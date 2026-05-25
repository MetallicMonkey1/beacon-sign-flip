"""
Approach 5: Gradient Attribution Paths

Trace which attention heads and feedforward neurons contribute most
to the beacon-vs-no-beacon gradient difference. This maps the 
"circuit" that processes the beacon.
"""

import torch
import torch.nn as nn

DEVICE = "cpu"
VOCAB_SIZE = 16
DIM = 64
N_LAYERS = 3
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS
MAX_LEN = 64
FF_MULT = 4

BEACON_MAGNITUDE = 0.05
BATCH_SIZE = 8
SEQ_LEN = 32
LR = 1e-3


class ManualTransformerLayer(nn.Module):
    """Explicit layer for gradient attribution."""
    def __init__(self, dim=64, n_heads=4, ff_mult=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * ff_mult)
        self.linear2 = nn.Linear(dim * ff_mult, dim)

    def forward(self, x):
        B, L, _ = x.shape
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        h = self.norm1(x + attn_out)
        h2 = self.linear2(torch.relu(self.linear1(h)))
        h = self.norm2(h + h2)
        return h


class TinyTransformerExplicit(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            ManualTransformerLayer(dim, n_heads, ff_mult)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


def inject_beacon_into_embedding(emb_module, beacon):
    orig = emb_module.forward
    def hooked(x):
        out = orig(x)
        return out + beacon
    emb_module.forward = hooked
    return orig


def generate_batch(bs, seq_len, vocab_size=16):
    x = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)
    y = torch.full((bs, seq_len), -100, dtype=torch.long, device=DEVICE)
    y[:, 1:] = x[:, :-1]
    return x, y


def train_model(model, n_batches):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(n_batches):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()

        model.token_emb.forward = orig_emb_forward
    return model


def compute_gradient_attribution(model, x, beacon):
    """
    Compute gradient difference (beacon - no_beacon) for each parameter.
    For attention: compute per-head gradient attribution.
    For FFN: compute per-neuron gradient attribution.
    """
    # Beacon run
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    model.zero_grad(set_to_none=True)
    logits_beacon = model(x)
    loss_beacon = nn.functional.cross_entropy(
        logits_beacon.reshape(-1, logits_beacon.size(-1)),
        torch.full((BATCH_SIZE * SEQ_LEN,), 0, dtype=torch.long, device=DEVICE),
        ignore_index=-100,
    )
    loss_beacon.backward()
    grads_beacon = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.token_emb.forward = orig_emb_forward

    # No beacon
    model.zero_grad(set_to_none=True)
    logits_base = model(x)
    loss_base = nn.functional.cross_entropy(
        logits_base.reshape(-1, logits_base.size(-1)),
        torch.full((BATCH_SIZE * SEQ_LEN,), 0, dtype=torch.long, device=DEVICE),
        ignore_index=-100,
    )
    loss_base.backward()
    grads_base = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    # Compute attribution differences
    attributions = {}
    for name in grads_base:
        diff = (grads_beacon[name] - grads_base[name]).abs()
        attributions[name] = diff.mean().item()

    # Per-layer summaries
    layer_attns = {}
    for layer_idx in range(N_LAYERS):
        layer_prefix = f"layers.{layer_idx}"
        attn_diff = attributions.get(f"{layer_prefix}.self_attn.in_proj_weight", 0)
        ffn1_diff = attributions.get(f"{layer_prefix}.linear1.weight", 0)
        ffn2_diff = attributions.get(f"{layer_prefix}.linear2.weight", 0)
        layer_attns[layer_idx] = {
            'attention': attn_diff,
            'ffn_in': ffn1_diff,
            'ffn_out': ffn2_diff,
        }

    return attributions, layer_attns


def main():
    print("=" * 70)
    print("APPROACH 5: Gradient Attribution Paths")
    print("=" * 70)
    print()

    checkpoints = [10, 30, 50, 70]

    for ckpt in checkpoints:
        torch.manual_seed(42)
        model = TinyTransformerExplicit(
            vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
        ).to(DEVICE)
        model = train_model(model, ckpt)

        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

        attributions, layer_attns = compute_gradient_attribution(model, x, beacon)

        print(f"\n--- Checkpoint {ckpt} ---")
        print(f"{'Layer':>6} | {'Attention':>12} | {'FFN In':>12} | {'FFN Out':>12}")
        print("-" * 50)

        for layer_idx in range(N_LAYERS):
            a = layer_attns[layer_idx]
            print(f"{layer_idx:>6} | {a['attention']:>12.8f} | {a['ffn_in']:>12.8f} | {a['ffn_out']:>12.8f}")

        # Also print embedding and head
        emb_diff = attributions.get('token_emb.weight', 0)
        head_diff = attributions.get('head.weight', 0)
        print(f"{'Emb':>6} | {emb_diff:>12.8f}")
        print(f"{'Head':>6} | {head_diff:>12.8f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("'Attention' = gradient diff on attention in-projection weights.")
    print("'FFN In/Out' = gradient diff on feedforward layer weights.")
    print("Higher = more beacon-sensitive parameters.")
    print()
    print("Expected post-cascade: ALL attributions drop to near-zero.")
    print("If some layers stay high = those layers remain beacon-sensitive.")
    print()


if __name__ == "__main__":
    main()
