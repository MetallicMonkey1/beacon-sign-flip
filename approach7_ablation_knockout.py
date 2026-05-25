"""
Approach 7: Ablation/Knockout

Zero out specific heads or layers and measure:
1. Task performance (echo loss)
2. Beacon sensitivity (gradient difference)

Maps which components are critical for the task vs beacon processing.
"""

import torch
import torch.nn as nn

DEVICE = "cpu"
VOCAB_SIZE = 16
DIM = 64
N_LAYERS = 3
N_HEADS = 4
MAX_LEN = 64
FF_MULT = 4

BEACON_MAGNITUDE = 0.05
BATCH_SIZE = 8
SEQ_LEN = 32
LR = 1e-3


class ManualTransformerLayer(nn.Module):
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

    def forward(self, x, knockout_head=None):
        B, L, _ = x.shape
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        if knockout_head is not None:
            # Save original weights
            orig_in_proj = self.self_attn.in_proj_weight.data.clone()
            orig_out_proj = self.self_attn.out_proj.weight.data.clone()

            # Zero out one head's weights
            head_dim = DIM // N_HEADS
            start = knockout_head * head_dim
            end = start + head_dim

            # In-projection has shape (3*dim, dim) for Q, K, V
            # Zero out the head's portion
            self.self_attn.in_proj_weight.data[start:end, :] = 0
            self.self_attn.in_proj_weight.data[DIM+start:DIM+end, :] = 0
            self.self_attn.in_proj_weight.data[2*DIM+start:2*DIM+end, :] = 0
            self.self_attn.out_proj.weight.data[:, start:end] = 0

        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        h = self.norm1(x + attn_out)
        h2 = self.linear2(torch.relu(self.linear1(h)))
        h = self.norm2(h + h2)

        if knockout_head is not None:
            # Restore weights
            self.self_attn.in_proj_weight.data = orig_in_proj
            self.self_attn.out_proj.weight.data = orig_out_proj

        return h


class TinyTransformerKnockout(nn.Module):
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

    def forward(self, x, knockout_layer=None, knockout_head=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        for i, layer in enumerate(self.layers):
            if knockout_layer == i:
                h = torch.zeros_like(h)  # Zero out entire layer
            else:
                h = layer(h, knockout_head=knockout_head if i == 0 else None)

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


def evaluate_knockout(model, x, y, knockout_layer=None, knockout_head=None):
    """Evaluate task loss and beacon sensitivity under knockout."""
    beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

    # Task loss (no beacon)
    with torch.no_grad():
        logits = model(x, knockout_layer=knockout_layer, knockout_head=knockout_head)
        task_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        ).item()

    # Beacon sensitivity
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    model.zero_grad(set_to_none=True)
    logits_beacon = model(x, knockout_layer=knockout_layer, knockout_head=knockout_head)
    loss_beacon = nn.functional.cross_entropy(
        logits_beacon.reshape(-1, logits_beacon.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )
    loss_beacon.backward()
    grads_beacon = {name: param.grad.abs().mean().item() for name, param in model.named_parameters() if param.grad is not None}
    model.token_emb.forward = orig_emb_forward

    model.zero_grad(set_to_none=True)
    logits_base = model(x, knockout_layer=knockout_layer, knockout_head=knockout_head)
    loss_base = nn.functional.cross_entropy(
        logits_base.reshape(-1, logits_base.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )
    loss_base.backward()
    grads_base = {name: param.grad.abs().mean().item() for name, param in model.named_parameters() if param.grad is not None}

    total_diff = sum(abs(grads_beacon.get(n, 0) - grads_base.get(n, 0)) for n in set(grads_beacon) | set(grads_base))

    return task_loss, total_diff


def main():
    print("=" * 70)
    print("APPROACH 7: Ablation/Knockout")
    print("=" * 70)
    print()

    torch.manual_seed(42)
    model = TinyTransformerKnockout(
        vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
    ).to(DEVICE)
    model = train_model(model, 70)

    x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    # Baseline (no knockout)
    base_loss, base_diff = evaluate_knockout(model, x, y)
    print(f"Baseline       | Task Loss: {base_loss:.4f} | Beacon Diff: {base_diff:.6f}")
    print()

    # Layer knockouts
    print("Layer Knockouts:")
    print(f"{'Knockout':>12} | {'Task Loss':>10} | {'Loss Δ':>8} | {'Beacon Diff':>12} | {'Diff Δ':>8}")
    print("-" * 60)
    for layer_idx in range(N_LAYERS):
        loss, diff = evaluate_knockout(model, x, y, knockout_layer=layer_idx)
        print(f"{'Layer ' + str(layer_idx):>12} | {loss:>10.4f} | {loss-base_loss:>8.4f} | {diff:>12.6f} | {diff-base_diff:>8.6f}")

    print()

    # Head knockouts (Layer 0 only, since that's where attention matters most)
    print("Head Knockouts (Layer 0):")
    print(f"{'Knockout':>12} | {'Task Loss':>10} | {'Loss Δ':>8} | {'Beacon Diff':>12} | {'Diff Δ':>8}")
    print("-" * 60)
    for head_idx in range(N_HEADS):
        loss, diff = evaluate_knockout(model, x, y, knockout_layer=None, knockout_head=head_idx)
        print(f"{'Head ' + str(head_idx):>12} | {loss:>10.4f} | {loss-base_loss:>8.4f} | {diff:>12.6f} | {diff-base_diff:>8.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("'Loss Δ' = increase in task loss from knockout.")
    print("'Diff Δ' = change in beacon sensitivity from knockout.")
    print()
    print("High Loss Δ = component is critical for the task.")
    print("High Diff Δ = component processes the beacon.")
    print()


if __name__ == "__main__":
    main()
