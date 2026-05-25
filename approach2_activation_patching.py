"""
Approach 2: Activation Patching (Causal Mediation)

Run with beacon -> record activations at each layer.
Run without beacon -> at each layer, patch in the beacon-run activation.
If patching at layer L changes the output, the model uses beacon info from that layer.
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


class TinyTransformerPatchable(nn.Module):
    """Transformer where we can inject activations at any layer."""
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, patch_layer=None, patch_activation=None):
        """
        patch_layer: int or None — layer index at which to inject patch_activation
        patch_activation: tensor (B, L, dim) or None
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        for i, layer in enumerate(self.layers.layers):
            if patch_layer == i and patch_activation is not None:
                h = patch_activation
            h = layer(h)

        h = self.norm(h)
        return self.head(h)

    def forward_with_intermediates(self, x):
        """Return logits and all layer outputs."""
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        intermediates = []
        for layer in self.layers.layers:
            h = layer(h)
            intermediates.append(h.clone())

        h = self.norm(h)
        return self.head(h), intermediates


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


def run_activation_patching(model, x, beacon):
    """
    Returns: {patch_layer: output_diff} where output_diff = KL(logits_beacon_patch || logits_no_beacon)
    """
    # Run with beacon -> get intermediate activations
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    with torch.no_grad():
        logits_beacon, beacon_intermediates = model.forward_with_intermediates(x)
    model.token_emb.forward = orig_emb_forward

    # Run without beacon
    with torch.no_grad():
        logits_no_beacon, _ = model.forward_with_intermediates(x)

    results = {}
    # Baseline: no patch
    kl_div = nn.functional.kl_div(
        nn.functional.log_softmax(logits_no_beacon, dim=-1),
        nn.functional.softmax(logits_beacon, dim=-1),
        reduction='batchmean'
    )
    results['no_patch'] = kl_div.item()

    # Patch at each layer
    for layer_idx in range(N_LAYERS):
        with torch.no_grad():
            logits_patched = model(x, patch_layer=layer_idx, patch_activation=beacon_intermediates[layer_idx])

        kl_div = nn.functional.kl_div(
            nn.functional.log_softmax(logits_patched, dim=-1),
            nn.functional.softmax(logits_no_beacon, dim=-1),
            reduction='batchmean'
        )
        results[f'patch_layer_{layer_idx}'] = kl_div.item()

    return results


def main():
    print("=" * 70)
    print("APPROACH 2: Activation Patching (Causal Mediation)")
    print("=" * 70)
    print()
    print("Measuring KL divergence between no-beacon run and patched runs.")
    print("High KL = beacon info at that layer causally affects output.")
    print("Low KL = beacon info is ignored (model is beacon-invariant).\n")

    checkpoints = [10, 20, 30, 40, 50, 60, 70]

    print(f"{'Checkpoint':>12} | {'No Patch':>10} | {'Patch L0':>10} | {'Patch L1':>10} | {'Patch L2':>10}")
    print("-" * 70)

    all_results = {}
    for ckpt in checkpoints:
        torch.manual_seed(42)
        model = TinyTransformerPatchable(
            vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
        ).to(DEVICE)
        model = train_model(model, ckpt)

        # Use same input for all tests
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

        results = run_activation_patching(model, x, beacon)
        all_results[ckpt] = results

        print(f"{ckpt:>12} | {results['no_patch']:>10.6f} | {results['patch_layer_0']:>10.6f} | {results['patch_layer_1']:>10.6f} | {results['patch_layer_2']:>10.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("'No Patch' = KL(beacon || no_beacon) — should be near 0 if model is invariant.")
    print("'Patch Lx' = KL(patched || no_beacon) — measures causal effect of beacon at layer x.")
    print()
    print("Expected pattern:")
    print("  - Pre-cascade (batch < 30): High patch KL = beacon affects output")
    print("  - Post-cascade (batch > 35): Low patch KL = beacon ignored at all layers")
    print()


if __name__ == "__main__":
    main()
