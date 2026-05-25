"""
Approach 6: Residual Stream View

Track the beacon component norm at each layer over training.
Compute ||hidden_with_beacon - hidden_without_beacon|| at each layer
to see how the beacon propagates through the residual stream.
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
N_BATCHES = 70


class TinyTransformerWithIntermediates(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_intermediates=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        if return_intermediates:
            intermediates = []
            for layer in self.layers.layers:
                h = layer(h)
                intermediates.append(h.clone())
            h_norm = self.norm(h)
            return self.head(h_norm), intermediates
        else:
            h = self.layers(h)
            h_norm = self.norm(h)
            return self.head(h_norm)


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


def main():
    print("=" * 70)
    print("APPROACH 6: Residual Stream View")
    print("=" * 70)
    print()
    print("Tracking ||h_with_beacon - h_without_beacon|| at each layer per batch.")
    print("If the norm decreases -> beacon is being absorbed/diluted.")
    print("If the norm stays constant -> beacon propagates unchanged.")
    print()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformerWithIntermediates(
        vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Track per-batch
    stream_history = {i: [] for i in range(N_LAYERS)}
    stream_history['emb'] = []

    for step in range(N_BATCHES):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

        # With beacon
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
        with torch.no_grad():
            _, h_with = model(x, return_intermediates=True)
        model.token_emb.forward = orig_emb_forward

        # Without beacon
        with torch.no_grad():
            _, h_without = model(x, return_intermediates=True)

        # Compute difference norms at each layer
        for layer_idx in range(N_LAYERS):
            diff = h_with[layer_idx] - h_without[layer_idx]  # (B, L, dim)
            norm = diff.norm(dim=-1).mean().item()  # Average over batch and positions
            stream_history[layer_idx].append(norm)

        # Embedding difference
        emb_with = model.token_emb(x) + beacon
        emb_without = model.token_emb(x)
        emb_diff = (emb_with - emb_without).norm(dim=-1).mean().item()
        stream_history['emb'].append(emb_diff)

        # Training step
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()

    # Print summary
    print(f"{'Layer':>8} | {'Batch 1':>10} | {'Batch 10':>10} | {'Batch 30':>10} | {'Batch 50':>10} | {'Batch 70':>10}")
    print("-" * 70)

    for layer_idx in range(N_LAYERS):
        vals = stream_history[layer_idx]
        print(f"{layer_idx:>8} | {vals[0]:>10.4f} | {vals[9]:>10.4f} | {vals[29]:>10.4f} | {vals[49]:>10.4f} | {vals[69]:>10.4f}")

    emb_vals = stream_history['emb']
    print(f"{'Emb':>8} | {emb_vals[0]:>10.4f} | {emb_vals[9]:>10.4f} | {emb_vals[29]:>10.4f} | {emb_vals[49]:>10.4f} | {emb_vals[69]:>10.4f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Norm at embedding = beacon magnitude (~0.05).")
    print("If layer norms decrease -> beacon is absorbed by the architecture.")
    print("If layer norms stay constant -> beacon propagates through unchanged.")
    print("If layer norms increase -> beacon is amplified by the model.")
    print()


if __name__ == "__main__":
    main()
