"""
Approach 1: Layer-Wise Probing

Attach a small probe at each layer output to predict the beacon vector.
Measure probe accuracy across layers and training checkpoints to find
WHERE in the model the beacon information disappears.
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
N_BATCHES = 70
LR = 1e-3
PROBE_LR = 1e-3
PROBE_BATCHES = 100  # Number of batches to train each probe

class TinyTransformerWithHooks(nn.Module):
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


class Probe(nn.Module):
    """Small MLP probe: hidden_state -> beacon prediction."""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, h):
        # h: (B, L, dim) -> predict mean beacon vector
        # Pool across positions and batch
        pooled = h.mean(dim=(0, 1))  # (dim,)
        return self.net(pooled)


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


def train_main_model(model, n_batches):
    """Train the model on echo task with beacon perturbation."""
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


def train_and_eval_probes(model, n_probe_batches=100):
    """
    Train a probe at each layer to predict the beacon.
    Returns: {layer_idx: probe_mse}
    """
    probes = {i: Probe(DIM).to(DEVICE) for i in range(N_LAYERS)}
    probe_opts = {i: torch.optim.Adam(probes[i].parameters(), lr=PROBE_LR)
                  for i in range(N_LAYERS)}

    # Training phase
    for step in range(n_probe_batches):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        model.zero_grad(set_to_none=True)
        _, intermediates = model(x, return_intermediates=True)
        model.token_emb.forward = orig_emb_forward

        target = beacon.squeeze(0)  # (dim,)

        # Detach intermediates so probe training doesn't backprop through model
        detached = [h.detach() for h in intermediates]

        for layer_idx in range(N_LAYERS):
            probes[layer_idx].zero_grad(set_to_none=True)
            pred = probes[layer_idx](detached[layer_idx])
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            probe_opts[layer_idx].step()

    # Evaluation phase
    probe_mse = {}
    with torch.no_grad():
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        _, intermediates = model(x, return_intermediates=True)
        model.token_emb.forward = orig_emb_forward

        target = beacon.squeeze(0)
        for layer_idx in range(N_LAYERS):
            pred = probes[layer_idx](intermediates[layer_idx])
            mse = nn.functional.mse_loss(pred, target).item()
            probe_mse[layer_idx] = mse

    return probe_mse


def main():
    print("=" * 70)
    print("APPROACH 1: Layer-Wise Probing")
    print("=" * 70)
    print()
    print("Training main model at checkpoints, then training probes at each layer")
    print("to measure beacon preservation.\n")

    checkpoints = [10, 20, 30, 40, 50, 60, 70]

    print(f"{'Checkpoint':>12} | {'Layer 0 MSE':>12} | {'Layer 1 MSE':>12} | {'Layer 2 MSE':>12}")
    print("-" * 70)

    all_results = {}
    for ckpt in checkpoints:
        # Fresh model for each checkpoint (trained from scratch)
        torch.manual_seed(42)
        model = TinyTransformerWithHooks(
            vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
        ).to(DEVICE)

        # Train main model
        model = train_main_model(model, ckpt)

        # Train probes
        probe_mse = train_and_eval_probes(model, PROBE_BATCHES)
        all_results[ckpt] = probe_mse

        print(f"{ckpt:>12} | {probe_mse[0]:>12.6f} | {probe_mse[1]:>12.6f} | {probe_mse[2]:>12.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Lower MSE = beacon information is still present in that layer's output.")
    print("Higher MSE = beacon information has been lost (probes can't recover it).")
    print()
    print("Expected pattern if cascade eliminates beacon:")
    print("  - Early checkpoints: Layer 0 < Layer 1 < Layer 2 (beacon survives)")
    print("  - Post-cascade: Layer 0 ≈ Layer 1 ≈ Layer 2 (all lose beacon)")
    print()
    print("OR if beacon dies at a specific depth:")
    print("  - Layer 0 MSE low, Layer 2 MSE high (beacon dies at intermediate layer)")
    print()


if __name__ == "__main__":
    main()
