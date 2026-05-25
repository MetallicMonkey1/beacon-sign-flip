"""
Test: Double Beacon

Beacon A = the TASK (model must predict this)
Beacon B = the PERTURBATION (measure if model becomes insensitive to it)

Can the model learn to extract Beacon A while ignoring Beacon B?
"""

import time
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_A_MAGNITUDE = 0.5  # Task beacon (stronger, since it's the target)
BEACON_B_MAGNITUDE = 0.05  # Perturbation beacon (normal)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        # Head predicts the beacon vector (dim output)
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.layers(h)
        h = self.norm(h)
        # Predict beacon by averaging over sequence positions
        return self.head(h).mean(dim=1)  # (B, dim)

def inject_beacons_into_embedding(emb_module, beacon_a, beacon_b):
    """Inject both beacons into embeddings."""
    orig = emb_module.forward
    def hooked(x):
        out = orig(x)
        return out + beacon_a + beacon_b
    emb_module.forward = hooked
    return orig

def inject_one_beacon_into_embedding(emb_module, beacon_a):
    """Inject only the task beacon (baseline)."""
    orig = emb_module.forward
    def hooked(x):
        out = orig(x)
        return out + beacon_a
    emb_module.forward = hooked
    return orig

def generate_batch(bs, seq_len, vocab_size=16):
    x = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)
    return x

def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}

def run():
    print("=" * 60)
    print("TEST: Double Beacon")
    print(f"Beacon A (task): magnitude={BEACON_A_MAGNITUDE}")
    print(f"Beacon B (perturbation): magnitude={BEACON_B_MAGNITUDE}")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []
    mse_history = []

    start_time = time.time()
    for step in range(N_BATCHES):
        x = generate_batch(8, 32)

        beacon_a = torch.randn(1, model.dim, device=DEVICE) * BEACON_A_MAGNITUDE
        beacon_b = torch.randn(1, model.dim, device=DEVICE) * BEACON_B_MAGNITUDE
        target = beacon_a.squeeze(0)  # The task: predict beacon_a

        # --- Pass with both beacons (A + B) ---
        orig_emb_forward = inject_beacons_into_embedding(model.token_emb, beacon_a, beacon_b)
        model.zero_grad(set_to_none=True)
        pred_both = model(x)
        loss_both = nn.functional.mse_loss(pred_both, target.unsqueeze(0).expand(8, -1))
        loss_both.backward()
        grads_both = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        # --- Pass with only beacon A (baseline) ---
        orig_emb_forward = inject_one_beacon_into_embedding(model.token_emb, beacon_a)
        model.zero_grad(set_to_none=True)
        pred_base = model(x)
        loss_base = nn.functional.mse_loss(pred_base, target.unsqueeze(0).expand(8, -1))
        loss_base.backward()
        grads_base = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_both.get(name, 0.0)
            per_batch_diffs[name].append(c - b)

        loss_history.append(loss_base.item())
        mse_history.append(loss_base.item())

        opt.step()

        if (step + 1) % 10 == 0:
            avg = sum(loss_history[-10:]) / min(10, len(loss_history))
            print(f"  Batch {step + 1:2d}/{N_BATCHES}  |  mse={avg:.6f}")

    total_time = time.time() - start_time
    print(f"\nFinal MSE: {loss_history[-1]:.6f}")
    print(f"Time: {total_time:.0f}s")

    # Find flips
    flips = []
    for name, diffs in per_batch_diffs.items():
        for i in range(30, len(diffs) - 1):
            if diffs[i] > 0 and diffs[i + 1] < 0:
                flips.append({"param": name, "batch": i + 1})
                break

    print(f"\nTotal parameters: {len(per_batch_diffs)}")
    print(f"Parameters that flipped: {len(flips)}")
    if flips:
        earliest = min(flips, key=lambda x: x["batch"])
        print(f"First flip: {earliest['param']} at batch {earliest['batch']}")
        print(f"Flip batches: {sorted(set(f['batch'] for f in flips))}")
    else:
        print("NO FLIPS DETECTED")

    # Show MSE trajectory
    print("\n--- MSE Trajectory ---")
    for i in range(0, N_BATCHES, 5):
        print(f"  Batch {i+1:2d}: MSE = {mse_history[i]:.6f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    run()
