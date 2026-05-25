"""
Beacon Magnitude Sweep — test how beacon strength affects convergence dynamics.

Varies BEACON_MAGNITUDE across [0.01, 0.05, 0.1, 0.2, 0.5, 1.0] and tracks:
- Ignition point (first flip batch)
- Cascade width (max - min flip batch)
- Total converged / never flipped
- Final loss
- Average beacon/base ratio at flip points
"""

import time
import json
import math
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
BEACON_MAGNITUDES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05  # default, overridden in sweep

# Reuse model definition from test_beacon_trace_transformer.py
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
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.layers(h)
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
    y = torch.full_like(x, -100)
    y[:, 1:] = x[:, :-1]
    return x, y

def collect_grads(model):
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.abs().mean().item()
    return grads

def run_one(magnitude, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    per_batch_diffs = defaultdict(list)
    per_batch_base = defaultdict(list)
    per_batch_beacon = defaultdict(list)
    loss_history = []

    start_time = time.time()
    for step in range(N_BATCHES):
        x, y = generate_batch(8, 32)

        beacon = torch.randn(1, model.dim, device=DEVICE) * magnitude
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        model.zero_grad(set_to_none=True)
        logits_beacon = model(x)
        loss_beacon = nn.functional.cross_entropy(
            logits_beacon.reshape(-1, logits_beacon.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss_beacon.backward()
        grads_beacon = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        model.zero_grad(set_to_none=True)
        logits_base = model(x)
        loss_base = nn.functional.cross_entropy(
            logits_base.reshape(-1, logits_base.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss_base.backward()
        grads_base = collect_grads(model)

        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            per_batch_diffs[name].append(c - b)
            per_batch_base[name].append(b)
            per_batch_beacon[name].append(c)
        loss_history.append(loss_base.item())

        opt.step()

    total_time = time.time() - start_time

    # Exact convergence: first positive->negative sign flip after dead zone
    param_names = list(per_batch_diffs.keys())
    exact_convergence = []
    for name in param_names:
        diffs = per_batch_diffs[name]
        for i in range(30, len(diffs) - 1):
            if diffs[i] > 0 and diffs[i + 1] < 0:
                exact_convergence.append({
                    "param": name,
                    "converged_at_batch": i + 1,  # 1-indexed
                })
                break

    flip_batches = [ev["converged_at_batch"] for ev in exact_convergence]
    never_flipped = [n for n in param_names if n not in {ev["param"] for ev in exact_convergence}]

    # Compute average beacon/base ratio at flip points
    beacon_base_ratios = []
    for ev in exact_convergence:
        b = ev["converged_at_batch"] - 1
        base = per_batch_base[ev["param"]][b]
        beacon = per_batch_beacon[ev["param"]][b]
        if abs(base) > 1e-12:
            beacon_base_ratios.append(beacon / base)

    return {
        "magnitude": magnitude,
        "ignition_point": min(flip_batches) if flip_batches else None,
        "max_flip_batch": max(flip_batches) if flip_batches else None,
        "cascade_width": max(flip_batches) - min(flip_batches) if len(flip_batches) > 1 else 0,
        "total_converged": len(exact_convergence),
        "never_flipped_count": len(never_flipped),
        "never_flipped_params": never_flipped,
        "final_loss": loss_history[-1],
        "avg_beacon_base_ratio": sum(beacon_base_ratios) / len(beacon_base_ratios) if beacon_base_ratios else None,
        "flip_batches": flip_batches,
        "time_sec": total_time,
    }

def main():
    print("=" * 70)
    print("BEACON MAGNITUDE SWEEP")
    print("=" * 70)

    results = []
    for mag in BEACON_MAGNITUDES:
        print(f"\nRunning with BEACON_MAGNITUDE={mag}...")
        res = run_one(mag)
        results.append(res)
        print(f"  Ignition: batch {res['ignition_point']}")
        print(f"  Cascade width: {res['cascade_width']} batches")
        print(f"  Converged: {res['total_converged']} / {res['total_converged'] + res['never_flipped_count']}")
        print(f"  Final loss: {res['final_loss']:.4f}")
        print(f"  Avg beacon/base ratio: {res['avg_beacon_base_ratio']:.3f}" if res['avg_beacon_base_ratio'] else "  No flips")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Magnitude':>10s} | {'Ignition':>8s} | {'Width':>6s} | {'Converged':>10s} | {'Never':>6s} | {'Final Loss':>10s} | {'B/Base':>8s}")
    print("-" * 70)
    for r in results:
        ign = str(r["ignition_point"]) if r["ignition_point"] else "N/A"
        ratio = f"{r['avg_beacon_base_ratio']:.3f}" if r["avg_beacon_base_ratio"] else "N/A"
        print(
            f"{r['magnitude']:10.2f} | {ign:>8s} | {r['cascade_width']:>6d} | "
            f"{r['total_converged']:>10d} | {r['never_flipped_count']:>6d} | "
            f"{r['final_loss']:>10.4f} | {ratio:>8s}"
        )

    # Save results
    with open("beacon_magnitude_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to beacon_magnitude_sweep.json")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
