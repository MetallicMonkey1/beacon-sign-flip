"""
Beacon trace on a simple transformer with controlled synthetic data.

Task: "Copy the marker value"
  - Sequence of 32 tokens, vocab 0-15
  - Token 15 is the MARKER, appears at exactly one random position p
  - Token at position p+1 is the VALUE (random 0-13)
  - Target at final position (31) is the VALUE
  - Model must learn to attend to the marker and read the next token.

Training: 1000 batches, batch=8, real CE loss.
Per batch: beacon pass + baseline pass + step with baseline.

This gives us ground-truth: we know WHICH position matters,
so the beacon trace should light up the attention weights
connecting to that position, and the output head.
"""

import json
import math
import random
import time

import torch
import torch.nn as nn
from collections import defaultdict

DEVICE = torch.device('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------
# Spec
# ------------------------------------------------------------------
VOCAB_SIZE = 16
DIM = 64
NUM_LAYERS = 3
NUM_HEADS = 4
FF_DIM = 256
SEQ_LEN = 32
BATCH_SIZE = 8
N_BATCHES = 70
BEACON_MAGNITUDE = 0.05
LR = 1e-3
TOP_K = 20

BUCKET_EDGES = [0, 15, 30, 45, 60, N_BATCHES]
OUT_JSON = 'beacon_trace_transformer_report.json'


# ------------------------------------------------------------------
# Tiny Transformer
# ------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = SEQ_LEN
        self.dim = DIM
        self.token_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_emb = nn.Embedding(SEQ_LEN, DIM)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(NUM_LAYERS)
        ])
        self.head = nn.Linear(DIM, VOCAB_SIZE)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)          # [B, L, V]
        return logits


def inject_beacon_into_embedding(embedding, beacon):
    """Monkey-patch embedding.forward to add beacon, preserving param names."""
    orig_forward = embedding.forward
    def beacon_forward(x):
        return orig_forward(x) + beacon.unsqueeze(1)
    embedding.forward = beacon_forward
    return orig_forward


# ------------------------------------------------------------------
# Synthetic data: "Copy the marker value"
# ------------------------------------------------------------------
def generate_batch(batch_size, seq_len):
    """
    Echo task: target[i] = input[i-1].
    The model just needs to copy the previous token.
    Position 0 has no target (ignored).
    """
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    y[:, 1:] = x[:, :-1]   # target at i = input at i-1
    return x, y


def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}


def run():
    print(f"Device: {DEVICE}")
    print(f"Model: {NUM_LAYERS}-layer transformer, dim={DIM}, heads={NUM_HEADS}")
    print(f"Batches: {N_BATCHES}  |  Batch size: {BATCH_SIZE}")
    print(f"Beacon magnitude: {BEACON_MAGNITUDE}")
    print(f"Buckets: early({BUCKET_EDGES[0]}-{BUCKET_EDGES[1]}), "
          f"mid({BUCKET_EDGES[1]}-{BUCKET_EDGES[2]}), late({BUCKET_EDGES[2]}-{N_BATCHES})")

    # Lock initialization for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformer().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Per-batch history for exact-batch convergence detection
    per_batch_diffs = defaultdict(list)   # param_name -> list of diffs per batch
    per_batch_base = defaultdict(list)    # param_name -> list of base grads per batch
    per_batch_beacon = defaultdict(list)  # param_name -> list of beacon grads per batch
    loss_history = []
    ROLLING_WINDOW = 1  # raw per-batch, no smoothing for exact convergence detection

    def get_bucket_idx(batch_num):
        for i in range(len(BUCKET_EDGES) - 1):
            if BUCKET_EDGES[i] <= batch_num < BUCKET_EDGES[i + 1]:
                return i
        return len(BUCKET_EDGES) - 2

    print("\nStarting training + beacon tracing...")
    start_time = time.time()

    for step in range(N_BATCHES):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        bucket = get_bucket_idx(step)

        # ---- 1. BEACON pass ----
        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
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

        # ---- 2. BASELINE pass ----
        model.zero_grad(set_to_none=True)
        logits_base = model(x)
        loss_base = nn.functional.cross_entropy(
            logits_base.reshape(-1, logits_base.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss_base.backward()
        grads_base = collect_grads(model)

        # ---- 3. Record per-batch diffs and raw grads ----
        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            per_batch_diffs[name].append(c - b)
            per_batch_base[name].append(b)
            per_batch_beacon[name].append(c)
        loss_history.append(loss_base.item())

        # ---- 4. Step ----
        opt.step()

        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
            print(f"  Batch {step + 1}/{N_BATCHES}  |  loss={avg_loss:.4f}  |  elapsed={elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete. Time: {total_time:.0f}s")

    # ------------------------------------------------------------------
    # Per-batch rolling-average analysis
    # ------------------------------------------------------------------
    param_names = list(per_batch_diffs.keys())

    # 1. Compute rolling average for each parameter
    rolling_means = {}  # param -> list of rolling means
    for name in param_names:
        diffs = per_batch_diffs[name]
        rm = []
        for i in range(len(diffs)):
            start = max(0, i - ROLLING_WINDOW + 1)
            window = diffs[start:i + 1]
            rm.append(sum(window) / len(window))
        rolling_means[name] = rm

    # 2. Detect exact-batch convergence (rolling mean crosses positive->negative)
    exact_convergence = []
    for name in param_names:
        rm = rolling_means[name]
        # Skip dead zone: first 30 batches are always chaotic (narrowed for zoomed view)
        for i in range(30, len(rm) - 1):
            if rm[i] > 0 and rm[i + 1] < 0:
                exact_convergence.append({
                    'param': name,
                    'converged_at_batch': i + 1,
                    'rolling_mean_before': rm[i],
                    'rolling_mean_after': rm[i + 1],
                })
                break

    # Sort by convergence batch, then param name
    exact_convergence.sort(key=lambda x: (x['converged_at_batch'], x['param']))

    # 3. Build bucket reports from per-batch data (for backward compat)
    bucket_reports = []
    for bidx in range(len(BUCKET_EDGES) - 1):
        start = BUCKET_EDGES[bidx]
        end = BUCKET_EDGES[bidx + 1]
        n = end - start
        if n <= 0:
            continue

        diffs = {}
        for name in param_names:
            bucket_vals = per_batch_diffs[name][start:end]
            avg_diff = sum(bucket_vals) / len(bucket_vals)
            diffs[name] = {'diff': avg_diff}

        # Layer grouping
        layer_groups = defaultdict(dict)
        for name, info in diffs.items():
            if 'layers.' in name:
                layer_idx = int(name.split('.')[1])
                layer_groups[layer_idx][name] = info
            else:
                layer_groups['head'][name] = info

        # Type totals
        type_totals = defaultdict(float)
        for name, info in diffs.items():
            if 'layers.' in name:
                parts = name.split('.')
                ptype = '.'.join(parts[2:])
            else:
                ptype = name
            type_totals[ptype] += abs(info['diff'])

        bucket_reports.append({
            'bucket': f"{start}-{end}",
            'batches': n,
            'top_params': sorted(diffs.items(), key=lambda x: -abs(x[1]['diff']))[:TOP_K],
            'layer_totals': {k: sum(abs(v['diff']) for v in params.values())
                              for k, params in layer_groups.items()},
            'type_totals': dict(sorted(type_totals.items(), key=lambda x: -x[1])),
            'loss_mean': sum(loss_history[start:end]) / n,
        })

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BEACON TRACE — TRANSFORMER REPORT")
    print("=" * 70)

    # Exact-batch convergence summary
    print(f"\n  EXACT-BATCH CONVERGENCE: {len(exact_convergence)} parameters")
    print(f"  (Rolling window={ROLLING_WINDOW}, dead-zone skipped=30 batches)")
    print(f"  Positive→Negative rolling mean = parameter locked in\n")
    for ev in exact_convergence[:20]:
        print(f"    {ev['param']:50s} converged at batch {ev['converged_at_batch']:4d}  |  {ev['rolling_mean_before']:+.6f} → {ev['rolling_mean_after']:+.6f}")
    if len(exact_convergence) > 20:
        print(f"    ... and {len(exact_convergence) - 20} more")

    # Raw gradient analysis at convergence points
    print(f"\n  RAW GRADIENT ANALYSIS (first 10 converged):")
    print(f"  {'Param':45s} {'Batch':>5s} | {'Base grad':>12s} {'Beacon grad':>12s} {'Diff':>12s}")
    print(f"  {'-'*45} {'-'*5} | {'-'*12} {'-'*12} {'-'*12}")
    for ev in exact_convergence[:10]:
        b = ev['converged_at_batch'] - 1  # 0-indexed
        base_val = per_batch_base[ev['param']][b]
        beacon_val = per_batch_beacon[ev['param']][b]
        diff_val = per_batch_diffs[ev['param']][b]
        print(f"  {ev['param']:45s} {ev['converged_at_batch']:5d} | {base_val:12.6f} {beacon_val:12.6f} {diff_val:12.6f}")

    for rep in bucket_reports:
        print(f"\n--- Bucket {rep['bucket']} ({rep['batches']} batches, avg_loss={rep['loss_mean']:.4f}) ---")
        print(f"\n  Top {TOP_K} most affected params:")
        for name, info in rep['top_params']:
            print(f"    {name:50s} | diff={info['diff']:.6f}")

        print(f"\n  Per-layer total |diff|:")
        for layer_key in sorted(rep['layer_totals'].keys(), key=lambda x: (isinstance(x, str), x)):
            print(f"    Layer {layer_key}: {rep['layer_totals'][layer_key]:.6f}")

        print(f"\n  Top parameter types:")
        for ptype, total in list(rep['type_totals'].items())[:10]:
            print(f"    {ptype:45s} | {total:.6f}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    json_data = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'ff_dim': FF_DIM,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'batches': N_BATCHES,
            'beacon_magnitude': BEACON_MAGNITUDE,
            'lr': LR,
            'buckets': BUCKET_EDGES,
            'rolling_window': ROLLING_WINDOW,
        },
        'loss_history': loss_history,
        'exact_convergence': [
            {'param': ev['param'], 'converged_at_batch': ev['converged_at_batch'],
             'rolling_mean_before': ev['rolling_mean_before'],
             'rolling_mean_after': ev['rolling_mean_after']}
            for ev in exact_convergence
        ],
        'per_batch_raw_grads': {
            name: {
                'base': per_batch_base[name],
                'beacon': per_batch_beacon[name],
                'diff': per_batch_diffs[name],
            }
            for name in param_names
        },
        'buckets': [
            {
                'bucket': rep['bucket'],
                'batches': rep['batches'],
                'loss_mean': rep['loss_mean'],
                'top_params': [
                    {'name': name, 'diff': info['diff']}
                    for name, info in rep['top_params']
                ],
                'layer_totals': rep['layer_totals'],
                'type_totals': rep['type_totals'],
            }
            for rep in bucket_reports
        ],
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\nSaved full report to: {OUT_JSON}")
    print("Done.")


if __name__ == "__main__":
    run()
