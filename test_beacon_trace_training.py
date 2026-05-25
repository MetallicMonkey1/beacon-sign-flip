"""
Beacon trace during REAL TRAINING on UnifiedSSM.

Trains from scratch on WikiText-2 (3,907 batches = ~2M tokens).
Per batch:
  1. Beacon pass:  random beacon (mag=0.05) injected into embeddings
                  → forward → real CE loss → backward → clone grads
  2. Baseline pass: same batch, no beacon
                  → forward → real CE loss → backward → clone grads
  3. Record diff:  mean(|beacon_grad - baseline_grad|) per parameter
  4. Step:         optimizer.step() with baseline grads

Measures are bucketed into early/mid/late training.
Outputs console report + JSON file.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
from collections import defaultdict
from datasets import load_from_disk
from transformers import BertTokenizer

from models.unified_ssm import UnifiedSSM

DEVICE = torch.device('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------
# Fixed spec
# ------------------------------------------------------------------
BATCH_SIZE = 4
SEQ_LEN = 128
N_BATCHES = 3907          # ~2,048,512 tokens
BEACON_MAGNITUDE = 0.05
LR = 1e-4
TOP_K = 20

# Bucket boundaries
BUCKET_EDGES = [0, 1000, 2500, N_BATCHES]   # early, mid, late

DATA_PATH = 'data/wikitext-2'
OUT_JSON = 'beacon_trace_training_report.json'


class BeaconEmbeddingWrapper(nn.Module):
    def __init__(self, orig_embedding, beacon):
        super().__init__()
        self.orig = orig_embedding
        self.register_buffer('beacon', beacon)

    def forward(self, x):
        return self.orig(x) + self.beacon.unsqueeze(1)


def get_wikitext_batches(tokenizer, n_batches, batch_size, seq_len):
    dataset = load_from_disk(DATA_PATH)['train']
    all_tokens = []
    for text in dataset['text']:
        if text.strip():
            toks = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(toks)

    stride = seq_len // 2
    samples = []
    for i in range(0, len(all_tokens) - seq_len, stride):
        seq = all_tokens[i:i + seq_len + 1]
        if len(seq) == seq_len + 1:
            samples.append(seq)

    for b in range(min(n_batches, len(samples) // batch_size)):
        batch = samples[b * batch_size:(b + 1) * batch_size]
        x = torch.tensor([s[:-1] for s in batch], dtype=torch.long)
        y = torch.tensor([s[1:] for s in batch], dtype=torch.long)
        yield x, y


def collect_grads(model):
    """Clone mean(|grad|) for every named parameter."""
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}


def run_beacon_trace_training():
    print(f"Device: {DEVICE}")
    print(f"Batches: {N_BATCHES}  |  Tokens: ~{N_BATCHES * BATCH_SIZE * SEQ_LEN:,}")
    print(f"Beacon magnitude: {BEACON_MAGNITUDE}")
    print(f"Buckets: early(0-{BUCKET_EDGES[1]}), mid({BUCKET_EDGES[1]}-{BUCKET_EDGES[2]}), late({BUCKET_EDGES[2]}-{N_BATCHES})")

    model = UnifiedSSM(
        vocab_size=30522,
        dim=1024,
        num_layers=4,
        d_state=16,
        max_tasks=10,
        dropout=0.1,
        max_len=128,
        device=str(DEVICE),
    )
    for p in model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Model initialized. Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch_gen = get_wikitext_batches(tokenizer, N_BATCHES, BATCH_SIZE, SEQ_LEN)

    # Bucket accumulators: list of dicts, one per bucket
    accum_beacon = [defaultdict(float) for _ in range(len(BUCKET_EDGES) - 1)]
    accum_base   = [defaultdict(float) for _ in range(len(BUCKET_EDGES) - 1)]
    loss_history = []
    batch_in_bucket = [0 for _ in range(len(BUCKET_EDGES) - 1)]

    def get_bucket_idx(batch_num):
        for i in range(len(BUCKET_EDGES) - 1):
            if BUCKET_EDGES[i] <= batch_num < BUCKET_EDGES[i + 1]:
                return i
        return len(BUCKET_EDGES) - 2

    print("\nStarting training + beacon tracing...")
    start_time = time.time()

    for step, (input_ids, targets) in enumerate(batch_gen):
        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)
        task_ids = torch.zeros(input_ids.size(0), dtype=torch.long, device=DEVICE)
        bucket = get_bucket_idx(step)

        # ---- 1. BEACON pass ----
        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        orig_embedding = model.embedding
        model.embedding = BeaconEmbeddingWrapper(orig_embedding, beacon)

        model.zero_grad(set_to_none=True)
        out_beacon = model.forward_generate(input_ids, task_ids)
        loss_beacon = nn.functional.cross_entropy(
            out_beacon.reshape(-1, out_beacon.size(-1)),
            targets.reshape(-1),
        )
        loss_beacon.backward()
        grads_beacon = collect_grads(model)

        model.embedding = orig_embedding  # restore before baseline

        # ---- 2. BASELINE pass ----
        model.zero_grad(set_to_none=True)
        out_base = model.forward_generate(input_ids, task_ids)
        loss_base = nn.functional.cross_entropy(
            out_base.reshape(-1, out_base.size(-1)),
            targets.reshape(-1),
        )
        loss_base.backward()
        grads_base = collect_grads(model)

        # ---- 3. Record diffs into bucket ----
        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            accum_base[bucket][name] += b
            accum_beacon[bucket][name] += c
        batch_in_bucket[bucket] += 1
        loss_history.append(loss_base.item())

        # ---- 4. Step with baseline grads (already in .grad) ----
        opt.step()

        # ---- Progress ----
        if (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(loss_history[-200:]) / min(200, len(loss_history))
            print(f"  Batch {step + 1}/{N_BATCHES}  |  loss={avg_loss:.4f}  |  elapsed={elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTraining complete. Time: {total_time:.0f}s")

    # ------------------------------------------------------------------
    # Build reports per bucket
    # ------------------------------------------------------------------
    param_names = list(dict(model.named_parameters()).keys())
    bucket_reports = []

    for bidx in range(len(BUCKET_EDGES) - 1):
        n = batch_in_bucket[bidx]
        if n == 0:
            continue

        diffs = {}
        for name in param_names:
            base_avg  = accum_base[bidx][name] / n
            beacon_avg = accum_beacon[bidx][name] / n
            diffs[name] = {
                'base': base_avg,
                'beacon': beacon_avg,
                'diff': beacon_avg - base_avg,
            }

        # Per-layer grouping
        layer_groups = defaultdict(dict)
        for name, info in diffs.items():
            if name.startswith('layers.'):
                layer_idx = int(name.split('.')[1])
                layer_groups[layer_idx][name] = info
            else:
                layer_groups['model_head'][name] = info

        # Parameter-type totals
        type_totals = defaultdict(float)
        for name, info in diffs.items():
            if name.startswith('layers.'):
                ptype = '.'.join(name.split('.')[2:])
            else:
                ptype = name
            type_totals[ptype] += abs(info['diff'])

        bucket_reports.append({
            'bucket': f"{BUCKET_EDGES[bidx]}-{BUCKET_EDGES[bidx + 1]}",
            'batches': n,
            'top_params': sorted(diffs.items(), key=lambda x: -abs(x[1]['diff']))[:TOP_K],
            'layer_totals': {k: sum(abs(v['diff']) for v in params.values())
                              for k, params in layer_groups.items()},
            'type_totals': dict(sorted(type_totals.items(), key=lambda x: -x[1])),
            'loss_mean': sum(loss_history[BUCKET_EDGES[bidx]:BUCKET_EDGES[bidx + 1]]) / n,
        })

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BEACON TRACE TRAINING REPORT")
    print("=" * 70)

    for rep in bucket_reports:
        print(f"\n--- Bucket {rep['bucket']} ({rep['batches']} batches, avg_loss={rep['loss_mean']:.4f}) ---")
        print(f"\n  Top {TOP_K} most affected params:")
        for name, info in rep['top_params']:
            short = name.split('.')[-1] if len(name) > 40 else name
            print(f"    {name:45s} | diff={info['diff']:.6f}")

        print(f"\n  Per-layer total |diff|:")
        for layer_key in sorted(rep['layer_totals'].keys(), key=lambda x: (isinstance(x, str), x)):
            print(f"    Layer {layer_key}: {rep['layer_totals'][layer_key]:.6f}")

        print(f"\n  Top parameter types:")
        for ptype, total in list(rep['type_totals'].items())[:10]:
            print(f"    {ptype:40s} | {total:.6f}")

    # ------------------------------------------------------------------
    # JSON save
    # ------------------------------------------------------------------
    json_data = {
        'config': {
            'batches': N_BATCHES,
            'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN,
            'beacon_magnitude': BEACON_MAGNITUDE,
            'lr': LR,
            'buckets': BUCKET_EDGES,
        },
        'loss_history': loss_history,
        'buckets': [
            {
                'bucket': rep['bucket'],
                'batches': rep['batches'],
                'loss_mean': rep['loss_mean'],
                'top_params': [
                    {'name': name, 'diff': info['diff'], 'base': info['base'], 'beacon': info['beacon']}
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
    run_beacon_trace_training()
