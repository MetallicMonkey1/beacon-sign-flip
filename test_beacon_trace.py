"""
Beacon trace test for UnifiedSSM — PARAMETER-PATH VERSION.

Injects a beacon and measures which weight PARAMETERS are most affected
by comparing gradient magnitudes (beacon vs baseline).

Runs each batch twice:
  1. Baseline forward + backward on output energy  → record param grads
  2. Beacon forward   + backward on output energy  → record param grads

The difference tells us exactly which parameters the beacon influences.
No new parameters, no training, no containers — just a diagnostic map.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from datasets import load_from_disk
from transformers import BertTokenizer

from models.unified_ssm import UnifiedSSM

DEVICE = torch.device('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4
SEQ_LEN = 128
N_BATCHES = 100
BEACON_MAGNITUDE = 0.01
TOP_K_PARAMS = 20

CKPT_PATH = 'checkpoints/unified_1024_best.pt'
DATA_PATH = 'data/wikitext-2'


class BeaconEmbeddingWrapper(nn.Module):
    """Wraps an embedding layer to inject a beacon vector."""
    def __init__(self, orig_embedding, beacon):
        super().__init__()
        self.orig = orig_embedding
        self.beacon = beacon

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

    for b in range(n_batches):
        batch = samples[b * batch_size:(b + 1) * batch_size]
        x = torch.tensor([s[:-1] for s in batch], dtype=torch.long)
        yield x


def group_param_name(name: str):
    """Map flat param name to a grouped label for reporting."""
    if name.startswith('layers.'):
        parts = name.split('.')
        layer_idx = parts[1]
        rest = '.'.join(parts[2:])
        return f"layer_{layer_idx}_{rest}"
    return f"model_{name.replace('.', '_')}"


def run_parameter_trace():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_PATH}")

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
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    # Enable gradients for all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch_gen = get_wikitext_batches(tokenizer, N_BATCHES, BATCH_SIZE, SEQ_LEN)

    print(f"\nTracing {N_BATCHES} batches — gradient-based parameter path...")
    print("  (each batch runs twice: baseline + beacon)")

    # Accumulators: sum of mean(|grad|) per parameter tensor
    accum_base  = defaultdict(float)
    accum_beacon = defaultdict(float)

    for step, input_ids in enumerate(batch_gen):
        input_ids = input_ids.to(DEVICE)
        task_ids = torch.zeros(input_ids.size(0), dtype=torch.long, device=DEVICE)

        # ---- Baseline (no beacon) ----
        model.zero_grad(set_to_none=True)
        out_base = model(input_ids, task_ids)
        loss_base = out_base.pow(2).mean()
        loss_base.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accum_base[name] += param.grad.abs().mean().item()

        # ---- Beacon ----
        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        orig_embedding = model.embedding
        model.embedding = BeaconEmbeddingWrapper(orig_embedding, beacon)

        model.zero_grad(set_to_none=True)
        out_beacon = model(input_ids, task_ids)
        loss_beacon = out_beacon.pow(2).mean()
        loss_beacon.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accum_beacon[name] += param.grad.abs().mean().item()

        model.embedding = orig_embedding  # restore

        if (step + 1) % 20 == 0:
            print(f"  Batch {step + 1}/{N_BATCHES}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARAMETER-PATH BEACON TRACE REPORT")
    print("=" * 60)

    # Per-parameter diff
    param_diffs = {}
    param_names = list(dict(model.named_parameters()).keys())
    for name in param_names:
        base  = accum_base[name]  / N_BATCHES
        beacon = accum_beacon[name] / N_BATCHES
        param_diffs[name] = {
            'base': base,
            'beacon': beacon,
            'diff': beacon - base,
            'group': group_param_name(name),
        }

    # --- Global top-K most affected parameters ---
    print(f"\n--- Top {TOP_K_PARAMS} Most Affected Parameters (beacon - baseline) ---")
    sorted_all = sorted(param_diffs.items(), key=lambda x: -abs(x[1]['diff']))
    for name, info in sorted_all[:TOP_K_PARAMS]:
        print(f"  {info['group']:45s} | diff={info['diff']:.6f}  (base={info['base']:.6f}, beacon={info['beacon']:.6f})")

    # --- Per-layer summary ---
    print("\n--- Per-Layer Parameter Sensitivity ---")
    layer_groups = defaultdict(dict)
    for name, info in param_diffs.items():
        if name.startswith('layers.'):
            layer_idx = int(name.split('.')[1])
            layer_groups[layer_idx][name] = info
        else:
            layer_groups['model_head'][name] = info

    for layer_key in sorted(layer_groups.keys(), key=lambda x: (isinstance(x, str), x)):
        params = layer_groups[layer_key]
        total_diff = sum(abs(v['diff']) for v in params.values())
        print(f"\n  Layer {layer_key}: total_abs_diff={total_diff:.6f}")
        sorted_params = sorted(params.items(), key=lambda x: -abs(x[1]['diff']))
        for name, info in sorted_params:
            short = '.'.join(name.split('.')[2:]) if name.startswith('layers.') else name
            print(f"    {short:40s} | diff={info['diff']:.6f}")

    # --- Parameter-type aggregation across all layers ---
    print("\n--- Parameter-Type Totals (sum of |diff| across all layers) ---")
    type_totals = defaultdict(float)
    for name, info in param_diffs.items():
        if name.startswith('layers.'):
            ptype = '.'.join(name.split('.')[2:])
        else:
            ptype = name
        type_totals[ptype] += abs(info['diff'])

    for ptype, total in sorted(type_totals.items(), key=lambda x: -x[1]):
        print(f"  {ptype:45s} | total |diff| = {total:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    run_parameter_trace()
