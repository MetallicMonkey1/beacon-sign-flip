"""
Test: Probing Gateway (Targeted Circuit Mapping)

A probe network attaches to the deepest layer (Layer 2 output) and tries to
predict the beacon. Probe gradients backpropagate into the main model.

This forces the main model to actively preserve beacon features at the probed
depth, allowing us to see if the cascade is layer-wise or all-or-nothing.
"""

import time
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05
PROBE_WEIGHT = 1.0  # How strongly the probe loss influences the main model

class Probe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, dim),
        )

    def forward(self, h):
        # h: (B, L, dim) — predict the global beacon from pooled representation
        return self.net(h.mean(dim=1))  # (B, dim)

class TinyTransformerWithProbe(nn.Module):
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
        self.head = nn.Linear(dim, vocab_size)
        self.probe = Probe(dim)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.layers(h)  # deepest layer output (before norm + head)
        h_norm = self.norm(h)
        return {
            'logits': self.head(h_norm),
            'deep': h,  # raw output of deepest encoder layer
            'probe_out': self.probe(h),  # probe prediction
        }

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

def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}

def run():
    print("=" * 60)
    print("TEST: Probing Gateway (Probe at Deepest Layer)")
    print(f"Probe weight: {PROBE_WEIGHT}")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformerWithProbe().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []
    main_loss_history = []
    probe_loss_history = []

    start_time = time.time()
    for step in range(N_BATCHES):
        x, y = generate_batch(8, 32)

        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        # --- Beacon pass: main + probe loss ---
        model.zero_grad(set_to_none=True)
        out_beacon = model(x)
        loss_main_beacon = nn.functional.cross_entropy(
            out_beacon['logits'].reshape(-1, out_beacon['logits'].size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        # Probe tries to predict the injected beacon
        pred_beacon = out_beacon['probe_out'].mean(dim=0)  # (dim,)
        true_beacon = beacon.squeeze(0)  # (dim,)
        loss_probe_beacon = nn.functional.mse_loss(pred_beacon, true_beacon)
        loss_beacon = loss_main_beacon + PROBE_WEIGHT * loss_probe_beacon
        loss_beacon.backward()
        grads_beacon = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        # --- Baseline pass: main loss only ---
        model.zero_grad(set_to_none=True)
        out_base = model(x)
        loss_main_base = nn.functional.cross_entropy(
            out_base['logits'].reshape(-1, out_base['logits'].size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss_base = loss_main_base
        loss_base.backward()
        grads_base = collect_grads(model)

        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            per_batch_diffs[name].append(c - b)

        loss_history.append(loss_base.item())
        main_loss_history.append(loss_main_base.item())
        probe_loss_history.append(loss_probe_beacon.item())

        opt.step()

        if (step + 1) % 10 == 0:
            avg_main = sum(main_loss_history[-10:]) / min(10, len(main_loss_history))
            avg_probe = sum(probe_loss_history[-10:]) / min(10, len(probe_loss_history))
            print(f"  Batch {step + 1:2d}/{N_BATCHES}  |  main_loss={avg_main:.4f}  probe_loss={avg_probe:.6f}")

    total_time = time.time() - start_time
    print(f"\nFinal main loss: {main_loss_history[-1]:.4f}")
    print(f"Final probe loss: {probe_loss_history[-1]:.6f}")
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

        # Categorize by layer
        deep_params = [f for f in flips if 'layers.2' in f['param'] or 'probe' in f['param']]
        mid_params = [f for f in flips if 'layers.1' in f['param']]
        early_params = [f for f in flips if 'layers.0' in f['param']]
        head_params = [f for f in flips if 'head' in f['param']]
        emb_params = [f for f in flips if 'emb' in f['param']]
        print(f"\nBy layer:")
        print(f"  Deep (Layer 2 + probe): {len(deep_params)} flips")
        print(f"  Mid (Layer 1): {len(mid_params)} flips")
        print(f"  Early (Layer 0): {len(early_params)} flips")
        print(f"  Head: {len(head_params)} flips")
        print(f"  Embeddings: {len(emb_params)} flips")
    else:
        print("NO FLIPS DETECTED")

    # Show diff trajectory for probe params vs others
    probe_param = 'probe.net.0.weight'
    if probe_param in per_batch_diffs:
        diffs = per_batch_diffs[probe_param]
        print(f"\n--- Probe param diff (last 10 batches) ---")
        for i in range(max(0, len(diffs)-10), len(diffs)):
            print(f"  Batch {i+1}: {diffs[i]:.8f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    run()
