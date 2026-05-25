"""
Test: Can the model reconstruct the beacon alongside the main task?

The model must output both:
- main task prediction (echo)
- beacon vector reconstruction

We track whether beacon reconstruction degrades as the main task is learned.
"""

import time
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05

class TinyTransformerWithBeaconHead(nn.Module):
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
        self.head = nn.Linear(dim, vocab_size)  # main task head
        self.beacon_head = nn.Linear(dim, dim)  # beacon reconstruction head

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.layers(h)
        h = self.norm(h)
        return {
            'logits': self.head(h),
            'beacon': self.beacon_head(h),
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
    print("TEST: Beacon Reconstruction Head")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformerWithBeaconHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []
    main_loss_history = []
    beacon_loss_history = []
    beacon_cos_history = []

    start_time = time.time()
    for step in range(N_BATCHES):
        x, y = generate_batch(8, 32)

        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        # --- Beacon pass ---
        model.zero_grad(set_to_none=True)
        out_beacon = model(x)
        loss_main_beacon = nn.functional.cross_entropy(
            out_beacon['logits'].reshape(-1, out_beacon['logits'].size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        # Beacon reconstruction loss: predict the global beacon from all positions
        pred_beacon = out_beacon['beacon'].mean(dim=(0, 1))  # (dim,)
        true_beacon = beacon.squeeze(0)  # (dim,)
        loss_beacon_recon = nn.functional.mse_loss(pred_beacon, true_beacon)
        loss_beacon = loss_main_beacon + loss_beacon_recon
        loss_beacon.backward()
        grads_beacon = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        # --- Baseline pass ---
        model.zero_grad(set_to_none=True)
        out_base = model(x)
        loss_main_base = nn.functional.cross_entropy(
            out_base['logits'].reshape(-1, out_base['logits'].size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        pred_beacon_base = out_base['beacon'].mean(dim=(0, 1))
        loss_beacon_recon_base = nn.functional.mse_loss(pred_beacon_base, true_beacon)
        loss_base = loss_main_base + loss_beacon_recon_base
        loss_base.backward()
        grads_base = collect_grads(model)

        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            per_batch_diffs[name].append(c - b)

        loss_history.append(loss_base.item())
        main_loss_history.append(loss_main_base.item())
        beacon_loss_history.append(loss_beacon_recon_base.item())
        cos_sim = nn.functional.cosine_similarity(pred_beacon_base.unsqueeze(0), true_beacon.unsqueeze(0), dim=1)
        beacon_cos_history.append(cos_sim.item())

        opt.step()

        if (step + 1) % 10 == 0:
            avg_main = sum(main_loss_history[-10:]) / min(10, len(main_loss_history))
            avg_beacon = sum(beacon_loss_history[-10:]) / min(10, len(beacon_loss_history))
            avg_cos = sum(beacon_cos_history[-10:]) / min(10, len(beacon_cos_history))
            print(f"  Batch {step + 1:2d}/{N_BATCHES}  |  main_loss={avg_main:.4f}  beacon_loss={avg_beacon:.6f}  cos_sim={avg_cos:.4f}")

    total_time = time.time() - start_time
    print(f"\nFinal main loss: {main_loss_history[-1]:.4f}")
    print(f"Final beacon loss: {beacon_loss_history[-1]:.6f}")
    print(f"Final beacon cos_sim: {beacon_cos_history[-1]:.4f}")
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

    # Print trajectory
    print("\n--- Beacon Reconstruction Trajectory ---")
    print(f"{'Batch':>6} | {'Main Loss':>10} | {'Beacon Loss':>12} | {'Cos Sim':>10}")
    for i in range(0, N_BATCHES, 5):
        print(f"{i+1:>6} | {main_loss_history[i]:>10.4f} | {beacon_loss_history[i]:>12.6f} | {beacon_cos_history[i]:>10.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    run()
