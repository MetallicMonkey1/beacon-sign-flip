"""
Test: What happens when we remove positional encoding?

The model can't distinguish positions, so it can't learn "copy previous token."
"""

import time
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05

# Model WITHOUT positional encoding
class TinyTransformerNoPos(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.token_emb = nn.Embedding(vocab_size, dim)
        # NO pos_emb
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.token_emb(x)  # NO + self.pos_emb(pos)
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
    y = torch.full((bs, seq_len), -100, dtype=torch.long, device=DEVICE)
    y[:, 1:] = x[:, :-1]
    return x, y

def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}

def run():
    print("=" * 60)
    print("TEST: No Positional Encoding")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformerNoPos().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []

    start_time = time.time()
    for step in range(N_BATCHES):
        x, y = generate_batch(8, 32)

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
        loss_history.append(loss_base.item())

        opt.step()

        if (step + 1) % 10 == 0:
            avg = sum(loss_history[-10:]) / min(10, len(loss_history))
            print(f"  Batch {step + 1:2d}/{N_BATCHES}  |  loss={avg:.4f}")

    total_time = time.time() - start_time
    print(f"\nFinal loss: {loss_history[-1]:.4f}")
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

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    run()
