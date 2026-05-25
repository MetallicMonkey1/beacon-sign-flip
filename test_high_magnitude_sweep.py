"""
Test: High Magnitude Sweep

Sweep beacon magnitude from 1.0 to 10.0 in 0.5 increments to map
the breakdown of the sign-flip cascade as the perturbation dominates.
"""

from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3

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
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
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
    y = torch.full((bs, seq_len), -100, dtype=torch.long, device=DEVICE)
    y[:, 1:] = x[:, :-1]
    return x, y

def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}

def run_experiment(magnitude):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    per_batch_diffs = defaultdict(list)
    loss_history = []

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
        loss_history.append(loss_base.item())

        opt.step()

    # Find flips
    flips = []
    for name, diffs in per_batch_diffs.items():
        for i in range(30, len(diffs) - 1):
            if diffs[i] > 0 and diffs[i + 1] < 0:
                flips.append({"param": name, "batch": i + 1})
                break

    first_flip = min(flips, key=lambda x: x["batch"]) if flips else None
    flip_batches = sorted(set(f['batch'] for f in flips)) if flips else []

    return {
        'magnitude': magnitude,
        'final_loss': loss_history[-1],
        'flips': len(flips),
        'total': len(per_batch_diffs),
        'first_flip': first_flip['batch'] if first_flip else None,
        'flip_batches': flip_batches,
    }

if __name__ == "__main__":
    magnitudes = [round(x * 0.5 + 1.0, 1) for x in range(19)]  # 1.0, 1.5, ..., 10.0

    print("=" * 80)
    print("HIGH MAGNITUDE SWEEP")
    print("=" * 80)
    print(f"{'Magnitude':>10} | {'Final Loss':>10} | {'Flips':>7} | {'First Flip':>12} | {'Cascade Width':>14}")
    print("-" * 80)

    results = []
    for mag in magnitudes:
        res = run_experiment(mag)
        results.append(res)
        width = len(res['flip_batches']) if res['flip_batches'] else 0
        first = str(res['first_flip']) if res['first_flip'] else "None"
        print(f"{mag:>10.1f} | {res['final_loss']:>10.4f} | {res['flips']:>7} | {first:>12} | {width:>14}")

    print("=" * 80)
