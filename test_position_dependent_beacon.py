"""
Test: Position-Dependent Beacon

Instead of adding the same beacon to ALL positions, each position gets
a different vector. This breaks common-mode cancellation and forces the
model to actually process the beacon.
"""

from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05
SEQ_LEN = 32

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

def inject_position_beacon(emb_module, beacon_matrix):
    """
    beacon_matrix: (seq_len, dim) — different vector per position
    """
    orig = emb_module.forward
    def hooked(x):
        out = orig(x)  # (B, L, dim)
        return out + beacon_matrix.unsqueeze(0)  # broadcast across batch
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

def run_experiment(position_dependent=False):
    label = "Position-dependent beacon" if position_dependent else "Uniform beacon"
    print("=" * 60)
    print(f"TEST: {label}")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []

    for step in range(N_BATCHES):
        x, y = generate_batch(8, SEQ_LEN)

        if position_dependent:
            # Each position gets a DIFFERENT random vector
            beacon = torch.randn(SEQ_LEN, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        else:
            # Same vector at all positions (baseline comparison)
            beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE

        orig_emb_forward = inject_position_beacon(model.token_emb, beacon)

        # --- Beacon pass ---
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

        # --- Baseline pass (NO beacon) ---
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

    print(f"\nFinal loss: {loss_history[-1]:.4f}")

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

    # Average diff magnitude during dead zone vs late training
    avg_diff_early = sum(sum(per_batch_diffs[name][:30]) / 30 for name in per_batch_diffs) / len(per_batch_diffs)
    avg_diff_late = sum(sum(per_batch_diffs[name][60:]) / 10 for name in per_batch_diffs) / len(per_batch_diffs)
    print(f"\nAvg |diff| batches 1-30:  {avg_diff_early:.8f}")
    print(f"Avg |diff| batches 60-70: {avg_diff_late:.8f}")

    print("\n" + "=" * 60)
    return len(flips)

if __name__ == "__main__":
    print("\n")
    uniform_flips = run_experiment(position_dependent=False)
    print("\n")
    posdep_flips = run_experiment(position_dependent=True)
    print("\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Uniform beacon:        {uniform_flips} parameters flipped")
    print(f"Position-dependent:    {posdep_flips} parameters flipped")
    print("=" * 60)
