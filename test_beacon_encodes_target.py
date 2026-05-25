"""
Test: Beacon Encodes the Target Token

The beacon at each position encodes WHICH token to predict.
Without extracting the beacon, the model CANNOT solve the task.
This makes the beacon indispensable.

Task:
- Input: random tokens (irrelevant to output)
- Beacon at position i: a vector that maps to the target token for position i
- Target: decode the beacon to predict the correct token
"""

from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.5
SEQ_LEN = 32
VOCAB_SIZE = 16
DIM = 64

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
        return out + beacon_matrix.unsqueeze(0)
    emb_module.forward = hooked
    return orig

def generate_batch_with_beacon_target(bs, seq_len, vocab_size=16, dim=64):
    """
    Generate random tokens and a beacon that encodes the target.
    The beacon is the ONLY source of target information.
    """
    # Random input tokens (irrelevant)
    x = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)

    # Random target tokens
    y = torch.randint(0, vocab_size, (bs, seq_len), device=DEVICE)

    # Create beacon: for each position, a vector that "points" to the target token
    # We use a learned mapping: each token gets a unique random direction in embedding space
    beacon = torch.zeros(seq_len, dim, device=DEVICE)
    for i in range(seq_len):
        # Seed by target token to create deterministic but unique beacon per token
        torch.manual_seed(2000 + y[0, i].item())
        beacon[i] = torch.randn(dim, device=DEVICE)
    beacon = beacon / beacon.norm(dim=-1, keepdim=True) * BEACON_MAGNITUDE

    return x, y, beacon

def collect_grads(model):
    return {name: param.grad.abs().mean().item()
            for name, param in model.named_parameters()
            if param.grad is not None}

def run_experiment():
    print("=" * 60)
    print("TEST: Beacon Encodes Target Token (Indispensable Beacon)")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformer(vocab_size=VOCAB_SIZE, dim=DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    loss_history = []

    for step in range(N_BATCHES):
        x, y, beacon = generate_batch_with_beacon_target(8, SEQ_LEN, vocab_size=VOCAB_SIZE, dim=DIM)

        # --- Beacon pass: beacon encodes target ---
        orig_emb_forward = inject_position_beacon(model.token_emb, beacon)
        model.zero_grad(set_to_none=True)
        logits_beacon = model(x)
        loss_beacon = nn.functional.cross_entropy(
            logits_beacon.reshape(-1, logits_beacon.size(-1)),
            y.reshape(-1),
        )
        loss_beacon.backward()
        grads_beacon = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        # --- Baseline pass: NO beacon (model can't know target) ---
        model.zero_grad(set_to_none=True)
        logits_base = model(x)
        loss_base = nn.functional.cross_entropy(
            logits_base.reshape(-1, logits_base.size(-1)),
            y.reshape(-1),
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

    # Average diff magnitude
    avg_diff_early = sum(sum(per_batch_diffs[name][:30]) / 30 for name in per_batch_diffs) / len(per_batch_diffs)
    avg_diff_late = sum(sum(per_batch_diffs[name][60:]) / 10 for name in per_batch_diffs) / len(per_batch_diffs)
    print(f"\nAvg diff batches 1-30:  {avg_diff_early:.8f}")
    print(f"Avg diff batches 60-70: {avg_diff_late:.8f}")

    print("\n" + "=" * 60)
    return len(flips)

if __name__ == "__main__":
    run_experiment()
