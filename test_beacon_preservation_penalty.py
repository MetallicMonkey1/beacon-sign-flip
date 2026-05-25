"""
Test: Beacon Preservation Penalty

Add a penalty term that forces the model to preserve beacon information
in hidden states while doing the echo task.

Options:
  1 = Final hidden state (before head)
  2 = Intermediate layer (Layer 1 output)
  3 = Per-position hidden states
"""

import sys
from collections import defaultdict
import torch
import torch.nn as nn

DEVICE = "cpu"
N_BATCHES = 70
LR = 1e-3
BEACON_MAGNITUDE = 0.05
PENALTY_LAMBDA = 1.0  # Strength of beacon preservation penalty

class TinyTransformerWithPenalty(nn.Module):
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

    def forward(self, x, return_intermediate=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        if return_intermediate:
            # Return outputs at each layer
            intermediate = []
            for layer in self.layers.layers:
                h = layer(h)
                intermediate.append(h.clone())
            h_norm = self.norm(h)
            return self.head(h_norm), intermediate
        else:
            h = self.layers(h)
            h_norm = self.norm(h)
            return self.head(h_norm)

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

def run_experiment(option):
    option_names = {1: "Final hidden state", 2: "Intermediate layer (Layer 1)", 3: "Per-position hidden states"}
    print("=" * 60)
    print(f"TEST: Beacon Preservation Penalty — Option {option}: {option_names[option]}")
    print(f"Penalty lambda: {PENALTY_LAMBDA}")
    print("=" * 60)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = TinyTransformerWithPenalty().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    per_batch_diffs = defaultdict(list)
    main_loss_history = []
    penalty_loss_history = []

    for step in range(N_BATCHES):
        x, y = generate_batch(8, 32)
        beacon = torch.randn(1, model.dim, device=DEVICE) * BEACON_MAGNITUDE
        true_beacon = beacon.squeeze(0)  # (dim,)

        # --- Beacon pass: main + penalty ---
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
        model.zero_grad(set_to_none=True)

        if option == 1:
            logits = model(x)
            loss_main = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            # Penalty on final hidden state (get it via forward hook or recompute)
            # We'll compute it manually
            h = model.token_emb(x)
            model.token_emb.forward = orig_emb_forward
            h = model.layers(h)
            h_norm = model.norm(h)
            # Penalty: final hidden state should preserve beacon
            pred_beacon = h_norm.mean(dim=(0, 1))  # (dim,)
            loss_penalty = nn.functional.mse_loss(pred_beacon, true_beacon)

        elif option == 2:
            logits, intermediates = model(x, return_intermediate=True)
            model.token_emb.forward = orig_emb_forward
            loss_main = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            # Penalty on Layer 1 output (index 0 in the layers list is Layer 1)
            layer1_out = intermediates[0]
            pred_beacon = layer1_out.mean(dim=(0, 1))
            loss_penalty = nn.functional.mse_loss(pred_beacon, true_beacon)

        elif option == 3:
            logits = model(x)
            model.token_emb.forward = orig_emb_forward
            loss_main = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            # Penalty on final hidden states per-position
            h = model.token_emb(x)
            model.token_emb.forward = orig_emb_forward
            h = model.layers(h)
            h_norm = model.norm(h)
            # Penalty: each position should correlate with beacon
            # We want the hidden states to be beacon-preserving
            pred_beacon = h_norm.mean(dim=0)  # (L, dim)
            target_expanded = true_beacon.unsqueeze(0).expand(pred_beacon.size(0), -1)
            loss_penalty = nn.functional.mse_loss(pred_beacon, target_expanded)

        loss_beacon = loss_main + PENALTY_LAMBDA * loss_penalty
        loss_beacon.backward()
        grads_beacon = collect_grads(model)
        model.token_emb.forward = orig_emb_forward

        # --- Baseline pass: main loss only ---
        model.zero_grad(set_to_none=True)
        logits_base = model(x)
        loss_base = nn.functional.cross_entropy(
            logits_base.reshape(-1, logits_base.size(-1)), y.reshape(-1), ignore_index=-100
        )
        loss_base.backward()
        grads_base = collect_grads(model)

        for name in grads_base:
            b = grads_base.get(name, 0.0)
            c = grads_beacon.get(name, 0.0)
            per_batch_diffs[name].append(c - b)

        main_loss_history.append(loss_base.item())
        penalty_loss_history.append(loss_penalty.item())

        opt.step()

        if (step + 1) % 10 == 0:
            avg_main = sum(main_loss_history[-10:]) / min(10, len(main_loss_history))
            avg_pen = sum(penalty_loss_history[-10:]) / min(10, len(penalty_loss_history))
            print(f"  Batch {step + 1:2d}/{N_BATCHES}  |  main_loss={avg_main:.4f}  penalty={avg_pen:.6f}")

    print(f"\nFinal main loss: {main_loss_history[-1]:.4f}")
    print(f"Final penalty loss: {penalty_loss_history[-1]:.6f}")

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
    return len(flips), flips

if __name__ == "__main__":
    for opt_num in [1, 2, 3]:
        run_experiment(opt_num)
        print("\n")
