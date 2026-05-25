"""
Approach 3: Attention Pattern Analysis

Extract and analyze attention weights from each head at each layer.
For the echo task, we expect a "copy head" that attends to the previous position.
Compare attention patterns with and without beacon to see if beacon affects attention.
"""

import torch
import torch.nn as nn

DEVICE = "cpu"
VOCAB_SIZE = 16
DIM = 64
N_LAYERS = 3
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS
MAX_LEN = 64
FF_MULT = 4

BEACON_MAGNITUDE = 0.05
BATCH_SIZE = 8
SEQ_LEN = 32
LR = 1e-3


class ManualTransformerLayer(nn.Module):
    """Explicit transformer layer that can return attention weights."""
    def __init__(self, dim=64, n_heads=4, ff_mult=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * ff_mult)
        self.linear2 = nn.Linear(dim * ff_mult, dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, return_attention=False):
        B, L, _ = x.shape
        # Generate causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Self-attention
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False,
            attn_mask=causal_mask,
        )
        h = self.norm1(x + self.dropout(attn_out))

        # FFN
        h2 = self.linear2(self.dropout(torch.relu(self.linear1(h))))
        h = self.norm2(h + self.dropout(h2))

        if return_attention:
            return h, attn_weights  # attn_weights: (B, n_heads, L, L)
        return h, None


class TinyTransformerWithAttention(nn.Module):
    """Transformer that returns attention weights."""
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            ManualTransformerLayer(dim, n_heads, ff_mult)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_attention=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        all_attns = []
        for layer in self.layers:
            h, attn = layer(h, return_attention=return_attention)
            if return_attention:
                all_attns.append(attn)

        h = self.norm(h)
        return self.head(h), all_attns


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


def train_model(model, n_batches):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(n_batches):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()

        model.token_emb.forward = orig_emb_forward
    return model


def analyze_attention_pattern(attn_weights):
    """
    attn_weights: (B, n_heads, L, L)
    Returns summary stats for each head.
    """
    B, n_heads, L, _ = attn_weights.shape
    results = {}

    for h in range(n_heads):
        head_attn = attn_weights[:, h, :, :]  # (B, L, L)
        mean_attn = head_attn.mean(dim=0)  # (L, L)

        # For echo task: position i should attend to i-1
        # Check diagonal-1 pattern (attend to previous position)
        diagonal_scores = torch.diag(mean_attn, diagonal=-1)  # attends to prev
        off_diagonal_mean = (mean_attn.sum() - torch.trace(mean_attn)) / (L * L - L)

        results[h] = {
            'prev_pos_attn': diagonal_scores.mean().item(),
            'prev_pos_attn_std': diagonal_scores.std().item(),
            'off_diagonal_mean': off_diagonal_mean.item(),
            'max_attention_idx': mean_attn.argmax(dim=-1).float().mean().item(),
        }

    return results


def compare_attention(model, x, beacon):
    """Compare attention patterns with and without beacon."""
    # With beacon
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    with torch.no_grad():
        _, attn_with = model(x, return_attention=True)
    model.token_emb.forward = orig_emb_forward

    # Without beacon
    with torch.no_grad():
        _, attn_without = model(x, return_attention=True)

    return attn_with, attn_without


def main():
    print("=" * 70)
    print("APPROACH 3: Attention Pattern Analysis")
    print("=" * 70)
    print()

    checkpoints = [10, 30, 50, 70]

    for ckpt in checkpoints:
        torch.manual_seed(42)
        model = TinyTransformerWithAttention(
            vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
        ).to(DEVICE)
        model = train_model(model, ckpt)

        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

        attn_with, attn_without = compare_attention(model, x, beacon)

        print(f"\n--- Checkpoint {ckpt} ---")
        print(f"{'Head':>6} | {'PrevPos Attn':>14} | {'Off-Diag':>10} | {'Beacon Δ':>10}")
        print("-" * 50)

        for layer_idx in range(N_LAYERS):
            print(f"\n  Layer {layer_idx}:")
            stats_with = analyze_attention_pattern(attn_with[layer_idx])
            stats_without = analyze_attention_pattern(attn_without[layer_idx])

            for h in range(N_HEADS):
                prev_with = stats_with[h]['prev_pos_attn']
                prev_without = stats_without[h]['prev_pos_attn']
                off_diag = stats_with[h]['off_diagonal_mean']
                delta = abs(prev_with - prev_without)
                print(f"  {h:>4} | {prev_with:>14.6f} | {off_diag:>10.6f} | {delta:>10.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("'PrevPos Attn' = average attention to previous position (echo pattern).")
    print("'Off-Diag' = average attention to non-diagonal positions.")
    print("'Beacon Δ' = difference in attention pattern with vs without beacon.")
    print()
    print("Expected: High PrevPos Attn in some head = copy head for echo task.")
    print("Expected: Low Beacon Δ post-cascade = beacon doesn't affect attention.")
    print()


if __name__ == "__main__":
    main()
