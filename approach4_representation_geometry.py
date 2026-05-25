"""
Approach 4: Representation Geometry

Analyze the geometric structure of hidden states using:
1. PCA variance distribution
2. Cosine similarity between positions
3. Beacon subspace projection norm
"""

import torch
import torch.nn as nn

DEVICE = "cpu"
VOCAB_SIZE = 16
DIM = 64
N_LAYERS = 3
N_HEADS = 4
MAX_LEN = 64
FF_MULT = 4

BEACON_MAGNITUDE = 0.05
BATCH_SIZE = 8
SEQ_LEN = 32
LR = 1e-3


class TinyTransformerWithIntermediates(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, return_intermediates=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)

        if return_intermediates:
            intermediates = []
            for layer in self.layers.layers:
                h = layer(h)
                intermediates.append(h.clone())
            h_norm = self.norm(h)
            return self.head(h_norm), intermediates
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


def train_model(model, n_batches):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(n_batches):
        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE
        orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()

        model.token_emb.forward = orig_emb_forward
    return model


def compute_pca_variance(hidden_states):
    """
    hidden_states: (B, L, dim) -> flatten to (B*L, dim)
    Returns top-5 eigenvalues of covariance matrix
    """
    flat = hidden_states.reshape(-1, hidden_states.size(-1))  # (B*L, dim)
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.T @ flat) / flat.size(0)
    eigvals = torch.linalg.eigvalsh(cov)
    return eigvals.flip(0)[:5]  # Top 5


def compute_cosine_similarity(hidden_states):
    """
    hidden_states: (B, L, dim)
    Returns average cosine similarity between different positions
    """
    B, L, dim = hidden_states.shape
    # Normalize
    norms = hidden_states.norm(dim=-1, keepdim=True)
    normalized = hidden_states / (norms + 1e-8)

    # Cosine similarity between all position pairs
    sim_matrix = torch.bmm(normalized, normalized.transpose(1, 2))  # (B, L, L)
    # Exclude diagonal
    mask = torch.eye(L, device=sim_matrix.device).bool().unsqueeze(0)
    off_diag = sim_matrix.masked_select(~mask)
    return off_diag.mean().item(), off_diag.std().item()


def compute_beacon_projection(hidden_states, beacon_direction):
    """
    hidden_states: (B, L, dim)
    beacon_direction: (dim,) unit vector
    Returns average projection norm
    """
    # Project each hidden state onto beacon direction
    projections = hidden_states @ beacon_direction  # (B, L)
    return projections.abs().mean().item(), projections.abs().std().item()


def analyze_geometry(model, x, beacon):
    """Extract and analyze hidden states from beacon and no-beacon runs."""
    # Beacon run
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    with torch.no_grad():
        _, intermediates_with = model(x, return_intermediates=True)
    model.token_emb.forward = orig_emb_forward

    # No beacon
    with torch.no_grad():
        _, intermediates_without = model(x, return_intermediates=True)

    # Normalize beacon direction
    beacon_dir = beacon.squeeze(0) / beacon.squeeze(0).norm()

    results = {}
    for layer_idx in range(N_LAYERS):
        h_with = intermediates_with[layer_idx]  # (B, L, dim)
        h_without = intermediates_without[layer_idx]

        # PCA
        eigvals_with = compute_pca_variance(h_with)
        eigvals_without = compute_pca_variance(h_without)

        # Cosine similarity
        cos_sim_with = compute_cosine_similarity(h_with)
        cos_sim_without = compute_cosine_similarity(h_without)

        # Beacon projection
        beacon_proj_with = compute_beacon_projection(h_with, beacon_dir)
        beacon_proj_without = compute_beacon_projection(h_without, beacon_dir)

        results[layer_idx] = {
            'pca_ratio': (eigvals_with[0] / eigvals_with.sum()).item(),
            'pca_top5': eigvals_with[:5].tolist(),
            'cos_sim_mean': cos_sim_with[0],
            'cos_sim_std': cos_sim_with[1],
            'beacon_proj_mean': beacon_proj_with[0],
            'beacon_proj_std': beacon_proj_with[1],
        }

    return results


def main():
    print("=" * 70)
    print("APPROACH 4: Representation Geometry")
    print("=" * 70)
    print()

    checkpoints = [10, 30, 50, 70]

    for ckpt in checkpoints:
        torch.manual_seed(42)
        model = TinyTransformerWithIntermediates(
            vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
        ).to(DEVICE)
        model = train_model(model, ckpt)

        x, y = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

        results = analyze_geometry(model, x, beacon)

        print(f"\n--- Checkpoint {ckpt} ---")
        print(f"{'Layer':>6} | {'PCA 1st EV':>12} | {'Cos Sim':>10} | {'Beacon Proj':>12}")
        print("-" * 50)

        for layer_idx in range(N_LAYERS):
            r = results[layer_idx]
            print(f"{layer_idx:>6} | {r['pca_ratio']:>12.4f} | {r['cos_sim_mean']:>10.4f} | {r['beacon_proj_mean']:>12.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("'PCA 1st EV' = proportion of variance in first principal component.")
    print("'Cos Sim' = average cosine similarity between positions (common-mode structure).")
    print("'Beacon Proj' = average projection of hidden states onto beacon direction.")
    print()
    print("Expected for echo task:")
    print("  - High PCA 1st EV = positions share common structure")
    print("  - High Cos Sim = all positions treated similarly")
    print("  - Low Beacon Proj = beacon direction is not a principal axis")
    print()


if __name__ == "__main__":
    main()
