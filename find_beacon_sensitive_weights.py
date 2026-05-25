"""
Find Individual Parameters That Interact With the Beacon

For each weight in the model, compute:
1. Gradient with beacon
2. Gradient without beacon
3. Rank by |grad_beacon - grad_base|

This reveals the EXACT parameters that "see" the beacon.
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
LR = 1e-3


class ManualTransformerLayer(nn.Module):
    def __init__(self, dim=64, n_heads=4, ff_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * ff_mult)
        self.linear2 = nn.Linear(dim * ff_mult, dim)

    def forward(self, x):
        B, L, _ = x.shape
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        h = self.norm1(x + attn_out)
        h2 = self.linear2(torch.relu(self.linear1(h)))
        return self.norm2(h + h2)


class BeaconSensitiveTransformer(nn.Module):
    def __init__(self, vocab_size=16, dim=64, n_layers=3, n_heads=4, max_len=64, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
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

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
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


def train_model(model, n_batches):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(n_batches):
        x, y = generate_batch(8, 32, VOCAB_SIZE)
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


def compute_individual_beacon_sensitivity(model, x, y):
    """
    For each individual parameter, compute |grad_beacon - grad_base|.
    Returns sorted list of (param_name, flat_index, weight_value, grad_diff, grad_beacon, grad_base).
    """
    beacon = torch.randn(1, DIM, device=DEVICE) * BEACON_MAGNITUDE

    # Beacon run
    orig_emb_forward = inject_beacon_into_embedding(model.token_emb, beacon)
    model.zero_grad(set_to_none=True)
    logits_beacon = model(x)
    loss_beacon = nn.functional.cross_entropy(
        logits_beacon.reshape(-1, logits_beacon.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )
    loss_beacon.backward()
    grads_beacon = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_beacon[name] = param.grad.clone().flatten()
    model.token_emb.forward = orig_emb_forward

    # Base run
    model.zero_grad(set_to_none=True)
    logits_base = model(x)
    loss_base = nn.functional.cross_entropy(
        logits_base.reshape(-1, logits_base.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    )
    loss_base.backward()
    grads_base = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_base[name] = param.grad.clone().flatten()

    # Compute sensitivity for each parameter
    sensitivities = []
    for name, param in model.named_parameters():
        if name not in grads_beacon:
            continue

        g_b = grads_beacon[name]
        g_base = grads_base[name]
        diff = (g_b - g_base).abs()
        weights = param.data.flatten()

        for i in range(diff.numel()):
            sensitivities.append({
                'param_name': name,
                'flat_index': i,
                'weight_value': weights[i].item(),
                'grad_diff': diff[i].item(),
                'grad_beacon': g_b[i].item(),
                'grad_base': g_base[i].item(),
            })

    # Sort by grad_diff descending
    sensitivities.sort(key=lambda x: x['grad_diff'], reverse=True)
    return sensitivities


def analyze_beacon_weights(sensitivities, model):
    """Analyze which specific weights are most beacon-sensitive."""
    print("=" * 70)
    print("INDIVIDUAL PARAMETERS THAT INTERACT WITH THE BEACON")
    print("=" * 70)
    print(f"\nTotal parameters analyzed: {len(sensitivities)}")
    print(f"Beacon magnitude: {BEACON_MAGNITUDE}")

    # Overall statistics
    all_diffs = [s['grad_diff'] for s in sensitivities]
    print(f"\nGradient difference statistics:")
    print(f"  Max: {max(all_diffs):.8f}")
    print(f"  Min: {min(all_diffs):.8f}")
    print(f"  Mean: {sum(all_diffs)/len(all_diffs):.8f}")
    print(f"  Std: {(torch.tensor(all_diffs).std().item()):.8f}")

    # Top 30 most sensitive individual weights
    print(f"\n{'='*70}")
    print("TOP 30 MOST BEACON-SENSITIVE INDIVIDUAL WEIGHTS")
    print(f"{'='*70}")
    print(f"{'Rank':>5} | {'Param':>35} | {'Index':>7} | {'Grad Diff':>12} | {'Grad Beacon':>12} | {'Grad Base':>12} | {'Weight':>12}")
    print("-" * 110)

    for i in range(min(30, len(sensitivities))):
        s = sensitivities[i]
        print(f"{i+1:>5} | {s['param_name']:>35} | {s['flat_index']:>7} | {s['grad_diff']:>12.8f} | {s['grad_beacon']:>12.8f} | {s['grad_base']:>12.8f} | {s['weight_value']:>12.8f}")

    # Analyze by component
    print(f"\n{'='*70}")
    print("BEACON SENSITIVITY BY COMPONENT")
    print(f"{'='*70}")

    component_sens = {}
    for s in sensitivities:
        name = s['param_name']
        # Categorize by component type
        if 'self_attn.in_proj_weight' in name:
            cat = 'Attention Q/K/V'
        elif 'self_attn.out_proj.weight' in name:
            cat = 'Attention Output'
        elif 'self_attn.out_proj.bias' in name:
            cat = 'Attention Output Bias'
        elif 'linear1.weight' in name:
            cat = 'FFN In'
        elif 'linear1.bias' in name:
            cat = 'FFN In Bias'
        elif 'linear2.weight' in name:
            cat = 'FFN Out'
        elif 'linear2.bias' in name:
            cat = 'FFN Out Bias'
        elif 'norm1' in name or 'norm2' in name:
            cat = 'LayerNorm'
        elif 'token_emb.weight' in name:
            cat = 'Token Embedding'
        elif 'pos_emb.weight' in name:
            cat = 'Positional Embedding'
        elif 'head.weight' in name:
            cat = 'Head Weight'
        elif 'head.bias' in name:
            cat = 'Head Bias'
        else:
            cat = 'Other'

        if cat not in component_sens:
            component_sens[cat] = []
        component_sens[cat].append(s['grad_diff'])

    # Sort by total sensitivity
    print(f"\n{'Component':>25} | {'Total Sens':>12} | {'Mean Sens':>12} | {'Max Sens':>12} | {'Count':>8}")
    print("-" * 80)
    sorted_components = sorted(component_sens.items(), key=lambda x: sum(x[1]), reverse=True)
    for cat, diffs in sorted_components:
        total = sum(diffs)
        mean = total / len(diffs)
        max_val = max(diffs)
        count = len(diffs)
        print(f"{cat:>25} | {total:>12.6f} | {mean:>12.8f} | {max_val:>12.8f} | {count:>8}")

    # Per-layer attention breakdown
    print(f"\n{'='*70}")
    print("PER-LAYER ATTENTION WEIGHT SENSITIVITY")
    print(f"{'='*70}")

    for layer_idx in range(N_LAYERS):
        layer_prefix = f'layers.{layer_idx}.self_attn'
        layer_sens = [s for s in sensitivities if layer_prefix in s['param_name']]
        if layer_sens:
            total = sum(s['grad_diff'] for s in layer_sens)
            max_val = max(s['grad_diff'] for s in layer_sens)
            print(f"Layer {layer_idx}: total={total:.6f}, max={max_val:.8f}, count={len(layer_sens)}")

            # Top 5 weights in this layer
            layer_sens_sorted = sorted(layer_sens, key=lambda x: x['grad_diff'], reverse=True)
            for i in range(min(5, len(layer_sens_sorted))):
                s = layer_sens_sorted[i]
                print(f"    #{i+1}: {s['param_name']}[{s['flat_index']}] diff={s['grad_diff']:.8f}")

    # Show specific weight values for top beacon-sensitive parameters
    print(f"\n{'='*70}")
    print("SPECIFIC WEIGHT VALUES FOR TOP BEACON-SENSITIVE PARAMETERS")
    print(f"{'='*70}")

    for i in range(min(10, len(sensitivities))):
        s = sensitivities[i]
        name = s['param_name']
        idx = s['flat_index']
        diff = s['grad_diff']
        w = s['weight_value']
        g_b = s['grad_beacon']
        g_base = s['grad_base']

        # Try to map flat index to meaningful coordinates
        param = dict(model.named_parameters())[name]
        shape = param.shape

        # Convert flat index to multi-dimensional indices
        indices = []
        remaining = idx
        for dim in reversed(shape):
            indices.append(remaining % dim)
            remaining = remaining // dim
        indices = list(reversed(indices))

        print(f"\nRank {i+1}: {name}")
        print(f"  Shape: {shape}")
        print(f"  Indices: {indices}")
        print(f"  Weight value: {w:.8f}")
        print(f"  Gradient (with beacon): {g_b:.8f}")
        print(f"  Gradient (no beacon):   {g_base:.8f}")
        print(f"  |Diff|: {diff:.8f}")

        # Special interpretation for attention weights
        if 'self_attn.in_proj_weight' in name:
            # in_proj = [W_Q; W_K; W_V]
            dim = param.shape[1]
            if indices[0] < DIM:
                proj_type = 'W_Q'
                head = indices[0] // HEAD_DIM
                head_row = indices[0] % HEAD_DIM
            elif indices[0] < 2 * DIM:
                proj_type = 'W_K'
                head = (indices[0] - DIM) // HEAD_DIM
                head_row = (indices[0] - DIM) % HEAD_DIM
            else:
                proj_type = 'W_V'
                head = (indices[0] - 2 * DIM) // HEAD_DIM
                head_row = (indices[0] - 2 * DIM) % HEAD_DIM
            print(f"  -> This is {proj_type}, Head {head}, row {head_row}, col {indices[1]}")

        elif 'self_attn.out_proj.weight' in name:
            head = indices[1] // HEAD_DIM
            head_col = indices[1] % HEAD_DIM
            print(f"  -> This is W_O, Head {head}, row {indices[0]}, col {head_col} (within head)")

        elif 'linear1.weight' in name:
            print(f"  -> FFN input weight: row {indices[0]} (neuron), col {indices[1]} (input dim)")

        elif 'linear2.weight' in name:
            print(f"  -> FFN output weight: row {indices[0]} (output dim), col {indices[1]} (neuron)")


def main():
    print("=" * 70)
    print("FINDING BEACON-SENSITIVE INDIVIDUAL WEIGHTS")
    print("=" * 70)
    print("\nTraining model...")

    torch.manual_seed(42)
    model = BeaconSensitiveTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, max_len=MAX_LEN, ff_mult=FF_MULT
    ).to(DEVICE)
    model = train_model(model, 70)

    print("\nComputing beacon sensitivity for every individual parameter...")
    x, y = generate_batch(8, 32, VOCAB_SIZE)
    sensitivities = compute_individual_beacon_sensitivity(model, x, y)

    analyze_beacon_weights(sensitivities, model)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
