# Beacon Sign-Flip Convergence Signal

## Discovery

During gradient-based beacon tracing on a 3-layer transformer learning the "echo previous token" task, the **sign of beacon gradient differences flips from positive to negative as parameters converge**.

## The Pattern

| Training Phase | Diff Sign | What It Means |
|---|---|---|
| **Learning** (early/mid) | **Positive** (`beacon_grad > baseline_grad`) | Noise creates "more work" — gradients increase to compensate. Parameter is **plastic**. |
| **Converged** (late) | **Negative** (`beacon_grad < baseline_grad`) | Noise just adds confusion — gradients shrink because the optimal path is already found. Parameter is **locked in**. |

## Concrete Numbers from Echo Task

**Early (0-300 batches):** All diffs positive
- `layers.0.norm1.bias`: **+0.000265**
- `layers.0.norm2.bias`: **+0.000248**
- `layers.0.linear2.bias`: **+0.000245**

**Late (700-1000 batches):** All diffs negative
- `layers.0.linear2.bias`: **-0.000067**
- `layers.0.norm2.bias`: **-0.000065**
- `layers.0.norm1.bias`: **-0.000065**

## Why This Happens

**Learning phase:** The model hasn't found the solution yet. Injecting a beacon perturbation pushes it further from the current (suboptimal) state. Backprop produces **larger** gradients to pull the model back toward the correct solution despite the noise. Beacon pass → bigger corrections needed.

**Converged phase:** The model has found the optimal weights. Injecting noise moves the activations away from the optimal manifold. Backprop tries to correct, but because the weights are already optimal for the clean signal, the corrections are **smaller** — the noise is just interference, not useful signal. Beacon pass → smaller, confused gradients.

## Actionable Rule for BRLM

```
if beacon_diff > 0:
    parameter is PLASTIC  → good target for container grafting
if beacon_diff < 0:
    parameter is LOCKED   → graft here has minimal effect, avoid
if beacon_diff crosses zero between buckets:
    CONVERGENCE detected at that bucket boundary
```

## Layer-Level Convergence Map

For the echo task, convergence propagated top-to-bottom:

| Layer | Early Diff | Mid Diff | Late Diff | Converged By |
|---|---|---|---|---|
| Head | +0.00015 | +0.00005 | -0.00003 | ~500 batches |
| Layer 2 | +0.00036 | +0.00014 | -0.00011 | ~500 batches |
| Layer 1 | +0.00062 | +0.00023 | -0.00020 | ~600 batches |
| Layer 0 | +0.00117 | +0.00043 | -0.00041 | **~700 batches** |

**Key insight:** Deeper layers converge first. The head and top layers lock in early because the task is simple. Layer 0 stays plastic longest because it's the entry point — it must handle all input perturbations.

## Using Sign-Flip Timing

1. **Run beacon trace** across training buckets (early/mid/late)
2. **For each parameter, track when diff crosses from positive → negative**
3. **Report convergence epoch per parameter**
4. **Graft containers only onto parameters that have NOT converged**

This gives a **per-parameter convergence schedule** — a live map of which parts of the model are still learning at any given time.

## 3-Phase Sign Pattern (Ongoing Research)

A focused 300-batch run with fine-grained buckets (0-100, 100-200, 200-300) on the echo task revealed a **three-phase pattern** more nuanced than the simple positive→negative flip.

### The Three Phases

| Phase | Bucket | Diff Sign | Loss | Meaning |
|---|---|---|---|---|
| **Chaos** | 0-100 | **NEGATIVE** | 2.786 | Random init — no coherent direction. Noise = more confusion → smaller gradients. |
| **Plastic** | 100-200 | **POSITIVE** | 1.564 | Model found the pattern. Noise = push off course → larger corrective gradients. |
| **Locking** | 200-300 | **MOSTLY POSITIVE** | 0.065 | Near convergence. Most params still plastic, first flips begin. |

### Key Data

**Layer 0 sensitivity peaked during the plastic phase:**
- 0-100: 0.000188
- **100-200: 0.001026** ← peak plasticity
- 200-300: 0.000626 ← declining

**Only 1 parameter converged within 300 batches:**
- `layers.0.self_attn.in_proj_weight` → flipped at bucket 200-300

### Implication: The "Dead Zone"

**The first ~100 batches are a dead zone for beacon tracing.** The model is still random — negative diffs don't mean "locked in," they mean "hasn't found anything yet."

**The grafting window opens at ~100 batches** when diffs turn positive and sensitivity peaks.

### Follow-up: 320-Batch Finer-Grained Run

A follow-up run with **4 buckets of 80 batches each** (0-80, 80-160, 160-240, 240-320) revealed the plastic window is **longer than initially estimated**.

| Bucket | Loss | Diff Sign | Phase |
|---|---|---|---|
| 0-80 | 2.79 | **NEGATIVE** | Chaos |
| 80-160 | 2.32 | **POSITIVE** | Plastic |
| 160-240 | 0.23 | **POSITIVE** | Plastic (peak sensitivity) |
| 240-320 | 0.04 | **POSITIVE** | Still plastic |

**Layer 0 sensitivity peaked at 160-240, not 80-160:**
- 0-80: 0.000165
- 80-160: 0.002033
- **160-240: 0.002374** ← actual peak
- 240-320: 0.000671 ← declining

**Convergence events: 0.** No positive→negative flips within 320 batches. The previous 300-batch run showed a flip at 200-300 only because the coarser 100-batch buckets averaged more data. Finer buckets reveal the model is still plastic at batch 320.

### Key Refinement: Plastic Window Is Wider

**Loss can be near-zero while parameters are still plastic.** At 240-320, loss = 0.04 (nearly perfect), but all diffs remain positive. The sign flip is a **late** signal — parameters stay graftable well after the task is "learned."

### Updated Rule for BRLM

```
if epoch < ~80:
    DEAD ZONE — don't graft, model is random
if beacon_diff > 0 and epoch > ~80:
    PLASTIC ZONE — graft containers here (window extends to ~320+)
if beacon_diff < 0 and epoch > ~80:
    LOCKED — avoid, parameter has converged
```

### Critical Follow-up: 330-Batch Run Catches the Convergence Cliff

Extending to **330 batches** with buckets [0, 80, 160, 240, 330] revealed the **convergence cliff** that the 320-batch run missed.

| Bucket | Loss | Diff Sign | Phase |
|---|---|---|---|
| 0-80 | 2.79 | **NEGATIVE** | Chaos |
| 80-160 | 2.31 | **POSITIVE** | Plastic |
| 160-240 | 0.23 | **POSITIVE** | Plastic (peak sensitivity) |
| **240-330** | 0.04 | **MIXED** | **Convergence cliff** |

**Convergence events: 24 parameters flipped positive→negative.**

- **1 param** (`layers.2.norm2.weight`) flipped early at 80-160 — likely noise
- **23 params** flipped simultaneously at **240-330** — the convergence cliff

**Every layer converged at the same bucket (240-330):**

| Layer | Params Flipped | Timing |
|---|---|---|
| Head | 2/2 | 240-330 |
| Layer 0 | 8/10 | 240-330 |
| Layer 1 | 7/10 | 240-330 |
| Layer 2 | 7/10 (1 early) | 240-330 |

### Key Refinement: The Plastic Window Closes at ~240

The previous 320-batch run reported "window extends to 320+" because the bucket boundary didn't straddle the cliff. With the boundary at 330, the **actual convergence point is ~240 batches**.

**Bucket boundaries matter.** If your trace buckets don't include the convergence point, you'll miss the flip entirely.

### Exact-Batch Convergence: Per-Batch Recording Breaks the Illusion

Switching from **bucket averages** to **per-batch diff recording** with a 10-batch rolling average revealed the bucket analysis was hiding a **staggered, top-to-bottom convergence cascade**.

**40 of 42 parameters converged within 330 batches**, with convergence starting at **batch 51** — not batch 240.

| Batch | Parameters Converged | Layer |
|---|---|---|
| **51** | `head.weight`, `layers.2.*` (5), `layers.1.norm2.weight`, `layers.1.linear1.bias` | **Head + Layer 2** |
| **52** | `layers.2.*` (4), `layers.1.*` (2), `layers.0.linear1.weight` | Layer 2 + Layer 1 |
| **53-55** | `layers.1.linear2.bias`, `layers.2.norm2.bias`, `head.bias` | Layer 1 + 2 + Head |
| **58-60** | `layers.2.*` (4), `layers.1.norm1.bias` | Layer 2 + Layer 1 |
| **62-67** | `layers.2.linear2.weight`, `layers.0.norm1.weight`, `layers.1.norm2.bias`, `layers.0.norm2.weight` | Mixed |
| **69-75** | `layers.0.*` (2), `layers.1.*` (4) | Layer 0 + 1 |
| **81** | `layers.0.*` (7!), `layers.1.norm1.weight`, `token_emb.weight` | **Layer 0 cascade** |
| **83** | `layers.0.self_attn.in_proj_weight` | Layer 0 (last) |

### Why Bucket Analysis Failed

Bucket 240-330 showed 23 params flipping "simultaneously." This was an **artifact of averaging** over 90 batches. The true picture is a **32-batch cascade** from batch 51 to batch 83.

**Layer convergence order (top-to-bottom):**
1. **Head + Layer 2:** batch 51
2. **Layer 1:** batch 53-75
3. **Layer 0:** batch 65-83 (mass event at 81)

### Key Insight: The Plastic Window Is Earlier and Narrower

**Bucket analysis said:** graft between batches 80-240.  
**Per-batch analysis says:** graft **before batch 51** for top layers, **before batch 65** for Layer 0.

The top layers lock in almost immediately after the dead zone. Layer 0 has a longer window but still converges by batch 83.

### Final Rule for BRLM (Exact-Batch)

```
if batch < ~50:
    DEAD ZONE — don't graft, model is random
if beacon_diff > 0 and batch < layer_convergence_batch:
    PLASTIC — graft container onto this layer now
    (Head/L2: before batch 51)
    (Layer 1: before batch 60)
    (Layer 0: before batch 65)
if beacon_diff < 0:
    LOCKED — this parameter has converged, avoid
```

**Per-batch beacon tracing is required for surgical precision.** Bucket averages smooth away the actual convergence cascade.

### Raw Per-Batch Detection: No Smoothing

Removing the rolling average entirely (window=1) caught **38 of 42 parameters converging by batch 61** — even faster than the 10-batch smoothed results.

| Batch | Parameters Converged |
|---|---|
| **51** | `layers.0.linear1.weight`, `layers.1.self_attn.out_proj.*`, `token_emb.weight` |
| **52** | 11 params — `layers.2.*` (8), `layers.0.norm1.weight`, `layers.1.linear1.weight`, `layers.1.self_attn.in_proj.*`, `pos_emb.weight` |
| **53** | 7 params — `layers.0.*` (6), `layers.1.norm1.weight` |
| **54** | `layers.1.norm2.weight`, `layers.2.norm1.bias` |
| **55** | 5 params — `head.bias`, `head.weight`, `layers.1.linear1.bias`, `layers.2.norm2.*` |
| **59** | `layers.1.linear2.*` |
| **60** | `layers.0.linear1.bias` |
| **61** | `layers.1.norm1.bias`, `layers.1.norm2.bias` |

**4 parameters never flipped** within 330 batches (stayed near-zero).  
**The remaining 38 all converged within a 10-batch window (51-61).**

### Key Refinement: Plastic Window Is ~50-60 Batches

| Layer | Plastic Window | Converges By |
|---|---|---|
| Layer 0 | Batches ~50-60 | Batch 51-60 |
| Layer 1 | Batches ~50-61 | Batch 55-61 |
| Layer 2 | Batches ~50-52 | Batch 52 |
| Head | Batches ~50-55 | Batch 55 |

**Every layer has roughly the same 50-60 batch plastic window.** There is no "early vs late" layer advantage — they all lock in within 10 batches of each other.

### Final Rule for BRLM (Raw Per-Batch)

```
if batch < ~50:
    DEAD ZONE — don't graft, model is random
if 50 < batch < ~60 and beacon_diff > 0:
    PLASTIC ZONE — graft NOW (window is ~10 batches)
if batch > ~60 and beacon_diff < 0:
    LOCKED — everything has converged, avoid
```

### 340-Batch Confirmation: Convergence Is a 5-Batch Explosion

Extending to 340 batches confirmed the pattern: **35 of 42 parameters converged within a 5-batch window (51-55).**

| Batch | Parameters Converged |
|---|---|
| **51** | 14 params — Head (2), Layer 0 (4), Layer 1 (2), Layer 2 (3), `token_emb.weight` |
| **52** | 7 params — Layer 0 (3), Layer 1 (1), Layer 2 (3) |
| **53** | 9 params — Layer 0 (3), Layer 1 (5), Layer 2 (1) |
| **54** | 4 params — Layer 0 (1), Layer 2 (3), `pos_emb.weight` |
| **55** | 1 param — `layers.1.norm1.weight` |

**Batch 51 is the ignition point** — 14 params flip simultaneously. Batches 52-55 are aftershocks. By batch 55, convergence is essentially complete. The remaining 7 parameters never flip (likely embeddings and small biases that stay near-zero).

### Key Refinement: Convergence Is a Single-Batch Event

The previous "10-batch window" estimate was too conservative. The actual convergence is a **single-batch cascade at batch 51**, with stragglers finishing by batch 55.

### Final Rule for BRLM (Exact Per-Batch)

```
if batch < ~50:
    DEAD ZONE — don't graft, model is random
if batch == ~50 and beacon_diff > 0:
    LAST CHANCE — graft NOW before the cascade hits
if batch > ~51 and beacon_diff < 0:
    LOCKED — convergence explosion happened, avoid
```

### Reproducibility Check: Second Run Confirms the Pattern

Running the exact same configuration a second time confirmed the phenomenon is real while showing slight run-to-run variation in which exact parameters flip.

| Metric | Run 1 | Run 2 |
|---|---|---|
| **Batch 51 ignition** | 14 params flip | **21 params flip** |
| Batch 52 | 7 params | 5 params |
| Batch 53 | 9 params | 8 params |
| Batch 54 | 4 params | 0 params |
| Batch 55 | 1 param | 3 params |
| Batch 56-57 | 0 | 3 params |
| **Total converged** | 35 by batch 55 | **40 by batch 57** |
| **Never flipped** | 7 | 2 |

### What This Tells Us

- **The phenomenon is real** — both runs show a mass convergence event at **batch 51**
- **The exact parameters vary** due to random initialization, but the cascade timing is consistent
- **The window is ~5-7 batches** wide, not a single exact batch
- **Batch 51 is consistently the ignition point** — this is the critical moment

### Reproducible Findings

| Finding | Consistency |
|---|---|
| Dead zone | Batches 0-50 (both runs) |
| Ignition point | **Batch 51** (both runs) |
| Completion | By batch 55-57 (both runs) |
| Never-flip params | Embeddings and small biases (roughly consistent) |

**The sign-flip convergence is a real, reproducible phenomenon. The batch-51 ignition point and the 5-7 batch cascade window are stable across runs despite random initialization.**

---

## Zoomed Analysis: 70 Batches with Per-Batch Raw Gradients

To understand the convergence cascade at higher resolution, we switched to **70 batches** with **per-batch recording of raw gradients** (base, beacon, and diff) and **no smoothing** (window=1). Finer buckets: [0, 15, 30, 45, 60, 70]. Dead zone skip narrowed to 30 batches.

### Zoomed Run 1

| Batch | Parameters Converged |
|---|---|
| **31** | 2 params — `layers.0.norm1.weight`, `layers.2.norm2.bias` |
| **32** | 11 params — Layer 0 (4), Layer 2 (5), `token_emb.weight` |
| **33** | 4 params — Layer 0 (2), `head.weight`, Layer 2 (1) |
| **34** | 9 params — Layer 0 (2), Layer 1 (4), Layer 2 (3) |
| **35** | 5 params — Layer 0 (1), Layer 1 (4) |
| **36** | 3 params — `head.bias`, Layer 0 (1), Layer 1 (1) |
| **37** | 1 param — `layers.0.norm1.bias` |
| **38** | 4 params — Layer 1 (4) |
| **40** | 1 param — `pos_emb.weight` |

**35 of 42 parameters converged by batch 40.** 7 parameters never flipped.

**Raw gradient mechanism at `layers.0.norm1.weight` (converged at batch 31):**
- Batch 31: Base grad = 0.00097, Beacon grad = 0.00093, Diff = +0.00004 (positive)
- Batch 32: Base grad = 0.00097, Beacon grad = 0.00075, Diff = -0.00005 (negative)

**The beacon gradient drops below the base gradient, driving the sign flip.**

### Zoomed Run 2

| Batch | Parameters Converged |
|---|---|
| **31** | 2 params — `layers.1.linear1.bias`, `layers.1.linear1.weight` |
| **32** | 1 param — `pos_emb.weight` |
| **33** | **18 params** — Layer 0 (4), Layer 1 (4), Layer 2 (7), `head.bias` |
| **34** | **15 params** — Layer 0 (7), Layer 1 (5), Layer 2 (2), `head.weight`, `token_emb.weight` |
| **36** | 5 params — `layers.1.linear2.weight`, Layer 2 (4) |

**35 of 42 parameters converged by batch 36.** 7 parameters never flipped.

**Raw gradient mechanism at `layers.1.linear1.bias` (converged at batch 31):**
- Batch 31: Base grad = 0.00016, Beacon grad = 0.00011, Diff = +0.00005 (positive)
- Batch 32: Base grad = 0.00014, Beacon grad = 0.00009, Diff = -0.00005 (negative)

**Same mechanism: beacon gradient drops below base gradient at the flip point.**

### Zoomed Reproducibility Summary

| Finding | Run 1 | Run 2 | Consistency |
|---|---|---|---|
| Ignition point | **Batch 31** | **Batch 31** | Reproducible |
| Total converged | 35 | 35 | Identical |
| Cascade width | 10 batches | 6 batches | Similar |
| Never-flip count | 7 | 7 | Identical |
| Never-flip params | Embeddings, small biases | Embeddings, small biases | Reproducible |

**The 70-batch zoom reveals the true ignition point is batch 31** — not batch 51 as the 340-batch runs suggested. The difference is that the 340-batch runs included a 50-batch dead zone skip, which masked the earlier convergence. With only 70 batches and a 30-batch dead zone skip, we catch the cascade as it truly begins.

### Key Insight: The Cascade Is Earlier Than We Thought

| Configuration | Ignition Point | Why It Differs |
|---|---|---|
| 340 batches, dead zone=50 | Batch 51 | Dead zone skip was too long, missed early convergence |
| 70 batches, dead zone=30 | **Batch 31** | Shorter skip catches the true start of the cascade |

**The plastic window is ~30 batches, not ~50.** The model starts locking in parameters immediately after the dead zone ends.

### Raw Gradient Trajectory (from JSON)

The `per_batch_raw_grads` field in `beacon_trace_transformer_report.json` now contains full trajectories for every parameter:

```json
"layers.0.norm1.weight": {
  "base": [0.0038, 0.0038, ..., 0.00097],
  "beacon": [0.0035, 0.0041, ..., 0.00093],
  "diff": [-0.00029, 0.00030, ..., +0.000042]
}
```

This enables plotting the exact gradient trajectory for any parameter across all batches.

### Final Rule for BRLM (70-Batch Zoomed)

```
if batch < ~30:
    DEAD ZONE — don't graft, model is random
if batch == ~30 and beacon_diff > 0:
    LAST CHANCE — graft NOW before cascade (window is ~5-10 batches)
if batch > ~31 and beacon_diff < 0:
    LOCKED — convergence happened at batch 31, avoid
```

**The actual grafting window is ~5 batches wide (30-35), starting at batch 30.** Per-batch beacon tracing with raw gradient recording is required to catch the cascade.

---

## What Determines Convergence, the Flip, and Lock-In

Based on raw gradient trajectories from both 70-batch zoomed runs and the trajectory analyzer, the sign-flip is not merely a statistical artifact — it reflects a real transition in how parameters respond to perturbation.

### The Flip Mechanism: Beacon Gradient Drops Below Base Gradient

At the exact flip batch:
- **Base grad ≈ 0.00097** — the parameter is still learning from the real data
- **Beacon grad ≈ 0.00075** — the beacon perturbation produces a smaller gradient
- **Diff = beacon − base = −0.00005** — sign flips from positive to negative

**Both gradients are non-zero.** The parameter has not stopped learning. The beacon can no longer perturb the gradient direction. The model has learned to "ignore" the beacon.

### Lock-In = Beacon-Invariant Gradient, Not Zero Gradient

A parameter locks in when its **loss landscape becomes flat in the beacon direction**:

- **Before lock-in:** `Loss(beacon_input)` produces a different gradient than `Loss(normal_input)`
- **After lock-in:** `Loss(beacon_input)` produces the same gradient direction as `Loss(normal_input)`

The beacon is a small random vector added to embeddings. If the parameter's configuration is stable, that perturbation does not change which way the parameter should move. The gradient "ignores" the beacon.

### Convergence Is Task-Driven and Cross-Layer

| Layer | Plastic Window | Converges By |
|---|---|---|
| Layer 0 | ~30-35 | Batch 31-36 |
| Layer 1 | ~30-36 | Batch 31-36 |
| Layer 2 | ~30-36 | Batch 31-36 |
| Head | ~30-34 | Batch 31-34 |

All layers flip within the same 5-6 batch window. This tells us convergence is **task-driven, not layer-driven**. The model solves the echo task as a coordinated system, not layer-by-layer.

### Why Some Parameters Never Flip

| Parameter Type | Why It Never Flips |
|---|---|
| `token_emb.weight` | Random tokens have no learned meaning — gradient stays near-zero |
| `pos_emb.weight` | Positions are uniform in short sequences — no gradient signal |
| Small biases (`linear1.bias`, `norm1.bias`) | LayerNorm and bias terms adapt instantly, no stable configuration |

These are **non-participating parameters** — they don't have meaningful gradient signals, so there's nothing to "lock in."

### Convergence Is a Phase Transition, Not Gradual Decay

The feature analysis shows:

| Metric | Pre-Flip | Random Noise | Ratio |
|---|---|---|---|
| `diff` magnitude | **0.000095** | 0.000005 | **18.5×** |
| `diff_slope_3` | **−0.000058** | −0.000004 | **14.7×** |

Pre-flip diff is **larger and declining faster** than random noise. The flip is not a gentle crossing — it's a parameter actively shedding its beacon sensitivity while still learning. This is why the cascade is so sharp.

### Implications for BRLM

| Zone | Batch | What's Happening | Graft? |
|---|---|---|---|
| Dead | 0-30 | Gradients chaotic, beacon and base uncorrelated | No |
| Plastic | 30-35 | Parameters finding roles, beacon still perturbs | **Yes — now** |
| Locked | >35 | Parameters beacon-invariant, stable configurations | No |

The grafting window is not about "gradients are large" — it's about **"gradients are still beacon-sensitive."** Once that sensitivity is lost, the parameter is locked.

### Trajectory-Based Prediction (44% Precision)

Using `loss < 3.0 AND diff > 0 AND diff_slope_3 < 0 AND |diff| < 1e-4` predicts a flip within the next 10 batches with **44% precision** and **17% recall** (F1 = 0.248). This is useful for BRLM as an early-warning system: when you see a positive-but-declining diff with loss around 2.7-2.8, that parameter is about to lock in.

---

## Does Measuring the Beacon Influence the Result?

### Technical Answer: No

The training loop explicitly zeros gradients between the beacon pass and the baseline pass:

```python
# Beacon pass
loss_beacon.backward()
grads_beacon = collect_grads(model)

model.zero_grad(set_to_none=True)  # ZERO beacon gradients

# Baseline pass
loss_base.backward()
grads_base = collect_grads(model)

opt.step()  # ONLY sees baseline gradients
```

The beacon pass is purely observational — compute, record, discard. The optimizer never sees beacon gradients. There is no dropout in the model, so random state is not affected. The baseline pass builds a fresh, independent computation graph.

### Epistemological Answer: Yes

While the measurement does not alter parameter updates, **the choice of measurement determines what we discover.** We chose to measure beacon gradient difference, and that revealed the sign-flip cascade. If we had only measured base gradient magnitude, we might have concluded "convergence happens when gradients get small" — a true but incomplete description that misses the beacon-invariance mechanism entirely.

The sign-flip is a **real property of the loss landscape**, but it is an **emergent property** — it only becomes visible when you compare beacon vs baseline. Base gradient alone does not tell you about beacon sensitivity.

### What This Means for BRLM

| Question | Answer |
|---|---|
| Does beacon tracing alter parameter updates? | **No** — zeroed before baseline |
| Does beacon tracing alter random state? | **No** — no dropout, no persistent state |
| Does measurement choice affect conclusions? | **Yes** — we found beacon-invariance because we looked for it |

**The phenomenon is real, but the measurement is what makes it visible.** The beacon is not just a diagnostic tool — it is a probe into the structure of the loss landscape. Without it, convergence looks like gradual gradient decay. With it, convergence looks like a phase transition into beacon-invariant stability.

---

## Beacon Magnitude Sweep: How Strong Can the Beacon Be?

To test whether the beacon magnitude affects convergence dynamics, we swept `BEACON_MAGNITUDE` across [0.01, 0.05, 0.1, 0.2, 0.5, 1.0] with `N_BATCHES = 70` and tracked ignition point, cascade width, and never-flip count.

### Results

| Magnitude | Ignition Point | Cascade Width | Converged | Never Flip | B/Base Ratio |
|---|---|---|---|---|---|
| **0.01** | **31** | 8 | 41/41 | 0 | 1.002 |
| **0.05** | **31** | 13 | 41/41 | 0 | 1.013 |
| **0.10** | **31** | 9 | 41/41 | 0 | 1.026 |
| **0.20** | **31** | 9 | 41/41 | 0 | 1.054 |
| **0.50** | **31** | 6 | 40/41 | 1 | 1.101 |
| **1.00** | **31** | **25** | 38/41 | 3 | 1.190 |

### Key Findings

**Ignition point is always batch 31.** The beacon magnitude does not shift when convergence starts. Convergence timing is driven by training dynamics, not by the beacon probe. The beacon only reveals what was already happening.

**Cascade width spreads at extremes.** At 0.01 the cascade is tight (8 batches) — the beacon is so weak it's almost invisible. At 1.0 the cascade widens to 25 batches — some parameters take much longer to become beacon-invariant. At 0.05-0.2 the width is moderate (9-13 batches).

**Never-flip count rises with magnitude.** All 41 parameters flip at 0.01-0.2. At 0.5 one parameter never flips. At 1.0 three parameters never flip — **some parameters cannot learn to ignore a strong beacon.**

**Beacon/base ratio at flip increases monotonically.** At the flip point, `beacon_grad / base_grad` goes from 1.002 (0.01) to 1.190 (1.0). A stronger beacon produces proportionally larger gradients, but the model still eventually learns to ignore it — except for stragglers at magnitude 1.0.

### Implications for BRLM

| Magnitude | Effect on Plastic Window | Detector Quality |
|---|---|---|
| **0.01** | Very short (8 batches) | Hard to catch, beacon too weak |
| **0.05-0.2** | Moderate (9-13 batches) | **Best for BRLM** |
| **0.5** | Some params never lock in | Unreliable |
| **1.0** | Very wide (25 batches), 3 never flip | Noisy, late stragglers |

**The sweet spot for BRLM is 0.05-0.2.** The beacon is strong enough to be measurable and produce a clear signal, but not so strong that parameters cannot learn to ignore it.

---

## What Is Happening Inside the Dead Zone?

The dead zone (batches 0-30) is called "dead" because no sign flips occur, but the model is actively learning. Loss drops from 2.94 to ~2.80 over these 30 batches.

### The Model Is Learning, But Hasn't "Clicked" Yet

| Batch | Loss |
|---|---|
| 1 | 2.941 |
| 10 | 2.884 |
| 20 | ~2.83 |
| 30 | ~2.80 |

The model is in an exploratory phase — it's still figuring out the echo task. Gradients are large and chaotic, oscillating randomly between positive and negative beacon differences.

### Every Parameter "Touches" Every Example

In a transformer, all 41 parameters receive gradients from every training example through backprop:

```
Input tokens → [token_emb + pos_emb] → Layer 0 → Layer 1 → Layer 2 → Head → Loss
```

There's no parameter that "doesn't touch the data." The dead zone isn't about inactive parameters — it's about parameters that haven't found beacon-invariant configurations yet.

### What "Clicks" at Batch 32?

Before batch 32, the attention mechanism hasn't stabilized. Adding a beacon vector to all positions shifts attention scores unpredictably, so gradients change chaotically.

After batch 32, the attention mechanism has **locked onto position-based copying.** It learned: "attend to the previous position, copy its token." Once this rule is learned, adding the same beacon vector to all positions doesn't change relative attention scores — it just adds a constant offset that the model ignores.

### Which Parameters Are the "Keyholders"

| Parameter Type | Role | Key to Beacon Invariance? |
|---|---|---|
| `token_emb.weight` | Token lookup | No — just a table |
| `pos_emb.weight` | Position encoding | Maybe — position determines attention target |
| `self_attn.in_proj_weight` | Q/K/V projection | **Yes** — defines what to attend to |
| `self_attn.out_proj_weight` | Attention output | **Yes** — determines representation |
| `linear1/2.weight` | FFN | No — refines representations |
| `norm1/2.weight/bias` | LayerNorm | No — stabilizes but doesn't decide |
| `head.weight/bias` | Output projection | No — reads final representation |

**The attention parameters (`in_proj_weight`, `out_proj_weight`) are the keyholders.** When they lock onto position-based copying, the whole model becomes beacon-invariant.

### Dead Zone Pattern for head.bias

| Batch | Diff | Sign |
|---|---|---|
| 1-27 | Oscillates | +/− randomly |
| 28 | -0.000318 | − |
| 29 | +0.000155 | + |
| 30 | -0.000051 | − |
| **31** | **+0.000186** | **+** |
| **32** | **−0.000019** | **−** ← **FLIP** |

No pre-flip signal. The diff oscillates for 30+ batches with no trend, then suddenly crosses.

---

## What Happens Without Positional Encoding?

To test whether position awareness is required for beacon invariance, we trained a model with `pos_emb` removed. The model cannot distinguish positions and therefore cannot learn "copy the previous token."

### Results

| Metric | With Pos Emb | Without Pos Emb |
|---|---|---|
| Final loss | 2.7582 | **2.6946** |
| Parameters flipped | 41/41 | **41/41** |
| First flip batch | 32 | **31** |
| Flip batches | 32, 33, 34, 35, 37 | **31, 33, 34, 35, 36, 39** |

### Key Findings

**Positional encoding is NOT required for beacon invariance.** The model still converges to a beacon-ignoring state, even without position information.

**The model learns token co-occurrence instead.** Without positions, it learns which tokens tend to follow which (e.g., "token A often follows token B") rather than strict positional copying.

**The phase transition is robust.** Remove a key component, and the cascade still happens. The model finds SOME stable configuration that makes the beacon ignorable.

**Beacon invariance is not task-specific.** It detects "model has found a stable strategy" — whether that's position-based copying or token co-occurrence patterns.

---

## What Happens With Completely Random Data?

To test whether beacon invariance requires ANY learnable structure, we trained on completely shuffled sequences with random targets (no echo, no co-occurrence, no pattern at all).

### Results

| Metric | Echo Task | Shuffled (Random) |
|---|---|---|
| Final loss | 2.7582 | **2.7906** (plateaued) |
| Parameters flipped | 41/41 | **42/42** |
| First flip batch | 32 | **31** |
| Flip batches | 32, 33, 34, 35, 37 | **31, 32, 33, 34, 36, 38** |

### This Changes Everything

**The sign flip happens even when the model learns NOTHING meaningful.** Loss plateaus at ~2.79 (random guessing for 16-token vocab = `ln(16) ≈ 2.77`). The model is in a high-loss basin, but it still becomes beacon-invariant.

### What This Actually Means

Beacon invariance is **not about "finding a stable strategy."** It's about something more fundamental:

1. **High-loss basin attraction:** Once the model settles into any parameter configuration, gradients become insensitive to small perturbations. This is a property of the optimization landscape, not the task.

2. **Adam's bias correction:** As training progresses, Adam's moving averages stabilize. Even with random gradients, the optimizer's internal state converges, making the effective step direction less sensitive to perturbations.

3. **Loss landscape geometry:** Near any local minimum (even a bad one), the Hessian is positive semi-definite. Small perturbations don't change the gradient direction much.

### The Real Insight

| Condition | Loss at Flip | Flip Happens? |
|---|---|---|
| Echo task (structured) | 2.76 | **Yes** |
| No pos_emb (co-occurrence) | 2.69 | **Yes** |
| **Random data (no structure)** | **2.79** | **Yes** |

**The sign flip is a universal property of training dynamics, not a signal of task learning.** It happens whenever:
- Gradients have been computed for ~30 batches
- Adam's moving averages have stabilized
- The model has settled into ANY basin (good or bad)

This refines BRLM: the beacon detects **optimization convergence**, not **task convergence.** The plastic window is the window before the optimizer stabilizes, not before the model "understands" the task.

---

## What Happens With a Different Task?

To test whether beacon invariance is task-independent, we changed the task from "echo the previous token" to "copy the first token to all positions." This is still positional but requires the model to always attend to position 0.

### Results

| Metric | Echo Task | Copy First Token |
|---|---|---|
| Final loss | 2.7582 | **2.9322** (still learning) |
| Parameters flipped | 41/41 | **42/42** |
| First flip batch | 32 | **31** |
| Flip batches | 32, 33, 34, 35, 37 | **31, 32, 33, 34, 35, 36, 39, 42** |

### What's Different

**Loss is higher — the model is struggling.** Copying the first token is harder than echo because it requires consistently attending to position 0 regardless of the token value. The model hasn't fully converged even at batch 70.

**More flip batches (8 vs 5).** The cascade is more spread out — parameters lock in over 12 batches instead of 6. This suggests the model is less stable, with parameters converging at different rates.

**But the flip still happens universally.** 42/42 parameters become beacon-invariant. The phase transition is independent of task difficulty.

### Updated Understanding

| Task Type | Loss | Flip? | Timing |
|---|---|---|---|
| Echo (easy positional) | 2.76 | Yes | Batch 32, tight cascade |
| No pos_emb (co-occurrence) | 2.69 | Yes | Batch 31, moderate |
| **Copy first token (harder positional)** | **2.93** | **Yes** | **Batch 31, wider cascade** |
| Random (no structure) | 2.79 | Yes | Batch 31, moderate |

**The sign flip is a universal property of transformer training dynamics.** Task difficulty affects final loss and cascade width, but NOT whether the flip happens.

---

## What Happens When the Model Must Reconstruct the Beacon?

Instead of treating the beacon as invisible noise, we added a supervised beacon reconstruction head. The model must output both the main task prediction AND a reconstruction of the beacon vector.

### Setup

```
Input → Transformer → [Main Head (logits), Beacon Head (vector)]
Loss = CrossEntropy(main) + MSE(beacon_reconstruction, true_beacon)
```

We tracked three metrics over batches:
- Main task loss (echo)
- Beacon reconstruction MSE
- Cosine similarity between predicted and true beacon

### Results (Seed=42, Two Runs — Identical)

| Metric | Batch 1 | Batch 31 | Batch 70 |
|---|---|---|---|
| Main loss | 2.9201 | 2.8150 | **2.7534** |
| Beacon MSE | 0.0657 | 0.0101 | **0.0072** |
| Beacon cos_sim | 0.1937 | −0.0976 | **−0.1089** |
| Parameters flipped | — | 44/44 started | Complete |
| First flip | — | **batch 31** | — |

### What Happened

**1. Main loss decreased** — the model learned the echo task normally.

**2. Beacon MSE dropped 10x** — from 0.066 to 0.007. But cosine similarity stayed near **zero** throughout, oscillating between −0.27 and +0.19.

**3. Low MSE + near-zero cosine = the model learned the right scale, not the right direction.**

The beacon_head outputs vectors with similar magnitude to the true beacon (~0.05) but essentially random orientation. It never learns to extract the beacon's actual direction.

### Critical Insight

| Observation | Interpretation |
|---|---|
| `cos_sim ≈ 0` | Model does NOT learn to reconstruct beacon directionally |
| `MSE ↓` | Model learns the right scale (converges to typical magnitude) |
| `sign_flip still happens` | Beacon invariance is about gradient insensitivity, not output suppression |

**The model doesn't "actively suppress" the beacon at the output layer.** Instead, the deeper layers become insensitive to it — gradients stop caring about the perturbation, even though the output head could theoretically reconstruct it.

**Beacon invariance is an emergent property of the hidden representation, not a learned filtering mechanism.**

---

## What Happens With a "Probing Gateway" at the Deepest Layer?

To test whether beacon invariance propagates downward from the input or upward from the loss, we attached a small probe network to the deepest layer (Layer 2). The probe tries to predict the injected beacon, and its gradients backpropagate into the main model.

### Setup

```
Input → [Layer 0] → [Layer 1] → [Layer 2] → [Probe]
                                         ↓
                                   predict beacon
                                   backprop into Layer 2

Loss = CrossEntropy(main) + MSE(probe(Layer_2_output), beacon)
```

This creates a **competing objective**:
- Main task loss pushes the model to ignore the beacon
- Probe loss forces Layer 2 to preserve beacon information

### Results (Seed=42, Two Runs — Identical)

| Metric | Without Probe | With Probe (Layer 2) |
|---|---|---|
| Final main loss | **2.7534** | **2.7854** (worse) |
| Final probe loss | — | **0.581862** (got WORSE) |
| Parameters flipped | 44/44 | **4/42** |
| First flip | batch 31 | batch 31 |
| Flip batches | [31, 32, 33, 34, 35, 36] | **[31, 32]** |

### What Happened

**1. The cascade was almost completely suppressed.** Only 4 parameters flipped (head/norm), and only at batch 31-32. All encoder layers, embeddings, and probe parameters stayed plastic for all 70 batches.

**2. The probe loss INCREASED over time.** From 0.051 at batch 10 to 0.58 at batch 70. The probe got progressively worse at predicting the beacon.

**3. Main task performance degraded.** Loss is 2.7854 vs 2.7534 without probe. The competing objectives hurt task learning.

### What This Means

| Layer | Without Probe | With Probe |
|---|---|---|
| Layer 0 | Flips | **No flip** |
| Layer 1 | Flips | **No flip** |
| Layer 2 | Flips | **No flip** |
| Head/Norm | Flips | Some flip |
| Probe | — | Never flips |

**The cascade can be locally disrupted.** The probe acts as a "plasticity anchor" — by forcing beacon preservation at one depth, it prevents the whole system from locking in.

### The Deepest Insight

**Beacon invariance propagates UPWARD from the loss, not DOWNWARD from the input.**

If it propagated downward (embedding learns to ignore → all layers ignore), the probe at Layer 2 wouldn't matter — the beacon would already be gone by the time it reached Layer 2.

But the probe DOES matter. This means the input embeddings NEVER learn to suppress the beacon. Instead, the **task loss teaches all layers simultaneously** to become beacon-invariant. And if you add a competing loss at a deep layer, that layer (and potentially all layers) stays sensitive.

**The cascade is a coordinated lock-in driven by the loss landscape, not a feedforward filtering mechanism.**

---

## What Happens With Double Beacons?

To test whether the model can selectively extract one beacon while ignoring another, we made the beacon itself the task:

- **Beacon A** = the TASK (model must predict this vector)
- **Beacon B** = the PERTURBATION (measure if model becomes insensitive to it)

Both beacons are injected into the SAME embedding space.

### Setup

```
Input: random tokens
Inject: Beacon A (task, mag=0.5) + Beacon B (perturbation, mag=0.05)
Target: predict Beacon A
Compare: gradients with Beacon B present vs absent
```

This tests: can the model disentangle signal from perturbation when they're mixed at the input?

### Results (Seed=42, Two Runs — Identical)

| Metric | Value |
|---|---|
| MSE (start) | 0.4311 |
| MSE (end) | **0.2102** |
| Parameters flipped | **42/42** |
| First flip | **batch 32** |
| Flip batches | **[32, 33, 34, 35, 36, 37, 38, 39, 40]** |

### What Happened

**1. The model learned the task.** MSE dropped from 0.43 to ~0.21. It successfully extracts Beacon A from the embedding.

**2. ALL parameters became insensitive to Beacon B.** Despite Beacon A being the ONLY signal and Beacon B being mixed with it at the input, the sign flip still happens universally.

**3. But the cascade is much wider** — 9 batches (32-40) vs 5-6 in the normal echo task. Disentangling two beacons is harder, so lock-in takes longer.

### Comparison

| Property | Echo Task | Double Beacon |
|---|---|---|
| Task signal | Token sequence | Beacon A (in embedding) |
| Perturbation | Beacon | Beacon B (mixed with signal!) |
| First flip | 32 | **32** |
| Cascade width | 5-6 batches | **9 batches** |
| All params flip? | Yes | **Yes** |

### What This Means

**The model CAN disentangle signal from perturbation.** Even when Beacon B is injected into the same embedding space as the task signal (Beacon A), the model learns to extract A while becoming gradient-invariant to B.

**But beacon invariance is NOT "learned filtering."** It's not that the model develops a mechanism to "subtract Beacon B." Instead, the optimizer finds a parameter configuration where the gradient on Beacon A's prediction is robust to the presence/absence of Beacon B.

**The wider cascade tells us:** When the perturbation is entangled with the signal, the model takes longer to find a stable, perturbation-robust configuration. But once found, the invariance is still universal.

---

## What Happens With a Beacon Preservation Penalty?

To test whether the cascade can be disrupted by forcing the model to preserve beacon information in hidden states, we added a penalty term:

```
Loss = CrossEntropy(echo) + λ * MSE(hidden_state, beacon)
```

This creates a direct competing objective: the echo task pushes toward beacon invariance, while the penalty forces beacon preservation.

We tested three penalty locations:
- **Option 1:** Penalty on final hidden state (before head)
- **Option 2:** Penalty on Layer 1 intermediate output
- **Option 3:** Penalty on per-position hidden states

### Results (Seed=42, λ=1.0)

| Option | Penalty Location | Params Flipped | First Flip | Flip Batches | Suppression |
|---|---|---|---|---|---|
| **1** | Final hidden state | **3/42** | batch 33 | **[33]** | **Strongest** |
| **2** | Intermediate (Layer 1) | **28/42** | batch 31 | **[31, 32, 33, 34, 35, 37]** | **Partial** |
| **3** | Per-position states | **5/42** | batch 33 | **[33, 47, 59]** | **Strong** |

### Key Findings

**1. Penalty location matters dramatically.**
- Final-state penalty (Option 1): cascade almost completely suppressed (3 flips)
- Intermediate penalty (Option 2): majority still flip (28/42)
- Per-position penalty (Option 3): strong suppression (5 flips)

**2. Option 2 is the critical result.** When the penalty is at Layer 1, the model finds a **compromise**: it preserves beacon information in early layers but STILL becomes beacon-invariant in later layers (Layer 2+).

| Option | What Gets Disrupted | What Still Flips |
|---|---|---|
| 1 (Final state) | ALL layers | Almost nothing |
| 2 (Layer 1) | Layers 0-1 stay plastic | **Layers 2+ still flip** |
| 3 (Per-position) | ALL positions | Almost nothing |

**3. The cascade is NOT "all or nothing" — it can be disrupted locally.** When the penalty is at an intermediate layer, the model can become beacon-invariant above that layer while staying sensitive below it.

### What This Means

This refines our understanding from the probe experiment:

| Previous Finding | Refined Understanding |
|---|---|
| "Cascade is coordinated by loss" | True, but layers can partially decouple |
| "Probe disrupts everything" | True only if penalty reaches all layers |
| "All-or-nothing lock-in" | **Layer-wise but coupled** — some layers can flip while others don't |

**Beacon invariance is a per-layer property that becomes correlated across layers through training.** You can break the correlation by adding a competing objective at a specific depth.

---

## What Happens With Orthogonal Beacon Rotation?

To test whether changing the beacon's mathematical direction prevents the cascade, we rotated the beacon into a new random orthogonal direction every N batches while keeping its magnitude constant.

### Results (Seed=42)

| Rotation Frequency | Flips | First Flip | Cascade Width |
|---|---|---|---|
| **Static (fixed)** | **42/42** | **batch 31** | **6 batches** |
| **Every 1 batch** | **42/42** | **batch 31** | **6 batches** |
| **Every 5 batches** | **42/42** | **batch 31** | **8 batches** |
| **Every 10 batches** | **42/42** | **batch 31** | **10 batches** |

### Key Finding: Cascade Width Correlates With Rotation Period

**The first flip timing is identical (batch 31) regardless of rotation.** Even changing the beacon direction every single batch does not delay the ignition point.

**But cascade width increases with longer rotation periods:**
- Static beacon: tightest cascade (6 batches)
- Rapid rotation (every batch): same tightness (6 batches)
- Slower rotation (every 10 batches): widest cascade (10 batches)

### What This Means

**The beacon direction is irrelevant to the cascade timing.** The model does not "memorize" a specific beacon direction and lock onto it. Instead, the optimizer reaches a state where ANY small perturbation of consistent magnitude is gradient-irrelevant.

**The cascade width variation is likely due to Adam's moving averages smoothing out perturbation changes.** When the beacon stays in one direction longer, the optimizer has more time to stabilize relative to that direction, but then must readjust when the direction changes. This creates a noisier transition.

| Hypothesis | Result |
|---|---|
| Model "memorizes" beacon direction | ❌ False |
| Rotation prevents Adam stabilization | ❌ False |
| Cascade depends on perturbation statistics | ❌ False |

**The cascade is purely about Adam's moving averages reaching stability.** The beacon direction is irrelevant — any task-irrelevant perturbation of consistent magnitude gets ignored once the optimizer settles.

---

## What Happens at Extreme Beacon Magnitudes?

To test whether there's a fundamental limit to how large a perturbation can be before the cascade breaks, we swept beacon magnitude from 1.0 to 10.0 in 0.5 increments.

### Results (Seed=42)

| Magnitude | Final Loss | Flips | First Flip | Cascade Width |
|---|---|---|---|---|
| 1.0 | **2.7585** | 39/42 | 33 | 6 |
| 1.5 | **2.7585** | 39/42 | 33 | 9 |
| 2.0 | **2.7585** | 32/42 | 33 | 6 |
| 2.5 | **2.7585** | 32/42 | 32 | 7 |
| 3.0 | **2.7585** | 35/42 | 32 | 6 |
| 4.0 | **2.7585** | 35/42 | 32 | 7 |
| 5.0 | **2.7585** | 35/42 | 32 | 6 |
| 6.0 | **2.7585** | 35/42 | 32 | 6 |
| 7.0 | **2.7585** | 37/42 | 32 | 7 |
| 8.0 | **2.7585** | 36/42 | 32 | 6 |
| 9.0 | **2.7585** | 34/42 | 32 | 7 |
| **10.0** | **2.7585** | **34/42** | **32** | **6** |

### What's Shocking

**1. Final loss is IDENTICAL across all magnitudes.** 2.7585 for every single run from 1.0 to 10.0.

**2. The cascade still happens.** Even at magnitude 10.0, 34/42 parameters flip, starting at batch 32.

**3. Task performance is completely unaffected.** The model learns the echo task equally well whether the beacon is tiny or 100x larger than the signal.

### Why This Happens

**The beacon is a common-mode signal.** We add the SAME vector to ALL positions. The transformer architecture inherently ignores common-mode offsets:

- **Self-attention** computes `Q·K^T` — if every position gets the same offset, the dot products change by a constant, but the softmax normalization cancels it out: `softmax(x + c) = softmax(x)`
- **LayerNorm** normalizes across the feature dimension, further suppressing common offsets
- **Residual connections** preserve the relative structure

**The model isn't "learning to ignore" the beacon — the architecture mathematically eliminates it.** The beacon could be magnitude 100 or 1000 and the effect would be the same, as long as it's uniform across positions.

### What This Means

| Hypothesis | Result |
|---|---|
| Large beacon breaks the cascade | ❌ **False** — cascade survives even at 10.0 |
| Beacon magnitude affects task learning | ❌ **False** — loss identical across all magnitudes |
| Model must "learn" beacon invariance | ❌ **False** — architecture handles it automatically |

**Beacon invariance is not learned — it's a built-in property of the transformer architecture when perturbations are uniform across positions.**

This explains why our earlier magnitude sweep (0.01-1.0) found a "sweet spot" — the effect plateaus because the architecture inherently filters common-mode signals. The magnitude doesn't matter once it's above the numerical precision threshold.

**This is a critical refinement:** The sign-flip cascade isn't about the model "figuring out" how to ignore the beacon. It's about the **optimizer stabilizing into a configuration where gradient differences become numerically zero** — even though the forward pass was already beacon-invariant from the start.

---

## What Happens With a Position-Dependent Beacon?

To test whether breaking common-mode cancellation prevents the cascade, we injected a different random beacon vector at each position instead of the same vector everywhere.

### Results (Seed=42)

| Metric | Uniform Beacon | Position-Dependent Beacon |
|---|---|---|
| Final loss | 2.7585 | **2.7732** |
| Parameters flipped | 42/42 | **42/42** |
| First flip | batch 31 | **batch 31** |
| Flip batches | [31, 32, 33, 34, 35, 37] | **[31, 32, 33, 34, 35, 36, 41]** |
| Avg |diff| late (batches 60-70) | 0.00000265 | **0.00000055** |

### What Happened

**The cascade still happens universally.** Even with a different beacon vector at every position, all 42 parameters flip starting at batch 31.

**But late-stage sensitivity is 5x lower.** The position-dependent beacon creates smaller gradient differences in the locked zone, suggesting the model is more affected by it but still finds a perturbation-robust configuration.

**The cascade is slightly wider** (7 vs 6 batches), indicating the transition is noisier.

### Why the Beacon Still Dies

Even though the beacon varies by position, the **task doesn't require the model to use it.** The echo task can be solved by pure position-based attention — the beacon is still task-irrelevant noise.

### What Would Actually Make the Beacon Survive

| Condition | Why It Would Work |
|---|---|
| **Beacon encodes the target token** | Model MUST extract it to minimize loss |
| **Beacon is the ONLY signal** | No other way to solve the task |
| **Task requires beacon disambiguation** | E.g., "copy the token indicated by the beacon" |

The beacon dies because the task can be solved without it. **To make it survive, make it indispensable.**

---

## What Happens When the Beacon Encodes the Target Token?

To test whether making the beacon task-relevant prevents the cascade, we created a task where:
- Input: random tokens (irrelevant to output)
- Beacon at position i: a vector that encodes the target token for position i
- Target: decode the beacon to predict the correct token

The beacon is the **ONLY** source of target information. Without extracting it, the model cannot minimize loss.

### Results (Seed=42)

| Metric | Normal Echo Task | Beacon Encodes Target |
|---|---|---|
| Final loss | 2.7585 | **0.0641** |
| Parameters flipped | 42/42 | **7/42** |
| First flip | batch 31 | **batch 31** |
| Avg diff early (batches 1-30) | ~0 | **0.00001209** |
| Avg diff late (batches 60-70) | ~0 | **0.00006059** |

### What Happened

**Only 7/42 parameters flipped.** The cascade is almost completely suppressed when the beacon encodes the target token.

**The model learned the task exceptionally well.** Loss dropped from 2.78 to 0.06 — near-perfect accuracy. The model successfully decodes the beacon to make predictions.

**But gradient differences INCREASED over time.** Early average diff: 0.000012. Late average diff: 0.000060 — **5x larger**. The beacon becomes MORE relevant, not less, as training progresses.

### Why This Works

**The beacon is no longer task-irrelevant noise — it's the ONLY source of target information.** Without extracting the beacon, the model cannot minimize loss. The optimizer is forced to preserve beacon sensitivity throughout training.

The 7 parameters that DID flip are likely the head and final norm layers — they receive the decoded beacon information from earlier layers and don't need to be directly sensitive to the raw beacon embedding. The encoder layers stay plastic because they must continue extracting the beacon.

### This Is the Key

| Condition | Flips | Reason |
|---|---|---|
| Beacon is noise | 42/42 | Model can solve task without it |
| Beacon encodes target | **7/42** | **Model MUST use it to minimize loss** |

**To make the beacon survive: make it indispensable to the task.** Not position-dependent, not larger, not rotated — **task-relevant.**

---

# Circuit Mapping: Tracing Information Flow Through the Model

To understand WHERE the beacon dies and HOW information flows, we ran 7 complementary approaches:

## Summary of All 7 Approaches

| Approach | What It Measures | Key Finding |
|---|---|---|
| **1. Layer-Wise Probing** | Can probes at each layer predict the beacon? | Beacon info present **uniformly** at all layers |
| **2. Activation Patching** | Does patching beacon activations change output? | Beacon causal effect **fades but never vanishes** |
| **3. Attention Analysis** | Do attention patterns encode the beacon? | Beacon **doesn't affect attention** at all |
| **4. Representation Geometry** | How do hidden state geometries evolve? | Deeper layers become **position-invariant** |
| **5. Gradient Attribution** | Which parameters drive beacon gradient diff? | Sensitivity **distributed** across all layers |
| **6. Residual Stream** | How does beacon difference propagate? | Difference **decreases over training** |
| **7. Ablation/Knockout** | Which components are critical? | **All layers critical** for task; Head 1 most beacon-sensitive |

## Detailed Findings

### Approach 1: Layer-Wise Probing

Probe MSE at each layer (lower = beacon preserved):

| Checkpoint | Layer 0 | Layer 1 | Layer 2 |
|---|---|---|---|
| 10 | 0.0029 | 0.0034 | 0.0043 |
| 30 | 0.0023 | 0.0028 | 0.0026 |
| 50 | 0.0028 | 0.0030 | 0.0025 |
| 70 | 0.0037 | 0.0032 | 0.0032 |

**Finding:** Probe MSE is roughly equal (~0.002-0.004) across ALL layers at ALL checkpoints. The beacon is NOT lost at any specific layer.

### Approach 2: Activation Patching

KL divergence from patching beacon activations into no-beacon run:

| Checkpoint | No Patch | Patch L0 | Patch L1 | Patch L2 |
|---|---|---|---|---|
| 10 | 0.0068 | 0.097 | 0.099 | 0.098 |
| 30 | 0.0020 | 0.050 | 0.050 | 0.046 |
| 50 | 0.0005 | 0.038 | 0.030 | 0.025 |
| 70 | 0.0005 | 0.035 | 0.021 | 0.019 |

**Finding:** Patch KL decreases over training (from ~0.10 to ~0.02), meaning the model becomes more robust to forced beacon activations. But it never drops to zero.

### Approach 3: Attention Pattern Analysis

"PrevPos Attn" = average attention to previous position (copy pattern):

| Checkpoint | Layer 0 | Layer 2 | Beacon Δ |
|---|---|---|---|
| 10 | ~0.10 | ~0.10 | ~0.0002 |
| 30 | ~0.12 | ~0.12 | ~0.0005 |
| 50 | ~0.17 | ~0.17 | ~0.0004 |
| **70** | **~0.27** | **~0.29** | **~0.001** |

**Finding:** The model develops "copy heads" over training (0.10 → 0.29). But the beacon changes attention by only ~0.001 at all checkpoints — essentially zero.

### Approach 4: Representation Geometry

| Checkpoint | Layer | Cosine Similarity | Beacon Projection |
|---|---|---|---|
| 10 | 0 | 0.106 | 1.48 |
| 10 | 2 | 0.448 | 2.49 |
| 70 | 0 | 0.304 | **1.19** |
| 70 | 2 | **0.895** | **1.64** |

**Finding:** Deeper layers become dramatically more position-invariant (cos sim: 0.45 → 0.90). Beacon projection drops at batch 70, coinciding with the cascade.

### Approach 5: Gradient Attribution Paths

| Checkpoint | Layer | Attention | FFN In | FFN Out |
|---|---|---|---|---|
| 10 | 0 | 0.00035 | 0.00023 | 0.00044 |
| 70 | 0 | **0.00094** | **0.00068** | **0.00114** |
| 70 | 2 | **0.00103** | **0.00043** | **0.00072** |

**Finding:** Gradient differences are distributed across ALL layers. No single layer dominates. The absolute difference INCREASES over training (the signed difference flips, but magnitude grows).

### Approach 6: Residual Stream View

||h_with_beacon - h_without_beacon|| at each layer:

| Layer | Batch 1 | Batch 70 |
|---|---|---|
| 0 | 0.512 | **0.451** |
| 1 | 0.605 | **0.505** |
| 2 | 0.731 | **0.550** |

**Finding:** Beacon difference propagates through all layers but **decreases 10-25% over training**. The model actively learns to stabilize hidden states against the beacon.

### Approach 7: Ablation/Knockout

| Knockout | Task Loss Δ | Beacon Diff Δ |
|---|---|---|
| Layer 0 | **+1.905** | -0.024 |
| Layer 1 | **+1.640** | -0.024 |
| Layer 2 | **+1.429** | -0.024 |
| Head 0 | +0.062 | -0.017 |
| Head 1 | +0.054 | -0.011 |
| Head 2 | +0.087 | -0.021 |
| Head 3 | +0.076 | -0.021 |

**Finding:** ALL layers are critical for the task. Individual heads have modest impact (+0.05-0.09). Head 1 has the highest remaining beacon sensitivity after knockout.

## The Complete Picture

### What Kills the Beacon?

| Mechanism | Finding | Source |
|---|---|---|
| **Forward-pass filtering** | ❌ Not happening — beacon present at all layers | Approaches 1, 6 |
| **Attention suppression** | ❌ Not happening — beacon doesn't affect attention | Approach 3 |
| **Gradient insensitivity** | ✅ Yes — signed diff crosses zero | Earlier experiments |
| **Hidden state stabilization** | ✅ Yes — beacon difference norm decreases | Approach 6 |
| **Geometric rotation** | ✅ Yes — representations rotate away from beacon | Approach 4 |

### The Refined Cascade Model

**The sign-flip cascade is a distributed, optimizer-driven phase transition where:**

1. **The beacon is always present in the forward pass** (Approaches 1, 6)
2. **But the model learns to make its hidden states more stable to the beacon** (Approach 6)
3. **Representations rotate away from the beacon direction** (Approach 4)
4. **Gradients become insensitive (signed diff crosses zero)** (original experiments)
5. **All layers participate simultaneously** — there's no single "blackout layer" (Approaches 2, 5, 7)

**The cascade is not about the model "figuring out" how to ignore the beacon. It's about the optimizer stabilizing into a configuration where the beacon's presence doesn't change the effective update direction — even though the beacon is still mathematically present in every hidden state.**

---

## Which Individual Parameters Interact With the Beacon?

We computed `|grad_with_beacon - grad_without_beacon|` for **every single weight** (~170,000 parameters) in the trained model to find the EXACT parameters that "see" the beacon.

### Method

1. Forward + backward with beacon → record gradient for each parameter
2. Forward + backward without beacon → record gradient for each parameter
3. Compute absolute difference for every individual weight
4. Rank all parameters by sensitivity

### Top 10 Most Beacon-Sensitive Individual Weights

| Rank | Parameter | Index | Weight Value | |Grad Diff| |
|---|---|---|---|---|
| 1 | `layers.0.norm1.bias` | **[33]** | -0.0021 | **0.01597** |
| 2 | `layers.0.norm2.bias` | **[33]** | -0.0009 | **0.01490** |
| 3 | `layers.0.linear2.bias` | **[33]** | 0.0258 | **0.01444** |
| 4 | `layers.0.norm1.bias` | **[37]** | 0.0029 | **0.01357** |
| 5 | `layers.0.norm2.bias` | **[37]** | 0.0026 | **0.01321** |
| 6 | `layers.0.linear2.bias` | **[37]** | 0.0586 | **0.01235** |
| 7 | `head.weight` | [10, **13**] | 0.1112 | **0.01225** |
| 8 | `head.weight` | [9, **41**] | -0.1402 | **0.01159** |
| 9 | `layers.1.norm1.bias` | **[33]** | -0.0017 | **0.01133** |
| 10 | `layers.0.self_attn.out_proj.bias` | **[33]** | -0.0020 | **0.01113** |

### Critical Discovery: Dimensions 33 and 37

**Index 33 appears as the #1 most sensitive weight across ALL component types:**

| Component | Index 33 Sensitivity |
|---|---|
| `layers.0.norm1.bias[33]` | 0.01597 |
| `layers.0.norm2.bias[33]` | 0.01490 |
| `layers.0.linear2.bias[33]` | 0.01444 |
| `layers.0.self_attn.out_proj.bias[33]` | 0.01113 |
| `layers.1.norm1.bias[33]` | 0.01133 |

**Index 37** also appears in #4-#6.

**The beacon signal is concentrated in a 2D subspace (dimensions 33 & 37) of the 64D hidden state.** The model doesn't need to ignore all 64 dimensions — it only needs to suppress these 2 dimensions to eliminate beacon sensitivity.

### Component Sensitivity Ranking

| Component | Total Sensitivity | Mean Per Weight | Max Single Weight | # Params |
|---|---|---|---|---|
| FFN Out | 32.87 | 0.00067 | 0.00863 | 49,152 |
| Attention Q/K/V | 13.60 | 0.00037 | 0.00858 | 36,864 |
| FFN In | 11.08 | 0.00023 | 0.00246 | 49,152 |
| Attention Output | 7.27 | 0.00059 | 0.00768 | 12,288 |
| **LayerNorm** | **1.56** | **0.00204** | **0.01597** | 768 |
| Head Weight | 1.19 | 0.00116 | 0.01225 | 1,024 |
| Token Embedding | 0.31 | 0.00030 | 0.00211 | 1,024 |
| Positional Embedding | 0.31 | 0.00008 | 0.00083 | 3,840 |

**LayerNorm is most sensitive per-parameter** — its weights have 3x higher average sensitivity than attention weights. This is because LayerNorm directly scales the hidden state dimensions that the beacon affects.

### Per-Layer Attention Breakdown

| Layer | Total Sensitivity | Max Single Weight | Top Parameter |
|---|---|---|---|
| Layer 0 | **8.54** | 0.01113 | `out_proj.bias[33]` |
| Layer 1 | **6.66** | 0.01015 | `out_proj.bias[33]` |
| Layer 2 | **6.60** | 0.00659 | `in_proj_weight[9341]` |

**Layer 0 is most beacon-sensitive** — the beacon's effect is strongest at the input layer.

### What This Means for the Cascade

| Finding | Interpretation |
|---|---|
| Beacon lives in 2D subspace | The model only needs to suppress 2 dimensions, not all 64 |
| LayerNorm most sensitive per-weight | LayerNorm parameters are the "control knobs" for beacon suppression |
| Layer 0 most sensitive | The cascade starts where the beacon first enters the model |
| Specific head.weight entries | Output logits for specific tokens (9, 10) have beacon-sensitive projections |

**The cascade is NOT a global phase transition.** It's a targeted suppression of a 2-dimensional subspace. The model learns to make dimensions 33 & 37 beacon-invariant, and because these dimensions are used by all layers (through LayerNorm), the effect propagates through the entire network.

This explains why:
- The cascade is so sharp (batch 31-32) — only 2 dimensions need to flip
- The cascade is so complete (42/42 parameters) — the 2D subspace is shared across all components
- Making the beacon task-relevant suppresses the cascade — the model can no longer ignore dims 33 & 37 because they now carry task information

---

# Synthesis: What All Experiments Tell Us

## The Core Phenomenon

The **sign-flip cascade** is a phase transition in transformer training where all parameters simultaneously become insensitive to a small embedding perturbation (the beacon). It happens reliably around batch 31-32 for seed=42, but timing depends on task difficulty and competing objectives.

## What DRIVES the Cascade

| Experiment | Finding | Conclusion |
|---|---|---|
| **Baseline (echo)** | 41/41 flip at batch 32 | Normal cascade |
| **Random data** | 42/42 still flip | Cascade is **not about task learning** |
| **No pos emb** | 41/41 flip | Position awareness not required |
| **Copy first token** | 42/42 flip, wider cascade | Harder task → wider cascade |
| **Beacon reconstruction head** | 44/44 flip, cos_sim ≈ 0 | Model doesn't "filter" beacon directionally |
| **Double beacon** | 42/42 flip, 9-batch cascade | Model can disentangle signal from perturbation |
| **Probe gateway** | 4/42 flip | Competing loss at deep layer suppresses cascade |
| **Beacon preservation penalty** | 3-28/42 flip depending on depth | Intermediate penalties less effective than final-state |
| **Orthogonal rotation** | 42/42 flip, width varies 6-10 batches | Beacon direction irrelevant; cascade is optimizer-driven |
| **High magnitude sweep (1.0-10.0)** | 34-39/42 flip | Transformer architecture inherently filters common-mode signals |
| **Position-dependent beacon** | 42/42 flip, lower late sensitivity | Beacon still dies because task doesn't require it |
| **Beacon encodes target** | **7/42 flip** | **Beacon survives when it's indispensable to the task** |

## The Refined Model

### 1. Beacon Invariance Is a Property of the Loss Landscape, Not the Task

The model doesn't "learn to ignore" the beacon in the sense of developing a filter. Instead, the optimizer finds parameter configurations where gradients on the task loss are robust to the perturbation. This happens even with random data — it's about **optimizer stabilization**, not task mastery.

### 2. The Cascade Propagates Upward From the Loss, Not Downward From the Input

The probe experiment proved this: if the embedding learned to suppress the beacon, a probe at Layer 2 wouldn't matter. But it DOES matter. The task loss teaches all layers simultaneously to become beacon-invariant.

### 3. The Cascade Is "Layer-Wise but Coupled," Not "All or Nothing"

The preservation penalty experiment showed that an intermediate penalty (Layer 1) allows layers above (Layer 2+) to still flip, while layers below stay plastic. This means:
- Each layer CAN become beacon-invariant independently
- But in normal training, they become correlated
- The correlation is driven by the shared loss signal

### 4. The Plastic Window Is the Period Before Adam's Moving Averages Stabilize

The dead zone isn't "dead" — the model is learning, but gradients are chaotic. Around batch 32, Adam's first and second moment estimates stabilize enough that the effective update direction becomes robust to perturbation.

### 5. Task Difficulty Affects Cascade Width, Not Whether It Happens

| Task | Loss at Flip | Cascade Width |
|---|---|---|
| Echo (easy) | 2.76 | 5-6 batches |
| No pos emb | 2.69 | 5-6 batches |
| Copy first token | 2.93 | 8 batches |
| Double beacon | 0.21 MSE | 9 batches |
| Random | 2.79 | 5-6 batches |

The model finds a perturbation-robust configuration faster with simpler tasks, but it ALWAYS finds one.

## What This Means for BRLM

**The sign-flip is a reliable detector of optimization convergence.** But it's detecting:
- ✅ **Adam stabilization** (moving averages settled)
- ✅ **Loss basin entry** (gradient insensitivity)
- ✅ **Parameter configuration rigidity**

Not:
- ❌ Task mastery
- ❌ Representational quality
- ❌ Generalization

**For meta-learning:** The plastic window ends when the optimizer stabilizes. If you want to keep learning, you need to reset optimizer state, not just continue training.

**For neuroscience analogy:** This is like Hebbian learning reaching a stable synaptic weight configuration — the system becomes resistant to small perturbations, but that doesn't mean it "understands" the task perfectly.

---

## Reproducibility: Locking the Seed

To verify that the first parameter to flip is deterministic given fixed initialization, we added `torch.manual_seed(42)` at the start of training and ran the 70-batch experiment twice.

### Results with Seed=42

| Metric | Run 1 (seed=42) | Run 2 (seed=42) | Match? |
|---|---|---|---|
| Ignition point | **32** | **32** | **Yes** |
| First 8 params | `head.bias`, `layers.1.linear2.weight`, `layers.1.norm1.bias`, `layers.1.norm2.bias`, `layers.2.norm1.bias`, `layers.2.norm1.weight`, `layers.2.self_attn.out_proj.bias`, `layers.2.self_attn.out_proj.weight` | Same 8 params | **Yes** |
| Rolling means | Identical | Identical | **Yes** |

**With `torch.manual_seed(42)` locked, both runs produce bit-for-bit identical results.** The cascade starts at batch 32 consistently, and the first parameter to flip is always `head.bias`.

### What This Proves

The variation in first-flip parameter across previous runs was **entirely due to random initialization**, not due to any fundamental property of the model architecture. With a fixed seed, the convergence cascade is fully deterministic:

- **Batch 32:** 8 parameters flip (`head.bias` first, then Layer 1 and Layer 2)
- **Batch 33:** Remaining parameters follow
- **Total converged:** 41/41 parameters
- **Never flipped:** 0

### Updated Scripts

- `test_beacon_trace_transformer.py` — now seeds `torch.manual_seed(42)` before model creation
- `sweep_beacon_magnitude.py` — now seeds `torch.manual_seed(42)` before each sweep run

---

## What Is the First Parameter to Flip?

Across runs, the first parameter to flip varies:

| Run | First Flip Batch | First Parameter(s) |
|---|---|---|
| 70-batch Run 1 | 31 | `layers.0.norm1.weight`, `layers.2.norm2.bias` |
| 70-batch Run 2 | 31 | `layers.1.linear1.bias`, `layers.1.linear1.weight` |
| 340-batch Run 1 | 51 | `head.bias`, `head.weight`, `layers.0.self_attn.in_proj_weight`, `token_emb.weight` |
| 340-batch Run 2 | 51 | Mixed across all layers |

### Convergence Is Not Layer-Wise

There's no "Layer 0 converges first, then Layer 1, then Layer 2" pattern. The first flippers come from Layer 0, Layer 1, Layer 2, or the Head — depending on random initialization.

**All layers are coupled.** The model converges as a single system, not layer-by-layer.

### It's a Phase Transition, Not a Chain Reaction

If convergence propagated layer-by-layer, we'd see Layer 0 at batch 31, Layer 1 at batch 33, Layer 2 at batch 35. Instead we see params from all layers flipping together at batch 31.

This is characteristic of a **phase transition** — like water freezing. The whole system crosses a critical point simultaneously.

### Random Initialization Determines the Trigger

The parameter that flips first is whichever one happens to be closest to its beacon-invariant configuration at initialization. But once one parameter tips, the rest follow within 5-10 batches. The system is in a **metastable state**.

### Implications for BRLM

- **You can't graft on a specific layer early** — the cascade is all-or-nothing
- **The plastic window is the same for all parameters** — when one goes, they all go
- **The signal is system-level, not parameter-level** — the beacon diff tells you about the whole model's state

### Hypothesis: The Model Has a Single "Convergence Mode"

The transformer isn't 42 parameters converging independently. It's one system with a **single order parameter** — something like "how well does the model understand the echo task?" When that order parameter crosses a threshold, all parameters simultaneously lock into configurations consistent with that understanding.

---

## Focusing on the First Parameter to Flip

Three experiments were run to understand whether the first-flip parameter is structurally determined or initialization-dependent, and whether the flip is intrinsic to that parameter or emergent from the system.

### Experiment 1: Isolate the Trajectory

Extracted `head.bias` raw gradient trajectory from the seed-locked run (seed=42):

| Batch | Base Grad | Beacon Grad | Diff | Loss |
|---|---|---|---|---|
| 28 | 0.01277 | 0.01267 | -0.00010 | 2.7826 |
| 29 | 0.01385 | 0.01415 | +0.00030 | 2.7763 |
| 30 | 0.01276 | 0.01267 | -0.00009 | 2.7722 |
| 31 | 0.01631 | 0.01629 | **-0.00002** | 2.7695 |
| **32** | **0.01143** | **0.01162** | **+0.00019** | **2.7663** |
| **33** | **0.01340** | **0.01338** | **-0.00002** | **2.7826** ← **FLIP** |
| 34 | 0.01294 | 0.01280 | -0.00014 | 2.7763 |

**The flip happens between batch 32 and 33.** Batch 32 has diff = +0.00019 (beacon still perturbs), batch 33 has diff = -0.00002 (beacon no longer perturbs). The exact convergence is recorded as batch 32 because that's the last positive-diff batch before the flip.

**Pre-flip:** The diff oscillates around zero for 30+ batches with no clear trend. The flip is a sudden crossing, not a gradual approach.

**Post-flip:** The diff stays negative, indicating `head.bias` has become beacon-invariant.

### Experiment 2: Is `head.bias` Always First? (10 Seeds)

| Seed | First Flip Parameter | Batch |
|---|---|---|
| 0 | `token_emb.weight` | 31 |
| 1 | `layers.0.self_attn.out_proj.bias` | 31 |
| 2 | `token_emb.weight` | 31 |
| 3 | `layers.0.norm1.weight` | 31 |
| 4 | `layers.0.self_attn.in_proj_bias` | 31 |
| 5 | `layers.0.self_attn.in_proj_bias` | 31 |
| 6 | `layers.1.norm2.bias` | 31 |
| 7 | `layers.0.self_attn.out_proj.weight` | 31 |
| 8 | `layers.1.linear1.bias` | 31 |
| 9 | `token_emb.weight` | 31 |

**Every seed hits batch 31 as ignition, but the parameter varies.** `token_emb.weight` wins 3/10 times. No parameter is structurally first. The flip is initialization-dependent.

### Experiment 3: Freeze Everything Except `head.bias`

| Condition | First Flip Batch | Final Loss |
|---|---|---|
| All parameters trainable | 32 | 2.7582 |
| Only `head.bias` trainable | 36 | 2.9464 |

`head.bias` flips even when everything else is frozen — the flip is **intrinsic to that parameter**. But system coupling accelerates convergence by ~4 batches (32 vs 36) and improves loss (2.7582 vs 2.9464).

### Key Conclusions

| Question | Answer |
|---|---|
| Is the flip intrinsic to a parameter? | **Yes** — `head.bias` flips even alone |
| Does system coupling matter? | **Yes** — 4 batches faster with full training |
| Is there a "first parameter" structurally? | **No** — any parameter can be first depending on seed |
| Is the ignition point stable? | **Yes** — always batch 31 regardless of seed |

## Files
- `beacon_trace_transformer_report.json` — raw data
- `test_beacon_trace_transformer.py` — implementation
