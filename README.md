# beacon-sign-flip
A Lightweight Beacon Test Reveals Rapid Optimizer-Driven Phase Transitions in Transformer Training. Detecting Per-Parameter Plasticity Windows.
Beacon Sign-Flip Convergence Detector
================================================================================

A lightweight, one-pass diagnostic that determines which parameters in a neural
network are still plastic (adaptable) versus locked (converged) -- by measuring
how each parameter's gradient responds to a small input perturbation.

================================================================================
THE CORE DISCOVERY
================================================================================

During transformer training on the "echo previous token" task, we observed a sharp
phase transition: parameters flip from beacon-sensitive to beacon-invariant
within a 5-batch window around batch 31. This is not gradual. It is a cascade.

Training Phase          beacon_grad - baseline_grad    Interpretation
--------------------    ---------------------------    ------------------------
Dead zone (0-30)          Oscillates, near-zero          Model is random; no
coherent direction

Plastic zone (~30-35)   POSITIVE                       Noise pushes the model
off-course; gradients
increase to compensate

Locked zone (>35)       NEGATIVE                       Noise adds confusion;
gradients shrink because
weights have stabilized

The flip happens simultaneously across ALL LAYERS -- not layer-by-layer. It is
a system-level phase transition, like water freezing.

THE MOST SURPRISING FINDING
---------------------------

The cascade occurs even with completely random data and no learnable task.

Condition                       Final Loss    Flipped    Meaning
------------------------------  ----------    -------    ------------------------
Echo task (structured)          2.76          42/42      Task learned; optimizer
settled

No positional encoding            2.69          41/41      Task unlearnable;
cascade still happens

Random data (no structure)      2.79          42/42      Optimizer stabilized;
task was never learned

Beacon encodes target             0.06          7/42       Task mastered; most
params stay plastic

Conclusion: The sign-flip detects OPTIMIZER STABILIZATION, not TASK MASTERY.
These are orthogonal properties.

================================================================================
HOW IT WORKS
================================================================================

One-shot plasticity probe for any differentiable model:

    # 1. Forward + backward with a small random perturbation (beacon)
    loss_beacon = model(x + beacon)
    loss_beacon.backward()
    grad_beacon = collect_grads(model)

    model.zero_grad(set_to_none=True)  # discard beacon gradients

    # 2. Forward + backward on clean input
    loss_base = model(x)
    loss_base.backward()
    grad_base = collect_grads(model)

    # 3. Per-parameter sensitivity
    diff = grad_beacon - grad_base

    if diff > 0:
        parameter is PLASTIC  -> still adapting, safe to fine-tune
    if diff < 0:
        parameter is LOCKED   -> optimizer has settled, grafting has minimal effect

The beacon is OBSERVATIONAL ONLY -- it never touches the optimizer step. The
measurement has no side effects.

================================================================================
KEY EXPERIMENTAL RESULTS
================================================================================

1. THE 2D SUBSPACE
------------------

The beacon signal concentrates in a 2-dimensional subspace (dimensions 33 and 37 in
a 64D model) across all components. The cascade is sharp because the model only
needs to suppress these 2 dimensions.

Component           Total Sensitivity    Mean Per Weight
----------------    -----------------    ---------------
LayerNorm           1.56                 0.00204
Attention Q/K/V     13.60                0.00037
FFN                 32.87                0.00067

LayerNorm is the most sensitive per-weight -- it directly scales the
beacon-carrying dimensions.

2. UPWARD PROPAGATION FROM LOSS
-------------------------------

Convergence propagates UPWARD FROM THE LOSS, not downward from the input.

Evidence: attaching a probe network at Layer 2 that forces beacon preservation
suppresses the cascade in all layers below. If embeddings learned to filter the
beacon, the probe wouldn't matter. It does matter -- the loss signal coordinates
lock-in across all layers simultaneously.

3. THE PLASTIC WINDOW IS ~5 BATCHES WIDE
-----------------------------------------

With raw per-batch recording (no smoothing):

  Batch 31:  ignition point -- 14 parameters flip simultaneously
  Batches 32-35:  aftershocks -- remaining parameters follow
  Batch 55:  essentially complete -- 35/42 flipped

Bucket analysis is misleading. Averaging over 100-batch buckets makes the
cascade look gradual. Per-batch recording reveals it is explosive.

4. BEACON MAGNITUDE IS IRRELEVANT (ABOVE A THRESHOLD)
-----------------------------------------------------

Even at magnitude 10.0 (100x the signal), the cascade still happens.
Transformers mathematically eliminate common-mode perturbations via softmax and
LayerNorm. The sign-flip is not "learned filtering" -- it is optimizer
stabilization into a configuration where gradient differences become numerically
zero.

================================================================================
FILE INVENTORY
================================================================================

Main Files
----------
BRLM/beacon_sign_flip_convergence.md      Full research log -- all experiments,
                                           data tables, and interpretations
BRLM/beacon_tracer.py                     Hook-based tracer that injects beacons
                                           and builds live path graphs
test_beacon_trace_transformer.py          Main sign-flip experiment script (echo
                                           task, 70-batch runs)
test_beacon_trace_training.py             Training-time beacon tracing with
                                           per-batch recording
test_beacon_trace.py                      Lightweight smoke test for beacon
                                           gradient diff
sweep_beacon_magnitude.py                 Magnitude sweep (0.01-1.0) showing
                                           sweet spot at 0.05-0.2
test_high_magnitude_sweep.py              Extreme magnitudes (1.0-10.0)
                                           proving architecture handles
                                           common-mode
test_no_pos_emb.py                        Ablates positional encoding --
                                           cascade still happens
test_shuffled_positions.py                Random data baseline -- cascade still
                                           happens
test_copy_first_token.py                  Harder task -- wider cascade, same
                                           ignition point
test_beacon_reconstruction.py             Adds beacon reconstruction head --
                                           cos_sim stays near zero
test_double_beacon.py                     Two mixed beacons -- model disentangles
                                           signal from noise
test_probe_gateway.py                     Probe at Layer 2 -- suppresses
                                           cascade, proves upward propagation
test_beacon_preservation_penalty.py       Competing loss at various depths --
                                           partial suppression
test_orthogonal_rotation.py               Rotates beacon every N batches --
                                           direction irrelevant
test_position_dependent_beacon.py         Different beacon per position -- still
                                           dies (task doesn't need it)
test_beacon_encodes_target.py             Beacon IS the target -- cascade
                                           suppressed (7/42 flip)
analyze_beacon_trajectories.py            Trajectory analyzer: predicts flips
                                           from slope + magnitude
find_beacon_sensitive_weights.py          Ranks all ~170K individual weights by
                                           beacon sensitivity
beacon_trace_transformer_report.json      Raw per-batch gradients from 70-batch
                                           runs
beacon_trace_training_report.json         Raw data from training-time tracing
beacon_magnitude_sweep.json               Results from magnitude sweep

Circuit Mapping (Complementary Analysis)
----------------------------------------
approach1_layer_wise_probing.py           Probe MSE per layer
                                           Finding: Beacon present uniformly at
                                           all layers

approach2_activation_patching.py          KL divergence from patching
                                           Finding: Robustness increases but
                                           never vanishes

approach3_attention_analysis.py           Copy-head attention scores
                                           Finding: Beacon changes attention by
                                           ~0.001 -- essentially zero

approach4_representation_geometry.py      Cosine similarity, projection
                                           Finding: Deeper layers become
                                           position-invariant

approach5_gradient_attribution.py         Per-path gradient diff
                                           Finding: Sensitivity distributed
                                           across all layers

approach6_residual_stream.py              Hidden state diff norm
                                           Finding: Beacon difference decreases
                                           10-25% over training

approach7_ablation_knockout.py            Component knockout
                                           Finding: All layers critical; Head 1
                                           most beacon-sensitive

================================================================================
QUICK START
================================================================================

Run the main sign-flip experiment
---------------------------------

    python test_beacon_trace_transformer.py

This trains a 3-layer transformer on the echo task for 70 batches, recording
per-batch beacon vs baseline gradients for all 42 parameters.

Output: beacon_trace_transformer_report.json

Run the trajectory analyzer
---------------------------

    python analyze_beacon_trajectories.py

Predicts which parameters will flip within the next 10 batches using slope and
magnitude features.

Find the most beacon-sensitive individual weights
-------------------------------------------------

    python find_beacon_sensitive_weights.py

Ranks every single scalar weight by |grad_with_beacon - grad_without_beacon|.

================================================================================
RELATIONSHIP TO BRLM
================================================================================

The Beacon Sign-Flip is a diagnostic primitive within the larger Beacon-Routed
Localist Memory (BRLM) framework:

  - beacon_tracer.py uses the same perturbation idea, but to build live path
    graphs during training rather than measure convergence.
  - The sign-flip tells you WHEN and WHERE the model is still trainable.
  - The path graph tells you WHICH CONTAINERS carry which knowledge.
  - Together, they enable surgical edits: graft onto plastic containers
    identified by the tracer, avoid locked ones flagged by the sign-flip.

BRLM files:
  beacon_tracer.py      -- live path graph construction
  dynamic_router.py     -- sparse routing using the path graph
  container_pool.py     -- growable container pool with grafting support

================================================================================
LIMITATIONS & HONEST CAVEATS
================================================================================

1. Detects optimizer stabilization, not task mastery.
   A model locked in a bad basin will show all-negative diffs and appear "done"
   despite poor performance.

2. Requires forward/backward access.
   You need the model graph and autodiff. Pure black-box weight inspection
   won't work.

3. Architecture-dependent.
   The 2D subspace finding is specific to this transformer. CNNs, RNNs, or other
   architectures will have different sensitivity profiles.

4. Task-relevant perturbations survive.
   If the perturbation encodes task-critical information (as in
   test_beacon_encodes_target.py), the cascade is suppressed. The beacon must be
   task-irrelevant noise.

5. Not a replacement for Fisher/Hessian analysis.
   It is a cheap, one-pass approximation. For surgical precision, full FIM is
   still gold standard.

================================================================================
CITATION
================================================================================

If you use this work, please cite the key findings:

The sign-flip cascade is a universal property of transformer training dynamics:
a discrete phase transition where all parameters simultaneously become
insensitive to small embedding perturbations, driven by optimizer stabilization
rather than task learning. The cascade propagates upward from the loss, happens
even with random data, and concentrates in a low-dimensional subspace shared
across all layers via LayerNorm.
