# Beacon Sign-Flip — Independent Test Results

Independent test, 2026-10-22, prepared at the repository owner's request.
Subject: the public repo
https://github.com/metallicmonkey1/beacon-sign-flip (MIT). All runs on CPU,
PyTorch 2.14.1, Python 3.12, in an isolated venv. Raw outputs for the new runs are
included with this submission (report_*.json).

## Round 1 — Reproduction of the repo as shipped

- Cloned the repo; ran all 24 scripts. 22 ran clean; 2 fail as shipped
  (`test_beacon_trace.py`, `test_beacon_trace_training.py` — they import
  `datasets` and `models.unified_ssm`, neither of which is in the repo).
- Main experiment (`test_beacon_trace_transformer.py`) is deterministic:
  two runs produced bit-identical output.
- Shipped report: 40 of 42 parameter tensors converge, cascade batches
  32–37. Local run: 40 again, cascade batches 31–39 (cross-environment
  numeric drift, same phenomenon as measured by the repo's detector).
- Controls as shipped: no-positional-encoding variant flipped 41/41 (first
  flip batch 31); beacon-encodes-target variant flipped only 8/42 and its
  model actually learned (loss 2.79 → 0.06).
- Magnitude sweep: ignition at batch 31 at every beacon magnitude
  0.01–1.0, 38–41 of 41 converged each time.
- `find_beacon_sensitive_weights.py`: top beacon-sensitive individual
  weights concentrate at dimensions 33 and 37, dominated by LayerNorm
  biases — matching the README's specific claims.
- Weakness found: `analyze_beacon_trajectories.py` as a flip *predictor*
  is poor — best configuration caught 23/200 upcoming flips within
  5 batches with 110 false positives (its input features do carry signal:
  pre-flip diffs run 9–12× baseline).
- Repo gaps: no `requirements.txt` / pinned versions; two broken files
  (above); main script docstring describes a different task ("copy the
  marker value", 1000 batches) than the code runs (echo task, 70 batches).

## Round 2 — Stress tests + corrected detector

Variants run: SGD (70/150/300 batches), larger model (6 layers, dim 256,
76 params), seeds 7/1234, Adam LR 1e-2 and 3e-4, 300-batch Adam and SGD.

- Every variant still "ignited" at batch 31 under the repo detector —
  because that detector skips the first 30 batches as a "dead zone" and
  records the first positive→negative crossing of a raw (window = 1)
  series after batch 30. Ignition before batch 31 is impossible by
  construction.
- Null test: time-shuffling each parameter's gradient-difference series
  (destroying all training dynamics) and applying the repo detector gave
  40/40 "converged" in all 200 trials. Fraction of negative diffs:
  batches 1–30 = 0.51, batches 31–70 = 0.49. Without the skip, first
  crossings occur at batches 1–11.
- Corrected detector `detector_v2.py` (no skip; smoothing window 10;
  lock = smoothed diff crosses +→− and stays negative ≥80% of the next
  15 batches; shuffle null reported alongside):
  - Shipped report: 26 locks vs null mean 19.6 (range 14–26); lock
    batches spread 10–55.
  - Adam 70: 22 vs null 30.2. SGD 70: 25 vs null 29.4. Big model: 18 vs
    null 45.1. Seed 7: 28 vs null 27.2. SGD 150: 38 vs null 36.8.
  - Adam/SGD 300 batches: 40 locks each vs null ~39–40 each; lock times
    scattered across batches 10–277.
  - Only Adam LR 1e-2 exceeded its null (38 vs 29.1, range 22–36).
- The real phase transition in the 300-batch runs is in the loss
  (Adam collapse ~batch 151–165, final 0.033; SGD ~batch 240–250, final
  0.089). Beacon lock times do not cluster near either collapse.
- Verdict: the synchronized batch-31 sign-flip cascade is a detector
  artifact. The sign of the beacon difference behaves as near-zero-mean
  noise in echo conditions. Surviving from round 1: the magnitude
  concentration (dims 33/37, LayerNorm) and the encodes-target asymmetry.

## Round 3 — Magnitude analyzer

`analyzer_v3.py`: tracks per-parameter beacon sensitivity
|mean|grad_beacon| − mean|grad_base|| over training; "recruitment" =
smoothed (W=10) sensitivity exceeds max(2× early mean, early mean +
global early median) and stays above ≥80% of the next 15 batches.

New run created for this round: random-labels control (Adam, 300
batches, targets pure noise; loss stayed at chance ≈2.79 throughout).
Plus an Adam seed-7 replication at 300 batches.

- Adam 300 (seed 42): sensitivity/loss block correlation −0.75;
  34/40 params >2× growth; recruited 34/40, median batch 129
  (loss <1.0 at batch 151, <0.5 at 165).
- Adam 300 (seed 7, replication): correlation −0.85; 34/40 >2×;
  recruited 35/40, median batch 136 (collapse at 152/165).
- SGD 300: correlation −0.89; 40/40 >2×; recruited 35/40, median
  batch 203 (collapse at 240/250). Recruitment timing follows each
  optimizer's own learning schedule (~75 batches apart), not step count.
- Sensitivity rises before the loss visibly moves (Adam: ~2–3× up by
  batch ~101 with loss still 2.47; SGD: rising from ~batch 126, loss
  2.73) — a leading indicator of circuit formation.
- Random-labels control: sensitivity *decays* monotonically
  (6.7e-05 → 2.1e-05 over 300 batches); 0/40 recruited; best growth
  ratio 0.8×. Big model at 70 batches (not yet learning): flat/declining
  sensitivity, 0/76 recruited.
- Top growers in both Adam seeds: layer-0 `self_attn.in_proj_weight`
  (43×/57×), `token_emb.weight` (32×/27×), `pos_emb.weight` (31×/38×),
  layer-0 `linear1.weight` (23×/20×). Output head and late biases grow
  least (≈1×) — they are already beacon-sensitive at initialization.
- Population mean sensitivity: Adam 7.1e-05 → peak 3.9e-04 → settles
  ≈2.4–3.3e-04; SGD 8.8e-05 → 5.0e-04; random-labels 6.7e-05 → 2.1e-05.

## Overall conclusion

The repo's headline (synchronized sign-flip lock-in cascade at batch 31)
does not hold: it is produced by the detector's 30-batch skip plus
window-1 first-crossing rule, and a corrected detector with a shuffle
null finds lock counts indistinguishable from noise.

The beacon probe itself does measure something real, in magnitude:
per-parameter beacon sensitivity grows as parameters are recruited into
the task circuit, tracks each optimizer's learning schedule, precedes
visible loss improvement, is concentrated (dims 33/37, LayerNorm, and
early-layer/embeddings growth), and vanishes (decays) when the task is
unlearnable. Reframed: beacon sensitivity is a relevance/recruitment
indicator, not a lock-in clock.

## Files in this submission

- `INDEPENDENT_REVIEW.md` — this document.
- `detector_v2.py` — corrected sign-flip detector with shuffle null model.
- `analyzer_v3.py` — magnitude/recruitment analyzer.
- `detector_v2_results.log` — detector v2 output on all pre-existing runs.
- `analyzer_v3_results.log` — analyzer v3 output (Adam/SGD 300, big model).
- `report_adam300.json`, `report_sgd300.json`,
  `report_adam300_seed7.json`, `report_random_labels_control.json` —
  per-batch raw gradient data for the new 300-batch runs.
