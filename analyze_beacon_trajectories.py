"""
Beacon Trajectory Analyzer — predict sign-flip convergence from per-batch gradients.

Reads beacon_trace_transformer_report.json, extracts raw gradient trajectories,
computes predictive features, and tests simple rules for early flip detection.
"""

import json
import math
from collections import defaultdict


def load_report(path="beacon_trace_transformer_report.json"):
    with open(path, "r") as f:
        return json.load(f)


def compute_slope(values, window=3):
    """Simple linear slope over the last `window` values."""
    if len(values) < window:
        return 0.0
    recent = values[-window:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0


def compute_features(base, beacon, diff, loss_history, t):
    """Compute features for a single parameter at batch t (0-indexed)."""
    b = base[t]
    c = beacon[t]
    d = diff[t]

    # Slope features
    diff_slope_3 = compute_slope(diff[: t + 1], window=3)
    diff_slope_5 = compute_slope(diff[: t + 1], window=5)

    # Beacon / base ratio
    beacon_base_ratio = c / b if abs(b) > 1e-12 else 1.0

    # Beacon shrinking faster than base?
    if t >= 1:
        beacon_delta = abs(c) - abs(beacon[t - 1])
        base_delta = abs(b) - abs(base[t - 1])
        beacon_trend = beacon_delta - base_delta  # negative = beacon shrinking faster
    else:
        beacon_trend = 0.0

    # Diff declining for N consecutive batches?
    diff_declining_3 = False
    if t >= 2:
        diff_declining_3 = diff[t - 2] > diff[t - 1] > diff[t]

    # Distance from zero
    diff_magnitude = abs(d)

    return {
        "batch": t,
        "diff": d,
        "diff_slope_3": diff_slope_3,
        "diff_slope_5": diff_slope_5,
        "beacon_base_ratio": beacon_base_ratio,
        "beacon_trend": beacon_trend,
        "diff_declining_3": diff_declining_3,
        "diff_magnitude": diff_magnitude,
        "loss": loss_history[t] if t < len(loss_history) else 0.0,
    }


def build_dataset(report):
    """Build a dataset of (features, label) for each (param, batch) pair."""
    raw = report["per_batch_raw_grads"]
    converged = {ev["param"]: ev["converged_at_batch"] for ev in report["exact_convergence"]}
    loss_history = report.get("loss_history", [])

    # Max batches across all params
    max_batches = max(len(v["diff"]) for v in raw.values())

    dataset = []
    for param, grads in raw.items():
        base = grads["base"]
        beacon = grads["beacon"]
        diff = grads["diff"]
        flip_batch = converged.get(param, None)  # 1-indexed batch where flip happens

        for t in range(len(diff)):
            # 1-indexed batch number
            batch_1idx = t + 1
            feats = compute_features(base, beacon, diff, loss_history, t)

            # Label: how many batches until flip? (None if never flips)
            if flip_batch is None:
                batches_until_flip = None
            else:
                batches_until_flip = flip_batch - batch_1idx
                if batches_until_flip < 0:
                    batches_until_flip = None  # already flipped

            feats["param"] = param
            feats["batches_until_flip"] = batches_until_flip
            feats["will_flip"] = flip_batch is not None
            feats["already_flipped"] = (
                flip_batch is not None and batches_until_flip is None
            )
            dataset.append(feats)

    return dataset


def evaluate_window_rule(dataset, window=5, diff_threshold=5e-5, loss_gate=2.5):
    """Evaluate rule for predicting flip within next `window` batches.

    Positive: flip happens in [batch+1, batch+window]
    Negative: flip happens later or never
    Loss gate: only predict when loss < threshold (avoids dead zone noise)
    """
    tp = fp = tn = fn = 0
    positive_examples = []
    negative_examples = []

    for row in dataset:
        if row["already_flipped"]:
            continue

        bu = row["batches_until_flip"]
        pred = False

        # Window prediction rule:
        # 1. Loss has dropped (out of dead zone)
        # 2. diff is positive but small (approaching zero from above)
        # 3. diff has negative slope (declining)
        if (
            row["loss"] < loss_gate
            and row["diff"] > 0
            and row["diff_slope_3"] < 0
            and row["diff_magnitude"] < diff_threshold
        ):
            pred = True

        # Positive: flip within window
        if bu is not None and 1 <= bu <= window:
            positive_examples.append(row)
            if pred:
                tp += 1
            else:
                fn += 1
        # Negative: flip later or never
        elif bu is None or bu > window:
            negative_examples.append(row)
            if pred:
                fp += 1
            else:
                tn += 1

    total_pos = tp + fn
    total_neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_pos if total_pos > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "window": window,
        "diff_threshold": diff_threshold,
        "loss_gate": loss_gate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_positive_examples": total_pos,
        "total_negative_examples": total_neg,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
    }


def evaluate_diff_slope_rule(dataset, lookahead=1, slope_threshold=-1e-6):
    """Evaluate a rule based purely on diff slope declining."""
    tp = fp = tn = fn = 0

    for row in dataset:
        if row["already_flipped"]:
            continue

        bu = row["batches_until_flip"]
        # Predict if diff_slope_3 < threshold AND diff is still positive
        pred = row["diff_slope_3"] < slope_threshold and row["diff"] > 0

        if bu == lookahead:
            if pred:
                tp += 1
            else:
                fn += 1
        elif bu is None or bu > lookahead:
            if pred:
                fp += 1
            else:
                tn += 1

    total_pos = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_pos if total_pos > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "lookahead": lookahead,
        "slope_threshold": slope_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_positive_examples": total_pos,
    }


def main():
    print("=" * 70)
    print("BEACON TRAJECTORY ANALYZER")
    print("=" * 70)

    report = load_report()
    dataset = build_dataset(report)

    # Summary stats
    will_flip = [r for r in dataset if r["will_flip"]]
    wont_flip = [r for r in dataset if not r["will_flip"]]
    print(f"\nDataset: {len(dataset):,} (param, batch) pairs")
    print(f"  From params that WILL flip: {len(will_flip):,}")
    print(f"  From params that NEVER flip: {len(wont_flip):,}")

    # Convergence timing distribution
    flip_batches = [r["batches_until_flip"] for r in dataset if r["batches_until_flip"] == 0]
    print(f"  Total flip events in dataset: {len(flip_batches)}")

    # Evaluate window-based rule with loss gating
    print("\n" + "-" * 70)
    print("RULE: loss<gate AND diff>0 AND slope_3<0 AND |diff|<T")
    print("Predicts: flip will happen within next `window` batches")
    print("-" * 70)

    thresholds = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    loss_gates = [3.0, 2.5, 2.0]
    windows = [3, 5, 10]
    best_results = {}

    for window in windows:
        print(f"\n  Window={window} batch(es):")
        best_f1 = -1
        best_res = None
        for loss_gate in loss_gates:
            for thresh in thresholds:
                res = evaluate_window_rule(
                    dataset, window=window, diff_threshold=thresh, loss_gate=loss_gate
                )
                marker = ""
                if res["f1"] > best_f1:
                    best_f1 = res["f1"]
                    best_res = res
                    marker = " <-- BEST"
                print(
                    f"    loss<{loss_gate:.1f} T={thresh:10.0e}:  "
                    f"Precision={res['precision']:.3f}  "
                    f"Recall={res['recall']:.3f}  "
                    f"F1={res['f1']:.3f}  "
                    f"(TP={res['tp']}, FP={res['fp']}, FN={res['fn']}){marker}"
                )
        best_results[window] = best_res

    # Show best predictions for window=5
    print("\n" + "-" * 70)
    print("BEST PREDICTIONS (flip within 5 batches, best config)")
    print("-" * 70)
    best5 = best_results.get(5)
    if best5 and best5["positive_examples"]:
        for row in best5["positive_examples"][:8]:
            param = row["param"]
            batch = row["batch"]
            bu = row["batches_until_flip"]
            print(
                f"  {param:50s} batch={batch:2d} | "
                f"flip_in={bu:2d} batches | "
                f"diff={row['diff']:12.6f} slope_3={row['diff_slope_3']:12.6f} "
                f"loss={row['loss']:.4f}"
            )
        print(
            f"\n  Correctly predicted {best5['tp']}/{best5['total_positive_examples']} "
            f"upcoming flips (within {best5['window']} batches)"
        )
        print(f"  False positives: {best5['fp']}")
        print(f"  Best config: loss<{best5['loss_gate']:.1f}, |diff|<{best5['diff_threshold']:.0e}")
    else:
        print("  No predictions made")

    # Feature distribution analysis: what do flips look like vs non-flips?
    print("\n" + "-" * 70)
    print("FEATURE DISTRIBUTION (1 batch before flip vs random non-flip batch)")
    print("-" * 70)
    flip_feats = [r for r in dataset if r["batches_until_flip"] == 1]
    nonflip_feats = [r for r in dataset if r["batches_until_flip"] is None or r["batches_until_flip"] > 5]
    nonflip_feats = nonflip_feats[:len(flip_feats)]  # sample same size

    def avg(feats, key):
        vals = [abs(r[key]) for r in feats if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print(f"  Feature           | Flip-next-batch | Non-flip (random) | Ratio")
    print(f"  {'-'*18}|{'-'*17}|{'-'*19}|{'-'*6}")
    for key in ["diff", "diff_slope_3", "diff_magnitude"]:
        a = avg(flip_feats, key)
        b = avg(nonflip_feats, key)
        ratio = a / b if b > 0 else 0
        print(f"  {key:18s}| {a:15.6f} | {b:17.6f} | {ratio:6.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
