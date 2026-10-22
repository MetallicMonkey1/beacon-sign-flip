"""
Corrected beacon sign-flip detector (v2).

Fixes vs the repo's detector:
  - No 30-batch dead-zone skip: scanning starts at batch 0.
  - Smoothing: rolling mean window W (default 10) instead of raw window=1 values.
  - Persistence: a "lock" requires the smoothed diff to cross + -> - AND stay
    negative for at least 80% of the next P batches (default 15) of smoothed
    values, with at least P batches of data after the crossing.
  - Null model: the same detector is run on time-shuffled copies of each
    parameter's diff series (200 trials) so 'locks found' can be compared
    against what pure noise with identical values produces.

Input: a beacon_trace_transformer_report.json (uses per_batch_raw_grads).
Usage: python detector_v2.py <report.json> [label]
"""
import json, sys, collections, random, statistics

W = 10          # smoothing window
P = 15          # persistence horizon (smoothed batches)
PERSIST_FRAC = 0.8
NULL_TRIALS = 200

def load_diffs(path):
    d = json.load(open(path))
    raw = d['per_batch_raw_grads']
    diffs = {}
    for p, series in raw.items():
        n = len(series['base'])
        diffs[p] = [series['beacon'][i] - series['base'][i] for i in range(n)]
    return diffs

def smooth(s, w=W):
    return [sum(s[i - w + 1:i + 1]) / w for i in range(w - 1, len(s))]

def detect_locks(diffs, w=W, p=P):
    """Return {param: lock_batch} using smoothed series; batch numbers are
    1-based positions in the smoothed series mapped back (+w-1)."""
    locks = {}
    for name, s in diffs.items():
        sm = smooth(s, w)
        for i in range(len(sm) - 1):
            if sm[i] > 0 and sm[i + 1] < 0:
                after = sm[i + 1:i + 1 + p]
                if len(after) < p:
                    break  # not enough horizon left to confirm persistence
                if sum(1 for v in after if v < 0) / len(after) >= PERSIST_FRAC:
                    locks[name] = i + 1 + (w - 1)  # batch where crossing completes
                    break
    return locks

def sign_balance_curve(diffs, w=W, block=20):
    """Fraction of params with negative smoothed diff, per block of batches."""
    sms = {p: smooth(s, w) for p, s in diffs.items()}
    n = min(len(v) for v in sms.values())
    out = []
    for b0 in range(0, n, block):
        vals = [sms[p][i] for p in sms for i in range(b0, min(b0 + block, n))]
        # fraction of (param,batch) points negative, and fraction of params whose block mean is negative
        frac_pts = sum(1 for v in vals if v < 0) / len(vals)
        pmeans = [sum(sms[p][b0:min(b0 + block, n)]) / len(sms[p][b0:min(b0 + block, n)]) for p in sms]
        frac_params = sum(1 for v in pmeans if v < 0) / len(pmeans)
        out.append((b0 + w, frac_pts, frac_params))
    return out

def null_locks(diffs, trials=NULL_TRIALS, seed=0):
    rng = random.Random(seed)
    counts = []
    for _ in range(trials):
        sh = {}
        for p, s in diffs.items():
            s2 = s[:]
            rng.shuffle(s2)
            sh[p] = s2
        counts.append(len(detect_locks(sh)))
    return counts

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    diffs = load_diffs(path)
    n_batches = len(next(iter(diffs.values())))
    print(f'=== {label}: {len(diffs)} params, {n_batches} batches')
    for w in (5, 10, 20):
        locks = detect_locks(diffs, w=w)
        hist = collections.Counter(locks.values())
        spread = (max(hist) - min(hist)) if hist else None
        print(f'  W={w:2d} P={P}: locked {len(locks)}/{len(diffs)}'
              + (f' | lock batches {min(hist)}..{max(hist)} spread={spread} | hist {dict(sorted(hist.items()))}' if hist else ''))
    locks = detect_locks(diffs)
    null = null_locks(diffs)
    print(f'  NULL (shuffled, W={W}): locked mean {statistics.mean(null):.1f}, range {min(null)}-{max(null)}  vs REAL {len(locks)}')
    print('  sign-balance (block start batch | frac smoothed points negative | frac params with negative block mean):')
    for b, fp, fpar in sign_balance_curve(diffs):
        print(f'    batch {b:3d}: points {fp:.2f} | params {fpar:.2f}')

if __name__ == '__main__':
    main()
