"""
Magnitude analyzer v3 for beacon data.

Hypothesis under test (from rounds 1-2): the SIGN of the beacon gradient
difference is noise, but its MAGNITUDE is signal - per-parameter beacon
sensitivity |mean|grad_beacon| - mean|grad_base|| grows as a parameter is
recruited into the task, tracking learning (loss collapse), not step count.

Analyses per report JSON:
  1. Population curve: block mean sensitivity vs block mean loss (10-batch
     blocks), and their correlation.
  2. Growth ranking: per-param late-mean / early-mean sensitivity.
  3. Recruitment timing: first batch where smoothed (W=10) sensitivity
     exceeds max(2x early mean, early mean + global early median) and stays
     above it for >=80% of the next 15 batches. Histogram + median.
  4. Loss-collapse timing for comparison (first batch loss < 1.0 and < 0.5).
"""
import json, sys, collections, statistics

W = 10
PERSIST = 15

def load(path):
    d = json.load(open(path))
    raw = d['per_batch_raw_grads']
    sens = {}
    for p, s in raw.items():
        n = len(s['base'])
        sens[p] = [abs(s['beacon'][i] - s['base'][i]) for i in range(n)]
    return sens, d['loss_history']

def smooth(s, w=W):
    return [sum(s[i - w + 1:i + 1]) / w for i in range(w - 1, len(s))]

def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0

def analyze(path, label):
    sens, loss = load(path)
    params = list(sens)
    nb = len(loss)
    print(f'=== {label}: {len(params)} params, {nb} batches')
    # loss collapse timing
    b1 = next((i + 1 for i, v in enumerate(loss) if v < 1.0), None)
    b05 = next((i + 1 for i, v in enumerate(loss) if v < 0.5), None)
    print(f'  loss: first {loss[0]:.2f} -> final {loss[-1]:.3f} | <1.0 at batch {b1} | <0.5 at batch {b05}')
    # 1) population curve in 25-batch blocks
    blocks = []
    for b0 in range(0, nb, 25):
        seg = [sens[p][i] for p in params for i in range(b0, min(b0 + 25, nb))]
        blocks.append((b0 + 1, sum(seg) / len(seg), sum(loss[b0:min(b0 + 25, nb)]) / len(loss[b0:min(b0 + 25, nb)])))
    print('  blocks (start | mean sensitivity | mean loss):')
    for b, s, l in blocks:
        print(f'    {b:3d} | {s:.2e} | {l:.3f}')
    corr = pearson([s for _, s, _ in blocks], [l for _, _, l in blocks])
    print(f'  corr(block sensitivity, block loss) = {corr:.2f}')
    # 2) growth ranking (early = batches 1-25, late = last 50)
    early = {p: sum(sens[p][:25]) / 25 for p in params}
    late = {p: sum(sens[p][-50:]) / 50 for p in params}
    growth = sorted(params, key=lambda p: late[p] / max(early[p], 1e-9), reverse=True)
    doubled = sum(1 for p in params if late[p] > 2 * early[p])
    print(f'  params with late sensitivity >2x early: {doubled}/{len(params)}')
    print('  top growth (param | early -> late | ratio):')
    for p in growth[:8]:
        print(f'    {p:45s} {early[p]:.2e} -> {late[p]:.2e} | {late[p] / max(early[p], 1e-9):.1f}x')
    print('  bottom growth:')
    for p in growth[-3:]:
        print(f'    {p:45s} {early[p]:.2e} -> {late[p]:.2e} | {late[p] / max(early[p], 1e-9):.1f}x')
    # 3) recruitment timing
    all_early = sorted(early.values())
    gmed = all_early[len(all_early) // 2]
    recruit = {}
    sms = {p: smooth(sens[p]) for p in params}
    for p in params:
        thr = max(2 * early[p], early[p] + gmed)
        sm = sms[p]
        for i in range(len(sm)):
            if sm[i] > thr:
                after = sm[i:i + PERSIST]
                if len(after) >= PERSIST and sum(1 for v in after if v > thr) / len(after) >= 0.8:
                    recruit[p] = i + W  # approximate batch
                    break
    if recruit:
        times = sorted(recruit.values())
        hist = collections.Counter((t // 25) * 25 + 1 for t in times)
        print(f'  recruited: {len(recruit)}/{len(params)} | median batch {times[len(times) // 2]} | by 25-batch block: {dict(sorted(hist.items()))}')
    else:
        print('  recruited: 0')
    print()

if __name__ == '__main__':
    for path, label in [
        ('exp/adam300/beacon_trace_transformer_report.json', 'Adam-300 (learns ~batch 160)'),
        ('exp/sgd300/beacon_trace_transformer_report.json', 'SGD-300 (learns ~batch 245)'),
        ('exp/big/beacon_trace_transformer_report.json', 'BIG-70 (no learning yet)'),
    ]:
        analyze(path, label)
