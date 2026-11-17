"""Adaptive sampling-only estimate of the TV distance between two providers' next-token distributions.

No logprobs needed. Per provider, in rounds:
  - plain samples at T=1 give absolute shares (with an OTHER bucket for everything unnamed);
  - hot samples at T=2 with top-K=10 reach the tail; a pooled multinomial fit over both runs gives the named tokens'
    relative shares (logit gaps), scaled by the plain-sample mass of the named set to absolute probabilities.
TV is computed over the union of named tokens plus one OTHER bucket (a lower bound on the full TV, matching the
logprob-based comparisons, which also see only the top tokens). A nonparametric bootstrap (multinomial resampling of
every run) gives a 95% interval. Rounds continue until the interval half-width is below TARGET or the budget is spent,
and each round goes to the provider whose own sampling noise contributes more of the TV variance.

Validation (this file, local, no API calls): Llama-3.2-3B fp32 as provider A; B = bf16 (true TV ~1.3%) or fp32 logits
plus Gaussian noise (TV ~5%, like real provider gaps). Truth is used only to score: coverage of the partition TV and
error against the full TV. Hosted use: `Sampler` is the only thing to swap (adaptive_tv_hosted_v1.py).
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))
import logit_reconstruction_benchmark_v1 as bench  # noqa: E402

PLAIN, HOT = (1.0, 0, 1.0), (2.0, 10, 1.0)
ROUND = 500           # samples per provider per round (half plain, half hot)
BUDGET = 8_000        # samples per provider per prompt
TARGET = 0.005        # stop when the 95% CI half-width on TV is below 0.5 points
BOOT = 120
MIN_COUNT = 3
OUT = ROOT / 'reports/adaptive-tv-2026-11-17'


class LocalSampler:
    def __init__(self, logits, generator):
        self.logits, self.generator = logits, generator

    def sample(self, setting, count):
        return bench.draw(self.logits, setting, count, self.generator)


class Provider:
    def __init__(self, sampler):
        self.sampler, self.plain, self.hot, self.used = sampler, Counter(), Counter(), 0

    def round(self, n):
        self.plain += self.sampler.sample(PLAIN, n // 2)
        self.hot += self.sampler.sample(HOT, n - n // 2)
        self.used += n


def estimate(plain, hot):
    """Absolute probabilities of named tokens + 'OTHER', from plain and hot counts."""
    named = [t for t, c in (plain + hot).most_common(30) if c >= MIN_COUNT]
    total = sum(plain.values())
    if not named or not total:
        return {'OTHER': 1.0}
    observations = [bench.run_observation(Counter({t: plain[t] for t in named}), 1.0, minimum=1),
                    bench.run_observation(Counter({t: hot[t] for t in named}), 2.0, minimum=1)]
    fitted = bench.fit(observations)
    named_mass = sum(plain[t] for t in named) / total
    if fitted is None:  # fall back to plain shares
        probs = {t: plain[t] / total for t in named}
    else:
        logits = {t: v[0] for t, v in fitted[0].items()}
        keys = list(logits)
        z = np.array([logits[t] for t in keys])
        shares = np.exp(z - z.max())
        shares /= shares.sum()
        probs = dict(zip(keys, named_mass * shares))
        for t in named:  # tokens the fit could not link keep their plain share
            if t not in probs:
                probs[t] = plain[t] / total
    probs['OTHER'] = max(0.0, 1.0 - sum(probs.values()))
    return probs


def tv(a, b):
    keys = (set(a) | set(b)) - {'OTHER'}
    other_a = a.get('OTHER', 0) + sum(a.get(k, 0) for k in keys if k not in a)
    other_b = b.get('OTHER', 0) + sum(b.get(k, 0) for k in keys if k not in b)
    return 0.5 * (sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) + abs(a.get('OTHER', 0) - b.get('OTHER', 0)))


def resample(counter, rng):
    n = sum(counter.values())
    if not n:
        return Counter()
    keys = list(counter)
    draws = rng.multinomial(n, np.array([counter[k] for k in keys], dtype=float) / n)
    return Counter({k: int(d) for k, d in zip(keys, draws) if d})


def bootstrap(pa, pb, rng, only=None):
    """TV draws; only='a'/'b' resamples just one provider (for allocation)."""
    out = []
    base_a, base_b = estimate(pa.plain, pa.hot), estimate(pb.plain, pb.hot)
    for _ in range(BOOT):
        ea = estimate(resample(pa.plain, rng), resample(pa.hot, rng)) if only in (None, 'a') else base_a
        eb = estimate(resample(pb.plain, rng), resample(pb.hot, rng)) if only in (None, 'b') else base_b
        out.append(tv(ea, eb))
    return np.array(out)


def interval(raw, draws):
    """Plug-in TV is biased upward by sampling noise (|pA - pB| > 0 even when equal): bias-correct the point
    (2*raw - mean of bootstrap) and use the basic bootstrap interval [2*raw - q97.5, 2*raw - q2.5]. Near TV = 0 the
    basic upper bound collapses (to [0, 0] when the providers agree), so the upper bound is never below the
    percentile bound q97.5, which is biased upward and so conservative."""
    q_lo, q_hi = np.percentile(draws, [2.5, 97.5])
    point = max(0.0, 2 * raw - float(draws.mean()))
    return point, max(0.0, 2 * raw - q_hi), max(2 * raw - q_lo, float(q_hi))


def adaptive(sa, sb, rng):
    pa, pb = Provider(sa), Provider(sb)
    pa.round(ROUND)
    pb.round(ROUND)
    history = []
    while True:
        draws = bootstrap(pa, pb, rng)
        raw = tv(estimate(pa.plain, pa.hot), estimate(pb.plain, pb.hot))
        point, lo, hi = interval(raw, draws)
        history.append(dict(used_a=pa.used, used_b=pb.used, tv=point, tv_raw=raw, lo=float(lo), hi=float(hi)))
        if (hi - lo) / 2 < TARGET or (pa.used >= BUDGET and pb.used >= BUDGET):
            break
        va, vb = bootstrap(pa, pb, rng, 'a').var(), bootstrap(pb, pa, rng, 'a').var()
        target = pa if (va >= vb and pa.used < BUDGET) or pb.used >= BUDGET else pb
        target.round(ROUND)
    return pa, pb, history


def main():
    torch.set_num_threads(8)
    items = bench.prompts()[:int(__import__('os').environ.get('ATV_PROMPTS', '60'))]
    truth = bench.next_logits(torch.float32, items)
    bf16 = bench.next_logits(torch.bfloat16, items)
    generator = torch.Generator().manual_seed(7)
    rng = np.random.default_rng(7)
    noise = torch.Generator().manual_seed(11)
    rows = []
    for p in range(len(items)):
        noisy = truth[p] + torch.randn(truth[p].shape, generator=noise) * 0.35
        for label, other in (('bf16', bf16[p]), ('noise', noisy)):
            pa, pb, hist = adaptive(LocalSampler(truth[p], generator), LocalSampler(other, generator), rng)
            ea, eb = estimate(pa.plain, pa.hot), estimate(pb.plain, pb.hot)
            ta, tb = torch.softmax(truth[p].double(), 0), torch.softmax(other.double(), 0)
            full = float(0.5 * (ta - tb).abs().sum())
            keys = (set(ea) | set(eb)) - {'OTHER'}
            part_a = {k: float(ta[k]) for k in keys}
            part_b = {k: float(tb[k]) for k in keys}
            part_a['OTHER'], part_b['OTHER'] = 1 - sum(part_a.values()), 1 - sum(part_b.values())
            part = tv(part_a, part_b)
            last = hist[-1]
            rows.append(dict(prompt=p, pair=label, tv_est=last['tv'], tv_raw=last['tv_raw'], lo=last['lo'], hi=last['hi'], tv_partition=part,
                             tv_full=full, covered=last['lo'] <= part <= last['hi'], samples=pa.used + pb.used,
                             rounds=len(hist)))
        if p % 10 == 9:
            print(p + 1, 'prompts', flush=True)
    summary = {}
    for label in ('bf16', 'noise'):
        r = [x for x in rows if x['pair'] == label]
        summary[label] = dict(
            prompts=len(r), coverage_partition=float(np.mean([x['covered'] for x in r])),
            median_abs_err_partition=float(np.median([abs(x['tv_est'] - x['tv_partition']) for x in r])),
            median_abs_err_full=float(np.median([abs(x['tv_est'] - x['tv_full']) for x in r])),
            median_abs_err_raw=float(np.median([abs(x['tv_raw'] - x['tv_partition']) for x in r])),
            median_bias_raw=float(np.median([x['tv_raw'] - x['tv_partition'] for x in r])),
            median_bias_corrected=float(np.median([x['tv_est'] - x['tv_partition'] for x in r])),
            median_tv_full=float(np.median([x['tv_full'] for x in r])),
            median_tv_partition=float(np.median([x['tv_partition'] for x in r])),
            median_ci_halfwidth=float(np.median([(x['hi'] - x['lo']) / 2 for x in r])),
            median_samples=float(np.median([x['samples'] for x in r])))
        print(label, json.dumps(summary[label]))
    OUT.parent.mkdir(exist_ok=True)
    Path(f'{OUT}.json').write_text(json.dumps(dict(round=ROUND, budget=BUDGET, target=TARGET, boot=BOOT,
                                                   summary=summary), indent=1))
    Path(f'{OUT}-records.json').write_text(json.dumps(rows))


if __name__ == '__main__':
    main()
