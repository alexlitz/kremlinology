"""Sniffing cache tiers from hit latency (pooled prefill-sweep and retention runs; no API calls).

A cache hit served from GPU memory costs (almost) nothing per cached token; a hit that has to be brought in from
host memory, SSD or another node costs time proportional to its size. So, per endpoint:
  hit slope   = d(warm TTFT)/d(cached tokens), from reported hits across prompt sizes (per-size medians, bootstrap CI)
  miss slope  = d(cold TTFT)/d(prompt tokens)  (the recompute cost per token)
  ratio       = hit slope / miss slope: 0 = resident (free hits), 1 = a "hit" costs as much as recomputing
  load rate   = 1 / hit slope in tokens/s, and in GB/s under a stated KV size (bf16; fp8 halves it)
Tiers show up as (a) a hit slope clearly above zero, (b) a hit slope that changes with the cache's age.
Part B reads the retention probes (1.5k-token prompts): elapsed time of reported hits by delay, per endpoint.
Run: python experiments/cache_tier_sniff_v1.py
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from prefill_recovery_analysis_v1 import load  # noqa: E402

DATA = Path(__import__('os').environ.get('KREMLINOLOGY_DATA', 'data'))
SWEEPS = [DATA / f'kremlinology-prefill-sweep-v{v}' for v in ('1', '2', '3', '4', '5', '6', '6b', '7', '7b')]
TTLS = [DATA / f'kremlinology-cache-ttl-probe-v{v}' for v in (1, 2, 3, 4)]
KV = {'moonshotai/kimi-k2.6': 70272, 'deepseek/deepseek-v4.1-flash': 78080}  # bytes/token at bf16; V4.1 assumed = V3.2
BOOT = 2000
rng = np.random.default_rng(20261123)


def slope_ci(rows, xkey):
    sizes = sorted({r['size'] for r in rows})
    groups = [[r for r in rows if r['size'] == s] for s in sizes]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 3:
        return None

    def fit(sample):
        x = [np.median([r[xkey] for r in g]) for g in sample]
        y = [np.median([r['ttft'] for r in g]) for g in sample]
        return np.polyfit(x, y, 1)
    slope, intercept = fit(groups)
    draws = [fit([[g[i] for i in rng.integers(0, len(g), len(g))] for g in groups])[0] for _ in range(BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(slope), float(lo), float(hi), float(intercept), len(groups)


def main():
    out = dict(slopes=[], age=[])
    by = load([s for s in SWEEPS if s.exists()])
    print('A. hit slope vs miss slope (microseconds per token)')
    for (model, provider), rows in sorted(by.items()):
        cold = [r for r in rows if r['kind'] == 'cold' and r['cached'] < 0.1 * r['prompt']]
        hits = [dict(r, cachedtok=r['cached']) for r in rows if r['kind'] == 'warm' and r['cached'] >= 0.5 * r['prompt']]
        c, h = slope_ci(cold, 'prompt'), slope_ci(hits, 'cachedtok')
        if not c or not h:
            continue
        ratio = h[0] / c[0] if c[0] > 0 else None
        kv = KV.get(model)
        rate = 1 / h[0] if h[0] > 0 else None
        e = dict(model=model, provider=provider, miss_us_per_token=c[0] * 1e6, miss_ci=[c[1] * 1e6, c[2] * 1e6],
                 hit_us_per_token=h[0] * 1e6, hit_ci=[h[1] * 1e6, h[2] * 1e6], ratio=ratio,
                 hit_intercept_s=h[3], miss_intercept_s=c[3], sizes=h[4], max_size=max(r['size'] for r in hits),
                 hit_tokens_per_s=rate, hit_gb_per_s=(rate * kv / 1e9 if rate and kv else None),
                 resident=bool(h[1] <= 0 or (ratio is not None and ratio < 0.1)))
        out['slopes'].append(e)
        gb = f", {e['hit_gb_per_s']:.1f} GB/s" if e['hit_gb_per_s'] else ''
        print(f"  {model.split('/')[1][:20]:20s} {provider:14s} miss {e['miss_us_per_token']:6.1f} us/tok | hit "
              f"{e['hit_us_per_token']:6.2f} [{e['hit_ci'][0]:6.2f},{e['hit_ci'][1]:6.2f}] | ratio "
              f"{ratio:5.2f} | hit path {'resident' if e['resident'] else f'{rate / 1000:,.0f}k tok/s' + gb} "
              f"(to {e['max_size'] // 1000}k)")
    print('\nB. reported-hit latency by cache age (retention probes, ~1.5k-token prompts; seconds, client clock)')
    lat = defaultdict(lambda: defaultdict(list))
    miss = defaultdict(list)
    for k, run in enumerate(TTLS):
        if not (run / 'requests.jsonl').exists():
            continue
        trials = defaultdict(dict)
        for line in (run / 'requests.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trials[(k, r['trial'])][r['role']] = r
        for pair in trials.values():
            a, b = pair.get('A'), pair.get('B')
            if not a or not b or a.get('status') != 200 or b.get('status') != 200:
                continue
            key = (b['model'].split('/')[-1], b['provider'])
            miss[key].append(a['elapsed_s'])
            u = b.get('usage') or {}
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
            if u.get('prompt_tokens') and cached >= 0.5 * u['prompt_tokens']:
                lat[key][b['delay']].append(b['elapsed_s'])
    for key in sorted(lat):
        cells = {d: v for d, v in sorted(lat[key].items()) if len(v) >= 4}
        if len(cells) < 2:
            continue
        row = dict(model=key[0], provider=key[1], miss_median=st.median(miss[key]),
                   by_delay={d: dict(n=len(v), median=st.median(v)) for d, v in cells.items()})
        out['age'].append(row)
        print(f"  {key[0][:22]:22s} {key[1]:16s} miss {row['miss_median']:.2f} | " +
              '  '.join(f"{d}s: {st.median(v):.2f} (n={len(v)})" for d, v in cells.items()))
    (ROOT / 'reports/cache-tier-sniff-2026-11-23.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
