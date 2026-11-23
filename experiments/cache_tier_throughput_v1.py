"""Bounds on how fast providers load cached KV, from witnessed time to first token (no new requests).

A cache hit's TTFT contains the time to bring the cached prefix to the GPU (zero if it never left HBM), plus fixed
overhead and the new tokens' prefill. So, per endpoint:

1. Marginal rate across cached sizes (prefill sweep warm hits; cache grid with 500 new tokens):
   rate >= (C2 - C1) / (TTFT_hit(C2) - TTFT_hit(C1)), because overhead doesn't shrink with C and attention over a
   longer context only adds time. One-sided 95%: bootstrap the TTFT difference (medians), use its 95th percentile.
   A difference within noise means the load is too fast to see: only the bound from the 95th percentile is given.
2. A slower tier (follow-ups after >= 10 min that still hit, or hit unreported), sandwiched against the fast hits of
   the same prefix size C:
   - upper: rate_slow <= C / (TTFT_slow - TTFT_fast)   (the slow hit does the fast hit's work plus a longer load)
   - lower: rate_slow >= C / (TTFT_slow - floor)        (floor: 5th percentile TTFT of the endpoint's smallest
                                                          uncached requests: overhead every request pays)
Rates are tokens/s; bytes/s = rate x KV bytes per token, which depends on the KV precision and, for sliding-window
models, on whether the sliding layers' cache is kept for every token or only for the last window. Both products are
reported (bf16 KV; fp8 halves them).
Run: python experiments/cache_tier_throughput_v1.py   (reads the runs under DATA; writes reports/cache-tier-throughput-2026-11-17.json)
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import cache_latency_analysis_v1 as lat  # noqa: E402
from prefill_cache_sweep_analysis_v1 import token_times  # noqa: E402
from cache_routing_honesty_analysis_v1 import generation_of  # noqa: E402

DATA = Path(__import__('os').environ.get('KREMLINOLOGY_DATA', 'data'))
SWEEPS = [DATA / f'kremlinology-prefill-sweep-v{v}' for v in (1, 2, 3, 4)]
LATENCY = [DATA / f'kremlinology-cache-latency-{x}' for x in ('v1', 'age-v2', 'age-v3', 'grid-v2', 'grid-v3', 'grid-v4', 'grid-v5')]
# KV bytes per token at bf16 (reports/kv-cache-economics/fig1_bytes_per_token_32k.csv). gpt-oss-120b: 18 full-attention
# layers 36,864 B; its 18 sliding layers add another 36,864 B per token if their cache is kept for every token.
KV = {'moonshotai/kimi-k2.6': {'MLA (as K2.5)': 70272},
      'openai/gpt-oss-120b': {'sliding windows only': 36864, 'all sliding-layer KV kept': 73728},
      'deepseek/deepseek-v4.1-flash': {}}  # V4.1 architecture not published in our configs: tokens/s only
BOOT = 2000
rng = np.random.default_rng(20261117)


def boot_median_diff(a, b):
    """Bootstrap distribution of median(b) - median(a)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.array([np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))) for _ in range(BOOT)])


def bytes_rates(model, rate):
    return {k: rate * v / 1e9 for k, v in KV.get(model, {}).items()}  # GB/s


def sweep_rows():
    by = defaultdict(list)
    for k, run in enumerate(SWEEPS):
        root = run / 'session'
        times = {}
        for line in (root / 'requests.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('status') == 200:
                gen = generation_of(record, root)
                if gen:
                    times[gen] = token_times(record)
        for r in json.loads((run / 'log.json').read_text())['rows']:
            t = times.get(r['generation']) if r['status'] == 200 else None
            if not t or not r.get('prompt_tokens'):
                continue
            size, _, kind = r['tag'].split(':')
            by[(r['model'], r['provider'])].append(dict(size=int(size), kind=kind, ttft=t[0], prompt=r['prompt_tokens'],
                                                        cached=r['cached_tokens']))
    return by


def marginal(rows_small, rows_large, source):
    c1 = statistics.median(r['cached'] for r in rows_small)
    c2 = statistics.median(r['cached'] for r in rows_large)
    d = boot_median_diff([r['ttft'] for r in rows_small], [r['ttft'] for r in rows_large])
    point = float(np.median([r['ttft'] for r in rows_large]) - np.median([r['ttft'] for r in rows_small]))
    hi = float(np.percentile(d, 95))
    bound = (c2 - c1) / hi if hi > 0 else float('inf')
    return dict(source=source, cached_from=c1, cached_to=c2, n=(len(rows_small), len(rows_large)), ttft_diff_s=point,
                ttft_diff_95=hi, min_tokens_per_s=bound)


def main():
    out = dict(sweep=[], grid=[], slow_tier=[])
    for (model, provider), rows in sorted(sweep_rows().items()):
        hits = [r for r in rows if r['kind'] == 'warm' and r['cached'] >= 0.5 * r['prompt']]
        sizes = sorted({r['size'] for r in hits})
        if len(sizes) < 2:
            continue
        m = marginal([r for r in hits if r['size'] == sizes[0]], [r for r in hits if r['size'] == sizes[-1]], 'prefill sweep')
        m.update(model=model, provider=provider, min_gb_per_s=bytes_rates(model, m['min_tokens_per_s']))
        out['sweep'].append(m)
    rows = [r for r in lat.load(LATENCY) if r.get('load', 0) < lat.LOADED]
    grid = defaultdict(list)
    for r in rows:
        p = r['tag'].split(':')
        if p[0] == 'grid' and p[-1] == 'read' and int(p[2]) == 500 and int(p[1]) > 0 and r['cached_tokens'] > 0:
            grid[(r['model'], r['provider'], int(p[1]))].append(dict(ttft=r['ttft'], cached=r['cached_tokens']))
    for model, provider in sorted({(m, p) for m, p, _ in grid}):
        cs = sorted(c for m, p, c in grid if (m, p) == (model, provider))
        g = marginal(grid[(model, provider, cs[0])], grid[(model, provider, cs[-1])], 'cache grid (500 new tokens)')
        g.update(model=model, provider=provider, min_gb_per_s=bytes_rates(model, g['min_tokens_per_s']))
        out['grid'].append(g)
    # slower tier: age follow-ups at >= 600 s vs <= 300 s, same prefix; floor from the endpoint's smallest cold sends
    age = defaultdict(list)
    for r in rows:
        p = r['tag'].split(':')
        if p[0] == 'age' and p[-1] == 'read':
            age[(r['model'], r['provider'])].append(dict(delay=int(p[1]), ttft=r['ttft'], cached=r['cached_tokens'],
                                                         prompt=r['prompt_tokens']))
    small_cold = defaultdict(list)
    for r in rows:
        p = r['tag'].split(':')
        if p[0] == 'grid' and int(p[1]) == 0 and int(p[2]) == 500 and p[-1] == 'read':
            small_cold[(r['model'], r['provider'])].append(r['ttft'])
    for (model, provider), rs in sorted(age.items()):
        fast = [r for r in rs if r['delay'] <= 300 and r['cached'] > 0]
        slow = [r for r in rs if r['delay'] >= 600]
        if len(fast) < 10 or len(slow) < 10:
            continue
        reported = statistics.median(r['cached'] for r in fast)  # the prefix the slow hits must also be reusing
        slow_reported = sum(r['cached'] > 0 for r in slow)
        d = boot_median_diff([r['ttft'] for r in fast], [r['ttft'] for r in slow])
        floor = float(np.percentile(small_cold[(model, provider)], 5)) if small_cold.get((model, provider)) else 0.0
        t_slow = np.asarray([r['ttft'] for r in slow])
        slow_med = float(np.median(t_slow))
        slow_hi = float(np.percentile([np.median(rng.choice(t_slow, len(t_slow))) for _ in range(BOOT)], 95))
        lo_rate = reported / (slow_hi - floor)
        d_lo = float(np.percentile(d, 5))
        hi_rate = reported / d_lo if d_lo > 0 else float('inf')
        out['slow_tier'].append(dict(model=model, provider=provider, cached=reported, n_fast=len(fast), n_slow=len(slow),
                                     slow_hits_reported=slow_reported, ttft_fast=float(np.median([r['ttft'] for r in fast])),
                                     ttft_slow=slow_med, floor_s=floor, extra_s=float(np.median(d)),
                                     min_tokens_per_s=lo_rate, max_tokens_per_s=hi_rate,
                                     gb_per_s={k: (lo_rate * v / 1e9, hi_rate * v / 1e9) for k, v in KV.get(model, {}).items()}))
    for key in ('sweep', 'grid'):
        for m in out[key]:
            gb = ', '.join(f'{k} >= {v:.2f} GB/s' for k, v in m['min_gb_per_s'].items())
            print(f"{key:5s} {m['model'].split('/')[1]:20s} {m['provider']:15s} cached {m['cached_from']:.0f}->{m['cached_to']:.0f} "
                  f"dTTFT {m['ttft_diff_s']:+.3f} s (95%: {m['ttft_diff_95']:+.3f})  >= {m['min_tokens_per_s'] / 1e3:,.0f}k tok/s  {gb}")
    for s in out['slow_tier']:
        gb = ', '.join(f'{k}: {a:.2f}-{b:.2f} GB/s' for k, (a, b) in s['gb_per_s'].items())
        print(f"slow  {s['model'].split('/')[1]:20s} {s['provider']:15s} C={s['cached']:.0f} fast {s['ttft_fast']:.2f} s, slow {s['ttft_slow']:.2f} s "
              f"(+{s['extra_s']:.2f}), floor {s['floor_s']:.2f}, slow hits reported {s['slow_hits_reported']}/{s['n_slow']}: "
              f"{s['min_tokens_per_s'] / 1e3:.0f}k-{s['max_tokens_per_s'] / 1e3:.0f}k tok/s  {gb}")
    (ROOT / 'reports/cache-tier-throughput-2026-11-17.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
