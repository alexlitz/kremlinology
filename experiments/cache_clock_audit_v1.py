"""Use the miss distribution as the reference clock for cache-hit auditing (2026-11-20; no API calls).

Idea (user's): a cache miss is compute-bound prefill -- its TTFT is comparatively deterministic for a given prompt
size and endpoint -- while a hit's time depends on lookup, tier and placement. So per endpoint, fit the COLD (miss)
TTFT distribution and classify each warm trial by the clock alone:
  clock-hit:  warm below the cold distribution's 10th percentile (faster than misses plausibly are), and separately
              the stricter below-min(cold) floor test;
  clock-miss: warm inside or above the cold distribution.
Comparing that with the provider's own reported cached tokens gives a per-trial confusion count:
  reported hit + clock miss  -> phantom-hit candidates (counter not corroborated; usually discounted, customer-favoured)
  reported miss + clock hit  -> hidden-hit candidates (reuse not reported; billed in full, provider-favoured)
Also reports the coefficient of variation of cold vs reported-hit warm times (the determinism check itself).
Input: reports/cache-honesty-pooled-2026-11-17.json (per-trial cells). Run: python experiments/cache_clock_audit_v1.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n):
    if not n:
        return (None, None)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def main():
    data = json.load(open(ROOT / 'reports/cache-honesty-pooled-2026-11-17.json'))
    out = []
    for t in data['honesty']:
        if not t.get('n'):
            continue
        cells = t['cells']
        colds = sorted(c['cold_ttft'] for c in cells)
        p10 = colds[max(0, int(0.1 * len(colds)) - (1 if len(colds) > 10 else 0))]
        floor = min(colds)
        mean_c, sd_c = st.mean(colds), st.stdev(colds)
        reported_hits = [c for c in cells if c['cached_frac'] > 0.5]
        reported_miss = [c for c in cells if c['cached_frac'] <= 0.5]
        clock_hit = lambda w: w < p10  # noqa: E731
        phantom = [c for c in reported_hits if not clock_hit(c['warm_ttft'])]
        hidden = [c for c in reported_miss if clock_hit(c['warm_ttft'])]
        strict_hidden = [c for c in reported_miss if c['warm_ttft'] < floor]
        strict_phantom = [c for c in reported_hits if c['warm_ttft'] >= st.median(colds)]
        cv = lambda xs: st.stdev(xs) / st.mean(xs) if len(xs) > 3 and st.mean(xs) else None  # noqa: E731
        row = dict(provider=t['provider'], model=t['model'], n=len(cells),
                   cold=dict(mean=round(mean_c, 3), sd=round(sd_c, 3), cv=round(cv(colds), 3) if cv(colds) else None,
                             p10=round(p10, 3), floor=round(floor, 3)),
                   reported_hits=len(reported_hits), reported_miss=len(reported_miss),
                   hit_cv=round(cv([c['warm_ttft'] for c in reported_hits]), 3) if len(reported_hits) > 3 else None,
                   phantom_hits=len(phantom), phantom_ci=wilson(len(phantom), max(1, len(reported_hits))),
                   hidden_hits=len(hidden), hidden_ci=wilson(len(hidden), max(1, len(reported_miss))),
                   strict_hidden=sorted(round(c['warm_ttft'], 2) for c in strict_hidden),
                   verdict=('counter matches clock' if not phantom and not hidden else
                            f"{len(phantom)} phantom / {len(hidden)} hidden of {len(cells)}"))
        out.append(row)
        print(f"{t['provider']:20s} n={len(cells):2d} cold cv {row['cold']['cv'] if row['cold']['cv'] else '-':>5} "
              f"hit cv {row['hit_cv'] if row['hit_cv'] else '-':>5} | p10 {p10:.2f}s floor {floor:.2f}s | "
              f"reported {len(reported_hits)}H/{len(reported_miss)}M | clock: phantom {len(phantom)}, "
              f"hidden {len(hidden)} (strict-below-floor {len(strict_hidden)}) | {row['verdict']}")
    (ROOT / 'reports/cache-clock-audit-2026-11-20.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
