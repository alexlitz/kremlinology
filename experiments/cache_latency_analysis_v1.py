"""Analyse cache_latency_grid_age_v1 runs (pooled over run directories): witnessed TTFT of the follow-up request.

grid: per endpoint, TTFT vs new tokens U, one curve per cached-prefix size C (median, 95% bootstrap CI), and the
      cached tokens actually reported.
age:  per endpoint, TTFT and reported cached share vs time since the cache was written (median, 95% bootstrap CI;
      hit rate with Wilson CI).
Self-load: a request sent while >= LOADED of our own requests to the same endpoint were still before their first token
is excluded from the age panel and its cold reference (age-v2 spread 126 Moonshot writes over 4 minutes and queued
behind itself: 10 s TTFTs; age-v3 re-ran Moonshot spread over 30 minutes).
Writes reports/cache-latency-grid-<date>.png, reports/cache-latency-age-<date>.png and a JSON summary.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import os as _os
DATA_ROOT = _os.environ.get('KREMLINOLOGY_DATA', 'data')  # where the release archives are extracted

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402
from cache_routing_honesty_analysis_v1 import generation_of, witnessed_ttft  # noqa: E402
from ci_utils import bootstrap, median, wilson  # noqa: E402

LOADED = 3
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8d5bd6']
NAMES = {'parasail/fp8': 'Parasail', 'nebius/fp4': 'Nebius', 'moonshotai': 'Moonshot', 'deepinfra/fp8': 'DeepInfra'}


def load(runs):
    rows = []
    for run in runs:
        log = json.loads((run / 'log.json').read_text())
        ttft, sent = {}, {}
        root = run / 'session'
        for line in (root / 'requests.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('status') == 200:
                gen = generation_of(record, root)
                if gen:
                    ttft[gen] = witnessed_ttft(record)
                    sent[gen] = record.get('sent_at')
        mine = [dict(r, ttft=ttft[r['generation']], sent=sent[r['generation']], run=str(run)) for r in log['rows']
                if r['status'] == 200 and ttft.get(r['generation']) is not None]
        for r in mine:  # own requests to the same endpoint still waiting for their first token when this one was sent
            r['load'] = sum(1 for o in mine if o is not r and o['provider'] == r['provider'] and r['sent'] is not None
                            and o['sent'] is not None and o['sent'] <= r['sent'] < o['sent'] + o['ttft'])
        rows += mine
    return rows


def label(model, provider):
    return f"{model.split('/')[1]} · {NAMES.get(provider, provider)}"


def summarise(values):
    lo, hi = bootstrap(values, median)
    return dict(median=median(values), lo=lo, hi=hi, n=len(values))


def main():
    runs = [Path(p) for p in sys.argv[1:]] or [Path(DATA_ROOT, 'kremlinology-cache-latency-v1')]
    rows = load(runs)
    grid, age = defaultdict(list), defaultdict(list)
    writes = defaultdict(list)
    for r in rows:
        parts = r['tag'].split(':')
        if parts[0] == 'age' and r['load'] >= LOADED:
            continue
        if parts[-1] == 'write':
            if parts[0] == 'age':
                writes[(r['model'], r['provider'])].append(r['ttft'])
            continue
        if parts[0] == 'grid':
            grid[(r['model'], r['provider'], int(parts[1]), int(parts[2]))].append(r)
        elif parts[0] == 'age':
            age[(r['model'], r['provider'], int(parts[1]))].append(r)
    summary = dict(grid=[], age=[])
    sweep_cold = {}
    sweep_path = ROOT / 'reports/prefill-cache-sweep-2026-11-17.json'
    if sweep_path.exists():
        for e in json.loads(sweep_path.read_text()):
            sweep_cold[(e['model'], e['provider'])] = {int(k): v['cold_ttft'] for k, v in e['sizes'].items()
                                                       if v.get('cold_ttft') is not None}
    endpoints = sorted({(m, p) for m, p, _, _ in grid})
    fig, axes = plt.subplots(1, len(endpoints), figsize=(5.2 * len(endpoints), 4.4), squeeze=False)
    fig.patch.set_facecolor(base.SURFACE)
    for ax, (model, provider) in zip(axes[0], endpoints):
        cached_sizes = sorted({c for m, p, c, _ in grid if (m, p) == (model, provider)})
        for k, c in enumerate(cached_sizes):
            xs, ys, lo, hi = [], [], [], []
            for u in sorted({u for m, p, cc, u in grid if (m, p, cc) == (model, provider, c)}):
                cell = grid[(model, provider, c, u)]
                s = summarise([r['ttft'] for r in cell])
                cached = median([r['cached_tokens'] for r in cell])
                summary['grid'].append(dict(model=model, provider=provider, cached=c, new=u, cached_reported=cached, **s))
                xs.append(c + u)
                ys.append(s['median'])
                lo.append(max(0.0, s['median'] - (s['lo'] if s['lo'] is not None else s['median'])))
                hi.append(max(0.0, (s['hi'] if s['hi'] is not None else s['median']) - s['median']))
            ax.errorbar(xs, ys, yerr=[lo, hi], color=COLORS[k], marker='o', markersize=5, linewidth=2, capsize=3,
                        label=f'{c // 1000}k of it cached' if c else 'nothing cached (short prompts)')
        cold = sweep_cold.get((model, provider))
        if cold:
            sizes = sorted(cold)
            ax.plot(sizes, [cold[x] for x in sizes], color=base.INK2, linestyle='--', linewidth=1.4,
                    label='all uncached (prefill sweep)')
        ax.set_xscale('log')
        ax.set_xticks([500, 1000, 4000, 8000, 16000, 32000], ['0.5k', '1k', '4k', '8k', '16k', '32k'])
        base.style(ax, label(model, provider), xlabel='total prompt tokens (cached + new)',
                   ylabel='time to first token, s (median, 95% CI)')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=base.INK)
    trials = sorted({len(v) for v in grid.values()})
    fig.suptitle(f'Time to first token by total prompt size, with part of it already cached (witnessed timing; {trials[0]}–{trials[-1]} trials per point)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=base.INK)
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/cache-latency-grid-2026-11-17.png', dpi=160, facecolor=base.SURFACE)

    endpoints = sorted({(m, p) for m, p, _ in age})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7.4), sharex=True)
    fig.patch.set_facecolor(base.SURFACE)
    for k, (model, provider) in enumerate(endpoints):
        delays = sorted({d for m, p, d in age if (m, p) == (model, provider)})
        xs, ys, lo, hi, share, slo, shi = [], [], [], [], [], [], []
        for d in delays:
            cell = age[(model, provider, d)]
            s = summarise([r['ttft'] for r in cell])
            hits = sum(1 for r in cell if r['cached_tokens'] > 0.5 * (r['prompt_tokens'] or 1) * 0.5)
            wl, wh = wilson(hits, len(cell))
            summary['age'].append(dict(model=model, provider=provider, delay_s=d, hits=hits, **s))
            xs.append(d)
            ys.append(s['median'])
            lo.append(max(0.0, s['median'] - (s['lo'] if s['lo'] is not None else s['median'])))
            hi.append(max(0.0, (s['hi'] if s['hi'] is not None else s['median']) - s['median']))
            share.append(hits / len(cell))
            slo.append(max(0.0, hits / len(cell) - wl))
            shi.append(max(0.0, wh - hits / len(cell)))
        top.errorbar(xs, ys, yerr=[lo, hi], color=COLORS[k], marker='o', linewidth=2, capsize=3, label=label(model, provider))
        cold = writes.get((model, provider))
        if cold:
            top.axhline(median(cold), color=COLORS[k], linestyle='--', linewidth=1.2, alpha=0.8)
            top.text(1500, median(cold) + (0.12 if k == 0 else 0.0), f'no cache ({len(cold)} cold sends)', color=COLORS[k],
                     fontsize=7.5, va='bottom' if k != 2 else 'top', ha='right')
        unreported = [d for d in delays if sum(r['cached_tokens'] == 0 for r in age[(model, provider, d)]) * 2 > len(age[(model, provider, d)])
                      and median([r['ttft'] for r in age[(model, provider, d)]]) < 0.7 * median(cold or [1e9])]
        for d in unreported[:1]:
            top.annotate('from 10 min: 0 cached tokens reported, billed in full,\nyet still about 1.8x as fast as no cache', (d, median([r['ttft'] for r in age[(model, provider, d)]])),
                         xytext=(-10, 22), textcoords='offset points', ha='right', fontsize=7.5, color=COLORS[k],
                         arrowprops=dict(arrowstyle='-', color=COLORS[k], lw=0.8))
        bottom.errorbar([x * (1 + 0.04 * (k - 1)) for x in xs], share, yerr=[slo, shi], color=COLORS[k], marker='o',
                        linewidth=2, capsize=3)
    top.set_xscale('log')
    base.style(top, 'Follow-up time to first token vs time since the cache was written (dashed: no cache)', xlabel='',
               ylabel='TTFT, s (median, 95% CI)')
    top.set_ylim(bottom=0)
    top.set_xlim(3, 4000)
    top.legend(frameon=False, fontsize=8.5, labelcolor=base.INK, loc='upper left')
    base.style(bottom, 'Share of follow-ups that hit the cache (95% Wilson CI)', xlabel='seconds since the cache was written',
               ylabel='hit rate')
    bottom.set_xscale('log')
    bottom.set_xticks([5, 30, 60, 120, 300, 600, 1200], ['5 s', '30 s', '1 min', '2 min', '5 min', '10 min', '20 min'])
    bottom.set_ylim(-0.05, 1.05)
    trials = sorted({len(v) for v in age.values()})
    fig.text(0.01, 0.005, f'{trials[0]}–{trials[-1]} trials per point (8k cached on Kimi, 16k on the others; 500 new tokens). '
             f'Hit = reported cached tokens.\nRequests sent while {LOADED}+ of our own were queued at the endpoint are '
             'excluded. Witnessed timing, 2026-11-17.', fontsize=7.5, color=base.INK2)
    bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ROOT / 'reports/cache-latency-age-2026-11-17.png', dpi=160, facecolor=base.SURFACE)
    (ROOT / 'reports/cache-latency-2026-11-17.json').write_text(json.dumps(summary, indent=1))
    for g in summary['grid']:
        print(f"grid {label(g['model'], g['provider']):28s} C={g['cached']:6d} U={g['new']:6d} ttft {g['median']:.3f} "
              f"[{g['lo'] or 0:.3f},{g['hi'] or 0:.3f}] n={g['n']} cached_reported={g['cached_reported']}")
    for a in summary['age']:
        print(f"age  {label(a['model'], a['provider']):28s} d={a['delay_s']:5d}s ttft {a['median']:.3f} "
              f"[{a['lo'] or 0:.3f},{a['hi'] or 0:.3f}] hits {a['hits']}/{a['n']}")


if __name__ == '__main__':
    main()
