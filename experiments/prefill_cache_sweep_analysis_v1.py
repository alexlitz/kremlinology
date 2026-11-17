"""Analyse prefill_cache_sweep_v1: prefill speed, cache effect by size, disaggregation signature (witnessed timing).

Per call (joined to session evidence by generation id): TTFT = first token-bearing SSE event minus end of request;
gap1 = second token-bearing event minus first; later = median gap between later token-bearing events (events within
2 ms merged). Per endpoint:
  - cold: least-squares TTFT = a + b * prompt_tokens (b -> prefill tokens/s, a -> fixed overhead), on uncached calls;
  - warm: median warm/cold TTFT ratio and cached share per size;
  - gap1 vs prompt size: slope of gap1 (ms per 1k prompt tokens) on cold calls, against the later-gap median.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_routing_honesty_analysis_v1 import generation_of  # noqa: E402
from ci_utils import bootstrap, median  # noqa: E402
import kv_cache_economics_v1 as base  # noqa: E402
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False
ROOT = Path(__file__).resolve().parents[1]
ROOT_REPORT = ROOT / 'reports/prefill-cache-sweep-2026-11-17.json'
DISAGG = {'fireworks': 'disaggregated (own docs)', 'baseten': 'disaggregated (own docs)',
          'moonshotai': 'disaggregated (Mooncake)', 'together': 'third-party claim only'}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8d5bd6', '#d6455b', '#4a9fb5', '#7a7a7a', '#b08a3c']


def figure(by):
    """Per model: cold and warm TTFT vs prompt size (median, 95% bootstrap CI) and first-gap vs size."""
    models = sorted({m for m, _ in by})
    fig, axes = plt.subplots(len(models), 3, figsize=(19, 5.0 * len(models)), squeeze=False)
    fig.patch.set_facecolor(base.SURFACE)
    for row, model in zip(axes, models):
        providers = sorted(p for m, p in by if m == model)
        for k, provider in enumerate(providers):
            rows = by[(model, provider)]
            name = provider.split('/')[0] + (' †' if provider == 'together' else ' *' if provider in DISAGG else '')
            for ax, kind, key in ((row[0], 'cold', 'ttft'), (row[1], 'warm', 'ttft'), (row[2], 'cold', 'gap1')):
                xs, ys, lo, hi = [], [], [], []
                for size in sorted({r['size'] for r in rows}):
                    values = [r[key] for r in rows if r['size'] == size and r['kind'] == kind and r[key] is not None
                              and (kind == 'warm' or r['cached'] < 0.1 * r['prompt'])]
                    if len(values) < 2:
                        continue
                    m = median(values)
                    l, h = bootstrap(values, median)
                    scale = 1000 if key == 'gap1' else 1
                    xs.append(size * (1 + 0.03 * (k - len(providers) / 2)))
                    ys.append(m * scale)
                    lo.append((m - l) * scale)
                    hi.append((h - m) * scale)
                ax.errorbar(xs, ys, yerr=[lo, hi], color=COLORS[k % len(COLORS)], marker='o', markersize=4,
                            linewidth=1.6, capsize=2, label=name)
        titles = ('cold TTFT (prefill)', 'warm TTFT (repeat 3 s later)', 'first-to-second-token gap (cold)')
        ylabels = ('seconds (median, 95% CI)', 'seconds (median, 95% CI)', 'milliseconds (median, 95% CI)')
        for ax, title, ylabel in zip(row, titles, ylabels):
            base.style(ax, f"{model.split('/')[1]} · {title}", xlabel='prompt tokens', ylabel=ylabel)
            ax.set_xscale('log')
            ax.set_xticks([1000, 2000, 4000, 8000, 16000, 32000], ['1k', '2k', '4k', '8k', '16k', '32k'])
            ax.set_ylim(bottom=0)
        row[0].legend(frameon=False, fontsize=8, labelcolor=base.INK, ncol=2)
    fig.text(0.01, 0.005, '* says it disaggregates prefill and decode (Fireworks, Baseten: own docs; Moonshot: Mooncake '
             'paper). † third-party claims only (Together). Witnessed timing (relay record timestamps), OpenRouter pinned '
             'endpoints, 2026-11-17.', fontsize=8, color=base.INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(ROOT / 'reports/prefill-cache-sweep-2026-11-17.png', dpi=140, facecolor=base.SURFACE)



def token_times(record):
    receipt = record.get('witness_receipt') or {}
    start = receipt.get('request_complete_ms')
    times = []
    for t_ms, data in (record.get('timings') or {}).get('witness_sse_events') or []:
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')):
                if not times or t_ms - times[-1] > 2:
                    times.append(t_ms)
                break
    if start is None or not times:
        return None
    return [(t - start) / 1000 for t in times]


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if not sxx:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    out = runs[0]
    times, all_rows = {}, []
    for k, run in enumerate(runs):
        root = run / 'session'
        for line in (root / 'requests.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('status') == 200:
                gen = generation_of(record, root)
                if gen:
                    times[gen] = token_times(record)
        for r in json.loads((run / 'log.json').read_text())['rows']:
            size, trial, kind = r['tag'].split(':')
            all_rows.append(dict(r, tag=f'{size}:{k * 100 + int(trial)}:{kind}'))
    by = defaultdict(list)
    for r in all_rows:
        if r['status'] != 200 or not r.get('prompt_tokens'):
            continue
        t = times.get(r['generation'])
        if not t:
            continue
        size, trial, kind = r['tag'].split(':')
        by[(r['model'], r['provider'])].append(dict(size=int(size), trial=int(trial), kind=kind, ttft=t[0],
                                                     gap1=t[1] - t[0] if len(t) > 1 else None,
                                                     later=statistics.median(b - a for a, b in zip(t[1:], t[2:]))
                                                     if len(t) > 3 else None,
                                                     prompt=r['prompt_tokens'], cached=r['cached_tokens'], cost=r['cost']))
    report = []
    for (model, provider), rows in sorted(by.items()):
        cold = [r for r in rows if r['kind'] == 'cold' and r['cached'] < 0.1 * r['prompt']]
        a, b = fit([r['prompt'] for r in cold], [r['ttft'] for r in cold])
        g0, g1 = fit([r['prompt'] for r in cold if r['gap1'] is not None], [r['gap1'] for r in cold if r['gap1'] is not None])
        sizes = sorted({r['size'] for r in rows})
        per_size = {}
        for size in sizes:
            c = [r for r in rows if r['size'] == size and r['kind'] == 'cold']
            w = [r for r in rows if r['size'] == size and r['kind'] == 'warm']
            pairs = [(x, y) for x in c for y in w if x['trial'] == y['trial']]
            per_size[size] = dict(
                cold_ttft=round(statistics.median(x['ttft'] for x in c), 3) if c else None,
                warm_ttft=round(statistics.median(y['ttft'] for y in w), 3) if w else None,
                warm_over_cold=round(statistics.median(y['ttft'] / x['ttft'] for x, y in pairs), 2) if pairs else None,
                cached_share=round(statistics.median(y['cached'] / y['prompt'] for y in w), 2) if w else None,
                cold_gap1_ms=round(statistics.median(x['gap1'] for x in c if x['gap1'] is not None) * 1000)
                if any(x['gap1'] is not None for x in c) else None,
                n=len(pairs))
        later = [r['later'] for r in rows if r['later'] is not None]
        report.append(dict(model=model, provider=provider, calls=len(rows),
                           prefill_tokens_per_s=round(1 / b) if b and b > 0 else None,
                           fixed_overhead_s=round(a, 3) if a is not None else None,
                           gap1_ms_per_1k_tokens=round(g1 * 1000 * 1000, 1) if g1 is not None else None,
                           later_gap_ms=round(statistics.median(later) * 1000, 1) if later else None,
                           sizes=per_size))
    Path(ROOT_REPORT).write_text(json.dumps(report, indent=1))
    figure(by)
    for e in report:
        print(f"\n{e['model'].split('/')[1]} @ {e['provider']}  calls {e['calls']}  prefill ~{e['prefill_tokens_per_s']} tok/s  "
              f"overhead {e['fixed_overhead_s']} s  gap1 slope {e['gap1_ms_per_1k_tokens']} ms/1k tok  later gap {e['later_gap_ms']} ms")
        for size, v in e['sizes'].items():
            print(f"   {size:6d}: cold {v['cold_ttft']}  warm {v['warm_ttft']}  w/c {v['warm_over_cold']}  cached {v['cached_share']}  "
                  f"gap1 {v['cold_gap1_ms']} ms  (n={v['n']})")


if __name__ == '__main__':
    main()
