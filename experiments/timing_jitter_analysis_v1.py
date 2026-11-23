"""Inter-token latency jitter in a timing-speculation run: load, stalls, or speculation?

Uses the per-chunk SSE arrival times already recorded (client-side, temperature 0, prompts interleaved in time).
Per stream, over chunks that carry text (content or reasoning):
  - gap_ms: median gap between chunks; gap_cv: coefficient of variation of the gaps (burstiness);
  - stall_share: share of decode time spent in gaps longer than 4x the stream's own median gap;
  - chars_per_chunk: median characters per chunk (several tokens per chunk = bursts).
Readings:
  - speculation: predictable text (copy, count) arrives in bigger, burstier chunks than story text, consistently
    across repeats;
  - load or scheduling: stalls appear regardless of content and vary between repeats;
  - consistency: speculation is a property of the content, so its pattern repeats across repeats of the same
    prompt; load-driven stalls don't (the spread of stall share across repeats is reported).
Events read in the same network read share a timestamp, so chunks arriving within 2 ms are merged into one
delivery and gaps are measured between deliveries. The run sent one request at a time, so stalls on different
providers can't be compared moment by moment.
"""
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
sys_path_added = None
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT / 'experiments'))
from ci_utils import bootstrap, median as _median  # noqa: E402
PROBES = ('copy', 'count', 'story', 'story2', 'random')
STALL = 4.0
MERGE_MS = 2.0


def chunks(row):
    out = []
    for event in row['events']:
        data = event.get('data')
        if not data or data == '[DONE]':
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        text = ''
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            text += ''.join(delta.get(k) or '' for k in ('content', 'reasoning', 'reasoning_content'))
        if text:
            if out and event['received_ms'] - out[-1][0] <= MERGE_MS:
                out[-1] = (out[-1][0], out[-1][1] + len(text))
            else:
                out.append((event['received_ms'], len(text)))
    return out


def stream_stats(row):
    c = chunks(row)
    if len(c) < 8:
        return None
    times = [t for t, _ in c]
    gaps = [b - a for a, b in zip(times, times[1:])]
    median = statistics.median(gaps)
    if median <= 0:
        median = max(statistics.mean(gaps), 1e-3)
    stalls = [(times[i], g) for i, g in enumerate(gaps) if g > STALL * median]
    span = times[-1] - times[0]
    start = datetime.fromisoformat(row['started_at']).timestamp() * 1000
    return dict(gap_ms=median, gap_cv=statistics.pstdev(gaps) / statistics.mean(gaps) if statistics.mean(gaps) else 0,
                stall_share=sum(g for _, g in stalls) / span if span else 0,
                stalls=len(stalls), chunks=len(c), chars_per_chunk=statistics.median(n for _, n in c),
                tokens_per_delivery=((row.get('usage') or {}).get('completion_tokens') or 0) / len(c),
                stall_times=[start + t for t, _ in stalls], start=start, span=span)


def main():
    args = sys.argv[1:]
    stem = args.pop() if args and args[-1].startswith('reports/') else 'reports/timing-jitter-2026-11-16'
    rows = []
    for run in args:
        path = Path(run) / 'observations.jsonl'
        rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by = defaultdict(lambda: defaultdict(list))
    streams = []
    for row in rows:
        if row['status'] != 'success':
            continue
        s = stream_stats(row)
        if s:
            s['target'] = row['target_id']
            by[row['target_id']][row['probe_id']].append(s)
            streams.append(s)
    table = []
    for target, probes in sorted(by.items()):
        entry = dict(target=target)
        for probe in PROBES:
            values = probes.get(probe) or []
            if not values:
                continue
            for key in ('gap_ms', 'gap_cv', 'stall_share', 'chars_per_chunk', 'tokens_per_delivery'):
                entry[f'{probe}_{key}'] = statistics.median(v[key] for v in values)
            for key in ('tokens_per_delivery', 'stall_share'):
                entry[f'{probe}_{key}_ci'] = bootstrap([v[key] for v in values], _median, seed=len(values))
            entry[f'{probe}_streams_with_stalls'] = sum(1 for v in values if v['stalls'])
            entry[f'{probe}_n'] = len(values)
        predictable = [entry.get(f'{p}_tokens_per_delivery') for p in ('copy', 'count')
                       if entry.get(f'{p}_tokens_per_delivery')]
        if predictable and entry.get('story_tokens_per_delivery'):
            entry['burst_ratio'] = max(predictable) / entry['story_tokens_per_delivery']
        all_values = [v for vs in probes.values() for v in vs]
        entry['stall_share_all'] = statistics.median(v['stall_share'] for v in all_values)
        entry['streams_with_stalls'] = f"{sum(1 for v in all_values if v['stalls'])}/{len(all_values)}"
        spread = [v['stall_share'] for v in all_values]
        entry['stall_share_max'] = max(spread)
        entry['stall_share_p10_p90'] = [sorted(spread)[len(spread) // 10], sorted(spread)[len(spread) * 9 // 10]]
        table.append(entry)
    (ROOT / f'{stem}.json').write_text(json.dumps(dict(targets=table), indent=1))
    figure(table, ROOT / f'{stem}.png')
    for e in table:
        print(f"{e['target']:30s} gap(story) {e.get('story_gap_ms', 0):6.1f} ms  cv {e.get('story_gap_cv', 0):4.2f}  "
              f"tok/delivery copy {e.get('copy_tokens_per_delivery', 0):5.2f} count {e.get('count_tokens_per_delivery', 0):5.2f} "
              f"story {e.get('story_tokens_per_delivery', 0):5.2f} story2 {e.get('story2_tokens_per_delivery', 0):5.2f}  "
              f"burst {e.get('burst_ratio', 0):4.2f}  "
              f"stall share med {e['stall_share_all']:.3f} p10-90 {e['stall_share_p10_p90'][0]:.2f}-{e['stall_share_p10_p90'][1]:.2f}  "
              f"streams w/ stalls {e['streams_with_stalls']}")


def figure(table, path):
    table = [e for e in table if 'story_gap_cv' in e]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.62 * len(table) + 1.8), sharey=True)
    labels = [e['target'] for e in table]
    y = range(len(table))
    colors = {'copy': '#2a78d6', 'count': '#eb6834', 'story': '#1baf7a', 'random': '#8d5bd6'}
    for k, (probe, color) in enumerate(colors.items()):
        offset = [i + (k - 1.5) * 0.17 for i in y]
        for ax, key, scale in ((axes[0], 'tokens_per_delivery', 1), (axes[1], 'stall_share', 100)):
            xs = [e.get(f'{probe}_{key}', float('nan')) * scale for e in table]
            cis = [e.get(f'{probe}_{key}_ci') or (None, None) for e in table]
            lo = [max(0, x - (c[0] * scale)) if c[0] is not None else 0 for x, c in zip(xs, cis)]
            hi = [max(0, (c[1] * scale) - x) if c[1] is not None else 0 for x, c in zip(xs, cis)]
            ax.errorbar(xs, offset, xerr=[lo, hi], fmt='o', color=color, markersize=4.5, elinewidth=1.1, capsize=2,
                        label=probe if ax is axes[0] else None, zorder=3)
    axes[0].set_xscale('log')
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    axes[0].xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30]))
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel('tokens per delivery (log); median over streams, 95% bootstrap CI')
    axes[0].set_title('Burst size by content', loc='left', fontweight='bold')
    axes[1].set_xlabel('% of decode time in stalls (gaps > 4x median); median, 95% CI')
    axes[1].set_title('Stalls by content', loc='left', fontweight='bold')
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, loc='upper center', ncol=4, title='prompt', bbox_to_anchor=(0.5, 1.0))
    for ax in axes:
        ax.grid(axis='x', color='#e5e5e5')
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
