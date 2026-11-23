"""Decode speed, speculation signature and hardware bounds from a timing-speculation run.

Per request: decode rate = (completion_tokens - 1) / (last - first text-event time), from
client-side SSE arrival times of content and reasoning deltas (network and buffering add noise; chunks can hold several
tokens). Speculation signature: with weights read the same way whatever the content,
decode speed should not depend on predictability; a consistent copy/count speed-up over
story text (bootstrap CI over repeats, requests interleaved in time) points to speculative
or lookahead decoding.

Hardware bound: a decode step without speculation reads at least the active weights, so
rate x active-weight bytes is a lower bound on memory bandwidth behind one stream (KV reads
add more). Uses the story probe (least speculation-friendly). Compared with H100 (3.35 TB/s),
8xH100 (26.8), 8xB200 (64). Exceeding a single node's bandwidth implies speculation, SRAM
hardware (Groq, Cerebras) or a sub-assumed precision. Active-parameter counts are published
approximations.
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402
from ci_utils import bootstrap as _boot  # noqa: E402

ACTIVE = {'llama70b': 70.6e9, 'gptoss120b': 5.1e9, 'gemma4-31b': 30.7e9, 'deepseekv32': 37e9}
NODES = [(3.35e12, '1x H100'), (26.8e12, '8x H100'), (64e12, '8x B200')]
PROBES = ('copy', 'count', 'story', 'story2', 'random')  # story2 (v3+): negative control; random (v4): unpredictable
COLORS = {'copy': '#2a78d6', 'count': '#eb6834', 'story': '#1baf7a', 'story2': '#eda100', 'random': '#8d5bd6'}


def rate(row):
    times = []
    for event in row['events']:
        data = event.get('data')
        if not data or data == '[DONE]':
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')):
                times.append(event['received_ms'])  # reasoning tokens count in completion_tokens too
    tokens = (row.get('usage') or {}).get('completion_tokens')
    if not tokens or len(times) < 3 or times[-1] <= times[0]:
        return None
    return dict(rate=(tokens - 1) / ((times[-1] - times[0]) / 1000), tokens_per_chunk=tokens / len(times),
                tokens=tokens, ttft_ms=row.get('ttft_ms'))


def bootstrap_ratio(a, b, seed=0):
    generator = random.Random(seed)
    ratios = []
    for _ in range(2000):
        x = statistics.median(generator.choice(a) for _ in a)
        y = statistics.median(generator.choice(b) for _ in b)
        ratios.append(x / y)
    ratios.sort()
    return ratios[50], ratios[1949]


def main():
    args = sys.argv[1:]
    stem_arg = args.pop() if args and args[-1].startswith('reports/') else None
    rows = []
    for run in args:
        rows += [json.loads(line) for line in (Path(run) / 'observations.jsonl').read_text().splitlines() if line.strip()]
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['status'] != 'success':
            continue
        r = rate(row)
        if r:
            data[row['target_id']][row['probe_id']].append(r)
    results = []
    for target, probes in sorted(data.items()):
        model = target.split('@')[0]
        entry = dict(target=target, model=model)
        for probe in PROBES:
            values = probes.get(probe, [])
            if values:
                entry[f'{probe}_median_tps'] = statistics.median(v['rate'] for v in values)
                entry[f'{probe}_tps_ci'] = list(_boot([v['rate'] for v in values], statistics.median, seed=len(values)))
                entry[f'{probe}_tokens_per_chunk'] = statistics.median(v['tokens_per_chunk'] for v in values)
                entry[f'{probe}_n'] = len(values)
        for probe in ('copy', 'count', 'story2'):
            a = [v['rate'] for v in probes.get(probe, [])]
            b = [v['rate'] for v in probes.get('story', [])]
            if len(a) >= 3 and len(b) >= 3:
                low, high = bootstrap_ratio(a, b)
                entry[f'{probe}_over_story'] = statistics.median(a) / statistics.median(b)
                entry[f'{probe}_over_story_ci'] = [low, high]
        # Bound from the least speculation-friendly prompt available: random text (v4) if measured, else story.
        bound_probe = 'random' if entry.get('random_n', 0) >= 5 else 'story'
        if f'{bound_probe}_median_tps' in entry:
            entry['bound_probe'] = bound_probe
            for name, size in (('fp4', 0.5), ('fp8', 1), ('bf16', 2)):
                entry[f'min_bandwidth_tbs_{name}'] = entry[f'{bound_probe}_median_tps'] * ACTIVE[model] * size / 1e12
        results.append(entry)
    stem = stem_arg or 'reports/timing-speculation-2026-11-13'
    base.write_csv(ROOT / f'{stem}.csv', results)
    figure(results, ROOT / f'{stem}.png')
    for e in results:
        ratio = lambda p: f"{e[p + '_over_story']:.2f} [{e[p + '_over_story_ci'][0]:.2f},{e[p + '_over_story_ci'][1]:.2f}]" \
            if p + '_over_story' in e else 'n/a'  # noqa: E731
        print(f"{e['target']:34s} story {e.get('story_median_tps', 0):6.0f} tps  copy/story {ratio('copy'):20s} "
              f"count/story {ratio('count'):20s} story2/story {ratio('story2'):20s} tok/chunk story {e.get('story_tokens_per_chunk', 0):.1f} "
              f"copy {e.get('copy_tokens_per_chunk', 0):.1f}  min BW fp8 {e.get('min_bandwidth_tbs_fp8', 0):.1f} TB/s")


def figure(results, path):
    repeats = max(e.get(f'{p}_n', 0) for e in results for p in PROBES)
    results = sorted(results, key=lambda e: (e['model'], e.get('story_median_tps', 0)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 0.55 * len(results) + 2.4), sharey=True)
    base.style(left, 'Decode speed by prompt (median, 95% CI)',
               xlabel='tokens per second (log scale); temperature 0, random text at T=1')
    seen = set()
    for i, e in enumerate(results):
        for k, probe in enumerate(PROBES):
            if f'{probe}_median_tps' in e:
                m = e[f'{probe}_median_tps']
                lo, hi = e.get(f'{probe}_tps_ci') or (None, None)
                left.errorbar(m, i + (k - 2) * 0.15, xerr=[[m - lo if lo else 0], [hi - m if hi else 0]], fmt='o',
                              color=COLORS[probe], markersize=5, elinewidth=1, capsize=2,
                              label=probe if probe not in seen else None)
                seen.add(probe)
    left.set_xscale('log')
    left.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    left.set_yticks(range(len(results)), [e['target'] for e in results], fontsize=8.5, color=base.INK)
    left.legend(frameon=False, fontsize=8.5, labelcolor=base.INK, loc='lower right',
                title='prompt', title_fontsize=8)
    left.grid(axis='y', visible=False)
    base.style(right, 'Minimum memory bandwidth (from random-text speed)',
               xlabel='TB/s if each step reads all active weights once (log)\n'
               'dot: fp8 weights, whiskers 95% CI; bar: fp4 to bf16. Groq, Cerebras: SRAM')
    for i, e in enumerate(results):
        if 'min_bandwidth_tbs_fp8' in e:
            lo, mid, hi = e['min_bandwidth_tbs_fp4'], e['min_bandwidth_tbs_fp8'], e['min_bandwidth_tbs_bf16']
            right.plot([lo, hi], [i, i], color='#2a78d6', linewidth=2, alpha=0.4)
            bp = e.get('bound_probe', 'story')
            ci = e.get(f'{bp}_tps_ci') or (None, None)
            scale = mid / e[f'{bp}_median_tps']
            right.errorbar(mid, i, xerr=[[mid - ci[0] * scale if ci[0] else 0], [ci[1] * scale - mid if ci[1] else 0]],
                           fmt='o', color='#2a78d6', markersize=6, elinewidth=1.2, capsize=2)
    for bandwidth, label in NODES:
        right.axvline(bandwidth / 1e12, color=base.INK2, linewidth=0.9, linestyle=(0, (3, 3)))
        right.text(bandwidth / 1e12 * 1.04, -0.9, label, rotation=0, ha='left', fontsize=7.5, color=base.INK2)
    right.set_xscale('log')
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    right.set_ylim(len(results) - 0.5, -1.3)
    right.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
