"""Analyse cache_ttl_probe_v1 runs: did a unique prompt's cache survive each delay?

A trial is a hit when request B reports cached prompt tokens >= half its prompt tokens
(the shared chat-template header alone is far below that). Request A's cached tokens
show the template-only baseline. Outputs a per-endpoint x delay hit table and figure.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402

OUT = ROOT / 'reports/kv-cache-economics'


from matplotlib.colors import LinearSegmentedColormap as _LSC
_cmap = _LSC.from_list('hits', ['#f0efec', '#9ec5f4', '#2a78d6', '#103f7a'])
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT_DIR / 'experiments'))
from ci_utils import wilson  # noqa: E402


def cached(usage):
    details = (usage or {}).get('prompt_tokens_details') or {}
    return details.get('cached_tokens')


def analyse(runs):
    trials = {}
    for k, run in enumerate(runs if isinstance(runs, (list, tuple)) else [runs]):
        for line in (Path(run) / 'requests.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trials.setdefault((k, r['trial']), {})[r['role']] = r
    table = {}
    for pair in trials.values():
        if 'A' not in pair or 'B' not in pair:
            continue
        a, b = pair['A'], pair['B']
        key = (b['model'].split('/')[-1], b['provider'], b['delay'])
        entry = table.setdefault(key, dict(model=key[0], provider=key[1], delay_s=key[2], trials=0, hits=0,
                                           reported=0, a_cached=[], b_cached=[], gap_s=[], errors=0))
        entry['trials'] += 1
        if a['status'] != 200 or b['status'] != 200:
            entry['errors'] += 1
            continue
        prompt = (b['usage'] or {}).get('prompt_tokens') or 0
        cb = cached(b['usage'])
        entry['a_cached'].append(cached(a['usage']))
        entry['b_cached'].append(cb)
        entry['gap_s'].append(b['sent'] - a['sent'])
        if cb is not None:
            entry['reported'] += 1
            entry['hits'] += cb >= 0.5 * prompt
    return sorted(table.values(), key=lambda e: (e['model'], e['provider'], e['delay_s']))


def figure(entries, path):
    endpoints = list(dict.fromkeys((e['model'], e['provider']) for e in entries))
    delays = sorted({e['delay_s'] for e in entries})
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(endpoints) + 2.2))
    base.style(ax, 'Did a unique prompt still hit the provider cache after each delay?',
               xlabel='delay between identical requests')
    lookup = {(e['model'], e['provider'], e['delay_s']): e for e in entries}
    for i, endpoint in enumerate(endpoints):
        for j, delay in enumerate(delays):
            e = lookup.get(endpoint + (delay,))
            if not e:
                continue
            if e['reported'] == 0:
                text, color = 'n/r', '#f0efec'  # provider did not report cached tokens
            else:
                share = e['hits'] / e['reported']
                color = _cmap(share)
                lo, hi = wilson(e['hits'], e['reported'])
                text = f"{e['hits']}/{e['reported']}"
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=color))
            ink = '#ffffff' if e['reported'] and e['hits'] / e['reported'] > 0.6 else base.INK
            ax.text(j, i - 0.1, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=ink)
            if e['reported']:
                ax.text(j, i + 0.2, f'{lo:.0%}–{hi:.0%}', ha='center', va='center', fontsize=6.5, color=ink)
    labels = {2: '~2 s', 60: '1 min', 300: '5 min', 900: '15 min', 1800: '30 min', 3600: '60 min'}
    ax.set_xticks(range(len(delays)), [labels.get(d, f'{d}s') for d in delays])
    ax.set_yticks(range(len(endpoints)), [f'{m} @ {p}' for m, p in endpoints], fontsize=8.5, color=base.INK)
    ax.set_xlim(-0.6, len(delays) - 0.4)
    ax.set_ylim(len(endpoints) - 0.5, -0.5)
    ax.grid(False)
    ax.text(0, -0.08, 'cells: hits / trials with cached-token reports (hit = at least half the prompt cached), and the '
            '95% Wilson interval on the hit rate;\nn/r = no cached-token field reported. '
            f'{len(sys.argv) - 1} runs pooled (2026-11-11 to 11-17).', transform=ax.transAxes, fontsize=7.5, color=base.INK2, va='top')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    entries = analyse(runs)
    for e in entries:
        e['median_a_cached'] = sorted(x for x in e['a_cached'] if x is not None)[len(e['a_cached']) // 2] \
            if any(x is not None for x in e['a_cached']) else None
    base.write_csv(OUT / 'fig16_cache_retention_probe.csv',
                   [{k: v for k, v in e.items() if k not in ('a_cached', 'b_cached', 'gap_s')} for e in entries])
    figure(entries, OUT / 'fig16_cache_retention_probe.png')
    figure(entries, ROOT_DIR / 'reports/cache-retention-2026-11-17.png')
    for e in entries:
        print(f"{e['model']:24s} {e['provider']:18s} {e['delay_s']:5d}s  hits {e['hits']}/{e['reported']} "
              f"(trials {e['trials']}, errors {e['errors']}) B cached {e['b_cached']} A cached {e['a_cached']}")


if __name__ == '__main__':
    main()
