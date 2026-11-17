"""Figure for the Llama-3.3-70B dataset TV campaign (no API calls).

Left: mean first-token TV vs CoreWeave fp16 per provider and dataset (per-prompt interval
midpoints at a 0.01-nat report-error allowance, normal 95% CI over prompts), with
CoreWeave's own repeat-to-repeat TV as the noise floor. Right: mean TV lower bound by
CoreWeave top-2 logprob margin tercile (near-tie -> confident).
"""
import json
import math
import statistics
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from kremlinology.bench.analysis import first_law, partial_tv_bounds  # noqa: E402
import kv_cache_economics_v1 as base  # noqa: E402
from ci_utils import bootstrap  # noqa: E402

RUN = Path(DATA_ROOT, 'kremlinology-llama70b-dataset-tv-v2')
OUT = ROOT / 'reports/llama70b-dataset-tv-2026-11-10.png'
COLORS = {'CoreWeave vs itself': '#52514e', 'akashml-fp8': '#2a78d6', 'parasail-fp8': '#eb6834', 'sambanova': '#1baf7a'}
LABELS = {'akashml-fp8': 'AkashML (fp8)', 'parasail-fp8': 'Parasail (fp8)', 'sambanova': 'SambaNova',
          'CoreWeave vs itself': 'CoreWeave vs itself'}


def main():
    rows = [json.loads(line) for line in (RUN / 'observations.jsonl').read_text().splitlines()]
    laws = defaultdict(dict)
    for r in rows:
        laws[r['probe_id']][(r['target_id'], r['repeat'])] = first_law(r)
    mids, tercile = defaultdict(list), defaultdict(list)
    for probe, cell in laws.items():
        dataset = probe.split('-')[0]
        ref0, ref1 = cell[('coreweave-fp16', 0)], cell[('coreweave-fp16', 1)]
        b = partial_tv_bounds(ref0, ref1, 0.01)
        mids[('CoreWeave vs itself', dataset)].append((b['lower'] + b['upper']) / 2)
        top = sorted(ref0.values(), reverse=True)
        margin = top[0] - top[1] if len(top) > 1 else math.inf
        for target in ('akashml-fp8', 'parasail-fp8', 'sambanova'):
            pairs = [partial_tv_bounds(cell[(target, rep)], cell[('coreweave-fp16', rep)], 0.01) for rep in (0, 1)]
            mids[(target, dataset)].append(sum((p['lower'] + p['upper']) / 2 for p in pairs) / 2)
            tercile[target].append((margin, sum(p['lower'] for p in pairs) / 2))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    base.style(left, 'Average first-token TV vs CoreWeave fp16 (95% CI)', xlabel='mean total variation distance')
    names = list(COLORS)
    for i, name in enumerate(names):
        for j, (dataset, marker) in enumerate((('code', 'o'), ('wiki', 's'))):
            values = mids[(name, dataset)]
            mean = statistics.mean(values)
            half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
            y = i + (j - 0.5) * 0.3
            left.plot([mean - half, mean + half], [y, y], color=COLORS[name], linewidth=2)
            left.plot(mean, y, marker, color=COLORS[name], markersize=7, markeredgecolor=base.SURFACE,
                      label=f'{dataset} prompts' if i == 0 else None)
            left.text(mean + half + 0.002, y, f'{mean:.1%}', va='center', fontsize=8, color=base.INK2)
    left.set_yticks(range(len(names)), [LABELS[n] for n in names], color=base.INK, fontsize=9)
    left.set_xlim(0, 0.11)
    left.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    left.grid(axis='y', visible=False)
    left.invert_yaxis()
    handles = [plt.Line2D([], [], color=base.INK2, marker=m, linestyle='') for m in ('o', 's')]
    left.legend(handles, ['CodeAlpaca prompts (200)', 'wikitext prompts (200)'], frameon=False, fontsize=8.5,
                loc='upper right', labelcolor=base.INK)
    base.style(right, "Difference by CoreWeave's top-2 margin",
               xlabel='CoreWeave top-2 logprob margin tercile', ylabel='mean TV lower bound (95% bootstrap CI)')
    groups = ['near-tie', 'middle', 'confident']
    for k, target in enumerate(('akashml-fp8', 'parasail-fp8', 'sambanova')):
        ordered = sorted(tercile[target])
        third = len(ordered) // 3
        parts = [ordered[:third], ordered[third:2 * third], ordered[2 * third:]]
        values = [statistics.mean(v for _, v in part) for part in parts]
        cis = [bootstrap([v for _, v in part], statistics.mean, seed=k * 10 + i) for i, part in enumerate(parts)]
        xs = [g + (k - 1) * 0.25 for g in range(3)]
        right.bar(xs, values, width=0.23, color=COLORS[target], edgecolor=base.SURFACE, linewidth=2,
                  label=LABELS[target])
        right.errorbar(xs, values, yerr=[[v - lo for v, (lo, _) in zip(values, cis)],
                                         [hi - v for v, (_, hi) in zip(values, cis)]],
                       fmt='none', ecolor=base.INK, elinewidth=1, capsize=3)
    right.set_xticks(range(3), groups, color=base.INK)
    right.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    right.legend(frameon=False, fontsize=8.5, labelcolor=base.INK)
    right.grid(axis='x', visible=False)
    fig.suptitle('Llama 3.3-70B on OpenRouter: 400 random prompts, first token, top-20 logprobs (2026-11-10)',
                 x=0.01, ha='left', color=base.INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == '__main__':
    main()
