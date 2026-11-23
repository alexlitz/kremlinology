"""Figure: exact top-5 logprob identity between Qwen3.5-397B providers (reseller fingerprint).

Cell (a, b), a != b: share of same prompt+repeat pairs whose top-5 logprob sets are equal within
0.002 nats. Diagonal: the provider's own repeat-0 vs repeat-1 match rate (its nondeterminism floor).
Two routes on one backend match each other about as often as each matches itself.
Run: python experiments/reseller_identity_figure_v1.py RUN_DIR
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from kremlinology.bench.analysis import first_law  # noqa: E402
import kv_cache_economics_v1 as base  # noqa: E402
from ci_utils import wilson  # noqa: E402

TARGETS = [('parasail-fp8', 'Parasail'), ('venice', 'Venice'), ('alibaba', 'Alibaba'), ('novita', 'Novita'),
           ('streamlake', 'StreamLake')]


def same(a, b, tol=2e-3):
    return bool(a and b) and set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


def main():
    run = Path(sys.argv[1])
    laws = defaultdict(dict)
    for line in (run / 'observations.jsonl').read_text().splitlines():
        o = json.loads(line)
        if o['status'] == 'success':
            law = first_law(o)
            if law:
                laws[(o['probe_id'], o['repeat'])][o['target_id']] = law
    ids = [t for t, _ in TARGETS]
    matrix = [[None] * len(ids) for _ in ids]
    counts = [[None] * len(ids) for _ in ids]
    hits = [[None] * len(ids) for _ in ids]
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            hit = n = 0
            for (probe, repeat), v in laws.items():
                if a == b:
                    other = laws.get((probe, 1), {})
                    if repeat == 0 and a in v and a in other:
                        n += 1
                        hit += same(v[a], other[a])
                elif a in v and b in v:
                    n += 1
                    hit += same(v[a], v[b])
            matrix[i][j], counts[i][j] = (hit / n if n else None), n
            hits[i][j] = hit
    cmap = LinearSegmentedColormap.from_list('blue', ['#f3f7fd', '#2a78d6', '#103f7a'])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    fig.patch.set_facecolor(base.SURFACE)
    ax.set_facecolor(base.SURFACE)
    for i in range(len(ids)):
        for j in range(len(ids)):
            value = matrix[i][j]
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, facecolor=cmap(value / 0.8),
                                       edgecolor=base.INK if i == j else 'none', linewidth=1.2 if i == j else 0))
            ink = 'white' if value > 0.45 else base.INK
            lo, hi = wilson(hits[i][j], counts[i][j])
            ax.text(j, i - 0.12, f'{value:.0%}', ha='center', va='center', fontsize=11, fontweight='bold', color=ink)
            ax.text(j, i + 0.13, f'95% CI {lo:.0%}–{hi:.0%}', ha='center', va='center', fontsize=6.8, color=ink)
            ax.text(j, i + 0.3, f'n = {counts[i][j]}', ha='center', va='center', fontsize=6.5, color=ink)
    names = [name for _, name in TARGETS]
    ax.set_xticks(range(len(ids)), names, fontsize=9.5, color=base.INK)
    ax.set_yticks(range(len(ids)), names, fontsize=9.5, color=base.INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, len(ids) - 0.5)
    ax.set_ylim(len(ids) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Qwen3.5-397B: how often two routes return identical logprobs', x=0.03, ha='left',
                 fontsize=12, fontweight='bold', color=base.INK)
    fig.text(0.03, 0.02, 'Same prompt and repeat; all top-5 logprobs within 0.002 nats. Outlined diagonal: a '
             "provider's own repeat\nvs repeat (its nondeterminism floor). 400 dataset prompts, OpenRouter, "
             '2026-11-13. Intervals: 95% Wilson.', fontsize=7.5, color=base.INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(ROOT / 'reports/reseller-identity-2026-11-13.png', dpi=160, facecolor=base.SURFACE)


if __name__ == '__main__':
    main()
