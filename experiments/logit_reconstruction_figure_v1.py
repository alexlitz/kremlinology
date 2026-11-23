"""Figure for logit_reconstruction_benchmark_v1: gap error and recovery vs sample budget."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402
from ci_utils import median as _median  # noqa: E402
import random as _random


def _cells(records, strategy, budget):
    """Per prompt: (errors on 2-5 nat gaps, (recovered, total) on 5-10 nat gaps), fp32 truth, naive support."""
    out = []
    for r in records:
        if r['source'] == 'fp32' and r['support'] == 'naive' and r['strategy'] == strategy and r['budget'] == budget:
            errs, rec, tot = [], 0, 0
            for rank, true_gap, est, _se in r['gaps']:
                g = -true_gap
                if 2 <= g < 5 and est is not None:
                    errs.append(abs(est - true_gap))
                if 5 <= g < 10:
                    tot += 1
                    rec += est is not None
            out.append((errs, rec, tot))
    return out


def _stats(cells):
    errs = [e for c in cells for e in c[0]]
    rec, tot = sum(c[1] for c in cells), sum(c[2] for c in cells)
    return (_median(errs) if errs else float('nan')), (rec / tot if tot else float('nan'))


def _cluster_ci(cells, seed=0, iterations=1000):
    """95% bootstrap over prompts (gaps within a prompt are correlated)."""
    rng = _random.Random(seed)
    draws = [_stats([rng.choice(cells) for _ in cells]) for _ in range(iterations)]
    out = []
    for k in range(2):
        v = sorted(d[k] for d in draws if d[k] == d[k])
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float('nan'), float('nan')))
    return out

STYLE = {'plain': ('#2a78d6', 'T=1, no truncation'), 'hot-top10': ('#eb6834', 'T=2, top-K=10'),
         'ladder': ('#1baf7a', 'top-K ladder 2/4/10'), 'adaptive': ('#eda100', 'adaptive pair design')}


def main():
    data = json.loads((ROOT / 'reports/logit-reconstruction-2026-11-13.json').read_text())
    rows = [r for r in data['gaps'] if r['support'] == 'naive']
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(base.SURFACE)
    base.style(left, 'Error on gaps 2-5 nats below top-1 (ranks 2-10)', xlabel='samples per prompt',
               ylabel='median |error| (nats)')
    base.style(right, 'Recovered: hard gaps (5-10 nats below top-1)', xlabel='samples per prompt',
               ylabel='share of ranks 2-10 recovered')
    records = json.loads((ROOT / 'reports/logit-reconstruction-2026-11-13-records.json').read_text())['records']
    for k, (strategy, (color, label)) in enumerate(STYLE.items()):
        series = sorted((r['budget'], r) for r in rows if r['source'] == 'fp32' and r['strategy'] == strategy)
        x = [b * (1 + 0.04 * (k - 1.5)) for b, _ in series]
        e_mid, e_lo, e_hi, r_mid, r_lo, r_hi = [], [], [], [], [], []
        for budget, summary in series:
            cells = _cells(records, strategy, budget)
            err, rec = _stats(cells)
            (el, eh), (rl, rh) = _cluster_ci(cells, seed=budget + k)
            e_mid.append(err); e_lo.append(err - el); e_hi.append(eh - err)
            r_mid.append(rec); r_lo.append(rec - rl); r_hi.append(rh - rec)
            print(strategy, budget, 'err', round(err, 3), 'vs', summary['by_true_gap']['2-5'].get('median_abs_error'),
                  'rec', round(rec, 3), 'vs', summary['by_true_gap']['5-10'].get('recovered'))
        left.errorbar(x, e_mid, yerr=[e_lo, e_hi], fmt='-o', color=color, linewidth=2, markersize=6, capsize=3,
                      markeredgecolor=base.SURFACE, label=label)
        right.errorbar(x, r_mid, yerr=[r_lo, r_hi], fmt='-o', color=color, linewidth=2, markersize=6, capsize=3,
                       markeredgecolor=base.SURFACE, label=label)
    floor = [r['by_true_gap']['2-5']['median_abs_error'] for r in rows if r['source'] == 'bf16' and r['strategy'] == 'hot-top10'][0]
    left.axhline(floor, color=base.INK2, linestyle=(0, (3, 3)), linewidth=1)
    left.text(110, floor * 1.12, f'bf16 server vs fp32 truth, 100k samples: {floor:.2f}', fontsize=8, color=base.INK2)
    for ax in (left, right):
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    left.set_yscale('log')
    left.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5], ['0.02', '0.05', '0.1', '0.2', '0.5'])
    left.minorticks_off()
    right.set_ylim(-0.03, 1.05)
    right.legend(frameon=False, fontsize=8.5, labelcolor=base.INK, loc='lower right')
    fig.text(0.01, 0.01, f"Llama-3.2-3B-Instruct, {data['prompts']} dataset prompts, exact provider-order sampling; error bars: 95% bootstrap over prompts; "
             'labels with >= 5 samples.\nT=1 errors at 100 samples cover only the 14% of 2-5 nat gaps it finds. 95% intervals cover 93-97% at 1,000+ samples (fp32).', fontsize=7.5,
             color=base.INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / 'reports/logit-reconstruction-2026-11-13.png', dpi=160, facecolor=base.SURFACE)


if __name__ == '__main__':
    main()
