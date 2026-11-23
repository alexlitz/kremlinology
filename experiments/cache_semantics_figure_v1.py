"""Figure: which prompt prefixes each API reuses from cache (pools cache_semantics v1-v3).

Cell = mean fraction of the reusable prefix that the second request read from cache:
  repeat     the whole prompt (no breakpoint; for Claude also with one)
  diverge    the first half, shared verbatim, then new text
  follow-up  the previous request's whole prompt (chat turn appended)
  extend     the previous prompt, with a new block appended and no breakpoint at its end
Responses reporting 0 prompt tokens (an OpenAI reporting glitch) and non-200s are skipped.
Run after cache_semantics_analysis_v1.py on each run directory.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
plt.rcParams['text.parse_math'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402

RUNS = [Path(DATA_ROOT, f'kremlinology-cache-semantics-v{v}') for v in (1, 2, 3)]
ROWS = [  # display name, labels pooled
    ('Claude Haiku 5.5', ['claude-haiku-5.5', 'claude-haiku-5.5@vertex']),
    ('Claude Sonnet 5.5', ['claude-sonnet-5.5']),
    ('Claude Opus 5.5', ['claude-opus-5.5']),
    ('Claude Haiku 4.5', ['claude-haiku-4.5']),
    ('GPT-6.1 Terra / Sol', ['gpt-6.1-terra', 'gpt-6.1-sol']),
    ('GPT-6 Luna / Sol', ['gpt-6-luna', 'gpt-6-sol']),
    ('GPT-5.6 Luna', ['gpt-5.6-luna']),
    ('GPT-5 / 5.4 nano', ['gpt-5-nano', 'gpt-5.4-nano']),
    ('DeepSeek-V4 Flash (DeepInfra)', ['deepseek-v4-flash']),
]
COLUMNS = [  # display, tests for breakpoint APIs, tests for automatic APIs, reusable share of the step-0 prompt
    ('same prompt,\nno breakpoint', ['no_breakpoint_repeat', 'no_bp_repeat'], ['no_breakpoint_repeat', 'repeat'], 1.0),
    ('same prompt,\nbreakpoint', ['breakpoint_repeat', 'bp_repeat'], [], 1.0),
    ('first half shared,\nthen new text', ['diverge_half'], ['diverge_half', 'diverge_half_auto'], 0.5),
    ('follow-up\nchat turn', ['followup'], ['followup'], 1.0),
    ('prompt extended,\nno new breakpoint', ['extend_without_bp'], ['extend_without_bp'], 1.0),
]
BREAKPOINT_LABELS = ('claude',)


def collect():
    cells = defaultdict(list)
    for run in RUNS:
        steps = defaultdict(dict)
        for r in json.loads((run / 'analysis.json').read_text()):
            steps[(r['label'], r['test'], r.get('repeat', 0))][r['step']] = r
        for (label, test, _), s in steps.items():
            first, second = s.get(0), s.get(1)
            if not first or not second or first.get('status') != 200 or second.get('status') != 200:
                continue
            if not first.get('prompt') or not second.get('prompt'):
                continue
            for name, labels in ROWS:
                if label not in labels:
                    continue
                for c, (_, bp_tests, auto_tests, share) in enumerate(COLUMNS):
                    tests = bp_tests if label.startswith(BREAKPOINT_LABELS) else auto_tests
                    if test in tests:
                        reusable = first['prompt'] * share
                        cells[(name, c)].append(min(1.0, (second.get('cached') or 0) / reusable))
    return cells


def figure(cells, path):
    cmap = LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#103f7a'])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(base.SURFACE)
    ax.set_facecolor(base.SURFACE)
    for r, (name, _) in enumerate(ROWS):
        for c in range(len(COLUMNS)):
            values = cells.get((name, c))
            if not values:
                ax.add_patch(plt.Rectangle((c - 0.46, r - 0.44), 0.92, 0.88, facecolor=base.SURFACE,
                                           edgecolor=base.GRID, linewidth=1))
                automatic = not any(label.startswith(BREAKPOINT_LABELS) for label in ROWS[r][1])
                ax.text(c, r, 'n/a (automatic)' if automatic and c == 1 else 'not tested', ha='center', va='center', fontsize=8, color=base.INK2)
                continue
            mean = sum(values) / len(values)
            ax.add_patch(plt.Rectangle((c - 0.46, r - 0.44), 0.92, 0.88, facecolor=cmap(mean), edgecolor='none'))
            ink = 'white' if mean > 0.55 else base.INK
            ax.text(c, r - 0.08, f'{mean:.0%}', ha='center', va='center', fontsize=10.5, fontweight='bold', color=ink)
            ax.text(c, r + 0.22, f'n = {len(values)}', ha='center', va='center', fontsize=7.5, color=ink)
    ax.set_xlim(-0.5, len(COLUMNS) - 0.5)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xticks(range(len(COLUMNS)), [c[0] for c in COLUMNS], fontsize=9, color=base.INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS], fontsize=9.5, color=base.INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(3.5, color=base.INK2, linewidth=0.8)
    ax.axhline(6.5, color=base.INK2, linewidth=0.8)
    fig.suptitle('Share of the reusable prefix read from cache on the second request', x=0.02, ha='left',
                 fontsize=12, fontweight='bold', color=base.INK)
    fig.text(0.02, 0.012, 'Claude rows send cache_control breakpoints (except "no breakpoint"); OpenAI and DeepSeek '
             'cache automatically.\nCells are means over trials; the 46% and 66% cells mix near-full hits with complete misses. '
             'OpenRouter usage fields, 2026-11-13.', fontsize=7.5, color=base.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=base.SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    cells = collect()
    figure(cells, ROOT / 'reports/cache-semantics-2026-11-13.png')
    for (name, c), v in sorted(cells.items()):
        print(f'{name:32s} {COLUMNS[c][0].replace(chr(10), " "):36s} {[round(x, 2) for x in v]}')
