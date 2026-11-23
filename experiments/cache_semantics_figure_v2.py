"""Figure v2: which prompt prefixes each API reuses from cache. Pools cache_semantics v1-v5 (closed models, direct
OpenRouter probes) and the witnessed kremlin cache-prefix runs on open models (open-v1, open-v2), grouped by
attention type. Open-model cells use the same definitions (share of the reusable prefix read from cache).

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
from ci_utils import wilson  # noqa: E402

RUNS = [Path(DATA_ROOT, f'kremlinology-cache-semantics-v{v}') for v in ('1', '2', '3', '4', '4b', '5', '6', '7a', '7b', '8', '9')]
OPEN_RUNS = [Path(DATA_ROOT, f'kremlinology-cache-semantics-open-v{v}') for v in (1, 2, 3)]
ROWS = [  # display name, labels pooled (closed: probe labels; open: kremlin route specs "model@provider")
    ('# Closed, explicit breakpoints', []),
    ('Claude Haiku 5.5', ['claude-haiku-5.5', 'claude-haiku-5.5@vertex']),
    ('Claude Sonnet 5.5', ['claude-sonnet-5.5']),
    ('Claude Opus 5.5', ['claude-opus-5.5']),
    ('Claude Haiku 4.5', ['claude-haiku-4.5']),
    ('# Closed, automatic', []),
    ('GPT-6.1 Terra / Sol', ['gpt-6.1-terra', 'gpt-6.1-sol']),
    ('GPT-6 Luna / Sol', ['gpt-6-luna', 'gpt-6-sol']),
    ('GPT-5.6 Luna', ['gpt-5.6-luna']),
    ('GPT-5 / 5.4 nano', ['gpt-5-nano', 'gpt-5.4-nano']),
    ('DeepSeek-V4 Flash · DeepInfra', ['deepseek-v4-flash']),
    ('# Open, full attention', []),
    ('Llama 3.3-70B · CoreWeave', ['meta-llama/llama-3.3-70b-instruct@coreweave/fp16']),
    ('# Open, sliding window + global', []),
    ('gpt-oss-120b · Nebius', ['openai/gpt-oss-120b@nebius/fp4']),
    ('gpt-oss-120b · CoreWeave', ['openai/gpt-oss-120b@coreweave/fp4']),
    ('gpt-oss-120b · Parasail', ['openai/gpt-oss-120b@parasail/fp4']),
    ('Gemma 4 31B · DeepInfra', ['google/gemma-4-31b-it@deepinfra/turbo']),
    ('Gemma 4 26B · CoreWeave', ['google/gemma-4-26b-a4b-it@coreweave/bf16']),
    ('# Open, linear attention + global (hybrid)', []),
    ('Qwen3.5-397B · Parasail', ['qwen/qwen3.5-397b-a17b@parasail/fp8']),
    ('Qwen3.5-397B · DeepInfra', ['qwen/qwen3.5-397b-a17b@deepinfra/fp8']),
    ('Qwen3.6-35B · CoreWeave', ['qwen/qwen3.6-35b-a3b@coreweave/fp8']),
    ('Qwen3-Next 80B · Parasail', ['qwen/qwen3-next-80b-a3b-instruct@parasail/fp8']),
]
HIT = 0.10  # a hit reads more than 10% of the reusable prefix (the 32-64-token chat-template header doesn't count)
OPEN_TESTS = {'repeat': (0, 1.0), 'diverge_half': (2, 0.5), 'followup': (3, 1.0), 'extend': (4, 1.0)}
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
    for run in OPEN_RUNS:
        if not run.exists():
            continue
        for results in sorted(run.glob('*/results.json')):
            meta = json.loads((results.parent / 'run.json').read_text())
            spec = meta['args']['routes'][0].split(':', 1)[1]
            body = json.loads(results.read_text()).get('results')
            if not body:  # an interrupted route (e.g. API credit exhausted) has no results
                continue
            tests = body['tests']
            for name, labels in ROWS:
                if spec not in labels:
                    continue
                for test, values in tests.items():
                    if test not in OPEN_TESTS:
                        continue
                    column, share = OPEN_TESTS[test]
                    for v in values:
                        if v.get('first_prompt') and v.get('cached') is not None:
                            cells[(name, column)].append(min(1.0, v['cached'] / (v['first_prompt'] * share)))
    return cells


def figure(cells, path):
    cmap = LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#103f7a'])
    fig, ax = plt.subplots(figsize=(11.5, 0.6 * len(ROWS) + 1.8))
    fig.patch.set_facecolor(base.SURFACE)
    ax.set_facecolor(base.SURFACE)
    for r, (name, _) in enumerate(ROWS):
        if name.startswith('# '):
            ax.text(-0.5, r + 0.15, name[2:], ha='left', va='center', fontsize=9, fontweight='bold', color=base.INK2)
            continue
        for c in range(len(COLUMNS)):
            values = cells.get((name, c))
            if not values:
                ax.add_patch(plt.Rectangle((c - 0.46, r - 0.44), 0.92, 0.88, facecolor=base.SURFACE,
                                           edgecolor=base.GRID, linewidth=1))
                automatic = not any(label.startswith(BREAKPOINT_LABELS) for label in ROWS[r][1])
                ax.text(c, r, 'n/a (automatic)' if automatic and c == 1 else 'not tested', ha='center', va='center', fontsize=8, color=base.INK2)
                continue
            median = sorted(values)[len(values) // 2] if len(values) % 2 else \
                (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
            hits = sum(1 for v in values if v > HIT)
            ax.add_patch(plt.Rectangle((c - 0.46, r - 0.44), 0.92, 0.88, facecolor=cmap(median), edgecolor='none'))
            ink = 'white' if median > 0.55 else base.INK
            lo, hi = wilson(hits, len(values))
            ax.text(c, r - 0.16, f'{median:.0%}', ha='center', va='center', fontsize=10.5, fontweight='bold', color=ink)
            ax.text(c, r + 0.12, f'hits {hits}/{len(values)}', ha='center', va='center', fontsize=7.3, color=ink)
            ax.text(c, r + 0.32, f'hit rate {lo:.0%}–{hi:.0%}', ha='center', va='center', fontsize=6.6, color=ink)
    ax.set_xlim(-0.5, len(COLUMNS) - 0.5)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xticks(range(len(COLUMNS)), [c[0] for c in COLUMNS], fontsize=9, color=base.INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS)), ['' if r[0].startswith('# ') else r[0] for r in ROWS], fontsize=9.5, color=base.INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Which prefixes are reused from cache? Share of the reusable prefix read on the second request', x=0.02, ha='left',
                 fontsize=12, fontweight='bold', color=base.INK)
    fig.text(0.02, 0.006, 'Cell: median share over trials; "hits" = trials reading > 10% of the reusable prefix, with the 95% Wilson interval on the hit rate (misses '
             'on otherwise-hitting rows are requests that reached a replica without the cache).\nClaude rows send '
             'cache_control breakpoints (except "no breakpoint"); all others cache automatically. Hybrid models save state '
             'only at whole blocks, so a full hit reads ~75-90% of the prompt.\nOpenRouter usage fields, 2026-11-13 to '
             '11-17; open-model rows are witnessed runs.\nNo cache at all on Qwen3.5-35B (3 providers), Nemotron-3 Nano '
             '(Mamba hybrid, Nebius), Gemma 4 26B (AkashML, Darkbloom).', fontsize=7.3, color=base.INK2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.985))
    fig.savefig(path, dpi=160, facecolor=base.SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    cells = collect()
    figure(cells, ROOT / 'reports/cache-semantics-2026-11-17.png')
    for (name, c), v in sorted(cells.items()):
        print(f'{name:32s} {COLUMNS[c][0].replace(chr(10), " "):36s} {[round(x, 2) for x in v]}')
