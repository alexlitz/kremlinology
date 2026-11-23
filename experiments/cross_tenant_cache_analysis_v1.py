"""Pool cross_tenant_cache_probe runs: does a prompt cached through one route hit when re-sent through the other?

Per model and order (VP = Venice writes, Parasail reads; PV; VV and PP = same-route controls): the second request is
a hit if at least half its prompt is reported cached. Hit rates with 95% Wilson intervals.
Run: python experiments/cross_tenant_cache_analysis_v1.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_utils import wilson  # noqa: E402


def main():
    cells = defaultdict(lambda: [0, 0])
    for k, run in enumerate(sys.argv[1:]):
        trials = defaultdict(dict)
        for line in (Path(run) / 'requests.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trials[(k, r['trial'])][r['step']] = r
        for steps in trials.values():
            first, second = steps.get(0), steps.get(1)
            if not first or not second or first.get('status') != 200 or second.get('status') != 200:
                continue
            usage = second.get('usage') or {}
            prompt = usage.get('prompt_tokens') or 0
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
            cell = cells[(second['model'], second['order'])]
            cell[0] += prompt and cached >= 0.5 * prompt
            cell[1] += 1
    out = []
    for (model, order), (hits, n) in sorted(cells.items()):
        lo, hi = wilson(hits, n)
        out.append(dict(model=model, order=order, hits=hits, n=n, lo=lo, hi=hi))
        print(f'{model:34s} {order}  {hits:3d}/{n:<3d}  95% CI {lo:.0%}-{hi:.0%}')
    Path('reports/cross-tenant-cache-pooled-2026-11-17.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
