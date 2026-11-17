"""Pool max_tokens_probe runs: does each endpoint stop at the requested cap (64), and what is billed?

Per endpoint and variant (max_tokens, streamed max_tokens, max_completion_tokens): completion tokens billed (median,
range), reasoning tokens billed, and the share of requests over the cap with a 95% Wilson interval.
Run: python experiments/max_tokens_analysis_v1.py RUN_DIR [RUN_DIR ...]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_utils import wilson  # noqa: E402

CAP = 64


def main():
    cells = defaultdict(list)
    for run in sys.argv[1:]:
        for line in (Path(run) / 'requests.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('status') != 200 or not r.get('usage'):
                continue
            u = r['usage']
            reasoning = (u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
            cells[(r['label'], r['field'] + (' (stream)' if r['stream'] else ''))].append(
                dict(completion=u.get('completion_tokens') or 0, reasoning=reasoning, cost=u.get('cost') or 0))
    out = []
    for (label, variant), rows in sorted(cells.items()):
        over = sum(1 for x in rows if x['completion'] > CAP)
        lo, hi = wilson(over, len(rows))
        c = [x['completion'] for x in rows]
        rec = dict(label=label, variant=variant, n=len(rows), over_cap=over, over_lo=lo, over_hi=hi,
                   completion_median=statistics.median(c), completion_min=min(c), completion_max=max(c),
                   reasoning_median=statistics.median(x['reasoning'] for x in rows),
                   cost_median=statistics.median(x['cost'] for x in rows))
        out.append(rec)
        print(f"{label:26s} {variant:28s} n={len(rows):2d} over cap {over:2d} ({lo:.0%}-{hi:.0%}) billed "
              f"{rec['completion_median']:.0f} [{min(c)}-{max(c)}], reasoning {rec['reasoning_median']:.0f}, "
              f"${rec['cost_median']:.5f}/req")
    Path('reports/max-tokens-pooled-2026-11-17.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
