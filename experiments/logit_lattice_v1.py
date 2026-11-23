"""Are a provider's logits rounded to a low-precision grid before the softmax? (no new requests)

Logprob differences between two tokens of one response equal their logit difference (the softmax shift cancels). If the
final-layer logits are stored in bf16, both logits lie on the bf16 grid, so for logits in [2^e, 2^(e+1)) their
difference is a multiple of 2^(e-7) (1/8 for logits of 16-32, 1/16 for 8-16). fp32 logits give no grid. Per response:
the coarsest step s in {1/2 ... 1/1024} such that every top-20 gap (top-1 minus rank k) is within 2e-4 of a multiple of
s (fp32 reporting error). Gaps are only multiples of the grid when both logits share an exponent, so the score is the
share of responses whose gaps fit a grid of 1/64 or coarser, and the most common step.
This reads the numeric pipeline (lm_head output precision, rounding before softmax), not the GPU vendor.
Run: python experiments/logit_lattice_v1.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

STEPS = [2 ** -k for k in range(1, 11)]
TOL = 2e-4


def step_of(gaps):
    for s in STEPS:
        if all(abs(g / s - round(g / s)) * s < TOL for g in gaps):
            return s
    return None


def main():
    by = defaultdict(list)
    for run in sys.argv[1:]:
        for line in open(Path(run) / 'observations.jsonl'):
            o = json.loads(line)
            if o.get('status') != 'success' or not o.get('records'):
                continue
            top = o['records'][0].get('top_logprobs') or []
            lps = [t['logprob'] for t in top if t.get('logprob') is not None]
            if len(lps) < 5:
                continue
            gaps = [lps[0] - x for x in lps[1:] if lps[0] - x > 1e-9]
            by[(Path(run).name, o['target_id'])].append(step_of(gaps) if gaps else None)
    out = []
    for (run, target), steps in sorted(by.items()):
        n = len(steps)
        on_grid = sum(1 for s in steps if s and s >= 1 / 64)
        common = Counter(s for s in steps if s).most_common(2)
        out.append(dict(run=run, target=target, responses=n, share_on_grid=on_grid / n,
                        common_steps={f'1/{round(1 / s)}': c for s, c in common}))
        print(f"{run:40s} {target:24s} n={n:4d} on a 1/64-or-coarser grid: {on_grid / n:6.1%}  common steps {out[-1]['common_steps']}")
    Path('reports/logit-lattice-2026-11-18.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
