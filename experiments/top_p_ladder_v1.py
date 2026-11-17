"""Top-P ladder: absolute probabilities of the top tokens from sampling alone (local ground truth, no API calls).

Rule (inclusive nucleus): a token is in the top-P support iff the total probability of the tokens ranked above it is
below P. So if a token ranked below the top K is ever sampled at top_p = P, the top-K mass is < P. Not seeing it in m
draws rules out "in the nucleus" only statistically, with miss probability (1 - share)^m.

Generalising the K = 1 bisection in logit_reconstruction_benchmark_v1 (which pins p1): for each K = 1..9, bisect on
top_p while watching the (K+1)-th token named by a T=1 pilot, which brackets C_K = mass ranked above it. Differences
of consecutive brackets bracket each top token's absolute probability: p_k = C_k - C_{k-1}.

Black-box throughout: the pilot names the targets and estimates their shares from samples; targets are then ordered
by their brackets (the pilot can misorder near-ties); truth is used only to score. Llama-3.2-3B-Instruct fp32 logits on the same 100 dataset prompts; exact provider sampling order.
Output: reports/top-p-ladder-2026-11-17.{json,png}
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))
import logit_reconstruction_benchmark_v1 as bench  # noqa: E402
from kremlinology.sampled_logit_estimates import nucleus_cutoff_bracket  # noqa: E402

K_MAX = 9
BUDGETS = [10_000, 100_000]  # total samples per prompt across the whole ladder
ALPHA = 0.01
OUT = ROOT / 'reports/top-p-ladder-2026-11-17'


def ladder(logits, budget, generator):
    """Brackets of C_K (mass above the (K+1)-th pilot-named token) for K = 1..K_MAX."""
    pilot_n = budget // 5
    counts = bench.draw(logits, (1.0, 0, 1.0), pilot_n, generator)
    named = [t for t, c in counts.most_common(K_MAX + 1) if c >= 3]
    used, out = pilot_n, []
    per_target = (budget - pilot_n) // max(1, len(named) - 1)
    for k in range(1, len(named)):
        target = named[k]
        share = counts[target] / pilot_n / 2  # half the estimate, for safety
        per_probe = max(1, math.ceil(math.log(ALPHA) / math.log1p(-share)))
        probes, spent = [], 0
        lower, upper = 0.0, 1.0
        for _ in range(40):
            if spent + per_probe > per_target:
                break
            middle = (lower + upper) / 2
            seen = bench.draw(logits, (1.0, 0, middle), per_probe, generator).get(target, 0) > 0
            spent += per_probe
            probes.append((middle, per_probe, seen))
            b = nucleus_cutoff_bracket(probes, minimum_share=share, alpha=ALPHA)
            lower, upper = b['lower'], b['upper']
        used += spent
        out.append(dict(k=k, target=target, lower=lower, upper=upper, probes=len(probes)))
    return out, used


def score(brackets, logits):
    probs = torch.softmax(logits.double(), 0)
    rows = []
    prev = None
    # Pilot order. A token's own probability is the gap between the mass above it and the mass above the next token,
    # which is only identified when the two brackets are disjoint and ordered (otherwise the pilot may have swapped a
    # near-tie and the order is unknown): those pairs are reported as unresolved, not guessed.
    for b in brackets:
        p_t = float(probs[b['target']])
        truth_c = float(probs[probs > p_t].sum())  # mass ranked above the target
        row = dict(k=b['k'], c_true=truth_c, c_lower=b['lower'], c_upper=b['upper'],
                   covered=b['lower'] <= truth_c <= b['upper'], width=b['upper'] - b['lower'])
        if prev is not None:
            resolved = prev['c_upper'] <= b['lower']
            row.update(resolved=resolved)
            if resolved:
                p_lower, p_upper = b['lower'] - prev['c_upper'], b['upper'] - prev['c_lower']
                p_true = truth_c - prev['c_true']  # probability between the two targets (the earlier target's own)
                row.update(p_true=p_true, p_lower=p_lower, p_upper=p_upper, p_covered=p_lower <= p_true <= p_upper,
                           p_width=p_upper - p_lower)
        rows.append(row)
        prev = row
    return rows


def main():
    torch.set_num_threads(8)
    items = bench.prompts()
    truth = bench.next_logits(torch.float32, items)
    generator = torch.Generator().manual_seed(20261117)
    results = {}
    for budget in BUDGETS:
        rows = []
        for p in range(len(items)):
            brackets, used = ladder(truth[p], budget, generator)
            for r in score(brackets, truth[p]):
                rows.append(dict(r, prompt=p))
        results[budget] = rows
        ks = sorted({r['k'] for r in rows})
        print(f'budget {budget}:')
        for k in ks:
            sel = [r for r in rows if r['k'] == k]
            cov = np.mean([r['covered'] for r in sel])
            width = np.median([r['width'] for r in sel])
            pk = [r for r in sel if 'p_true' in r]
            pcov = np.mean([r['p_covered'] for r in pk]) if pk else float('nan')
            pw = np.median([r['p_width'] for r in pk]) if pk else float('nan')
            resolved = sum(1 for r in sel if r.get('resolved'))
            print(f'  K={k}: n={len(sel):3d} C_K coverage {cov:.2f} median width {width:.4f} | '
                  f'p resolved {resolved}/{len(sel)}, coverage {pcov:.2f}, median width {pw:.4f}')
    OUT.parent.mkdir(exist_ok=True)
    summary = {}
    for budget, rows in results.items():
        summary[str(budget)] = [dict(k=k, n=len([r for r in rows if r['k'] == k]),
                                     c_coverage=float(np.mean([r['covered'] for r in rows if r['k'] == k])),
                                     c_median_width=float(np.median([r['width'] for r in rows if r['k'] == k])),
                                     p_coverage=float(np.mean([r['p_covered'] for r in rows if r['k'] == k and 'p_true' in r]))
                                     if any('p_true' in r for r in rows if r['k'] == k) else None,
                                     resolved=sum(1 for r in rows if r['k'] == k and r.get('resolved')),
                                     p_median_width=float(np.median([r['p_width'] for r in rows if r['k'] == k and 'p_true' in r]))
                                     if any('p_true' in r for r in rows if r['k'] == k) else None)
                                for k in sorted({r['k'] for r in rows})]
    Path(f'{OUT}.json').write_text(json.dumps(dict(model=bench.MODEL, prompts=len(items), budgets=BUDGETS,
                                                   alpha=ALPHA, summary=summary), indent=1))
    Path(f'{OUT}-records.json').write_text(json.dumps({str(b): r for b, r in results.items()}))


if __name__ == '__main__':
    main()
