"""How well do sampling strategies reconstruct next-token logits? Local ground truth.

Ground truth: fp32 next-token logits of Llama-3.2-3B-Instruct on 100 dataset prompts
(50 CodeAlpaca, 50 wikitext; the Llama-70B campaign's manifest), chat template applied.
Samples are drawn exactly as a provider would (temperature, then top-K, then inclusive
top-P, multinomial), so this isolates the statistics from server bugs; the HTTP path and
fault modes are covered by local_sampler_pipeline_v1. The n parameter only changes how
many requests a budget costs: n independent choices per request equal n requests here.

Budget B = total sampled tokens per prompt, split equally across a strategy's runs.
Strategies:
  plain       T=1, no truncation
  hot-top10   T=2, top-K=10 (flatten, and stop spending samples on the tail)
  ladder      top-K=2 at T=1, top-K=4 at T=1.5, top-K=10 at T=2.5, pooled MLE
  adaptive    pilot B/4 as hot-top10, then top-K=2 at the pair_design temperature for
              the estimated top-1/top-2 gap, the rest at T=2.5 top-K=10; pooled MLE
Estimands: gaps z_r - z_1 for true ranks r=2..10 (logits are identified only up to a shift),
and the top-10 conditional distribution (TV). Gaps use labels with >= MIN_COUNT samples in
the run (naive; at most the 40 most sampled), or labels chosen on one half and counted on the other (split).
Absolute mass: top-P bisection on the top-1 probability (nucleus_cutoff_bracket) with the
same budget, which pins the shift so log-probs become absolute.
Floor: the same estimators fed samples from the bf16 model, scored against fp32 truth.
Local only (GPU0): CUDA_VISIBLE_DEVICES=0 python experiments/logit_reconstruction_benchmark_v1.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.sampled_logit_estimates import nucleus_cutoff_bracket, pair_design, pooled_logit_fit  # noqa: E402

MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
BUDGETS = (100, 1_000, 10_000, 100_000)
RANKS = 10
MIN_COUNT = 5
MAX_LABELS = 40  # most-sampled labels kept per run (conditional likelihood on that subset), as the harness does
SEED = 20261113
STRATEGIES = {
    'plain': [(1.0, 0, 1.0)],
    'hot-top10': [(2.0, 10, 1.0)],
    'ladder': [(1.0, 2, 1.0), (1.5, 4, 1.0), (2.5, 10, 1.0)],
    'adaptive': None,
}
OUT = ROOT / 'reports/logit-reconstruction-2026-11-13'


def prompts():
    manifest = json.loads((ROOT / 'configs/bench/llama70b-dataset-tv-v1.prompts.json').read_text())['prompts']
    return [p for p in manifest if p['dataset'] == 'code'][:50] + [p for p in manifest if p['dataset'] == 'wiki'][:50]


@torch.no_grad()
def next_logits(dtype, items):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to('cuda').eval()
    out = []
    for item in items:
        ids = tokenizer.apply_chat_template([dict(role='user', content=item['text'])], add_generation_prompt=True,
                                            return_tensors='pt').to('cuda')
        out.append(model(ids).logits[0, -1].float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.stack(out)


def sampling_law(logits, temperature, top_k, top_p):
    """Provider order: temperature, top-K, inclusive top-P (keep while preceding mass < p)."""
    scaled = logits.double() / temperature
    order = torch.argsort(scaled, descending=True)
    if top_k:
        order = order[:top_k]
    probs = torch.softmax(scaled[order], 0)
    if top_p < 1:
        before = torch.cumsum(probs, 0) - probs
        keep = before < top_p
        order, probs = order[keep], probs[keep] / probs[keep].sum()
    return order, probs


def draw(logits, setting, count, generator):
    if count <= 0:
        return Counter()
    order, probs = sampling_law(logits, *setting)
    picks = torch.multinomial(probs.float(), count, replacement=True, generator=generator)
    return Counter(order[picks].tolist())


def run_observation(counts, temperature, minimum=MIN_COUNT):
    ids = [t for t, c in counts.most_common(MAX_LABELS) if c >= minimum]
    return (ids, [counts[t] for t in ids], temperature) if len(ids) >= 2 else None


def split_observation(counts_a, counts_b, temperature):
    """Support chosen on half A (>= MIN_COUNT); counts from half B on it, zero-count labels dropped."""
    chosen = [t for t, c in counts_a.most_common(MAX_LABELS) if c >= MIN_COUNT]
    ids = [t for t in chosen if counts_b.get(t, 0) > 0]
    return (ids, [counts_b[t] for t in ids], temperature) if len(ids) >= 2 else None


def fit(observations):
    observations = [o for o in observations if o]
    if not observations:
        return None
    # Keep the component connected to the most-sampled token; others are unidentified.
    tokens = Counter()
    for ids, values, _ in observations:
        for token, value in zip(ids, values):
            tokens[token] += value
    anchor = tokens.most_common(1)[0][0]
    linked, changed = {anchor}, True
    while changed:
        changed = False
        for ids, _, _ in observations:
            if linked & set(ids) and not set(ids) <= linked:
                linked |= set(ids)
                changed = True
    observations = [o for o in observations if set(o[0]) <= linked]
    try:
        result = pooled_logit_fit(observations, tokens=[anchor] + sorted(linked - {anchor}))
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(result['logits']).all():
        return None
    return dict(zip(result['tokens'], zip(result['logits'], result['standard_error']))), result


def strategy_runs(name, logits_sample, budget, generator):
    """Returns list of (setting, counts_a, counts_b) with the budget split in halves."""
    if name != 'adaptive':
        settings = STRATEGIES[name]
        share = budget // len(settings)
        return [(s, draw(logits_sample, s, share // 2, generator), draw(logits_sample, s, share - share // 2,
                                                                       generator)) for s in settings]
    pilot_setting = (2.0, 10, 1.0)
    pilot = budget // 4
    pa, pb = draw(logits_sample, pilot_setting, pilot // 2, generator), draw(logits_sample, pilot_setting,
                                                                            pilot - pilot // 2, generator)
    runs = [(pilot_setting, pa, pb)]
    total = pa + pb
    top = [t for t, _ in total.most_common(2)]
    temperature = 1.0
    if len(top) == 2 and total[top[1]] > 0:
        gap = 2.0 * math.log(total[top[0]] / total[top[1]])
        if gap > 0:
            temperature = pair_design(gap, maximum_temperature=4.0)['temperature']
    rest = budget - pilot
    for setting, share in (((temperature, 2, 1.0), rest // 2), ((2.5, 10, 1.0), rest - rest // 2)):
        runs.append((setting, draw(logits_sample, setting, share // 2, generator),
                     draw(logits_sample, setting, share - share // 2, generator)))
    return runs


def score_gaps(estimate, truth_logits, top):
    """Per true rank 2..RANKS: (rank, true gap, estimated gap or None, se)."""
    rows = []
    if not estimate or top[0] not in estimate[0]:
        return [(r + 1, float(truth_logits[top[r]] - truth_logits[top[0]]), None, None) for r in range(1, RANKS)]
    values, result = estimate
    index = {t: i for i, t in enumerate(result['tokens'])}
    covariance = result['covariance']
    z1, i1 = values[top[0]][0], index[top[0]]
    for r in range(1, RANKS):
        truth = float(truth_logits[top[r]] - truth_logits[top[0]])
        token = top[r]
        if token not in values:
            rows.append((r + 1, truth, None, None))
            continue
        i = index[token]
        variance = covariance[i, i] + covariance[i1, i1] - 2 * covariance[i, i1]
        rows.append((r + 1, truth, float(values[token][0] - z1), float(math.sqrt(max(variance, 0)))))
    return rows


def top10_tv(estimate, truth_logits, top):
    truth = torch.softmax(truth_logits[top].double(), 0).numpy()
    if not estimate:
        return None
    values = estimate[0]
    est = np.array([values[t][0] if t in values else -np.inf for t in top], dtype=float)
    if not np.isfinite(est).any():
        return None
    est = np.exp(est - est[np.isfinite(est)].max())
    est = est / est.sum()
    return 0.5 * float(np.abs(est - truth).sum())


def mass_bisection(logits_sample, truth_logits, top, budget, generator, alpha=0.01, steps_max=40):
    """Bisection on top-P to bracket p1 = P(top-1) at T=1 (the runner-up is in the nucleus iff p1 < top_p).

    Black-box: a plain T=1 pilot (a fifth of the budget) names the runner-up and estimates its
    probability p2. Inside any nucleus that includes it, its share is at least p2, so absence in
    m draws rules it out at level alpha once (1 - p2/2)^m <= alpha (half the estimate, for safety).
    """
    pilot = max(budget // 5, 1)
    counts = draw(logits_sample, (1.0, 0, 1.0), pilot, generator)
    used = pilot
    lower, upper, probes = 0.0, 1.0, []
    common = counts.most_common(2)
    p1 = float(torch.softmax(truth_logits.double(), 0)[top[0]])
    if len(common) == 2 and common[1][1] >= 3:
        second, share = common[1][0], common[1][1] / pilot / 2
        per_probe = max(1, math.ceil(math.log(alpha) / math.log1p(-share)))
        for _ in range(steps_max):
            if used + per_probe > budget:
                break
            middle = (lower + upper) / 2
            seen = draw(logits_sample, (1.0, 0, middle), per_probe, generator).get(second, 0) > 0
            used += per_probe
            probes.append((middle, per_probe, seen))
            bracket = nucleus_cutoff_bracket(probes, minimum_share=share, alpha=alpha)
            lower, upper = bracket['lower'], bracket['upper']
    return dict(p1=p1, lower=lower, upper=upper, covered=lower <= p1 <= upper, width=upper - lower,
                log_width=math.log(upper) - math.log(lower) if lower > 0 else math.inf, probes=len(probes),
                gap_1_2=float(truth_logits[top[0]] - truth_logits[top[1]]))


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['source'], r['strategy'], r['support'], r['budget'])].append(r)
    table = []
    for (source, strategy, support, budget), rows in sorted(groups.items()):
        gaps = [g for r in rows for g in r['gaps']]
        found = [g for g in gaps if g[2] is not None]
        errors = [abs(g[2] - g[1]) for g in found]
        covered = [abs(g[2] - g[1]) <= 1.96 * g[3] for g in found if g[3] and g[3] > 0]
        tvs = [r['tv'] for r in rows if r['tv'] is not None]
        bins = {}
        for low, high in ((0, 2), (2, 5), (5, 10), (10, 1e9)):
            members = [g for g in gaps if low <= -g[1] < high]
            hit = [abs(g[2] - g[1]) for g in members if g[2] is not None]
            bins[f'{low}-{high if high < 1e9 else "inf"}'] = dict(
                n=len(members), recovered=len(hit) / len(members) if members else None,
                median_abs_error=float(np.median(hit)) if hit else None)
        table.append(dict(source=source, strategy=strategy, support=support, budget=budget, prompts=len(rows),
                          recovered_fraction=len(found) / len(gaps), median_abs_error=float(np.median(errors))
                          if errors else None, p90_abs_error=float(np.percentile(errors, 90)) if errors else None,
                          ci95_coverage=float(np.mean(covered)) if covered else None,
                          median_top10_tv=float(np.median(tvs)) if tvs else None,
                          tv_available=len(tvs) / len(rows), by_true_gap=bins))
    return table


def main():
    torch.set_num_threads(8)
    items = prompts()
    truth = next_logits(torch.float32, items)
    bf16 = next_logits(torch.bfloat16, items)
    generator = torch.Generator().manual_seed(SEED)
    records, mass = [], []
    for source, sampled in (('fp32', truth), ('bf16', bf16)):
        for p in range(len(items)):
            top = torch.argsort(truth[p], descending=True)[:RANKS].tolist()
            for budget in BUDGETS:
                if source == 'bf16' and budget != BUDGETS[-1]:
                    continue
                for strategy in STRATEGIES:
                    runs = strategy_runs(strategy, sampled[p], budget, generator)
                    naive = fit([run_observation(a + b, s[0]) for s, a, b in runs])
                    split = fit([split_observation(a, b, s[0]) for s, a, b in runs])
                    for support, estimate in (('naive', naive), ('split', split)):
                        records.append(dict(source=source, prompt=items[p]['probe_id'], strategy=strategy,
                                            support=support, budget=budget,
                                            gaps=score_gaps(estimate, truth[p], top),
                                            tv=top10_tv(estimate, truth[p], top)))
                if source == 'fp32' or budget == BUDGETS[-1]:
                    mass.append(dict(source=source, prompt=items[p]['probe_id'], budget=budget,
                                     **mass_bisection(sampled[p], truth[p], top, budget, generator)))
            if p % 10 == 9:
                print(source, p + 1, 'prompts', flush=True)
    table = summarize(records)
    mass_table = []
    for source, budget in [('fp32', b) for b in BUDGETS] + [('bf16', BUDGETS[-1])]:
        rows = [m for m in mass if m['budget'] == budget and m['source'] == source]
        finite = [m['log_width'] for m in rows if math.isfinite(m['log_width'])]
        mass_table.append(dict(source=source, budget=budget, coverage=float(np.mean([m['covered'] for m in rows])),
                               median_probes=float(np.median([m['probes'] for m in rows])),
                               median_width=float(np.median([m['width'] for m in rows])),
                               median_log_width=float(np.median(finite)) if finite else None,
                               bracketed_fraction=len(finite) / len(rows)))
    truth_p1 = torch.softmax(truth.double(), -1).max(-1).values.numpy()
    floor = torch.softmax(bf16.double(), -1)
    exact = torch.softmax(truth.double(), -1)
    bf16_tv = (0.5 * (floor - exact).abs().sum(-1)).numpy()
    OUT.parent.mkdir(exist_ok=True)
    Path(f'{OUT}.json').write_text(json.dumps(dict(
        model=MODEL, prompts=len(items), budgets=BUDGETS, ranks=RANKS, min_count=MIN_COUNT, seed=SEED,
        top1_probability_quartiles=np.percentile(truth_p1, [25, 50, 75]).tolist(),
        bf16_vs_fp32_full_tv_median=float(np.median(bf16_tv)), gaps=table, mass=mass_table), indent=1))
    Path(f'{OUT}-records.json').write_text(json.dumps(dict(records=records, mass=mass)))
    for row in table:
        print(f"{row['source']:5s} {row['strategy']:10s} {row['support']:6s} B={row['budget']:>6d} "
              f"recovered {row['recovered_fraction']:.2f} |err| med {row['median_abs_error'] or float('nan'):.3f} "
              f"p90 {row['p90_abs_error'] or float('nan'):.3f} cover95 {row['ci95_coverage'] or float('nan'):.2f} "
              f"TV10 {row['median_top10_tv'] or float('nan'):.4f}")
    for row in mass_table:
        print('mass', row)


if __name__ == '__main__':
    main()
