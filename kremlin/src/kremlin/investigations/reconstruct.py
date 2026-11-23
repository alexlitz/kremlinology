"""Compare routes that do not return logprobs, by reconstructing first-token distributions from samples.

For each prompt, every route draws one-token samples at temperature T (default 2) with top_k K (default 10),
using n per request where honoured. Under the stated sampler (temperature then top-k), the T=1 distribution
over the sampled tokens is proportional to count^T, so a route's first-token law (restricted to its top-K
support) is reconstructed without any logprobs. Routes are compared by the TV between reconstructed laws
(union of tokens, averaged over prompts), with a multinomial bootstrap 95% interval. Routes that also report
logprobs get a self-check: reconstructed vs reported law. More than K distinct tokens means top_k was ignored.
Accuracy (local ground truth, 1,000-10,000 samples): median gap error 0.07-0.2 nats.
"""
import random
from itertools import combinations

from .. import report, text
from .sampler import draw


def plan(routes, prompts=5, samples=200, **_):
    return dict(requests_without_n=len(routes) * prompts * samples,
                requests_with_n=len(routes) * prompts * ((samples + 9) // 10))


def law_from_counts(counts, temperature, minimum=1):
    weights = {t: c ** temperature for t, c in counts.items() if c >= minimum}
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()} if total else {}


def tv(p, q):
    return 0.5 * sum(abs(p.get(t, 0) - q.get(t, 0)) for t in set(p) | set(q))


def bootstrap(counts_a, counts_b, temperature, iterations=200, seed=0):
    generator = random.Random(seed)

    def resample(counts):
        tokens, weights = list(counts), list(counts.values())
        n = sum(weights)
        out = {}
        for t in generator.choices(tokens, weights, k=n):
            out[t] = out.get(t, 0) + 1
        return out
    values = sorted(tv(law_from_counts(resample(counts_a), temperature), law_from_counts(resample(counts_b), temperature))
                    for _ in range(iterations))
    return values[int(0.025 * iterations)], values[int(0.975 * iterations) - 1]


def run(client, routes, prompts=5, samples=200, temperature=2.0, top_k=10, prompt_file=None, seed=0, **_):
    items = text.prompts(prompts, seed, prompt_file)
    jobs = [(route, i) for route in routes for i in range(len(items))]

    def sample(route, i):
        counts, err = draw(client, route, samples, items[i], f'recon{i}', temperature=temperature, top_k=top_k)
        return counts, err

    def reported(route, i):
        r = client.chat(route, [dict(role='user', content=items[i])], max_tokens=1, temperature=1.0, top_logprobs=20,
                        require_parameters=True, tag=dict(reconstruct='reported', prompt=i),
                        key=f'recon:{route.label}:reported:{i}')
        if not r.ok or not r.first_token_logprobs:
            return None
        import math
        law = {}
        for key, lp in r.first_token_logprobs.items():
            label = key[2:] if key.startswith('s:') else bytes.fromhex(key).decode('utf-8', 'replace')
            law[label] = math.exp(lp)
        return law
    sampled = dict(zip(jobs, client.parallel([lambda j=j: sample(*j) for j in jobs])))
    reports = dict(zip(jobs, client.parallel([lambda j=j: reported(*j) for j in jobs])))
    per_route, rows = {}, []
    for route in routes:
        laws, errors, ignored, self_check = {}, [], 0, []
        for i in range(len(items)):
            counts, err = sampled[(route, i)] or (None, 'cancelled')
            if err:
                errors.append(err)
                continue
            if len(counts) > top_k:
                ignored += 1
            laws[i] = law_from_counts(counts, temperature)
            rep = reports.get((route, i))
            if rep:
                support = [t for t in laws[i] if t in rep]
                if support:
                    norm = sum(rep[t] for t in support)
                    self_check.append(tv({t: laws[i][t] for t in support}, {t: rep[t] / norm for t in support}))
        per_route[route.label] = dict(laws=laws, errors=errors[:3], top_k_ignored_prompts=ignored,
                                      self_check_tv=sum(self_check) / len(self_check) if self_check else None)
        rows.append([route.label, len(laws), ignored,
                     f"{per_route[route.label]['self_check_tv']:.3f}" if self_check else '— (no logprobs)'])
    pairs = {}
    for a, b in combinations(routes, 2):
        shared = [i for i in range(len(items)) if i in per_route[a.label]['laws'] and i in per_route[b.label]['laws']]
        if not shared:
            continue
        values = [tv(per_route[a.label]['laws'][i], per_route[b.label]['laws'][i]) for i in shared]
        intervals = [bootstrap(sampled[(a, i)][0], sampled[(b, i)][0], temperature, seed=i) for i in shared]
        pairs[f'{a.label} ~ {b.label}'] = dict(mean_tv=sum(values) / len(values),
                                              ci=(sum(x[0] for x in intervals) / len(intervals),
                                                  sum(x[1] for x in intervals) / len(intervals)), prompts=len(shared))
    results = dict(samples=samples, temperature=temperature, top_k=top_k, prompts=len(items),
                   routes={k: {kk: vv for kk, vv in v.items() if kk != 'laws'} for k, v in per_route.items()},
                   pairs=pairs)
    md = '\n'.join(['# Reconstructed distributions (no logprobs needed)', '',
                    f'{samples} samples per prompt at T={temperature}, top_k={top_k}; T=1 law ∝ count^T on the sampled '
                    f'tokens; {len(items)} prompts.', '',
                    report.table(['route', 'prompts reconstructed', 'prompts with > top_k tokens',
                                  'TV vs its own reported logprobs'], rows), '',
                    report.table(['pair', 'mean TV', 'bootstrap 95% (per-prompt mean)', 'prompts'],
                                 [[k, f"{v['mean_tv']:.3f}", f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]", v['prompts']]
                                  for k, v in pairs.items()]), '',
                    '_Sampling noise alone gives a nonzero TV between identical routes: compare a pair with the '
                    'bootstrap interval, or include a route twice (different labels) as a noise control._'])
    return results, md
