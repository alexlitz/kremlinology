"""Output differences beyond the first token: next-token distributions at forced later positions.

1. The reference route (first) samples a continuation of each prompt at T=1 with per-token logprobs.
2. Every route is asked for the next-token distribution at positions t (default 1, 4, 8) after the first
   t-1 generated tokens, sent as a final partial assistant message (prefill).
3. Validity per route and position:
   - faithfulness (reference only): its forced logprob of the token it sampled must match the logprob it
     reported while generating (within 0.05 nats); otherwise the route does not continue prefills;
   - prompt-token parity: a route's prompt tokens at t minus at t=1 must equal the reference's (the prefill
     is tokenised the same way and no closing turn is added).
4. Per position: dataset-average partial TV bounds vs the reference with 95% intervals (valid cells only).
Providers that do not support assistant prefill fail validity and are reported as such, not compared.
"""
from collections import defaultdict

from .. import report, stats, text


def plan(routes, prompts=40, positions=(1, 4, 8), **_):
    return dict(requests=prompts + len(routes) * prompts * len(positions))


def run(client, routes, prompts=40, positions=(1, 4, 8), top_logprobs=20, prompt_file=None, seed=0, error=0.01,
        extra=None, **_):
    items = text.prompts(prompts, seed, prompt_file)
    reference = routes[0]
    length = max(positions)
    generations = client.parallel([lambda i=i: client.chat(
        reference, [dict(role='user', content=items[i])], max_tokens=length, temperature=1.0, top_logprobs=1,
        extra=extra, require_parameters=True, tag=dict(prompt=i), key=f'later:generate:{i}') for i in range(len(items))])
    continuations = {}
    for i, result in enumerate(generations):
        if result is not None and result.ok and len(result.tokens) >= length:
            continuations[i] = result.tokens
    jobs = [(route, i, t) for route in routes for i in continuations for t in positions]

    def ask(route, i, t):
        prefix = ''.join(token for token, _ in continuations[i][:t - 1])
        messages = [dict(role='user', content=items[i])] + ([dict(role='assistant', content=prefix)] if t > 1 else [])
        return client.chat(route, messages, max_tokens=1, top_logprobs=top_logprobs, extra=extra,
                           require_parameters=True, tag=dict(prompt=i, position=t), key=f'later:{route.label}:{i}:{t}')
    answers = {}
    for job, result in zip(jobs, client.parallel([lambda j=j: ask(*j) for j in jobs])):
        if result is not None and result.ok:
            answers[(job[0].label, job[1], job[2])] = result
    # validity
    faithful = {}
    for i in continuations:
        for t in positions:
            got = answers.get((reference.label, i, t))
            token, logprob = continuations[i][t - 1]
            reported = None
            if got and got.first_token_logprobs:
                key = bytes(token.encode()).hex()
                reported = got.first_token_logprobs.get(key, got.first_token_logprobs.get('s:' + token))
            faithful[(i, t)] = reported is not None and logprob is not None and abs(reported - logprob) <= 0.05
    results = dict(reference=reference.label, positions=list(positions), prompts=len(continuations), routes={})
    for route in routes:
        per_position = {}
        for t in positions:
            lowers, uppers, invalid = [], [], 0
            for i in continuations:
                mine, ref = answers.get((route.label, i, t)), answers.get((reference.label, i, t))
                base_mine, base_ref = answers.get((route.label, i, 1)), answers.get((reference.label, i, 1))
                if not (mine and ref and mine.first_token_logprobs and ref.first_token_logprobs and faithful[(i, t)]):
                    invalid += 1
                    continue
                if base_mine and base_ref and None not in (mine.prompt_tokens, base_mine.prompt_tokens,
                                                         ref.prompt_tokens, base_ref.prompt_tokens):
                    if mine.prompt_tokens - base_mine.prompt_tokens != ref.prompt_tokens - base_ref.prompt_tokens:
                        invalid += 1
                        continue
                bounds = stats.partial_tv_bounds(mine.first_token_logprobs, ref.first_token_logprobs, error)
                if bounds:
                    lowers.append(bounds[0])
                    uppers.append(bounds[1])
            per_position[t] = dict(interval=stats.average_interval(lowers, uppers), valid=len(lowers), invalid=invalid)
        results['routes'][route.label] = per_position

    def cell(v):
        a = v['interval']
        if not a:
            return f"— ({v['valid']} valid)"
        return f"{100 * a['mean_lower']:.1f}–{100 * a['mean_upper']:.1f}% [{100 * a['ci_lower']:.0f}, {100 * a['ci_upper']:.0f}] (n={v['valid']})"
    md = '\n'.join(['# Later-token output differences', '',
                    f"Reference **{reference.label}** generated {length} tokens per prompt ({len(continuations)} prompts); "
                    f'every route was forced onto that history by assistant prefill. Partial TV vs the reference, '
                    f'report-error allowance {error} nats; cells failing faithfulness or prompt-token parity are excluded.',
                    '', report.table(['route'] + [f'position {t}' for t in positions],
                                     [[label] + [cell(v[t]) for t in positions] for label, v in results['routes'].items()]),
                    '', '_The reference row is the reference against itself at the same forced history (repeatability). '
                        'Many open-model providers do not continue assistant prefills; their cells show as invalid._'])
    return results, md
