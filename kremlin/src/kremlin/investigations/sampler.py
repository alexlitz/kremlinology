"""Is the sampler what the route claims? First-token sampling checked against the route's own logprobs.

The prompt is chosen per route: several candidates are probed and the one whose reported first-token
distribution has the highest entropy is used (a peaked prompt cannot reveal an ignored top_k or
temperature). The route's reported top-20 logprobs at T=1 are the reference. Then one-token samples (n per request when supported, else repeated requests):
  top_k=1       every sample must be the reported argmax
  top_k=2       samples must stay within the reported top two
  top_p         at temperature 1, top_p halfway between p1 and p1 + p2 (from the reported logprobs): a token is in the
                nucleus iff the mass ranked above it is below top_p, so exactly the top two may appear; a third-ranked
                sample means the top-2 mass would have to be below top_p, i.e. top_p was not applied
  T=1           sample frequencies vs reported probabilities (chi-square, pooled small cells)
  T=0.5         temperature fitted from the counts (maximum likelihood) should be about 0.5
Assumes provider order temperature -> top-k and that reported logprobs are pre-sampling at T=1;
a failure means the parameter is ignored or the reported logprobs are not the sampled law (or the
logits vary between requests; repeat to separate).
"""
from collections import Counter

from .. import report, stats

import math

MARGIN = 0.5  # nats: outside-top-k tokens closer than this to the cutoff are near-ties, not definite violations
CANDIDATES = [
    'Name a color. Reply with one word only.',
    'Pick a random fruit. Reply with one word only.',
    'Say a random first name. Reply with the name only.',
    'Choose a random animal. Reply with one word only.',
    'Pick a random integer from 1 to 20. Reply with the number only.',
    'Name a city anywhere in the world. Reply with the city name only.',
    'Write a random English noun. Reply with the word only.',
    'Pick a random letter of the alphabet. Reply with the letter only.',
]


def entropy(law):
    probs = [math.exp(v) for v in law.values()]
    rest = max(0.0, 1 - sum(probs))
    return -sum(p * math.log(p) for p in probs + [rest] if p > 0)


def pick_prompt(client, route):
    """Reference logprobs for each candidate (in parallel); the most uncertain one wins (top-1 at most 0.7 preferred)."""
    refs = client.parallel([lambda j=j, p=p: client.chat(
        route, [dict(role='user', content=p)], max_tokens=1, temperature=1.0, top_logprobs=20, require_parameters=True,
        tag=dict(sampler='reference', prompt=p), key=f'sampler:{route.label}:candidate:{j}')
        for j, p in enumerate(CANDIDATES)])
    best = None
    for prompt, ref in zip(CANDIDATES, refs):
        if ref is None or not ref.ok or not ref.first_token_logprobs:
            if best is None:
                best = (None, prompt, ref)
            continue
        top = max(ref.first_token_logprobs.values())
        score = entropy(ref.first_token_logprobs) - (10 if math.exp(top) > 0.7 else 0)
        if best is None or best[0] is None or score > best[0]:
            best = (score, prompt, ref)
    return best[1], best[2]


def plan(routes, samples=60, **_):
    return dict(requests_without_n=len(routes) * (len(CANDIDATES) + 5 * samples),
                requests_with_n=len(routes) * (len(CANDIDATES) + 5 * (samples + 9) // 10))


def draw(client, route, samples, prompt, setting, **params):
    """samples first tokens as the API reports them. The first request asks for n=10; if the route honours n the
    rest also use n, otherwise single-sample requests. Remaining requests run in parallel."""
    extra = {key: v for key, v in params.items() if key in ('top_k', 'top_p')}
    temperature = params.get('temperature', 1.0)

    def call(index, k):
        body_extra = dict(extra, **({'n': k} if k > 1 else {}))
        return client.chat(route, [dict(role='user', content=prompt)], max_tokens=1, temperature=temperature,
                           top_logprobs=1, extra=body_extra or None, require_parameters=True,
                           tag=dict(sampler=setting), key=f'sampler:{route.label}:{setting}:{index}')
    out = Counter()
    first = call(0, min(10, samples))
    if not first.ok and first.status and 400 <= first.status < 500:  # n rejected: start over one at a time
        first = call(1, 1)
        offset = 2
    else:
        offset = 1
    if not first.ok:
        return None, f'HTTP {first.status}'
    texts = first.choice_texts or [first.text]
    for text_ in texts:
        out[text_] += 1
    per = min(10, samples) if len(texts) > 1 else 1
    remaining = samples - sum(out.values())
    sizes = [min(per, remaining - i * per) for i in range((remaining + per - 1) // per)] if remaining > 0 else []
    results = client.parallel([lambda i=i, k=k: call(offset + i, k) for i, k in enumerate(sizes)])
    for result in results:
        if result is None or not result.ok:
            continue
        for text_ in result.choice_texts or [result.text]:
            out[text_] += 1
    return out, None


def run(client, routes, samples=60, **_):
    results, rows = {}, []
    for route in routes:
        prompt, ref = pick_prompt(client, route)
        if not ref.ok or not ref.first_token_logprobs:
            results[route.label] = dict(error=f'no reference logprobs (HTTP {ref.status})')
            rows.append([route.label, 'no logprobs', '', '', ''])
            continue
        # reference keyed by token text (samples are texts)
        law = {}
        for key, lp in ref.first_token_logprobs.items():
            text = bytes.fromhex(key).decode('utf-8', 'replace') if not key.startswith('s:') else key[2:]
            law[text] = lp
        ranked = sorted(law, key=law.get, reverse=True)
        p1 = math.exp(law[ranked[0]])
        informative = samples * (1 - p1) >= 5  # an ignored top_k would show at least ~5 non-argmax samples
        out = dict(prompt=prompt, reference=law, top1_probability=p1, informative=informative)
        def classify(counts, k):
            """Outside-top-k samples: definite if far below the k-th reference token, else a near-tie that
            request-to-request logit variation (batching, low precision) could reorder."""
            cutoff = law[ranked[k - 1]]
            definite = sum(v for t, v in counts.items() if t not in ranked[:k] and law.get(t, -99) < cutoff - MARGIN)
            near = sum(v for t, v in counts.items() if t not in ranked[:k] and law.get(t, -99) >= cutoff - MARGIN)
            return dict(counts=dict(counts), violations=definite, near_ties=near)
        k1, err = draw(client, route, samples, prompt, 'top_k_1', top_k=1)
        out['top_k_1'] = dict(err=err) if err else classify(k1, 1)
        k2, err = draw(client, route, samples, prompt, 'top_k_2', top_k=2)
        out['top_k_2'] = dict(err=err) if err else classify(k2, 2)
        if len(ranked) >= 2:
            p2 = math.exp(law[ranked[1]])
            nucleus = p1 + p2 / 2  # mid-gap: robust to inclusive/exclusive boundary conventions
            tp, err = draw(client, route, samples, prompt, 'top_p', temperature=1.0, top_p=round(nucleus, 6))
            out['top_p'] = dict(err=err) if err else dict(classify(tp, 2), top_p=nucleus)
        else:
            out['top_p'] = dict(err='fewer than two reported logprobs')
        t1, err = draw(client, route, samples, prompt, 't1', temperature=1.0)
        if err:
            out['t1'] = dict(err=err)
        else:
            labels = ranked + [t for t in t1 if t not in law]
            probs = [math.exp(law[t]) for t in ranked]
            probs.append(max(0.0, 1 - sum(probs)))
            counts = [t1.get(t, 0) for t in ranked] + [sum(t1.get(t, 0) for t in labels[len(ranked):])]
            out['t1'] = dict(counts=dict(t1), gof=stats.multinomial_gof(counts, probs))
        th, err = draw(client, route, samples, prompt, 't05', temperature=0.5)
        if err:
            out['t05'] = dict(err=err)
        else:
            out['t05'] = dict(counts=dict(th), fitted_temperature=stats.temperature_mle(
                [law[t] for t in ranked], [th.get(t, 0) for t in ranked]))
        results[route.label] = out

        def verdict(cell, name):
            if 'err' in cell:
                return f'unsupported ({cell["err"]})'
            if name in ('top_k_1', 'top_k_2', 'top_p'):
                n = sum(cell['counts'].values())
                if cell['violations']:
                    return f"VIOLATED ({cell['violations']}/{n} far outside top-k)"
                if cell['near_ties']:
                    return (f"honoured up to near-ties ({cell['near_ties']}/{n} within {MARGIN} nats of the cutoff: "
                            'varying logits can reorder them)')
                return 'honoured' if out['informative'] else 'no violation (uninformative: prompt too peaked)'
            if name == 't1':
                g = cell['gof']
                return '—' if not g else (f"matches logprobs (p={g['p_value']:.2g})" if g['p_value'] >= 0.001
                                          else f"DIFFERS from logprobs (p={g['p_value']:.1g})")
            t = cell['fitted_temperature']
            return '—' if t is None else (f'T≈{t:.2f}' + (' (ok)' if 0.3 <= t <= 0.8 else ' (WRONG for 0.5)'))
        rows.append([f"{route.label} (p1={p1:.2f})", verdict(out['top_k_1'], 'top_k_1'), verdict(out['top_k_2'], 'top_k_2'),
                     verdict(out['top_p'], 'top_p'), verdict(out['t1'], 't1'), verdict(out['t05'], 't05')])
    md = '\n'.join(['# Sampler behaviour', '', f'{samples} one-token samples per setting; per-route prompt: '
                    + '; '.join(f"{k}: \"{v.get('prompt')}\"" for k, v in results.items()), '',
                    report.table(['route', 'top_k=1', 'top_k=2', 'top_p (top two only)', 'T=1 vs reported logprobs', 'T=0.5 fitted'], rows),
                    '', '_Violations are definite only if the route applies the declared parameters exactly; a T=1 '
                        'mismatch can also mean reported logprobs are post-processed or logits vary between requests._'])
    return dict(samples=samples, routes=results), md
