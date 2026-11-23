"""Do two routes share a backend? Exact first-token logprob identity vs each route's own repeatability.

Every route answers the same prompts twice (max_tokens=1, top-k logprobs), interleaved. For
each route, the self-match rate is how often its two repeats return identical logprobs (within
0.002 nats): inference is rarely bit-deterministic, so this is its fingerprint floor. For each
pair, the cross-match rate is how often the two routes agree on the same prompt and repeat.
Independent deployments essentially never agree to 0.002 nats; a pair that agrees about as often
as each agrees with itself is serving from the same numerics (engine, weights, hardware,
batching). 'Lands-on' adds: when route A's own repeats differ, how often B equals one of them.
Metadata (latency, response-id shape, OpenRouter upstream-id shape, prompt-token parity) is
reported alongside; it is supporting evidence, not proof.
"""
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations

from .. import report, stats, text


def plan(routes, prompts=60, repeats=2, **_):
    return dict(requests=len(routes) * prompts * repeats, max_tokens_each=1)


def shape(identifier):
    """Format of an ID with the variable parts abstracted: 'gen-1794562124-CkrNL2...' -> 'gen-#10-*20'."""
    if not identifier:
        return None
    parts = []
    for part in re.split(r'([-_:.])', identifier):
        if part.isdigit():
            parts.append(f'#{len(part)}')
        elif part.isalnum() and (len(part) > 12 or (len(part) > 6 and any(c.isdigit() for c in part))):
            parts.append(f'*{len(part)}')
        else:
            parts.append(part)
    return ''.join(parts)[:40]


def run(client, routes, prompts=60, repeats=2, top_logprobs=5, prompt_file=None, seed=0, extra=None,
        lookups=20, **_):
    items = text.prompts(prompts, seed, prompt_file)
    generator = random.Random(seed)
    calls = [(route, i, r) for r in range(repeats) for i in range(len(items)) for route in routes]
    generator.shuffle(calls)  # interleave routes, prompts and repeats

    def ask(route, i, repeat):
        return client.chat(route, [dict(role='user', content=items[i])], max_tokens=1, top_logprobs=top_logprobs,
                           extra=extra, require_parameters=True, tag=dict(prompt=i, repeat=repeat),
                           key=f'fp:{route.label}:{i}:{repeat}')
    results = client.parallel([lambda c=c: ask(*c) for c in calls])
    laws, meta = {}, defaultdict(lambda: defaultdict(list))
    for (route, i, repeat), result in zip(calls, results):
        if result is None:
            continue
        if result.ok and result.first_token_logprobs:
            laws[(route.label, i, repeat)] = result.first_token_logprobs
        m = meta[route.label]
        m['ok'].append(result.ok)
        if result.elapsed_s is not None and result.ok:
            m['latency'].append(result.elapsed_s)
        m['id_shape'].append(shape(result.response_id))
        m['served_by'].append(result.served_by)
        m['prompt_tokens'].append((i, result.prompt_tokens))
        if route.is_openrouter and len(m['generation_ids']) < lookups and result.response_id:
            m['generation_ids'].append(result.response_id)
    lookups_todo = [(route, gid) for route in routes for gid in meta[route.label].pop('generation_ids', [])]
    records = client.parallel([lambda r=r, g=g: client.generation(r, g) for r, g in lookups_todo])
    for (route, _), record in zip(lookups_todo, records):
        if record:
            meta[route.label]['upstream'].append(shape(record.get('upstream_id')))
    labels = [r.label for r in routes]
    self_rate = {}
    for label in labels:
        hits = [stats.same_law(laws[(label, i, 0)], laws[(label, i, 1)]) for i in range(len(items))
                if (label, i, 0) in laws and (label, i, 1) in laws] if repeats >= 2 else []
        self_rate[label] = stats.proportion_interval(sum(hits), len(hits))
    pairs = {}
    for a, b in combinations(labels, 2):
        hits = [stats.same_law(laws[(a, i, r)], laws[(b, i, r)]) for i in range(len(items)) for r in range(repeats)
                if (a, i, r) in laws and (b, i, r) in laws]
        lands = []
        for i in range(len(items)):
            own = [laws.get((a, i, r)) for r in range(repeats)]
            if repeats >= 2 and all(own) and not stats.same_law(own[0], own[1]):
                lands += [any(stats.same_law(laws[(b, i, r)], o) for o in own)
                          for r in range(repeats) if (b, i, r) in laws]
        cross = stats.proportion_interval(sum(hits), len(hits))
        pairs[f'{a} ~ {b}'] = dict(cross=cross, lands_on=stats.proportion_interval(sum(lands), len(lands)),
                                   verdict=verdict(cross, self_rate[a], self_rate[b]),
                                   prompt_token_parity=parity(meta[a]['prompt_tokens'], meta[b]['prompt_tokens']))
    summary = {}
    for label in labels:
        m = meta[label]
        summary[label] = dict(success_rate=sum(m['ok']) / len(m['ok']) if m['ok'] else None,
                              median_latency_s=statistics.median(m['latency']) if m['latency'] else None,
                              response_id_shapes=Counter(m['id_shape']).most_common(2),
                              upstream_id_shapes=Counter(m['upstream']).most_common(2),
                              served_by=Counter(m['served_by']).most_common(2), self_match=self_rate[label])
    results = dict(prompts=len(items), repeats=repeats, top_logprobs=top_logprobs, routes=summary, pairs=pairs)
    return results, render(results)


def parity(a, b):
    a, b = dict(a), dict(b)
    common = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    return dict(equal=sum(a[i] == b[i] for i in common), n=len(common))


def verdict(cross, self_a, self_b):
    floor = min(self_a['fraction'] or 0, self_b['fraction'] or 0)
    if not cross['n'] or not self_a['n'] or not self_b['n']:
        return 'not enough data'
    if self_a['upper'] < 0.05 and self_b['upper'] < 0.05:
        return 'inconclusive: neither route repeats itself exactly, so exact matching cannot fingerprint'
    if cross['lower'] > 0.05 and cross['fraction'] >= 0.5 * floor:
        return 'same numerics: matches as often as the routes match themselves (shared backend likely)'
    if cross['upper'] < max(0.02, 0.25 * floor):
        return 'different numerics: never or rarely identical (independent deployments)'
    return 'inconclusive'


def render(results):
    lines = ['# Same-backend fingerprint', '',
             f"{results['prompts']} prompts x {results['repeats']} repeats, first token, top-{results['top_logprobs']} "
             'logprobs. Identical = same reported tokens, every logprob within 0.002 nats.', '',
             '## Routes', '',
             report.table(['route', 'self-match (repeat vs repeat)', 'median latency', 'served by', 'response ids',
                           'upstream ids (OpenRouter)'],
                          [[k, report.interval(v['self_match']),
                            f"{v['median_latency_s']:.2f} s" if v['median_latency_s'] else '—',
                            ', '.join(f'{a} ({n})' for a, n in v['served_by'] if a) or '—',
                            ', '.join(f'`{a}`' for a, _ in v['response_id_shapes'] if a) or '—',
                            ', '.join(f'`{a}`' for a, _ in v['upstream_id_shapes'] if a) or '—']
                           for k, v in results['routes'].items()]), '',
             '## Pairs', '',
             report.table(['pair', 'identical (same prompt+repeat)', 'lands on A\'s states', 'prompt tokens equal',
                           'verdict'],
                          [[k, report.interval(v['cross']), report.interval(v['lands_on']),
                            f"{v['prompt_token_parity']['equal']}/{v['prompt_token_parity']['n']}", v['verdict']]
                           for k, v in results['pairs'].items()]), '',
             '_A bitwise match shows shared numerics, not a contract; metadata columns are supporting evidence._']
    return '\n'.join(lines)
