"""Prompt-cache probes. Every trial uses fresh text (random nonce first), so trials never hit
each other's caches. Cache reads come from the API's usage fields (cached_tokens /
cache_read_input_tokens); routes that report none can only be probed by timing (not done here).

  share A B      write with A, re-send with B 4 s later; also B->A, A->A, B->B controls.
                 Same model through two providers -> do they share one cache (resale)?
                 Same route with two keys (;key=ENV_B) -> is the cache shared across accounts?
  blocks R       repeat prompts of increasing length: cache granularity and minimum length.
  ttl R          write, then re-read after each delay: how long entries survive.
  prefix R       resume semantics: half-shared prompt, extended prompt, follow-up chat turn.

For Anthropic models (anthropic: routes, or openrouter:anthropic/... models) a cache_control
breakpoint is put on the document block; other providers cache automatically.
"""
import math
import random
import time
from collections import defaultdict
from functools import reduce

from .. import report, text

ASK = 'Summarize the document above in one sentence.'


def breakpoints(route):
    return route.kind == 'anthropic' or route.model.startswith('anthropic/')


def message(route, *blocks, bp_index=-1, ttl=None):
    content = []
    for k, block in enumerate(blocks):
        item = dict(type='text', text=block)
        if breakpoints(route) and k == (bp_index % len(blocks)):
            item['cache_control'] = dict(type='ephemeral', **({'ttl': ttl} if ttl else {}))
        content.append(item)
    content.append(dict(type='text', text=ASK))
    return [dict(role='user', content=content)]


def hit(result, shared_tokens_estimate=None):
    return bool(result.ok and (result.cached_tokens or 0) > 64)


# ---- share --------------------------------------------------------------------------------
def plan_share(routes, trials=3, **_):
    return dict(requests=4 * trials * 2)


def run_share(client, routes, trials=3, chars=24000, gap_s=4.0, seed=0, salt=None, **_):
    salt = text.fresh_salt(salt)
    a, b = routes
    generator = random.Random(seed)
    orders = dict(AB=(a, b), BA=(b, a), AA=(a, a), BB=(b, b))
    jobs = [(name, t) for name in orders for t in range(trials)]
    generator.shuffle(jobs)
    docs = {job: text.prose(chars, generator, salt) for job in jobs}  # drawn in job order: deterministic

    def trial(name, t):
        first, second = orders[name]
        doc = docs[(name, t)]
        write = client.chat(first, message(first, doc), max_tokens=8, tag=dict(order=name, trial=t, step=0),
                            key=f'share:{name}:{t}:write')
        time.sleep(gap_s)
        read = client.chat(second, message(second, doc), max_tokens=8, tag=dict(order=name, trial=t, step=1),
                           key=f'share:{name}:{t}:read')
        return dict(write_ok=write.ok, read_ok=read.ok, cached=read.cached_tokens, prompt=read.prompt_tokens,
                    hit=hit(read))
    cells = defaultdict(list)
    for (name, _), cell in zip(jobs, client.parallel([lambda j=j: trial(*j) for j in jobs])):
        if cell is not None:
            cells[name].append(cell)
    def rate(name):
        valid = [c for c in cells[name] if c['write_ok'] and c['read_ok']]
        return dict(hits=sum(c['hit'] for c in valid), n=len(valid),
                    median_cached=sorted(c['cached'] or 0 for c in valid)[len(valid) // 2] if valid else None)
    summary = {name: rate(name) for name in orders}
    controls = summary['AA']['hits'] + summary['BB']['hits']
    cross = summary['AB']['hits'] + summary['BA']['hits']
    if not controls:
        verdict = 'inconclusive: neither route showed same-route cache hits'
    elif cross == 0:
        verdict = 'separate caches: each route caches for itself but never reads the other\'s'
    elif cross / max(1, summary['AB']['n'] + summary['BA']['n']) >= 0.5 * controls / max(
            1, summary['AA']['n'] + summary['BB']['n']):
        verdict = 'shared cache: one route reads prompts the other wrote, about as often as its own'
    else:
        verdict = 'partly shared: some cross-route hits, fewer than same-route hits'
    results = dict(a=a.label, b=b.label, trials=trials, chars=chars, gap_s=gap_s, cells=summary, verdict=verdict,
                   detail=cells)
    md = '\n'.join(['# Cache sharing between routes', '',
                    f'A = **{a.label}**, B = **{b.label}**. Fresh ~{chars // 4}-token documents; second request '
                    f'{gap_s:g} s after the first. A hit = more than 64 cached tokens on the second request.', '',
                    report.table(['order', 'hits', 'median cached tokens'],
                                 [[f'{o[0]} writes, {o[1]} reads', f"{v['hits']}/{v['n']}", v['median_cached']]
                                  for o, v in summary.items()]), '', f'**Verdict:** {verdict}', '',
                    '_Same model via two providers + shared cache = one backend serving both tenants. '
                    'Same route with two different API keys + shared cache = cross-account cache sharing._'])
    return results, md


# ---- blocks -------------------------------------------------------------------------------
def plan_blocks(routes, lengths=(1000, 2000, 4000, 8000, 16000, 32000), **_):
    return dict(requests=2 * len(lengths))


def run_blocks(client, routes, lengths=(1000, 2000, 4000, 8000, 16000, 32000), gap_s=3.0, seed=0, salt=None, **_):
    salt = text.fresh_salt(salt)
    route = routes[0]
    generator = random.Random(seed)
    docs = [(chars, text.prose(chars, generator, salt)) for chars in lengths]

    def probe(chars, doc):
        client.chat(route, message(route, doc), max_tokens=4, tag=dict(chars=chars, step=0), key=f'blocks:{chars}:write')
        time.sleep(gap_s)
        read = client.chat(route, message(route, doc), max_tokens=4, tag=dict(chars=chars, step=1),
                           key=f'blocks:{chars}:read')
        return dict(chars=chars, prompt_tokens=read.prompt_tokens, cached=read.cached_tokens, write=read.cache_write_tokens)
    rows = [r for r in client.parallel([lambda d=d: probe(*d) for d in docs]) if r]
    cached = [r['cached'] for r in rows if r['cached']]
    block = reduce(math.gcd, cached) if len(cached) >= 2 else None
    minimum = min((r['prompt_tokens'] for r in rows if r['cached']), default=None)
    largest_miss = max((r['prompt_tokens'] or 0 for r in rows if not r['cached']), default=None)
    results = dict(route=route.label, rows=rows, block_tokens=block, smallest_cached_prompt=minimum,
                   largest_uncached_prompt=largest_miss)
    md = '\n'.join(['# Cache granularity', '', f'Route **{route.label}**.', '',
                    report.table(['document chars', 'prompt tokens', 'cached on repeat', 'cache writes'],
                                 [[r['chars'], r['prompt_tokens'], r['cached'], r['write']] for r in rows]), '',
                    f'- Common divisor of cached counts (block size candidate): **{block}**',
                    f'- Smallest prompt that was cached: {minimum}; largest that was not: {largest_miss}', '',
                    '_Block sizes from a gcd need several distinct cached counts; one-token chat-template offsets '
                    'can hide the block (look at the raw counts)._'])
    return results, md


# ---- ttl ----------------------------------------------------------------------------------
def plan_ttl(routes, delays=(30, 120, 300, 600), trials=1, **_):
    return dict(requests=2 * len(routes) * len(delays) * trials, duration_s=max(delays) + 30)


def run_ttl(client, routes, delays=(30, 120, 300, 600), trials=1, seed=0, salt=None, ttl_hint=None, **_):
    """Every route x delay x trial runs concurrently on one timeline (a multi-provider retention sweep)."""
    salt = text.fresh_salt(salt)
    generator = random.Random(seed)
    start = time.time() + 2
    jobs = []
    for route in routes:
        for t in range(trials):
            for delay in delays:
                jobs.append((route, delay, t, text.prose(24000, generator, salt), start + generator.uniform(0, 20)))

    def probe(route, delay, t, doc, when):
        time.sleep(max(0.0, when - time.time()))
        client.chat(route, message(route, doc, ttl=ttl_hint), max_tokens=4, tag=dict(delay=delay, trial=t, step=0),
                    key=f'ttl:{route.label}:{delay}:{t}:write')
        time.sleep(max(0.0, when + delay - time.time()))
        read = client.chat(route, message(route, doc, ttl=ttl_hint), max_tokens=4,
                           tag=dict(delay=delay, trial=t, step=1), key=f'ttl:{route.label}:{delay}:{t}:read')
        return hit(read)
    hits = defaultdict(lambda: defaultdict(list))
    outcomes = client.parallel([lambda j=j: probe(*j) for j in jobs], workers=max(1, len(jobs)))
    for (route, delay, *_), outcome in zip(jobs, outcomes):
        if outcome is not None:
            hits[route.label][delay].append(outcome)
    table = {label: [dict(delay_s=d, hits=sum(v), n=len(v)) for d, v in sorted(by.items())] for label, by in hits.items()}
    results = dict(routes=table, delays=list(delays), trials=trials)
    md = '\n'.join(['# Cache lifetime', '', f'{len(routes)} route(s), delays {list(delays)} s, {trials} trial(s); '
                    'each probe uses a fresh document; all probes share one timeline.', '',
                    report.table(['route'] + [f'{d} s' for d in delays],
                                 [[label] + [f"{r['hits']}/{r['n']}" for r in rows] for label, rows in table.items()]),
                    '', '_A miss can also mean the re-read landed on a different server; use --trials to separate '
                        'eviction from routing._'])
    return results, md


# ---- prefix -------------------------------------------------------------------------------
def plan_prefix(routes, trials=2, **_):
    return dict(requests=4 * 2 * trials)


def run_prefix(client, routes, trials=2, chars=24000, gap_s=3.0, seed=0, salt=None, **_):
    salt = text.fresh_salt(salt)
    route = routes[0]
    generator = random.Random(seed)
    jobs = []
    for t in range(trials):
        for test in ('repeat', 'diverge_half', 'extend', 'followup'):
            doc = text.prose(chars, generator, salt)
            first = message(route, doc)
            if test == 'repeat':
                second = message(route, doc)
            elif test == 'diverge_half':
                second = message(route, doc[:len(doc) // 2] + ' ' + text.prose(chars // 2, generator, salt))
            elif test == 'extend':
                second = message(route, doc, text.prose(chars // 2, generator, salt), bp_index=0)
                if breakpoints(route):  # no breakpoint on the new block: tests lookup without a new breakpoint
                    second[0]['content'][1].pop('cache_control', None)
            else:
                second = first + [dict(role='assistant', content='It is a reference document.'),
                                  dict(role='user', content='Name one thing it mentions.')]
            jobs.append((test, t, first, second))

    def probe(test, t, first, second):
        a = client.chat(route, first, max_tokens=8, tag=dict(test=test, trial=t, step=0), key=f'prefix:{test}:{t}:first')
        time.sleep(gap_s)
        b = client.chat(route, second, max_tokens=8, tag=dict(test=test, trial=t, step=1), key=f'prefix:{test}:{t}:second')
        share = (b.cached_tokens or 0) / a.prompt_tokens if a.prompt_tokens else None
        return dict(first_prompt=a.prompt_tokens, second_prompt=b.prompt_tokens, cached=b.cached_tokens,
                    share_of_first=share)
    rows = defaultdict(list)
    for (test, *_), row in zip(jobs, client.parallel([lambda j=j: probe(*j) for j in jobs])):
        if row is not None:
            rows[test].append(row)
    results = dict(route=route.label, tests=rows)
    expected = dict(repeat='whole prompt', diverge_half='about half (any-prefix cache) or 0 (breakpoint-only)',
                    extend='the first document (any-prefix) or 0 (needs a breakpoint at the end)',
                    followup='the previous prompt')
    md = '\n'.join(['# Cache resume semantics', '', f'Route **{route.label}**.', '',
                    report.table(['test', 'cached / first prompt', 'what each design predicts'],
                                 [[k, ', '.join(report.pct(r['share_of_first']) for r in v), expected[k]]
                                  for k, v in rows.items()]), '',
                    '_Block-paged prefix caches reuse any shared prefix; breakpoint caches (Anthropic, and OpenAI '
                    'from GPT-5.6 in our tests) resume only at boundaries where an earlier request ended or a '
                    'breakpoint was written._'])
    return results, md
