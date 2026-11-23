"""Do routes honour the parameters they accept? Cheap behavioural checks (about 12 requests per route).

  logprobs      top_logprobs returned with the requested count, sorted, and consistent with the
                chosen token (its logprob must not exceed the reported top-1; GMI-style misalignment)
  n             n=3 returns three choices
  seed          same seed at T=1 -> same text; different seed -> different text (seed honoured)
  temperature0  two T=0 requests agree (determinism at temperature 0)
  stop          a stop sequence ends the output before it appears
  max_tokens    completion tokens never exceed the cap
  template      prompt tokens for the same message, compared across routes (template/tokenizer parity)
A check is 'unsupported' when the route rejects the parameter (HTTP 4xx), 'pass'/'fail' otherwise.
OpenRouter routes add require_parameters so unsupported parameters are rejected, not dropped.
"""
from collections import defaultdict

from .. import report

PROMPT = 'Write one sentence about a lighthouse.'
COUNT = 'Count from 1 to 20 separated by commas and spaces.'


def plan(routes, **_):
    return dict(requests=12 * len(routes))


def _call(client, route, content, key, **kw):
    extra = kw.pop('extra', {})
    return client.chat(route, [dict(role='user', content=content)], require_parameters=True,
                       extra=extra or None, key=key, **kw)


def _status(result):
    if result.ok:
        return None
    if result.status and 400 <= result.status < 500:
        return 'unsupported'
    return 'error'


def check_logprobs(client, route):
    r = _call(client, route, PROMPT, f'params:{route.label}:logprobs', max_tokens=1, top_logprobs=5)
    if _status(r):
        return (_status(r), f'HTTP {r.status}'), r.prompt_tokens
    if not r.first_token_logprobs:
        return ('fail', 'accepted but returned no logprobs'), r.prompt_tokens
    values = sorted(r.first_token_logprobs.values(), reverse=True)
    chosen = (r.raw or {}).get('choices', [{}])[0].get('logprobs', {}).get('content', [{}])[0].get('logprob')
    problems = []
    if len(values) != 5:
        problems.append(f'{len(values)} alternatives, asked for 5')
    if chosen is not None and chosen > values[0] + 1e-6:
        problems.append(f'chosen token logprob {chosen:.3f} above reported top-1 {values[0]:.3f}')
    if r.chosen_first and r.chosen_first not in r.first_token_logprobs and chosen is not None \
            and chosen > values[-1] + 1e-6:
        problems.append('chosen token missing from the top list despite a higher logprob')
    return ('fail' if problems else 'pass', '; '.join(problems) or 'consistent'), r.prompt_tokens


def check_n(client, route):
    r = _call(client, route, PROMPT, f'params:{route.label}:n', max_tokens=4, temperature=1.0, extra=dict(n=3))
    if _status(r):
        return _status(r), f'HTTP {r.status}'
    if len(r.choice_texts) == 3:
        return 'pass', '3 choices'
    return 'fail', f'silently ignored: {len(r.choice_texts)} choice(s) for n=3, no error'


def check_seed(client, route):
    texts, last = [], None
    for i, seed in enumerate((1234, 1234, 4321)):
        last = _call(client, route, PROMPT, f'params:{route.label}:seed:{i}', max_tokens=24, temperature=1.0,
                     extra=dict(seed=seed))
        texts.append(last.text if last.ok else None)
    if None in texts:
        return ('unsupported' if last.status and 400 <= last.status < 500 else 'error'), f'HTTP {last.status}'
    if texts[0] == texts[1] and texts[0] != texts[2]:
        return 'pass', 'same seed reproduces; different seed differs'
    if texts[0] != texts[1]:
        return 'fail', 'same seed gave different text (seed ignored or nondeterministic)'
    return 'inconclusive', 'different seeds gave the same text (low-entropy prompt?)'


def check_temperature0(client, route):
    a = _call(client, route, PROMPT, f'params:{route.label}:t0:0', max_tokens=24, temperature=0.0)
    b = _call(client, route, PROMPT, f'params:{route.label}:t0:1', max_tokens=24, temperature=0.0)
    if _status(a) or _status(b):
        return _status(a) or _status(b), 'request failed'
    return ('pass', 'identical') if a.text == b.text else ('fail', 'two T=0 requests differ')


def check_stop(client, route):
    r = _call(client, route, COUNT, f'params:{route.label}:stop', max_tokens=60, temperature=0.0, extra=dict(stop=['7']))
    if _status(r):
        return _status(r), f'HTTP {r.status}'
    if '7' in r.text:
        return 'fail', f'output contains the stop sequence: {r.text[:40]!r}'
    return 'pass', f'finish_reason={r.finish_reason}'


def check_max_tokens(client, route):
    r = _call(client, route, COUNT, f'params:{route.label}:max_tokens', max_tokens=5, temperature=0.0)
    if _status(r):
        return _status(r), f'HTTP {r.status}'
    used = r.completion_tokens
    return ('pass', f'{used} tokens') if used is not None and used <= 5 else (
        'fail', f'{used} completion tokens for max_tokens=5')


CHECKS = dict(n=check_n, seed=check_seed, temperature0=check_temperature0, stop=check_stop,
              max_tokens=check_max_tokens)


def run(client, routes, **_):
    jobs = [(route, 'logprobs') for route in routes] + [(route, name) for route in routes for name in CHECKS]

    def one(route, name):
        return check_logprobs(client, route) if name == 'logprobs' else CHECKS[name](client, route)
    checks, template = defaultdict(dict), {}
    for (route, name), outcome in zip(jobs, client.parallel([lambda j=j: one(*j) for j in jobs])):
        if outcome is None:
            continue
        if name == 'logprobs':
            checks[route.label]['logprobs'], tokens = outcome
            template[route.label] = (route.model, tokens)
        else:
            checks[route.label][name] = outcome
    by_model = defaultdict(dict)
    for label, (model, count) in template.items():
        if count is not None:
            by_model[model][label] = count
    parity = {model: ('all equal' if len(set(c.values())) <= 1 else 'differ') + ': ' + ', '.join(
        f'{k}={v}' for k, v in c.items()) for model, c in by_model.items() if len(c) >= 2}
    results = dict(checks={k: dict(v) for k, v in checks.items()},
                   prompt_tokens={k: v[1] for k, v in template.items()}, template_parity=parity)
    names = ['logprobs', 'n', 'seed', 'temperature0', 'stop', 'max_tokens']
    md = '\n'.join(['# Parameter behaviour', '',
                    report.table(['route'] + names, [[label] + [f'{v[name][0]}' for name in names]
                                                     for label, v in checks.items()]), '',
                    '## Details', '',
                    *[f'- **{label}** {name}: {v[name][0]} - {v[name][1]}' for label, v in checks.items() for name in names],
                    '', '## Template parity (same model, same message)', '',
                    *([f'- **{m}**: {v}' for m, v in parity.items()] or ['- (needs two routes serving the same model)']),
                    '', '_Different prompt-token counts for the same model mean different chat templates, injected system '
                        'prompts or tokenizers._'])
    return results, md
