"""Turn an accepted run's results into uniform finding rows for the hub's collated pages.

Every row: kind, model, subjects (route labels / providers involved), metric (short text), value (number or
None), verdict (short text), flag ('notable' | 'normal' | 'inconclusive'). Rows carry run_id when stored.
Extraction only reads results.json produced by the investigation; it never re-measures anything.
"""
from . import routes as routes_mod

VERSION = 2  # bump when extraction changes: the hub recomputes findings for runs stored with an older version

KINDS = {
    'same-backend': 'Same backend (reseller detection)',
    'output-difference': 'Output differences vs a reference',
    'later-tokens': 'Output differences at later positions',
    'reconstructed-difference': 'Differences without logprobs (reconstructed)',
    'cache-sharing': 'Cache sharing between routes / accounts',
    'cache-granularity': 'Cache block size and minimum length',
    'cache-lifetime': 'Cache lifetime',
    'cache-semantics': 'Cache resume semantics',
    'billing': 'Output cap and billing',
    'parameters': 'Parameter honouring',
    'sampler': 'Sampler behaviour',
    'speculation': 'Decode speed and speculation',
    'served-by': 'Who served it (witnessed sessions)',
}


def describe_routes(specs):
    """label -> dict(model, provider, host) from the run's route specs."""
    out = {}
    for spec in specs:
        try:
            route = routes_mod.parse_route(spec)
        except ValueError:
            continue
        host = route.base_url.split('//')[-1].split('/')[0]
        out[route.label] = dict(model=route.model, provider=route.provider or host, kind=route.kind)
    return out


def _row(kind, model, subjects, metric, value, verdict, flag):
    return dict(kind=kind, model=model, subjects=subjects, metric=metric, value=value, verdict=verdict, flag=flag)


def extract(command, results, specs):
    """Finding rows for one run (empty if the run produced no results)."""
    if not results:
        return []
    routes = describe_routes(specs)

    def model(label):
        return (routes.get(label) or {}).get('model', '?')

    def who(label):
        info = routes.get(label) or {}
        return info.get('provider') or label
    rows = []
    if command == 'session':
        return session_rows(results)
    if command == 'fingerprint':
        for pair, v in results.get('pairs', {}).items():
            a, b = pair.split(' ~ ')
            cross = v['cross']
            verdict = v['verdict'].split(':')[0]
            flag = 'notable' if verdict == 'same numerics' else 'inconclusive' if verdict.startswith('inconclusive') \
                else 'normal'
            rows.append(_row('same-backend', model(a), [who(a), who(b)],
                             f"identical {cross['successes']}/{cross['n']} (self {results['routes'][a]['self_match']['fraction'] or 0:.0%} / "
                             f"{results['routes'][b]['self_match']['fraction'] or 0:.0%})", cross['fraction'], verdict, flag))
    elif command == 'tv':
        ref = results['reference']
        floor = (results.get('reference_self') or {}).get('mean_upper')
        for label, a in results.get('routes', {}).items():
            if not a:
                continue
            notable = floor is not None and a['ci_lower'] > floor
            rows.append(_row('output-difference', model(label), [who(label), f'ref: {who(ref)}'],
                             f"TV {a['mean_lower']:.1%}–{a['mean_upper']:.1%} (95% {a['ci_lower']:.1%}–{a['ci_upper']:.1%})"
                             + (f'; reference self {floor:.1%}' if floor is not None else ''), a['mean_lower'],
                             'above the noise floor' if notable else 'within or near the noise floor',
                             'notable' if notable else 'normal'))
    elif command == 'later':
        ref = results['reference']
        for label, positions in results.get('routes', {}).items():
            if label == ref:
                continue
            for t, v in positions.items():
                a = v['interval']
                if not a:
                    rows.append(_row('later-tokens', model(label), [who(label), f'ref: {who(ref)}'],
                                     f'position {t}: no valid cells ({v["invalid"]} invalid)', None,
                                     'prefill not honoured or no data', 'inconclusive'))
                    continue
                rows.append(_row('later-tokens', model(label), [who(label), f'ref: {who(ref)}'],
                                 f"position {t}: TV {a['mean_lower']:.1%}–{a['mean_upper']:.1%} (n={v['valid']})",
                                 a['mean_lower'], 'differs' if a['ci_lower'] > 0.02 else 'close',
                                 'notable' if a['ci_lower'] > 0.02 else 'normal'))
    elif command == 'reconstruct':
        for pair, v in results.get('pairs', {}).items():
            a, b = pair.split(' ~ ')
            rows.append(_row('reconstructed-difference', model(a), [who(a), who(b)],
                             f"TV {v['mean_tv']:.3f} (bootstrap {v['ci'][0]:.3f}–{v['ci'][1]:.3f}, {v['prompts']} prompts)",
                             v['mean_tv'], 'compare with a same-route control', 'normal'))
        for label, v in results.get('routes', {}).items():
            if v.get('top_k_ignored_prompts'):
                rows.append(_row('sampler', model(label), [who(label)], 'more tokens than top_k allows', None,
                                 'top_k ignored', 'notable'))
    elif command == 'cache-share':
        verdict = results['verdict'].split(':')[0]
        a, b = results['a'], results['b']
        rows.append(_row('cache-sharing', model(a), [who(a), who(b)],
                         ', '.join(f"{k} {v['hits']}/{v['n']}" for k, v in results['cells'].items()), None, verdict,
                         'notable' if verdict in ('shared cache', 'partly shared') else
                         'inconclusive' if verdict == 'inconclusive' else 'normal'))
    elif command == 'cache-blocks':
        rows.append(_row('cache-granularity', model(results['route']), [who(results['route'])],
                         f"block {results['block_tokens']} tokens; smallest cached prompt {results['smallest_cached_prompt']}",
                         results['block_tokens'], 'measured', 'normal'))
    elif command == 'cache-ttl':
        for label, rows_ in results.get('routes', {}).items():
            text = ', '.join(f"{r['delay_s']}s {r['hits']}/{r['n']}" for r in rows_)
            alive = [r['delay_s'] for r in rows_ if r['hits']]
            rows.append(_row('cache-lifetime', model(label), [who(label)], text, max(alive) if alive else 0,
                             f'hits up to {max(alive)} s' if alive else 'no hits', 'normal'))
    elif command == 'cache-prefix':
        label = results['route']
        tests = results.get('tests', {})

        def share(name):
            values = [r['share_of_first'] for r in tests.get(name, []) if r['share_of_first'] is not None]
            return sum(values) / len(values) if values else None
        half = share('diverge_half')
        verdict = 'breakpoint-only (no partial reuse)' if half is not None and half < 0.05 else \
            'any-prefix reuse' if half is not None else 'no data'
        rows.append(_row('cache-semantics', model(label), [who(label)],
                         ', '.join(f'{k} {share(k):.0%}' for k in tests if share(k) is not None), half, verdict,
                         'notable' if verdict.startswith('breakpoint') else 'normal'))
    elif command == 'billing':
        for label, verdict in results.get('verdicts', {}).items():
            billed = max((r['completion'] or 0) for r in results['rows'][label]) if results['rows'].get(label) else None
            rows.append(_row('billing', model(label), [who(label)], f"cap {results['cap']}, max billed {billed}",
                             billed, verdict.split(':')[0], 'normal' if verdict.startswith('honoured') else 'notable'))
    elif command == 'params':
        for label, checks in results.get('checks', {}).items():
            failed = [k for k, v in checks.items() if v[0] == 'fail']
            rows.append(_row('parameters', model(label), [who(label)],
                             ', '.join(f'{k}: {v[0]}' for k, v in checks.items()), len(failed),
                             ('fails: ' + ', '.join(failed)) if failed else 'all honoured or unsupported',
                             'notable' if failed else 'normal'))
    elif command == 'sampler':
        for label, v in results.get('routes', {}).items():
            if 'error' in v:
                continue
            violated = [k for k in ('top_k_1', 'top_k_2', 'top_p') if v.get(k, {}).get('violations')]
            gof = (v.get('t1') or {}).get('gof') or {}
            rows.append(_row('sampler', model(label), [who(label)],
                             f"top_k violations {[v.get(k, {}).get('violations') for k in ('top_k_1', 'top_k_2')]}, "
                             f"T=1 fit p={gof.get('p_value', float('nan')):.2g}, "
                             f"T=0.5 fitted {(v.get('t05') or {}).get('fitted_temperature') or float('nan'):.2f}",
                             None, ('violates ' + ', '.join(violated)) if violated else 'consistent',
                             'notable' if violated else 'normal'))
    elif command == 'timing':
        for label, v in results.get('routes', {}).items():
            ratio = v.get('count_over_story') or v.get('copy_over_story')
            control = v.get('story2_over_story')
            speculates = bool(ratio and ratio[1] > 1 and (not control or control[1] <= 1 <= control[2]))
            rows.append(_row('speculation', model(label), [who(label)],
                             f"story {v.get('story_tps', 0):.0f} tok/s; count/story "
                             f"{ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}]" if ratio else
                             f"story {v.get('story_tps', 0):.0f} tok/s", v.get('story_tps'),
                             'speculation signature' if speculates else 'no signature',
                             'notable' if speculates else 'normal'))
    return rows


def session_rows(results):
    """One row per (host, model, serving provider) in a witnessed session: what the API said served the calls, how
    fast (witnessed time to first byte), and whether the served model differs from the requested one."""
    import statistics
    from urllib.parse import urlsplit
    groups = {}
    for call in results.get('calls', []):
        if not call.get('witnessed') or not 200 <= (call.get('status') or 0) < 300:
            continue  # only answered calls say who served them
        model = call.get('model_served') or call.get('model_requested')
        if not model:
            continue  # not an inference call (model lists, files, ...)
        key = (urlsplit(call['url']).hostname, model, call.get('provider') or '')
        groups.setdefault(key, []).append(call)
    rows = []
    for (host, model, provider), calls in sorted(groups.items()):
        latencies = [c['latency_s'] for c in calls if c.get('latency_s') is not None]
        statuses = {}
        for c in calls:
            statuses[str(c['status'])] = statuses.get(str(c['status']), 0) + 1
        requested = {c.get('model_requested') for c in calls if c.get('model_requested')}
        swapped = sorted(r for r in requested if not (model.startswith(r) or r.startswith(model)))
        median = statistics.median(latencies) if latencies else None
        rows.append(_row('served-by', model, [host] + ([provider] if provider else []),
                         f"{len(calls)} call{'s' if len(calls) != 1 else ''}, "
                         f"statuses {', '.join(f'{k}×{v}' for k, v in sorted(statuses.items()))}"
                         + (f', median time to first byte {median:.2f} s' if median is not None else ''),
                         median, (f'served {model} for requested {", ".join(swapped)}' if swapped else
                                  f'served by {provider}' if provider else 'provider not reported'),
                         'notable' if swapped else 'normal'))
    return rows
