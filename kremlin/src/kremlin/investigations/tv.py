"""How different are a route's outputs from a reference? Dataset-average first-token TV with bounds.

Every route (the first is the reference) answers each prompt twice with top-k logprobs. Per
prompt, partial-report TV bounds [L, U] compare a route with the reference (same repeat index,
averaged over repeats); the reference against itself gives the noise floor. Dataset averages
get distribution-free 95% intervals (Waudby-Smith & Ramdas betting). With the built-in prompts
this describes those prompts; pass --prompts FILE for a dataset you care about.
"""
import random

from .. import report, stats, text


def plan(routes, prompts=100, repeats=2, **_):
    return dict(requests=len(routes) * prompts * repeats, max_tokens_each=1)


def run(client, routes, prompts=100, repeats=2, top_logprobs=20, prompt_file=None, seed=0, error=0.0, extra=None,
        **_):
    items = text.prompts(prompts, seed, prompt_file)
    generator = random.Random(seed)
    calls = [(route, i, r) for r in range(repeats) for i in range(len(items)) for route in routes]
    generator.shuffle(calls)

    def ask(route, i, repeat):
        return client.chat(route, [dict(role='user', content=items[i])], max_tokens=1, top_logprobs=top_logprobs,
                           extra=extra, require_parameters=True, tag=dict(prompt=i, repeat=repeat),
                           key=f'tv:{route.label}:{i}:{repeat}')
    laws = {}
    for (route, i, repeat), result in zip(calls, client.parallel([lambda c=c: ask(*c) for c in calls])):
        if result is not None and result.ok and result.first_token_logprobs:
            laws[(route.label, i, repeat)] = result.first_token_logprobs
    reference = routes[0].label

    def per_prompt(label, compare_self=False):
        lowers, uppers = [], []
        for i in range(len(items)):
            if compare_self:
                pairs = [(laws.get((label, i, 0)), laws.get((label, i, 1)))]
            else:
                pairs = [(laws.get((label, i, r)), laws.get((reference, i, r))) for r in range(repeats)]
            bounds = [stats.partial_tv_bounds(a, b, error) for a, b in pairs if a and b]
            bounds = [b for b in bounds if b]
            if bounds:
                lowers.append(sum(b[0] for b in bounds) / len(bounds))
                uppers.append(sum(b[1] for b in bounds) / len(bounds))
        return stats.average_interval(lowers, uppers)
    results = dict(reference=reference, prompts=len(items), repeats=repeats, top_logprobs=top_logprobs, error=error,
                   reference_self=per_prompt(reference, True) if repeats >= 2 else None,
                   routes={r.label: per_prompt(r.label) for r in routes[1:]})
    return results, render(results)


def fmt(a):
    if not a:
        return '—', '—', '—'
    point = (f"{100 * a['mean_lower']:.1f}%" if abs(a['mean_upper'] - a['mean_lower']) < 0.002
             else f"{100 * a['mean_lower']:.1f}–{100 * a['mean_upper']:.1f}%")
    return point, f"[{100 * a['ci_lower']:.1f}, {100 * a['ci_upper']:.1f}]", a['prompts']


def render(results):
    rows = [[f"{results['reference']} vs itself", *fmt(results['reference_self'])]]
    rows += [[label, *fmt(a)] for label, a in results['routes'].items()]
    return '\n'.join([
        '# Output differences vs a reference', '',
        f"Reference: **{results['reference']}**. First token, top-{results['top_logprobs']} logprobs, "
        f"{results['prompts']} prompts x {results['repeats']} repeats, report-error allowance "
        f"{results['error']} nats.", '',
        report.table(['route', 'average TV', '95% CI', 'prompts'], rows), '',
        '_The reference-vs-itself row is the noise floor: differences at that level are indistinguishable from the '
        "reference's own nondeterminism. Causes (precision, kernels, templates) are not attributed._"])
