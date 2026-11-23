"""Do routes honour the output cap, and what are you billed for?

Each route gets a long-output prompt with a small cap, sent as max_tokens (plain and streamed)
and as max_completion_tokens. Reported completion and reasoning tokens and cost are compared
with the cap; for OpenRouter routes the generation record (OpenRouter's billing view) is
fetched too. Over-cap completions mean the cap was ignored; over-cap reasoning with a capped
visible answer means reasoning is excluded from the cap but still billed.
"""
from collections import defaultdict

from .. import report

PROMPT = 'Write an original 600-word story about a cartographer who maps a city that keeps changing.'
VARIANTS = [('max_tokens', False), ('max_tokens', True), ('max_completion_tokens', False)]


def plan(routes, repeats=2, **_):
    return dict(requests=len(routes) * len(VARIANTS) * repeats)


def run(client, routes, cap=64, repeats=2, **_):
    jobs = [(route, field, stream, repeat) for repeat in range(repeats) for route in routes
            for field, stream in VARIANTS]

    def one(route, field, stream, repeat):
        result = client.chat(route, [dict(role='user', content=PROMPT)], max_tokens=cap, cap_field=field, stream=stream,
                             tag=dict(field=field, stream=stream, repeat=repeat),
                             key=f'bill:{route.label}:{field}:{int(stream)}:{repeat}')
        record = client.generation(route, result.response_id) if result.ok else None
        billed = (record or {}).get('native_tokens_completion') or result.completion_tokens
        reasoning = (record or {}).get('native_tokens_reasoning') or result.reasoning_tokens or 0
        return dict(field=field, stream=stream, ok=result.ok, completion=billed, reasoning=reasoning,
                    visible=(billed or 0) - (reasoning or 0), cost=(record or {}).get('total_cost') or result.cost_usd,
                    finish=result.finish_reason)
    rows = defaultdict(list)
    for (route, *_), row in zip(jobs, client.parallel([lambda j=j: one(*j) for j in jobs])):
        if row is not None:
            rows[route.label].append(row)
    verdicts = {}
    for label, items in rows.items():
        ok = [r for r in items if r['ok'] and r['completion'] is not None]
        if not ok:
            verdicts[label] = 'no successful responses'
        elif all(r['completion'] <= cap for r in ok):
            verdicts[label] = 'honoured: never billed more than the cap'
        elif all(r['visible'] <= cap + 4 for r in ok):
            verdicts[label] = 'reasoning excluded from the cap: visible output capped, extra reasoning billed'
        else:
            verdicts[label] = 'cap ignored: completions billed beyond the cap'
    results = dict(cap=cap, rows=rows, verdicts=verdicts)
    table_rows = [[label, r['field'] + (' (stream)' if r['stream'] else ''), r['completion'], r['reasoning'],
                   r['finish'], f"${r['cost']:.6f}" if r['cost'] else '—'] for label, items in rows.items() for r in items]
    md = '\n'.join(['# Output cap compliance and billing', '', f'Cap: **{cap}** tokens; prompt asks for ~800.', '',
                    report.table(['route', 'cap sent as', 'billed completion tokens', 'of which reasoning', 'finish',
                                  'cost'], table_rows), '', '## Verdicts', '',
                    *[f'- **{k}**: {v}' for k, v in verdicts.items()]])
    return results, md
