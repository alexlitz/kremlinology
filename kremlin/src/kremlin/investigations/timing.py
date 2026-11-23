"""Decode speed by content: the speculative-decoding signature, and memory-bandwidth bounds.

Without speculation, decode speed should not depend on how predictable the text is. Each route
streams four prompts (copy a given text, count to 300, two different stories) at temperature 0,
interleaved, several times. Rate = (completion tokens - 1) / (last - first token-bearing event).
Copy/story and count/story speed ratios with bootstrap CIs are the signature; story2/story is
the control and should be about 1. With --active-params, story speed x active-weight bytes is a
lower bound on the memory bandwidth behind one stream (fp8 and bf16 shown).
Witnessed runs time records at the relay (honest timing the client cannot fake).
"""
import random
import statistics
from collections import defaultdict

from .. import report, stats

COPY = ('ember quiet mosaic stone lantern willow harbor thread beacon glacier maple copper orbit canyon cipher '
        'meadow tundra signal velvet river ') * 6
PROBES = dict(
    copy='Copy the following text exactly, word for word, with no other output:\n\n' + COPY.strip(),
    count='Write the integers from 1 to 300 separated by single spaces, with no other output.',
    story='Write an original short story about a lighthouse keeper who finds an unusual object. Do not use lists.',
    story2='Write an original short story about a night-shift baker who receives a strange order. Do not use lists.')


def plan(routes, repeats=6, max_tokens=256, **_):
    return dict(requests=len(routes) * len(PROBES) * repeats, max_output_tokens=len(routes) * len(PROBES) * repeats
                * max_tokens)


def run(client, routes, repeats=6, max_tokens=256, active_params=None, seed=0, timing_concurrency=1, **_):
    """Sequential by default: concurrent requests to one provider can change its batching and therefore the
    decode speed being measured. --timing-concurrency raises it."""
    generator = random.Random(seed)
    jobs = [(route, probe, r) for r in range(repeats) for route in routes for probe in PROBES]
    generator.shuffle(jobs)
    def one(route, probe, r):
        return client.chat(route, [dict(role='user', content=PROBES[probe])], max_tokens=max_tokens, stream=True,
                           tag=dict(probe=probe, repeat=r), key=f'time:{route.label}:{probe}:{r}')
    rates = defaultdict(lambda: defaultdict(list))
    results = client.parallel([lambda j=j: one(*j) for j in jobs], workers=timing_concurrency)
    for (route, probe, _), result in zip(jobs, results):
        if result is None:
            continue
        times = result.token_times_s
        if result.ok and result.completion_tokens and len(times) >= 3 and times[-1] > times[0]:
            rates[route.label][probe].append((result.completion_tokens - 1) / (times[-1] - times[0]))
    rows = {}
    for label, probes in rates.items():
        row = {f'{p}_tps': statistics.median(v) for p, v in probes.items() if v}
        for p in ('copy', 'count', 'story2'):
            ci = stats.bootstrap_ratio_ci(probes.get(p, []), probes.get('story', []))
            if ci:
                row[f'{p}_over_story'] = ci
        if active_params and 'story_tps' in row:
            row['min_bandwidth_tb_s'] = {name: row['story_tps'] * active_params * b / 1e12
                                         for name, b in (('fp8', 1), ('bf16', 2))}
        rows[label] = row
    results = dict(repeats=repeats, max_tokens=max_tokens, active_params=active_params, routes=rows)

    def ratio(row, key):
        v = row.get(key)
        return f'{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]' if v else '—'
    md = '\n'.join(['# Decode speed and speculation', '',
                    f'{repeats} repeats per prompt, temperature 0, up to {max_tokens} tokens, streamed.', '',
                    report.table(['route', 'story tok/s', 'copy/story', 'count/story', 'story2/story (control)',
                                  'min bandwidth fp8 / bf16'],
                                 [[k, f"{v.get('story_tps', 0):.0f}", ratio(v, 'copy_over_story'),
                                   ratio(v, 'count_over_story'), ratio(v, 'story2_over_story'),
                                   ' / '.join(f'{x:.1f} TB/s' for x in v['min_bandwidth_tb_s'].values())
                                   if 'min_bandwidth_tb_s' in v else '—'] for k, v in rows.items()]), '',
                    '_A copy/count speed-up whose CI excludes 1, with the story2 control near 1, points to '
                    'speculative or lookahead decoding. Reasoning models: rates include reasoning tokens. SRAM '
                    'hardware (Groq, Cerebras) breaks the HBM bandwidth bound._'])
    return results, md
