"""Analyse cache_routing_honesty_v1: routing stickiness and reported-vs-actual caching (witnessed TTFT).

TTFT per call = time of the first token-bearing SSE event minus the end of the request, both from the witness's
record timestamps (receipt), not the client clock. Joins the run log to the session evidence by generation id.
"""
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
import os as _os
DATA_ROOT = _os.environ.get('KREMLINOLOGY_DATA', 'data')  # where the release archives are extracted


def witnessed_ttft(record):
    receipt = record.get('witness_receipt') or {}
    start = receipt.get('request_complete_ms')
    for t_ms, data in (record.get('timings') or {}).get('witness_sse_events') or []:
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')):
                return (t_ms - start) / 1000 if start is not None else None
    return None


def generation_of(record, root):
    import base64
    body = base64.b64decode(json.loads((root / 'captures' / f"{record['capture']}.json").read_text())['body_base64'])
    for line in body.decode('utf-8', 'replace').splitlines():
        if line.startswith('data:') and '"id"' in line:
            try:
                return json.loads(line[5:]).get('id')
            except ValueError:
                pass
    return None


def sign_test(k, n):
    """One-sided binomial P(X >= k | n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def floor_test(cold, warm):
    """Warm trials faster than every cold trial: count m and P(the m fastest of all 2n are all warm) under
    exchangeability, C(n, m) / C(2n, m). Catches a bimodal hit/miss mix (hits on some replicas only) that a sign test
    on the median misses when queueing noise is large."""
    if not cold or not warm:
        return 0, 1.0
    m = sum(w < min(cold) for w in warm)
    n = len(cold)
    return m, math.comb(len(warm), m) / math.comb(len(warm) + n, m)


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [Path(DATA_ROOT, 'kremlinology-cache-honesty-v1')]
    out = dirs[0] if len(dirs) == 1 else Path(str(dirs[0]) + '-pooled')
    out.mkdir(exist_ok=True)
    rows, witnessed, spent, run_ids = [], 0, 0.0, []
    for k, d in enumerate(dirs):
        log = json.loads((d / 'log.json').read_text())
        root = d / 'session'
        ttft = {}
        for line in (root / 'requests.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record.get('witness_receipt'):
                witnessed += 1
            gen = generation_of(record, root) if record.get('status') == 200 else None
            if gen:
                ttft[gen] = witnessed_ttft(record)
        for r in log['rows']:
            r = dict(r, ttft_s=ttft.get(r['generation']))
            if r['tag'] and r['tag'].startswith(('honest:', 'route:')):  # keep trials distinct across runs
                parts = r['tag'].split(':')
                parts[2] = f'{k}-{parts[2]}' if parts[0] == 'honest' else parts[2]
                if parts[0] == 'route':
                    parts[1] = f'{k}-{parts[1]}'
                r['tag'] = ':'.join(parts)
            rows.append(r)
        spent += log['spent']
        run_ids.append(log['run_id'])
    log = dict(run_id='+'.join(run_ids), spent=spent)
    report = dict(run_id=log['run_id'], spent_usd=round(log['spent'], 4), calls=len(rows), witnessed_records=witnessed)

    # A. routing
    groups = defaultdict(list)
    for r in rows:
        if r['tag'] and r['tag'].startswith('route:') and r['status'] == 200:
            groups[r['tag'].split(':')[1]].append(r)
    firsts = Counter(g[0]['served_by'] for g in groups.values() if g)
    repeats = [(g[0], r) for g in groups.values() for r in g[1:]]
    same = sum(1 for first, r in repeats if r['served_by'] == first['served_by'])
    # expected share if repeats were routed like fresh prompts: sum over providers of p(first)^2-ish; use the
    # empirical distribution of all unpinned calls
    all_served = Counter(r['served_by'] for g in groups.values() for r in g)
    total = sum(all_served.values())
    expected = sum((n / total) ** 2 for n in all_served.values())
    hits_same = [r['cached_tokens'] for first, r in repeats if r['served_by'] == first['served_by']]
    hits_other = [r['cached_tokens'] for first, r in repeats if r['served_by'] != first['served_by']]
    report['routing'] = dict(
        prompts=len(groups), repeats=len(repeats), providers_seen=dict(all_served.most_common()),
        first_request_providers=dict(firsts.most_common()),
        repeats_on_first_provider=same, share_on_first_provider=round(same / len(repeats), 3) if repeats else None,
        share_expected_without_stickiness=round(expected, 3),
        median_cached_tokens_same_provider=statistics.median(hits_same) if hits_same else None,
        median_cached_tokens_other_provider=statistics.median(hits_other) if hits_other else None,
        median_prompt_tokens=statistics.median(r['prompt_tokens'] for g in groups.values() for r in g),
        sequences=[[r['served_by'] + (f" ({r['cached_tokens']} cached)" if r['cached_tokens'] else '') for r in g]
                   for _, g in sorted(groups.items())])

    # B. honesty
    pairs = defaultdict(dict)
    for r in rows:
        if r['tag'] and r['tag'].startswith('honest:'):
            _, provider, trial, kind = r['tag'].split(':')
            pairs[(r['model'], provider)].setdefault(trial, {})[kind] = r
    table = []
    for (model, provider), trials in sorted(pairs.items()):
        cells = []
        for trial, pair in sorted(trials.items()):
            cold, warm = pair.get('cold'), pair.get('warm')
            if not cold or not warm or cold['status'] != 200 or warm['status'] != 200:
                continue
            if cold.get('ttft_s') is None or warm.get('ttft_s') is None:
                continue
            cells.append(dict(cold_ttft=cold['ttft_s'], warm_ttft=warm['ttft_s'],
                              cached_frac=warm['cached_tokens'] / warm['prompt_tokens'] if warm['prompt_tokens'] else 0,
                              cold_cached=cold['cached_tokens'], prompt_tokens=warm['prompt_tokens'],
                              cold_cost=cold['cost'], warm_cost=warm['cost']))
        if not cells:
            table.append(dict(model=model, provider=provider, n=0))
            continue
        speedup = statistics.median(c['cold_ttft'] / c['warm_ttft'] for c in cells if c['warm_ttft'] > 0)
        cached = statistics.median(c['cached_frac'] for c in cells)
        cost_ratio = statistics.median(c['warm_cost'] / c['cold_cost'] for c in cells if c['cold_cost'])
        for c in cells:
            c['faster'] = c['warm_ttft'] < 0.8 * c['cold_ttft']
            c['reported'] = c['cached_frac'] > 0.5
        faster = sum(c['faster'] for c in cells)
        reported = sum(c['reported'] for c in cells)
        both = sum(c['faster'] and c['reported'] for c in cells)
        reported_only = sum(c['reported'] and not c['faster'] for c in cells)
        hidden = sum(c['faster'] and not c['reported'] for c in cells)
        # Hidden caching is a speed-up on repeats that report no cache. With ~20 trials, a few "faster" trials happen
        # by noise, so require >= 8 unreported trials and either (a) warm faster than cold in significantly more than
        # half (one-sided sign test, p < 0.01) with a median cold/warm ratio >= 1.25, or (b) significantly many warm
        # trials faster than the fastest cold trial (floor test, p < 0.01): hits on some replicas only.
        unreported = [c for c in cells if not c['reported']]
        warm_wins = sum(c['warm_ttft'] < c['cold_ttft'] for c in unreported)
        hidden_p = sign_test(warm_wins, len(unreported))
        hidden_speedup = (statistics.median(c['cold_ttft'] / c['warm_ttft'] for c in unreported if c['warm_ttft'] > 0)
                          if unreported else None)
        floor_m, floor_p = floor_test([c['cold_ttft'] for c in unreported], [c['warm_ttft'] for c in unreported])
        if len(unreported) >= 8 and ((hidden_p < 0.01 and hidden_speedup >= 1.25) or floor_p < 0.01):
            verdict = 'faster when repeated, no cache reported'
        elif both and both >= reported_only:
            verdict = 'cache reported and used' + ('' if 2 * reported >= len(cells) else ' (intermittent)')
        elif reported >= 2:
            verdict = 'cache reported, speed-up not visible'
        else:
            verdict = 'no cache'
        table.append(dict(model=model, provider=provider, n=len(cells), prompt_tokens=cells[0]['prompt_tokens'],
                          cold_ttft_s=round(statistics.median(c['cold_ttft'] for c in cells), 3),
                          warm_ttft_s=round(statistics.median(c['warm_ttft'] for c in cells), 3),
                          speedup=round(speedup, 2), faster_trials=faster, cached_share=round(cached, 3),
                          reported_trials=reported, both_trials=both, hidden_trials=hidden,
                          warm_over_cold_cost=round(cost_ratio, 3), unreported_trials=len(unreported),
                          unreported_warm_faster=warm_wins, unreported_sign_p=round(hidden_p, 5),
                          unreported_below_cold_floor=floor_m, floor_p=round(floor_p, 6),
                          unreported_speedup=None if hidden_speedup is None else round(hidden_speedup, 2),
                          cells=cells, verdict=verdict))
    report['honesty'] = table
    (out / 'analysis.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ('honesty',)}, indent=1))
    for t in table:
        if t.get('n'):
            print(f"{t['model'].split('/')[1][:14]:14s} {t['provider']:22s} n={t['n']} prompt {t['prompt_tokens']:6d}  "
                  f"ttft cold {t['cold_ttft_s']:.2f} warm {t['warm_ttft_s']:.2f} x{t['speedup']:.2f} ({t['faster_trials']} faster)  "
                  f"unrep {t['unreported_warm_faster']}/{t['unreported_trials']} p={t['unreported_sign_p']:.3g} x{t['unreported_speedup'] or 0:.2f}  "
                  f"cached {t['cached_share']:.2f} ({t['reported_trials']} rep, {t['both_trials']} both, {t['hidden_trials']} hidden)  cost w/c {t['warm_over_cold_cost']:.2f}  -> {t['verdict']}")
        else:
            print(f"{t['model'].split('/')[1][:14]:14s} {t['provider']:22s} no usable pairs")


if __name__ == '__main__':
    main()
