"""Reconcile a benchmark run against OpenRouter generation stats (free lookups, no inference).

For each successful OpenRouter response ID in a run, GET /api/v1/generation and
keep an allowlist of fields: final cost, native token counts, upstream provider
and endpoint IDs, latency and response-cache source. Account, workspace,
application and user identifiers are dropped. Writes derived/
generation_stats.json bound to the run manifest. Requires OPENROUTER_API_KEY;
the key is never written or printed. Stats are OpenRouter's records, not
independent billing or execution attestations.
Usage: python tools/reconcile_openrouter_generations.py RUN_DIR [--interval 0.2]
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kremlinology.bench.runner import write_derived  # noqa: E402

KEEP = ('id', 'created_at', 'model', 'streamed', 'cancelled', 'latency', 'generation_time',
        'moderation_latency', 'tokens_prompt', 'tokens_completion', 'native_tokens_prompt',
        'native_tokens_completion', 'native_tokens_reasoning', 'native_tokens_cached', 'finish_reason',
        'native_finish_reason', 'service_tier', 'usage', 'total_cost', 'cache_discount',
        'upstream_inference_cost', 'provider_name', 'upstream_id', 'request_id', 'router',
        'response_cache_source_id', 'data_region', 'is_byok')
KEEP_RESPONSE = ('endpoint_id', 'id', 'latency', 'model_permaslug', 'provider_name', 'status', 'is_byok')
URL = 'https://openrouter.ai/api/v1/generation'


def response_ids(run, row):
    if row.get('response_ids'):
        return row['response_ids']
    capture = Path(run) / 'captures' / f"{row['call_id']}.json"
    if not capture.is_file():
        return []
    try:
        body = json.loads(base64.b64decode(json.loads(capture.read_text())['body_base64']))
    except (ValueError, KeyError):
        return []
    return [body['id']] if isinstance(body.get('id'), str) else []


def fetch(client, key, generation, retries=4):
    for attempt in range(retries + 1):
        response = client.get(URL, params=dict(id=generation), headers=dict(authorization='Bearer ' + key))
        if response.status_code not in (404, 429, 502, 503, 504) or attempt == retries:
            return response
        time.sleep(2 ** attempt)  # stats can lag briefly; rate limits back off
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--interval', type=float, default=0.2)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise SystemExit('OPENROUTER_API_KEY required')
    rows = [json.loads(line) for line in (Path(args.run) / 'observations.jsonl').read_text().splitlines()]
    records, failures = [], 0
    with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
        for row in rows:
            if row['adapter'] != 'openrouter' or row['status'] != 'success':
                continue
            for generation in response_ids(args.run, row):
                response = fetch(client, key, generation)
                record = dict(call_id=row['call_id'], target_id=row['target_id'], generation_id=generation,
                              http_status=response.status_code)
                if response.status_code == 200:
                    data = response.json().get('data') or {}
                    stats = {k: data.get(k) for k in KEEP}
                    stats['provider_responses'] = [{k: r.get(k) for k in KEEP_RESPONSE}
                                                   for r in data.get('provider_responses') or []]
                    record['stats'] = stats
                    observed = row.get('accounted_cost_usd')
                    if isinstance(observed, (int, float)) and isinstance(stats['total_cost'], (int, float)):
                        record['cost_difference_usd'] = stats['total_cost'] - observed
                else:
                    failures += 1
                records.append(record)
                time.sleep(args.interval)
    ok = [r for r in records if 'stats' in r]
    by_target = {}
    for r in ok:
        t = by_target.setdefault(r['target_id'], dict(generations=0, total_cost=0.0, providers={}, endpoints={},
                                                     response_cache_hits=0, cancelled=0,
                                                     max_abs_cost_difference=0.0))
        s = r['stats']
        t['generations'] += 1
        t['total_cost'] += s['total_cost'] or 0
        t['providers'][str(s['provider_name'])] = t['providers'].get(str(s['provider_name']), 0) + 1
        for p in s['provider_responses']:
            t['endpoints'][str(p['endpoint_id'])] = t['endpoints'].get(str(p['endpoint_id']), 0) + 1
        t['response_cache_hits'] += s['response_cache_source_id'] is not None
        t['cancelled'] += bool(s['cancelled'])
        t['max_abs_cost_difference'] = max(t['max_abs_cost_difference'], abs(r.get('cost_difference_usd') or 0))
    result = dict(lookups=len(records), failures=failures, by_target=by_target, records=records,
                  scope='OpenRouter generation records; identifiers outside an allowlist removed. '
                        'Not independent billing or execution attestation.')
    path = write_derived(args.run, 'generation_stats', result, Path(__file__))
    print(json.dumps(dict(output=str(path), lookups=len(records), failures=failures, by_target=by_target),
                     indent=2))


if __name__ == '__main__':
    main()
