"""Is reasoning counted against max_tokens? Kimi K2.5 on every OpenRouter provider (2026-11-19; paid, small, witnessed).

Found in cache_refresh_probe_v1: Novita returned and billed up to 81,921 completion tokens (81,920 of them reasoning)
for max_tokens=1. Here, per provider, two ordinary prompts x 5 repeats under four settings: default reasoning with
max_tokens 1 and 64; reasoning disabled with max_tokens 1; and an explicit reasoning budget (reasoning.max_tokens=64,
max_tokens 1). Recorded: completion and reasoning tokens billed, visible content length, finish reason, cost, time.
All requests go through a local witnessing proxy (KREMLINOLOGY_OPENROUTER_URL, `kremlin proxy --upstream
https://openrouter.ai`); this script's own log is joined to the session evidence by generation id.
Run: KREMLINOLOGY_OPENROUTER_URL=http://127.0.0.1:PORT/api/v1/chat/completions python experiments/reasoning_cap_probe_v1.py OUT
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
MODEL = 'moonshotai/kimi-k2.5'
PROVIDERS = ['novita', 'atlas-cloud/int4', 'siliconflow/int4', 'amazon-bedrock/us-east-2']
PROMPTS = {
    'arith': 'What is 17 times 23? Answer with just the number.',
    'summary': ('Summarize in one sentence: The Hanseatic League was a medieval commercial and defensive network of '
                'merchant guilds and market towns in Central and Northern Europe. It grew from a few North German towns '
                'in the late 12th century to dominate Baltic maritime trade for three centuries.'),
}
SETTINGS = {
    'default_max1': dict(max_tokens=1),
    'default_max64': dict(max_tokens=64),
    'reasoning_off_max1': dict(max_tokens=1, reasoning=dict(enabled=False)),
    'reasoning_budget64_max1': dict(max_tokens=1, reasoning=dict(max_tokens=64)),
}
REPEATS = 5


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    jobs = [(p, pr, s, k) for p in PROVIDERS for pr in PROMPTS for s in SETTINGS for k in range(REPEATS)]
    client = httpx.Client(timeout=1800)

    def run(job):
        provider, prompt, setting, k = job
        body = dict(model=MODEL, messages=[dict(role='user', content=PROMPTS[prompt])], temperature=0,
                    usage=dict(include=True), provider=dict(only=[provider], allow_fallbacks=False), **SETTINGS[setting])
        t = time.time()
        try:
            r = client.post(URL, json=body, headers={'Authorization': f'Bearer {key}'})
            j = r.json() if r.status_code == 200 else {}
            status, err = r.status_code, (r.text[:200] if r.status_code != 200 else None)
        except httpx.HTTPError as e:
            j, status, err = {}, None, type(e).__name__
        u = j.get('usage') or {}
        ch = (j.get('choices') or [{}])[0]
        msg = ch.get('message') or {}
        return dict(provider=provider, prompt=prompt, setting=setting, repeat=k, status=status, error=err,
                    generation=j.get('id'), served=j.get('provider'), elapsed_s=time.time() - t,
                    completion_tokens=u.get('completion_tokens'),
                    reasoning_tokens=(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                    cost=u.get('cost'), finish=ch.get('finish_reason'), content=(msg.get('content') or '')[:200],
                    reasoning_chars=len(msg.get('reasoning') or ''))

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(run, jobs))
    client.close()
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(dict(requests=len(rows), spent=sum(r['cost'] or 0 for r in rows))))


if __name__ == '__main__':
    main()
