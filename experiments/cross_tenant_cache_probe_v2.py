"""Cross-tenant prefix caching through a reseller: Venice -> Parasail (paid, small, capped).

If Venice relays to Parasail, Parasail sees Venice as one customer and OpenRouter (our direct
pinned requests) as another. A fresh prompt sent first through Venice and then directly to
Parasail is a cache hit on the second request only if Parasail's prefix cache is shared
across its customers (and Venice really sent it there). Orders per trial:
  V>P, P>V   cross-tenant tests
  P>P, V>V   same-route controls (does each route cache at all for this model?)
Only our own fresh prompts (random nonce first, wikitext prose) are used; nothing tries to
read anyone else's cache. Cache reads come from OpenRouter usage fields; upstream IDs and
latency from OpenRouter generation records fetched afterwards.
Run: HF_HUB_OFFLINE=1 python experiments/cross_tenant_cache_probe_v2.py OUT [--cap 0.5]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import redact  # noqa: E402

URL = 'https://openrouter.ai/api/v1/chat/completions'
GEN = 'https://openrouter.ai/api/v1/generation'
MODELS = [  # model, venice tag, parasail tag
    ('google/gemma-4-26b-a4b-it', 'venice/bf16', 'parasail/bf16'),
    ('z-ai/glm-5.3-flash', 'venice', 'parasail/fp4'),
    ('minimax/minimax-m3', 'venice/fp8', 'parasail/fp8'),
    ('moonshotai/kimi-k2.6', 'venice/int4', 'parasail/int4'),
]
ORDERS = ['VP', 'PV', 'PP', 'VV']
REPEATS = 17  # v2: top-up to 20 per cell with v1
GAP_S = 4
CHARS = 24000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=0.5)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    out.mkdir(parents=True)
    data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    paragraphs = [r['text'].strip() for r in data if len(r['text']) > 400 and not r['text'].strip().startswith('=')]
    generator = random.Random(20261180)

    def fresh():
        parts, total = [f'Reference document {generator.getrandbits(64):016x}.'], 0
        while total < CHARS:
            parts.append(generator.choice(paragraphs))
            total += len(parts[-1])
        return '\n\n'.join(parts)

    trials = [(model, tags, order, r) for model, *tags in MODELS for order in ORDERS for r in range(REPEATS)]
    generator.shuffle(trials)
    (out / 'config.json').write_text(json.dumps(dict(
        started_at=datetime.now(timezone.utc).isoformat(), models=MODELS, orders=ORDERS, repeats=REPEATS,
        gap_s=GAP_S, chars=CHARS, cap_usd=args.cap,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1))
    headers = dict(authorization='Bearer ' + key)
    spent, stop, rows = 0.0, None, []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for number, (model, (venice, parasail), order, repeat) in enumerate(trials):
            if spent >= args.cap:
                stop = 'cap_reached'
                break
            text = fresh()
            messages = [dict(role='user', content=text + '\n\nSummarize the document above in one sentence.')]
            for step, route in enumerate(order):
                if step:
                    time.sleep(GAP_S)
                provider = venice if route == 'V' else parasail
                body = dict(model=model, max_tokens=8, temperature=0, messages=messages,
                            provider=dict(only=[provider], allow_fallbacks=False))
                sent = time.time()
                try:
                    response = client.post(URL, json=body, headers=headers)
                    status, obj = response.status_code, response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    status, obj = None, dict(error=type(exc).__name__)
                usage = obj.get('usage') if isinstance(obj, dict) else None
                if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                    spent += usage['cost']
                rows.append(dict(trial=number, model=model, order=order, repeat=repeat, step=step, route=route,
                                 provider=provider, status=status, elapsed_s=time.time() - sent,
                                 generation_id=obj.get('id') if isinstance(obj, dict) else None,
                                 reported_provider=obj.get('provider') if isinstance(obj, dict) else None,
                                 usage=usage, error=str(obj.get('error'))[:300] if isinstance(obj, dict)
                                 and obj.get('error') else None,
                                 prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
            print(number, model, order, [(r['route'], r['status'], ((r['usage'] or {}).get('prompt_tokens_details')
                                           or {}).get('cached_tokens')) for r in rows[-len(order):]], flush=True)
    time.sleep(20)
    with httpx.Client(timeout=60, trust_env=False) as client:
        for row in rows:
            if not row['generation_id']:
                continue
            for _ in range(4):
                response = client.get(GEN, params=dict(id=row['generation_id']), headers=headers)
                if response.status_code == 200:
                    s = response.json().get('data') or {}
                    row['generation'] = {k: s.get(k) for k in ('upstream_id', 'provider_name', 'latency',
                                                               'generation_time', 'native_tokens_prompt',
                                                               'native_tokens_cached', 'total_cost')}
                    break
                time.sleep(5)
    (out / 'requests.jsonl').write_text(''.join(redact(json.dumps(r).encode(), [key]).decode() + '\n' for r in rows))
    summary = dict(stop=stop or 'completed', spent_usd=spent, requests=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(schema_version=1, files=files), indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
