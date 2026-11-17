"""Do providers honour the output cap, and are over-cap tokens billed? (paid, tiny, capped)

Each target gets a long-output prompt (an original story) with a cap of 64 tokens, sent as
max_tokens or max_completion_tokens, streamed or not, 3 repeats. Billing comes from
OpenRouter generation stats (native completion and reasoning tokens, total cost), fetched
after a short delay. Controls are other providers of the same models.
Run: python experiments/max_tokens_probe_v1.py OUT [--cap 0.10]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import redact  # noqa: E402

URL = 'https://openrouter.ai/api/v1/chat/completions'
GEN = 'https://openrouter.ai/api/v1/generation'
CAP_TOKENS = 64
TARGETS = [  # label, model, provider
    ('gptoss120b@siliconflow', 'openai/gpt-oss-120b', 'siliconflow/fp8'),
    ('gptoss120b@baseten', 'openai/gpt-oss-120b', 'baseten/fp4'),
    ('deepseekv32@siliconflow', 'deepseek/deepseek-v3.2', 'siliconflow/fp8'),
    ('deepseekv32@gmicloud', 'deepseek/deepseek-v3.2', 'gmicloud/fp8'),
    ('deepseekv32@deepinfra', 'deepseek/deepseek-v3.2', 'deepinfra/fp4'),
]
VARIANTS = [('max_tokens', False), ('max_tokens', True), ('max_completion_tokens', False)]
PROMPT = 'Write an original 600-word story about a cartographer who maps a city that keeps changing.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=0.10)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps(dict(
        started_at=datetime.now(timezone.utc).isoformat(), targets=TARGETS, variants=VARIANTS, prompt=PROMPT,
        cap_tokens=CAP_TOKENS, cap_usd=args.cap, repeats=args.repeats,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1))
    spent, stop, rows = 0.0, None, []
    headers = dict(authorization='Bearer ' + key)
    with httpx.Client(timeout=180, follow_redirects=False, trust_env=False) as client:
        for repeat in range(args.repeats):
            for label, model, provider in TARGETS:
                for field, stream in VARIANTS:
                    if spent >= args.cap:
                        stop = 'cap_reached'
                        break
                    body = {'model': model, 'temperature': 0, field: CAP_TOKENS, 'stream': stream,
                            'messages': [dict(role='user', content=PROMPT)],
                            'provider': dict(only=[provider], allow_fallbacks=False)}
                    if stream:
                        body['stream_options'] = dict(include_usage=True)
                    sent = time.time()
                    gen_id, usage, finish, text, error, status = None, None, None, '', None, None
                    try:
                        if stream:
                            with client.stream('POST', URL, json=body, headers=headers) as response:
                                status = response.status_code
                                for line in response.iter_lines():
                                    if not line.startswith('data: ') or line == 'data: [DONE]':
                                        continue
                                    try:
                                        obj = json.loads(line[6:])
                                    except ValueError:
                                        continue
                                    gen_id = obj.get('id') or gen_id
                                    usage = obj.get('usage') or usage
                                    error = obj.get('error') or error
                                    for choice in obj.get('choices') or []:
                                        text += (choice.get('delta') or {}).get('content') or ''
                                        finish = choice.get('finish_reason') or finish
                        else:
                            response = client.post(URL, json=body, headers=headers)
                            status = response.status_code
                            obj = response.json()
                            gen_id, usage, error = obj.get('id'), obj.get('usage'), obj.get('error')
                            if obj.get('choices'):
                                text = obj['choices'][0]['message'].get('content') or ''
                                finish = obj['choices'][0].get('finish_reason')
                    except (httpx.HTTPError, ValueError) as exc:
                        error = type(exc).__name__
                    if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                        spent += usage['cost']
                    rows.append(dict(label=label, provider=provider, field=field, stream=stream, repeat=repeat,
                                     status=status, generation_id=gen_id, usage=usage, finish_reason=finish,
                                     visible_chars=len(text), elapsed_s=time.time() - sent,
                                     error=str(error)[:300] if error else None))
    time.sleep(20)
    with httpx.Client(timeout=60, trust_env=False) as client:
        for row in rows:
            if not row['generation_id']:
                continue
            for _ in range(4):
                response = client.get(GEN, params=dict(id=row['generation_id']), headers=headers)
                if response.status_code == 200:
                    s = response.json().get('data') or {}
                    row['billed'] = {k: s.get(k) for k in ('native_tokens_completion', 'native_tokens_reasoning',
                                                           'tokens_completion', 'total_cost', 'finish_reason',
                                                           'native_finish_reason', 'provider_name')}
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
