"""Cache isolation policy probe: who can read a prompt cache someone else wrote? (paid, capped)

Each trial writes a fresh prompt (random nonce first, wikitext prose) with one identity and
re-sends it 4 s later with another; a cache read on the second request means the cache is
shared across that identity boundary. Variants:
  same          same key, same `user`              (control: does this route cache at all?)
  other_user    same key, different `user` field   (per-end-user scoping?)
  no_user       first with `user`, then without
  other_cache_key  same user, different OpenAI `prompt_cache_key` (OpenAI targets only)
  other_account second request from a different OpenRouter account (needs --second-key-env)
Upstream providers see all OpenRouter traffic as OpenRouter; a cross-account hit means
OpenRouter customers share that provider's cache. Only our own prompts are involved.
Run: HF_HUB_OFFLINE=1 python experiments/cache_scope_probe_v1.py OUT [--cap 0.5]
         [--second-key-env OPENROUTER_API_KEY_B]
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
TARGETS = [  # label, model, provider, breakpoint (Anthropic), openai
    ('claude-haiku-5.5', 'anthropic/claude-haiku-5.5', 'anthropic', True, False),
    ('gpt-5.4-nano', 'openai/gpt-5.4-nano', 'openai', False, True),
    ('gpt-6-luna', 'openai/gpt-6-luna', 'openai', False, True),
    ('deepseek-v4-flash@deepinfra', 'deepseek/deepseek-v4-flash', 'deepinfra/fp8', False, False),
    ('llama70b@coreweave', 'meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16', False, False),
    ('llama70b@parasail', 'meta-llama/llama-3.3-70b-instruct', 'parasail/fp8', False, False),
    ('llama70b@akashml', 'meta-llama/llama-3.3-70b-instruct', 'akashml/fp8', False, False),
    ('gemma4-26b@parasail', 'google/gemma-4-26b-a4b-it', 'parasail/bf16', False, False),
]
REPEATS = 2
GAP_S = 4
CHARS = 24000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=0.5)
    parser.add_argument('--second-key-env')
    args = parser.parse_args()
    keys = dict(a=os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required'))
    if args.second_key_env:
        keys['b'] = os.environ.get(args.second_key_env) or sys.exit(f'{args.second_key_env} not set')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    out.mkdir(parents=True)
    data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    paragraphs = [r['text'].strip() for r in data if len(r['text']) > 400 and not r['text'].strip().startswith('=')]
    generator = random.Random(20261116)

    def fresh():
        parts, total = [f'Reference document {generator.getrandbits(64):016x}.'], 0
        while total < CHARS:
            parts.append(generator.choice(paragraphs))
            total += len(parts[-1])
        return '\n\n'.join(parts)

    variants = ['same', 'other_user', 'no_user', 'other_cache_key'] + (['other_account'] if 'b' in keys else [])
    trials = [(t, v, r) for t in TARGETS for v in variants for r in range(REPEATS) if v != 'other_cache_key' or t[4]]
    generator.shuffle(trials)
    (out / 'config.json').write_text(json.dumps(dict(
        started_at=datetime.now(timezone.utc).isoformat(), targets=TARGETS, variants=variants, repeats=REPEATS,
        gap_s=GAP_S, chars=CHARS, cap_usd=args.cap, accounts=sorted(keys),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=1))
    first_user = 'kremlinology-probe-alice'
    plans = dict(same=[('a', first_user, 'k1'), ('a', first_user, 'k1')],
                 other_user=[('a', first_user, 'k1'), ('a', 'kremlinology-probe-bob', 'k1')],
                 no_user=[('a', first_user, 'k1'), ('a', None, 'k1')],
                 other_cache_key=[('a', first_user, 'k1'), ('a', first_user, 'k2')],
                 other_account=[('a', first_user, 'k1'), ('b', first_user, 'k1')])
    spent, stop, rows = 0.0, None, []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for number, ((label, model, provider, breakpoint, openai), variant, repeat) in enumerate(trials):
            if spent >= args.cap:
                stop = 'cap_reached'
                break
            block = dict(type='text', text=fresh())
            if breakpoint:
                block['cache_control'] = dict(type='ephemeral')
            messages = [dict(role='user', content=[block, dict(type='text', text='Summarize the document in one sentence.')])]
            for step, (account, user, cache_key) in enumerate(plans[variant]):
                if step:
                    time.sleep(GAP_S)
                body = dict(model=model, max_tokens=8, temperature=0, messages=messages,
                            provider=dict(only=[provider], allow_fallbacks=False))
                if user:
                    body['user'] = user
                if openai:
                    body['prompt_cache_key'] = f'kremlinology-{number}-{cache_key}'
                sent = time.time()
                try:
                    response = client.post(URL, json=body, headers=dict(authorization='Bearer ' + keys[account]))
                    status, obj = response.status_code, response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    status, obj = None, dict(error=type(exc).__name__)
                usage = obj.get('usage') if isinstance(obj, dict) else None
                if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                    spent += usage['cost']
                rows.append(dict(trial=number, label=label, variant=variant, repeat=repeat, step=step, account=account,
                                 user=user, cache_key=cache_key if openai else None, status=status,
                                 elapsed_s=time.time() - sent, usage=usage,
                                 generation_id=obj.get('id') if isinstance(obj, dict) else None,
                                 error=str(obj.get('error'))[:300] if isinstance(obj, dict) and obj.get('error') else None))
            cells = [(r['account'], r['status'], ((r['usage'] or {}).get('prompt_tokens_details') or {}).get('cached_tokens'))
                     for r in rows[-2:]]
            print(number, label, variant, cells, flush=True)
    (out / 'requests.jsonl').write_text(''.join(redact(json.dumps(r).encode(), list(keys.values())).decode() + '\n'
                                                for r in rows))
    summary = dict(stop=stop or 'completed', spent_usd=spent, requests=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(schema_version=1, files=files), indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
