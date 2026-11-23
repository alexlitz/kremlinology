"""Measure prefix-cache retention on OpenRouter endpoints (paid, small, capped).

Each trial sends a unique ~2,000-token prompt (random nonce first, so it can only hit its
own cache) twice: request A, then request B after a delay d. Cached prompt tokens reported
on B, compared with the prompt length and with the shared chat-template header, show
whether the prefix survived d. Very short delays (~2 s) separate eviction from routing to a
replica without the cache. All trials are interleaved over about an hour, one request at a
time, pinned to one provider endpoint with fallbacks disabled; spend stops at a hard cap.
Raw responses, timing, safe headers and usage are retained with a manifest.
Run: python experiments/cache_ttl_probe_v1.py OUT [--cap 0.5]
"""
import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import SAFE_RESPONSE_HEADERS, redact  # noqa: E402

ENDPOINTS = [
    ('meta-llama/llama-3.3-70b-instruct', 'akashml/fp8'), ('meta-llama/llama-3.3-70b-instruct', 'parasail/fp8'),
    ('meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16'),
    ('deepseek/deepseek-v3.2', 'gmicloud/fp8'), ('deepseek/deepseek-v3.2', 'deepinfra/fp4'),
    ('deepseek/deepseek-v3.2', 'baidu/fp8'),
    ('openai/gpt-oss-120b', 'groq'), ('openai/gpt-oss-120b', 'nebius/fp4'), ('openai/gpt-oss-120b', 'siliconflow/fp8'),
    ('google/gemma-4-31b-it', 'chutes/fp4'), ('google/gemma-4-31b-it', 'deepinfra/turbo'),
    ('qwen/qwen3.5-397b-a17b', 'deepinfra/fp8'), ('moonshotai/kimi-k2.5', 'novita'),
]
DELAYS = (2, 60, 300, 900, 1800, 3600)
REPLICATES = 2
URL = 'https://openrouter.ai/api/v1/chat/completions'


def filler(generator, words=1500):
    vocabulary = ('system cache token latency memory block prefix model request provider window layer state '
                  'attention vector matrix budget schedule replica storage compute network kernel batch').split()
    return ' '.join(generator.choice(vocabulary) for _ in range(words))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=0.5)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    (out / 'captures').mkdir(parents=True)
    generator = random.Random(20261111)
    trials, events = [], []
    start = time.time() + 5
    for model, provider in ENDPOINTS:
        for delay in DELAYS:
            for replicate in range(REPLICATES):
                trial = dict(id=len(trials), model=model, provider=provider, delay=delay, replicate=replicate,
                             nonce=f'{generator.getrandbits(64):016x}')
                trial['prompt'] = (f"Reference {trial['nonce']}. Read the notes below, then reply with OK.\n\n"
                                   + filler(generator))
                # Spread A sends over the first ~5 minutes so the hour-long delays finish around 65 minutes.
                first = start + generator.uniform(0, 300)
                trials.append(trial)
                events += [(first, trial['id'], 'A'), (first + delay, trial['id'], 'B')]
    events.sort()
    config = dict(started_at=datetime.now(timezone.utc).isoformat(), endpoints=ENDPOINTS, delays=DELAYS,
                  replicates=REPLICATES, cap_usd=args.cap, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  trials=[{k: v for k, v in t.items() if k != 'prompt'} | dict(prompt_sha256=hashlib.sha256(
                      t['prompt'].encode()).hexdigest(), prompt_chars=len(t['prompt'])) for t in trials])
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    spent, stop = 0.0, None
    with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
        for count, (when, trial_id, role) in enumerate(events):
            wait = when - time.time()
            if wait > 0:
                time.sleep(wait)
            if spent >= args.cap:
                stop = 'cap_reached'
                break
            trial = trials[trial_id]
            body = dict(model=trial['model'], max_tokens=1, temperature=0,
                        messages=[dict(role='user', content=trial['prompt'])],
                        provider=dict(only=[trial['provider']], allow_fallbacks=False))
            sent = time.time()
            try:
                response = client.post(URL, json=body, headers=dict(authorization='Bearer ' + key))
                status, raw = response.status_code, response.content
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_RESPONSE_HEADERS}
            except httpx.HTTPError as error:
                status, raw, headers = None, b'', dict(error=type(error).__name__)
            elapsed = time.time() - sent
            safe = redact(raw, [key])
            name = f'{count:05d}'
            (out / 'captures' / f'{name}.json').write_text(json.dumps(dict(
                body_base64=base64.b64encode(safe).decode(), sha256=hashlib.sha256(safe).hexdigest())))
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {}
            usage = parsed.get('usage') if isinstance(parsed, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                spent += usage['cost']
            record = dict(call=name, trial=trial_id, role=role, model=trial['model'], provider=trial['provider'],
                          delay=trial['delay'], replicate=trial['replicate'], scheduled=when, sent=sent,
                          sent_at=datetime.fromtimestamp(sent, timezone.utc).isoformat(), elapsed_s=elapsed,
                          status=status, headers=headers, usage=usage,
                          response_id=parsed.get('id') if isinstance(parsed, dict) else None,
                          reported_provider=parsed.get('provider') if isinstance(parsed, dict) else None,
                          system_fingerprint=parsed.get('system_fingerprint') if isinstance(parsed, dict) else None)
            with (out / 'requests.jsonl').open('a') as handle:
                handle.write(redact(json.dumps(record).encode(), [key]).decode() + '\n')
            (out / 'progress.json').write_text(json.dumps(dict(done=count + 1, of=len(events), spent_usd=spent)))
    summary = dict(stop=stop or 'completed', spent_usd=spent, events=len(events),
                   finished_at=datetime.now(timezone.utc).isoformat())
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(schema_version=1, files=files), indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
