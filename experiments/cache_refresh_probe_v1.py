"""Does hitting a cache keep it alive, and does a partial hit do it too? (paid, capped; 2026-11-18)

Per endpoint, three arms, TRIALS each, every trial with its own unique prompt (random nonce first):
  control:  write the prompt at t=0, probe with the same prompt at D;
  full:     write at 0, re-send the same prompt at PING (< usual lifetime), probe at D;
  partial:  write at 0, send only the first 25% of the prompt plus new text at PING (a partial hit), probe at D.
D is past the endpoint's usual lifetime (from the retention runs), so the control mostly misses. The probe's cached
token count says what survived: the whole prompt (any hit refreshes the entry), about the first 25% (blocks are
refreshed one by one, least-recently-used per block) or nothing. Anthropic only caches at breakpoints, so its prompt has
two breakpoints (50% and 100%) and its partial ping re-sends just the first half, ending at the 50% breakpoint.
A ping that misses writes the entry again, which would also make the probe hit, so the analysis conditions on
whether the ping itself hit. Requests go direct to OpenRouter (pinned endpoint, no fallbacks), max_tokens=1; every
response is kept (key redacted) with timing and usage. Writes are spread over SPREAD seconds per endpoint so our own
requests don't queue behind each other.
Run: python experiments/cache_refresh_probe_v1.py OUT [--cap 4.0] [--trials 20]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import SAFE_RESPONSE_HEADERS, redact  # noqa: E402

# KREMLINOLOGY_OPENROUTER_URL=http://127.0.0.1:PORT/api/v1/chat/completions sends everything through a local
# `kremlin proxy --upstream https://openrouter.ai` (witnessed; evidence in the proxy's --out directory).
URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
# (model, provider, ping_s, probe_s, words). Delays from the retention / cache-semantics / age runs:
ENDPOINTS = [
    ('anthropic/claude-haiku-4.5', 'anthropic', 240, 420, 9000),      # hits to 5.5 min, misses at 7: 4 + 5 > 7
    ('openai/gpt-5.4-nano', 'openai', 450, 900, 5000),                # hits at >= 7 min; docs: 5-10 min of inactivity
    ('moonshotai/kimi-k2.6', 'moonshotai', 300, 600, 5000),           # reported hits 18/20 at 5 min, 0/40 at 10
    ('meta-llama/llama-3.3-70b-instruct', 'coreweave/fp16', 1800, 3600, 5000),  # 15/20 at 30 min, 0/20 at 60
    ('openai/gpt-oss-120b', 'nebius/fp4', 600, 1800, 5000),           # 0/20 at 30 min
    ('deepseek/deepseek-v3.2', 'deepinfra/fp4', 60, 600, 5000),       # 13/20 at 1 min, ~20% at 5-15 min
    ('deepseek/deepseek-v3.2', 'gmicloud/fp8', 30, 300, 5000),        # 19/20 at 2 s, 8/19 at 1 min, 4/20 at 5 min
    ('google/gemma-4-31b-it', 'chutes/fp4', 30, 300, 5000),           # 20/20 at 2 s, 12/20 at 1 min, 5/20 at 5 min
    ('moonshotai/kimi-k2.5', 'novita', 20, 300, 5000),                # 15/20 at 2 s, 7/20 at 1 min, 2/20 at 5 min
]
ARMS = ('control', 'full', 'partial')
SPREAD = 900
REASONING_OFF = os.environ.get('KREMLINOLOGY_REASONING_OFF', '1') == '1'  # off by default after v1
VOCAB = ('system cache token latency memory block prefix model request provider window layer state attention vector '
         'matrix budget schedule replica storage compute network kernel batch').split()


def words(generator, n):
    return ' '.join(generator.choice(VOCAB) for _ in range(n))


def messages(trial, part):
    """part: 'whole' or 'partial'. Anthropic: two breakpoint blocks; others: one user message."""
    ask = 'Reply with OK.'
    if trial['provider'] == 'anthropic':
        first = dict(type='text', text=trial['text_a'], cache_control=dict(type='ephemeral'))
        second = dict(type='text', text=trial['text_b'], cache_control=dict(type='ephemeral'))
        blocks = [first, second] if part == 'whole' else [first]
        return [dict(role='user', content=blocks + [dict(type='text', text=ask)])]
    if part == 'whole':
        return [dict(role='user', content=trial['text'] + '\n\n' + ask)]
    return [dict(role='user', content=trial['head'] + ' ' + trial['tail'] + '\n\n' + ask)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=4.0)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    (out / 'captures').mkdir(parents=True)
    generator = random.Random(20261184)
    trials, events = [], []
    start = time.time() + 5
    for model, provider, ping, probe, n_words in ENDPOINTS:
        for arm in ARMS:
            for replicate in range(args.trials):
                nonce = f'{generator.getrandbits(64):016x}'
                body = words(generator, n_words)
                t = dict(id=len(trials), model=model, provider=provider, arm=arm, replicate=replicate, ping_s=ping,
                         probe_s=probe, nonce=nonce)
                text = f'Reference {nonce}. Read the notes below.\n\n' + body
                if provider == 'anthropic':
                    cut = len(text) // 2
                    t['text_a'], t['text_b'] = text[:cut], text[cut:]
                else:
                    cut = len(text) // 4
                    t['text'], t['head'], t['tail'] = text, text[:cut], words(generator, 100)
                first = start + generator.uniform(0, SPREAD)
                t['done'] = {r: threading.Event() for r in ('write', 'ping')}
                trials.append(t)
                events.append((first, t['id'], 'write'))
                if arm != 'control':
                    events.append((first + ping, t['id'], 'ping'))
                events.append((first + probe, t['id'], 'probe'))
    events.sort()
    config = dict(started_at=datetime.now(timezone.utc).isoformat(), endpoints=ENDPOINTS, arms=ARMS, trials=args.trials,
                  spread_s=SPREAD, cap_usd=args.cap, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  trials_index=[dict(id=t['id'], model=t['model'], provider=t['provider'], arm=t['arm'],
                                     replicate=t['replicate'], ping_s=t['ping_s'], probe_s=t['probe_s'], nonce=t['nonce'])
                                for t in trials])
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    state = dict(spent=0.0, stop=None, count=0)
    lock = threading.Lock()
    client = httpx.Client(timeout=120, follow_redirects=False, trust_env=False)

    def send(trial, role):
        if role == 'ping':
            trial['done']['write'].wait(1200)
        if role == 'probe':
            trial['done']['write'].wait(1200)
            if trial['arm'] != 'control':
                trial['done']['ping'].wait(1200)
        try:
            with lock:
                if state['spent'] >= args.cap:
                    state['stop'] = 'cap_reached'
                    return
                state['count'] += 1
                name = f"{state['count']:05d}"
            part = 'partial' if role == 'ping' and trial['arm'] == 'partial' else 'whole'
            body = dict(model=trial['model'], max_tokens=1, temperature=0, messages=messages(trial, part),
                        usage=dict(include=True), provider=dict(only=[trial['provider']], allow_fallbacks=False))
            if REASONING_OFF:  # v1 ran without this: Novita's Kimi K2.5 then ignored max_tokens (up to 81,921 tokens)
                body['reasoning'] = dict(enabled=False)
            sent = time.time()
            try:
                response = client.post(URL, json=body, headers=dict(authorization='Bearer ' + key))
                status, raw = response.status_code, response.content
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_RESPONSE_HEADERS}
            except httpx.HTTPError as error:
                status, raw, headers = None, b'', dict(error=type(error).__name__)
            elapsed = time.time() - sent
            safe = redact(raw, [key])
            (out / 'captures' / f'{name}.json').write_text(json.dumps(dict(
                body_sha256=hashlib.sha256(safe).hexdigest(), body=safe.decode('utf-8', 'replace'))))
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {}
            usage = parsed.get('usage') if isinstance(parsed, dict) else None
            with lock:
                if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                    state['spent'] += usage['cost']
                record = dict(call=name, trial=trial['id'], role=role, arm=trial['arm'], part=part, model=trial['model'],
                              provider=trial['provider'], ping_s=trial['ping_s'], probe_s=trial['probe_s'],
                              sent=sent, elapsed_s=elapsed, status=status, headers=headers, usage=usage,
                              response_id=parsed.get('id') if isinstance(parsed, dict) else None,
                              reported_provider=parsed.get('provider') if isinstance(parsed, dict) else None)
                with (out / 'requests.jsonl').open('a') as handle:
                    handle.write(redact(json.dumps(record).encode(), [key]).decode() + '\n')
                (out / 'progress.json').write_text(json.dumps(dict(done=state['count'], of=len(events),
                                                                   spent_usd=state['spent'])))
        finally:
            if role in trial['done']:
                trial['done'][role].set()

    t0 = time.time()
    with ThreadPoolExecutor(24) as pool:
        futures = []
        for when, trial_id, role in events:
            wait = when - time.time()
            if wait > 0:
                time.sleep(wait)
            futures.append(pool.submit(send, trials[trial_id], role))
        for f in futures:
            f.result()
    client.close()
    summary = dict(stop=state['stop'] or 'completed', spent_usd=state['spent'], events=len(events),
                   duration_s=time.time() - t0, finished_at=datetime.now(timezone.utc).isoformat())
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(schema_version=1, files=files), indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
