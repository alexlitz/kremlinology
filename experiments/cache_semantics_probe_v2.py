"""Follow-up cache-semantics probe with natural-prose filler (paid, small, capped).

v1's random word salad tripped content filters on some Claude requests, so this uses
wikitext-103 paragraphs (fresh nonce first). Adds a chat follow-up test: request 2 is a
second turn whose prompt contains request 1's whole prompt, then the assistant reply and a
new user message. A cache that resumes at previous request ends hits there even if it
cannot resume at arbitrary prefixes.
Run: HF_HUB_OFFLINE=1 python experiments/cache_semantics_probe_v2.py OUT [--cap 1.0]
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
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kremlinology.bench.adapters import SAFE_RESPONSE_HEADERS, redact  # noqa: E402

URL = 'https://openrouter.ai/api/v1/chat/completions'
TARGETS = [  # label, model, provider, breakpoints, tests
    ('claude-haiku-5.5', 'anthropic/claude-haiku-5.5', 'anthropic', True,
     ['extend_without_bp', 'edit_before_bp', 'diverge_half', 'followup', 'ttl_330', 'ttl_420']),
    ('gpt-6-luna', 'openai/gpt-6-luna', 'openai', False, ['repeat', 'diverge_half', 'followup', 'ttl_420']),
    ('gpt-6-sol', 'openai/gpt-6-sol', 'openai', False, ['repeat', 'diverge_half', 'followup']),
    ('gpt-5.6-luna', 'openai/gpt-5.6-luna', 'openai', False, ['repeat', 'diverge_half', 'followup']),
    ('gpt-5.4-nano', 'openai/gpt-5.4-nano', 'openai', False, ['repeat', 'diverge_half', 'followup']),
]


def corpus():
    data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    return [r['text'].strip() for r in data if len(r['text']) > 400 and not r['text'].strip().startswith('=')]


def block(content, breakpoint=False):
    item = dict(type='text', text=content)
    if breakpoint:
        item['cache_control'] = dict(type='ephemeral')
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=1.0)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    (out / 'captures').mkdir(parents=True)
    paragraphs = corpus()
    generator = random.Random(20261113)

    def prose(chars=30000):
        parts, total = [], 0
        while total < chars:
            parts.append(generator.choice(paragraphs))
            total += len(parts[-1])
        return ' '.join(parts)

    def fresh(chars=30000):
        return f'Reference document {generator.getrandbits(64):016x}.\n\n' + prose(chars)

    ask = 'Summarize the document above in one short sentence.'
    events, start = [], time.time() + 3
    for label, model, provider, bp, tests in TARGETS:
        for test in tests:
            a = fresh()
            msg = lambda blocks: [dict(role='user', content=blocks + [block(ask)])]  # noqa: E731
            if test == 'repeat':
                steps = [(0, msg([block(a)])), (3, msg([block(a)]))]
            elif test == 'diverge_half':
                steps = [(0, msg([block(a, bp)])), (3, msg([block(a[:len(a) // 2] + ' ' + prose(15000), bp)]))]
            elif test == 'extend_without_bp':
                steps = [(0, msg([block(a, True)])), (3, msg([block(a), block(prose(15000))]))]
            elif test == 'edit_before_bp':
                b = prose(15000)
                steps = [(0, msg([block(a), block(b, True)])), (3, msg([block(a[:-600] + ' ' + prose(400)), block(b, True)]))]
            elif test == 'followup':
                first = msg([block(a, bp)])
                second = first + [dict(role='assistant', content='It is a reference document.'),
                                  dict(role='user', content='Now name one topic it mentions.')]
                steps = [(0, first), (3, second)]
            elif test.startswith('ttl_'):
                delay = int(test.split('_')[1])
                steps = [(0, msg([block(a, bp)])), (delay, msg([block(a, bp)]))]
            when = start + generator.uniform(0, 90)
            for step, (delay, messages) in enumerate(steps):
                when += delay
                events.append(dict(when=when, label=label, model=model, provider=provider, test=test, step=step,
                                   messages=messages))
    events.sort(key=lambda e: e['when'])
    (out / 'config.json').write_text(json.dumps(dict(
        started_at=datetime.now(timezone.utc).isoformat(), targets=TARGETS, cap_usd=args.cap,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        events=[{k: v for k, v in e.items() if k != 'messages'} | dict(
            messages_sha256=hashlib.sha256(json.dumps(e['messages']).encode()).hexdigest()) for e in events]), indent=1))
    spent, stop = 0.0, None
    with httpx.Client(timeout=120, follow_redirects=False, trust_env=False) as client:
        for count, event in enumerate(events):
            wait = event['when'] - time.time()
            if wait > 0:
                time.sleep(wait)
            if spent >= args.cap:
                stop = 'cap_reached'
                break
            body = dict(model=event['model'], max_tokens=24, temperature=0, messages=event['messages'],
                        provider=dict(only=[event['provider']], allow_fallbacks=False))
            sent = time.time()
            try:
                response = client.post(URL, json=body, headers=dict(authorization='Bearer ' + key))
                status, raw = response.status_code, response.content
                headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in SAFE_RESPONSE_HEADERS}
            except httpx.HTTPError as error:
                status, raw, headers = None, b'', dict(error=type(error).__name__)
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
            finish = None
            if isinstance(parsed, dict) and parsed.get('choices'):
                finish = parsed['choices'][0].get('finish_reason')
            record = {k: v for k, v in event.items() if k != 'messages'} | dict(
                call=name, sent=sent, elapsed_s=time.time() - sent, status=status, headers=headers, usage=usage,
                finish_reason=finish, response_id=parsed.get('id') if isinstance(parsed, dict) else None,
                reported_provider=parsed.get('provider') if isinstance(parsed, dict) else None,
                error=parsed.get('error') if isinstance(parsed, dict) else None)
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
