"""Probe prompt-caching semantics: breakpoint caches (Anthropic) vs automatic prefix caches.

Paid, small, capped. Every test uses fresh random text (unique nonce first) so trials cannot
hit each other's caches. Cache reads and writes come from usage fields
(prompt_tokens_details.cached_tokens / cache_write_tokens) returned by OpenRouter.

Tests (each a short sequence of requests to one pinned model/provider):
  no_breakpoint_repeat   A twice, no cache_control
  breakpoint_repeat      A[bp] twice
  diverge_half           A[bp], then first half of A + new second half [bp]
  diverge_half_auto      same as diverge_half without any cache_control
  extend_with_bp         A[bp], then A + B[bp]
  extend_without_bp      A[bp], then A + B (no cache_control)
  edit_before_bp         A B[bp], then A' B[bp] (A' edited)
  edit_after_bp          A[bp] B, then A[bp] B' (B' edited)
  min_length_<n>         A_n[bp] twice for n in 1k..8k tokens
  ttl_<seconds>_<ttl>    A[bp] then repeat after a delay, default or 1h TTL (Anthropic only)
Run: python experiments/cache_semantics_probe_v1.py OUT [--cap 2.0]
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

URL = 'https://openrouter.ai/api/v1/chat/completions'
TARGETS = [  # (label, model, provider slug or None, uses breakpoints, full test set)
    ('claude-haiku-5.5', 'anthropic/claude-haiku-5.5', 'anthropic', True, True),
    ('claude-haiku-4.5', 'anthropic/claude-haiku-4.5', 'anthropic', True, False),
    ('gpt-6-luna', 'openai/gpt-6-luna', 'openai', False, True),
    ('gpt-5-nano', 'openai/gpt-5-nano', 'openai', False, False),
    ('deepseek-v4-flash', 'deepseek/deepseek-v4-flash', 'deepinfra/fp8', False, True),
    ('claude-haiku-5.5@vertex', 'anthropic/claude-haiku-5.5', 'google-vertex/global', True, False),
]
WORDS = ('river stone lantern harbor quiet signal maple orbit velvet canyon ember thread mosaic copper '
         'meadow cipher glacier willow beacon tundra prism falcon saffron quarry nectar basalt').split()


def text(generator, words):
    return ' '.join(generator.choice(WORDS) for _ in range(words))


def block(content, breakpoint=False, ttl=None):
    item = dict(type='text', text=content)
    if breakpoint:
        item['cache_control'] = dict(type='ephemeral', **({'ttl': ttl} if ttl else {}))
    return item


def plan(generator, label, breakpoints, full):
    """List of (test, step, delay_before_s, content blocks)."""
    def fresh(words=6000):
        return f'Document {generator.getrandbits(64):016x}. ' + text(generator, words)
    tests = []
    bp = breakpoints  # automatic-cache providers get identical content without cache_control
    a = fresh()
    tests.append(('no_breakpoint_repeat', [(0, [block(a)]), (3, [block(a)])]))
    a = fresh()
    tests.append(('breakpoint_repeat', [(0, [block(a, bp)]), (3, [block(a, bp)])]))
    a = fresh()
    half = a[:len(a) // 2]
    tests.append(('diverge_half', [(0, [block(a, bp)]), (3, [block(half + ' ' + text(generator, 3000), bp)])]))
    if full:
        a = fresh()
        half = a[:len(a) // 2]
        tests.append(('diverge_half_auto', [(0, [block(a)]), (3, [block(half + ' ' + text(generator, 3000))])]))
        a, b = fresh(), text(generator, 3000)
        tests.append(('extend_with_bp', [(0, [block(a, bp)]), (3, [block(a), block(b, bp)])]))
        a, b = fresh(), text(generator, 3000)
        tests.append(('extend_without_bp', [(0, [block(a, bp)]), (3, [block(a), block(b)])]))
        a, b = fresh(), text(generator, 3000)
        edited = a[:-200] + ' ' + text(generator, 40)
        tests.append(('edit_before_bp', [(0, [block(a), block(b, bp)]), (3, [block(edited), block(b, bp)])]))
        a, b = fresh(), text(generator, 3000)
        tests.append(('edit_after_bp', [(0, [block(a, bp), block(b)]), (3, [block(a, bp), block(text(generator, 3000))])]))
    for words in (800, 1600, 3200, 6400):
        a = fresh(words)
        tests.append((f'min_length_{words}w', [(0, [block(a, bp)]), (3, [block(a, bp)])]))
    if breakpoints and full:
        for delay, ttl in ((240, None), (420, None), (420, '1h')):
            a = fresh()
            tests.append((f'ttl_{delay}s_{ttl or "default"}', [(0, [block(a, True, ttl)]), (delay, [block(a, True, ttl)])]))
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--cap', type=float, default=2.0)
    args = parser.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    if out.exists():
        sys.exit('new output directory required')
    (out / 'captures').mkdir(parents=True)
    generator = random.Random(20261113)
    events, start = [], time.time() + 3
    for label, model, provider, breakpoints, full in TARGETS:
        for test, steps in plan(generator, label, breakpoints, full):
            when = start + generator.uniform(0, 120)  # spread test starts over two minutes
            for step, (delay, blocks) in enumerate(steps):
                when += delay
                events.append(dict(when=when, label=label, model=model, provider=provider, test=test, step=step,
                                   blocks=blocks))
    events.sort(key=lambda e: e['when'])
    (out / 'config.json').write_text(json.dumps(dict(
        started_at=datetime.now(timezone.utc).isoformat(), targets=TARGETS, cap_usd=args.cap,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        events=[{k: v for k, v in e.items() if k != 'blocks'} | dict(
            block_sha256=[hashlib.sha256(b['text'].encode()).hexdigest() for b in e['blocks']],
            block_chars=[len(b['text']) for b in e['blocks']],
            cache_control=[b.get('cache_control') for b in e['blocks']]) for e in events]), indent=1))
    spent, stop = 0.0, None
    with httpx.Client(timeout=120, follow_redirects=False, trust_env=False) as client:
        for count, event in enumerate(events):
            wait = event['when'] - time.time()
            if wait > 0:
                time.sleep(wait)
            if spent >= args.cap:
                stop = 'cap_reached'
                break
            body = dict(model=event['model'], max_tokens=16, temperature=0,
                        messages=[dict(role='user', content=event['blocks'] + [block('Reply with OK.')])])
            if event['provider']:
                body['provider'] = dict(only=[event['provider']], allow_fallbacks=False)
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
            record = {k: v for k, v in event.items() if k != 'blocks'} | dict(
                call=name, sent=sent, elapsed_s=time.time() - sent, status=status, headers=headers, usage=usage,
                response_id=parsed.get('id') if isinstance(parsed, dict) else None,
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
