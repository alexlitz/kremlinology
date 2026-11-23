"""Live end-to-end validation: every investigation and request type through a witness relay, audited and published.

Starts a local relay and hub, then for each case:
  1. runs it through the relay with request disclosure (real providers, small budgets);
  2. audits the evidence with the relay's public key, requiring every call to be witnessed;
  3. submits it to the hub and waits for acceptance.
Request-type cases (vision, audio input, image output, streaming, GET) check each receipt directly: signature,
disclosed request body equal to what was sent, response consistent. Also: a budget-stopped run is resumed.
Needs OPENROUTER_API_KEY. Spends a few cents. Usage: python tools/validate_live.py OUT_DIR [--only NAME ...]
"""
import argparse
import base64
import io
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
import wave
import zlib
from pathlib import Path

import httpx

BIN = Path(sys.executable).parent
QWEN = 'openrouter:qwen/qwen3.5-397b-a17b'
LLAMA = 'openrouter:meta-llama/llama-3.3-70b-instruct'
GEMMA = 'openrouter:google/gemma-4-26b-a4b-it'
NO_REASONING = '{"reasoning":{"enabled":false}}'

INVESTIGATIONS = [
    ('fingerprint', ['fingerprint', f'{QWEN}@parasail/fp8;label=parasail', f'{QWEN}@venice;label=venice',
                     '--prompts', '6', '--top-logprobs', '5', '--extra', NO_REASONING], 0.02),
    ('tv', ['tv', f'{LLAMA}@coreweave/fp16;label=coreweave', f'{LLAMA}@parasail/fp8;label=parasail',
            '--prompts', '8', '--top-logprobs', '20'], 0.02),
    ('cache-share', ['cache-share', f'{GEMMA}@parasail/bf16;label=parasail', f'{GEMMA}@venice/bf16;label=venice',
                     '--trials', '1', '--chars', '12000'], 0.05),
    ('cache-blocks', ['cache-blocks', 'openrouter:deepseek/deepseek-v4-flash@deepinfra/fp8;label=deepinfra'], 0.05),
    ('cache-prefix', ['cache-prefix', 'openrouter:anthropic/claude-haiku-5.5@anthropic;label=haiku',
                      '--trials', '1', '--chars', '12000'], 0.05),
    ('cache-ttl', ['cache-ttl', 'openrouter:anthropic/claude-haiku-5.5@anthropic;label=haiku',
                   'openrouter:openai/gpt-6-luna@openai;label=luna', '--delays', '20,60'], 0.05),
    ('billing', ['billing', 'openrouter:openai/gpt-oss-120b@baseten/fp4;label=baseten', '--repeats', '1'], 0.02),
    ('timing', ['timing', f'{LLAMA}@coreweave/fp16;label=coreweave', f'{LLAMA}@groq;label=groq',
                '--repeats', '3', '--max-tokens', '64'], 0.02),
    ('params', ['params', f'{LLAMA}@coreweave/fp16;label=coreweave', f'{LLAMA}@parasail/fp8;label=parasail'], 0.02),
    ('sampler', ['sampler', f'{LLAMA}@coreweave/fp16;label=coreweave', '--samples', '30'], 0.03),
    ('later', ['later', f'{LLAMA}@coreweave/fp16;label=coreweave', f'{LLAMA}@parasail/fp8;label=parasail',
               '--prompts', '5', '--positions', '1,4', '--top-logprobs', '20'], 0.02),
    ('reconstruct', ['reconstruct', f'{LLAMA}@coreweave/fp16;label=coreweave', f'{LLAMA}@parasail/fp8;label=parasail',
                     '--prompts', '2', '--samples', '40'], 0.03),
]


def png(width=64, height=64):
    """A small solid-colour PNG, no imaging library needed."""
    row = b'\x00' + bytes([200, 30, 30]) * width
    raw = zlib.compress(row * height)

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', raw) + chunk(b'IEND', b''))


def wav(seconds=1.0, rate=16000, hz=440):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * hz * i / rate)))
                                 for i in range(int(seconds * rate))))
    return buffer.getvalue()


def vision_body():
    image = dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(png()).decode()))
    return dict(model='google/gemma-4-26b-a4b-it', max_tokens=16,
                provider=dict(only=['parasail/bf16'], allow_fallbacks=False),
                messages=[dict(role='user', content=[dict(type='text', text='What colour is this image? One word.'),
                                                     image])])


def audio_body():
    audio = dict(type='input_audio', input_audio=dict(data=base64.b64encode(wav()).decode(), format='wav'))
    return dict(model='openai/gpt-audio-mini', max_tokens=16, modalities=['text'],
                messages=[dict(role='user', content=[dict(type='text', text='Describe this sound in three words.'),
                                                     audio])])


def image_out_body():
    return dict(model='google/gemini-3.1-flash-lite-image', modalities=['image', 'text'], max_tokens=1500,
                messages=[dict(role='user', content='A small red square on white. Image only.')])


def stream_body():
    return dict(model='meta-llama/llama-3.3-70b-instruct', max_tokens=40, stream=True,
                provider=dict(only=['coreweave/fp16'], allow_fallbacks=False),
                messages=[dict(role='user', content='Count from 1 to 15.')])


REQUESTS = [
    ('get-models', 'openrouter:openai/gpt-oss-120b', 'models/openai/gpt-oss-120b/endpoints', None, 'GET'),
    ('vision-input', 'openrouter:google/gemma-4-26b-a4b-it', 'chat/completions', vision_body, 'POST'),
    ('audio-input', 'openrouter:openai/gpt-audio-mini', 'chat/completions', audio_body, 'POST'),
    ('image-output', 'openrouter:google/gemini-3.1-flash-lite-image', 'chat/completions', image_out_body, 'POST'),
    ('streaming', 'openrouter:meta-llama/llama-3.3-70b-instruct', 'chat/completions', stream_body, 'POST'),
]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def kremlin(*args, timeout=1800):
    proc = subprocess.run([str(BIN / 'kremlin'), *args], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--witness', help='use a deployed relay: tls://HOST:PORT,https://HOST/witness[,PUBKEY]')
    parser.add_argument('--hub', help='use a deployed hub: https://HOST/hub')
    args = parser.parse_args()
    if not os.environ.get('OPENROUTER_API_KEY'):
        sys.exit('OPENROUTER_API_KEY required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    relay = hub = None
    if args.witness:
        parts = args.witness.split(',')
        pubkey = parts[2] if len(parts) > 2 else httpx.get(parts[1] + '/pubkey').json()['public_key']
        witness = f'{parts[0]},{parts[1]},{pubkey}'
    else:
        tunnel, control = free_port(), free_port()
        relay = subprocess.Popen([str(BIN / 'kremlin-witness'), '--key', str(out / 'witness.key'), '--allow',
                                  'openrouter.ai', '--tunnel', f'127.0.0.1:{tunnel}', '--control',
                                  f'127.0.0.1:{control}', '--receipts', str(out / 'receipts'), '--quiet'],
                                 stdout=subprocess.PIPE, text=True)
        time.sleep(2)
        pubkey = httpx.get(f'http://127.0.0.1:{control}/pubkey').json()['public_key']
        witness = f'127.0.0.1:{tunnel},http://127.0.0.1:{control},{pubkey}'
    if args.hub:
        hub_url = args.hub.rstrip('/')
    else:
        hub_port = free_port()
        hub = subprocess.Popen([str(BIN / 'kremlin-hub'), '--dir', str(out / 'hub'), '--witness-key', pubkey,
                                '--listen', f'127.0.0.1:{hub_port}', '--workers', '2'], stdout=subprocess.PIPE, text=True)
        time.sleep(2)
        hub_url = f'http://127.0.0.1:{hub_port}'
    rows = []
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from kremlin.replay import AuditFailure, audit
    from kremlin.witness import receipts as receipt_mod
    from kremlin.witness.client import parse_http_response
    try:
        for name, argv, budget in INVESTIGATIONS:
            if args.only and name not in args.only:
                continue
            run = out / name
            started = time.time()
            code, stdout, stderr = kremlin(*argv, '--out', str(run), '--witness', witness, '--max-usd', str(budget))
            row = dict(case=name, ran=code == 0, seconds=round(time.time() - started, 1))
            try:
                results = json.loads((run / 'results.json').read_text())
                row.update(status=results['status'], requests=results['usage']['requests'],
                           spend=results['usage']['spent_usd'])
                report = audit(run, pubkey, require_witness=True)
                row.update(audited=report['attested'] and report['results_reproduced'],
                           witnessed=f"{report['witnessed']}/{report['records']}",
                           disclosed=report['requests_disclosed'])
            except (AuditFailure, OSError, KeyError, ValueError) as error:
                row.update(audited=False, error=str(error)[:200] or stderr[-200:])
            if row.get('audited'):
                code, stdout, _ = kremlin('submit', str(run), '--hub', hub_url)
                row['hub'] = 'accepted' if code == 0 else 'rejected: ' + stdout[-200:]
            rows.append(row)
            print(json.dumps(row), flush=True)
        if not args.only or 'resume' in args.only:
            run = out / 'resume'
            argv = INVESTIGATIONS[0][1]
            kremlin(*argv, '--out', str(run), '--witness', witness, '--max-usd', '0.0003', '--concurrency', '2')
            stopped = json.loads((run / 'results.json').read_text())['status']
            code, stdout, _ = kremlin('resume', str(run), '--max-usd', '0.05')
            results = json.loads((run / 'results.json').read_text())
            report = audit(run, pubkey, require_witness=True)
            code_hub, _, _ = kremlin('submit', str(run), '--hub', hub_url)
            row = dict(case='resume', first=stopped, status=results['status'],
                       resumed_calls=results['usage']['resumed_calls'], requests=results['usage']['requests'],
                       audited=report['attested'] and report['results_reproduced'],
                       hub='accepted' if code_hub == 0 else 'rejected')
            rows.append(row)
            print(json.dumps(row), flush=True)
        for name, route, path, body_fn, method in REQUESTS:
            if args.only and name not in args.only:
                continue
            run = out / name
            extra = []
            body = None
            if body_fn:
                body = body_fn()
                (out / f'{name}.json').write_text(json.dumps(body))
                extra = ['--json', f'@{out / name}.json']
            code, stdout, stderr = kremlin('request', route, path, '--method', method, *extra, '--out', str(run),
                                           '--witness', witness, '--max-usd', '0.1')
            row = dict(case=name, ran=code == 0)
            try:
                record = json.loads((run / 'requests.jsonl').read_text().splitlines()[-1])
                receipt = record['witness_receipt']
                raw = receipt_mod.verify(receipt, pubkey)
                status, headers, response = parse_http_response(raw)
                sent = base64.b64decode(receipt['request']['body_base64']) if receipt.get('request') else None
                row.update(status=status, receipt_checks=len(receipt['checks']),
                           request_body_witnessed=(json.loads(sent) == body) if body else 'GET (empty body)',
                           request_kb=round(len(sent or b'') / 1024, 1), response_kb=round(len(response) / 1024, 1),
                           sse_events=response.count(b'data:') if body and body.get('stream') else None)
                if name == 'image-output' and status == 200:
                    message = json.loads(response)['choices'][0]['message']
                    row['images_returned'] = len(message.get('images') or [])
                if status == 200 and name in ('vision-input', 'audio-input'):
                    row['answer'] = json.loads(response)['choices'][0]['message'].get('content', '')[:60]
            except (OSError, KeyError, ValueError, IndexError) as error:
                row.update(error=f'{type(error).__name__}: {error}'[:200], stderr=stderr[-200:])
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        for process in (relay, hub):
            if process:
                process.terminate()
    (out / 'validation.json').write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if (r.get('audited') and r.get('hub') == 'accepted')
          or (r.get('receipt_checks') and r.get('status') == 200
              and r.get('request_body_witnessed') in (True, 'GET (empty body)'))]
    print(f'\n{len(ok)}/{len(rows)} cases passed; total spend ${sum(r.get("spend") or 0 for r in rows):.4f} '
          '(investigations only)')


if __name__ == '__main__':
    main()
