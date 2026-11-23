"""A local OpenAI-compatible HTTPS provider with known behaviour, for end-to-end tests.

Models map to simulated backends:
  shared-a, shared-b  one backend (same numerics, same prefix cache): a reseller pair
  other               an independent backend (different numerics and cache)
  ignorer             same as other but ignores max_tokens and bills reasoning
Logprobs are deterministic per (backend, prompt) with a few nondeterministic 'numeric states'.
Prefix cache: 16-token blocks of whitespace tokens, per backend. Streaming sends one event per
token with a delay; predictable prompts (copy/count) send two tokens per event (speculation).
Also: POST /v1/audio/transcriptions (multipart) and /v1/images/generations (large base64).
"""
import base64
import hashlib
import json
import math
import random
import socket
import ssl
import threading
import time

BACKENDS = {'shared-a': 'shared', 'shared-b': 'shared', 'other': 'other', 'ignorer': 'other', 'liar': 'other',
            'nolog': 'other', 'gmi': 'other', 'noprefill': 'shared'}
VOCAB = ['The', 'A', 'In', 'Once', 'Here', 'It', 'One', 'Red', 'Blue', 'Cat', 'Paris', 'Oak', 'To', 'My']


class Provider:
    def __init__(self, cert, key):
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.context.load_cert_chain(cert, key)
        self.caches = {'shared': set(), 'other': set()}
        self.lock = threading.Lock()
        self.random = random.Random(1)
        self.listener = socket.create_server(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.seen_auth = []
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def close(self):
        self.listener.close()

    # ---- HTTP -------------------------------------------------------------------------
    def handle(self, conn):
        try:
            with self.context.wrap_socket(conn, server_side=True) as tls:
                data = b''
                while b'\r\n\r\n' not in data:
                    chunk = tls.recv(65536)
                    if not chunk:
                        return
                    data += chunk
                head, _, body = data.partition(b'\r\n\r\n')
                lines = head.decode('latin-1').split('\r\n')
                method, path, _ = lines[0].split(' ', 2)
                headers = {k.strip().lower(): v.strip() for k, v in (l.split(':', 1) for l in lines[1:] if ':' in l)}
                length = int(headers.get('content-length', 0))
                while len(body) < length:
                    body += tls.recv(65536)
                self.seen_auth.append(headers.get('authorization'))
                self.route(tls, path, headers, body)
        except (ssl.SSLError, OSError, ValueError):
            pass

    def send_json(self, tls, obj, status=200):
        raw = json.dumps(obj).encode()
        tls.sendall(f'HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n'
                    f'Connection: close\r\n\r\n'.encode() + raw)

    def route(self, tls, path, headers, body):
        if path.endswith('/audio/transcriptions'):
            assert b'multipart/form-data' in headers.get('content-type', '').encode()
            size = len(body)
            return self.send_json(tls, dict(text=f'transcribed {size} bytes'))
        if path.endswith('/images/generations'):
            blob = random.Random(7).randbytes(300_000)
            return self.send_json(tls, dict(data=[dict(b64_json=base64.b64encode(blob).decode())]))
        if path.endswith('/chat/completions'):
            return self.chat(tls, json.loads(body))
        self.send_json(tls, dict(error='not found'), 404)

    # ---- model behaviour --------------------------------------------------------------------
    def law(self, backend, prompt):
        seed = int(hashlib.sha256(f'{backend}|{prompt}'.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        logits = [rng.gauss(0, 2) for _ in VOCAB]
        with self.lock:
            state = self.random.choice([0, 0, 0, 1])  # nondeterministic numeric state
        logits = [x + 0.01 * state * (i % 3) for i, x in enumerate(logits)]
        total = max(logits)
        norm = total + __import__('math').log(sum(__import__('math').exp(x - total) for x in logits))
        return {tok: round(x - norm, 6) for tok, x in zip(VOCAB, logits)}

    def cached_prefix(self, backend, tokens):
        blocks = [hash(tuple(tokens[:i + 16])) for i in range(0, len(tokens) - 15, 16)]
        with self.lock:
            cache = self.caches[backend]
            hit = 0
            for b in blocks:
                if b in cache:
                    hit += 16
                else:
                    break
            cache.update(blocks)
        return hit

    def chat(self, tls, request):
        model = request['model']
        backend = BACKENDS[model]
        prompt_text = json.dumps(request['messages'])
        tokens = prompt_text.split()
        cached = self.cached_prefix(backend, tokens)
        cap = request.get('max_tokens') or request.get('max_completion_tokens') or 16
        users = [m for m in request['messages'] if m['role'] == 'user']
        content = users[-1]['content'] if users else ''
        content = content if isinstance(content, str) else ' '.join(c.get('text', '') for c in content)
        last = request['messages'][-1]
        prefill = last['content'] if last['role'] == 'assistant' and model != 'noprefill' else ''
        if last['role'] == 'assistant' and len(request['messages']) > 2:
            prefill = ''  # a multi-turn conversation, not a prefill
        predictable = content.startswith(('Copy', 'Write the integers'))
        n_out = 200 if model == 'ignorer' else cap
        usage = dict(prompt_tokens=len(tokens), completion_tokens=n_out,
                     prompt_tokens_details=dict(cached_tokens=cached),
                     completion_tokens_details=dict(reasoning_tokens=150 if model == 'ignorer' else 0),
                     cost=1e-6 * (len(tokens) + n_out))
        def law_at(prefix):
            return self.law(backend, content if not prefix else content + '\x00' + prefix)
        law = law_at(prefill)
        temperature, top_k, top_p = request.get('temperature', 1.0), request.get('top_k'), request.get('top_p')
        if model == 'liar':  # ignores sampling parameters
            temperature, top_k, top_p = 1.0, None, None
        if request.get('seed') is not None:
            rng = random.Random(request['seed'])
        elif temperature == 0:
            rng = random.Random(content)  # greedy decoding is deterministic
        else:
            rng = random.Random()
        ranked = sorted(law, key=law.get, reverse=True)
        support = ranked[:top_k] if top_k else ranked
        if top_p is not None and top_p < 1 and temperature != 0:  # inclusive nucleus after temperature
            peak0 = max(law[t] / temperature for t in support)
            mass = [math.exp(law[t] / temperature - peak0) for t in support]
            total, kept, above = sum(mass), [], 0.0
            for t, m in zip(support, mass):
                if above < top_p:
                    kept.append(t)
                above += m / total
            support = kept
        if temperature == 0:
            weights = [1.0] + [0.0] * (len(support) - 1)
        else:
            peak = max(law[t] / temperature for t in support)
            weights = [math.exp(law[t] / temperature - peak) for t in support]

        def one():
            first = rng.choices(support, weights)[0]
            if content.startswith('Count from 1'):
                words = ', '.join(str(i) for i in range(1, 21)).split(' ')
            else:
                words = [first] + [rng.choice(VOCAB).lower() for _ in range(n_out - 1)]
            text = ' '.join(words[:n_out])
            finish = 'length'
            for stop in (request.get('stop') or []):
                if model != 'liar' and stop in text:
                    text, finish = text[:text.index(stop)], 'stop'
            return first, text, finish
        def sequence():
            """Token-by-token generation with per-token logprobs (each step's law depends on the prefix so far)."""
            produced, entries = '', []
            for _ in range(n_out):
                step = law_at(prefill + produced)
                ranked_k = sorted(step, key=step.get, reverse=True)
                support_k = ranked_k[:top_k] if top_k else ranked_k
                if temperature == 0:
                    token = support_k[0]
                else:
                    peak_k = max(step[t] / temperature for t in support_k)
                    token = rng.choices(support_k, [math.exp(step[t] / temperature - peak_k) for t in support_k])[0]
                token = token if not (prefill or produced) else ' ' + token
                key = token.strip()
                space = ' ' if (prefill or produced) else ''
                entries.append(dict(token=token, logprob=step[key], bytes=list(token.encode()),
                                    top_logprobs=[dict(token=space + t, logprob=step[t], bytes=list((space + t).encode()))
                                                  for t in ranked_k[:request.get('top_logprobs', 5)]]))
                produced += token
            return produced, entries
        count = request.get('n', 1)
        choices = []
        for index in range(count):
            if request.get('logprobs') and n_out > 1 and model != 'nolog':
                text, entries = sequence()
                choices.append(dict(index=index, message=dict(role='assistant', content=text), finish_reason='length',
                                    logprobs=dict(content=entries)))
                continue
            first, text, finish = one()
            logprobs = None
            if request.get('logprobs') and model != 'nolog':
                top = ranked[:request.get('top_logprobs', 5)]
                chosen = law[first] + (0.7 if model == 'gmi' and first != ranked[0] else 0.0)
                if model == 'gmi':
                    chosen = law[ranked[0]] + 0.4
                space = ' ' if prefill else ''  # continuing a prefill: tokens carry the leading space
                logprobs = dict(content=[dict(token=space + first, logprob=chosen, bytes=list((space + first).encode()),
                                              top_logprobs=[dict(token=space + t, logprob=law[t],
                                                                 bytes=list((space + t).encode())) for t in top])])
            choices.append(dict(index=index, message=dict(role='assistant', content=text), finish_reason=finish,
                                logprobs=logprobs))
        if not request.get('stream'):
            return self.send_json(tls, dict(id=f'cmpl-{random.getrandbits(48):012x}', model=model, choices=choices,
                                            usage=usage))
        tls.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n'
                    b'Connection: close\r\n\r\n')

        def event(obj):
            raw = f'data: {json.dumps(obj)}\n\n'.encode()
            tls.sendall(f'{len(raw):x}\r\n'.encode() + raw + b'\r\n')
        per_event = 2 if predictable else 1
        sent = 0
        while sent < n_out:
            k = min(per_event, n_out - sent)
            event(dict(id='s', choices=[dict(index=0, delta=dict(content=' w' * k))]))
            sent += k
            time.sleep(0.01)
        event(dict(id='s', choices=[dict(index=0, delta={}, finish_reason='length')], usage=usage))
        raw = b'data: [DONE]\n\n'
        tls.sendall(f'{len(raw):x}\r\n'.encode() + raw + b'\r\n0\r\n\r\n')
