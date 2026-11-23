"""Witnessed sessions: make ordinary API calls (any SDK, curl, any language) through a witness relay and keep
auditable evidence of every exchange.

Two ways in, both writing the same evidence directory (command 'session'):

  - kremlin.http_client(out) / kremlin.async_http_client(out): httpx clients whose transport sends every request
    through the witness. Pass them to an SDK, e.g. OpenAI(base_url=..., http_client=kremlin.http_client('runs/s1')).
  - `kremlin proxy --out runs/s1`: a local HTTP proxy. Point any tool's base URL at
    http://127.0.0.1:8787/<host>/<path>, e.g. OPENAI_BASE_URL=http://127.0.0.1:8787/openrouter.ai/api/v1.

The witness sees responses (and, with request disclosure, requests minus the auth header); it never sees the key.
Responses are returned once complete: streaming calls work, but chunks arrive together at the end (the witness's
own record timestamps keep the real token timing in the evidence).
The audit (kremlin audit DIR --pubkey KEY) checks every receipt and recomputes results.json from the records.
"""
import base64
import hashlib
import itertools
import json
import os
import threading
import time
from urllib.parse import urlsplit

import httpx

from . import evidence as evidence_mod

# The public relay on alexlitzenberger.com (override with --witness or KREMLIN_WITNESS).
PUBLIC_WITNESS = ('tls://alexlitzenberger.com:8443,https://alexlitzenberger.com/kremlinology/witness,'
                  '1pUve6CoMVNVk6p9750A5cED9hyw80ut4kOr6DJ12aY=')
# Hosts the proxy decrypts in HTTPS_PROXY mode unless --hosts says otherwise ('*' = everything): the usual LLM APIs.
DEFAULT_HOSTS = ['openrouter.ai', 'api.openai.com', '*.openai.azure.com', 'api.anthropic.com',
                 'generativelanguage.googleapis.com', '*-aiplatform.googleapis.com', 'api.groq.com', 'api.deepseek.com',
                 'api.together.xyz', 'api.fireworks.ai', 'api.mistral.ai', 'api.deepinfra.com', 'api.x.ai',
                 'api.cohere.com', 'api.cohere.ai', 'api.perplexity.ai', 'api.cerebras.ai', 'api.sambanova.ai',
                 'api.novita.ai', 'api.parasail.io', 'api.hyperbolic.xyz', 'api.moonshot.ai', 'api.minimax.io',
                 'dashscope*.aliyuncs.com', 'api.studio.nebius.com', 'api.inference.net', 'router.huggingface.co',
                 'api-inference.huggingface.co', 'integrate.api.nvidia.com', 'api.venice.ai', 'llm.chutes.ai',
                 'api.siliconflow.com', 'api.siliconflow.cn', 'api.z.ai', 'open.bigmodel.cn', 'api.baseten.co',
                 'inference.baseten.co', 'api.lambda.ai', 'api.replicate.com', 'bedrock-runtime.*.amazonaws.com']
HOP_BY_HOP = {'connection', 'keep-alive', 'proxy-connection', 'transfer-encoding', 'te', 'trailer', 'upgrade',
              'content-length', 'host', 'accept-encoding'}


def default_witness_spec():
    return os.environ.get('KREMLIN_WITNESS') or PUBLIC_WITNESS


def make_witness(spec=None, trust_file=None, announce=True):
    """WitnessClient from 'TUNNEL,CONTROL_URL[,PUBKEY]' (default: KREMLIN_WITNESS or the public relay)."""
    from .witness.client import WitnessClient
    parts = (spec or default_witness_spec()).split(',')
    if len(parts) not in (2, 3):
        raise ValueError('witness spec must be TUNNEL_HOST:PORT,CONTROL_URL[,PUBKEY_BASE64]')
    client = WitnessClient(parts[0], parts[1], public_key=parts[2] if len(parts) == 3 else None, trust_file=trust_file)
    info = client.info()
    if announce and info.get('data_terms'):
        import sys
        print(f"witness data terms: {info['data_terms']}"
              + (f" ({info['terms_url']})" if info.get('terms_url') else ''), file=sys.stderr)
    return client


def _json_or_none(body):
    try:
        return json.loads(body) if body else None
    except ValueError:
        return None


def describe(record, body):
    """One summary row for an exchange, computed only from the record and the captured response body (so the
    audit can recompute it)."""
    from .client import witnessed_latency
    receipt = record.get('witness_receipt')
    latency = witnessed_latency(receipt) if receipt else None
    row = dict(n=record['n'], method=record['method'], url=record['url'], status=record.get('status'),
               witnessed=bool(receipt), request_disclosed=bool(receipt and receipt.get('request')),
               latency_s=round(latency, 4) if latency is not None else None,
               error=record.get('error'))
    # Request fields only when the witness saw the request (disclosed and checked against this record by the audit);
    # otherwise they would rest on the submitter's word, so they are left out.
    request = _json_or_none(base64.b64decode(record.get('request_base64') or '')) if row['request_disclosed'] else None
    if isinstance(request, dict):
        row.update(model_requested=request.get('model'), stream=bool(request.get('stream')))
    text = (body or b'').decode('utf-8', 'replace')
    objects = []
    if text.lstrip().startswith(('data:', 'event:')):
        for line in text.splitlines():
            if line.startswith('data:') and line[5:].strip() not in ('', '[DONE]'):
                obj = _json_or_none(line[5:].strip())
                if isinstance(obj, dict):
                    objects.append(obj.get('message') if obj.get('type') == 'message_start' else obj)
    else:
        obj = _json_or_none(body)
        if isinstance(obj, dict):
            objects.append(obj)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for field in ('id', 'model', 'provider'):
            if obj.get(field) and not row.get(field if field != 'model' else 'model_served'):
                row[field if field != 'model' else 'model_served'] = obj[field]
        if obj.get('usage'):
            row['usage'] = obj['usage']
    return row


def summarize(rows):
    witnessed = sum(r['witnessed'] for r in rows)
    return dict(exchanges=len(rows), witnessed=witnessed, requests_disclosed=sum(r['request_disclosed'] for r in rows),
                hosts=sorted({urlsplit(r['url']).hostname for r in rows}), calls=rows)


def report(results):
    lines = ['# Witnessed session', '',
             f"{results['exchanges']} exchanges, {results['witnessed']} witnessed, "
             f"{results['requests_disclosed']} with the request disclosed; hosts: {', '.join(results['hosts'])}", '',
             '| # | method | url | status | model | served by | witnessed latency |', '|---|---|---|---|---|---|---|']
    for r in results['calls']:
        latency = f"{r['latency_s']:.2f} s" if r.get('latency_s') is not None else '—'
        lines.append(f"| {r['n']} | {r['method']} | {r['url']} | {r.get('status') or r.get('error') or '—'} | "
                     f"{r.get('model_served') or r.get('model_requested') or '—'} | {r.get('provider') or '—'} | "
                     f"{latency}{'' if r['witnessed'] else ' (unwitnessed)'} |")
    return '\n'.join(lines) + '\n'


class Session:
    """Sends HTTP exchanges through a witness and records them. Thread-safe."""

    def __init__(self, out, witness=None, *, disclose=True, trust_file=None, max_requests=None, args=None,
                 _resume=False):
        if _resume:
            self.evidence = evidence_mod.Evidence(out, 'session', {}, resume=True)
            self.closed = False
            self.lock = threading.Lock()
            self.inflight = 0
            self.idle = threading.Condition()
            return
        self.witness = witness if witness is not None else make_witness(trust_file=trust_file)
        self.disclose = disclose
        self.max_requests = max_requests
        self.evidence = evidence_mod.Evidence(out, 'session', dict(
            args or {}, witness_public_key=self.witness.public_key, disclose=disclose))
        self._n = itertools.count()
        self.lock = threading.Lock()
        self.closed = False
        self.inflight = 0
        self.idle = threading.Condition()

    def send(self, method, url, headers=None, body=b'', stream=None):
        """(status, headers, body, receipt). Raises WitnessError if the relay or the TLS check fails.
        stream: a ResponseStream fed with the raw response as it arrives (live streaming); it is finished (with the
        error, if any) when the exchange is verified and recorded."""
        from .witness.client import AUTH_HEADERS, WitnessError
        n = next(self._n)
        if self.max_requests is not None and n >= self.max_requests:
            raise WitnessError(f'request limit ({self.max_requests}) reached for this session')
        headers = {k: v for k, v in (headers or {}).items() if k.lower() not in HOP_BY_HOP}
        for k, v in headers.items():  # never let a key into the evidence, even if echoed back
            if k.lower() in AUTH_HEADERS:
                secret = v.split(' ', 1)[1] if v.startswith('Bearer ') else v
                with self.lock:
                    if secret not in self.evidence.secrets:
                        self.evidence.secrets.append(secret)
        auth = [k for k in headers if k.lower() in AUTH_HEADERS]
        disclose = self.disclose and len(auth) == 1
        record = dict(kind='http', key=f'http:{n}', n=n, route=urlsplit(url).hostname, method=method, url=url,
                      request_base64=base64.b64encode(body).decode(), request_sha256=hashlib.sha256(body).hexdigest(),
                      request_headers={k: ('[REDACTED]' if k.lower() in AUTH_HEADERS else v) for k, v in headers.items()},
                      sent_at=time.time(), attempt=0)
        on_data = stream.feed if stream else None
        with self.idle:
            self.inflight += 1
        try:
            return self._exchange(method, url, headers, body, stream, on_data, disclose, record)
        finally:
            with self.idle:
                self.inflight -= 1
                self.idle.notify_all()

    def _exchange(self, method, url, headers, body, stream, on_data, disclose, record):
        from .witness.client import WitnessError
        try:
            try:
                response = self.witness.request(method, url, headers=headers, body=body, disclose_request=disclose,
                                                on_data=on_data)
            except WitnessError as error:
                if not (disclose and 'does not offer TLS 1.3' in str(error)):
                    raise
                # Fails during the handshake, before anything is sent: safe to retry response-only.
                disclose = False
                response = self.witness.request(method, url, headers=headers, body=body, disclose_request=False,
                                                on_data=on_data)
        except Exception as error:
            # If part of a response was already streamed to the caller it was answered but is not witnessed, and
            # the audit says so; with nothing received the attempt counts as unanswered.
            record.update(status=stream.status if stream else None, error=f'{type(error).__name__}: {error}')
            self.evidence.record(record)
            if stream:
                stream.finish(error)
            raise
        record.update(status=response.status, witness_receipt=response.receipt,
                      response_headers=dict(response.headers), timings=response.timings)
        self.evidence.record(record, response.body)
        if stream:
            stream.finish()
        return response.status, response.headers, response.body, response.receipt

    @classmethod
    def finish(cls, out):
        """Finalize a session whose process was killed before it closed (results, report, manifest)."""
        return cls(out, _resume=True).close()

    def close(self, wait_s=600):
        with self.idle:  # exchanges still running (e.g. a stream the caller stopped reading) finish first
            self.idle.wait_for(lambda: self.inflight == 0, timeout=wait_s)
        with self.lock:
            if self.closed:
                return
            self.closed = True
        records = sorted(self.evidence.records(), key=lambda r: r['n'])
        results = summarize([describe(r, self.evidence.capture(r['capture']) if r.get('status') is not None else b'')
                             for r in records])
        self.evidence.write_json('results.json', dict(status='completed', usage=None, results=results))
        self.evidence.write_text('report.md', report(results))
        self.evidence.close(status='completed')
        return self.evidence.root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ResponseStream:
    """Receives the raw HTTP response while the witness session is still running. Once the head is in, callers
    can see status/headers and, for event streams, consume the body live; everything is still verified against
    the receipt before the exchange is recorded."""

    def __init__(self):
        self.buffer = bytearray()
        self.status, self.headers, self.streaming, self.error = None, {}, False, None
        self.head_ready = threading.Event()
        self.chunks = __import__('queue').Queue()

    def feed(self, data):
        try:
            if self.status is not None:
                self.chunks.put(bytes(data))
                return
            self.buffer += data
            end = self.buffer.find(b'\r\n\r\n')
            if end < 0:
                return
            lines = bytes(self.buffer[:end]).decode('latin-1').split('\r\n')
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            self.status, self.headers = int(lines[0].split(' ')[1]), headers
            self.streaming = 'text/event-stream' in headers.get('content-type', '')
            rest = bytes(self.buffer[end + 4:])
            self.buffer = bytearray()
            if rest:
                self.chunks.put(rest)
            self.head_ready.set()
        except Exception:  # never disturb the witnessed session; the full response is still returned at the end
            pass

    def finish(self, error=None):
        self.error = error
        self.chunks.put(None)
        self.head_ready.set()

    def raw_body(self):
        """Raw body bytes after the head (still transfer-encoded), as they arrive; ends when the exchange ends."""
        while True:
            chunk = self.chunks.get()
            if chunk is None:
                return
            yield chunk

    @property
    def chunked(self):
        return 'chunked' in self.headers.get('transfer-encoding', '').lower()


class Dechunker:
    """Incremental HTTP/1.1 chunked-transfer decoder; passes bytes through when the body is not chunked."""

    def __init__(self, chunked):
        self.chunked, self.buf, self.state, self.need = chunked, bytearray(), 'size', 0

    def feed(self, data):
        if not self.chunked:
            return bytes(data)
        self.buf += data
        out = bytearray()
        while True:
            if self.state == 'size':
                end = self.buf.find(b'\r\n')
                if end < 0:
                    break
                size = int(bytes(self.buf[:end]).split(b';')[0].strip() or b'0', 16)
                del self.buf[:end + 2]
                if size == 0:
                    self.state = 'done'
                    break
                self.need, self.state = size, 'data'
            elif self.state == 'data':
                if not self.buf:
                    break
                take = min(len(self.buf), self.need)
                out += self.buf[:take]
                del self.buf[:take]
                self.need -= take
                if self.need:
                    break
                self.state = 'crlf'
            elif self.state == 'crlf':
                if len(self.buf) < 2:
                    break
                del self.buf[:2]
                self.state = 'size'
            else:
                break
        return bytes(out)


def _response_headers(headers):
    return [(k, v) for k, v in headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != 'content-encoding']


def _httpx_response(request, status, headers, body):
    return httpx.Response(status, headers=_response_headers(headers), content=body, request=request)


def _transport_error(error):
    return error if isinstance(error, httpx.HTTPError) else httpx.TransportError(f'{type(error).__name__}: {error}')


class _LiveBody(httpx.SyncByteStream):
    def __init__(self, stream, thread, box):
        self.stream, self.thread, self.box = stream, thread, box

    def __iter__(self):
        decoder = Dechunker(self.stream.chunked)
        for raw in self.stream.raw_body():
            data = decoder.feed(raw)
            if data:
                yield data
        self.thread.join()
        if 'error' in self.box:
            raise _transport_error(self.box['error'])


class WitnessTransport(httpx.BaseTransport):
    """Event streams (SSE) are passed to the caller as they arrive; other responses once verified."""

    def __init__(self, session):
        self.session = session

    def handle_request(self, request):
        body = request.read()
        stream, box = ResponseStream(), {}

        def run():
            try:
                box['result'] = self.session.send(request.method, str(request.url), dict(request.headers), body,
                                                  stream=stream)
            except Exception as error:
                box['error'] = error
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        stream.head_ready.wait()
        if stream.status is not None and stream.streaming:
            return httpx.Response(stream.status, headers=_response_headers(stream.headers),
                                  stream=_LiveBody(stream, thread, box), request=request)
        thread.join()
        if 'error' in box:
            raise _transport_error(box['error'])
        status, headers, content, _ = box['result']
        return _httpx_response(request, status, headers, content)

    def close(self):
        self.session.close()


class _AsyncLiveBody(httpx.AsyncByteStream):
    def __init__(self, stream, future):
        self.stream, self.future = stream, future

    async def __aiter__(self):
        import asyncio
        decoder = Dechunker(self.stream.chunked)
        while True:
            raw = await asyncio.to_thread(self.stream.chunks.get)
            if raw is None:
                break
            data = decoder.feed(raw)
            if data:
                yield data
        try:
            await self.future
        except Exception as error:
            raise _transport_error(error) from error


class AsyncWitnessTransport(httpx.AsyncBaseTransport):
    def __init__(self, session):
        self.session = session

    async def handle_async_request(self, request):
        import asyncio
        body = await request.aread()
        stream = ResponseStream()
        future = asyncio.ensure_future(asyncio.to_thread(
            self.session.send, request.method, str(request.url), dict(request.headers), body, stream))
        await asyncio.to_thread(stream.head_ready.wait)
        if stream.status is not None and stream.streaming:
            return httpx.Response(stream.status, headers=_response_headers(stream.headers),
                                  stream=_AsyncLiveBody(stream, future), request=request)
        try:
            status, headers, content, _ = await future
        except Exception as error:
            raise _transport_error(error) from error
        return _httpx_response(request, status, headers, content)

    async def aclose(self):
        self.session.close()


def _session(out, witness, disclose, trust_file, max_requests):
    if isinstance(witness, Session):
        return witness
    if witness is None or isinstance(witness, str):
        witness = make_witness(witness, trust_file)
    return Session(out, witness, disclose=disclose, trust_file=trust_file, max_requests=max_requests,
                   args=dict(via='http_client'))


def http_client(out, witness=None, *, disclose=True, trust_file=None, max_requests=None, timeout=300, **kwargs):
    """An httpx.Client that sends everything through the witness and records evidence in `out`.
    The evidence is finalized (results, report, manifest) when the client is closed (or at interpreter exit)."""
    session = _session(out, witness, disclose, trust_file, max_requests)
    import atexit
    atexit.register(session.close)
    client = httpx.Client(transport=WitnessTransport(session), timeout=timeout, **kwargs)
    client.session = session
    return client


def async_http_client(out, witness=None, *, disclose=True, trust_file=None, max_requests=None, timeout=300, **kwargs):
    session = _session(out, witness, disclose, trust_file, max_requests)
    import atexit
    atexit.register(session.close)
    client = httpx.AsyncClient(transport=AsyncWitnessTransport(session), timeout=timeout, **kwargs)
    client.session = session
    return client


# ---- local proxy -------------------------------------------------------------------------------------------------

def target_url(path, upstream=None):
    """'/openrouter.ai/api/v1/chat/completions' -> 'https://openrouter.ai/api/v1/chat/completions';
    with an upstream base, the path is appended to it."""
    if upstream:
        return upstream.rstrip('/') + path
    host, _, rest = path.lstrip('/').partition('/')
    if '.' not in host and ':' not in host and host != 'localhost':
        raise ValueError('path must start with the API host, e.g. /api.openai.com/v1/... (or run with --upstream)')
    return f'https://{host}/{rest}'


class LocalCA:
    """A private CA for the proxy's HTTPS_PROXY mode (like mitmproxy): clients that trust it let the proxy read their
    requests so it can send them through the witness. Kept in ~/.config/kremlin/proxy-ca, never leaves the machine."""

    def __init__(self, directory=None):
        from pathlib import Path
        self.dir = Path(directory or os.path.expanduser('~/.config/kremlin/proxy-ca'))
        self.cert_path, self.key_path = self.dir / 'ca.pem', self.dir / 'ca.key'
        self.lock = threading.Lock()
        self.contexts = {}
        if not self.cert_path.exists():
            self._create()
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        self.cert = x509.load_pem_x509_certificate(self.cert_path.read_bytes())
        self.key = serialization.load_pem_private_key(self.key_path.read_bytes(), None)

    def _create(self):
        import datetime
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'kremlin local proxy CA'),
                          x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'kremlin (this machine only)')])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=3650))
                .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                             content_commitment=False, key_encipherment=False, data_encipherment=False,
                                             key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
                .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
                .sign(key, hashes.SHA256()))
        self.key_path.touch(mode=0o600)
        self.key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                    serialization.NoEncryption()))
        self.cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    @property
    def bundle_path(self):
        """The public CA roots plus this CA: for tools whose setting replaces the trust store (SSL_CERT_FILE,
        REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE), so hosts tunnelled straight through still verify."""
        import certifi
        path = self.dir / 'bundle.pem'
        wanted = open(certifi.where(), 'rb').read() + b'\n' + self.cert_path.read_bytes()
        if not path.exists() or path.read_bytes() != wanted:
            path.write_bytes(wanted)
        return path

    def java_truststore(self):
        """The JDK's default roots plus this CA as a PKCS12 truststore (password 'changeit'), or None without
        keytool. For JVM languages (Java, Kotlin, Scala), which ignore SSL_CERT_FILE."""
        import shutil
        import subprocess
        path = self.dir / 'java-truststore.p12'
        if path.exists() and path.stat().st_mtime >= self.cert_path.stat().st_mtime:
            return path
        keytool = shutil.which('keytool')
        if not keytool:
            return None
        home = os.path.dirname(os.path.dirname(os.path.realpath(keytool)))
        cacerts = os.path.join(home, 'lib', 'security', 'cacerts')
        path.unlink(missing_ok=True)
        common = ['-noprompt', '-storepass', 'changeit']
        if os.path.exists(cacerts):
            subprocess.run([keytool, '-importkeystore', '-srckeystore', cacerts, '-srcstorepass', 'changeit',
                            '-destkeystore', str(path), '-deststoretype', 'PKCS12', '-deststorepass', 'changeit',
                            '-noprompt'], check=True, capture_output=True)
        subprocess.run([keytool, '-importcert', '-alias', 'kremlin-proxy-ca', '-file', str(self.cert_path),
                        '-keystore', str(path), '-storetype', 'PKCS12', *common], check=True, capture_output=True)
        return path

    def php_ini_dir(self):
        """An ini directory for PHP_INI_SCAN_DIR: PHP's curl and openssl ignore SSL_CERT_FILE."""
        directory = self.dir / 'php'
        directory.mkdir(exist_ok=True)
        bundle = self.bundle_path
        (directory / 'kremlin.ini').write_text(f'curl.cainfo={bundle}\nopenssl.cafile={bundle}\n')
        return directory

    def context(self, host):
        """Server-side SSLContext presenting a certificate for host, signed by this CA."""
        import datetime
        import ipaddress
        import ssl
        import tempfile
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
        with self.lock:
            if host in self.contexts:
                return self.contexts[host]
            key = ec.generate_private_key(ec.SECP256R1())
            try:
                san = x509.IPAddress(ipaddress.ip_address(host))
            except ValueError:
                san = x509.DNSName(host)
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (x509.CertificateBuilder()
                    .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host[:64])]))
                    .issuer_name(self.cert.subject).public_key(key.public_key())
                    .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
                    .not_valid_after(now + datetime.timedelta(days=90))
                    .add_extension(x509.SubjectAlternativeName([san]), critical=False)
                    .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                    .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                    .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()),
                                   critical=False)
                    .sign(self.key, hashes.SHA256()))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.set_alpn_protocols(['http/1.1'])
            with tempfile.TemporaryDirectory() as tmp:  # load_cert_chain only takes files
                os.chmod(tmp, 0o700)
                cert_file, key_file = os.path.join(tmp, 'c.pem'), os.path.join(tmp, 'k.pem')
                with open(cert_file, 'wb') as handle:
                    handle.write(cert.public_bytes(serialization.Encoding.PEM))
                with open(key_file, 'wb') as handle:
                    handle.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()))
                context.load_cert_chain(cert_file, key_file)
            self.contexts[host] = context
            return context


def env_vars(listen, ca):
    """[(name, value, comment)]: the settings that send each language's usual HTTP stack through the proxy."""
    base = f'http://{listen}'
    host, _, port = listen.rpartition(':')
    bundle = str(ca.bundle_path)
    out = [(name, base, 'proxy for most HTTP stacks') for name in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY',
                                                                  'http_proxy')]
    out += [(name, 'localhost,127.0.0.1', '') for name in ('NO_PROXY', 'no_proxy')]
    out += [('SSL_CERT_FILE', bundle, 'Python, Go, Ruby, Rust (native-tls / native roots), .NET on Linux'),
            ('REQUESTS_CA_BUNDLE', bundle, 'Python requests'),
            ('CURL_CA_BUNDLE', bundle, 'curl'),
            ('NODE_EXTRA_CA_CERTS', str(ca.cert_path), 'Node 24+ and Bun'),
            ('NODE_USE_ENV_PROXY', '1', 'Node fetch honours HTTPS_PROXY'),
            ('DENO_CERT', str(ca.cert_path), 'Deno'),
            ('PHP_INI_SCAN_DIR', os.pathsep + str(ca.php_ini_dir()), 'PHP (curl, openssl)'),
            ('CARGO_HTTP_CAINFO', bundle, 'cargo')]
    truststore = ca.java_truststore()
    if truststore:
        out.append(('JAVA_TOOL_OPTIONS',
                    f'-Dhttps.proxyHost={host} -Dhttps.proxyPort={port} -Dhttp.proxyHost={host} '
                    f'-Dhttp.proxyPort={port} -Dhttp.nonProxyHosts=localhost|127.0.0.1 '
                    f'-Djavax.net.ssl.trustStore={truststore} -Djavax.net.ssl.trustStorePassword=changeit',
                    'Java, Kotlin, Scala'))
    return out


SHELLS = ('sh', 'fish', 'powershell', 'cmd')


def env_lines(listen, ca, shell='sh'):
    """The env_vars as commands for a shell: sh (bash/zsh), fish, powershell or cmd."""
    import shlex
    lines = []
    for name, value, comment in env_vars(listen, ca):
        if shell == 'sh':
            line = f'export {name}={shlex.quote(value)}' + (f'  # {comment}' if comment else '')
        elif shell == 'fish':
            line = f'set -gx {name} {shlex.quote(value)}' + (f'  # {comment}' if comment else '')
        elif shell == 'powershell':
            quoted = value.replace("'", "''")
            line = f"$env:{name} = '{quoted}'" + (f'  # {comment}' if comment else '')
        elif shell == 'cmd':
            line = f'set "{name}={value}"'
        else:
            raise ValueError(f'shell must be one of {SHELLS}')
        lines.append(line)
    return lines


def trust_commands(ca, platform=None):
    """Commands that add the proxy CA to the current user's trust store (no admin rights needed on macOS/Windows).
    Needed where the env settings are not read: .NET on macOS/Windows, browsers, many desktop apps."""
    import sys
    platform = platform or sys.platform
    pem = str(ca.cert_path)
    if platform == 'darwin':
        return [['security', 'add-trusted-cert', '-r', 'trustRoot', '-k',
                 os.path.expanduser('~/Library/Keychains/login.keychain-db'), pem]]
    if platform.startswith('win'):
        return [['certutil', '-user', '-addstore', 'Root', pem]]
    nss = os.path.expanduser('~/.pki/nssdb')  # Chrome/Chromium on Linux; system-wide trust needs root
    return [['certutil', '-d', f'sql:{nss}', '-A', '-t', 'C,,', '-n', 'kremlin local proxy CA', '-i', pem]]


def untrust_commands(ca, platform=None):
    import sys
    platform = platform or sys.platform
    if platform == 'darwin':
        return [['security', 'remove-trusted-cert', str(ca.cert_path)],
                ['security', 'delete-certificate', '-c', 'kremlin local proxy CA']]
    if platform.startswith('win'):
        return [['certutil', '-user', '-delstore', 'Root', 'kremlin local proxy CA']]
    return [['certutil', '-d', f"sql:{os.path.expanduser('~/.pki/nssdb')}", '-D', '-n', 'kremlin local proxy CA']]


def host_matches(host, patterns):
    import fnmatch
    return any(fnmatch.fnmatch(host.lower(), p.lower()) for p in patterns)


def serve_proxy(session, listen='127.0.0.1:8787', upstream=None, quiet=False, ca=None, hosts=None, cors=None):
    """Base-URL mode: http://LISTEN/<host>/<path> (or --upstream). HTTPS_PROXY mode (needs ca): CONNECT to a host
    in `hosts` (all if None) is decrypted with the local CA and sent through the witness; other hosts are tunnelled
    straight through, unrecorded. The base-URL form is limited to `hosts` too. cors: browser origins allowed to call
    the proxy (['*'] for any); without it browsers are refused, so web pages you visit cannot use it."""
    import select
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from .witness.client import WitnessError

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        tunnel_host = None

        def log_message(self, fmt, *args):
            if not quiet:
                super().log_message(fmt, *args)

        def _cors_origin(self):
            origin = self.headers.get('Origin')
            if origin and cors and ('*' in cors or origin in cors):
                return origin
            return None

        def do_OPTIONS(self):
            origin = self._cors_origin()
            if not origin:
                return self._error(403, 'browser origin not allowed; start the proxy with --cors ORIGIN')
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers',
                             self.headers.get('Access-Control-Request-Headers') or '*')
            self.send_header('Access-Control-Max-Age', '600')
            self.send_header('Vary', 'Origin')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _reply(self, status, headers, body):
            self.send_response(status)
            origin = self._cors_origin()
            if origin:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Access-Control-Expose-Headers', 'X-Kremlin-Witnessed, X-Kremlin-Session')
                self.send_header('Vary', 'Origin')
            for k, v in headers.items():
                if k.lower() in HOP_BY_HOP or k.lower() == 'content-encoding':
                    continue
                if origin and k.lower().startswith('access-control-'):
                    continue  # ours replace the provider's: a duplicate Allow-Origin makes browsers reject it
                self.send_header(k, v)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _error(self, status, message):
            self._reply(status, {'Content-Type': 'application/json', 'X-Kremlin-Error': '1'},
                        json.dumps(dict(error=dict(message=message, type='kremlin_proxy_error'))).encode())

        def do_CONNECT(self):
            host, _, port = self.path.rpartition(':')
            host, port = host.strip('[]'), int(port or 443)
            if hosts is not None and not host_matches(host, hosts) or ca is None:
                return self._blind_tunnel(host, port)
            self.send_response(200, 'Connection established')
            self.end_headers()
            self.wfile.flush()
            try:
                tls = ca.context(host).wrap_socket(self.connection, server_side=True)
            except (OSError, ValueError) as error:
                self.log_message('TLS with the client failed for %s (does it trust the kremlin CA?): %s', host, error)
                self.close_connection = True
                return
            self.connection = tls
            self.rfile, self.wfile = tls.makefile('rb'), tls.makefile('wb')
            self.tunnel_host = host if port == 443 else f'{host}:{port}'
            self.close_connection = False  # handle() keeps reading requests, now from inside the TLS session

        def _blind_tunnel(self, host, port):
            try:
                upstream_sock = socket.create_connection((host, port), timeout=30)
            except OSError as error:
                return self._error(502, f'cannot reach {host}:{port}: {error}')
            self.send_response(200, 'Connection established')
            self.end_headers()
            self.wfile.flush()
            sockets = [self.connection, upstream_sock]
            try:
                while True:
                    ready, _, _ = select.select(sockets, [], [], 300)
                    if not ready:
                        break
                    for sock in ready:
                        data = sock.recv(65536)
                        if not data:
                            raise EOFError
                        (upstream_sock if sock is self.connection else self.connection).sendall(data)
            except (EOFError, OSError):
                pass
            finally:
                upstream_sock.close()
                self.close_connection = True

        def _handle(self):
            if self.headers.get('Origin') and not self.tunnel_host and not self._cors_origin():
                return self._error(403, 'browser origin not allowed; start the proxy with --cors ORIGIN')
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length) if length else b''
            if 'chunked' in (self.headers.get('Transfer-Encoding') or '').lower():
                body = self._read_chunked()
            try:
                if self.tunnel_host:
                    url = f'https://{self.tunnel_host}{self.path}'
                elif self.path.startswith('https://'):
                    url = self.path
                elif self.path.startswith('http://'):
                    host = urlsplit(self.path).hostname or ''
                    if hosts is None or host_matches(host, hosts):
                        raise ValueError('plain http cannot be witnessed; use https')
                    return self._forward_plain(body)
                else:
                    url = target_url(self.path, upstream)
                    if hosts is not None and not host_matches(urlsplit(url).hostname or '', hosts):
                        raise PermissionError(f'{urlsplit(url).hostname} is not in the witnessed hosts (--hosts)')
            except PermissionError as error:
                return self._error(403, str(error))
            except ValueError as error:
                return self._error(400, str(error))
            stream, box = ResponseStream(), {}

            def run():
                try:
                    box['result'] = session.send(self.command, url, dict(self.headers.items()), body, stream=stream)
                except Exception as error:
                    box['error'] = error
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            stream.head_ready.wait()
            if stream.status is not None and stream.streaming:
                return self._stream(stream, worker, box)
            worker.join()
            if 'error' in box:
                error = box['error']
                prefix = 'witness: ' if isinstance(error, WitnessError) else f'{type(error).__name__}: '
                return self._error(502, prefix + str(error))
            status, headers, content, receipt = box['result']
            headers = dict(headers)
            headers['X-Kremlin-Witnessed'] = 'request+response' if receipt.get('request') else 'response'
            headers['X-Kremlin-Session'] = receipt.get('session_id', '')
            self._reply(status, headers, content)

        def _stream(self, stream, worker, box):
            """Event stream: forward the provider's bytes as they arrive (same framing). The receipt is checked when
            the stream ends; X-Kremlin-Witnessed says 'streaming' because the verdict is not known yet, and a failed
            check is recorded in the evidence and closes the connection."""
            self.send_response(stream.status)
            origin = self._cors_origin()
            for k, v in stream.headers.items():
                if k in HOP_BY_HOP or k == 'content-encoding' or (origin and k.startswith('access-control-')):
                    continue
                self.send_header(k, v)
            if origin:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Access-Control-Expose-Headers', 'X-Kremlin-Witnessed')
                self.send_header('Vary', 'Origin')
            self.send_header('X-Kremlin-Witnessed', 'streaming')
            if stream.chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            elif 'content-length' in stream.headers:
                self.send_header('Content-Length', stream.headers['content-length'])
            else:
                self.send_header('Connection', 'close')
                self.close_connection = True
            self.end_headers()
            try:
                for raw in stream.raw_body():
                    self.wfile.write(raw)
                    self.wfile.flush()
            except OSError:  # the caller went away; the witnessed exchange still completes and is recorded
                self.close_connection = True
            worker.join()
            if 'error' in box:
                self.log_message('stream not verified: %s', box['error'])
                self.close_connection = True

        def _forward_plain(self, body):
            """HTTP_PROXY traffic for hosts that are not witnessed: forwarded as is, not recorded."""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            try:
                reply = httpx.request(self.command, self.path, headers=headers, content=body, timeout=120)
            except httpx.HTTPError as error:
                return self._error(502, f'{type(error).__name__}: {error}')
            self._reply(reply.status_code, dict(reply.headers), reply.content)

        def _read_chunked(self):
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b';')[0], 16)
                if not size:
                    self.rfile.readline()
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    host, _, port = listen.rpartition(':')
    server = ThreadingHTTPServer((host or '127.0.0.1', int(port)), Handler)
    server.daemon_threads = True
    return server
