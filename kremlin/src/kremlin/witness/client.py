"""Client for a kremlin-witness relay: witnessed HTTPS requests to any host.

  witness = WitnessClient('relay.example.com:8080', 'https://relay.example.com/witness')
  result = witness.request('POST', 'https://openrouter.ai/api/v1/chat/completions',
                           headers={'Authorization': 'Bearer ...', 'Content-Type': 'application/json'},
                           body=json.dumps(payload).encode())
  result.status, result.json(), result.receipt

The TLS session runs here (Python ssl, TLS 1.3 only, fresh context per request so there is no
resumption). The key log goes to a private temporary file; only SERVER_HANDSHAKE_TRAFFIC_SECRET
and SERVER_TRAFFIC_SECRET_0 are sent to the relay, after the connection is closed. The relay's
signed receipt is checked against the witness public key and must match the bytes received here.
"""
import json
import os
import socket
import ssl
import tempfile
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from . import receipts

DISCLOSED = ('SERVER_HANDSHAKE_TRAFFIC_SECRET', 'SERVER_TRAFFIC_SECRET_0')
AUTH_HEADERS = ('authorization', 'x-api-key', 'x-goog-api-key', 'api-key')  # the one header kept sealed


class WitnessError(Exception):
    pass


@dataclass
class WitnessedResponse:
    status: int
    headers: dict
    body: bytes
    receipt: dict
    timings: dict = field(default_factory=dict)

    def json(self):
        return json.loads(self.body)

    def text(self):
        return self.body.decode('utf-8', 'replace')


def decode_with_offsets(raw):
    """Parse an HTTP/1.1 response; return (status, headers, body, segments).

    segments map decoded-body offsets back to offsets in the raw response:
    [(body_start, raw_start, length)], so timestamps of raw TLS records can be attached to
    positions in the decoded body (needed for chunked streaming responses).
    """
    head, sep, _ = raw.partition(b'\r\n\r\n')
    if not sep:
        raise WitnessError('incomplete HTTP response')
    lines = head.decode('latin-1').split('\r\n')
    status = int(lines[0].split(' ')[1])
    headers = {}
    for line in lines[1:]:
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip().lower()] = value.strip()
    start = len(head) + 4
    body, segments = bytearray(), []
    if 'chunked' in headers.get('transfer-encoding', '').lower():
        i = start
        while True:
            end = raw.index(b'\r\n', i)
            size = int(raw[i:end].split(b';')[0], 16)
            if size == 0:
                break
            segments.append((len(body), end + 2, size))
            body += raw[end + 2:end + 2 + size]
            i = end + 2 + size + 2
    else:
        length = int(headers['content-length']) if 'content-length' in headers else len(raw) - start
        segments.append((0, start, length))
        body += raw[start:start + length]
    return status, headers, bytes(body), segments


def response_complete(raw):
    """True once a full HTTP/1.1 response is buffered (Content-Length reached or final chunk seen), so the client can
    close without waiting for the server to close the connection."""
    head, sep, rest = bytes(raw).partition(b'\r\n\r\n')
    if not sep:
        return False
    lines = head.decode('latin-1').lower().split('\r\n')
    headers = dict(l.split(':', 1) for l in lines[1:] if ':' in l)
    if 'chunked' in headers.get('transfer-encoding', ''):
        i = 0
        while True:
            end = rest.find(b'\r\n', i)
            if end < 0:
                return False
            try:
                size = int(rest[i:end].split(b';')[0], 16)
            except ValueError:
                return False
            if size == 0:
                return rest.find(b'\r\n\r\n', end) >= 0 or rest.endswith(b'\r\n\r\n')
            i = end + 2 + size + 2
            if i > len(rest):
                return False
    if 'content-length' in headers:
        return len(rest) >= int(headers['content-length'].strip())
    return False  # read until close


def parse_http_response(raw):
    status, headers, body, _ = decode_with_offsets(raw)
    return status, headers, body


def sse_event_times(raw, records):
    """[(t_ms, data_line)] for each 'data:' line, timed by the witness record holding its last byte."""
    import bisect
    _, _, body, segments = decode_with_offsets(raw)
    starts = [s[0] for s in segments]
    record_starts = [r['offset'] for r in records]
    out, i = [], 0
    while True:
        j = body.find(b'data:', i)
        if j < 0:
            break
        end = body.find(b'\n', j)
        end = len(body) if end < 0 else end
        k = bisect.bisect_right(starts, end - 1) - 1
        raw_offset = segments[k][1] + (end - 1 - segments[k][0])
        r = bisect.bisect_right(record_starts, raw_offset) - 1
        out.append((records[r]['t_ms'], body[j + 5:end].strip().decode('utf-8', 'replace')))
        i = end
    return out


def tls12_secrets(keylog_path, head):
    """TLS 1.2: derive server_write_key/IV locally from the master secret; only those are disclosed."""
    from . import tls12, tls13
    line = next((l.split() for l in open(keylog_path) if l.startswith('CLIENT_RANDOM ')), None)
    if not line:
        raise WitnessError('TLS 1.2 master secret not found in key log')
    _, random, master = line
    hello = next(m for kind, _, payload, _ in tls13.records(bytes(head)) if kind == tls13.HANDSHAKE
                 for m in tls13.Messages().feed(payload) if m[0] == 2)
    server = tls12.parse_server_hello12(hello)
    key, iv = tls12.key_material(bytes.fromhex(master), bytes.fromhex(random), server['random'], server['suite'])
    return random, dict(tls12=True, server_write_key=key.hex(), server_write_iv=iv.hex())


def server_secrets(keylog_path):
    """Read only the server-direction secrets from an NSS key log file."""
    found, random = {}, None
    with open(keylog_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[0] in DISCLOSED:
                if random and parts[1] != random:
                    raise WitnessError('key log holds more than one session')
                random = parts[1]
                found[parts[0].lower()] = parts[2]
    if set(found) != {d.lower() for d in DISCLOSED}:
        raise WitnessError('TLS 1.3 server secrets not found in key log')
    return random, found


class BioTLS:
    """TLS over any socket-like transport (including another TLS socket), via ssl.MemoryBIO."""

    def __init__(self, sock, context, host):
        self.sock, self.incoming, self.outgoing = sock, ssl.MemoryBIO(), ssl.MemoryBIO()
        self.head = bytearray()  # first received bytes (TLS 1.2: ServerHello random and suite)
        self.obj = context.wrap_bio(self.incoming, self.outgoing, server_hostname=host)
        self._io(self.obj.do_handshake)

    def _flush(self):
        data = self.outgoing.read()
        if data:
            self.sock.sendall(data)

    def _io(self, fn, *args):
        while True:
            try:
                result = fn(*args)
                self._flush()
                return result
            except ssl.SSLWantReadError:
                self._flush()
                data = self.sock.recv(65536)
                if data:
                    if len(self.head) < 65536:
                        self.head += data
                    self.incoming.write(data)
                else:
                    self.incoming.write_eof()

    def sendall(self, data):
        view = memoryview(data)
        while view:
            view = view[self._io(self.obj.write, view):]

    def recv(self, size):
        return self._io(self.obj.read, size)

    def close(self):
        try:
            self._io(self.obj.unwrap)
        except (ssl.SSLError, OSError):
            pass


class WitnessClient:
    def __init__(self, tunnel, control_url, public_key=None, trust_file=None, timeout=120, token=None,
                 relay_trust_file=None, allow_tls12=True):
        """tunnel: 'host:port' (plain TCP) or 'tls://host:port' (TLS to the relay; protects the access token and
        the target host name; the API session runs inside it)."""
        self.tunnel_tls = tunnel.startswith('tls://')
        tunnel = tunnel[len('tls://'):] if self.tunnel_tls else tunnel
        self.relay_trust_file = relay_trust_file
        self.allow_tls12 = allow_tls12  # response-only attestation for TLS 1.2 hosts
        self.token = token if token is not None else os.environ.get('KREMLIN_WITNESS_TOKEN')
        host, _, port = tunnel.rpartition(':')
        self.tunnel = (host, int(port))
        self.control_url = control_url.rstrip('/')
        self.trust_file = trust_file
        self.timeout = timeout
        self.http = httpx.Client(timeout=60, limits=httpx.Limits(max_keepalive_connections=32))  # kept alive
        self.public_key = public_key or self.fetch_public_key()

    def fetch_public_key(self):
        # Trust-on-first-use: pin the returned key (pass public_key=...) for real verification.
        return self.info()['public_key']

    def info(self):
        """The witness's key and its data terms (what it stores and how the operator may use it)."""
        if not getattr(self, '_info', None):
            self._info = self.http.get(f'{self.control_url}/pubkey').json()
        return self._info

    def _disclosing_tls(self, sock, host, method, path, headers, body, on_data=None):
        """TLS 1.3 via tlslite-ng: hidden prefix (request line, Host, auth header), KeyUpdate, then the rest.

        Returns (raw_response, client_random_hex, server_secrets, request_disclosure)."""
        from . import fast_tlslite
        fast_tlslite.install()
        from tlslite import HandshakeSettings, TLSConnection
        from tlslite.constants import KeyUpdateMessageType
        from tlslite.errors import TLSAbruptCloseError, TLSRemoteAlert
        from cryptography import x509 as cx509
        from datetime import datetime, timezone
        from . import tls13
        auth = [(k, v) for k, v in headers.items() if k.lower() in AUTH_HEADERS]
        if len(auth) != 1:
            raise WitnessError(f'request disclosure needs exactly one auth header ({", ".join(AUTH_HEADERS)})')
        query = path.partition('?')[2].lower()
        if any(part.split('=')[0] in ('key', 'api_key', 'apikey', 'access_token', 'token') for part in query.split('&')):
            raise WitnessError('the URL query carries a credential, which request disclosure would publish; '
                               'send it in a header (e.g. x-goog-api-key) or use disclose_request=False')
        auth_name, auth_value = auth[0]
        auth_prefix = 'Bearer ' if auth_value.startswith('Bearer ') else ''
        key = auth_value[len(auth_prefix):]
        conn = TLSConnection(sock)
        captured = []
        original = conn._recordLayer.calcTLS1_3PendingState

        def capture(suite, client_secret, server_secret, implementations):
            captured.append((bytes(client_secret), bytes(server_secret)))
            return original(suite, client_secret, server_secret, implementations)
        conn._recordLayer.calcTLS1_3PendingState = capture
        settings = HandshakeSettings()
        settings.minVersion = settings.maxVersion = (3, 4)
        try:
            conn.handshakeClientCert(serverName=host, alpn=[bytearray(b'http/1.1')], settings=settings)
        except Exception as error:
            if 'version' in str(error).lower():
                raise WitnessError(f'{host} does not offer TLS 1.3; request disclosure needs it '
                                   '(use disclose_request=False / --no-disclose for response-only attestation)') from None
            raise
        # tlslite does not validate certificates: do it before the key is written.
        certs = [cx509.load_der_x509_certificate(bytes(c.writeBytes())) for c in conn.session.serverCertChain.x509List]
        tls13.verify_chain(certs, host, datetime.now(timezone.utc), self.trust_file)
        prefix = f'{method} {path} HTTP/1.1\r\nHost: {host}\r\n{auth_name}: {auth_value}\r\n'.encode()
        conn.write(prefix)
        for _ in conn.send_keyupdate_request(KeyUpdateMessageType.update_not_requested):
            pass
        after = bytes(conn.session.cl_app_secret)
        rest = [f'{k}: {v}' for k, v in headers.items()
                if k.lower() not in ('host', 'connection', 'content-length', 'accept-encoding', auth_name.lower())]
        rest += ['Accept-Encoding: identity', f'Content-Length: {len(body)}', 'Connection: close']
        conn.write(('\r\n'.join(rest) + '\r\n\r\n').encode() + body)
        raw = bytearray()
        while True:
            try:
                data = conn.read()
            except (TLSAbruptCloseError, TLSRemoteAlert, OSError):
                break
            if not data:
                break
            raw += data
            if on_data:
                on_data(bytes(data))
            if response_complete(raw):
                break
        try:
            conn.close()
        except Exception:
            pass
        handshake, application = captured[0], captured[1]
        layout = dict(method=method, path=path, auth_header=auth_name, auth_prefix=auth_prefix, key_length=len(key))
        return (bytes(raw), bytes(conn._clientRandom).hex(),
                dict(server_handshake_traffic_secret=handshake[1].hex(), server_traffic_secret_0=application[1].hex()),
                dict(client_secret_after_update=after.hex(), layout=layout))

    def request(self, method, url, headers=None, body=b'', disclose_request=False, on_data=None):
        """disclose_request=True: the witness also verifies and records the request (all but the auth header).
        on_data(bytes): called with the raw HTTP response bytes as they arrive (for live streaming); the response is
        still verified against the receipt before request() returns."""
        parts = urlsplit(url)
        if parts.scheme != 'https':
            raise WitnessError('witnessed requests must be https')
        host, port = parts.hostname, parts.port or 443
        path = (parts.path or '/') + (f'?{parts.query}' if parts.query else '')
        phases = {'start': time.time()}
        sock = socket.create_connection(self.tunnel, timeout=self.timeout)
        phases['relay_connected'] = time.time()
        if self.tunnel_tls:
            outer = ssl.create_default_context(cafile=self.relay_trust_file)
            sock = outer.wrap_socket(sock, server_hostname=self.tunnel[0])
        try:
            auth = f'Proxy-Authorization: Bearer {self.token}\r\n' if self.token else ''
            sock.sendall(f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}\r\n'.encode())
            head = b''
            while b'\r\n\r\n' not in head:
                chunk = sock.recv(4096)
                if not chunk:
                    raise WitnessError('relay closed the tunnel')
                head += chunk
            status_line, *lines = head.split(b'\r\n\r\n')[0].decode('latin-1').split('\r\n')
            if ' 200 ' not in status_line + ' ':
                raise WitnessError(f'relay refused tunnel: {status_line}')
            session = next(l.split(':', 1)[1].strip() for l in lines if l.lower().startswith('x-witness-session:'))
            phases['tunnel_open'] = time.time()
            if disclose_request:
                raw, random, disclosed, request_disclosure = self._disclosing_tls(sock, host, method, path,
                                                                                headers or {}, body, on_data)
                arrivals = []
            else:
                request_disclosure = None
                raw, random, disclosed, arrivals = self._ssl_tls(sock, host, method, path, headers, body, on_data)
        finally:
            sock.close()
        phases['tls_done'] = time.time()
        payload = dict(client_random=random, **disclosed)
        if request_disclosure:
            payload['request_disclosure'] = request_disclosure
        reply = self.http.post(f'{self.control_url}/sessions/{session}/disclose', json=payload)
        phases['receipt'] = time.time()
        result = self._finish(reply, raw, arrivals, disclose_request, method, path, body)
        phases['verified'] = time.time()
        marks = list(phases.items())
        result.timings['phases_ms'] = {b: round((tb - ta) * 1000, 1) for (a, ta), (b, tb) in zip(marks, marks[1:])}
        return result

    def _ssl_tls(self, sock, host, method, path, headers, body, on_data=None):
        private_dir = tempfile.mkdtemp(prefix='kremlin-keylog-')
        os.chmod(private_dir, 0o700)
        keylog = os.path.join(private_dir, 'keys.log')
        try:
            context = ssl.create_default_context(cafile=self.trust_file)
            context.minimum_version = ssl.TLSVersion.TLSv1_2 if self.allow_tls12 else ssl.TLSVersion.TLSv1_3
            context.set_alpn_protocols(['http/1.1'])
            context.keylog_filename = keylog
            tls = BioTLS(sock, context, host)
            version = tls.obj.version()
            lines_out = [f'{method} {path} HTTP/1.1', f'Host: {host}', 'Connection: close',
                         'Accept-Encoding: identity', f'Content-Length: {len(body)}']
            lines_out += [f'{k}: {v}' for k, v in (headers or {}).items()
                          if k.lower() not in ('host', 'connection', 'content-length', 'accept-encoding')]
            sent = time.time()
            tls.sendall(('\r\n'.join(lines_out) + '\r\n\r\n').encode() + body)
            raw, arrivals = bytearray(), []
            while True:
                try:
                    data = tls.recv(65536)
                except (ssl.SSLEOFError, ssl.SSLZeroReturnError):
                    break
                if not data:
                    break
                arrivals.append((time.time() - sent, len(data)))
                raw += data
                if on_data:
                    on_data(data)
                if response_complete(raw):
                    break
            try:
                tls.close()
            except OSError:
                pass
            if version == 'TLSv1.2':
                random, disclosed = tls12_secrets(keylog, tls.head)
            else:
                random, disclosed = server_secrets(keylog)
        finally:
            for name in os.listdir(private_dir):
                path_ = os.path.join(private_dir, name)
                with open(path_, 'r+b') as handle:  # overwrite before unlinking: it held client secrets
                    handle.write(b'\0' * os.path.getsize(path_))
                os.unlink(path_)
            os.rmdir(private_dir)
        return bytes(raw), random, disclosed, arrivals

    def _finish(self, reply, raw, arrivals, disclose_request, method, path, body):
        if reply.status_code != 200:
            raise WitnessError(f'witness rejected disclosure ({reply.status_code}): {reply.text[:300]}')
        receipt = reply.json()
        witnessed = receipts.verify(receipt, self.public_key)
        if witnessed != raw:
            raise WitnessError('witnessed response differs from the bytes this client received')
        if disclose_request:
            import base64 as b64
            request = receipt.get('request') or {}
            if b64.b64decode(request.get('body_base64', '')) != body or request.get('path') != path \
                    or request.get('method') != method:
                raise WitnessError('witnessed request differs from the request this client sent')
        status, response_headers, response_body = parse_http_response(raw)
        timings = dict(client_arrivals_s=arrivals, witness_records_ms=[r['t_ms'] for r in receipt['response_records']])
        if 'text/event-stream' in response_headers.get('content-type', ''):
            timings['witness_sse_events'] = sse_event_times(raw, receipt['response_records'])
        return WitnessedResponse(status, response_headers, response_body, receipt, timings)


def verify_receipt_file(path, public_key_b64):
    receipt = json.loads(open(path).read())
    raw = receipts.verify(receipt, public_key_b64)
    status, headers, body = parse_http_response(raw)
    return dict(host=receipt['host'], status=status, opened_at=receipt['opened_at'],
                certificate_chain_sha256=receipt['certificate_chain_sha256'], checks=receipt['checks'],
                body=body.decode('utf-8', 'replace'))
