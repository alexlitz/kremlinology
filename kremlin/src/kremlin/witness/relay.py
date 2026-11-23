"""kremlin-witness: the relay service.

Two listeners:
  tunnel   raw TCP. Accepts 'CONNECT host:port HTTP/1.1', dials host:port itself (allowlist),
           answers with a session id, then forwards bytes both ways and records them with
           nanosecond timestamps. TLS terminates on the client; the relay holds no keys.
  control  HTTP. GET /pubkey; POST /sessions/<id>/disclose with the server-direction TLS
           secrets -> verifies the session and returns a signed receipt. Deploy this behind
           HTTPS (e.g. a reverse proxy): disclosed secrets decrypt the response.

The relay refuses client-direction secrets, keeps raw sessions in memory only until they are
disclosed or expire, and writes receipts (not raw traffic) to --receipts.
Run: kremlin-witness --key witness.key --allow api.openai.com --allow openrouter.ai
"""
import argparse
import asyncio
import base64
import fnmatch
import hashlib
import ipaddress
import socket
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from . import receipts, tls12, tls13

DNS_TTL = 60
DEFAULT_TERMS = ('Everything this witness sees is stored indefinitely: responses, disclosed request bodies and '
                 'session metadata (never API keys). The operator may use, analyse, publish and share it for any '
                 'purpose. Do not send personal or confidential data.')
FORBIDDEN = ('client_handshake_traffic_secret', 'client_traffic_secret_0', 'exporter_secret', 'master_secret')


class Session:
    def __init__(self, host, port, submitter=None):
        self.submitter = submitter
        self.id = secrets.token_hex(16)
        self.host, self.port = host, port
        self.opened = datetime.now(timezone.utc)
        self.opened_ns = time.time_ns()
        self.client, self.server = [], []  # [(t_ns, bytes)]
        self.size = 0
        self.closed = asyncio.Event()
        self.error = None


def token_id(token):
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def log(event, **fields):
    """One JSON line per event on stdout: metadata only (never prompts, responses or secrets)."""
    print(json.dumps(dict(t=datetime.now(timezone.utc).isoformat(), event=event, **fields)), flush=True)


class RateLimiter:
    """Sliding one-minute window of session starts per key (token id, or client IP without tokens)."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self.starts = {}

    def allow(self, key):
        if not self.per_minute:
            return True
        now = time.monotonic()
        recent = [t for t in self.starts.get(key, []) if now - t < 60]
        if len(recent) >= self.per_minute:
            self.starts[key] = recent
            return False
        recent.append(now)
        self.starts[key] = recent
        return True


class Relay:
    def __init__(self, key, allow, receipts_dir=None, trust_file=None, max_bytes=256_000_000, ttl_s=900,
                 tokens=None, max_sessions=200, max_per_ip=32, allow_private=False, tokens_file=None,
                 rate_per_minute=0, retention_days=None, quiet=False, allow_ports=(443,), terms=DEFAULT_TERMS,
                 terms_url=None):
        self.terms, self.terms_url = terms, terms_url
        self.allow_ports = set(allow_ports)
        self.allow_private = allow_private
        self.tokens_file, self.tokens_mtime = tokens_file, None
        self.rate = RateLimiter(rate_per_minute)
        self.retention_days = retention_days
        self.quiet = quiet
        self.tokens = {token_id(t) for t in tokens} if tokens else None  # only hashes are kept
        self.reload_tokens()
        from collections import Counter
        self.counters = Counter()
        self.max_sessions, self.max_per_ip = max_sessions, max_per_ip
        self.dns_cache = {}
        self.active = {}
        self.key = key
        self.allow = allow
        self.receipts_dir = Path(receipts_dir) if receipts_dir else None
        self.trust_file = trust_file
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self.sessions = {}

    async def resolve(self, host, port):
        """Resolve once; keep only public addresses (unless allow_private). Connecting to these exact
        addresses leaves no DNS-rebinding window. Validated answers are cached for DNS_TTL seconds."""
        cached = self.dns_cache.get((host, port))
        if cached and cached[0] > time.monotonic():
            return cached[1]
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        out = []
        for _, _, _, _, address in infos:
            ip = ipaddress.ip_address(address[0])
            if self.allow_private or not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                                          or ip.is_multicast or ip.is_unspecified):
                out.append(address[0])
        if not out:
            raise PermissionError(f'{host} resolves only to internal addresses')
        addresses = list(dict.fromkeys(out))
        self.dns_cache[(host, port)] = (time.monotonic() + DNS_TTL, addresses)
        return addresses

    async def connect(self, addresses, port):
        error = None
        for address in addresses:
            try:
                return await asyncio.wait_for(asyncio.open_connection(address, port), 10)
            except (OSError, asyncio.TimeoutError) as failure:
                error = failure
        raise error or OSError('no address')

    def reload_tokens(self):
        """Re-read the tokens file when it changes: revoking a token takes effect without a restart."""
        if not self.tokens_file:
            return
        try:
            mtime = Path(self.tokens_file).stat().st_mtime
        except OSError:
            return
        if mtime != self.tokens_mtime:
            lines = [l.strip() for l in Path(self.tokens_file).read_text().splitlines() if l.strip()]
            self.tokens, self.tokens_mtime = {token_id(t) for t in lines}, mtime
            self.event('tokens_reloaded', count=len(self.tokens))

    def event(self, name, **fields):
        if not self.quiet:
            log(name, **fields)

    def allowed(self, host):
        return any(fnmatch.fnmatch(host.lower(), pattern.lower()) for pattern in self.allow)

    # ---- tunnel -------------------------------------------------------------------------
    async def tunnel(self, reader, writer):
        session = None
        try:
            head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 15)
            lines = head.decode('latin-1').split('\r\n')
            method, target, _ = lines[0].split(' ', 2)
            fields = {k.strip().lower(): v.strip() for k, v in (l.split(':', 1) for l in lines[1:] if ':' in l)}
            submitter = None
            self.reload_tokens()
            if self.tokens is not None:
                supplied = fields.get('proxy-authorization', '')
                supplied = supplied[7:] if supplied.lower().startswith('bearer ') else ''
                submitter = token_id(supplied) if supplied else None
                if submitter not in self.tokens:
                    writer.write(b'HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Bearer\r\n'
                                 b'Content-Length: 0\r\n\r\n')
                    return
            peer = (writer.get_extra_info('peername') or ('?',))[0]
            if not self.rate.allow(submitter or peer):
                self.counters['rate_limited'] += 1
                writer.write(b'HTTP/1.1 429 Too Many Requests\r\nRetry-After: 10\r\nContent-Length: 0\r\n\r\n')
                return
            if sum(self.active.values()) >= self.max_sessions or self.active.get(peer, 0) >= self.max_per_ip:
                writer.write(b'HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\n\r\n')
                return
            host, _, port = target.rpartition(':')
            if method != 'CONNECT' or not host or not port.isdigit():
                writer.write(b'HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n')
                return
            if not self.allowed(host) or int(port) not in self.allow_ports:
                writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                return
            try:
                addresses = await self.resolve(host, int(port))
            except (PermissionError, OSError):
                writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                return
            # Answer at once and connect upstream meanwhile: the client's ClientHello overlaps our TCP connect
            # (it waits in the reader until upstream is ready).
            connecting = asyncio.ensure_future(self.connect(addresses, int(port)))
            session = Session(host, int(port), submitter)
            self.active[peer] = self.active.get(peer, 0) + 1
            self.counters['sessions'] += 1
            self.event('session_open', session=session.id, host=host, port=int(port), token=submitter)
            self.sessions[session.id] = session
            notice = f'X-Witness-Terms: {self.terms_url}\r\n' if self.terms_url else ''
            writer.write(f'HTTP/1.1 200 Connection Established\r\nX-Witness-Session: {session.id}\r\n{notice}\r\n'.encode())
            await writer.drain()
            try:
                upstream_reader, upstream_writer = await connecting
            except (OSError, asyncio.TimeoutError) as error:
                session.error = f'upstream connect failed: {type(error).__name__}'
                return

            # Server->client must never wait on the client: a client that reads slowly would otherwise slow the
            # relay's reads from the server and stretch the witnessed timing. Buffer up to the session limit.
            writer.transport.set_write_buffer_limits(high=self.max_bytes)

            async def pump(src, dst, log):
                while True:
                    try:
                        data = await src.read(65536)
                    except (ConnectionError, OSError):
                        break
                    if not data:
                        break
                    log.append((time.time_ns(), data))
                    session.size += len(data)
                    if session.size > self.max_bytes:
                        session.error = 'session exceeded byte limit'
                        break
                    dst.write(data)
                    if dst is not writer:  # client->server: normal flow control
                        await dst.drain()
                try:
                    if dst.can_write_eof():
                        dst.write_eof()
                    else:  # TLS tunnel: no half-close; one request per tunnel, so close it
                        dst.close()
                except (OSError, RuntimeError):
                    pass

            to_server = asyncio.ensure_future(pump(reader, upstream_writer, session.client))
            to_client = asyncio.ensure_future(pump(upstream_reader, writer, session.server))
            await asyncio.wait({to_server, to_client}, return_when=asyncio.FIRST_COMPLETED)
            if to_server.done() and not to_client.done():
                # the client hung up (it has its complete response): allow in-flight bytes, then end the session
                # instead of waiting for the server to close its side
                await asyncio.wait({to_client}, timeout=0.25)
            for task in (to_server, to_client):
                if not task.done():
                    task.cancel()
            await asyncio.gather(to_server, to_client, return_exceptions=True)
            upstream_writer.close()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError, OSError) as error:
            if session:
                session.error = type(error).__name__
            else:
                try:
                    writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
                except OSError:
                    pass
        finally:
            if session:
                session.closed.set()
                self.event('session_close', session=session.id, host=session.host, error=session.error,
                           client_bytes=sum(len(d) for _, d in session.client),
                           server_bytes=sum(len(d) for _, d in session.server),
                           seconds=round((time.time_ns() - session.opened_ns) / 1e9, 3))
                self.active[peer] -= 1
                if not self.active[peer]:
                    self.active.pop(peer)
            try:
                writer.close()
            except OSError:
                pass

    # ---- control ------------------------------------------------------------------------
    def disclose(self, session, payload):
        if any(k in payload for k in FORBIDDEN):
            raise ValueError('client-direction secrets must never be sent to the witness')
        if payload.get('tls12'):
            result = tls12.verify_session12(session.client, session.server, session.host, payload['server_write_key'],
                                            payload['server_write_iv'], payload['client_random'], session.opened,
                                            self.trust_file)
        else:
            result = tls13.verify_session(
                session.client, session.server, session.host,
                payload['server_handshake_traffic_secret'], payload['server_traffic_secret_0'],
                payload['client_random'], session.opened, self.trust_file, payload.get('request_disclosure'))
        response = result.pop('response')
        request = result.pop('request')
        if request:
            body = request.pop('body')
            request.pop('last_record_ns')
            request.update(body_base64=base64.b64encode(body).decode(), body_sha256=hashlib.sha256(body).hexdigest())
        result.pop('host')  # equals session.host (checked against SNI and the certificate)
        start = session.opened_ns
        records = result.get('response_records') or []
        first_response = records[0]['t_ns'] if records else None
        before = [t for t, _ in session.client if first_response is None or t < first_response]
        receipt = dict(
            session_id=session.id, host=session.host, port=session.port,
            opened_at=session.opened.isoformat(), closed_at=datetime.now(timezone.utc).isoformat(),
            witness_note='Server-to-client half verified and decrypted by the witness; the request is opaque to it.',
            response_base64=base64.b64encode(response).decode(),
            response_sha256=hashlib.sha256(response).hexdigest(),
            response_records=[dict(r, t_ms=(r.pop('t_ns') - start) / 1e6) for r in result.pop('response_records')],
            client_first_byte_ms=(session.client[0][0] - start) / 1e6 if session.client else None,
            request_complete_ms=(before[-1] - start) / 1e6 if before else None,
            client_last_byte_ms=(session.client[-1][0] - start) / 1e6 if session.client else None,
            relay_error=session.error, request=request, submitter_token_id=session.submitter, **result)
        signed = receipts.sign(receipt, self.key)
        if self.receipts_dir:
            self.receipts_dir.mkdir(parents=True, exist_ok=True)
            (self.receipts_dir / f'{session.id}.json').write_text(json.dumps(signed, indent=1))
        return signed

    async def control(self, reader, writer):
        """HTTP/1.1 with keep-alive: a client disclosing many sessions reuses one connection."""
        try:
            while True:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 60)
                except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    return
                if len(head) > 16_384:
                    return self.reply(writer, 431, dict(error='headers too large'), close=True)
                lines = head.decode('latin-1').split('\r\n')
                try:
                    method, path, _ = lines[0].split(' ', 2)
                except ValueError:
                    return self.reply(writer, 400, dict(error='bad request line'), close=True)
                headers = {k.strip().lower(): v.strip() for k, v in (l.split(':', 1) for l in lines[1:] if ':' in l)}
                close = headers.get('connection', '').lower() == 'close'
                length = int(headers.get('content-length', '0') or 0)
                if length > 100_000:
                    return self.reply(writer, 413, dict(error='body too large'), close=True)
                try:
                    body = await asyncio.wait_for(reader.readexactly(length), 15) if length else b''
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    return
                status, obj = await self.handle(method, path, body)
                self.reply(writer, status, obj, close=close)
                await writer.drain()
                if close:
                    return
        finally:
            try:
                writer.close()
            except OSError:
                pass

    async def handle(self, method, path, body):
        if method == 'GET' and path == '/pubkey':
            public = receipts.public_bytes(self.key)
            return 200, dict(algorithm='Ed25519', public_key=base64.b64encode(public).decode(),
                             key_id=receipts.key_id(public), receipt_version=receipts.RECEIPT_VERSION,
                             data_terms=self.terms, terms_url=self.terms_url,
                             stores_receipts=bool(self.receipts_dir), retention_days=self.retention_days)
        if method == 'GET' and path == '/metrics':
            return 200, self.metrics()
        parts = path.strip('/').split('/')
        if method == 'POST' and len(parts) == 3 and parts[0] == 'sessions' and parts[2] == 'disclose':
            session = self.sessions.get(parts[1])
            if not session:
                return 404, dict(error='unknown or expired session')
            try:
                await asyncio.wait_for(session.closed.wait(), 30)
            except asyncio.TimeoutError:
                return 409, dict(error='close the TLS connection before disclosing')
            try:
                payload = json.loads(body)
                signed = self.disclose(session, payload)
                self.counters['receipts'] += 1
                self.event('receipt', session=session.id, host=session.host, token=session.submitter,
                           request_disclosed=bool(signed.get('request')), checks=len(signed['checks']))
                return 200, signed
            except (ValueError, KeyError, tls13.VerificationError) as error:
                self.counters['rejected_disclosures'] += 1
                self.event('disclosure_rejected', session=session.id, host=session.host, reason=str(error)[:200])
                return 422, dict(error=str(error))
            except Exception as error:  # never drop the connection silently
                self.counters['errors'] += 1
                return 500, dict(error=f'{type(error).__name__}: {error}')
            finally:
                self.sessions.pop(session.id, None)  # one disclosure per session; raw bytes dropped
        return 404, dict(error='not found')

    def metrics(self):
        return dict(active_sessions=sum(self.active.values()), pending_disclosures=len(self.sessions),
                    **self.counters)

    @staticmethod
    def reply(writer, status, obj, close=False):
        body = json.dumps(obj).encode()
        writer.write(f'HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                     f'Connection: {"close" if close else "keep-alive"}\r\n\r\n'.encode() + body)

    async def expire(self):
        while True:
            await asyncio.sleep(30)
            cutoff = time.time_ns() - self.ttl_s * 1_000_000_000
            for sid in [s for s, v in self.sessions.items() if v.opened_ns < cutoff]:
                self.sessions.pop(sid, None)
            self.purge_receipts()

    def purge_receipts(self):
        """Delete receipts older than retention_days (they hold responses and disclosed request bodies)."""
        if not (self.receipts_dir and self.retention_days and self.receipts_dir.exists()):
            return 0
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        for path in self.receipts_dir.glob('*.json'):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            self.event('receipts_purged', count=removed)
        return removed

    async def serve(self, tunnel_addr, control_addr, ready=None, tunnel_ssl=None):
        tunnel = await asyncio.start_server(self.tunnel, *tunnel_addr, ssl=tunnel_ssl)
        control = await asyncio.start_server(self.control, *control_addr)
        janitor = asyncio.create_task(self.expire())
        if ready:
            ready(tunnel.sockets[0].getsockname(), control.sockets[0].getsockname())
        async with tunnel, control:
            try:
                await asyncio.gather(tunnel.serve_forever(), control.serve_forever())
            finally:
                janitor.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--key', required=True, help='Ed25519 signing key file (created if missing, mode 600)')
    parser.add_argument('--allow', action='append', default=[], help='host pattern the tunnel may dial (repeatable)')
    parser.add_argument('--tunnel', default='0.0.0.0:8080', help='tunnel listen address')
    parser.add_argument('--control', default='127.0.0.1:8081', help='control listen address (put HTTPS in front)')
    parser.add_argument('--receipts', help='directory to keep signed receipts')
    parser.add_argument('--trust', help='PEM roots for certificate validation (default: certifi)')
    parser.add_argument('--max-mb', type=int, default=256, help='per-session byte limit, both directions (MB)')
    parser.add_argument('--tokens', help='file with one access token per line; CONNECT must send '
                                         'Proxy-Authorization: Bearer TOKEN (receipts record the token id)')
    parser.add_argument('--tunnel-cert', help='serve the tunnel over TLS with this certificate chain (PEM)')
    parser.add_argument('--tunnel-key', help='private key for --tunnel-cert')
    parser.add_argument('--allow-ports', default='443', help='comma-separated destination ports (default 443 only)')
    parser.add_argument('--allow-private', action='store_true',
                        help='allow dialing private/loopback addresses (testing only: otherwise an SSRF risk)')
    parser.add_argument('--rate', type=int, default=0, help='max new sessions per minute per token (or per IP); 0 = off')
    parser.add_argument('--retention-days', type=float, help='delete stored receipts older than this (default: keep forever)')
    parser.add_argument('--terms', default=DEFAULT_TERMS, help='data-use statement shown to clients')
    parser.add_argument('--terms-url', help='URL of the full data terms (sent in a header on every tunnel)')
    parser.add_argument('--quiet', action='store_true', help='no JSON event log on stdout')
    parser.add_argument('--max-sessions', type=int, default=200)
    parser.add_argument('--max-per-ip', type=int, default=32, help='concurrent sessions per client IP (>= client --concurrency)')
    args = parser.parse_args()
    if not args.allow:
        parser.error('at least one --allow host pattern is required')

    def address(text):
        host, _, port = text.rpartition(':')
        return host, int(port)
    relay = Relay(receipts.load_or_create_key(args.key), args.allow, args.receipts, args.trust,
                  max_bytes=args.max_mb * 1_000_000, max_sessions=args.max_sessions, max_per_ip=args.max_per_ip,
                  allow_private=args.allow_private, tokens_file=args.tokens, rate_per_minute=args.rate,
                  retention_days=args.retention_days, quiet=args.quiet,
                  allow_ports=tuple(int(p) for p in args.allow_ports.split(',')), terms=args.terms,
                  terms_url=args.terms_url)
    public = receipts.public_bytes(relay.key)
    print(f'witness key id {receipts.key_id(public)} public {base64.b64encode(public).decode()}', flush=True)
    tunnel_ssl = None
    if args.tunnel_cert:
        import ssl
        tunnel_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tunnel_ssl.load_cert_chain(args.tunnel_cert, args.tunnel_key)
    asyncio.run(relay.serve(address(args.tunnel), address(args.control),
                            lambda t, c: print(f'tunnel {t[0]}:{t[1]}{" (TLS)" if tunnel_ssl else ""}  '
                                               f'control {c[0]}:{c[1]}', flush=True), tunnel_ssl))


if __name__ == '__main__':
    main()
