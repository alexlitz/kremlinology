"""Verify and decrypt the server-to-client half of a recorded TLS 1.3 session.

Inputs are the raw TCP byte streams the relay forwarded (client->server and server->client,
with arrival timestamps) and two disclosed secrets: SERVER_HANDSHAKE_TRAFFIC_SECRET and
SERVER_TRAFFIC_SECRET_0 (RFC 8446 section 7.1). Both are Derive-Secret outputs, so they do
not reveal the handshake or master secret, and the client-to-server keys stay unknown.

Checks performed:
  - ClientHello / ServerHello parse (plaintext); HelloRetryRequest handled per RFC 8446 4.4.1
  - every server handshake record decrypts under the disclosed handshake secret (AEAD tags)
  - Certificate chain validates for the expected host at the session time
  - CertificateVerify: the leaf key signed this session's transcript hash (binds the
    ClientHello random and key share, so the server really took part in this session)
  - server Finished matches HMAC(finished_key, transcript)
  - every application record decrypts under the disclosed application secret (server
    KeyUpdate followed by deriving the next secret, which is one-way)
"""
import hashlib
import hmac
import struct
from datetime import datetime, timezone

from cryptography import x509
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.x509.verification import PolicyBuilder, Store

CHANGE_CIPHER_SPEC, ALERT, HANDSHAKE, APPLICATION_DATA = 20, 21, 22, 23
CLIENT_HELLO, SERVER_HELLO, NEW_SESSION_TICKET, ENCRYPTED_EXTENSIONS = 1, 2, 4, 8
CERTIFICATE, CERTIFICATE_VERIFY, FINISHED, KEY_UPDATE, MESSAGE_HASH = 11, 15, 20, 24, 254
COMPRESSED_CERTIFICATE = 25  # RFC 8879
HRR_RANDOM = hashlib.sha256(b'HelloRetryRequest').digest()
SUITES = {0x1301: ('TLS_AES_128_GCM_SHA256', hashlib.sha256, AESGCM, 16),
          0x1302: ('TLS_AES_256_GCM_SHA384', hashlib.sha384, AESGCM, 32),
          0x1303: ('TLS_CHACHA20_POLY1305_SHA256', hashlib.sha256, ChaCha20Poly1305, 32)}
SIGNATURES = {0x0403: ('ecdsa', hashes.SHA256), 0x0503: ('ecdsa', hashes.SHA384), 0x0603: ('ecdsa', hashes.SHA512),
              0x0804: ('pss', hashes.SHA256), 0x0805: ('pss', hashes.SHA384), 0x0806: ('pss', hashes.SHA512),
              0x0809: ('pss', hashes.SHA256), 0x080a: ('pss', hashes.SHA384), 0x080b: ('pss', hashes.SHA512),
              0x0807: ('ed25519', None)}


class VerificationError(Exception):
    pass


def hkdf_expand_label(secret, label, context, length, hash_fn):
    full = b'tls13 ' + label
    info = struct.pack('>H', length) + bytes([len(full)]) + full + bytes([len(context)]) + context
    out, block, counter = b'', b'', 1
    while len(out) < length:
        block = hmac.new(secret, block + info + bytes([counter]), hash_fn).digest()
        out += block
        counter += 1
    return out[:length]


def records(stream):
    """Split a TLS byte stream into (type, header, payload, end_offset). Trailing partial bytes are ignored."""
    out, i = [], 0
    while i + 5 <= len(stream):
        kind, length = stream[i], struct.unpack('>H', stream[i + 3:i + 5])[0]
        if i + 5 + length > len(stream):
            break
        out.append((kind, stream[i:i + 5], stream[i + 5:i + 5 + length], i + 5 + length))
        i += 5 + length
    return out


class Messages:
    """Reassembles handshake messages from a stream of handshake-record payloads."""

    def __init__(self):
        self.buffer = b''

    def feed(self, data):
        self.buffer += data
        out = []
        while len(self.buffer) >= 4:
            length = int.from_bytes(self.buffer[1:4], 'big')
            if len(self.buffer) < 4 + length:
                break
            out.append(self.buffer[:4 + length])
            self.buffer = self.buffer[4 + length:]
        return out


def _vector(data, offset, size_bytes):
    length = int.from_bytes(data[offset:offset + size_bytes], 'big')
    start = offset + size_bytes
    if start + length > len(data):
        raise VerificationError('truncated vector')
    return data[start:start + length], start + length


def parse_client_hello(message):
    body = message[4:]
    random = body[2:34]
    _, i = _vector(body, 34, 1)          # legacy_session_id
    _, i = _vector(body, i, 2)           # cipher_suites
    _, i = _vector(body, i, 1)           # compression
    extensions, _ = _vector(body, i, 2)
    sni = None
    j = 0
    while j + 4 <= len(extensions):
        kind, length = struct.unpack('>HH', extensions[j:j + 4])
        data = extensions[j + 4:j + 4 + length]
        if kind == 0 and len(data) > 5:  # server_name: list, type 0 (host_name)
            name, _ = _vector(data, 3, 2)
            sni = name.decode('ascii', 'replace')
        j += 4 + length
    return dict(random=random, sni=sni)


def parse_server_hello(message):
    body = message[4:]
    random = body[2:34]
    _, i = _vector(body, 34, 1)
    suite = struct.unpack('>H', body[i:i + 2])[0]
    return dict(random=random, suite=suite, retry=random == HRR_RANDOM)


def parse_certificate(message):
    body = message[4:]
    _, i = _vector(body, 0, 1)           # certificate_request_context
    entries, _ = _vector(body, i, 3)
    certs, j = [], 0
    while j < len(entries):
        der, j = _vector(entries, j, 3)
        _, j = _vector(entries, j, 2)    # per-certificate extensions
        certs.append(x509.load_der_x509_certificate(der))
    if not certs:
        raise VerificationError('empty certificate list')
    return certs


def decompress_certificate(message):
    """RFC 8879 CompressedCertificate -> an equivalent Certificate message (transcript keeps the original)."""
    body = message[4:]
    algorithm = struct.unpack('>H', body[:2])[0]
    expected = int.from_bytes(body[2:5], 'big')
    data, _ = _vector(body, 5, 3)
    if algorithm == 1:
        import zlib
        raw = zlib.decompress(data)
    elif algorithm == 2:
        try:
            import brotli
        except ImportError:
            raise VerificationError('server sent a brotli-compressed certificate: pip install brotli') from None
        raw = brotli.decompress(data)
    elif algorithm == 3:
        try:
            import zstandard
        except ImportError:
            raise VerificationError('server sent a zstd-compressed certificate: pip install zstandard') from None
        raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=expected)
    else:
        raise VerificationError(f'unknown certificate compression algorithm {algorithm}')
    if len(raw) != expected:
        raise VerificationError('decompressed certificate length mismatch')
    return bytes([CERTIFICATE]) + len(raw).to_bytes(3, 'big') + raw


def verify_signature(certificate, scheme, signature, content):
    if scheme not in SIGNATURES:
        raise VerificationError(f'unsupported signature scheme 0x{scheme:04x}')
    kind, algorithm = SIGNATURES[scheme]
    key = certificate.public_key()
    try:
        if kind == 'ecdsa' and isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(signature, content, ec.ECDSA(algorithm()))
        elif kind == 'pss' and isinstance(key, rsa.RSAPublicKey):
            key.verify(signature, content, padding.PSS(padding.MGF1(algorithm()), algorithm().digest_size),
                       algorithm())
        elif kind == 'ed25519' and isinstance(key, ed25519.Ed25519PublicKey):
            key.verify(signature, content)
        else:
            raise VerificationError('signature scheme does not match certificate key')
    except InvalidSignature:
        raise VerificationError('CertificateVerify signature invalid') from None


_ROOTS = {}


def _roots(trust_file):
    """Parsed trust store, cached per file (parsing certifi's bundle costs ~2 ms per verification)."""
    import certifi
    import warnings
    path = trust_file or certifi.where()
    if path not in _ROOTS:
        with open(path, 'rb') as handle, warnings.catch_warnings():
            warnings.simplefilter('ignore')  # some bundled roots have non-positive serials (harmless here)
            _ROOTS[path] = Store(x509.load_pem_x509_certificates(handle.read()))
    return _ROOTS[path]


def verify_chain(certs, host, when, trust_file=None):
    verifier = PolicyBuilder().store(_roots(trust_file)).time(when).build_server_verifier(x509.DNSName(host))
    try:
        chain = verifier.verify(certs[0], certs[1:])
    except Exception as error:  # cryptography raises VerificationError subclasses; surface the reason
        raise VerificationError(f'certificate chain invalid for {host}: {error}') from None
    return [c.fingerprint(hashes.SHA256()).hex() for c in chain]


class Decryptor:
    def __init__(self, secret, suite):
        self.suite = suite
        self.hash_fn = SUITES[suite][1]
        self.secret = secret
        self._keys()

    def _keys(self):
        _, hash_fn, aead_cls, length = SUITES[self.suite]
        key = hkdf_expand_label(self.secret, b'key', b'', length, hash_fn)
        self.iv = hkdf_expand_label(self.secret, b'iv', b'', 12, hash_fn)
        self.aead = aead_cls(key)
        self.seq = 0

    def update(self):
        self.secret = hkdf_expand_label(self.secret, b'traffic upd', b'', self.hash_fn().digest_size, self.hash_fn)
        self._keys()

    def open(self, header, payload):
        nonce = (int.from_bytes(self.iv, 'big') ^ self.seq).to_bytes(12, 'big')
        try:
            inner = self.aead.decrypt(nonce, payload, header)
        except InvalidTag:
            raise VerificationError(f'AEAD tag failed on server record {self.seq}') from None
        self.seq += 1
        stripped = inner.rstrip(b'\x00')
        if not stripped:
            raise VerificationError('record with no content type')
        return stripped[-1], stripped[:-1]


def _timestamped_records(chunks):
    """chunks: [(t_ns, bytes)] in arrival order -> records with the arrival time of their last byte."""
    import bisect
    buffer, ends, times = bytearray(), [], []
    for t, data in chunks:
        buffer += data
        ends.append(len(buffer))
        times.append(t)
    stream = bytes(buffer)
    out = []
    for kind, header, payload, end in records(stream):
        out.append((kind, header, payload, times[bisect.bisect_left(ends, end)]))
    return out, len(stream)


def verify_request_suffix(client_records, suite, secret_hex, layout, host):
    """Decrypt the client's post-KeyUpdate records and check the hidden prefix length.

    client_records: client records after the plaintext ClientHello(s). The first encrypted record is
    the client Finished (handshake key); then the hidden prefix under the first application secret,
    then the KeyUpdate record; then everything under the disclosed secret. layout declares the hidden
    prefix: method, path, auth_header, auth_prefix, key_length.
    """
    encrypted = [(h, p, t) for kind, h, p, t in client_records if kind == APPLICATION_DATA]
    if len(encrypted) < 4:
        raise VerificationError('too few client records for a disclosed request')
    boundary = None
    for index in range(2, len(encrypted)):
        try:
            Decryptor(bytes.fromhex(secret_hex), suite).open(*encrypted[index][:2])
            boundary = index
            break
        except VerificationError:
            continue
    if boundary is None:
        raise VerificationError('disclosed client secret decrypts no client record')
    hidden = encrypted[1:boundary]  # [0] is client Finished; last hidden record is the KeyUpdate
    hidden_plaintext = sum(len(p) - 17 for _, p, _ in hidden) - 5  # 16-byte tag + type byte; KeyUpdate is 5 bytes
    if len(hidden[-1][1]) - 17 != 5:
        raise VerificationError('record before the disclosed suffix is not a KeyUpdate')
    for field in ('method', 'path', 'auth_header', 'auth_prefix', 'key_length'):
        if field not in layout:
            raise VerificationError(f'request layout missing {field}')
    expected = len(f"{layout['method']} {layout['path']} HTTP/1.1\r\nHost: {host}\r\n"
                   f"{layout['auth_header']}: {layout['auth_prefix']}".encode()) + int(layout['key_length']) + 2
    if hidden_plaintext != expected:
        raise VerificationError(f'hidden request prefix is {hidden_plaintext} bytes, layout implies {expected}')
    decryptor = Decryptor(bytes.fromhex(secret_hex), suite)
    suffix, last_t = bytearray(), None
    for header, payload, t in encrypted[boundary:]:
        inner_type, content = decryptor.open(header, payload)
        if inner_type == APPLICATION_DATA:
            suffix += content
            last_t = t
        elif inner_type == ALERT:
            break
        else:
            raise VerificationError('unexpected client record type after KeyUpdate')
    head, sep, body = bytes(suffix).partition(b'\r\n\r\n')
    if not sep:
        raise VerificationError('disclosed suffix has no end of headers')
    headers = []
    for line in head.decode('latin-1').split('\r\n'):
        if ':' not in line:
            raise VerificationError(f'malformed header line in disclosed suffix: {line[:60]!r}')
        name, value = line.split(':', 1)
        headers.append((name.strip(), value.strip()))
    names = {n.lower(): v for n, v in headers}
    if 'transfer-encoding' in names or 'expect' in names:
        raise VerificationError('transfer-encoding / expect not allowed in a disclosed request')
    if names.get('connection', '').lower() != 'close':
        raise VerificationError('disclosed request must be Connection: close (one request per session)')
    if int(names.get('content-length', '-1')) != len(body):
        raise VerificationError('Content-Length does not match the disclosed body (or trailing bytes follow it)')
    return dict(method=layout['method'], path=layout['path'], auth_header=layout['auth_header'],
                key_length=int(layout['key_length']), headers=headers, body=body, last_record_ns=last_t)


def verify_session(client_chunks, server_chunks, host, server_handshake_secret, server_traffic_secret,
                   client_random, session_time=None, trust_file=None, request_disclosure=None):
    """Return verified server-side facts and the decrypted application data, or raise VerificationError."""
    client_records, client_bytes = _timestamped_records(client_chunks)
    server_records, server_bytes = _timestamped_records(server_chunks)
    # Client plaintext handshake messages (ClientHello, and a second one after HelloRetryRequest).
    client_messages, assembler = [], Messages()
    for kind, _, payload, _ in client_records:
        if kind == HANDSHAKE:
            client_messages += assembler.feed(payload)
    hellos = [m for m in client_messages if m[0] == CLIENT_HELLO]
    if not hellos:
        raise VerificationError('no ClientHello in client stream')
    first = parse_client_hello(hellos[0])
    if first['random'].hex() != client_random.lower():
        raise VerificationError('disclosed client_random does not match this session')
    if first['sni'] != host:
        raise VerificationError(f'ClientHello SNI {first["sni"]!r} differs from tunnel host {host!r}')
    # Server plaintext part: ServerHello (or HRR then ServerHello).
    server_plain, assembler, index = [], Messages(), 0
    while index < len(server_records) and server_records[index][0] in (HANDSHAKE, CHANGE_CIPHER_SPEC):
        if server_records[index][0] == HANDSHAKE:
            server_plain += assembler.feed(server_records[index][2])
        index += 1
    hellos_s = [m for m in server_plain if m[0] == SERVER_HELLO]
    if not hellos_s:
        raise VerificationError('no ServerHello in server stream')
    final = parse_server_hello(hellos_s[-1])
    if final['retry']:
        raise VerificationError('session ended after HelloRetryRequest')
    if final['suite'] not in SUITES:
        raise VerificationError(f'unsupported cipher suite 0x{final["suite"]:04x}')
    name, hash_fn, _, _ = SUITES[final['suite']]
    if len(hellos_s) == 2:
        retry = parse_server_hello(hellos_s[0])
        if not retry['retry'] or len(hellos) < 2:
            raise VerificationError('unexpected handshake shape')
        digest = hash_fn(hellos[0]).digest()
        transcript = bytes([MESSAGE_HASH, 0, 0, len(digest)]) + digest + hellos_s[0] + hellos[1] + hellos_s[1]
    elif len(hellos_s) == 1:
        transcript = hellos[0] + hellos_s[0]
    else:
        raise VerificationError('unexpected number of ServerHello messages')
    # Encrypted server handshake.
    decryptor = Decryptor(bytes.fromhex(server_handshake_secret), final['suite'])
    assembler, certs, finished = Messages(), None, False
    while index < len(server_records) and not finished:
        kind, header, payload, _ = server_records[index]
        index += 1
        if kind == CHANGE_CIPHER_SPEC:
            continue
        if kind != APPLICATION_DATA:
            raise VerificationError('plaintext record during encrypted handshake')
        inner_type, content = decryptor.open(header, payload)
        if inner_type != HANDSHAKE:
            raise VerificationError('non-handshake content before server Finished')
        for message in assembler.feed(content):
            kind_m = message[0]
            if kind_m == CERTIFICATE:
                certs = parse_certificate(message)
            elif kind_m == COMPRESSED_CERTIFICATE:
                certs = parse_certificate(decompress_certificate(message))
            elif kind_m == CERTIFICATE_VERIFY:
                if certs is None:
                    raise VerificationError('CertificateVerify before Certificate')
                scheme = struct.unpack('>H', message[4:6])[0]
                signature, _ = _vector(message, 6, 2)
                content_signed = b'\x20' * 64 + b'TLS 1.3, server CertificateVerify\x00' + hash_fn(transcript).digest()
                verify_signature(certs[0], scheme, signature, content_signed)
            elif kind_m == FINISHED:
                finished_key = hkdf_expand_label(bytes.fromhex(server_handshake_secret), b'finished', b'',
                                                 hash_fn().digest_size, hash_fn)
                expected = hmac.new(finished_key, hash_fn(transcript).digest(), hash_fn).digest()
                if not hmac.compare_digest(expected, message[4:]):
                    raise VerificationError('server Finished does not match transcript')
                finished = True
            elif kind_m not in (ENCRYPTED_EXTENSIONS,):
                raise VerificationError(f'unexpected handshake message type {kind_m}')
            transcript += message
    if not finished or certs is None:
        raise VerificationError('server handshake incomplete (no Finished or no Certificate; resumption is not supported)')
    when = session_time or datetime.now(timezone.utc)
    chain = verify_chain(certs, host, when, trust_file)
    # Application data.
    decryptor = Decryptor(bytes.fromhex(server_traffic_secret), final['suite'])
    plaintext, chunks, closed, post_handshake = bytearray(), [], False, Messages()
    for kind, header, payload, t in server_records[index:]:
        if kind == CHANGE_CIPHER_SPEC:
            continue
        if kind != APPLICATION_DATA:
            raise VerificationError('plaintext record after handshake')
        inner_type, content = decryptor.open(header, payload)
        if inner_type == APPLICATION_DATA:
            chunks.append(dict(t_ns=t, offset=len(plaintext), length=len(content)))
            plaintext += content
        elif inner_type == HANDSHAKE:
            for message in post_handshake.feed(content):
                if message[0] == KEY_UPDATE:
                    decryptor.update()
                elif message[0] != NEW_SESSION_TICKET:
                    raise VerificationError(f'unexpected post-handshake message {message[0]}')
        elif inner_type == ALERT:
            closed = True
            break
    client_app = sum(len(p) for k, _, p, _ in client_records if k == APPLICATION_DATA)
    request = None
    if request_disclosure:
        request = verify_request_suffix(client_records, final['suite'], request_disclosure['client_secret_after_update'],
                                        request_disclosure['layout'], host)
        first_response = chunks[0]['t_ns'] if chunks else None
        if first_response is not None and request['last_record_ns'] is not None \
                and first_response < request['last_record_ns']:
            raise VerificationError('server responded before the disclosed request body arrived')
    return dict(request=request, cipher_suite=name, host=host, sni=first['sni'], certificate_chain_sha256=chain,
                leaf_subject=certs[0].subject.rfc4514_string(), hello_retry=len(hellos_s) == 2,
                response=bytes(plaintext), response_records=chunks, close_notify=closed,
                client_bytes=client_bytes, server_bytes=server_bytes, client_encrypted_payload_bytes=client_app,
                checks=['client_hello_matches', 'server_handshake_aead', 'certificate_chain', 'certificate_verify',
                        'server_finished', 'application_aead']
                + (['request_suffix_aead', 'hidden_prefix_length', 'single_request', 'response_after_request']
                   if request else []))
