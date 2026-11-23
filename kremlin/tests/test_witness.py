"""End-to-end witness tests against a local TLS 1.3 server (no network, no keys)."""
import asyncio
import datetime
import json
import socket
import ssl
import threading
import time

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from kremlin.witness import client as witness_client
from kremlin.witness import receipts
from kremlin.witness.relay import Relay

SECRET_KEY = 'sk-test-THIS-MUST-NEVER-REACH-THE-WITNESS'


def make_pki(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'kremlin test CA')])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(1).not_valid_before(now - datetime.timedelta(days=1))
          .not_valid_after(now + datetime.timedelta(days=30))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                       content_commitment=False, key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])).issuer_name(ca_name)
            .public_key(leaf_key.public_key()).serial_number(2)
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    (tmp_path / 'ca.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (tmp_path / 'leaf.pem').write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    (tmp_path / 'leaf.key').write_bytes(leaf_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return tmp_path / 'ca.pem'


def start_server(tmp_path, seen, tls12=False, pause=0.15, close_early=False):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    if tls12:
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    else:
        context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(tmp_path / 'leaf.pem', tmp_path / 'leaf.key')
    listener = socket.create_server(('127.0.0.1', 0))
    port = listener.getsockname()[1]

    def handle(conn):
        try:
            serve_one(conn)
        except (ssl.SSLError, OSError):
            pass  # tests that hang up early

    def serve_one(conn):
        with context.wrap_socket(conn, server_side=True) as tls:
            data = b''
            while b'\r\n\r\n' not in data:
                data += tls.recv(4096)
            head, _, body = data.partition(b'\r\n\r\n')
            length = int(next(l.split(b':')[1] for l in head.split(b'\r\n') if l.lower().startswith(b'content-length')))
            while len(body) < length:
                body += tls.recv(4096)
            seen.append(head + b'\r\n\r\n' + body)
            # one valid JSON object, split across two chunks sent 150 ms apart
            first = json.dumps(dict(id='gen-1', choices=[dict(message=dict(content='hel'))]))[:-1].encode() + b', '
            second = b'"usage":{"prompt_tokens":5,"completion_tokens":2}}'
            tls.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n'
                        + f'{len(first):x}\r\n'.encode() + first + b'\r\n')
            time.sleep(pause)
            if close_early:  # hang up mid-body: the response is incomplete
                return
            tls.sendall(f'{len(second):x}\r\n'.encode() + second + b'\r\n0\r\n\r\n')

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return listener, port


@pytest.fixture
def setup(tmp_path):
    ca = make_pki(tmp_path)
    seen = []
    listener, port = start_server(tmp_path, seen)
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536))
    ready, addresses = threading.Event(), {}

    def run():
        loop = asyncio.new_event_loop()
        addresses['loop'] = loop

        def on_ready(t, c):
            addresses.update(tunnel=t, control=c)
            ready.set()
        loop.run_until_complete(relay.serve(('127.0.0.1', 0), ('127.0.0.1', 0), on_ready))
    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(10)
    tunnel = f"127.0.0.1:{addresses['tunnel'][1]}"
    control = f"http://127.0.0.1:{addresses['control'][1]}"
    client = witness_client.WitnessClient(tunnel, control, trust_file=str(ca))
    yield dict(client=client, port=port, relay=relay, seen=seen, control=control, tunnel=tunnel, ca=ca)
    listener.close()


def witnessed(setup):
    body = json.dumps(dict(model='m', messages=[dict(role='user', content='hi')], max_tokens=2)).encode()
    return setup['client'].request('POST', f"https://localhost:{setup['port']}/v1/chat/completions",
                                   headers={'Authorization': f'Bearer {SECRET_KEY}',
                                            'Content-Type': 'application/json'}, body=body)


def test_witnessed_response_round_trip(setup):
    result = witnessed(setup)
    assert result.status == 200
    assert result.body.startswith(b'{"id": "gen-1"') and result.body.endswith(b'"completion_tokens":2}}')
    receipt = result.receipt
    assert receipt['host'] == 'localhost' and receipt['sni'] == 'localhost'
    assert {'certificate_chain', 'certificate_verify', 'server_finished', 'application_aead'} <= set(receipt['checks'])
    assert receipt['cipher_suite'].startswith('TLS_')
    # the witness saw the 150 ms pause between the two chunks
    times = [r['t_ms'] for r in receipt['response_records']]
    assert len(times) >= 2 and times[-1] - times[0] >= 100
    # the server really got the key; the witness's receipt does not contain it
    assert SECRET_KEY.encode() in setup['seen'][0]
    assert SECRET_KEY not in json.dumps(receipt)


def test_receipt_tamper_detected(setup):
    receipt = dict(witnessed(setup).receipt)
    public = setup['client'].public_key
    receipts.verify(receipt, public)
    forged = dict(receipt, host='evil.example')
    with pytest.raises(ValueError, match='signature'):
        receipts.verify(forged, public)
    other = receipts.public_bytes(Ed25519PrivateKey.generate())
    import base64
    with pytest.raises(ValueError, match='different witness key'):
        receipts.verify(receipt, base64.b64encode(other).decode())


def test_wrong_secret_rejected(setup, monkeypatch):
    real = witness_client.server_secrets

    def corrupt(path):
        random, found = real(path)
        flipped = found['server_traffic_secret_0']
        found['server_traffic_secret_0'] = ('0' if flipped[0] != '0' else '1') + flipped[1:]
        return random, found
    monkeypatch.setattr(witness_client, 'server_secrets', corrupt)
    with pytest.raises(witness_client.WitnessError, match='AEAD tag failed'):
        witnessed(setup)


def test_client_secrets_refused(setup):
    sock = socket.create_connection(tuple(setup['tunnel'].rsplit(':', 1)[0:1]) + (int(setup['tunnel'].rsplit(':')[1]),))
    sock.sendall(f"CONNECT localhost:{setup['port']} HTTP/1.1\r\n\r\n".encode())
    head = sock.recv(4096).decode()
    session = next(l.split(':', 1)[1].strip() for l in head.split('\r\n') if l.lower().startswith('x-witness-session'))
    sock.close()
    reply = httpx.post(f"{setup['control']}/sessions/{session}/disclose",
                       json=dict(client_random='00', client_traffic_secret_0='aa',
                                 server_handshake_traffic_secret='aa', server_traffic_secret_0='aa'))
    assert reply.status_code == 422 and 'never' in reply.json()['error']


def test_disallowed_host(setup):
    with pytest.raises(witness_client.WitnessError, match='refused'):
        setup['client'].request('GET', 'https://example.com/')


def test_keylog_parser_ignores_client_secrets(tmp_path):
    log = tmp_path / 'k.log'
    log.write_text('CLIENT_HANDSHAKE_TRAFFIC_SECRET r1 c1\nSERVER_HANDSHAKE_TRAFFIC_SECRET r1 s1\n'
                   'CLIENT_TRAFFIC_SECRET_0 r1 c2\nSERVER_TRAFFIC_SECRET_0 r1 s2\nEXPORTER_SECRET r1 e\n')
    random, found = witness_client.server_secrets(log)
    assert random == 'r1' and found == dict(server_handshake_traffic_secret='s1', server_traffic_secret_0='s2')


def disclosed(setup, **overrides):
    body = json.dumps(dict(model='m', messages=[dict(role='user', content='hi')], max_tokens=2)).encode()
    return setup['client'].request('POST', f"https://localhost:{setup['port']}/v1/chat/completions",
                                   headers={'Authorization': f'Bearer {SECRET_KEY}', 'Content-Type': 'application/json'},
                                   body=body, disclose_request=True), body


def test_request_disclosure_round_trip(setup):
    result, body = disclosed(setup)
    request = result.receipt['request']
    import base64
    assert base64.b64decode(request['body_base64']) == body and request['path'] == '/v1/chat/completions'
    assert request['method'] == 'POST' and request['key_length'] == len(SECRET_KEY)
    assert ('Content-Type', 'application/json') in [tuple(h) for h in request['headers']]
    assert {'request_suffix_aead', 'hidden_prefix_length', 'response_after_request'} <= set(result.receipt['checks'])
    assert SECRET_KEY not in json.dumps(result.receipt)
    assert setup['seen'][-1].endswith(body) and SECRET_KEY.encode() in setup['seen'][-1]


def test_request_disclosure_rejects_wrong_layout(setup, monkeypatch):
    real = witness_client.WitnessClient._disclosing_tls

    def lie(self, *args):
        raw, random, secrets_, request = real(self, *args)
        request['layout']['key_length'] -= 12  # pretend the hidden part was shorter (room for a smuggled request)
        return raw, random, secrets_, request
    monkeypatch.setattr(witness_client.WitnessClient, '_disclosing_tls', lie)
    with pytest.raises(witness_client.WitnessError, match='hidden request prefix'):
        disclosed(setup)


def test_request_disclosure_rejects_wrong_secret(setup, monkeypatch):
    real = witness_client.WitnessClient._disclosing_tls

    def lie(self, *args):
        raw, random, secrets_, request = real(self, *args)
        request['client_secret_after_update'] = '00' * 32
        return raw, random, secrets_, request
    monkeypatch.setattr(witness_client.WitnessClient, '_disclosing_tls', lie)
    with pytest.raises(witness_client.WitnessError, match='decrypts no client record'):
        disclosed(setup)


def test_tokens_and_limits(tmp_path):
    ca = make_pki(tmp_path)
    seen = []
    listener, port = start_server(tmp_path, seen)
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536), tokens=['good-token'])
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    tunnel, control = f"127.0.0.1:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}"
    bad = witness_client.WitnessClient(tunnel, control, trust_file=str(ca), token='wrong')
    with pytest.raises(witness_client.WitnessError, match='407'):
        bad.request('GET', f'https://localhost:{port}/')
    good = witness_client.WitnessClient(tunnel, control, trust_file=str(ca), token='good-token')
    body = b'{"x":1}'
    result = good.request('POST', f'https://localhost:{port}/v1/x', headers={'Authorization': 'Bearer k'}, body=body)
    from kremlin.witness.relay import token_id
    assert result.receipt['submitter_token_id'] == token_id('good-token')
    listener.close()


@pytest.mark.parametrize('disclose', [False, True])
def test_tls_tunnel(tmp_path, disclose):
    ca = make_pki(tmp_path)
    seen = []
    listener, port = start_server(tmp_path, seen)
    tunnel_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tunnel_ssl.load_cert_chain(tmp_path / 'leaf.pem', tmp_path / 'leaf.key')
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536), tokens=['t0k'])
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()), tunnel_ssl)),
        daemon=True).start()
    assert ready.wait(10)
    client = witness_client.WitnessClient(f"tls://localhost:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}",
                                          trust_file=str(ca), relay_trust_file=str(ca), token='t0k')
    result = client.request('POST', f'https://localhost:{port}/v1/x',
                            headers={'Authorization': f'Bearer {SECRET_KEY}'}, body=b'{"a":1}',
                            disclose_request=disclose)
    assert result.status == 200 and 'certificate_verify' in result.receipt['checks']
    assert bool(result.receipt.get('request')) == disclose
    listener.close()


@pytest.mark.parametrize('cipher', ['ECDHE-ECDSA-AES128-GCM-SHA256', 'ECDHE-ECDSA-AES256-GCM-SHA384',
                                    'ECDHE-ECDSA-CHACHA20-POLY1305'])
def test_tls12_response_attestation(tmp_path, cipher, monkeypatch):
    ca = make_pki(tmp_path)
    seen = []
    real_ctx = ssl.SSLContext

    listener, port = start_server(tmp_path, seen, tls12=True)
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536))
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    original = ssl.create_default_context

    def pinned(*a, **k):
        context = original(*a, **k)
        context.set_ciphers(cipher)
        return context
    monkeypatch.setattr(witness_client.ssl, 'create_default_context', pinned)
    client = witness_client.WitnessClient(f"127.0.0.1:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}",
                                          trust_file=str(ca), public_key=None)
    result = client.request('POST', f'https://localhost:{port}/v1/x', headers={'Authorization': f'Bearer {SECRET_KEY}'},
                            body=b'{"a":1}')
    assert result.status == 200 and 'tls12' in result.receipt['checks']
    assert 'server_key_exchange_signature' in result.receipt['checks']
    assert result.receipt['cipher_suite'].startswith('TLS12_')
    times = [r['t_ms'] for r in result.receipt['response_records']]
    assert times[-1] - times[0] >= 100
    assert SECRET_KEY not in json.dumps(result.receipt)
    listener.close()


def test_disclosure_header_variants_and_query_keys(setup):
    url = f"https://localhost:{setup['port']}/v1/x"
    result = setup['client'].request('POST', url, headers={'x-goog-api-key': SECRET_KEY}, body=b'{}',
                                     disclose_request=True)
    assert result.receipt['request']['auth_header'] == 'x-goog-api-key' and SECRET_KEY not in json.dumps(result.receipt)
    with pytest.raises(witness_client.WitnessError, match='URL query carries a credential'):
        setup['client'].request('POST', url + '?key=abc', headers={'api-key': 'x'}, body=b'{}', disclose_request=True)


def test_internal_addresses_refused(tmp_path):
    ca = make_pki(tmp_path)
    seen = []
    listener, port = start_server(tmp_path, seen)
    relay = Relay(Ed25519PrivateKey.generate(), ['*'], trust_file=str(ca))  # wide allowlist, private not allowed
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    client = witness_client.WitnessClient(f"127.0.0.1:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}",
                                          trust_file=str(ca))
    for target in (f'https://localhost:{port}/', 'https://127.0.0.1:22/', 'https://169.254.169.254/'):
        with pytest.raises(witness_client.WitnessError, match='403'):
            client.request('GET', target)
    listener.close()


def test_slow_reader_cannot_stretch_witnessed_timing(tmp_path, monkeypatch):
    """The server streams 48 MB as fast as it can. A client reading slowly must not stretch the arrival timeline the
    witness records (it would, if the relay stopped reading from the server whenever the client lagged and the
    kernel buffers filled), so slow reading cannot fake slow decoding."""
    ca = make_pki(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(tmp_path / 'leaf.pem', tmp_path / 'leaf.key')
    listener = socket.create_server(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    big = b'x' * 48_000_000

    def serve():
        conn, _ = listener.accept()
        with context.wrap_socket(conn, server_side=True) as tls:
            data = b''
            while b'\r\n\r\n' not in data:
                data += tls.recv(4096)
            tls.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n' % (len(big) + 5) + big)
            time.sleep(0.15)
            tls.sendall(b'tail!')
    threading.Thread(target=serve, daemon=True).start()
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536))
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    original = witness_client.BioTLS.recv
    stalled = []

    def slow_recv(self, size):
        if not stalled:  # stall 3 s before reading anything, then read at full speed
            stalled.append(1)
            time.sleep(3)
        return original(self, size)
    monkeypatch.setattr(witness_client.BioTLS, 'recv', slow_recv)
    client = witness_client.WitnessClient(f"127.0.0.1:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}",
                                          trust_file=str(ca))
    result = client.request('GET', f'https://localhost:{port}/big')
    assert result.body.endswith(b'tail!')
    records = result.receipt['response_records']
    span = records[-1]['t_ms'] - records[0]['t_ms']
    assert span < 2000, f'witnessed stream took {span:.0f} ms: the stalled reader stretched it'
    listener.close()


def test_relay_ops_revocation_rate_retention_logs(tmp_path, capsys):
    ca = make_pki(tmp_path)
    seen = []
    listener, port = start_server(tmp_path, seen)
    tokens = tmp_path / 'tokens.txt'
    tokens.write_text('alice\nbob\n')
    receipts_dir = tmp_path / 'receipts'
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], receipts_dir=receipts_dir, trust_file=str(ca),
                  allow_private=True, allow_ports=range(1, 65536), tokens_file=str(tokens), rate_per_minute=2, retention_days=1)
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)

    def client_for(token):
        return witness_client.WitnessClient(f"127.0.0.1:{addr['t'][1]}", f"http://127.0.0.1:{addr['c'][1]}",
                                            trust_file=str(ca), token=token)
    url = f'https://localhost:{port}/v1/x'
    headers = {'Authorization': f'Bearer {SECRET_KEY}'}
    bob = client_for('bob')
    bob.request('POST', url, headers=headers, body=b'{}')
    tokens.write_text('alice\n')                     # revoke bob
    import os
    os.utime(tokens, (time.time() + 5, time.time() + 5))
    with pytest.raises(witness_client.WitnessError, match='407'):
        bob.request('POST', url, headers=headers, body=b'{}')
    alice = client_for('alice')
    alice.request('POST', url, headers=headers, body=b'{}')
    alice.request('POST', url, headers=headers, body=b'{}')
    with pytest.raises(witness_client.WitnessError, match='429'):  # 3rd session within a minute
        alice.request('POST', url, headers=headers, body=b'{}')
    old = sorted(receipts_dir.glob('*.json'))[0]
    os.utime(old, (time.time() - 3 * 86400, time.time() - 3 * 86400))
    assert relay.purge_receipts() == 1 and not old.exists() and len(list(receipts_dir.glob('*.json'))) == 2
    logs = capsys.readouterr().out
    events = [json.loads(l)['event'] for l in logs.splitlines() if l.startswith('{')]
    assert {'session_open', 'session_close', 'receipt', 'tokens_reloaded', 'receipts_purged'} <= set(events)
    assert SECRET_KEY not in logs and '"id": "gen-1"' not in logs  # metadata only
    assert relay.metrics()['rate_limited'] == 1
    listener.close()


def test_only_allowed_ports(tmp_path):
    ca = make_pki(tmp_path)
    relay = Relay(Ed25519PrivateKey.generate(), ['*'], trust_file=str(ca))  # default: port 443 only
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    sock = socket.create_connection(('127.0.0.1', addr['t'][1]))
    sock.sendall(b'CONNECT example.com:25 HTTP/1.1\r\n\r\n')
    assert b'403' in sock.recv(200)
    sock.close()


def relay_setup(tmp_path, **server):
    """A relay and a test server with custom behaviour; returns (client factory, server port)."""
    ca = make_pki(tmp_path)
    listener, port = start_server(tmp_path, [], **server)
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True,
                  allow_ports=range(1, 65536))
    ready, addresses = threading.Event(), {}

    def run():
        loop = asyncio.new_event_loop()

        def on_ready(t, c):
            addresses.update(tunnel=t, control=c)
            ready.set()
        loop.run_until_complete(relay.serve(('127.0.0.1', 0), ('127.0.0.1', 0), on_ready))
    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(10)

    def make_client(**kwargs):
        return witness_client.WitnessClient(f"127.0.0.1:{addresses['tunnel'][1]}",
                                            f"http://127.0.0.1:{addresses['control'][1]}", trust_file=str(ca), **kwargs)
    return make_client, port, listener


def post(client, port, disclose):
    body = json.dumps(dict(model='m', messages=[dict(role='user', content='hi')])).encode()
    return client.request('POST', f'https://localhost:{port}/v1/chat/completions',
                          headers={'Authorization': f'Bearer {SECRET_KEY}', 'Content-Type': 'application/json'},
                          body=body, disclose_request=disclose)


@pytest.mark.parametrize('disclose', [False, True])
def test_read_timeout_is_an_error_not_a_short_response(tmp_path, disclose):
    """A stalled upstream (pause longer than the read timeout) must fail loudly. It used to be treated as the end of the
    response: the truncated bytes were submitted and either accepted or reported as a witness mismatch."""
    make_client, port, listener = relay_setup(tmp_path, pause=1.5)
    try:
        with pytest.raises(witness_client.WitnessError, match='timed out'):
            post(make_client(timeout=1), port, disclose)
    finally:
        listener.close()


@pytest.mark.parametrize('disclose', [False, True])
def test_connection_closed_mid_body_is_an_error(tmp_path, disclose):
    make_client, port, listener = relay_setup(tmp_path, close_early=True)
    try:
        with pytest.raises(witness_client.WitnessError, match='before the response was complete'):
            post(make_client(), port, disclose)
    finally:
        listener.close()


@pytest.mark.parametrize('disclose', [False, True])
def test_slow_upstream_within_timeout_is_witnessed(tmp_path, disclose):
    make_client, port, listener = relay_setup(tmp_path, pause=1.5)
    try:
        result = post(make_client(timeout=5), port, disclose)
        assert result.status == 200 and result.body.endswith(b'"completion_tokens":2}}')
    finally:
        listener.close()
