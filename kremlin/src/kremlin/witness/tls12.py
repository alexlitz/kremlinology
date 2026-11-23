"""Verify and decrypt the server-to-client half of a recorded TLS 1.2 (ECDHE, AEAD) session.

Disclosed: server_write_key and server_write_IV only (TLS 1.2 key block, RFC 5246 6.3). They are
separate PRF output bytes from the client-side keys, so the request stays sealed. Checks:
  - ServerHello / Certificate / ServerKeyExchange parse (plaintext in TLS 1.2)
  - certificate chain valid for the host at the session time
  - ServerKeyExchange signature over client_random + server_random + ECDHE params (binds the server
    to this session's randoms and key share)
  - the server Finished and every application record authenticate under the disclosed keys
Request disclosure is not available for TLS 1.2 (no KeyUpdate).
"""
import struct
from datetime import datetime, timezone

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from cryptography import x509

from .tls13 import (ALERT, APPLICATION_DATA, CHANGE_CIPHER_SPEC, HANDSHAKE, Messages, VerificationError,
                    _timestamped_records, _vector, parse_client_hello, verify_chain)

SERVER_HELLO, CERTIFICATE, SERVER_KEY_EXCHANGE = 2, 11, 12
SUITES12 = {0xC02B: ('ECDHE_ECDSA_AES128_GCM_SHA256', 'gcm', 16), 0xC02F: ('ECDHE_RSA_AES128_GCM_SHA256', 'gcm', 16),
            0xC02C: ('ECDHE_ECDSA_AES256_GCM_SHA384', 'gcm', 32), 0xC030: ('ECDHE_RSA_AES256_GCM_SHA384', 'gcm', 32),
            0xCCA8: ('ECDHE_RSA_CHACHA20_POLY1305', 'chacha', 32),
            0xCCA9: ('ECDHE_ECDSA_CHACHA20_POLY1305', 'chacha', 32)}
HASHES = {4: hashes.SHA256, 5: hashes.SHA384, 6: hashes.SHA512, 2: hashes.SHA1}
PSS = {0x0804: hashes.SHA256, 0x0805: hashes.SHA384, 0x0806: hashes.SHA512}


def key_material(master_secret, client_random, server_random, suite):
    """Client-side helper: TLS 1.2 key expansion -> (server_write_key, server_write_IV). Never send the master secret."""
    import hashlib
    import hmac
    hash_fn = hashlib.sha384 if suite in (0xC02C, 0xC030) else hashlib.sha256
    _, mode, key_len = SUITES12[suite]
    iv_len = 4 if mode == 'gcm' else 12
    seed = b'key expansion' + server_random + client_random
    out, a = b'', seed
    while len(out) < 2 * key_len + 2 * iv_len:
        a = hmac.new(master_secret, a, hash_fn).digest()
        out += hmac.new(master_secret, a + seed, hash_fn).digest()
    server_key = out[key_len:2 * key_len]           # AEAD suites: no MAC keys
    server_iv = out[2 * key_len + iv_len:2 * key_len + 2 * iv_len]
    return server_key, server_iv


def parse_server_hello12(message):
    body = message[4:]
    version = struct.unpack('>H', body[:2])[0]
    random = body[2:34]
    _, i = _vector(body, 34, 1)
    suite = struct.unpack('>H', body[i:i + 2])[0]
    return dict(version=version, random=random, suite=suite)


def verify_session12(client_chunks, server_chunks, host, server_write_key, server_write_iv, client_random,
                     session_time=None, trust_file=None):
    client_records, client_bytes = _timestamped_records(client_chunks)
    server_records, server_bytes = _timestamped_records(server_chunks)
    assembler, client_messages = Messages(), []
    for kind, _, payload, _ in client_records:
        if kind == HANDSHAKE:
            client_messages += assembler.feed(payload)
            break
    hello = parse_client_hello(client_messages[0])
    if hello['random'].hex() != client_random.lower() or hello['sni'] != host:
        raise VerificationError('ClientHello does not match the disclosure / tunnel host')
    assembler, messages, index = Messages(), [], 0
    while index < len(server_records) and server_records[index][0] == HANDSHAKE:
        messages += assembler.feed(server_records[index][2])
        index += 1
    by_type = {m[0]: m for m in messages}
    if SERVER_HELLO not in by_type or CERTIFICATE not in by_type or SERVER_KEY_EXCHANGE not in by_type:
        raise VerificationError('TLS 1.2 handshake incomplete (need ServerHello, Certificate, ServerKeyExchange)')
    server_hello = parse_server_hello12(by_type[SERVER_HELLO])
    if server_hello['version'] != 0x0303 or server_hello['suite'] not in SUITES12:
        raise VerificationError(f"unsupported TLS 1.2 parameters (suite 0x{server_hello['suite']:04x})")
    entries, _ = _vector(by_type[CERTIFICATE][4:], 0, 3)
    certs, j = [], 0
    while j < len(entries):
        der, j = _vector(entries, j, 3)
        certs.append(x509.load_der_x509_certificate(der))
    chain = verify_chain(certs, host, session_time or datetime.now(timezone.utc), trust_file)
    ske = by_type[SERVER_KEY_EXCHANGE][4:]
    if ske[0] != 3:
        raise VerificationError('ServerKeyExchange is not named-curve ECDHE')
    point_len = ske[3]
    params = ske[:4 + point_len]
    scheme = struct.unpack('>H', ske[4 + point_len:6 + point_len])[0]
    signature, _ = _vector(ske, 6 + point_len, 2)
    signed = hello['random'] + server_hello['random'] + params
    key = certs[0].public_key()
    try:
        if scheme in PSS and isinstance(key, rsa.RSAPublicKey):
            key.verify(signature, signed, padding.PSS(padding.MGF1(PSS[scheme]()), PSS[scheme]().digest_size),
                       PSS[scheme]())
        elif isinstance(key, rsa.RSAPublicKey) and (scheme & 0xff) == 1:
            key.verify(signature, signed, padding.PKCS1v15(), HASHES[scheme >> 8]())
        elif isinstance(key, ec.EllipticCurvePublicKey) and (scheme & 0xff) == 3:
            key.verify(signature, signed, ec.ECDSA(HASHES[scheme >> 8]()))
        else:
            raise VerificationError(f'unsupported ServerKeyExchange signature 0x{scheme:04x}')
    except InvalidSignature:
        raise VerificationError('ServerKeyExchange signature invalid') from None
    _, mode, _ = SUITES12[server_hello['suite']]
    aead = AESGCM(bytes.fromhex(server_write_key)) if mode == 'gcm' else ChaCha20Poly1305(bytes.fromhex(server_write_key))
    iv = bytes.fromhex(server_write_iv)
    seq, finished, plaintext, chunks, closed = 0, False, bytearray(), [], False
    while index < len(server_records) and server_records[index][0] == CHANGE_CIPHER_SPEC:
        index += 1
    for kind, header, payload, t in server_records[index:]:
        if kind == CHANGE_CIPHER_SPEC:
            continue
        if mode == 'gcm':
            nonce, ciphertext = iv + payload[:8], payload[8:]
        else:
            nonce, ciphertext = (int.from_bytes(iv, 'big') ^ seq).to_bytes(12, 'big'), payload
        aad = struct.pack('>Q', seq) + bytes([kind]) + header[1:3] + struct.pack('>H', len(ciphertext) - 16)
        try:
            content = aead.decrypt(nonce, ciphertext, aad)
        except InvalidTag:
            raise VerificationError(f'AEAD tag failed on server record {seq}') from None
        seq += 1
        if kind == HANDSHAKE and not finished:
            if content[0] != 20:
                raise VerificationError('first encrypted server record is not Finished')
            finished = True
        elif kind == APPLICATION_DATA:
            chunks.append(dict(t_ns=t, offset=len(plaintext), length=len(content)))
            plaintext += content
        elif kind == ALERT:
            closed = True
            break
    if not finished:
        raise VerificationError('no server Finished')
    client_app = sum(len(p) for k, _, p, _ in client_records if k == APPLICATION_DATA)
    return dict(request=None, cipher_suite='TLS12_' + SUITES12[server_hello['suite']][0], host=host, sni=hello['sni'],
                certificate_chain_sha256=chain, leaf_subject=certs[0].subject.rfc4514_string(), hello_retry=False,
                response=bytes(plaintext), response_records=chunks, close_notify=closed, client_bytes=client_bytes,
                server_bytes=server_bytes, client_encrypted_payload_bytes=client_app,
                checks=['client_hello_matches', 'certificate_chain', 'server_key_exchange_signature',
                        'server_finished_aead', 'application_aead', 'tls12'])
