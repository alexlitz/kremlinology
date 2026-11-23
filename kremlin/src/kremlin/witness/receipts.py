"""Witness receipts: canonical JSON signed with the relay's Ed25519 key.

A receipt states what the relay itself observed and verified for one tunnelled TLS session:
host and port it dialled, certificate chain (validated), cipher suite, timestamps of every
server record, byte counts in both directions, and the decrypted server response. It never
contains client-to-server plaintext (the relay cannot decrypt it).
"""
import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RECEIPT_VERSION = 'kremlin-witness/1'


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def load_or_create_key(path):
    path = Path(path)
    if path.exists():
        return Ed25519PrivateKey.from_private_bytes(path.read_bytes())
    key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                       serialization.NoEncryption()))
    return key


def public_bytes(key):
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(public):
    return hashlib.sha256(public).hexdigest()[:16]


def sign(receipt, key):
    body = dict(receipt, version=RECEIPT_VERSION, witness_key_id=key_id(public_bytes(key)))
    signature = key.sign(canonical(body))
    return dict(body, signature=base64.b64encode(signature).decode())


def verify(receipt, public_key_b64):
    """Check the signature and internal digests; returns the decoded response bytes."""
    public = base64.b64decode(public_key_b64)
    body = {k: v for k, v in receipt.items() if k != 'signature'}
    if body.get('version') != RECEIPT_VERSION:
        raise ValueError(f'unknown receipt version {body.get("version")!r}')
    if body.get('witness_key_id') != key_id(public):
        raise ValueError('receipt was signed by a different witness key')
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(base64.b64decode(receipt['signature']), canonical(body))
    except InvalidSignature:
        raise ValueError('receipt signature invalid') from None
    response = base64.b64decode(body['response_base64'])
    if hashlib.sha256(response).hexdigest() != body['response_sha256']:
        raise ValueError('response digest mismatch')
    return response
