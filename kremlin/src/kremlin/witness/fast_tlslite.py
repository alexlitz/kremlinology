"""Give tlslite-ng (used only for request disclosure) OpenSSL-backed AEADs via the `cryptography` package.

tlslite-ng falls back to pure-Python AES-GCM / ChaCha20-Poly1305 unless M2Crypto or PyCrypto is installed; that
made disclosure CPU-bound (~8 requests/s). The record layer only needs objects with seal/open, so we hand it
cryptography's AESGCM and ChaCha20Poly1305. Call install() once before opening tlslite connections.
"""
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

_INSTALLED = False


class _AEAD:
    isBlockCipher = False
    isAEAD = True
    nonceLength = 12
    tagLength = 16
    implementation = 'openssl'

    def __init__(self, cls, key, name):
        self.key = key
        self.name = name
        self._aead = cls(bytes(key))

    def seal(self, nonce, plaintext, data):
        return bytearray(self._aead.encrypt(bytes(nonce), bytes(plaintext), bytes(data)))

    def open(self, nonce, ciphertext, data):
        try:
            return bytearray(self._aead.decrypt(bytes(nonce), bytes(ciphertext), bytes(data)))
        except InvalidTag:
            return None


def aesgcm(key, implList=None):
    return _AEAD(AESGCM, key, 'aes128gcm' if len(key) == 16 else 'aes256gcm')


def chacha20(key, implList=None):
    return _AEAD(ChaCha20Poly1305, key, 'chacha20-poly1305')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import tlslite.recordlayer as recordlayer
    import tlslite.utils.cipherfactory as cipherfactory
    for module in (recordlayer, cipherfactory):
        module.createAESGCM = aesgcm
        module.createCHACHA20 = chacha20
    _INSTALLED = True
