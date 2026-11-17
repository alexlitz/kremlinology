"""Exact token IDs from reported token bytes, for byte-level BPE vocabularies.

Byte-level BPE (GPT-2, Llama 3, Qwen 2) stores each vocabulary entry as a
distinct byte string written in the bytes-to-unicode alphabet, so reported
bytes identify one token ID. Reads a Hugging Face tokenizer.json directly (no
tokenizers dependency). Bytes with no vocabulary entry are reporting anomalies
for that tokenizer. The tokenizer must be the served model's: identity holds
only if template and token counts were independently checked to match.
"""

import hashlib
import json
from pathlib import Path


def bytes_to_unicode():
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, characters)))


class ByteVocabulary:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        model = data.get("model") or {}
        if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
            raise ValueError("byte-level BPE tokenizer.json required")
        decoder = {c: b for b, c in bytes_to_unicode().items()}
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.ids = {}
        for piece, token in model["vocab"].items():
            if all(c in decoder for c in piece):
                self.ids[bytes(decoder[c] for c in piece)] = token
        # Special tokens are stored literally, not in the byte alphabet.
        self.special = {t["content"].encode(): t["id"] for t in data.get("added_tokens", [])}
        if len(self.ids) < len(model["vocab"]) * 0.99:
            raise ValueError("vocabulary is not byte-level")

    def token_id(self, raw):
        """Token ID for reported bytes, or None if no vocabulary entry matches."""
        raw = bytes(raw)
        return self.ids.get(raw, self.special.get(raw))
