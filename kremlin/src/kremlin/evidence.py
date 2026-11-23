"""Evidence directory: every request and response is kept, keys redacted, files hashed.

Layout of a run directory:
  run.json         command, arguments, tool version, start/end time, spend
  requests.jsonl   one record per HTTP attempt (including failures and retries)
  captures/        raw response bodies (redacted), one file per attempt
  results.json     the investigation's computed results
  report.md        human-readable summary
  manifest.json    SHA-256 of every file above
"""
import base64
import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def now():
    return datetime.now(timezone.utc).isoformat()


def redact(data, secrets):
    for secret in secrets:
        if secret:
            token = secret.encode() if isinstance(data, bytes) else secret
            data = data.replace(token, b'[REDACTED]' if isinstance(data, bytes) else '[REDACTED]')
    return data


class Evidence:
    def __init__(self, path, command, args, secrets=(), resume=False):
        self.root = Path(path)
        self.lock = threading.Lock()
        self.secrets = [s for s in secrets if s]
        if resume:
            self.meta = json.loads((self.root / 'run.json').read_text())
            self.meta.setdefault('resumed_at', []).append(now())
            (self.root / 'manifest.json').unlink(missing_ok=True)
            self.count = len(list((self.root / 'captures').glob('*.json')))
            self.write_json('run.json', self.meta)
            return
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f'{self.root} exists and is not empty; choose a new --out directory (or --resume)')
        (self.root / 'captures').mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.meta = dict(tool='kremlin', version=__version__, command=command, args=args, started_at=now(),
                         python=sys.version.split()[0])
        self.write_json('run.json', self.meta)

    def records(self):
        path = self.root / 'requests.jsonl'
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

    def capture(self, name):
        data = json.loads((self.root / 'captures' / f'{name}.json').read_text())
        return base64.b64decode(data['body_base64'])

    def write_json(self, name, obj):
        raw = json.dumps(obj, indent=1, default=str).encode()
        (self.root / name).write_bytes(redact(raw, self.secrets))

    def write_text(self, name, text):
        (self.root / name).write_text(redact(text, self.secrets))

    def record(self, record, body=b''):
        safe = redact(body or b'', self.secrets)
        line = None
        with self.lock:
            name = f'{self.count:06d}'
            self.count += 1
            (self.root / 'captures' / f'{name}.json').write_text(json.dumps(dict(
                body_base64=base64.b64encode(safe).decode(), sha256=hashlib.sha256(safe).hexdigest())))
            line = redact(json.dumps(dict(record, capture=name), default=str).encode(), self.secrets) + b'\n'
            with (self.root / 'requests.jsonl').open('ab') as handle:
                handle.write(line)
        return name

    def close(self, **extra):
        self.meta.update(finished_at=now(), **extra)
        self.write_json('run.json', self.meta)
        files = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(self.root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        self.write_json('manifest.json', dict(schema_version=1, files=files,
                                              scope='File integrity only; not signed provenance.'))


def verify(path):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text())
    bad = [name for name, digest in manifest['files'].items()
           if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    return dict(files=len(manifest['files']), mismatched=bad, ok=not bad)
