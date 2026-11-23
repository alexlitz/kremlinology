"""kremlin-hub: accept evidence bundles, audit them against trusted witness keys, publish what passes.

  kremlin-hub --dir hub-data --witness-key PUBKEY [--witness-key OLDKEY] --listen 127.0.0.1:8090

POST /runs        body: a bundle (tar.gz from `kremlin bundle`). Queued (202, {job, poll}); a worker unpacks it
                  safely, runs `kremlin audit --require-witness` against each trusted key, and publishes it only
                  if every request is witnessed, every receipt verifies and the results reproduce.
GET  /jobs/ID     queued | running | accepted (with the index entry) | rejected (with the reason)
GET  /            HTML index of accepted runs.
GET  /runs        JSON index (?offset=&limit=).
GET  /runs/ID/report.md | results.json | audit.json | bundle.tar.gz
DELETE /runs/ID   remove a published run (Authorization: Bearer ADMIN_TOKEN).
Put it behind HTTPS (same reverse proxy as the witness control API).
"""
import argparse
import hashlib
import hmac
import html
import io
import json
import queue
import shutil
import tarfile
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BUNDLE = 200_000_000
SERVED = {'report.md': 'text/markdown; charset=utf-8', 'results.json': 'application/json',
          'audit.json': 'application/json', 'bundle.tar.gz': 'application/gzip'}


def bundle(run, out=None):
    """Deterministic tar.gz of an evidence directory."""
    root = Path(run)
    if not (root / 'manifest.json').is_file():
        raise ValueError('not a finished evidence directory (no manifest.json)')
    out = Path(out or f'{root}.tar.gz')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', compresslevel=6) as tar:
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            info = tarfile.TarInfo(str(path.relative_to(root)))
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    out.write_bytes(buffer.getvalue())
    return out


def safe_extract(data, destination):
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError(f'unsafe bundle entry {member.name!r}')
            total += member.size
            if total > 4 * MAX_BUNDLE:
                raise ValueError('bundle expands too far')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())


def write_atomic(path, text):
    """Write via a temporary file and rename: concurrent readers never see a partial file."""
    temporary = path.with_name(f'.{path.name}.{threading.get_ident()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


class Hub:
    def __init__(self, directory, witness_keys, workers=1, max_pending=50, admin_token=None, base_path='',
                 public_url=None, witness_spec=None, submissions_per_hour=30, terms=None, downloads=None):
        self.downloads = Path(downloads) if downloads else None
        from .site import HUB_TERMS
        self.terms = terms or HUB_TERMS
        self.base = base_path.rstrip('/')
        self.public_url = public_url
        self.witness_spec = witness_spec
        self.submit_limit = submissions_per_hour
        self.submits = {}
        self.dir = Path(directory)
        for sub in ('runs', 'queue', 'jobs'):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.keys = witness_keys
        self.lock = threading.RLock()
        self.admin_token = admin_token
        self.max_pending = max_pending
        self.pending = queue.Queue()
        for path in sorted((self.dir / 'queue').glob('*.tar.gz')):  # survive restarts
            self.pending.put(path.name.split('.')[0])
        self.backfill()
        for _ in range(workers):
            threading.Thread(target=self.work, daemon=True).start()

    def backfill(self):
        """Compute findings for runs accepted before findings existed."""
        from .findings import VERSION, extract
        with self.lock:
            index, changed = self.index(), False
            for entry in index:
                if entry.get('findings_version') == VERSION:
                    continue
                entry['findings_version'] = VERSION
                changed = True
                folder = self.dir / 'runs' / entry['id']
                try:
                    results = json.loads((folder / 'results.json').read_text())['results']
                    with tarfile.open(folder / 'bundle.tar.gz') as tar:
                        meta = json.load(tar.extractfile('run.json'))
                    entry['route_specs'] = meta['args'].get('routes', [])
                    entry['findings'] = extract(entry['command'], results, entry['route_specs'])
                    changed = True
                except (OSError, KeyError, ValueError, tarfile.TarError):
                    entry.setdefault('findings', [])
            if changed:
                write_atomic(self.dir / 'index.json', json.dumps(index, indent=1))

    def allow_submission(self, ip):
        now = time.time()
        recent = [t for t in self.submits.get(ip, []) if now - t < 3600]
        if len(recent) >= self.submit_limit:
            self.submits[ip] = recent
            return False
        self.submits[ip] = recent + [now]
        return True

    def job(self, job_id):
        path = self.dir / 'jobs' / f'{job_id}.json'
        return json.loads(path.read_text()) if path.exists() else None

    def set_job(self, job_id, **fields):
        write_atomic(self.dir / 'jobs' / f'{job_id}.json', json.dumps(dict(id=job_id, **fields)))

    def enqueue(self, data):
        if len(data) > MAX_BUNDLE:
            raise ValueError('bundle too large')
        job_id = hashlib.sha256(data).hexdigest()[:20]
        existing = self.job(job_id)
        if existing and existing['status'] in ('queued', 'running', 'accepted'):
            return existing, False
        if self.pending.qsize() >= self.max_pending:
            raise OverflowError('queue full, try later')
        (self.dir / 'queue' / f'{job_id}.tar.gz').write_bytes(data)
        self.set_job(job_id, status='queued')
        self.pending.put(job_id)
        return self.job(job_id), True

    def work(self):
        while True:
            job_id = self.pending.get()
            path = self.dir / 'queue' / f'{job_id}.tar.gz'
            if not path.exists():
                continue
            self.set_job(job_id, status='running')
            try:
                entry, _ = self.submit(path.read_bytes())
                self.set_job(job_id, status='accepted', entry=entry)
            except Exception as error:  # audit failures, malformed bundles
                self.set_job(job_id, status='rejected', error=str(error)[:1000])
            finally:
                path.unlink(missing_ok=True)

    def append_index(self, entry):
        with self.lock:  # whole read-modify-write under one (re-entrant) lock
            index = self.index()
            index.append(entry)
            write_atomic(self.dir / 'index.json', json.dumps(index, indent=1))

    def delete(self, run_id):
        with self.lock:  # whole read-modify-write under one (re-entrant) lock
            index = [e for e in self.index() if e['id'] != run_id]
            write_atomic(self.dir / 'index.json', json.dumps(index, indent=1))
        shutil.rmtree(self.dir / 'runs' / run_id, ignore_errors=True)
        (self.dir / 'jobs' / f'{run_id}.json').unlink(missing_ok=True)

    def index(self):
        with self.lock:
            path = self.dir / 'index.json'
            return json.loads(path.read_text()) if path.exists() else []

    def submit(self, data):
        from .replay import AuditFailure, audit
        if len(data) > MAX_BUNDLE:
            raise ValueError('bundle too large')
        run_id = hashlib.sha256(data).hexdigest()[:20]  # same id as the job
        if (self.dir / 'runs' / run_id).exists():
            return next(r for r in self.index() if r['id'] == run_id), False
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            safe_extract(data, run)
            failures = []
            for key in self.keys:
                try:
                    report = audit(run, key, require_witness=True)
                except AuditFailure as error:
                    failures.append(str(error))
                    continue
                if report['attested'] and report['results_reproduced']:
                    break
                failures.append('results not reproduced' if not report['results_reproduced'] else 'not attested')
            else:
                raise AuditFailure('; '.join(failures) or 'no trusted witness key')
            receipts = [json.loads(line).get('witness_receipt') or {} for line in
                        (run / 'requests.jsonl').read_text().splitlines() if line.strip()]
            meta = json.loads((run / 'run.json').read_text())
            from .findings import VERSION, extract
            results_doc = json.loads((run / 'results.json').read_text())
            entry = dict(route_specs=meta['args'].get('routes', []), findings_version=VERSION,
                         findings=extract(report['command'], results_doc.get('results'), meta['args'].get('routes', [])),id=run_id, command=report['command'], routes=report['routes'],
                         witness_key_id=receipts[0].get('witness_key_id') if receipts else None,
                         submitters=sorted({r.get('submitter_token_id') for r in receipts if r.get('submitter_token_id')}),
                         records=report['records'], requests_disclosed=report['requests_disclosed'],
                         run_started=meta.get('started_at'), accepted_at=datetime.now(timezone.utc).isoformat())
            target = self.dir / 'runs' / run_id
            target.mkdir()
            for name in ('report.md', 'results.json'):
                if (run / name).exists():
                    shutil.copy(run / name, target / name)
            (target / 'audit.json').write_text(json.dumps(report, indent=1))
            (target / 'bundle.tar.gz').write_bytes(data)
        self.append_index(entry)
        return entry, True


def handler_for(hub):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            if kind.startswith('application/json') or kind.startswith('application/javascript'):
                self.send_header('Access-Control-Allow-Origin', '*')  # findings.json / embed.js on other pages
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from . import site
            url = urlsplit(self.path)
            parts = [p for p in url.path.split('/') if p]
            if not parts:
                return self.send(200, site.index_page(hub, hub.base), 'text/html; charset=utf-8')
            if parts == ['findings.json']:
                return self.send(200, site.findings_json(hub, parse_qs(url.query)))
            if len(parts) == 2 and parts[0] == 'download' and hub.downloads:
                name = parts[1]
                path = hub.downloads / name
                if '/' not in name and not name.startswith('.') and name.endswith(('.whl', '.tar.gz')) and path.is_file():
                    return self.send(200, path.read_bytes(), 'application/octet-stream')
                return self.send(404, b'{"error":"not found"}')
            if parts == ['terms']:
                return self.send(200, site.terms_page(hub, hub.base), 'text/html; charset=utf-8')
            if parts == ['embed.js']:
                return self.send(200, site.EMBED_JS, 'application/javascript; charset=utf-8')
            if len(parts) == 2 and parts[0] == 'runs' and parts[1].isalnum():
                body = site.run_page(hub, parts[1], hub.base)
                return self.send(200 if body else 404, body or b'not found', 'text/html; charset=utf-8')
            if parts == ['runs']:
                query = parse_qs(url.query)
                offset = int(query.get('offset', ['0'])[0])
                limit = min(500, int(query.get('limit', ['100'])[0]))
                index = hub.index()
                return self.send(200, json.dumps(dict(total=len(index), runs=index[offset:offset + limit])).encode())
            if len(parts) == 2 and parts[0] == 'jobs' and parts[1].isalnum():
                job = hub.job(parts[1])
                return self.send(200 if job else 404, json.dumps(job or dict(error='unknown job')).encode())
            if len(parts) == 3 and parts[0] == 'runs' and parts[2] in SERVED and parts[1].isalnum():
                path = hub.dir / 'runs' / parts[1] / parts[2]
                if path.is_file():
                    return self.send(200, path.read_bytes(), SERVED[parts[2]])
            self.send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path.rstrip('/') != '/runs':
                return self.send(404, b'{"error":"not found"}')
            ip = self.headers.get('X-Forwarded-For', self.client_address[0]).split(',')[0].strip()
            if not hub.allow_submission(ip):
                return self.send(429, b'{"error":"too many submissions; try later"}')
            length = int(self.headers.get('Content-Length', '0'))
            if length > MAX_BUNDLE:
                return self.send(413, b'{"error":"bundle too large"}')
            data = self.rfile.read(length)
            try:
                job, new = hub.enqueue(data)
            except OverflowError as error:
                return self.send(503, json.dumps(dict(error=str(error))).encode())
            except ValueError as error:
                return self.send(422, json.dumps(dict(accepted=False, error=str(error))).encode())
            self.send(202, json.dumps(dict(job=job['id'], status=job['status'], new=new,
                                           poll=f"/jobs/{job['id']}")).encode())

        def do_DELETE(self):
            parts = [p for p in self.path.split('/') if p]
            supplied = self.headers.get('Authorization', '').removeprefix('Bearer ').strip()
            if not hub.admin_token or not hmac.compare_digest(supplied, hub.admin_token):
                return self.send(401, b'{"error":"admin token required"}')
            if len(parts) != 2 or parts[0] != 'runs' or not parts[1].isalnum():
                return self.send(404, b'{"error":"not found"}')
            hub.delete(parts[1])
            self.send(200, json.dumps(dict(deleted=parts[1])).encode())

        def log_message(self, *args):
            pass
    return Handler


def serve(hub, address, ready=None):
    server = ThreadingHTTPServer(address, handler_for(hub))
    if ready:
        ready(server.server_address)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', required=True)
    parser.add_argument('--witness-key', action='append', required=True, help='trusted witness public key (base64)')
    parser.add_argument('--listen', default='127.0.0.1:8090')
    parser.add_argument('--workers', type=int, default=1, help='parallel audit workers')
    parser.add_argument('--max-pending', type=int, default=50)
    parser.add_argument('--admin-token-file', help='file holding the token that allows DELETE /runs/ID')
    parser.add_argument('--base-path', default='', help='URL prefix the hub is served under, e.g. /hub')
    parser.add_argument('--public-url', help='public URL of the hub, shown in instructions')
    parser.add_argument('--witness-spec', help='witness spec shown to contributors (TUNNEL,CONTROL_URL,PUBKEY)')
    parser.add_argument('--submissions-per-hour', type=int, default=30)
    parser.add_argument('--downloads', help='directory of client packages (.whl, .tar.gz) served at /download/')
    args = parser.parse_args()
    host, _, port = args.listen.rpartition(':')
    admin = Path(args.admin_token_file).read_text().strip() if args.admin_token_file else None
    serve(Hub(args.dir, args.witness_key, args.workers, args.max_pending, admin, args.base_path, args.public_url,
              args.witness_spec, args.submissions_per_hour, downloads=args.downloads), (host, int(port)),
          lambda a: print(f'kremlin-hub on {a[0]}:{a[1]}', flush=True))


if __name__ == '__main__':
    main()
