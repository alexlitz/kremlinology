"""Audit an evidence directory: verify receipts, check requests, recompute the results.

The investigation is re-run against a ReplayClient that serves the recorded responses in order.
For every call it checks:
  - the request the investigation code would send equals the recorded request (same seed and
    run salt, so even fresh documents are regenerated exactly);
  - witnessed records: the receipt signature verifies with the witness public key, the receipt's
    response equals the captured response, and a disclosed request equals the recorded request
    (so the request's properties, e.g. provider pin, max_tokens, logprobs, are attested too);
  - timing comes from the witness's record timestamps when witnessed, never from the client.
Finally the recomputed results are compared with the submitted results.json. Unwitnessed runs can
still be replayed (reproducibility), but nothing about them is attested.
"""
import base64
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from . import evidence, routes as routes_mod
from .client import Client, Result, _token_bearing, extract_generation, parse, witnessed_latency


class AuditFailure(Exception):
    pass


# Fixed key lengths: a declared length equal to the real one leaves no slack in the hidden request prefix.
KNOWN_KEY_LENGTHS = {'openrouter.ai': {73}}
# What the response must be for the disclosed path (the request line is length-bound, not content-bound).
RESPONSE_KINDS = [('/chat/completions', ('chat.completion', 'chat.completion.chunk')), ('/embeddings', ('list',)),
                  ('/completions', ('text_completion',)), ('/messages', ('message',))]


def check_response_kind(path, body, capture):
    """A same-length path to a different endpoint would return a different kind of object."""
    base = path.split('?')[0]
    for suffix, kinds in RESPONSE_KINDS:
        if base.endswith(suffix):
            text = body.decode('utf-8', 'replace').lstrip()
            if text.startswith('data:'):  # stream: first event
                text = text[5:].split('\n', 1)[0]
            try:
                obj = json.loads(text)
            except ValueError:
                return
            if not isinstance(obj, dict) or 'error' in obj:
                return
            kind = obj.get('object') or obj.get('type')
            if kind is not None and kind not in kinds:
                raise AuditFailure(f'capture {capture}: response is a {kind!r}, not what {suffix} returns')
            return


class ReplayClient:
    _anthropic = staticmethod(Client._anthropic)

    concurrency = 1

    def __init__(self, run, public_key=None):
        self.root = Path(run)
        records = [json.loads(l) for l in (self.root / 'requests.jsonl').read_text().splitlines() if l.strip()]
        self.by_key = defaultdict(list)
        for record in records:
            if not record.get('key'):
                raise AuditFailure(f"record {record.get('capture')} has no call key (made by an older kremlin)")
            self.by_key[record['key']].append(record)
        for chain in self.by_key.values():
            chain.sort(key=lambda r: r.get('attempt', 0))
        self._seq = itertools.count()
        self.public_key = public_key
        self.findings = []
        # Attempts that got no HTTP response at all (e.g. the relay refused the tunnel: rate limit) never reached a
        # provider and feed no result; they are allowed to be unwitnessed.
        self.counts = dict(records=len(records), witnessed=0, receipts_valid=0, requests_disclosed=0,
                           unanswered=sum(1 for r in records if response_status(r) is None))
        self.spent, self.requests, self.unknown_cost, self.max_usd = 0.0, 0, 0, float('inf')
        self.evidence = None
        self.witness = None

    def finding(self, text_):
        self.findings.append(text_)

    def capture(self, record):
        data = json.loads((self.root / 'captures' / f"{record['capture']}.json").read_text())
        return base64.b64decode(data['body_base64'])

    def response_of(self, record):
        """Response body (bytes), status, witness SSE times; verified against the receipt if present."""
        from .witness import client as wclient, receipts
        body = self.capture(record)
        receipt = record.get('witness_receipt')
        if not receipt:
            return body, record.get('status') or (record.get('result') or {}).get('status'), None
        self.counts['witnessed'] += 1
        if not self.public_key:
            self.finding(f"capture {record['capture']}: witnessed but no --pubkey given; receipt not checked")
            return body, record.get('status') or (record.get('result') or {}).get('status'), None
        try:
            raw = receipts.verify(receipt, self.public_key)
        except ValueError as error:
            raise AuditFailure(f"capture {record['capture']}: receipt invalid: {error}") from None
        self.counts['receipts_valid'] += 1
        status, headers, witnessed_body = wclient.parse_http_response(raw)
        if witnessed_body != body:
            raise AuditFailure(f"capture {record['capture']}: captured response differs from the witnessed one")
        url = urlsplit(record['url'])
        if receipt['host'] != url.hostname:
            raise AuditFailure(f"capture {record['capture']}: receipt host {receipt['host']} != {url.hostname}")
        disclosed = receipt.get('request')
        if disclosed:
            self.counts['requests_disclosed'] += 1
            path = url.path + (f'?{url.query}' if url.query else '')
            if disclosed['path'] != path:
                raise AuditFailure(f"capture {record['capture']}: witnessed path {disclosed['path']} != {path}")
            allowed = KNOWN_KEY_LENGTHS.get(url.hostname)
            if allowed and disclosed['key_length'] not in allowed:
                raise AuditFailure(f"capture {record['capture']}: declared key length {disclosed['key_length']} is "
                                   f'not a valid {url.hostname} key length (room for a hidden request)')
            check_response_kind(path, witnessed_body, record['capture'])
            if disclosed.get('method') and disclosed['method'] != record.get('method', disclosed['method']):
                raise AuditFailure(f"capture {record['capture']}: witnessed method {disclosed['method']} != "
                                   f"{record['method']}")
            if record.get('request_base64') is not None:  # session records keep the exact bytes sent
                if base64.b64decode(disclosed['body_base64']) != base64.b64decode(record['request_base64']):
                    raise AuditFailure(f"capture {record['capture']}: witnessed request body differs from the record")
            elif record.get('request') is not None:
                sent = base64.b64decode(disclosed['body_base64'])
                if sent != json.dumps(record['request']).encode():
                    raise AuditFailure(f"capture {record['capture']}: witnessed request body differs from the record")
        times = None
        if 'text/event-stream' in headers.get('content-type', ''):
            times = [t / 1000 for t, data in wclient.sse_event_times(raw, receipt['response_records'])
                     if _token_bearing(data)]
        return body, status, times

    def key(self, prefix='seq'):
        return f'{prefix}:{next(self._seq)}'

    def parallel(self, tasks, workers=None):
        return [task() for task in tasks]  # replay is sequential; keys make order irrelevant

    def _take(self, kind, route_label, call_key):
        """The recorded attempts for one logical call, matched by its key."""
        chain = self.by_key.pop(call_key, None)
        if not chain:
            raise AuditFailure(f'the investigation makes call {call_key!r} ({kind} to {route_label}), '
                               'which is not in the evidence')
        record = chain[-1]
        if record.get('kind', 'chat') != kind or record['route'] != route_label:
            raise AuditFailure(f"call {call_key!r} is recorded as a {record.get('kind')} to {record['route']}, "
                               f'but the investigation makes a {kind} call to {route_label}')
        return chain

    def chat(self, route, messages, *, max_tokens=1, temperature=0.0, top_logprobs=None, stream=False, extra=None,
             cap_field='max_tokens', require_parameters=False, tag=None, key=None):
        _, _, body, _ = Client._build(self, route, messages, max_tokens=max_tokens, temperature=temperature,
                                      top_logprobs=top_logprobs, stream=stream, extra=extra, cap_field=cap_field,
                                      require_parameters=require_parameters, key='-')
        result = None
        for record in self._take('chat', route.label, key or self.key()):
            self.requests += 1
            if record['request'] != json.loads(json.dumps(body)):
                raise AuditFailure(f"record {record['capture']}: the investigation would send a different request "
                                   'than the one recorded')
            raw, status, times = self.response_of(record)
            result = parse(route, status, raw, stream) if status is not None else Result(route.label, False, None)
            receipt = record.get('witness_receipt')
            result.elapsed_s = (witnessed_latency(receipt) if receipt and self.public_key
                                else (record.get('result') or {}).get('elapsed_s'))
            if stream:
                result.token_times_s = times if times is not None else (record.get('result') or {}).get(
                    'token_times_s', [])
            Client._account(self, route, result, body)
        return result

    def generation(self, route, generation_id, key=None, **_):
        if not route.is_openrouter or not generation_id:
            return None
        out = None
        for record in self._take('generation', route.label, key or f'gen:{generation_id}'):
            raw, status, _ = self.response_of(record)
            if status == 200:
                out = extract_generation(raw)
        return out

    def request(self, route, method, path, key=None, **_):
        record = self._take('request', route.label, key or self.key('request'))[-1]
        raw, status, _ = self.response_of(record)
        return status, raw, record.get('witness_receipt')

    def summary(self):
        return dict(spent_usd=round(self.spent, 6), requests=self.requests, unknown_cost_responses=self.unknown_cost)


def _audit_session(root, report, public_key, require_witness):
    """Sessions (kremlin proxy / http_client) have no investigation to re-run: every exchange's receipt is checked
    and the per-call summary is recomputed from the verified records."""
    from . import session
    client = ReplayClient(root, public_key)
    rows = []
    for chain in client.by_key.values():
        for record in chain:
            if record.get('kind') != 'http':
                raise AuditFailure(f"capture {record['capture']}: a session holds only http records")
            body, _, _ = client.response_of(record) if record.get('status') is not None else (b'', None, None)
            rows.append(session.describe(record, body))
    rows.sort(key=lambda r: r['n'])
    results = session.summarize(rows)
    submitted = json.loads((root / 'results.json').read_text())['results']
    differences = _diff(_normal(results), _normal(submitted))
    answered = client.counts['records'] - client.counts['unanswered']
    if require_witness and client.counts['witnessed'] < answered:
        raise AuditFailure(f"{answered - client.counts['witnessed']} answered records are not witnessed")
    report.update(command='session', routes=results['hosts'], **client.counts, results_reproduced=not differences,
                  differences=differences[:20], findings=client.findings,
                  attested=bool(public_key) and client.counts['witnessed'] == answered
                  and client.counts['receipts_valid'] == answered)
    return report


def response_status(record):
    if record.get('kind') in ('generation', 'request', 'http'):
        return record.get('status')
    return (record.get('result') or {}).get('status')


def _normal(obj):
    return json.loads(json.dumps(obj, default=str))


def _diff(a, b, path='results'):
    if type(a) != type(b):
        return [f'{path}: {a!r} != {b!r}']
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f'{path}.{k}') if k in a and k in b else [f'{path}.{k}: missing on one side']
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f'{path}[{i}]')]
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def audit(run, public_key=None, require_witness=False):
    from .cli import COMMANDS, options_for
    root = Path(run)
    if not (root / 'manifest.json').is_file():
        raise AuditFailure('no manifest.json: the run did not finish (or this is not an evidence directory)')
    report = dict(run=str(root), manifest=evidence.verify(root))
    if not report['manifest']['ok']:
        raise AuditFailure(f"manifest mismatch: {report['manifest']['mismatched'][:5]}")
    meta = json.loads((root / 'run.json').read_text())
    args = meta['args']
    command = meta['command']
    if command == 'session':
        return _audit_session(root, report, public_key, require_witness)
    if command not in COMMANDS:
        raise AuditFailure(f'cannot replay command {command!r}')
    _, run_fn, _, _ = COMMANDS[command]
    route_list = [routes_mod.parse_route(r) for r in args['routes']]
    options = options_for(args)
    client = ReplayClient(root, public_key)
    with mock.patch.object(time, 'sleep', lambda s: None):
        results, _ = run_fn(client, route_list, **options)
    if client.by_key:
        raise AuditFailure(f'{len(client.by_key)} recorded calls were not made by the investigation, e.g. '
                           f'{next(iter(client.by_key))!r}')
    submitted = json.loads((root / 'results.json').read_text())['results']
    differences = _diff(_normal(results), _normal(submitted))
    answered = client.counts['records'] - client.counts['unanswered']
    if require_witness and client.counts['witnessed'] < answered:
        raise AuditFailure(f"{answered - client.counts['witnessed']} answered records are not witnessed")
    report.update(command=command, routes=[r.label for r in route_list], **client.counts,
                  results_reproduced=not differences, differences=differences[:20], findings=client.findings,
                  attested=bool(public_key) and client.counts['witnessed'] == answered
                  and client.counts['receipts_valid'] == answered)
    return report
