"""Every investigation end to end against the fake provider, directly and through the witness relay."""
import asyncio
import json
import threading

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fakeprovider import Provider
from test_witness import make_pki

from kremlin import cli, evidence
from kremlin.client import Client
from kremlin.investigations import billing, cache, fingerprint, timing, tv
from kremlin.routes import parse_route
from kremlin.witness.client import WitnessClient
from kremlin.witness.relay import Relay

KEY = 'fake-key-must-stay-private'


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('pki')
    ca = make_pki(tmp)
    provider = Provider(tmp / 'leaf.pem', tmp / 'leaf.key')
    relay = Relay(Ed25519PrivateKey.generate(), ['localhost'], trust_file=str(ca), allow_private=True, allow_ports=range(1, 65536))
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(relay.serve(
        ('127.0.0.1', 0), ('127.0.0.1', 0), lambda t, c: (addr.update(t=t, c=c), ready.set()))), daemon=True).start()
    assert ready.wait(10)
    yield dict(ca=str(ca), provider=provider, witness=f"127.0.0.1:{addr['t'][1]},http://127.0.0.1:{addr['c'][1]}")
    provider.close()


def route(world, model, label=None):
    return parse_route(f"openai:https://localhost:{world['provider'].port}/v1|{model};key=FAKE_KEY;label={label or model}")


def client(world, tmp_path, witnessed):
    witness = None
    if witnessed:
        tunnel, control = world['witness'].split(',')
        witness = WitnessClient(tunnel, control, trust_file=world['ca'])
    return Client(evidence.Evidence(tmp_path / 'run', 'test', {}), max_usd=1.0, witness=witness,
                  trust_file=world['ca'], backoff_s=0.01)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv('FAKE_KEY', KEY)


@pytest.mark.parametrize('witnessed', [False, True])
def test_fingerprint_finds_reseller(world, tmp_path, witnessed):
    c = client(world, tmp_path, witnessed)
    routes = [route(world, 'shared-a'), route(world, 'shared-b'), route(world, 'other')]
    results, md = fingerprint.run(c, routes, prompts=25, repeats=2, top_logprobs=5)
    assert results['pairs']['shared-a ~ shared-b']['verdict'].startswith('same numerics')
    assert results['pairs']['shared-a ~ other']['verdict'].startswith('different numerics')
    assert '# Same-backend fingerprint' in md


@pytest.mark.parametrize('witnessed', [False, True])
def test_cache_share_detects_shared_cache(world, tmp_path, witnessed):
    c = client(world, tmp_path, witnessed)
    results, _ = cache.run_share(c, [route(world, 'shared-a'), route(world, 'shared-b')], trials=2, chars=3000, gap_s=0)
    assert results['verdict'].startswith('shared cache')
    results, _ = cache.run_share(c, [route(world, 'shared-a'), route(world, 'other')], trials=2, chars=3000, gap_s=0)
    assert results['verdict'].startswith('separate caches')


def test_cache_blocks_and_prefix(world, tmp_path):
    c = client(world, tmp_path, True)
    results, _ = cache.run_blocks(c, [route(world, 'other')], lengths=(1000, 2000, 4000), gap_s=0)
    assert results['block_tokens'] and results['block_tokens'] % 16 == 0
    results, _ = cache.run_prefix(c, [route(world, 'other')], trials=1, chars=3000, gap_s=0)
    assert results['tests']['diverge_half'][0]['share_of_first'] > 0.2  # any-prefix cache


@pytest.mark.parametrize('witnessed', [False, True])
def test_billing_flags_ignored_cap(world, tmp_path, witnessed):
    c = client(world, tmp_path, witnessed)
    results, _ = billing.run(c, [route(world, 'other'), route(world, 'ignorer')], cap=16, repeats=1)
    assert results['verdicts']['other'].startswith('honoured')
    assert not results['verdicts']['ignorer'].startswith('honoured')


@pytest.mark.parametrize('witnessed', [False, True])
def test_timing_sees_speculation(world, tmp_path, witnessed):
    c = client(world, tmp_path, witnessed)
    results, _ = timing.run(c, [route(world, 'other')], repeats=3, max_tokens=60)
    row = results['routes']['other']
    assert row['count_over_story'][0] > 1.4 and 0.6 < row['story2_over_story'][0] < 1.6


def test_tv_reference_self_and_difference(world, tmp_path):
    c = client(world, tmp_path, True)
    results, _ = tv.run(c, [route(world, 'shared-a'), route(world, 'shared-b'), route(world, 'other')], prompts=20,
                        top_logprobs=5)
    assert results['routes']['shared-b']['mean_lower'] < 0.02  # upper stays wide: top-5 leaves mass unreported
    assert results['routes']['other']['mean_lower'] > 0.2


def test_generic_endpoints_through_witness(world, tmp_path):
    c = client(world, tmp_path, True)
    r = route(world, 'other')
    status, body, receipt = c.request(r, 'POST', 'audio/transcriptions', files=dict(file=('a.wav', b'\x00' * 50_000)),
                                      data=dict(model='whisper'))
    assert status == 200 and json.loads(body)['text'].startswith('transcribed') and receipt
    status, body, receipt = c.request(r, 'POST', 'images/generations', json_body=dict(prompt='a cat'))
    assert status == 200 and len(json.loads(body)['data'][0]['b64_json']) >= 400_000 and receipt


def test_key_never_in_receipts_or_evidence(world, tmp_path):
    c = client(world, tmp_path, True)
    fingerprint.run(c, [route(world, 'shared-a'), route(world, 'other')], prompts=3, repeats=2)
    c.evidence.close()
    blob = b''.join(p.read_bytes() for p in (tmp_path / 'run').rglob('*') if p.is_file())
    assert KEY.encode() not in blob
    assert f'Bearer {KEY}' in world['provider'].seen_auth  # the provider did receive it
    assert evidence.verify(tmp_path / 'run')['ok']


def test_cli_plan_and_run(world, tmp_path, capsys):
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    cli.main(['billing', spec, '--plan'])
    assert '"requests": 6' in capsys.readouterr().out
    cli.main(['billing', spec, '--cap', '8', '--repeats', '1', '--out', str(tmp_path / 'cli'), '--trust', world['ca'],
              '--witness', world['witness']])
    out = capsys.readouterr().out
    assert 'honoured' in out and (tmp_path / 'cli' / 'report.md').exists()
    assert evidence.verify(tmp_path / 'cli')['ok']


def test_rerun_with_same_seed_uses_fresh_documents(world, tmp_path):
    c = client(world, tmp_path, False)
    first, _ = cache.run_share(c, [route(world, 'shared-a'), route(world, 'other')], trials=2, chars=3000, gap_s=0)
    again, _ = cache.run_share(c, [route(world, 'shared-a'), route(world, 'other')], trials=2, chars=3000, gap_s=0)
    assert first['verdict'].startswith('separate') and again['verdict'].startswith('separate')


def run_cli(world, tmp_path, name, *argv):
    out = tmp_path / name
    cli.main([*argv, '--out', str(out), '--trust', world['ca'], '--witness', world['witness']])
    return out


def pubkey(world):
    import httpx
    return httpx.get(world['witness'].split(',')[1] + '/pubkey').json()['public_key']


@pytest.mark.parametrize('command', [
    ['fingerprint', '{shared-a}', '{other}', '--prompts', '6'],
    ['cache-share', '{shared-a}', '{shared-b}', '--trials', '1', '--chars', '2000'],
    ['billing', '{ignorer}', '--cap', '8', '--repeats', '1'],
    ['timing', '{other}', '--repeats', '3', '--max-tokens', '30'],
    ['params', '{other}', '{liar}'],
    ['sampler', '{other}', '--samples', '30'],
])
def test_audit_reproduces_witnessed_runs(world, tmp_path, command, capsys):
    from kremlin.replay import audit
    specs = {m: f"openai:https://localhost:{world['provider'].port}/v1|{m};key=FAKE_KEY;label={m}"
             for m in ('shared-a', 'shared-b', 'other', 'ignorer', 'liar')}
    argv = [a.format(**specs) if a.startswith('{') else a for a in command]
    run = run_cli(world, tmp_path, 'run', *argv)
    capsys.readouterr()
    report = audit(run, pubkey(world), require_witness=True)
    assert report['results_reproduced'] and report['attested'], report['differences']
    assert report['requests_disclosed'] == report['records']


def test_audit_catches_tampering(world, tmp_path, capsys):
    from kremlin.replay import AuditFailure, audit
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    run = run_cli(world, tmp_path, 'run', 'billing', spec, '--cap', '8', '--repeats', '1')
    capsys.readouterr()
    results = json.loads((run / 'results.json').read_text())
    results['results']['verdicts']['other'] = 'cap ignored'          # a forged conclusion
    (run / 'results.json').write_text(json.dumps(results))
    with pytest.raises(AuditFailure, match='manifest'):              # manifest catches the edit
        audit(run, pubkey(world))
    import hashlib
    manifest = json.loads((run / 'manifest.json').read_text())       # re-sealing does not help:
    manifest['files']['results.json'] = hashlib.sha256((run / 'results.json').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    report = audit(run, pubkey(world))
    assert not report['results_reproduced'] and any('verdicts.other' in d for d in report['differences'])


def test_params_checks(world, tmp_path):
    from kremlin.investigations import params
    c = client(world, tmp_path, True)
    results, md = params.run(c, [route(world, 'other'), route(world, 'liar'), route(world, 'nolog'), route(world, 'gmi')])
    checks = results['checks']
    assert all(checks['other'][k][0] == 'pass' for k in ('logprobs', 'n', 'seed', 'temperature0', 'stop', 'max_tokens'))
    assert checks['liar']['stop'][0] == 'fail'
    assert checks['nolog']['logprobs'][0] == 'fail'
    assert checks['gmi']['logprobs'][0] == 'fail' and 'above reported top-1' in checks['gmi']['logprobs'][1]


def test_sampler_catches_ignored_parameters(world, tmp_path):
    from kremlin.investigations import sampler
    c = client(world, tmp_path, False)
    results, md = sampler.run(c, [route(world, 'other'), route(world, 'liar')], samples=200)
    good, bad = results['routes']['other'], results['routes']['liar']
    assert good['top_k_1']['violations'] == 0 and good['top_k_2']['violations'] == 0
    assert good['t1']['gof']['p_value'] > 0.001 and 0.35 < good['t05']['fitted_temperature'] < 0.7
    assert bad['top_k_1']['violations'] > 0 and bad['t05']['fitted_temperature'] > 0.8
    # top_p mid-gap between p1 and p1 + p2: an honest nucleus holds only the top two
    assert good['top_p']['violations'] == 0 and bad['top_p']['violations'] > 0


def test_hub_accepts_attested_and_rejects_forged(world, tmp_path, capsys):
    import httpx
    from kremlin.hub import Hub, bundle, serve
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    run = run_cli(world, tmp_path, 'good', 'billing', spec, '--cap', '8', '--repeats', '1')
    plain = tmp_path / 'plain'
    cli.main(['billing', spec, '--cap', '8', '--repeats', '1', '--out', str(plain), '--trust', world['ca']])
    capsys.readouterr()
    hub = Hub(tmp_path / 'hub', [pubkey(world)], admin_token='admintoken')
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: serve(hub, ('127.0.0.1', 0), lambda a: (addr.update(a=a), ready.set())),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{addr['a'][1]}"

    def submit(data):
        import time as t
        reply = httpx.post(base + '/runs', content=data)
        assert reply.status_code == 202, reply.text
        for _ in range(300):
            job = httpx.get(base + reply.json()['poll']).json()
            if job['status'] in ('accepted', 'rejected'):
                return job
            t.sleep(0.1)
        raise AssertionError('job did not finish')
    job = submit(bundle(run).read_bytes())
    assert job['status'] == 'accepted'
    run_id = job['entry']['id']
    assert submit(bundle(run).read_bytes())['status'] == 'accepted'  # idempotent
    assert 'Output cap compliance' in httpx.get(f'{base}/runs/{run_id}/report.md').text
    page = httpx.get(base + '/')
    assert page.status_code == 200 and run_id in page.text, (page.status_code, page.text[-500:])
    listing = httpx.get(base + '/runs?limit=1').json()
    assert listing['total'] == 1, listing
    rejected = submit(bundle(plain).read_bytes())  # not witnessed
    assert rejected['status'] == 'rejected' and 'not witnessed' in rejected['error']
    assert httpx.delete(f'{base}/runs/{run_id}').status_code == 401
    assert httpx.delete(f'{base}/runs/{run_id}', headers={'Authorization': 'Bearer admintoken'}).status_code == 200
    assert httpx.get(base + '/runs').json()['total'] == 0
    import io, tarfile
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode='w:gz') as tar:
        info = tarfile.TarInfo('../escape.txt')
        info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    assert submit(evil.getvalue())['status'] == 'rejected'
    assert not (tmp_path / 'hub' / 'escape.txt').exists() and not (tmp_path / 'escape.txt').exists()


def test_audit_response_kind_and_key_length_checks():
    from kremlin.replay import AuditFailure, check_response_kind
    check_response_kind('/v1/chat/completions', b'{"object":"chat.completion","choices":[]}', 'c')
    check_response_kind('/v1/chat/completions', b'data: {"object":"chat.completion.chunk"}\n\n', 'c')
    check_response_kind('/v1/chat/completions', b'{"error":{"message":"x"}}', 'c')  # errors carry no kind
    with pytest.raises(AuditFailure, match='not what /embeddings returns'):
        check_response_kind('/v1/embeddings', b'{"object":"chat.completion"}', 'c')


@pytest.mark.parametrize('witnessed', [False, True])
def test_resume_after_budget_stop(world, tmp_path, capsys, witnessed):
    from kremlin.replay import audit
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    out = tmp_path / 'run'
    extra = ['--witness', world['witness']] if witnessed else []
    cli.main(['fingerprint', spec, f"openai:https://localhost:{world['provider'].port}/v1|shared-a;key=FAKE_KEY",
              '--prompts', '10', '--max-usd', '0.0002', '--concurrency', '1', '--out', str(out),
              '--trust', world['ca'], *extra])
    first = json.loads((out / 'results.json').read_text())
    assert first['status'].startswith('stopped') and first['results'] is None
    sent_before = first['usage']['requests']
    cli.main(['resume', str(out), '--max-usd', '1'])
    done = json.loads((out / 'results.json').read_text())
    capsys.readouterr()
    assert done['status'] == 'completed' and done['usage']['resumed_calls'] >= sent_before - 1
    assert done['usage']['requests'] + done['usage']['resumed_calls'] == 40
    report = audit(out, pubkey(world) if witnessed else None, require_witness=witnessed)
    assert report['results_reproduced']


def test_later_positions(world, tmp_path):
    from kremlin.investigations import later
    c = client(world, tmp_path, True)
    routes = [route(world, 'shared-a'), route(world, 'shared-b'), route(world, 'other'), route(world, 'noprefill')]
    results, md = later.run(c, routes, prompts=12, positions=(1, 4), top_logprobs=5)
    r = results['routes']
    assert r['shared-b'][4]['interval']['mean_upper'] < 0.3 and r['shared-b'][4]['interval']['mean_lower'] < 0.02
    assert r['other'][4]['interval']['mean_lower'] > 0.2
    # noprefill ignores the forced history: same backend as the reference, yet far apart at position 4
    assert r['noprefill'][1]['interval']['mean_lower'] < 0.02 and r['noprefill'][4]['interval']['mean_lower'] > 0.1


def test_reconstruct_without_logprobs(world, tmp_path):
    from kremlin.investigations import reconstruct
    c = client(world, tmp_path, False)
    routes = [route(world, 'shared-a'), route(world, 'noprefill', 'shared-again'), route(world, 'other'),
              route(world, 'liar')]
    results, md = reconstruct.run(c, routes, prompts=3, samples=300)
    pairs = results['pairs']
    same, different = pairs['shared-a ~ shared-again']['mean_tv'], pairs['shared-a ~ other']['mean_tv']
    # 300 samples over 3 prompts: same-backend TV is sampling noise (0.07-0.17 observed), different backends ~0.78
    assert same < 0.3 < different
    assert results['routes']['liar']['top_k_ignored_prompts'] > 0          # liar ignores top_k
    assert results['routes']['shared-a']['self_check_tv'] < 0.3             # matches its own logprobs (noise ~0.1)


def test_multi_route_ttl(world, tmp_path):
    from kremlin.investigations import cache
    c = client(world, tmp_path, True)
    results, md = cache.run_ttl(c, [route(world, 'other'), route(world, 'shared-a')], delays=(1, 2), trials=1)
    assert set(results['routes']) == {'other', 'shared-a'}
    assert all(r['hits'] == r['n'] == 1 for rows in results['routes'].values() for r in rows)  # fake cache never expires


def test_hub_concurrent_workers_keep_every_run(world, tmp_path, capsys):
    """Regression: two workers once raced on index.json (partial reads, lost entries)."""
    import time as t
    import httpx
    from kremlin.hub import Hub, bundle, serve
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    runs = [run_cli(world, tmp_path, f'r{i}', 'billing', spec, '--cap', '8', '--repeats', '1') for i in range(5)]
    capsys.readouterr()
    hub = Hub(tmp_path / 'hub', [pubkey(world)], workers=4)
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: serve(hub, ('127.0.0.1', 0), lambda a: (addr.update(a=a), ready.set())),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{addr['a'][1]}"
    polls = [httpx.post(base + '/runs', content=bundle(r).read_bytes()).json()['poll'] for r in runs]
    for _ in range(600):
        states = [httpx.get(base + p).json()['status'] for p in polls]
        if all(s in ('accepted', 'rejected') for s in states):
            break
        t.sleep(0.1)
    assert states == ['accepted'] * 5, states
    assert httpx.get(base + '/runs').json()['total'] == 5


def test_hub_index_is_safe_under_concurrency(tmp_path):
    """8 writers x 50 appends with concurrent readers: no lost entries, no partial reads."""
    from kremlin.hub import Hub
    hub = Hub(tmp_path / 'hub', ['unused'], workers=0)
    errors, stop = [], threading.Event()

    def writer(w):
        for i in range(50):
            hub.append_index(dict(id=f'{w}-{i}'))

    def reader():
        while not stop.is_set():
            try:
                hub.index()
            except Exception as error:  # a partial file would raise here
                errors.append(error)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for thread in readers + writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()
    assert not errors and len(hub.index()) == 400


def test_site_pages_findings_and_embed(world, tmp_path, capsys):
    import httpx
    from kremlin.hub import Hub, bundle, serve
    spec = f"openai:https://localhost:{world['provider'].port}/v1|"
    run = run_cli(world, tmp_path, 'fp', 'fingerprint', spec + 'shared-a;key=FAKE_KEY;label=a',
                  spec + 'shared-b;key=FAKE_KEY;label=b', '--prompts', '12')
    capsys.readouterr()
    hub = Hub(tmp_path / 'hub', [pubkey(world)], base_path='/hub', submissions_per_hour=2)
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: serve(hub, ('127.0.0.1', 0), lambda a: (addr.update(a=a), ready.set())),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{addr['a'][1]}"
    poll = httpx.post(base + '/runs', content=bundle(run).read_bytes()).json()['poll']
    import time as t
    for _ in range(300):
        job = httpx.get(base + poll).json()
        if job['status'] in ('accepted', 'rejected'):
            break
        t.sleep(0.1)
    assert job['status'] == 'accepted', job
    run_id = job['entry']['id']
    page = httpx.get(base + '/').text
    assert 'Same backend (reseller detection)' in page and 'same numerics' in page
    assert f'href="/hub/runs/{run_id}/"' in page and 'href="/styles.css"' in page
    run_page = httpx.get(f'{base}/runs/{run_id}/').text
    assert 'audit: results reproduced' in run_page and '<table>' in run_page and 'Same-backend fingerprint' in run_page
    data = httpx.get(base + '/findings.json?kind=same-backend&provider=localhost')
    assert data.headers['access-control-allow-origin'] == '*'
    assert data.json()['count'] == 1 and data.json()['findings'][0]['flag'] == 'notable'
    assert 'data-kremlin-findings' in httpx.get(base + '/embed.js').text
    assert httpx.post(base + '/runs', content=b'x').status_code in (202, 422)  # 2nd submission this hour
    assert httpx.post(base + '/runs', content=b'y').status_code == 429      # 3rd: rate limited


def test_data_terms_are_announced(world, tmp_path):
    import httpx
    from kremlin.hub import Hub, serve
    info = httpx.get(world['witness'].split(',')[1] + '/pubkey').json()
    assert 'stored indefinitely' in info['data_terms'] and 'any purpose' in info['data_terms']
    hub = Hub(tmp_path / 'hub', [pubkey(world)], workers=0, base_path='/hub')
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: serve(hub, ('127.0.0.1', 0), lambda a: (addr.update(a=a), ready.set())),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{addr['a'][1]}"
    assert 'any purpose' in httpx.get(base + '/terms').text and 'Data use' in httpx.get(base + '/').text


def test_audit_tolerates_unanswered_attempts_only(world, tmp_path, capsys):
    """Relay-refused attempts (no HTTP response) may be unwitnessed; answered ones may not."""
    from kremlin.replay import AuditFailure, audit
    spec = f"openai:https://localhost:{world['provider'].port}/v1|other;key=FAKE_KEY"
    run = run_cli(world, tmp_path, 'run', 'billing', spec, '--cap', '8', '--repeats', '1')
    capsys.readouterr()
    lines = (run / 'requests.jsonl').read_text().splitlines()
    first = json.loads(lines[0])
    refused = dict(first, attempt=-1, result=dict(first['result'], status=None, ok=False,
                                                  error='WitnessError: relay refused tunnel: 429'))
    refused.pop('witness_receipt')
    (run / 'requests.jsonl').write_text('\n'.join([json.dumps(refused)] + lines) + '\n')
    import hashlib
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['files']['requests.jsonl'] = hashlib.sha256((run / 'requests.jsonl').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    report = audit(run, pubkey(world), require_witness=True)
    assert report['attested'] and report['results_reproduced'] and report['unanswered'] == 1
    answered = dict(first, attempt=-1)  # an answered attempt without its receipt must still fail
    answered.pop('witness_receipt')
    (run / 'requests.jsonl').write_text('\n'.join([json.dumps(answered)] + lines) + '\n')
    manifest['files']['requests.jsonl'] = hashlib.sha256((run / 'requests.jsonl').read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AuditFailure, match='not witnessed'):
        audit(run, pubkey(world), require_witness=True)


def test_hub_serves_client_downloads(tmp_path):
    import httpx
    from kremlin.hub import Hub, serve
    downloads = tmp_path / 'dl'
    downloads.mkdir()
    (downloads / 'kremlin-0.1.0-py3-none-any.whl').write_bytes(b'PK-wheel')
    (downloads / 'secret.txt').write_text('no')
    hub = Hub(tmp_path / 'hub', ['k'], workers=0, base_path='/kremlinology', public_url='https://x.test/kremlinology',
              downloads=downloads)
    ready, addr = threading.Event(), {}
    threading.Thread(target=lambda: serve(hub, ('127.0.0.1', 0), lambda a: (addr.update(a=a), ready.set())),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{addr['a'][1]}"
    assert httpx.get(base + '/download/kremlin-0.1.0-py3-none-any.whl').content == b'PK-wheel'
    assert httpx.get(base + '/download/secret.txt').status_code == 404
    assert httpx.get(base + '/download/..%2Fhub').status_code == 404
    assert "kremlin[witness] @ https://x.test/kremlinology/download/kremlin-0.1.0-py3-none-any.whl" in \
        httpx.get(base + '/').text.replace('&#x27;', "'")
