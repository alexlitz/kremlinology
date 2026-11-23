"""Witnessed sessions: SDK transport, local proxy (base-URL and HTTPS_PROXY modes), and their audit."""
import json
import threading

import httpx
import pytest

from kremlin import session as session_mod
from kremlin.replay import AuditFailure, audit

from test_witness import SECRET_KEY, setup  # noqa: F401  (fixture)

BODY = dict(model='m', messages=[dict(role='user', content='hi')], max_tokens=2)


def start_proxy(sess, **kwargs):
    server = session_mod.serve_proxy(sess, '127.0.0.1:0', quiet=True, **kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


def check_run(root, public_key, n):
    report = audit(root, public_key, require_witness=True)
    assert report['attested'] and report['results_reproduced'] and report['witnessed'] == n
    assert report['requests_disclosed'] == n
    text = ''.join(p.read_text() for p in root.rglob('*') if p.is_file())
    assert SECRET_KEY not in text
    return json.loads((root / 'results.json').read_text())['results']


def test_http_client_with_openai_sdk(setup, tmp_path):
    openai = pytest.importorskip('openai')
    root = tmp_path / 'run'
    http = session_mod.http_client(root, setup['client'])
    sdk = openai.OpenAI(base_url=f"https://localhost:{setup['port']}/v1", api_key=SECRET_KEY, http_client=http,
                        max_retries=0)
    reply = sdk.chat.completions.create(**BODY)
    assert reply.id == 'gen-1' and reply.usage.completion_tokens == 2
    http.close()
    results = check_run(root, setup['client'].public_key, 1)
    call = results['calls'][0]
    assert call['id'] == 'gen-1' and call['model_requested'] == 'm' and call['witnessed']
    assert SECRET_KEY.encode() in setup['seen'][0]  # the provider got the key; the evidence never did


def test_async_http_client(setup, tmp_path):
    import asyncio
    root = tmp_path / 'run'

    async def go():
        async with session_mod.async_http_client(root, setup['client']) as http:
            replies = await asyncio.gather(*[
                http.post(f"https://localhost:{setup['port']}/v1/chat/completions", json=BODY,
                          headers={'Authorization': f'Bearer {SECRET_KEY}'}) for _ in range(3)])
        return replies
    replies = asyncio.run(go())
    assert [r.status_code for r in replies] == [200, 200, 200]
    check_run(root, setup['client'].public_key, 3)


def test_proxy_base_url_and_https_proxy_modes(setup, tmp_path):
    root = tmp_path / 'run'
    sess = session_mod.Session(root, setup['client'])
    ca = session_mod.LocalCA(tmp_path / 'ca')
    server, base = start_proxy(sess, ca=ca, hosts=['localhost'])
    headers = {'Authorization': f'Bearer {SECRET_KEY}'}
    # 1. base URL: http://proxy/<host>/<path>
    reply = httpx.post(f"{base}/localhost:{setup['port']}/v1/chat/completions", json=BODY, headers=headers)
    assert reply.status_code == 200 and reply.json()['id'] == 'gen-1'
    assert reply.headers['x-kremlin-witnessed'] == 'request+response'
    # 2. HTTPS_PROXY with the local CA; two requests on one kept-alive client connection
    with httpx.Client(proxy=base, verify=str(ca.cert_path)) as client:
        for _ in range(2):
            reply = client.post(f"https://localhost:{setup['port']}/v1/chat/completions", json=BODY, headers=headers)
            assert reply.status_code == 200 and reply.json()['usage']['completion_tokens'] == 2
    # 3. errors come back as JSON, not as a hung connection
    assert httpx.get(f'{base}/no-host-here').status_code == 400
    server.shutdown()
    sess.close()
    results = check_run(root, setup['client'].public_key, 3)
    assert {c['url'] for c in results['calls']} == {f"https://localhost:{setup['port']}/v1/chat/completions"}
    assert 'kremlin' in ca.bundle_path.read_text() or ca.bundle_path.stat().st_size > ca.cert_path.stat().st_size


def test_session_audit_detects_tampering(setup, tmp_path):
    from kremlin import evidence
    root = tmp_path / 'run'
    with session_mod.Session(root, setup['client']) as sess:
        sess.send('POST', f"https://localhost:{setup['port']}/v1/chat/completions",
                  {'Authorization': f'Bearer {SECRET_KEY}', 'Content-Type': 'application/json'},
                  json.dumps(BODY).encode())
    lines = (root / 'requests.jsonl').read_text().splitlines()
    record = json.loads(lines[0])
    record['request_base64'] = __import__('base64').b64encode(b'{"model": "other"}').decode()
    (root / 'requests.jsonl').write_text(json.dumps(record) + '\n')
    evidence.Evidence(root, 'session', {}, resume=True).close()  # re-hash so only the receipt check can catch it
    with pytest.raises(AuditFailure, match='request body differs'):
        audit(root, setup['client'].public_key, require_witness=True)


def test_target_url():
    assert session_mod.target_url('/openrouter.ai/api/v1/chat/completions') == \
        'https://openrouter.ai/api/v1/chat/completions'
    assert session_mod.target_url('/v1/messages', 'https://api.anthropic.com') == 'https://api.anthropic.com/v1/messages'
    with pytest.raises(ValueError):
        session_mod.target_url('/v1/chat/completions')


def test_finish_after_kill(setup, tmp_path):
    root = tmp_path / 'run'
    sess = session_mod.Session(root, setup['client'])
    sess.send('POST', f"https://localhost:{setup['port']}/v1/chat/completions",
              {'Authorization': f'Bearer {SECRET_KEY}'}, json.dumps(BODY).encode())
    assert not (root / 'manifest.json').exists()  # process "killed" here
    session_mod.Session.finish(root)
    check_run(root, setup['client'].public_key, 1)


def test_proxy_cors_and_host_limits(setup, tmp_path):
    sess = session_mod.Session(tmp_path / 'run', setup['client'])
    server, base = start_proxy(sess, hosts=['localhost'], cors=['http://app.example'])
    url = f"{base}/localhost:{setup['port']}/v1/chat/completions"
    headers = {'Authorization': f'Bearer {SECRET_KEY}'}
    # a page from an allowed origin: preflight, then the call, with exactly one Allow-Origin
    pre = httpx.options(url, headers={'Origin': 'http://app.example', 'Access-Control-Request-Method': 'POST',
                                      'Access-Control-Request-Headers': 'authorization,content-type'})
    assert pre.status_code == 204 and pre.headers['access-control-allow-origin'] == 'http://app.example'
    assert 'authorization' in pre.headers['access-control-allow-headers']
    reply = httpx.post(url, json=BODY, headers={**headers, 'Origin': 'http://app.example'})
    assert reply.status_code == 200
    assert reply.headers.get_list('access-control-allow-origin') == ['http://app.example']
    assert 'X-Kremlin-Witnessed' in reply.headers['access-control-expose-headers']
    # any other web page you visit cannot use the proxy
    assert httpx.options(url, headers={'Origin': 'http://evil.example'}).status_code == 403
    assert httpx.post(url, json=BODY, headers={**headers, 'Origin': 'http://evil.example'}).status_code == 403
    # hosts outside --hosts are refused in base-URL form
    assert httpx.get(f'{base}/example.com/x', headers=headers).status_code == 403
    server.shutdown()
    sess.close()
    assert len(sess.evidence.records()) == 1


def test_env_lines_cover_each_stack(tmp_path):
    ca = session_mod.LocalCA(tmp_path / 'ca')
    text = '\n'.join(session_mod.env_lines('127.0.0.1:8787', ca))
    for name in ('HTTPS_PROXY', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'NODE_EXTRA_CA_CERTS', 'NODE_USE_ENV_PROXY',
                 'DENO_CERT', 'PHP_INI_SCAN_DIR', 'CURL_CA_BUNDLE'):
        assert name in text
    assert 'curl.cainfo' in (ca.dir / 'php' / 'kremlin.ini').read_text()
    import shutil
    if shutil.which('keytool'):
        assert 'JAVA_TOOL_OPTIONS' in text and (ca.dir / 'java-truststore.p12').exists()


def start_sse_server(cert_dir, gap=0.3):
    """TLS server that streams four SSE events, `gap` seconds apart, chunked."""
    import socket
    import ssl
    import time
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(cert_dir / 'leaf.pem', cert_dir / 'leaf.key')
    listener = socket.create_server(('127.0.0.1', 0))

    def serve_one(conn):
        try:
            with context.wrap_socket(conn, server_side=True) as tls:
                data = b''
                while b'\r\n\r\n' not in data:
                    data += tls.recv(4096)
                head, _, body = data.partition(b'\r\n\r\n')
                length = int(next(l.split(b':')[1] for l in head.split(b'\r\n')
                                  if l.lower().startswith(b'content-length')))
                while len(body) < length:
                    body += tls.recv(4096)
                tls.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n')
                for i in range(3):
                    event = ('data: ' + json.dumps(dict(id='gen-s', choices=[dict(index=0, delta=dict(content=f't{i}'))]))
                             + '\n\n').encode()
                    tls.sendall(f'{len(event):x}\r\n'.encode() + event + b'\r\n')
                    time.sleep(gap)
                done = b'data: [DONE]\n\n'
                tls.sendall(f'{len(done):x}\r\n'.encode() + done + b'\r\n0\r\n\r\n')
        except (ssl.SSLError, OSError):
            pass

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=serve_one, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    return listener, listener.getsockname()[1]


def test_dechunker_byte_by_byte():
    raw = b'5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n'
    decoder = session_mod.Dechunker(True)
    assert b''.join(decoder.feed(raw[i:i + 1]) for i in range(len(raw))) == b'hello world'
    assert session_mod.Dechunker(False).feed(b'abc') == b'abc'


def test_streaming_is_live_through_proxy_and_transports(setup, tmp_path):
    import asyncio
    import time
    listener, port = start_sse_server(setup['ca'].parent)
    url = f'https://localhost:{port}/v1/chat/completions'
    headers = {'Authorization': f'Bearer {SECRET_KEY}'}
    body = dict(BODY, stream=True)
    root = tmp_path / 'run'
    sess = session_mod.Session(root, setup['client'])

    def timed(lines):
        start, seen = time.time(), []
        for line in lines:
            if line.startswith('data:'):
                seen.append((round(time.time() - start, 2), line))
        return seen

    # 1. proxy, base-URL form
    server, base = start_proxy(sess, hosts=['localhost'])
    with httpx.stream('POST', f'{base}/localhost:{port}/v1/chat/completions', json=body, headers=headers) as reply:
        assert reply.headers['x-kremlin-witnessed'] == 'streaming'
        events = timed(reply.iter_lines())
    # 2. httpx transport (what SDKs use)
    http = httpx.Client(transport=session_mod.WitnessTransport(sess))
    with http.stream('POST', url, json=body, headers=headers) as reply:
        events2 = timed(reply.iter_lines())

    # 3. async transport
    async def go():
        client = httpx.AsyncClient(transport=session_mod.AsyncWitnessTransport(sess))
        async with client.stream('POST', url, json=body, headers=headers) as reply:
            start, seen = time.time(), []
            async for line in reply.aiter_lines():
                if line.startswith('data:'):
                    seen.append((round(time.time() - start, 2), line))
            return seen
    events3 = asyncio.run(go())
    for seen in (events, events2, events3):
        assert [line for _, line in seen][-1] == 'data: [DONE]' and len(seen) == 4
        assert seen[-1][0] - seen[0][0] >= 0.5, seen  # the first event arrived well before the last: live
    server.shutdown()
    sess.close()
    listener.close()
    results = check_run(root, setup['client'].public_key, 3)
    assert all(c['stream'] for c in results['calls'])


def test_openai_sdk_streams_live(setup, tmp_path):
    openai = pytest.importorskip('openai')
    import time
    listener, port = start_sse_server(setup['ca'].parent)
    root = tmp_path / 'run'
    http = session_mod.http_client(root, setup['client'])
    sdk = openai.OpenAI(base_url=f'https://localhost:{port}/v1', api_key=SECRET_KEY, http_client=http, max_retries=0)
    start, times, text = time.time(), [], ''
    for chunk in sdk.chat.completions.create(model='m', messages=[dict(role='user', content='hi')], stream=True):
        times.append(time.time() - start)
        text += chunk.choices[0].delta.content or ''
    assert text == 't0t1t2' and times[-1] - times[0] >= 0.5
    http.close()
    listener.close()
    check_run(root, setup['client'].public_key, 1)


def test_session_findings():
    from kremlin.findings import extract
    calls = [dict(n=i, url='https://openrouter.ai/api/v1/chat/completions', status=200, witnessed=True,
                  latency_s=0.2 + i / 10, model_requested='a/b', model_served='a/b', provider='Parasail')
             for i in range(3)]
    calls.append(dict(n=3, url='https://openrouter.ai/api/v1/chat/completions', status=200, witnessed=True,
                      latency_s=0.5, model_requested='a/b', model_served='c/d', provider='X'))
    calls.append(dict(n=4, url='https://openrouter.ai/x', status=None, witnessed=False))
    calls.append(dict(n=5, url='https://openrouter.ai/api/v1/chat/completions', status=429, witnessed=True,
                      model_requested='a/b'))
    rows = extract('session', dict(calls=calls), [])
    assert len(rows) == 2
    normal = next(r for r in rows if r['model'] == 'a/b')
    assert normal['subjects'] == ['openrouter.ai', 'Parasail'] and normal['value'] == pytest.approx(0.3)
    assert normal['flag'] == 'normal' and '3 calls' in normal['metric']
    swapped = next(r for r in rows if r['model'] == 'c/d')
    assert swapped['flag'] == 'notable' and 'requested a/b' in swapped['verdict']


def test_env_shells_and_trust_commands(tmp_path):
    ca = session_mod.LocalCA(tmp_path / 'ca')
    sh = session_mod.env_lines('127.0.0.1:8787', ca, 'sh')
    assert "export HTTPS_PROXY=http://127.0.0.1:8787" in sh[0]
    ps = session_mod.env_lines('127.0.0.1:8787', ca, 'powershell')
    assert ps[0].startswith("$env:HTTPS_PROXY = 'http://127.0.0.1:8787'")
    fish = session_mod.env_lines('127.0.0.1:8787', ca, 'fish')
    assert fish[0].startswith('set -gx HTTPS_PROXY ')
    cmd = session_mod.env_lines('127.0.0.1:8787', ca, 'cmd')
    assert cmd[0] == 'set "HTTPS_PROXY=http://127.0.0.1:8787"'
    with pytest.raises(ValueError):
        session_mod.env_lines('127.0.0.1:8787', ca, 'tcsh')
    assert session_mod.trust_commands(ca, 'darwin')[0][:2] == ['security', 'add-trusted-cert']
    assert session_mod.trust_commands(ca, 'win32')[0][:4] == ['certutil', '-user', '-addstore', 'Root']
    assert str(ca.cert_path) in session_mod.trust_commands(ca, 'linux')[0]
    assert session_mod.untrust_commands(ca, 'win32')[0][2] == '-delstore'


def test_describe_ignores_unwitnessed_request_fields():
    import base64
    record = dict(n=0, method='POST', url='https://x.example/v1/chat/completions', status=200, error=None,
                  request_base64=base64.b64encode(json.dumps(dict(model='claimed/model', stream=True)).encode()).decode(),
                  witness_receipt=dict(response_records=[], request_complete_ms=None))  # response-only receipt
    row = session_mod.describe(record, b'{"model": "served/model", "provider": "P"}')
    assert 'model_requested' not in row and 'stream' not in row  # submitter's claim, not witnessed
    assert row['model_served'] == 'served/model'  # from the witnessed response
    record['witness_receipt']['request'] = dict(path='/v1/chat/completions')
    assert session_mod.describe(record, b'{}')['model_requested'] == 'claimed/model'
