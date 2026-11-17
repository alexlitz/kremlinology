"""kremlin: black-box forensics for LLM inference APIs.

Routes (see `kremlin help-routes`):
  openrouter:MODEL@PROVIDER   openai:https://host/v1|MODEL;key=ENV   anthropic:MODEL
Every paid command needs --out DIR and respects --max-usd (default $0.25). --plan prints the
request count without sending anything. --witness TUNNEL,CONTROL[,PUBKEY] routes every request
through a kremlin-witness relay and stores its signed receipts in the evidence.
"""
import argparse
import json
import os
import sys

from . import evidence, routes as routes_mod
from .client import BudgetExceeded, Client
from .investigations import billing, cache, fingerprint, later, params, reconstruct, sampler, timing, tv

COMMANDS = {
    'fingerprint': (fingerprint.plan, fingerprint.run, 2, 'Do routes share a backend? (exact logprob identity)'),
    'tv': (tv.plan, tv.run, 2, 'Output differences vs the first route (dataset-average TV)'),
    'cache-share': (cache.plan_share, cache.run_share, 2, 'Is a prompt cached by route A readable by route B?'),
    'cache-blocks': (cache.plan_blocks, cache.run_blocks, 1, 'Cache granularity and minimum cached length'),
    'cache-ttl': (cache.plan_ttl, cache.run_ttl, 1, 'How long cache entries survive (one or many routes at once)'),
    'cache-prefix': (cache.plan_prefix, cache.run_prefix, 1, 'Partial-prefix / breakpoint resume semantics'),
    'billing': (billing.plan, billing.run, 1, 'Output-cap compliance and what is billed'),
    'timing': (timing.plan, timing.run, 1, 'Decode speed by content: speculation signature, bandwidth bound'),
    'reconstruct': (reconstruct.plan, reconstruct.run, 2, 'Compare routes without logprobs: laws from samples'),
    'later': (later.plan, later.run, 2, 'Output differences at later positions (prefill-forced history)'),
    'params': (params.plan, params.run, 1, 'Are logprobs, n, seed, stop, max_tokens, T=0 honoured? Template parity'),
    'sampler': (sampler.plan, sampler.run, 1, 'Are top_k and temperature applied to the reported distribution?'),
}


NON_OPTIONS = ('routes', 'command', 'out', 'max_usd', 'max_requests', 'plan', 'witness', 'trust',
               'no_disclose', 'witness_command', 'concurrency', 'resume')


def options_for(args):
    """Investigation keyword options from parsed CLI args (also used by `kremlin audit`)."""
    options = {k: v for k, v in args.items() if k not in NON_OPTIONS}
    if options.get('extra') and isinstance(options['extra'], str):
        options['extra'] = json.loads(options['extra'])
    if 'positions' in options and isinstance(options['positions'], str):
        options['positions'] = tuple(int(x) for x in options['positions'].split(','))
    if 'delays' in options and isinstance(options['delays'], str):
        options['delays'] = tuple(int(x) for x in options['delays'].split(','))
    if 'prompt_file' in options and options['prompt_file'] is None:
        options.pop('prompt_file')
    return options


def make_witness(spec, trust_file=None):
    if not spec:
        return None
    from .witness.client import WitnessClient
    parts = spec.split(',')
    if len(parts) not in (2, 3):
        raise SystemExit('--witness needs TUNNEL_HOST:PORT,CONTROL_URL[,PUBKEY_BASE64]')
    client = WitnessClient(parts[0], parts[1], public_key=parts[2] if len(parts) == 3 else None, trust_file=trust_file)
    info = client.info()
    if info.get('data_terms'):
        import sys as _sys
        print(f"witness data terms: {info['data_terms']}"
              + (f" ({info['terms_url']})" if info.get('terms_url') else ''), file=_sys.stderr)
    return client


def common(parser):
    parser.add_argument('--out', help='new evidence directory (required unless --plan)')
    parser.add_argument('--max-usd', type=float, default=0.25)
    parser.add_argument('--max-requests', type=int, default=5000)
    parser.add_argument('--plan', action='store_true', help='print what would be sent and exit')
    parser.add_argument('--witness', help='TUNNEL,CONTROL_URL[,PUBKEY] of a kremlin-witness relay')
    parser.add_argument('--no-disclose', action='store_true',
                        help='with --witness: attest responses only, keep request bodies private from the witness')
    parser.add_argument('--trust', help='extra CA bundle (PEM) for TLS verification')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=8, help='requests in flight at once (default 8)')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='kremlin', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, (_, _, minimum, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument('routes', nargs='+')
        common(p)
        if name == 'later':
            p.add_argument('--prompts', type=int, default=40)
            p.add_argument('--prompt-file')
            p.add_argument('--positions', default='1,4,8', help='token positions, comma separated')
            p.add_argument('--top-logprobs', type=int, default=20)
            p.add_argument('--extra')
        if name in ('fingerprint', 'tv'):
            p.add_argument('--prompts', type=int, default=60 if name == 'fingerprint' else 100)
            p.add_argument('--prompt-file', help='text (one per line) or JSONL with "text"')
            p.add_argument('--repeats', type=int, default=2)
            p.add_argument('--top-logprobs', type=int, default=5 if name == 'fingerprint' else 20)
            p.add_argument('--extra', help='JSON merged into every request body, e.g. {"reasoning":{"enabled":false}}')
        if name.startswith('cache'):
            p.add_argument('--trials', type=int, default=3 if name == 'cache-share' else 2)
            p.add_argument('--chars', type=int, default=24000)
            p.add_argument('--delays', default='30,120,300,600', help='cache-ttl: seconds, comma separated')
            p.add_argument('--ttl-hint', choices=['5m', '1h'], help='cache-ttl: Anthropic cache_control ttl')
        if name == 'billing':
            p.add_argument('--cap', type=int, default=64)
            p.add_argument('--repeats', type=int, default=2)
        if name == 'reconstruct':
            p.add_argument('--prompts', type=int, default=5)
            p.add_argument('--prompt-file')
            p.add_argument('--samples', type=int, default=200)
            p.add_argument('--temperature', type=float, default=2.0)
            p.add_argument('--top-k', type=int, default=10)
        if name == 'sampler':
            p.add_argument('--samples', type=int, default=60)
        if name == 'timing':
            p.add_argument('--timing-concurrency', type=int, default=1,
                           help='parallel streams (default 1: concurrency can change the speed being measured)')
            p.add_argument('--repeats', type=int, default=6)
            p.add_argument('--max-tokens', type=int, default=256)
            p.add_argument('--active-params', type=float, help='active parameters (e.g. 7.06e10) for bandwidth bounds')
    req = sub.add_parser('request', help='send one request to any endpoint (chat, embeddings, images, audio...)')
    req.add_argument('route')
    req.add_argument('path', help='path under the route base URL, e.g. embeddings or audio/transcriptions')
    req.add_argument('--json', help='JSON body (string or @file)')
    req.add_argument('-F', '--form', action='append', default=[], help='multipart field name=value or name=@file')
    req.add_argument('--method', default='POST')
    common(req)
    prx = sub.add_parser('proxy', help='local witnessed proxy: point any SDK / curl base URL at it')
    prx.add_argument('--out', help='new evidence directory for this session')
    prx.add_argument('--finish', metavar='DIR', help='finalize a session whose proxy was killed, then exit')
    prx.add_argument('--listen', default='127.0.0.1:8787')
    prx.add_argument('--upstream', help='forward every path to this base URL (default: the path starts with the host, '
                                        'e.g. /openrouter.ai/api/v1/chat/completions)')
    prx.add_argument('--witness', help='TUNNEL,CONTROL_URL[,PUBKEY] (default: $KREMLIN_WITNESS or the public relay)')
    prx.add_argument('--no-disclose', action='store_true', help='attest responses only; requests stay private')
    prx.add_argument('--max-requests', type=int, help='refuse further requests after this many')
    prx.add_argument('--trust')
    prx.add_argument('--hosts', help="comma-separated host globs to witness, or '*' (default: the common LLM APIs, "
                                     'see kremlin.session.DEFAULT_HOSTS); in HTTPS_PROXY mode other hosts are '
                                     'tunnelled through untouched and unrecorded')
    prx.add_argument('--cors', action='append', default=[], metavar='ORIGIN',
                     help="let browser pages from ORIGIN call the proxy (repeatable; '*' for any origin)")
    prx.add_argument('--no-mitm', action='store_true', help='disable HTTPS_PROXY mode (base-URL form only)')
    prx.add_argument('--ca-dir', help='local CA directory (default ~/.config/kremlin/proxy-ca)')
    prx.add_argument('--quiet', action='store_true')
    env = sub.add_parser('env', help='print shell exports that route any language through `kremlin proxy`')
    env.add_argument('--listen', default='127.0.0.1:8787')
    env.add_argument('--ca-dir')
    env.add_argument('--shell', choices=['sh', 'fish', 'powershell', 'cmd'], default='sh',
                     help='sh (bash/zsh, default), fish, powershell, cmd')
    tru = sub.add_parser('trust', help="add the proxy's local CA to your user trust store (macOS keychain, Windows "
                                      'user store, Linux NSS for Chrome); prints the commands unless --install')
    tru.add_argument('--install', action='store_true', help='run the commands')
    tru.add_argument('--remove', action='store_true', help='undo: remove the CA from the store')
    tru.add_argument('--ca-dir')
    prm = sub.add_parser('prompts', help='build a prompt file from a public dataset (needs `datasets`)')
    prm.add_argument('source', help='codealpaca | wikitext | hf:REPO:SPLIT:FIELD')
    prm.add_argument('--n', type=int, default=200)
    prm.add_argument('--seed', type=int, default=0)
    prm.add_argument('--out', required=True)
    rts = sub.add_parser('routes', help='list OpenRouter endpoints for a model (free) to build route specs')
    rts.add_argument('model')
    wit = sub.add_parser('witness', help='run a relay or verify a receipt')
    wsub = wit.add_subparsers(dest='witness_command', required=True)
    wsub.add_parser('serve', help='same as kremlin-witness', add_help=False)
    wv = wsub.add_parser('verify', help='verify a receipt JSON with the witness public key')
    wv.add_argument('receipt')
    wv.add_argument('--pubkey', required=True)
    res = sub.add_parser('resume', help='continue an interrupted run: completed calls are reused, the rest are sent')
    res.add_argument('run')
    res.add_argument('--max-usd', type=float, help='new total budget (default: the original)')
    res.add_argument('--witness', help='witness spec (default: the original)')
    res.add_argument('--trust')
    res.add_argument('--concurrency', type=int)
    aud = sub.add_parser('audit', help='verify receipts and recompute results from an evidence directory')
    aud.add_argument('run')
    aud.add_argument('--pubkey', help='witness public key (base64); without it nothing is attested')
    aud.add_argument('--require-witness', action='store_true', help='fail unless every record is witnessed')
    bun = sub.add_parser('bundle', help='pack an evidence directory into a tar.gz for submission')
    bun.add_argument('run')
    bun.add_argument('-o', '--output')
    sbm = sub.add_parser('submit', help='upload a run (directory or bundle) to a kremlin-hub')
    sbm.add_argument('run')
    sbm.add_argument('--hub', default=os.environ.get('KREMLIN_HUB', 'https://alexlitzenberger.com/kremlinology'),
                     help='hub base URL (default: $KREMLIN_HUB or the public hub)')
    ver = sub.add_parser('verify', help='check an evidence directory against its manifest')
    ver.add_argument('run')
    sub.add_parser('help-routes', help='route syntax')
    args, rest = parser.parse_known_args(argv)

    if args.command == 'help-routes':
        print(routes_mod.__doc__)
        return
    if args.command == 'audit':
        from .replay import AuditFailure, audit
        try:
            result = audit(args.run, args.pubkey, args.require_witness)
        except AuditFailure as error:
            print(json.dumps(dict(ok=False, failure=str(error)), indent=1))
            raise SystemExit(1)
        print(json.dumps(dict(ok=result['results_reproduced'], **result), indent=1))
        raise SystemExit(0 if result['results_reproduced'] else 1)
    if args.command == 'prompts':
        from . import datasets
        items = datasets.build(args.source, args.n, args.seed)
        datasets.write(items, args.out)
        print(f'{len(items)} prompts from {args.source} -> {args.out} (use --prompt-file {args.out})')
        return
    if args.command == 'resume':
        return resume_run(args)
    if args.command == 'bundle':
        from .hub import bundle
        print(bundle(args.run, args.output))
        return
    if args.command == 'submit':
        import httpx
        from pathlib import Path
        from .hub import bundle
        path = Path(args.run)
        data = path.read_bytes() if path.is_file() else bundle(path, Path(tempfile_name(path))).read_bytes()
        import time as time_module
        base = args.hub.rstrip('/')
        reply = httpx.post(base + '/runs', content=data, timeout=600, headers={'Content-Type': 'application/gzip'})
        if reply.status_code != 202:
            print(json.dumps(reply.json(), indent=1))
            raise SystemExit(1)
        job = reply.json()
        print(f"queued as job {job['job']}; waiting for the audit...", flush=True)
        deadline = time_module.time() + 3600
        while time_module.time() < deadline:
            status = httpx.get(base + job['poll'], timeout=60).json()
            if status['status'] in ('accepted', 'rejected'):
                print(json.dumps(status, indent=1))
                raise SystemExit(0 if status['status'] == 'accepted' else 1)
            time_module.sleep(2)
        raise SystemExit('timed out waiting for the hub')
    if args.command == 'verify':
        print(json.dumps(evidence.verify(args.run), indent=1))
        return
    if args.command == 'witness':
        if args.witness_command == 'serve':
            from .witness import relay
            sys.argv = ['kremlin-witness'] + rest
            return relay.main()
        from .witness.client import verify_receipt_file
        print(json.dumps(verify_receipt_file(args.receipt, args.pubkey), indent=1))
        return
    if rest:
        parser.error(f'unrecognised arguments: {rest}')
    if args.command == 'routes':
        return list_routes(args.model)
    if args.command == 'request':
        return single_request(args)
    if args.command == 'proxy':
        return run_proxy(args)
    if args.command == 'env':
        from . import session as session_mod
        print('\n'.join(session_mod.env_lines(args.listen, session_mod.LocalCA(args.ca_dir), args.shell)))
        return
    if args.command == 'trust':
        import shlex
        import subprocess
        from . import session as session_mod
        ca = session_mod.LocalCA(args.ca_dir)
        commands = (session_mod.untrust_commands if args.remove else session_mod.trust_commands)(ca)
        for argv in commands:
            print(' '.join(shlex.quote(a) for a in argv))
            if args.install or args.remove:
                done = subprocess.run(argv)
                if done.returncode:
                    raise SystemExit(f'failed ({done.returncode}): {argv[0]}')
        if not (args.install or args.remove):
            print('# run these yourself, or: kremlin trust --install   (undo: kremlin trust --remove)')
        return
    plan_fn, run_fn, minimum, _ = COMMANDS[args.command]
    route_list = routes_mod.unique_labels([routes_mod.parse_route(r) for r in args.routes])
    if len(route_list) < minimum or (args.command == 'cache-share' and len(route_list) != 2):
        parser.error(f'{args.command} needs {"exactly 2" if args.command == "cache-share" else f"at least {minimum}"} routes')
    options = options_for(vars(args))
    planned = plan_fn(route_list, **options)
    if args.plan:
        print(json.dumps(dict(command=args.command, routes=[r.label for r in route_list], **planned), indent=1))
        return
    if not args.out:
        parser.error('--out DIR is required (or use --plan)')
    from . import text
    try:
        witness = make_witness(args.witness, args.trust)  # fail before creating any evidence
    except Exception as error:
        raise SystemExit(f'witness unreachable: {error}')
    salt = options['salt'] = text.fresh_salt()
    run_dir = evidence.Evidence(args.out, args.command, dict(vars(args), routes=args.routes, salt=salt))
    client = Client(run_dir, max_usd=args.max_usd, max_requests=args.max_requests, witness=witness,
                    trust_file=args.trust, disclose_requests=not args.no_disclose, concurrency=args.concurrency)
    execute(run_fn, client, run_dir, route_list, options, args.command, bool(args.witness), args.out)


def execute(run_fn, client, run_dir, route_list, options, command, witnessed, out):
    status = 'completed'
    try:
        results, report_md = run_fn(client, route_list, **options)
    except BudgetExceeded as error:
        status, results, report_md = f'stopped: {error}', None, f'# Stopped\n\n{error}\n\nResume with `kremlin resume {out} --max-usd N`.\n'
    except KeyboardInterrupt:
        status, results, report_md = 'interrupted', None, f'# Interrupted\n\nResume with `kremlin resume {out}`.\n'
    run_dir.write_json('results.json', dict(status=status, usage=client.summary(), results=results))
    run_dir.write_text('report.md', report_md + f"\n\n---\nSpend: ${client.spent:.4f} reported/estimated over "
                                                f"{client.requests} requests ({client.unknown_cost} with unknown cost"
                                                f"{f', {client.resumed} calls reused from before a resume' if client.resumed else ''})."
                                                f"{' Witnessed.' if witnessed else ''}\n")
    run_dir.close(status=status, usage=client.summary())
    print(report_md)
    print(f'\n[{status}] evidence in {out} - spend ${client.spent:.4f}, {client.requests} requests')


def resume_run(args):
    """Re-run the same investigation with the same options and salt; recorded calls are served from evidence."""
    from pathlib import Path
    root = Path(args.run)
    meta = json.loads((root / 'run.json').read_text())
    original = meta['args']
    command = meta['command']
    plan_fn, run_fn, _, _ = COMMANDS[command]
    route_list = routes_mod.unique_labels([routes_mod.parse_route(r) for r in original['routes']])
    options = options_for(original)
    witness_spec = args.witness or original.get('witness')
    trust = args.trust or original.get('trust')
    try:
        witness = make_witness(witness_spec, trust)
    except Exception as error:
        raise SystemExit(f'witness unreachable: {error}')
    run_dir = evidence.Evidence(root, command, original, resume=True)
    client = Client(run_dir, max_usd=args.max_usd or original.get('max_usd', 0.25),
                    max_requests=original.get('max_requests', 5000), witness=witness, trust_file=trust,
                    disclose_requests=not original.get('no_disclose'),
                    concurrency=args.concurrency or original.get('concurrency', 8), resume_records=run_dir.records())
    print(f'resuming {command}: {len(client.completed)} completed calls on record')
    execute(run_fn, client, run_dir, route_list, options, command, bool(witness_spec), str(root))


def tempfile_name(path):
    import tempfile
    return tempfile.mkdtemp(prefix='kremlin-bundle-') + '/' + path.name + '.tar.gz'


def single_request(args):
    route = routes_mod.parse_route(args.route)
    body = None
    if args.json:
        body = json.loads(open(args.json[1:]).read() if args.json.startswith('@') else args.json)
    files, data = {}, {}
    for field in args.form:
        name, _, value = field.partition('=')
        if value.startswith('@'):
            files[name] = (value[1:].split('/')[-1], open(value[1:], 'rb').read())
        else:
            data[name] = value
    if args.plan:
        print(json.dumps(dict(route=route.label, url=f"{route.base_url}/{args.path}", json=body, files=list(files),
                              form=data), indent=1))
        return
    if not args.out:
        raise SystemExit('--out DIR is required')
    run_dir = evidence.Evidence(args.out, 'request', vars(args))
    client = Client(run_dir, max_usd=args.max_usd, witness=make_witness(args.witness, args.trust), trust_file=args.trust)
    status, raw, receipt = client.request(route, args.method, args.path, json_body=body, files=files or None,
                                          data=data or None)
    run_dir.close(status=status)
    shown = raw[:2000].decode('utf-8', 'replace') if raw else ''
    print(f'HTTP {status}{" (witnessed, receipt in evidence)" if receipt else ""}\n{shown}')


def run_proxy(args):
    import signal
    from . import session as session_mod
    if args.finish:
        print(f'finalized {session_mod.Session.finish(args.finish)}')
        return
    if not args.out:
        raise SystemExit('--out DIR is required')
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # `kill` finishes the session like Ctrl-C
    signal.signal(signal.SIGINT, signal.default_int_handler)
    witness = session_mod.make_witness(args.witness, args.trust)
    sess = session_mod.Session(args.out, witness, disclose=not args.no_disclose, max_requests=args.max_requests,
                               args=dict(via='proxy', upstream=args.upstream, listen=args.listen))
    ca = None if args.no_mitm else session_mod.LocalCA(args.ca_dir)
    hosts = [h.strip() for h in args.hosts.split(',')] if args.hosts else session_mod.DEFAULT_HOSTS
    if hosts == ['*'] or args.upstream:
        hosts = None
    server = session_mod.serve_proxy(sess, args.listen, args.upstream, quiet=args.quiet, ca=ca, hosts=hosts,
                                     cors=args.cors or None)
    base = f'http://{args.listen}'
    example = args.upstream and base or f'{base}/openrouter.ai/api/v1'
    print(f"""witnessed proxy on {base} -> evidence in {args.out}
  OpenAI-style SDKs:  OPENAI_BASE_URL={example}
  Anthropic SDK:      ANTHROPIC_BASE_URL={base}/api.anthropic.com
  curl:               curl {example}/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" ...
  any host:           {base}/<api host>/<path>
{'' if ca is None else f'''  or, with no code change (any language): HTTPS_PROXY={base} plus trust {ca.cert_path}
     (SSL_CERT_FILE / REQUESTS_CA_BUNDLE for Python, NODE_EXTRA_CA_CERTS for Node, CURL_CA_BUNDLE or --cacert for curl)
     all the usual exports: eval "$(kremlin env --listen {args.listen})"   (PowerShell: kremlin env --shell powershell | iex)
     apps that only read the OS trust store (.NET on macOS/Windows, browsers): kremlin trust --install
'''}Ctrl-C finishes the session; then: kremlin audit {args.out} --pubkey {witness.public_key}  /  kremlin submit {args.out}""",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        root = sess.close()
        records = sess.evidence.records()
        print(f'\nsession closed: {len(records)} exchanges recorded in {root}')


def list_routes(model):
    import httpx
    data = httpx.get(f'https://openrouter.ai/api/v1/models/{model}/endpoints', timeout=30).json()['data']
    print(f"{'route spec':58s} {'quant':8s} {'$in/M':>7s} {'$out/M':>7s} {'cache':>6s} logprobs")
    for e in data['endpoints']:
        price = e['pricing']
        logprobs = 'top_logprobs' in (e.get('supported_parameters') or [])
        cache_read = price.get('input_cache_read')
        print(f"openrouter:{model}@{e['tag']:<{max(0, 47 - len(model))}s} {str(e.get('quantization')):8s} "
              f"{float(price['prompt']) * 1e6:7.3f} {float(price['completion']) * 1e6:7.3f} "
              f"{float(cache_read) * 1e6 if cache_read else 0:6.3f} {'yes' if logprobs else 'no'}")


if __name__ == '__main__':
    main()
