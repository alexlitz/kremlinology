"""Check `kremlin proxy` from each language's usual HTTP stack, in both modes (base URL and HTTPS_PROXY).

Needs a running proxy (default 127.0.0.1:8787) started with --cors http://127.0.0.1:8899 for the browser case, and
OPENROUTER_API_KEY. Each case calls a free OpenRouter endpoint (the model's endpoint list) and must come back 200
with X-Kremlin-Witnessed: request+response. Runtimes that are not installed run in Docker (--network host).

    kremlin proxy --out runs/langs --cors http://127.0.0.1:8899 &
    python tools/lang_matrix.py [--only go,php]
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

PATH = '/api/v1/models/liquid/lfm-3-1.7b:free/endpoints'
DIRECT = 'https://openrouter.ai' + PATH

SNIPPETS = {
    'python-urllib': ('main.py', '''
        import os, urllib.request
        req = urllib.request.Request(os.environ['URL'], headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        try:
            r = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            r = e
        print(r.status, r.headers.get('x-kremlin-witnessed'))
        '''),
    'python-httpx': ('main.py', '''
        import os, httpx
        r = httpx.get(os.environ['URL'], headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
        print(r.status_code, r.headers.get('x-kremlin-witnessed'))
        '''),
    'python-openai': ('main.py', '''
        import os
        from openai import OpenAI
        base = os.environ['URL'].split('/models/')[0]
        c = OpenAI(base_url=base, api_key=os.environ['OPENROUTER_API_KEY'], max_retries=0)
        r = c.models.with_raw_response.list()
        print(r.http_response.status_code, r.headers.get('x-kremlin-witnessed'))
        '''),
    'node-fetch': ('main.mjs', '''
        const r = await fetch(process.env.URL, {headers: {Authorization: 'Bearer ' + process.env.OPENROUTER_API_KEY}});
        console.log(r.status, r.headers.get('x-kremlin-witnessed'));
        '''),
    'node-openai': ('main.mjs', '''
        import OpenAI from 'openai';
        const c = new OpenAI({baseURL: process.env.URL.split('/models/')[0], apiKey: process.env.OPENROUTER_API_KEY, maxRetries: 0});
        const {response} = await c.models.list().withResponse();
        console.log(response.status, response.headers.get('x-kremlin-witnessed'));
        '''),
    'bun-fetch': ('main.ts', '''
        const r = await fetch(process.env.URL!, {headers: {Authorization: 'Bearer ' + process.env.OPENROUTER_API_KEY}});
        console.log(r.status, r.headers.get('x-kremlin-witnessed'));
        '''),
    'deno-fetch': ('main.ts', '''
        const r = await fetch(Deno.env.get('URL')!, {headers: {Authorization: 'Bearer ' + Deno.env.get('OPENROUTER_API_KEY')}});
        console.log(r.status, r.headers.get('x-kremlin-witnessed'));
        '''),
    'go-net/http': ('main.go', '''
        package main
        import ("fmt"; "net/http"; "os")
        func main() {
            req, _ := http.NewRequest("GET", os.Getenv("URL"), nil)
            req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENROUTER_API_KEY"))
            r, err := http.DefaultClient.Do(req)
            if err != nil { fmt.Println("ERR", err); return }
            fmt.Println(r.StatusCode, r.Header.Get("X-Kremlin-Witnessed"))
        }
        '''),
    'java-httpclient': ('Main.java', '''
        import java.net.http.*; import java.net.URI;
        public class Main { public static void main(String[] a) throws Exception {
            var r = HttpClient.newHttpClient().send(HttpRequest.newBuilder(URI.create(System.getenv("URL")))
                .header("Authorization", "Bearer " + System.getenv("OPENROUTER_API_KEY")).build(),
                HttpResponse.BodyHandlers.ofString());
            System.out.println(r.statusCode() + " " + r.headers().firstValue("x-kremlin-witnessed").orElse("null"));
        } }
        '''),
    'csharp-httpclient': ('app.cs', '''
        using var c = new HttpClient();
        var req = new HttpRequestMessage(HttpMethod.Get, Environment.GetEnvironmentVariable("URL"));
        req.Headers.Add("Authorization", "Bearer " + Environment.GetEnvironmentVariable("OPENROUTER_API_KEY"));
        var r = await c.SendAsync(req);
        Console.WriteLine($"{(int)r.StatusCode} {(r.Headers.TryGetValues("X-Kremlin-Witnessed", out var v) ? string.Join(",", v) : "null")}");
        '''),
    'rust-reqwest': ('main.rs', '''
        fn main() {
            let url = std::env::var("URL").unwrap();
            let key = std::env::var("OPENROUTER_API_KEY").unwrap();
            let r = reqwest::blocking::Client::new().get(url).header("Authorization", format!("Bearer {key}")).send();
            match r {
                Ok(r) => println!("{} {}", r.status().as_u16(),
                                  r.headers().get("x-kremlin-witnessed").map(|v| v.to_str().unwrap()).unwrap_or("null")),
                Err(e) => println!("ERR {e:?}"),
            }
        }
        '''),
    'php-curl': ('main.php', '''
        <?php
        $h = curl_init(getenv('URL'));
        $w = 'null';
        curl_setopt_array($h, [CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Authorization: Bearer ' . getenv('OPENROUTER_API_KEY')],
            CURLOPT_HEADERFUNCTION => function ($h, $line) use (&$w) {
                if (stripos($line, 'x-kremlin-witnessed:') === 0) { $w = trim(substr($line, 20)); }
                return strlen($line); }]);
        if (curl_exec($h) === false) { echo 'ERR ' . curl_error($h) . "\\n"; exit; }
        echo curl_getinfo($h, CURLINFO_RESPONSE_CODE) . " $w\\n";
        '''),
    'ruby-net/http': ('main.rb', '''
        require 'net/http'
        u = URI(ENV['URL'])
        h = Net::HTTP.new(u.host, u.port, :ENV)
        h.use_ssl = u.scheme == 'https'
        r = h.get(u.request_uri, {'Authorization' => 'Bearer ' + ENV['OPENROUTER_API_KEY']})
        puts "#{r.code} #{r['x-kremlin-witnessed'] || 'null'}"
        '''),
    'powershell': ('main.ps1', '''
        $r = Invoke-WebRequest -Uri $env:URL -Headers @{Authorization = "Bearer $env:OPENROUTER_API_KEY"} -SkipHttpErrorCheck
        "$($r.StatusCode) $(if ($r.Headers['X-Kremlin-Witnessed']) { $r.Headers['X-Kremlin-Witnessed'] } else { 'null' })"
        '''),
    'curl': ('main.sh', '''
        curl -s -o /dev/null -D - "$URL" -H "Authorization: Bearer $OPENROUTER_API_KEY" |
          awk 'tolower($1)=="x-kremlin-witnessed:"{w=$2} /^HTTP/{s=$2} END{gsub("\\r","",w); print s, (w?w:"null")}'
        '''),
}

CARGO_TOML = '''[package]
name = "probe"
version = "0.1.0"
edition = "2021"
[dependencies]
reqwest = { version = "0.12", features = ["blocking"] }
'''


def command(case, workdir, file):
    """(argv, needs_docker_image or None)."""
    have = shutil.which
    return {
        'python-urllib': ([sys.executable, file], None),
        'python-httpx': ([sys.executable, file], None),
        'python-openai': ([sys.executable, file], None),
        'node-fetch': (['node', file], None),
        'node-openai': (['node', file], None),
        'bun-fetch': (['bun', 'run', file], None if have('bun') else 'oven/bun:1'),
        'deno-fetch': (['deno', 'run', '--allow-env', '--allow-net', file],
                       None if have('deno') and 'pyenv' not in os.path.realpath(have('deno')) else 'denoland/deno:latest'),
        'go-net/http': (['go', 'run', file], None if have('go') else 'golang:1'),
        'java-httpclient': (['java', file], None if have('java') else 'eclipse-temurin:21'),
        'csharp-httpclient': (['dotnet', 'run', file], None if have('dotnet') else 'mcr.microsoft.com/dotnet/sdk:10.0'),
        'rust-reqwest': (['cargo', 'run', '-q', '--manifest-path', str(workdir / 'Cargo.toml')],
                         None if have('cargo') else 'rust:1'),
        'php-curl': (['php', file], None if have('php') else 'php:8.4-cli'),
        'ruby-net/http': (['ruby', file], None if have('ruby') else 'ruby:3'),
        'curl': (['sh', file], None),
        'powershell': (['pwsh', '-NoProfile', '-File', file],
                       None if have('pwsh') else 'mcr.microsoft.com/powershell:latest'),
    }[case]


FORCE_DOCKER = {
    'bun-fetch': 'oven/bun:1', 'deno-fetch': 'denoland/deno:latest', 'go-net/http': 'golang:1',
    'java-httpclient': 'eclipse-temurin:21', 'csharp-httpclient': 'mcr.microsoft.com/dotnet/sdk:10.0',
    'rust-reqwest': 'rust:1', 'php-curl': 'php:8.4-cli', 'ruby-net/http': 'ruby:3',
    'powershell': 'mcr.microsoft.com/powershell:latest'}
forced = set()


def run_case(case, mode, base, env_proxy, ca_dir, workdir):
    name, source = SNIPPETS[case]
    case_dir = workdir / case.replace('/', '_')
    case_dir.mkdir(exist_ok=True)
    if case == 'rust-reqwest':
        (case_dir / 'src').mkdir(exist_ok=True)
        (case_dir / 'Cargo.toml').write_text(CARGO_TOML)
        (case_dir / 'src' / 'main.rs').write_text(textwrap.dedent(source).lstrip())
    elif case == 'node-openai' and not (case_dir / 'node_modules' / 'openai').exists():
        subprocess.run(['npm', 'install', '--silent', '--prefix', str(case_dir), 'openai'], check=True,
                       capture_output=True)
    path = case_dir / name
    path.write_text(textwrap.dedent(source).lstrip())
    argv, image = command(case, case_dir, str(path))
    if case in forced:
        image = FORCE_DOCKER[case]
    env = dict(URL=f'{base}/openrouter.ai{PATH}' if mode == 'base-url' else DIRECT,
               OPENROUTER_API_KEY=os.environ['OPENROUTER_API_KEY'])
    if mode == 'https-proxy':
        env.update(env_proxy)
    if case == 'go-net/http':
        env['CGO_ENABLED'] = '0'
    if image:
        flags = ['docker', 'run', '--rm', '--network', 'host', '-v', f'{case_dir}:{case_dir}', '-w', str(case_dir),
                 '-v', f'{ca_dir}:{ca_dir}:ro']
        for key in env:
            flags += ['-e', key]
        if case == 'csharp-httpclient':
            flags += ['-e', 'DOTNET_CLI_TELEMETRY_OPTOUT=1', '-e', 'DOTNET_NOLOGO=1']
        argv = flags + [image] + argv
    full_env = dict(os.environ, **env)
    if mode == 'base-url':  # make sure no proxy settings leak in
        for key in list(full_env):
            if key.lower() in ('https_proxy', 'http_proxy', 'all_proxy'):
                del full_env[key]
    try:
        out = subprocess.run(argv, env=full_env, capture_output=True, text=True, timeout=900, cwd=case_dir)
    except subprocess.TimeoutExpired:
        return case, mode, 'timeout', image
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    result = lines[-1] if lines else (out.stderr.strip().splitlines() or ['no output'])[-1][:160]
    return case, mode, result, image


def browser_case(base):
    """A page on http://127.0.0.1:8899 calls the proxy with fetch (base-URL mode; needs --cors)."""
    import http.server
    from playwright.sync_api import sync_playwright
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 8899), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto('http://127.0.0.1:8899/')
            result = page.evaluate('''async ([url, key]) => { try {
                const r = await fetch(url, {headers: {Authorization: 'Bearer ' + key}});
                return r.status + ' ' + r.headers.get('x-kremlin-witnessed'); } catch (e) { return 'ERR ' + e; } }''',
                                   [f'{base}/openrouter.ai{PATH}', os.environ['OPENROUTER_API_KEY']])
            browser.close()
            return 'browser-fetch (chromium)', 'base-url', result, None
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', default='127.0.0.1:8787')
    parser.add_argument('--only', help='comma-separated cases')
    parser.add_argument('--modes', default='base-url,https-proxy')
    parser.add_argument('--docker', default='', help='comma-separated cases to run in Docker even if installed')
    args = parser.parse_args()
    forced.update(c for c in args.docker.split(',') if c)
    from kremlin import session
    ca = session.LocalCA()
    env_proxy = {name: value for name, value, _ in session.env_vars(args.listen, ca)}
    base = f'http://{args.listen}'
    cases = args.only.split(',') if args.only else list(SNIPPETS) + ['browser']
    rows = []
    with tempfile.TemporaryDirectory(prefix='kremlin-langs-', ignore_cleanup_errors=True) as tmp:
        for case in cases:
            for mode in args.modes.split(','):
                if case == 'browser':
                    if mode == 'base-url':
                        rows.append(browser_case(base))
                    continue
                row = run_case(case, mode, base, env_proxy, ca.dir, Path(tmp))
                rows.append(row)
                print(f'{row[0]:22s} {row[1]:12s} {row[2]}' + (f'   [docker {row[3]}]' if row[3] else ''), flush=True)
        if any(r[3] for r in rows):  # containers ran as root: remove what they wrote
            subprocess.run(['docker', 'run', '--rm', '-v', f'{tmp}:/w', 'alpine', 'sh', '-c', 'rm -rf /w/*'],
                           capture_output=True)
    if 'browser' in cases:
        print(f'{rows[-1][0]:22s} {rows[-1][1]:12s} {rows[-1][2]}')
    ok = sum(1 for r in rows if r[2] == '200 request+response')
    print(f'\n{ok}/{len(rows)} witnessed with request+response')
    return 0 if ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
