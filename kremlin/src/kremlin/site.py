"""HTML pages for kremlin-hub: collated findings, per-run pages, JSON export and an embeddable script.

Designed to sit inside an existing site: pages link the host site's /styles.css (same origin when the hub is
reverse-proxied under a path such as /hub/) and carry a small inline fallback so they also look right alone.
"""
import html
import json
import re

from .findings import KINDS

FALLBACK_CSS = """
body{background:#000;color:#fff;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;line-height:1.6;margin:0}
a{color:#66b3ff;text-decoration:none}a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #333;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#1a1a1a}tr:nth-child(even){background:#0a0a0a}code,pre{background:#1a1a1a;border-radius:3px;font-family:'Courier New',monospace}
code{padding:2px 6px}pre{padding:12px;overflow-x:auto}h1{border-bottom:2px solid #333;padding-bottom:.3em}
.home-button{position:fixed;top:20px;right:20px;padding:10px 20px;background:#66b3ff;color:#000;border-radius:5px;font-weight:bold}
"""
PAGE_CSS = """
.kremlin{max-width:1100px;margin:0 auto;padding:20px}
.kremlin .lede{color:#ccc}
.kremlin .muted{color:#999;font-size:.9em}
.kremlin .badge{display:inline-block;border:1px solid #66b3ff;border-radius:4px;padding:2px 8px;margin:2px 6px 2px 0;font-size:.85em}
.kremlin .flag-notable td:first-child{border-left:4px solid #66b3ff}
.kremlin .tag{font-size:.8em;border:1px solid #555;border-radius:3px;padding:0 5px;color:#ddd;white-space:nowrap}
.kremlin .tag.notable{border-color:#66b3ff;color:#66b3ff}
.kremlin input[type=search]{background:#111;color:#fff;border:1px solid #333;border-radius:4px;padding:8px 10px;width:100%;max-width:420px;font-size:1em}
.kremlin td.num{text-align:right;font-variant-numeric:tabular-nums}
.kremlin .report table{font-size:.92em}
"""


def esc(value):
    return html.escape('' if value is None else str(value))


def page(title, body, base):
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)}</title><style>{FALLBACK_CSS}</style><link rel="stylesheet" href="/styles.css"><style>{PAGE_CSS}</style>
</head><body><a href="/" class="home-button">Home</a><main class="kremlin">{body}</main></body></html>""".encode()


def tag(flag):
    label = {'notable': 'notable', 'inconclusive': 'inconclusive'}.get(flag)
    return f'<span class="tag {esc(flag)}">{label}</span>' if label else ''


def wheel(hub):
    if not hub.downloads or not hub.downloads.is_dir():
        return None
    wheels = sorted(hub.downloads.glob('kremlin-*.whl'))
    return wheels[-1].name if wheels else None


def install_line(hub, base):
    name = wheel(hub)
    if not name:
        return "pipx install 'kremlin[witness]'"
    url = f"{hub.public_url or base}/download/{name}"
    return esc(f"pipx install 'kremlin[witness] @ {url}'")


def index_page(hub, base):
    index = hub.index()
    findings = [dict(f, run_id=e['id']) for e in index for f in e.get('findings', [])]
    by_kind = {}
    for finding in findings:
        by_kind.setdefault(finding['kind'], []).append(finding)
    sections = []
    for kind, title in KINDS.items():
        rows = by_kind.get(kind)
        if not rows:
            continue
        rows.sort(key=lambda f: (f['flag'] != 'notable', f['model'], ' '.join(f['subjects'])))
        body = ''.join(
            f'<tr class="flag-{esc(f["flag"])}"><td>{esc(f["model"])}</td><td>{esc(" vs ".join(f["subjects"]))}</td>'
            f'<td>{esc(f["metric"])}</td><td>{esc(f["verdict"])} {tag(f["flag"])}</td>'
            f'<td><a href="{base}/runs/{esc(f["run_id"])}/">{esc(f["run_id"][:8])}</a></td></tr>' for f in rows)
        sections.append(f'<section class="kind" data-kind="{esc(kind)}"><h2>{esc(title)}</h2><table>'
                        f'<thead><tr><th>Model</th><th>Providers</th><th>Measurement</th><th>Finding</th><th>Run</th></tr>'
                        f'</thead><tbody>{body}</tbody></table></section>')
    runs = ''.join(
        f'<tr><td><a href="{base}/runs/{esc(e["id"])}/">{esc(e["id"][:8])}</a></td><td>{esc(e["command"])}</td>'
        f'<td>{esc(", ".join(e["routes"]))}</td><td class="num">{esc(e["records"])}</td>'
        f'<td>{esc((e.get("run_started") or "")[:10])}</td></tr>' for e in reversed(index[-50:]))
    notable = sum(f['flag'] == 'notable' for f in findings)
    body = f"""<h1>LLM inference forensics</h1>
<p class="lede">What inference providers actually serve, measured from outside. Every result below comes from a run in
which <b>each API call was witnessed</b> by this site's TLS relay (it sees the responses and request bodies, never the
caller's API key) and was then <b>re-audited here</b>: receipts verified and the conclusions recomputed from the
witnessed data.</p>
<p><span class="badge">{len(index)} attested runs</span><span class="badge">{len(findings)} findings</span>
<span class="badge">{notable} notable</span> <a href="{base}/findings.json">findings.json</a> ·
<a href="{base}/runs">runs (JSON)</a> · <a href="#contribute">contribute a run</a></p>
<p><input type="search" id="filter" placeholder="Filter by model, provider or finding…" aria-label="Filter findings"></p>
{''.join(sections) or '<p class="muted">No accepted runs yet.</p>'}
<h2>Latest runs</h2><table><thead><tr><th>Run</th><th>Investigation</th><th>Routes</th><th>Calls</th><th>Date</th></tr>
</thead><tbody>{runs}</tbody></table>
<h2 id="contribute">Contribute a run</h2>
<p>Install the client, run an investigation through this site's witness with your own provider key, and submit the
evidence. Only runs where every call was witnessed and the results reproduce are published. <b>Everything you send
through the witness or submit here is stored and may be used and published for any purpose</b> (<a href="#data-use">data
use</a>); your API key is never seen.</p>
<pre><code>{install_line(hub, base)}
kremlin routes qwen/qwen3.5-397b-a17b
kremlin fingerprint 'openrouter:qwen/qwen3.5-397b-a17b@parasail/fp8' 'openrouter:qwen/qwen3.5-397b-a17b@venice' \\
    --out run1 --max-usd 0.10 --witness {esc(hub.witness_spec or 'TUNNEL,CONTROL_URL,PUBKEY')}
kremlin submit run1 --hub {esc(hub.public_url or 'https://this-site' + base)}</code></pre>
<h2 id="data-use">Data use</h2>
<p>{esc(hub.terms)} <a href="{base}/terms">Full terms</a>.</p>
<p class="muted">Witness keys trusted here: {' '.join('<code>' + esc(k[:16]) + '…</code>' for k in hub.keys)}</p>
<script>
document.getElementById('filter').addEventListener('input', function (event) {{
  var needle = event.target.value.toLowerCase();
  document.querySelectorAll('section.kind').forEach(function (section) {{
    var shown = 0;
    section.querySelectorAll('tbody tr').forEach(function (row) {{
      var hit = row.textContent.toLowerCase().indexOf(needle) >= 0;
      row.style.display = hit ? '' : 'none';
      shown += hit ? 1 : 0;
    }});
    section.style.display = shown ? '' : 'none';
  }});
}});
</script>"""
    return page('LLM inference forensics', body, base)


def markdown(text):
    """Small markdown renderer for kremlin reports: headings, tables, lists, paragraphs, code, emphasis."""
    def inline(value):
        value = esc(value)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
        value = re.sub(r'(?<![\w/])_([^_]+)_(?![\w/])', r'<i>\1</i>', value)
        return value
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                j += 1
            out.append('<pre><code>' + esc('\n'.join(lines[i + 1:j])) + '</code></pre>')
            i = j + 1
            continue
        heading = re.match(r'^(#{1,4}) (.*)', line)
        if heading:
            level = min(len(heading.group(1)) + 1, 5)  # page title is the only h1
            out.append(f'<h{level}>{inline(heading.group(2))}</h{level}>')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(set(c) <= set('-: ') for c in r)]
            out.append('<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in head) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body)
                       + '</tbody></table>')
            continue
        elif re.match(r'^\s*[-*] ', line):
            items = []
            while i < len(lines) and re.match(r'^\s*[-*] ', lines[i]):
                items.append(re.sub(r'^\s*[-*] ', '', lines[i]))
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + '</ul>')
            continue
        elif line.strip() == '---':
            out.append('<hr>')
        elif line.strip():
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r'^(#|\||\s*[-*] |```|---)', lines[i + 1]):
                i += 1
                para.append(lines[i])
            out.append('<p>' + inline(' '.join(para)) + '</p>')
        i += 1
    return '\n'.join(out)


def run_page(hub, run_id, base):
    entry = next((e for e in hub.index() if e['id'] == run_id), None)
    if not entry:
        return None
    folder = hub.dir / 'runs' / run_id
    report = (folder / 'report.md').read_text() if (folder / 'report.md').exists() else ''
    audit = json.loads((folder / 'audit.json').read_text()) if (folder / 'audit.json').exists() else {}
    badges = [f"witnessed {esc(audit.get('witnessed'))}/{esc(audit.get('records'))} calls",
              f"{esc(audit.get('requests_disclosed'))} requests disclosed",
              'audit: results reproduced' if audit.get('results_reproduced') else 'audit: NOT reproduced',
              f"witness key {esc(entry.get('witness_key_id'))}"]
    if entry.get('submitters'):
        badges.append('submitter ' + esc(', '.join(entry['submitters'])))
    findings = ''.join(f'<li>{esc(f["metric"])} — <b>{esc(f["verdict"])}</b> {tag(f["flag"])}</li>'
                       for f in entry.get('findings', []))
    body = f"""<p><a href="{base}/">← All findings</a></p>
<h1>{esc(entry['command'])}: {esc(', '.join(entry['routes']))}</h1>
<p>{''.join(f'<span class="badge">{b}</span>' for b in badges)}</p>
<p class="muted">Run started {esc(entry.get('run_started'))}; accepted {esc(entry.get('accepted_at'))}.</p>
{('<h2>Findings</h2><ul>' + findings + '</ul>') if findings else ''}
<div class="report">{markdown(report)}</div>
<h2>Verify it yourself</h2>
<pre><code>curl -O {esc(hub.public_url or base)}/runs/{esc(run_id)}/bundle.tar.gz
mkdir run &amp;&amp; tar -xzf bundle.tar.gz -C run
kremlin audit run --pubkey WITNESS_PUBLIC_KEY --require-witness</code></pre>
<p><a href="{base}/runs/{esc(run_id)}/bundle.tar.gz">bundle.tar.gz</a> ·
<a href="{base}/runs/{esc(run_id)}/results.json">results.json</a> ·
<a href="{base}/runs/{esc(run_id)}/audit.json">audit.json</a> ·
<a href="{base}/runs/{esc(run_id)}/report.md">report.md</a></p>"""
    return page(f"{entry['command']} — LLM inference forensics", body, base)


def findings_json(hub, query):
    rows = [dict(f, run_id=e['id'], run_started=e.get('run_started')) for e in hub.index() for f in e.get('findings', [])]
    for field in ('kind', 'model', 'flag'):
        if query.get(field):
            rows = [r for r in rows if query[field][0].lower() in str(r[field]).lower()]
    if query.get('provider'):
        needle = query['provider'][0].lower()
        rows = [r for r in rows if any(needle in s.lower() for s in r['subjects'])]
    return json.dumps(dict(count=len(rows), kinds=KINDS, findings=rows)).encode()


EMBED_JS = """(function () {
  // Usage: <div data-kremlin-findings data-hub="https://example.com/hub" data-kind="same-backend"></div>
  //        <script src="https://example.com/hub/embed.js" defer></script>
  function render(el) {
    var hub = el.getAttribute('data-hub') || '';
    var params = ['kind', 'model', 'provider', 'flag'].filter(function (k) { return el.getAttribute('data-' + k); })
      .map(function (k) { return k + '=' + encodeURIComponent(el.getAttribute('data-' + k)); }).join('&');
    fetch(hub + '/findings.json' + (params ? '?' + params : '')).then(function (r) { return r.json(); }).then(function (d) {
      var t = document.createElement('table');
      t.innerHTML = '<thead><tr><th>Model</th><th>Providers</th><th>Measurement</th><th>Finding</th><th>Run</th></tr></thead>';
      var b = document.createElement('tbody');
      d.findings.forEach(function (f) {
        var tr = document.createElement('tr');
        [f.model, f.subjects.join(' vs '), f.metric, f.verdict + (f.flag === 'notable' ? ' (notable)' : '')]
          .forEach(function (v) { var td = document.createElement('td'); td.textContent = v; tr.appendChild(td); });
        var td = document.createElement('td'), a = document.createElement('a');
        a.href = hub + '/runs/' + f.run_id + '/'; a.textContent = f.run_id.slice(0, 8); td.appendChild(a); tr.appendChild(td);
        b.appendChild(tr);
      });
      t.appendChild(b); el.innerHTML = ''; el.appendChild(t);
    }).catch(function () { el.textContent = 'Findings unavailable.'; });
  }
  document.querySelectorAll('[data-kremlin-findings]').forEach(render);
})();
""".encode()


HUB_TERMS = ('All data sent through this site\'s witness and all submitted evidence (requests, responses, timings, '
             'receipts and run metadata; never API keys) is stored indefinitely. The site operator may use, analyse, '
             'publish, share and relicense it for any purpose, without further notice. Do not send personal, '
             'confidential or third-party data you do not have the right to share.')


def terms_page(hub, base):
    body = f"""<p><a href="{base}/">← All findings</a></p><h1>Data terms</h1>
<p>{esc(hub.terms)}</p>
<h2>What is stored</h2>
<ul><li>Every witnessed TLS session's receipt: the server's response, its timing, and (with request disclosure) the
request line's length, non-secret headers and the request body.</li>
<li>Every submitted evidence bundle: all requests and responses of the run, its results and report.</li>
<li>Metadata: time, destination host, access-token identifier (a hash), and the client's IP address in logs.</li></ul>
<h2>What is never stored</h2><ul><li>API keys or other credentials: they are encrypted under keys the witness never
receives.</li></ul>
<h2>How it may be used</h2><p>Any purpose, including research, publication of findings and raw data, and sharing
with others.</p>
<p class="muted">By using the witness or submitting a run you agree to these terms.</p>"""
    return page('Data terms — LLM inference forensics', body, base)
