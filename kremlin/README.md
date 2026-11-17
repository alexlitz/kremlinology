# kremlin

Black-box forensics for LLM inference APIs. Point it at two or more *routes* (the same model
through different providers, or the same provider with different keys) and it measures what the
API gives away: whether two routes share a backend, how their prompt caches behave and whether
they leak across tenants, how far their outputs differ from a reference, whether they honour
output caps and what they bill, and how decode speed depends on content.

Every run writes an **evidence directory** (every request and response, keys redacted, SHA-256
manifest). Optionally every request goes through a **witness relay** that attests the responses
(and their timing) without ever seeing the caller's API key.

```
pip install -e '.[witness]'      # witness extra: cryptography, certifi, brotli, zstandard, tlslite-ng
export OPENROUTER_API_KEY=...
kremlin routes qwen/qwen3.5-397b-a17b                      # list endpoints -> route specs (free)
kremlin fingerprint 'openrouter:qwen/qwen3.5-397b-a17b@parasail/fp8' \
                    'openrouter:qwen/qwen3.5-397b-a17b@venice' --plan
kremlin fingerprint 'openrouter:qwen/qwen3.5-397b-a17b@parasail/fp8' \
                    'openrouter:qwen/qwen3.5-397b-a17b@venice' --out runs/fp1 --max-usd 0.10
```

Python 3.10+. Walkthrough: [docs/walkthrough.md](docs/walkthrough.md) (detecting a reseller).

## Routes

| spec | meaning |
|---|---|
| `openrouter:MODEL@PROVIDER` | OpenRouter pinned to one endpoint, no fallbacks |
| `openrouter:MODEL` | OpenRouter default routing |
| `openai:https://host/v1\|MODEL;key=ENV` | any OpenAI-compatible API (provider's own API, local vLLM, ...) |
| `anthropic:MODEL` | Anthropic Messages API (`ANTHROPIC_API_KEY`) |

Options after `;`: `key=ENV_VAR` (which key), `label=NAME` (report name), `price=IN/OUT` (USD per
million tokens, used when the API reports no cost). Two routes that differ only in `key=` test
**cross-account** behaviour.

## Investigations

All paid commands take `--out DIR`, `--max-usd` (default 0.25, hard stop), `--plan` (print the
request count, send nothing) and `--witness`.

| command | question | cost (typical) |
|---|---|---|
| `fingerprint A B [C...]` | Do routes share a backend? Exact first-token logprob identity vs each route's own repeatability. A reseller matches its supplier as often as the supplier matches itself; independent deployments never match to 0.002 nats. | 60 prompts x 2 x routes, 1 token each: cents |
| `tv REF A [B...]` | How different are outputs from a reference? Dataset-average first-token TV with distribution-free 95% CIs; the reference against itself is the noise floor. `--prompt-file` for your dataset. | 100 x 2 x routes, 1 token |
| `cache-share A B` | Does B read a prompt A just cached? (resale: same model via two providers; cross-account: same route, two keys). Controls A->A, B->B. | 24 requests x ~6k tokens |
| `cache-blocks R` | Cache granularity (block size) and minimum cached length. | 12 requests |
| `cache-ttl R [R...] --delays 30,120,300,600` | How long entries survive; many routes on one timeline (a retention sweep); `--ttl-hint 1h` for Anthropic. | 8 requests per route, runs for max delay |
| `cache-prefix R` | Resume semantics: any-prefix (block-paged) vs breakpoint-only caches; follow-up turns. | 16 requests |
| `billing A [B...] --cap 64` | Is the output cap honoured, and is reasoning billed outside it? Checks OpenRouter's own billing record too. | 6 requests per route |
| `timing A [B...]` | Speculative-decoding signature (copy/count vs story speed, with a story-vs-story control) and, with `--active-params`, a memory-bandwidth lower bound. | 24 streamed requests per route |
| `params A [B...]` | Are logprobs (count, consistency with the chosen token), `n`, `seed`, temperature 0, `stop` and `max_tokens` honoured? Prompt-token parity between routes serving the same model (template/tokenizer differences). | 12 requests per route |
| `sampler A [B...]` | Is the sampler what the route claims? Picks the most uncertain of 8 prompts, then checks `top_k=1/2` (definite violations vs near-ties), `top_p` set midway between p1 and p1 + p2 so only the top two may appear (a token is in the nucleus iff the mass ranked above it is below `top_p`), T=1 frequencies vs reported logprobs (chi-square), fitted temperature at T=0.5. | ~310 one-token requests per route without `n` |
| `later REF A [B...]` | Output differences at later positions: the reference generates, every route is forced onto that history by assistant prefill; faithfulness and prompt-token parity checks exclude routes that do not continue prefills. | prompts x positions x routes, 1 token |
| `reconstruct A B [...]` | Compare routes that return **no logprobs**: first-token laws reconstructed from T=2, top-k=10 samples (law ∝ count^T), bootstrap CIs, self-check against reported logprobs where available. | ~200 one-token samples per prompt per route (fewer with n) |
| `request R PATH` | One request to any endpoint: `--json`, or multipart `-F file=@audio.wav -F model=...` (embeddings, images, audio, responses...). | one request |

Requests run concurrently (`--concurrency`, default 8; `timing` defaults to 1 because parallel load can change the
speed being measured). Every call has a deterministic key, so an interrupted or budget-stopped run continues
with `kremlin resume DIR [--max-usd N]` (completed calls are reused from the evidence, nothing is re-sent) and
`audit` replays it exactly. `kremlin prompts codealpaca|wikitext|hf:REPO:SPLIT:FIELD --n 200 --out p.jsonl`
builds a dataset prompt file (`pip install datasets`) with source indices.

Reports are written to `report.md`, numbers to `results.json`. Interpretation notes are at the
bottom of each report. What these measurements cannot tell you: a difference does not identify its
cause (precision, kernels, templates); declarations (quantization labels) are not verified; a
bitwise match shows shared numerics, not a contract.

## Witnessed requests from any language (your own code, your usual SDK)

`kremlin proxy` runs a local proxy that sends every API call through the witness relay
(default: the public relay on alexlitzenberger.com; `--witness` or `$KREMLIN_WITNESS` for another)
and records an auditable session. Two ways to point code at it:

**1. Base URL** (what most SDKs already expose):

```
kremlin proxy --out runs/s1          # Ctrl-C (or kill) finishes the session
export OPENAI_BASE_URL=http://127.0.0.1:8787/openrouter.ai/api/v1      # OpenAI SDKs (Python, JS, ...), LangChain, LiteLLM
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/api.anthropic.com      # Anthropic SDKs
curl http://127.0.0.1:8787/api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" ...
```

Any `https://HOST/PATH` becomes `http://127.0.0.1:8787/HOST/PATH`. In JS: `new OpenAI({ baseURL: ... })`.

**2. `HTTPS_PROXY`, no code change at all.** The proxy creates a local CA (`~/.config/kremlin/proxy-ca`,
never leaves the machine) and decrypts only the hosts you list, the way mitmproxy does; other traffic is
tunnelled through untouched and unrecorded.

```
kremlin proxy --out runs/s1 --hosts 'openrouter.ai,api.openai.com,api.anthropic.com'
eval "$(kremlin env)"                 # HTTPS_PROXY plus each stack's CA setting
# PowerShell: kremlin env --shell powershell | Out-String | iex    (also --shell fish / cmd)
python my_script.py; node app.js; go run .; ruby x.rb; curl https://openrouter.ai/...
```

`kremlin env` covers each stack's own setting:

| language / runtime | client checked | base URL | HTTPS_PROXY (setting `kremlin env` exports) |
|---|---|---|---|
| Python | urllib, httpx, openai SDK (requests, anthropic use the same) | ✓ | ✓ `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` |
| Node.js 24+ / TypeScript | fetch, openai SDK | ✓ | ✓ `NODE_USE_ENV_PROXY=1`, `NODE_EXTRA_CA_CERTS` |
| Bun | fetch | ✓ | ✓ `NODE_EXTRA_CA_CERTS` |
| Deno | fetch | ✓ | ✓ `DENO_CERT` |
| Browser JS | fetch from a page (start the proxy with `--cors ORIGIN`) | ✓ | — (browsers don't read these settings) |
| Go | net/http | ✓ | ✓ `SSL_CERT_FILE` |
| Java / Kotlin / Scala | java.net.http.HttpClient | ✓ | ✓ `JAVA_TOOL_OPTIONS` (proxy properties + a truststore built with keytool) |
| C# / .NET 10 | HttpClient | ✓ | ✓ `SSL_CERT_FILE` (Linux; on macOS/Windows add `ca.pem` to the system store) |
| Rust | reqwest 0.12 (default features) | ✓ | ✓ `SSL_CERT_FILE` |
| PHP 8.4 | curl extension | ✓ | ✓ `PHP_INI_SCAN_DIR` (an ini setting `curl.cainfo`) |
| Ruby | Net::HTTP | ✓ | ✓ `http_proxy`, `SSL_CERT_FILE` |
| shell | curl | ✓ | ✓ `CURL_CA_BUNDLE` |
| PowerShell | Invoke-WebRequest | ✓ | ✓ `kremlin env --shell powershell` |

Each cell is a real call to OpenRouter through the public relay, witnessed with its request disclosed
(`python tools/lang_matrix.py`; runtimes that aren't installed run in Docker). SDKs in these languages
(openai-go, openai-java, the OpenAI .NET library, async-openai, openai-php, LangChain, LiteLLM, the Vercel AI SDK)
sit on these clients and take a base URL. In HTTPS_PROXY mode only the usual LLM API hosts are decrypted
(`--hosts` to change, `'*'` for all); everything else is tunnelled through untouched.

Programs that read only the operating system's trust store (.NET on macOS and Windows, browsers, many desktop
apps) need the CA there: `kremlin trust` prints the command for your OS (macOS login keychain, Windows user store,
Linux NSS for Chrome; no admin rights needed) and `kremlin trust --install` runs it (`--remove` undoes it). The
matrix above ran on Linux; macOS and Windows are untested.

**In Python, no proxy process:** pass a witnessed httpx client to the SDK.

```python
import kremlin
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", http_client=kremlin.http_client("runs/s2"))
# AsyncOpenAI(http_client=kremlin.async_http_client(...)), Anthropic(http_client=...) work the same way
```

Then `kremlin audit runs/s1 --pubkey PUBKEY --require-witness` checks every receipt, and
`kremlin submit runs/s1` publishes the session on the hub. Notes: event streams (`stream: true`) reach your code live, as the provider sends them; the receipt is checked
when the stream ends, so streamed responses carry `X-Kremlin-Witnessed: streaming` (the verdict is in the evidence
and the audit), other responses are returned once verified; each call opens a fresh
TLS session through the relay and then fetches its receipt, so calls are slower than over a kept-alive connection; requests need one auth header
(`Authorization`, `x-api-key`, `x-goog-api-key` or `api-key`) to be disclosed, otherwise only the response is
attested. `kremlin proxy --finish DIR` finalizes a session whose proxy was killed.

## Witness: attested results without sharing your key

`kremlin-witness` is a small relay (pure Python, no MPC) that lets a third party, such as a site
that collects investigation results, verify that a response really came from the API, with its
real timing, while the API key stays with the caller.

```
# on the witness host (control port behind HTTPS, e.g. a reverse proxy)
kremlin-witness --key witness.key --allow '*.openai.com' --allow openrouter.ai --receipts receipts/
# the caller
kremlin billing 'openrouter:...' --out runs/b1 --witness relay.example.com:8080,https://relay.example.com/witness,PUBKEY
kremlin witness verify runs/b1/... --pubkey PUBKEY
```

How it works:

1. The client opens `CONNECT host:443` on the relay's tunnel port. The relay dials the host itself
   and forwards raw bytes both ways, timestamping every TLS record. **TLS terminates on the
   client**: the relay holds no keys and cannot read the request (API key, prompt, parameters).
2. After the response, the client discloses only the TLS 1.3 **server-to-client** secrets
   (`SERVER_HANDSHAKE_TRAFFIC_SECRET`, `SERVER_TRAFFIC_SECRET_0`). They are one-way derivations
   (RFC 8446 7.1) that reveal neither the master secret nor the client-to-server keys. The relay
   refuses client-direction secrets.
3. The relay decrypts the server's handshake and checks the certificate chain for the host,
   **CertificateVerify** (the server's signature over this session's transcript, including the
   ClientHello random and key share) and the server Finished. Then it decrypts the application data
   *it recorded itself*; every AEAD tag must verify.
4. It signs (Ed25519) a receipt: host, certificate chain, cipher suite, decrypted response, the
   arrival time of every response record (streamed tokens get honest timestamps), and the opaque
   request size.

5. **Request disclosure (default with `--witness`).** The client sends the request line, `Host` and
   the auth header, then a TLS 1.3 **KeyUpdate**, then the remaining headers and the body under the new
   key, and discloses only that post-update secret (HKDF forward derivation: it cannot be run back to
   the key that encrypted the API key). The relay decrypts the suffix and checks:
   - every post-update record authenticates;
   - the hidden prefix is **exactly** as long as the declared layout (method, path, host, auth header
     name, key length), so no complete alternative request can hide in it;
   - the suffix is headers + exactly `Content-Length` bytes of body, `Connection: close`, nothing after;
   - the server's first response byte arrives after the last body byte.
   The receipt then records method, path, non-secret headers and body. `--no-disclose` keeps the request
   private from the witness (response-only attestation). Residual risk: a key shorter than its declared
   length would leave room for a few smuggled bytes; check key lengths per provider if that matters.

**What exactly is attested** (and what is not):

| | attested? | how |
|---|---|---|
| response body, every byte | yes | relay decrypts what it recorded from the server; certificate signature + AEAD tags |
| response timing (each record, time to first byte) | yes | relay timestamps; it reads from the server at full speed whatever the client does, so a slow reader cannot fake slow decoding (and nothing can fake fast) |
| request body (model, messages, params, provider pin) | yes, TLS 1.3 | post-KeyUpdate disclosure |
| request method and path | bound by length; `audit` also checks the response kind matches the path | they precede the key in HTTP, so they stay in the sealed part; a full proof needs a ZK proof of the first record |
| which account (API key) | no, by design | the key is the account |
| TLS 1.2 hosts' requests | no | no KeyUpdate in TLS 1.2 |

`audit` also rejects declared key lengths that are impossible for the host (e.g. not 73 for OpenRouter), which
would leave room for a smuggled request in the sealed part.

Why the caller cannot fake a receipt: the response bytes came over the relay's own connection to
the host, which the caller is not on, and only the real host can produce a valid CertificateVerify
for the session. Trust assumptions: verifiers trust the relay's signing key and its network path
(covered by certificate validation).

### Auditing a submitted run

```
kremlin audit runs/b1 --pubkey PUBKEY --require-witness
```

re-runs the investigation against the recorded responses (same seed and run salt, so the code
regenerates exactly the requests it sent) and checks, for every request: the receipt signature, that
the receipt's response equals the captured one, and that the witnessed request equals what the
investigation sends. Timing and latency come from the relay's timestamps. Finally it compares the
recomputed results with the submitted `results.json`: a forged conclusion fails even if the submitter
re-seals the manifest.

### Collecting attested runs: kremlin-hub

```
kremlin-hub --dir hub-data --witness-key PUBKEY --listen 127.0.0.1:8090   # on your site, behind HTTPS
kremlin submit runs/b1 --hub https://example.com/hub                       # a contributor
```

The hub unpacks each bundle safely, runs `kremlin audit --require-witness` against the trusted witness
key(s), and publishes only runs where every request was witnessed, every receipt verifies and the
results reproduce. It serves an HTML index, `GET /runs` (JSON) and each run's report, results, audit
and bundle. `kremlin bundle RUN` makes the tarball yourself.

### Running a relay

See `deploy/`: Dockerfile, docker-compose with Caddy (HTTPS control API), systemd unit, and a guide.
The tunnel can itself run over TLS (`--tunnel-cert/--tunnel-key`, clients use `tls://host:port`) so the
access token (`--tokens`, `KREMLIN_WITNESS_TOKEN`) and target host are not sent in clear; receipts name
the submitting token id. `--max-sessions`, `--max-per-ip`, `--max-mb` limit abuse.

Cost: the relay forwards bytes, then does one pass of AEAD decryption and one signature per
request. Streaming, images and audio (multipart) pass through unchanged. Measured overhead: a witnessed request
costs about what any fresh HTTPS connection costs (119 ms vs 114 ms direct to OpenRouter); the relay caches DNS,
answers CONNECT while dialling upstream, and the client closes as soon as the response is complete. Throughput
scales with `--concurrency` (32 req/s with request disclosure at 16 in flight, from one client).

**Provider support** (TLS 1.3 + HTTP/1.1, full verification, checked 2026-11-13 with keyless
requests): OpenRouter, OpenAI, Anthropic, Google Gemini, Google Vertex, Mistral, Together,
Fireworks, DeepInfra, Groq, Cerebras, SambaNova, Parasail, Venice, Hyperbolic, Baseten, xAI,
DeepSeek, Alibaba DashScope, Moonshot, Cohere, Perplexity, Hugging Face router, Nebius,
SiliconFlow, Chutes, Friendli, MiniMax, Z.ai, Featherless, AI21, AWS Bedrock, Cloudflare Workers
AI, Inference.net, GMI Cloud, Atlas Cloud, NVIDIA NIM, Replicate, and Novita over TLS 1.2 (39). Request disclosure (mid-request
KeyUpdate) verified on OpenRouter, OpenAI, Anthropic, Google, Groq, DeepSeek, DeepInfra, Together,
Parasail and Fireworks. TLS 1.2 hosts (Novita) get response-only attestation: TLS 1.2 has no KeyUpdate,
so the request cannot be disclosed. Certificate compression (RFC 8879) is handled.

## Responsible use

Probe only with your own prompts and accounts. The cache probes use fresh random documents and
never try to read anyone else's data; cross-account tests need two accounts you control. Keep
spend caps low; `--plan` first.

## Tests

`pytest` runs offline: a local TLS 1.3 server and a fake OpenAI-compatible provider (a reseller
pair, an independent backend, a prefix cache, a cap-ignoring model, speculation-like streaming,
multipart audio, large images). Every investigation runs both directly and through the witness.
