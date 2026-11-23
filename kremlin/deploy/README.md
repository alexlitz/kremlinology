# Deploying a witness relay

1. **DNS and certificates.** Point `witness.example.com` at the host. Get a certificate (certbot, or let
   Caddy issue it and mount Caddy's files). The tunnel port serves TLS with this certificate, so the
   access token and the target host name are not sent in clear.
2. **Signing key.** Started the first time, the relay creates `witness.key` (Ed25519, mode 600) and prints
   the public key. **Back the key up and publish the public key** on your site, for example at
   `https://witness.example.com/witness/pubkey` and in your documentation. Clients pin it:
   `--witness tls://witness.example.com:8443,https://witness.example.com/witness,PUBKEY`.
   Rotating the key invalidates nothing already signed, but verifiers need the old public key for old
   receipts: keep a list of key ids with validity dates.
3. **Access tokens.** Put one token per line in `tokens.txt` (`openssl rand -hex 24`). Clients send it as
   `KREMLIN_WITNESS_TOKEN`. Receipts carry `submitter_token_id` (a hash prefix, not the token), so you
   know whose run a receipt belongs to. Without `--tokens` the tunnel is open to anyone (the allowlist
   still applies). The relay never dials private, loopback or link-local addresses (no SSRF into your
   network or cloud metadata), and connects to the address it checked (no DNS rebinding);
   `--allow-private` turns this off for local testing only.
4. **Allowlist.** `--allow` patterns limit which hosts the tunnel will dial. Keep it to the APIs you
   want attested. The relay is otherwise an open TCP forwarder.
5. **Limits.** `--max-sessions`, `--max-per-ip` (keep it at least the clients' `--concurrency`), `--max-mb` (per
   session, both directions), `--rate N` (new sessions per minute per token, or per IP without tokens).
   **Revoking a token:** delete its line from `tokens.txt`; the relay re-reads the file on the next connection.
   **Data:** by default receipts are kept forever and the relay announces that the operator may use the data for
   any purpose (`--terms`, `--terms-url`; clients print the terms). `--retention-days N` purges instead.
   **Logs and metrics:** one JSON line per event on stdout (session open/close, receipt, rejection, token reload,
   purge; metadata only, never content); `GET /witness/metrics` returns counters.
6. **Run.** `docker compose up -d` in this directory (edit hostnames and paths first), or install the
   package and use `kremlin-witness.service`.

7. **Hub (optional).** Submissions are queued and audited by `--workers` background workers (`GET /jobs/ID`
   for status). Put an admin token in `admin-token.txt` (`openssl rand -hex 24`) to allow
   `DELETE /hub/runs/ID`. The `hub` service publishes contributors' runs at `/hub/` after re-auditing them
   against your witness key: set `--witness-key` to the key printed in step 2 (add old keys too).
   Contributors run `kremlin submit RUN --hub https://witness.example.com/hub`.

## What the relay sees and keeps

- **Never:** API keys or anything else in the hidden request prefix (request line, Host, auth header).
  Client-direction secrets are refused if sent.
- **During a session (memory only):** the encrypted bytes in both directions, the target host.
- **After disclosure:** the decrypted response; with request disclosure (default), the request headers
  other than auth, and the request body. Raw session bytes are dropped after disclosure or after 15 min.
- **Kept on disk:** signed receipts (`--receipts`), which contain responses and disclosed request bodies.
  Treat them as user data: retention policy, access control, deletion on request.

## Checking it

```
curl https://witness.example.com/witness/pubkey
KREMLIN_WITNESS_TOKEN=... kremlin request 'openrouter:openai/gpt-oss-120b' models --method GET --out /tmp/w1 \
  --witness tls://witness.example.com:8443,https://witness.example.com/witness,PUBKEY
```
