# Walkthrough: is one provider reselling another?

This reproduces, at small scale, how we found that Venice serves Qwen3.5-397B from Parasail's
backend (2026-11-13). Total cost: under $0.01.

## 1. Find the routes

```
kremlin routes qwen/qwen3.5-397b-a17b
```

lists every OpenRouter endpoint with its quantization, prices, cache price and logprob support. Two
providers with identical or suspiciously close prices for the same model are worth a look.

## 2. Fingerprint

```
kremlin fingerprint 'openrouter:qwen/qwen3.5-397b-a17b@parasail/fp8;label=parasail' \
                    'openrouter:qwen/qwen3.5-397b-a17b@venice;label=venice' \
                    'openrouter:qwen/qwen3.5-397b-a17b@alibaba;label=alibaba' \
                    --prompts 30 --top-logprobs 5 --extra '{"reasoning":{"enabled":false}}' \
                    --max-usd 0.05 --out runs/qwen-fp
```

Read the report's two tables:

- **self-match**: how often each route returns bit-identical first-token logprobs for the same prompt
  twice. Inference is rarely bit-deterministic, so this is each route's floor (Parasail ~70%, Alibaba
  ~18% on our full run).
- **identical (same prompt+repeat)** per pair: independent deployments essentially never match to
  0.002 nats. Venice matched Parasail as often as Parasail matched itself, and nobody else.
- **lands on A's states**: when Parasail's own repeats disagree, does Venice reproduce one of them? An
  independent copy of the same recipe would sometimes produce a third variant; a relay never does.

## 3. Corroborate

- `upstream ids (OpenRouter)` in the fingerprint report: Venice responses carry Parasail's
  `chatcmpl-*32` format.
- Latency: a relay adds a hop (about 90 ms here).
- Cache sharing, on a model both cache:

  ```
  kremlin cache-share 'openrouter:google/gemma-4-26b-a4b-it@parasail/bf16;label=parasail' \
                      'openrouter:google/gemma-4-26b-a4b-it@venice/bf16;label=venice' \
                      --max-usd 0.05 --out runs/gemma-share
  ```

  A prompt cached through one route was read through the other as often as a same-route repeat:
  one backend, and a prefix cache shared across that provider's customers.

## 4. Make it checkable by others

Run the same commands with `--witness tls://witness.example.com:8443,https://witness.example.com/witness,PUBKEY`.
Every request then has a signed receipt from the relay (responses, timing, and the request itself
except the API key). Anyone can re-check:

```
kremlin audit runs/qwen-fp --pubkey PUBKEY --require-witness
kremlin submit runs/qwen-fp --hub https://witness.example.com/hub
```

## What it does not show

A bitwise match shows shared numerics (same engine, weights, hardware and batching), not a contract or
who pays whom. Combine it with public facts: Parasail lists Venice as a customer.
