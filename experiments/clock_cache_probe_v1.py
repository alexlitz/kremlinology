"""Cold/warm clock test for a pinned endpoint, per trial (2026-11-21; paid, witnessed via KREMLINOLOGY_OPENROUTER_URL).

For each size and trial: a unique nonce prompt (cold = the cache write), then the identical request 3 s later
(warm = the read). Records per-trial elapsed, reported cached tokens, and cost, so the analysis can fit the cold
distribution (overhead + tokens/prefill_rate) and score each warm trial against it: the clock test for whether a
reported hit is corroborated by time. --bp adds Anthropic-style breakpoints (two blocks, cache_control on the big one).
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/clock_cache_probe_v1.py OUT --model M --provider P \
        --sizes 8000,32000 --trials 20 [--bp]
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
WORDS = ('system cache token latency memory block prefix model request provider window layer state attention '
         'vector matrix budget schedule replica storage compute network kernel batch').split()

def text(seed, tokens):
    rng = random.Random(seed); w = int(tokens / 1.15); out = []
    while len(out) < w:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]; out[-1] += '.'
    return ' '.join(out)

def messages(body_text, bp):
    if bp:
        return [dict(role='user', content=[dict(type='text', text=body_text, cache_control=dict(type='ephemeral')),
                                           dict(type='text', text='Reply with OK.')])]
    return [dict(role='user', content=body_text + '\n\nReply with OK.')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--model', required=True); ap.add_argument('--provider', required=True)
    ap.add_argument('--sizes', required=True); ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--bp', action='store_true'); ap.add_argument('--cap', type=float, default=3.0)
    ap.add_argument('--reasoning-off', action='store_true', default=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    rng = random.Random(20261121)
    cl = httpx.Client(timeout=600)
    rows, spent = [], 0.0
    for size in [int(x) for x in a.sizes.split(',')]:
        for trial in range(a.trials):
            body_text = f'Reference {rng.getrandbits(64):016x}. ' + text(rng.getrandbits(48), size)
            for kind in ('cold', 'warm'):
                body = dict(model=a.model, max_tokens=1, temperature=0, messages=messages(body_text, a.bp),
                            usage=dict(include=True), provider=dict(only=[a.provider], allow_fallbacks=False))
                t = time.time()
                r = cl.post(URL, json=body, headers={'Authorization': 'Bearer ' + key})
                el = time.time() - t
                j = r.json() if r.status_code == 200 else {}
                u = j.get('usage') or {}
                spent += u.get('cost') or 0
                rows.append(dict(size=size, trial=trial, kind=kind, status=r.status_code, elapsed_s=el,
                                 prompt=u.get('prompt_tokens'),
                                 cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                                 cache_read=(u.get('prompt_tokens_details') or {}).get('cache_read_input_tokens'),
                                 cache_write=(u.get('prompt_tokens_details') or {}).get('cache_creation_input_tokens'),
                                 cost=u.get('cost'), served=j.get('provider'), generation=j.get('id')))
                (out / 'rows.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
                if spent >= a.cap:
                    break
            if spent >= a.cap:
                break
            time.sleep(3)
        print(json.dumps(dict(done=len(rows), spent=spent)), flush=True)
    cl.close()
    (out / 'summary.json').write_text(json.dumps(dict(model=a.model, provider=a.provider, trials=a.trials,
                                                      rows=len(rows), spent=spent), indent=1))
    print(json.dumps(dict(stop='done', rows=len(rows), spent=spent)))

if __name__ == '__main__':
    main()
