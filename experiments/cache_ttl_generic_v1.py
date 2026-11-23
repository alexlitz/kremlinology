"""TTL probe for a pinned endpoint through the witness proxy (2026-11-21; KREMLINOLOGY_OPENROUTER_URL).

Unique nonce prompts (one per trial), each sent twice with a delay in between; writes spread over the first minutes.
Records elapsed per request so the clock (not just the reported cached count) says whether the prefix survived.
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/cache_ttl_generic_v1.py OUT --model M --provider P \
        --tokens 16000 --delays 300,1800,7200,21600,86400 --trials 8
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
WORDS = ('system cache token latency memory block prefix model request provider window layer state attention '
         'vector matrix budget schedule replica storage compute network kernel batch').split()

def text(rng, tokens):
    w = int(tokens / 1.15); out = []
    while len(out) < w:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]; out[-1] += '.'
    return ' '.join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--model', required=True); ap.add_argument('--provider', required=True)
    ap.add_argument('--tokens', type=int, default=16000); ap.add_argument('--delays', required=True)
    ap.add_argument('--trials', type=int, default=8); ap.add_argument('--cap', type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    rng = random.Random(20261122)
    events = []
    span = max(int(d) for d in a.delays.split(','))
    for trial in range(a.trials):
        for delay in [int(d) for d in a.delays.split(',')]:
            start = rng.uniform(0, max(60, a.trials * 60))
            events.append((start, trial, delay, 'A'))
            events.append((start + delay, trial, delay, 'B'))
    events.sort()
    cl = httpx.Client(timeout=600)
    rows, spent, t0 = [], 0.0, time.time() + 5
    prompts = {}
    for when, trial, delay, role in events:
        wait = t0 + when - time.time()
        if wait > 0:
            time.sleep(wait)
        if (trial, delay) not in prompts:
            prompts[(trial, delay)] = f'Reference {rng.getrandbits(64):016x}. ' + text(rng, a.tokens)
        body = dict(model=a.model, max_tokens=1, temperature=0,
                    messages=[dict(role='user', content=prompts[(trial, delay)] + '\n\nReply with OK.')],
                    usage=dict(include=True), provider=dict(only=[a.provider], allow_fallbacks=False))
        t = time.time()
        r = cl.post(URL, json=body, headers={'Authorization': 'Bearer ' + key})
        el = time.time() - t
        j = r.json() if r.status_code == 200 else {}
        u = j.get('usage') or {}
        spent += u.get('cost') or 0
        rows.append(dict(trial=trial, delay=delay, role=role, status=r.status_code, elapsed_s=el,
                         prompt=u.get('prompt_tokens'), cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                         cost=u.get('cost'), served=j.get('provider'), generation=j.get('id'), at=when))
        (out / 'rows.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        if spent >= a.cap:
            break
    cl.close()
    (out / 'summary.json').write_text(json.dumps(dict(model=a.model, provider=a.provider, trials=a.trials,
                                                      delays=a.delays, rows=len(rows), spent=spent), indent=1))
    print(json.dumps(dict(stop='done' if spent < a.cap else 'cap', rows=len(rows), spent=spent, hours=(time.time()-t0)/3600)))

if __name__ == '__main__':
    main()
