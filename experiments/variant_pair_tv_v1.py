"""Is a "turbo/fast/spot" variant the same backend as its standard sibling? (2026-11-22; paid, via proxy).

Pairs of endpoints of the same model on the same provider. Per prompt (n≈40, one-token, T=1, top-20 logprobs on
both sides): total variation over the union of reported top-20, and exact identity (logprob gap <= 0.002 on the
shared tokens, the reseller test). Self-noise for each side comes from repeat requests in the same run. Read: TV at
the self-noise level = same deployment tier; TV at the cross-provider level = different quantisation or weights.
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/variant_pair_tv_v1.py OUT
"""
import argparse, json, os, random, time
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
PAIRS = [
    ('openai/gpt-oss-120b', 'parasail/fp4', 'parasail/fast', 'gptoss@parasail-fp4~fast'),
    ('openai/gpt-oss-120b', 'fireworks', 'fireworks/spot', 'gptoss@fireworks~spot'),
    ('deepseek/deepseek-v4.1-flash', 'venice/fp8', 'venice/ultrafast', 'dsv4.1@venice-fp8~ultrafast'),
    ('deepseek/deepseek-v4.1-flash', 'deepinfra/fp8', 'deepinfra/fp8/priority', 'dsv4.1@deepinfra~priority'),
]
PROMPTS = ([f'Write the first line of a Python function that {x}.' for x in
            ('sorts a list of dicts by a key', 'reads a csv', 'retries on failure', 'parses iso dates',
             'normalises unicode', 'shuffles with a seed', 'validates an email', 'chunks a stream',
             'caches a result', 'walks a directory')] +
           [f'In one sentence, what is {x}?' for x in
            ('a database index', 'a cache line', 'the paging theorem', 'a syscall', 'tail latency',
             'a bloom filter', 'rpc batching', 'the memory wall', 'rowhammer', 'an rdma verb')] +
           ['Continue: The tower was', 'Continue: She opened the', 'Translate to French: the cat',
            'Summarise in three words: fast cheap good', 'Name a color:', 'Name a tool:',
            'What is 17*23?', 'What is the capital of Peru?', 'First word of your reply must be Hello.',
            'Reply with a single verb:', 'List two primes:', 'Opposite of hot is']) * 2


def fetch(cl, key, model, provider, prompt, seed):
    body = dict(model=model, max_tokens=1, temperature=1, seed=seed, logprobs=True, top_logprobs=20,
                usage=dict(include=True), provider=dict(only=[provider], allow_fallbacks=False),
                messages=[dict(role='user', content=prompt)])
    r = cl.post(URL, json=body, headers={'Authorization': 'Bearer ' + key})
    if r.status_code != 200:
        return None
    content = ((r.json()['choices'][0].get('logprobs') or {}).get('content') or [])
    return {e['token']: e['logprob'] for e in content[0]['top_logprobs']} if content else None


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, -30) - b.get(k, -30)) for k in keys)


def identity(a, b):
    shared = set(a) & set(b)
    return len(shared) >= 10 and max((abs(a[k] - b[k]) for k in shared), default=99) <= 0.002


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--cap', type=float, default=0.8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key = os.environ['OPENROUTER_API_KEY']
    cl = httpx.Client(timeout=120)
    rows = []
    for model, pa, pb, label in PAIRS:
        tvs, ids, selfa, selfb = [], 0, [], []
        for i, p in enumerate(PROMPTS):
            la, lb = fetch(cl, key, model, pa, p, i), fetch(cl, key, model, pb, p, i)
            if la and lb:
                tvs.append(tv(la, lb)); ids += identity(la, lb)
            la2 = fetch(cl, key, model, pa, p, i)
            if la and la2:
                selfa.append(tv(la, la2))
            lb2 = fetch(cl, key, model, pb, p, i)
            if lb and lb2:
                selfb.append(tv(lb, lb2))
            time.sleep(0.3)
        row = dict(pair=label, n=len(tvs), median_tv=sorted(tvs)[len(tvs) // 2] if tvs else None,
                   exact_identity=ids, self_tv_a=sorted(selfa)[len(selfa) // 2] if selfa else None,
                   self_tv_b=sorted(selfb)[len(selfb) // 2] if selfb else None)
        rows.append(row); print(json.dumps(row), flush=True)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
    cl.close()


if __name__ == '__main__':
    main()
