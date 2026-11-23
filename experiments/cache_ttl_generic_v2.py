"""Resumable TTL probe for a pinned endpoint through the witness proxy (2026-11-23).

v1 kept its schedule in memory and died with its session 33 writes into a 24-hour run. v2 persists the plan
(absolute send times and a seed per prompt) in OUT/plan.json and appends every request to OUT/rows.jsonl, so a
restart skips what is done and sends anything overdue at once, recording the actual delay. Prompts are regenerated
from their seeds. Each trial is a unique nonce prompt sent twice: A (the write), then B after its delay; elapsed time
is kept so the clock, not just the reported cached count, says whether the prefix survived.
Run: KREMLINOLOGY_OPENROUTER_URL=... python experiments/cache_ttl_generic_v2.py OUT --model M --provider P \
        --tokens 16000 --delays 1800,7200,21600,43200,86400 --trials 20 [--cap 0.55]
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import httpx

URL = os.environ.get('KREMLINOLOGY_OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions')
WORDS = ('system cache token latency memory block prefix model request provider window layer state attention '
         'vector matrix budget schedule replica storage compute network kernel batch').split()


def prompt(seed, tokens):
    rng = random.Random(seed)
    w = int(tokens / 1.15); out = []
    while len(out) < w:
        out += [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]; out[-1] += '.'
    return f'Reference {rng.getrandbits(64):016x}. ' + ' '.join(out) + '\n\nReply with OK.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--model', required=True); ap.add_argument('--provider', required=True)
    ap.add_argument('--tokens', type=int, default=16000); ap.add_argument('--delays', required=True)
    ap.add_argument('--trials', type=int, default=20); ap.add_argument('--cap', type=float, default=0.55)
    ap.add_argument('--spread', type=float, default=600, help='seconds over which the writes are spread')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan_path, rows_path = out / 'plan.json', out / 'rows.jsonl'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        rng = random.Random()
        t0 = time.time() + 10
        events = []
        for trial in range(a.trials):
            for delay in [int(d) for d in a.delays.split(',')]:
                start = t0 + rng.uniform(0, a.spread)
                seed = rng.getrandbits(60)
                events.append(dict(when=start, trial=trial, delay=delay, role='A', seed=seed))
                events.append(dict(when=start + delay, trial=trial, delay=delay, role='B', seed=seed))
        plan = dict(model=a.model, provider=a.provider, tokens=a.tokens, cap=a.cap, created=time.time(),
                    events=sorted(events, key=lambda e: e['when']))
        plan_path.write_text(json.dumps(plan))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r['trial'], r['delay'], r['role']): r for r in rows if r.get('status') == 200}
    spent = sum(r.get('cost') or 0 for r in rows)
    key = os.environ['OPENROUTER_API_KEY']
    cl = httpx.Client(timeout=600)
    for e in plan['events']:
        k = (e['trial'], e['delay'], e['role'])
        if k in done:
            continue
        if e['role'] == 'B' and (e['trial'], e['delay'], 'A') not in done:
            continue  # its write never succeeded
        if spent >= plan['cap']:
            print(json.dumps(dict(stop='cap', spent=spent))); sys.exit(0)
        wait = e['when'] - time.time()
        if wait > 0:
            time.sleep(wait)
        body = dict(model=plan['model'], max_tokens=1, temperature=0,
                    messages=[dict(role='user', content=prompt(e['seed'], plan['tokens']))],
                    usage=dict(include=True), provider=dict(only=[plan['provider']], allow_fallbacks=False))
        row = dict(trial=e['trial'], delay=e['delay'], role=e['role'], scheduled=e['when'])
        for attempt in range(3):
            t = time.time()
            try:
                r = cl.post(URL, json=body, headers={'Authorization': 'Bearer ' + key})
                j = r.json() if r.status_code == 200 else {}
                status = r.status_code
            except Exception as error:  # proxy down, timeouts: let the wrapper restart us
                print(json.dumps(dict(error=type(error).__name__, at=k)), flush=True)
                sys.exit(3)
            if status == 200:
                break
            time.sleep(5 + 10 * attempt)
        u = j.get('usage') or {}
        row.update(status=status, sent=t, elapsed_s=time.time() - t, prompt=u.get('prompt_tokens'),
                   cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens'), cost=u.get('cost'),
                   served=j.get('provider'), generation=j.get('id'))
        if e['role'] == 'B':
            row['actual_delay'] = t - done[(e['trial'], e['delay'], 'A')]['sent']
        with rows_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        if status == 200:
            done[k] = row
            spent += u.get('cost') or 0
    cl.close()
    (out / 'summary.json').write_text(json.dumps(dict(model=plan['model'], provider=plan['provider'], rows=len(done),
                                                      spent=spent, finished=time.time()), indent=1))
    print(json.dumps(dict(stop='done', rows=len(done), spent=spent)))


if __name__ == '__main__':
    main()
