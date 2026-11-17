"""Tabulate cache_semantics_probe_v1-v3: cache reads/writes per test step, per model/provider (and repeat)."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def usage_fields(usage):
    usage = usage or {}
    details = usage.get('prompt_tokens_details') or {}
    return usage.get('prompt_tokens'), details.get('cached_tokens'), details.get('cache_write_tokens')


def main():
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
    table = defaultdict(dict)
    for r in rows:
        table[(r['label'], r['test'], r.get('repeat', 0))][r['step']] = r
    labels = list(dict.fromkeys(r['label'] for r in rows))
    tests = list(dict.fromkeys((r['test'], r.get('repeat', 0)) for r in rows))
    out = []
    for label in labels:
        print(f'== {label}')
        for test, repeat in tests:
            steps = table.get((label, test, repeat))
            if not steps:
                continue
            cells = []
            for step in sorted(steps):
                r = steps[step]
                prompt, cached, write = usage_fields(r['usage'])
                if r['status'] != 200:
                    cells.append(f"step{step}: HTTP {r['status']} {str(r.get('error'))[:60]}")
                    out.append(dict(label=label, test=test, repeat=repeat, step=step, status=r['status']))
                    continue
                cells.append(f'step{step}: prompt {prompt} cached {cached} write {write}')
                out.append(dict(label=label, test=test, repeat=repeat, step=step, status=200, prompt=prompt, cached=cached,
                                write=write, provider=r.get('reported_provider')))
            print(f'  {test + "#" + str(repeat):28s} ' + " | ".join(cells))
    (run / 'analysis.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
