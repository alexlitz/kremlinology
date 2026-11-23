"""Analyse cache_refresh_probe_v1: does a full or partial hit keep a cache entry alive?

Per endpoint and arm, the probe at D (same full prompt as the write) is classified by its reported cached share:
  whole  >= 80% of the prompt cached;
  head   10-80% (a partial ping's first 25%, or Anthropic's first breakpoint at 50%);
  none   < 10%.
The ping itself is a hit if at least half its own prompt was reported cached. A missed full ping re-writes the entry,
so the probe after it says nothing about refreshing; results are given for all trials and for trials whose ping hit.
Hit rates with 95% Wilson intervals; probe latency (non-streamed one-token request, client clock) by outcome.
Run: python experiments/cache_refresh_analysis_v1.py RUN_DIR [RUN_DIR ...]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_utils import wilson  # noqa: E402


def cached_share(r):
    u = r.get('usage') or {}
    prompt = u.get('prompt_tokens') or 0
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    return cached / prompt if prompt else None


def main():
    trials = defaultdict(dict)
    for k, run in enumerate(sys.argv[1:]):
        for line in (Path(run) / 'requests.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                trials[(k, r['trial'])][r['role']] = r
    cells = defaultdict(list)
    for t in trials.values():
        probe, write = t.get('probe'), t.get('write')
        if not probe or not write or probe.get('status') != 200 or write.get('status') != 200:
            continue
        share = cached_share(probe)
        if share is None:
            continue
        ping = t.get('ping')
        ping_hit = None
        if probe['arm'] != 'control':
            if not ping or ping.get('status') != 200:
                continue
            ps = cached_share(ping)
            ping_hit = ps is not None and ps >= 0.5
        outcome = 'whole' if share >= 0.8 else 'head' if share >= 0.1 else 'none'
        cells[(probe['provider'], probe['model'], probe['arm'])].append(dict(
            outcome=outcome, share=share, ping_hit=ping_hit, latency=probe['elapsed_s'], ping_s=probe['ping_s'],
            probe_s=probe['probe_s']))
    report = []
    for (provider, model, arm), rows in sorted(cells.items(), key=lambda x: (x[0][0], ('control', 'full', 'partial', 'near', 'samectx').index(x[0][2]))):
        def summary(sel):
            n = len(sel)
            out = dict(n=n)
            for o in ('whole', 'head', 'none'):
                k = sum(r['outcome'] == o for r in sel)
                out[o] = k
                out[o + '_ci'] = wilson(k, n) if n else (None, None)
            for o in ('whole', 'head', 'none'):
                lat = [r['latency'] for r in sel if r['outcome'] == o]
                out[o + '_latency'] = statistics.median(lat) if lat else None
            return out
        entry = dict(provider=provider, model=model, arm=arm, ping_s=rows[0]['ping_s'], probe_s=rows[0]['probe_s'],
                     all=summary(rows))
        if arm != 'control':
            entry['ping_hits'] = sum(bool(r['ping_hit']) for r in rows)
            entry['given_ping_hit'] = summary([r for r in rows if r['ping_hit']])
        report.append(entry)
        a = entry['all']
        extra = ''
        if arm != 'control':
            g = entry['given_ping_hit']
            extra = f"  ping hit {entry['ping_hits']}/{a['n']}; given ping hit: whole {g['whole']}/{g['n']} head {g['head']}/{g['n']}"
        lat = ' '.join(f"{o} {a[o + '_latency']:.2f}s" for o in ('whole', 'head', 'none') if a[o + '_latency'] is not None)
        print(f"{provider:16s} {arm:8s} ping {rows[0]['ping_s']:>5}s probe {rows[0]['probe_s']:>5}s  "
              f"whole {a['whole']:2d}/{a['n']:<2d} head {a['head']:2d} none {a['none']:2d}{extra}  [{lat}]")
    Path(f'reports/cache-refresh-{Path(sys.argv[1]).name.replace("kremlinology-cache-refresh-", "")}.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
