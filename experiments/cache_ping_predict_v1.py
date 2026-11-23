"""How well can a prefix's return be predicted? Conversation-level study on the Mooncake traces (2026-11-19; free).

The v2 block-level simulation showed the value of keep-alive pinging is entirely in predicting which prefixes return;
block statistics (idle age, uses) predict that poorly. Here the unit is the conversation: requests are chained when
an earlier request's block-hash list is a prefix of a later one (greedy longest match, most recent first), which is
the thing a user-side pinger actually keeps alive. Per conversation, after each request, the predictor estimates
P(another request within I seconds | idle age, features) from: idle age, requests-so-far, the conversation's own
arrival rhythm so far, and the last output length -- all known at decision time, fitted on a held-out half of the
conversations. Reported: AUC of the return-within-I classifier, calibration, the share of block reuses that are
intra-conversation (pingeable by their owner) vs cross-conversation (provider-side only), and the ping-policy value
with the feature predictor vs age-only, blind, and the oracle.
Run: python experiments/cache_ping_predict_v1.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from kv_reuse_tiers_v1 import load_trace, block_events  # noqa: E402

BLOCK = 512
ENDPOINTS = [('haiku4.5', 300, 0.10, 1.25), ('kimi-k2.6', 600, 0.168, 1.0)]
VARIANTS = [('conversation', 'conversation_trace.jsonl', 1), ('conversation-x3', 'conversation_trace.jsonl', 3),
            ('conversation-x8', 'conversation_trace.jsonl', 8), ('toolagent', 'toolagent_trace.jsonl', 1),
            ('arxiv', 'arxiv_mooncake_trace.jsonl', 1), ('synthetic', 'synthetic_trace.jsonl', 1)]
AGE_BINS = [0, 15, 30, 60, 120, 240, 480, 960, 1920, 1e9]
N_BINS = (1, 2, 4, 8, 1 << 30)      # requests so far
RATE_BINS = (0, 30, 120, 600, 1e9)  # mean seconds between this conversation's requests so far
OUT_BINS = (0, 100, 500, 2000, 1e9)
MIN_CELL = 25


def bin_of(value, bins):
    return min(bins, key=lambda b: abs(b - value))


def bin_age(t):
    return next((a for a in AGE_BINS if t < a), AGE_BINS[-1])


def conversations(rows, stretch):
    """Greedy prefix-chain grouping: a request continues the most recent conversation whose last hashes it extends.
    Every prefix of a conversation's last hash list is indexed (a continuation may share only part of it, since the
    final partial block is re-hashed when the prompt grows)."""
    index = {}  # prefix tuple -> the most recent conversation under that prefix
    convs = []

    def reindex(c):
        for key in c['keys']:
            index[key] = c  # most recent wins; older ones stay reachable via their unique deeper prefixes

    def keys_of(hashes):
        return [hashes[:k] for k in range(2, len(hashes) + 1)]

    for r in rows:
        hashes = tuple(r['hash_ids'])
        keys = keys_of(hashes)
        match = next((index[hashes[:k]] for k in range(len(hashes), 1, -1) if hashes[:k] in index), None)
        item = dict(t=r['timestamp'] / 1000 * stretch, blocks=len(hashes), out=r['output_length'])
        if match is None:
            match = dict(events=[item], last=hashes, keys=keys)
            convs.append(match)
        else:
            match['events'].append(item)
            match['last'] = hashes
            match['keys'] = keys
        reindex(match)
    return convs


def features(conv):
    """Per idle period: (age-independent features, gap seconds or None for the terminal one)."""
    ev = conv['events']
    out = []
    for i in range(len(ev)):
        t = ev[i]['t']
        n = i + 1
        rate = (t - ev[0]['t']) / i if i else RATE_BINS[-1]
        f = dict(n=bin_of(n, N_BINS), rate=bin_of(rate, RATE_BINS), out=bin_of(ev[i]['out'], OUT_BINS),
                 blocks=ev[i]['blocks'])
        out.append((f, (ev[i + 1]['t'] - t) if i + 1 < len(ev) else None))
    return out


def fit(train_pairs, I, key_fn):
    at_risk, returned = defaultdict(int), defaultdict(int)
    for f, gap in train_pairs:
        gap = float('inf') if gap is None else gap  # a dead conversation never returns: a deployed model trains on history
        for age in AGE_BINS:
            if gap >= age:
                at_risk[key_fn(age, f)] += 1
                if gap - age <= I:
                    returned[key_fn(age, f)] += 1
    return {k: returned[k] / n for k, n in at_risk.items() if n >= MIN_CELL}


def auc(pairs, table, I, key_fn):
    """Rank AUC of P(return within I) at idle age 0 (the moment a conversation goes idle)."""
    scores, labels = [], []
    for f, gap in pairs:
        scores.append(table.get(key_fn(0, f), 0.0))
        labels.append(1 if gap is not None and gap <= I else 0)
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    # rank-based AUC with ties split evenly (Mann-Whitney), O(n log n)
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        shared = (i + j - 1) / 2 + 1  # average rank of the tied block (1-based)
        for k in range(i, j):
            ranks[order[k]] = shared
        i = j
    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def simulate(convs, L, rho, w, I, policy, table, key_fn):
    cost = pings = waste = 0
    cap = max(0, int(w / rho) - 1)
    for conv in convs:
        ev = conv['events']
        feats = features(conv)
        for i in range(len(ev)):
            f, gap = feats[i]
            blocks = ev[i]['blocks']
            g = gap if gap is not None else 1e9
            shared = blocks if gap is not None else blocks  # shared prefix at the next use (approx: this request's)
            if gap is not None and g <= L:
                continue
            if policy == 'oracle':
                if gap is not None:
                    cost += shared * min(w, (-(-g // L) - 1) * rho)
                continue
            expiry, t, n = L, I, 0
            while t < g and n < cap:
                # the ping only saves a miss for a return in (would-be death, new death]: hazard at that window
                if policy != 'blind' and table.get(key_fn(t + L - I, f), 0.0) * w < rho:
                    break
                cost += shared * rho
                pings += 1
                if gap is None:
                    waste += 1
                n += 1
                t += I
            if gap is not None and g > expiry:  # death happens at expiry: any use after it is a miss
                cost += shared * w
    return cost, pings, waste


def main():
    out = {}
    for vname, fname, stretch in VARIANTS:
        rows = load_trace(fname)
        convs = conversations(rows, stretch)
        convs = [c for c in convs if len(c['events']) > 1]
        convs.sort(key=lambda c: c['events'][0]['t'])
        train, test = convs[::2], convs[1::2]
        multi = sum(1 for c in convs)
        uses = block_events(rows)
        intra = sum(len(ev) - 1 for ev in uses.values())
        entry = dict(variant=vname, conversations_total=len(conversations(rows, stretch)), conversations_multi=multi,
                     endpoints={})
        print(f"\n{vname}: {entry['conversations_total']} conversations, {multi} with >1 request", flush=True)
        for label, L, rho, w in ENDPOINTS:
            I = int(0.9 * L)
            train_pairs = [p for c in train for p in features(c)]
            test_pairs = [p for c in test for p in features(c)]
            keys = {'age+n': lambda a, f: (bin_age(a), f['n']), 'age+rate': lambda a, f: (bin_age(a), f['rate']),
                    'age+out': lambda a, f: (bin_age(a), f['out']), 'age only': lambda a, f: bin_age(a)}
            tables = {k: fit(train_pairs, I, kf) for k, kf in keys.items()}
            aucs = {k: auc(test_pairs, tables[k], I, keys[k]) for k in tables}
            reuses = sum(1 for f, g in test_pairs if g is not None)
            none_c = simulate(test, L, rho, w, I, 'none', {}, lambda a, f: 0)[0]
            blind = simulate(test, L, rho, w, I, 'blind', {}, lambda a, f: 0)
            best_name = max((k for k in tables if aucs[k]), key=lambda k: aucs[k] or 0)
            pred = simulate(test, L, rho, w, I, 'pred', tables[best_name], keys[best_name])
            oracle_c = simulate(test, L, rho, w, I, 'oracle', {}, lambda a, f: 0)[0]
            per_m = lambda c: c / max(reuses, 1) * (1.0 if label == 'haiku4.5' else 0.95)
            e = dict(auc={k: (round(v, 3) if v else None) for k, v in aucs.items()}, best_feature=best_name,
                     miss_mtok=round(per_m(none_c), 4), blind_mtok=round(per_m(blind[0]), 4),
                     pred_mtok=round(per_m(pred[0]), 4), oracle_mtok=round(per_m(oracle_c), 4),
                     blind_waste=round(blind[2] / max(blind[1], 1), 3), pred_waste=round(pred[2] / max(pred[1], 1), 3))
            entry['endpoints'][label] = e
            print(f"  {label:10s} AUC {e['auc']} | $/Mtok: miss {e['miss_mtok']:.4f} blind {e['blind_mtok']:.4f} "
                  f"({e['blind_waste']:.0%} wasted) pred({best_name}) {e['pred_mtok']:.4f} ({e['pred_waste']:.0%}) "
                  f"oracle {e['oracle_mtok']:.4f}")
        out[vname] = entry
    (ROOT / 'reports/cache-ping-predict-2026-11-19.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
