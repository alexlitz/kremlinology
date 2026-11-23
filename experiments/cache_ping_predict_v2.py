"""How well CAN it be predicted? Real models on the same features, and the accuracy needed to profit (2026-11-19).

Extends cache_ping_predict_v1: instead of binned hazard tables, a logistic regression and a small neural net (plain
numpy, no framework) predict P(the conversation returns within the affordable bridge horizon) from deployable features --
idle age, requests-so-far, the conversation's arrival rhythm, last output length, prompt size, conversation age --
evaluated at each ping epoch exactly as the policy uses it. Held-out conversations: AUC, Brier, calibration. Then the
policy value of each predictor, and a degraded-oracle sweep (the oracle's answer flipped with probability eps) that
maps prediction quality to dollars and locates the break-even AUC: the accuracy a predictor needs before keep-alive
pinging pays at all.
Run: python experiments/cache_ping_predict_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from kv_reuse_tiers_v1 import load_trace  # noqa: E402
import cache_ping_predict_v1 as base  # noqa: E402

ENDPOINTS = [('haiku4.5', 300, 0.10, 1.25), ('kimi-k2.6', 600, 0.168, 1.0)]
TRACES = [('toolagent', 'toolagent_trace.jsonl', 1), ('arxiv', 'arxiv_mooncake_trace.jsonl', 1),
          ('conversation', 'conversation_trace.jsonl', 1)]
SEED = 20261119


def epoch_row(ev, i, t):
    """Feature row for the ping decision at idle age t after event i (shared by training and the policy)."""
    n = i + 1
    conv_age = ev[i]['t'] - ev[0]['t']
    rate = conv_age / i if i else 1e9
    return [np.log1p(t), np.log1p(ev[i]['blocks']), np.log1p(ev[i]['out']), np.log1p(conv_age), np.log1p(rate), n]


def dataset(convs, I, cap):
    """Rows at every ping epoch of every idle period (terminal ones included, label 0). The label is a return
    within the affordable bridge horizon cap*I: that is the event a ping chain can still monetise."""
    X, Y, owner = [], [], []
    for ci, conv in enumerate(convs):
        ev = conv['events']
        for i in range(len(ev)):
            nxt = ev[i + 1]['t'] if i + 1 < len(ev) else None
            t, epoch = I, 0
            while epoch < cap + 2 and (nxt is None or t < nxt):
                label = 1.0 if (nxt is not None and t <= nxt < t + cap * I) else 0.0
                X.append(epoch_row(ev, i, t))
                Y.append(label)
                owner.append((ci, i, t, nxt))
                t += I
                epoch += 1
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), owner


def auc(p, y):
    order = np.argsort(p, kind='stable')
    ranks = np.empty(len(p), np.float64)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    pos, neg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else 0.5


class Logistic:
    name = 'logistic'

    def fit(self, X, Y, Xv, Yv):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xa = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1), np.float32)])
        w = np.zeros(Xa.shape[1], np.float32)
        for _ in range(200):
            p = 1 / (1 + np.exp(-(Xa @ w).clip(-30, 30)))
            grad = Xa.T @ (p - Y) / len(Y) + 1e-4 * w
            hess = (Xa * (p * (1 - p))[:, None]).T @ Xa / len(Y) + 1e-4 * np.eye(len(w), dtype=np.float32)
            step = np.linalg.solve(hess, grad)
            w -= step.astype(np.float32)
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1), np.float32)])
        return 1 / (1 + np.exp(-(Z @ self.w).clip(-30, 30)))


class MLP:
    """One hidden layer (32), early stopping on validation AUC. Plain numpy."""

    name = 'mlp'

    def fit(self, X, Y, Xv, Yv, hidden=32, epochs=600, lr=0.02):
        rng = np.random.default_rng(SEED)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xa = (X - self.mu) / self.sd
        Xva = (Xv - self.mu) / self.sd
        W1 = rng.normal(0, 0.1, (X.shape[1], hidden)).astype(np.float32)
        W2 = rng.normal(0, 0.1, (hidden, 1)).astype(np.float32)
        b1 = np.zeros(hidden, np.float32)
        b2 = np.zeros(1, np.float32)
        best, patience, save = -1, 0, None
        for _ in range(epochs):
            H = np.maximum(Xa @ W1 + b1, 0)
            P = 1 / (1 + np.exp(-(H @ W2 + b2).clip(-30, 30)))[:, 0]
            D = (P - Y)[:, None]
            g2 = H.T @ D / len(Y) + 1e-4 * W2
            d1 = (D @ W2.T) * (H > 0)
            g1 = Xa.T @ d1 / len(Y) + 1e-4 * W1
            W2 -= lr * g2
            b2 -= lr * D.mean(0)
            W1 -= lr * g1
            b1 -= lr * d1.mean(0)
            a = auc(self._p(Xva, W1, b1, W2, b2), Yv)
            if a > best + 1e-4:
                best, patience = a, 0
                save = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            else:
                patience += 1
                if patience > 60:
                    break
        (self.W1, self.b1, self.W2, self.b2) = save
        self.val_auc = best
        return self

    @staticmethod
    def _p(X, W1, b1, W2, b2):
        H = np.maximum(X @ W1 + b1, 0)
        return (1 / (1 + np.exp(-(H @ W2 + b2).clip(-30, 30))))[:, 0]

    def predict(self, X):
        return self._p((X - self.mu) / self.sd, self.W1, self.b1, self.W2, self.b2)


def policy(convs, L, rho, w, I, cap, prob_at, noisy_oracle_eps=None, rng=None):
    """prob_at(ev, i, t) -> P(return in the saved window). Terminal idle periods are pinged until the cap."""
    cost = pings = waste = 0
    for conv in convs:
        ev = conv['events']
        for i in range(len(ev)):
            nxt = ev[i + 1]['t'] if i + 1 < len(ev) else None
            g = nxt - ev[i]['t'] if nxt is not None else np.inf
            shared = ev[i]['blocks']
            if nxt is not None and g <= L:
                continue
            expiry, t, n = L, I, 0
            while t < g and n < cap:
                if noisy_oracle_eps is None:
                    p = prob_at(ev, i, t)
                else:  # the bridging oracle's answer, flipped with probability eps: keep the chain alive while
                    # the return is beyond the current coverage and bridging to it costs less than the miss
                    if nxt is None:
                        truth = 0.0
                    else:
                        needed = max(1, -(-(nxt - expiry) // I))
                        truth = 1.0 if needed * rho <= w else 0.0
                    p = 1 - truth if rng.random() < noisy_oracle_eps else truth
                if p * w < rho:
                    break
                cost += shared * (rho if t < expiry else w)  # after death a ping re-writes
                expiry = t + L
                pings += 1
                if nxt is None:
                    waste += 1
                n += 1
                t += I
            if nxt is not None and g > expiry:
                cost += shared * w
    return cost


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for tname, fname, stretch in TRACES:
        convs = [c for c in base.conversations(load_trace(fname), stretch) if len(c['events']) > 1]
        convs.sort(key=lambda c: c['events'][0]['t'])
        train, test = convs[::2], convs[1::2]
        print(f"\n{tname}: {len(convs)} conversations", flush=True)
        entry = dict(conversations=len(convs), endpoints={})
        for label, L, rho, w in ENDPOINTS:
            I = int(0.9 * L)
            cap = max(0, int(w / rho) - 1)
            Xtr, Ytr, _ = dataset(train, I, cap)
            nv = len(Xtr) // 5
            models = [Logistic().fit(Xtr[nv:], Ytr[nv:], Xtr[:nv], Ytr[:nv]),
                      MLP().fit(Xtr[nv:], Ytr[nv:], Xtr[:nv], Ytr[:nv])]
            Xte, Yte, _ = dataset(test, I, cap)
            miss = policy(test, L, rho, w, I, cap, lambda ev, i, t: 0.0)
            oracle = policy(test, L, rho, w, I, cap, lambda ev, i, t: 0.0, noisy_oracle_eps=0.0, rng=rng)
            e = dict(rows=int(len(Xtr)), base_rate=round(float(Yte.mean()), 3), miss=miss, oracle=oracle, models={})
            age0 = np.array([r[0] == np.log1p(I) for r in Xte])  # the first epoch of each idle period
            for m in models:
                p = m.predict(Xte)
                a, brier = auc(p, Yte), float(((p - Yte) ** 2).mean())
                a0 = auc(p[age0], Yte[age0]) if age0.any() else None
                cal = [(round(float(lo), 1), round(float(Yte[(p >= lo) & (p < lo + .1)].mean()), 3),
                        int(((p >= lo) & (p < lo + .1)).sum()))
                       for lo in np.arange(0, .9, .1) if ((p >= lo) & (p < lo + .1)).sum() > 50]
                def cost_at(thr, _m=m):
                    return policy(test, L, rho, w, I, cap,
                                  lambda ev, i, t: float(_m.predict(np.asarray([epoch_row(ev, i, t)], np.float32))[0]) * w >= thr)
                # pick the threshold on a held-out slice of TRAIN (never on test)
                def val_cost(thr, _m=m):
                    return policy(train[-len(test) if len(test) < len(train) else len(train) // 5:], L, rho, w, I, cap,
                                  lambda ev, i, t: float(_m.predict(np.asarray([epoch_row(ev, i, t)], np.float32))[0]) * w >= thr)
                thresholds = [rho, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                thr_costs = {thr: val_cost(thr) for thr in thresholds}
                best_thr = min(thr_costs, key=thr_costs.get)
                cost = cost_at(best_thr)
                e['models'][m.name] = dict(auc=round(float(a), 3), auc_at_idle_start=round(float(a0), 3) if a0 else None,
                                           brier=round(brier, 4), cost=cost, threshold=best_thr, calibration=cal)
            sweep = []
            for eps in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4):
                # AUC of the degraded oracle, measured on the same rows the policy queries
                truth = np.array([1.0 if y else 0.0 for y in Yte], np.float32)
                flip = (rng.random(len(Yte)) < eps).astype(np.float32)
                noisy = np.abs(truth - flip)
                sweep.append(dict(eps=eps, auc=round(auc(noisy, Yte), 3),
                                  cost=policy(test, L, rho, w, I, cap, None, noisy_oracle_eps=eps, rng=rng)))
            e['degraded_oracle'] = sweep
            entry['endpoints'][label] = e
            mm = e['models']
            print(f"  {label:10s} rows {e['rows']}, base rate {e['base_rate']:.3f} | miss {miss} oracle {oracle} | "
                  + " ".join(f"{k}: AUC {v['auc']} thr {v['threshold']:.2f} cost {v['cost']}" for k, v in mm.items())
                  + " | sweep(AUC->cost): " + " ".join(f"{s['auc']:.2f}->{s['cost']:.0f}" for s in sweep))
        out[tname] = entry
    (ROOT / 'reports/cache-ping-predict-v2-2026-11-19.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
