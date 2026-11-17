"""Small, dependency-free statistics used by the investigations.

- partial_tv_bounds: total-variation bounds between two partially reported distributions
  (top-k logprobs), allowing each reported logprob to be off by +-error nats.
- average_interval: distribution-free confidence interval for a dataset average of per-prompt
  [lower, upper] intervals (Waudby-Smith & Ramdas betting bound).
- proportion_interval: Clopper-Pearson interval.
- bootstrap_ratio_ci: percentile bootstrap CI for a ratio of medians.
"""
import math
import random
import statistics


def partial_tv_bounds(left, right, error=0.0):
    """Bounds on TV(P, Q) when only top-k log-probabilities of each are reported.

    Each reported value is trusted to +-error nats; unreported tokens share the remaining mass.
    Returns (lower, upper) or None if the reports are inconsistent (mass > 1).
    """
    boxes = [{k: (math.exp(v - error), math.exp(min(0.0, v + error))) for k, v in law.items()} for law in (left, right)]
    masses = [math.fsum(v[0] for v in box.values()) for box in boxes]
    if any(m > 1 for m in masses):
        return None
    forward, reverse = [], []
    for k in set(left) | set(right):
        a = boxes[0].get(k, (0.0, 1 - masses[0]))
        b = boxes[1].get(k, (0.0, 1 - masses[1]))
        forward.append(max(0.0, a[0] - b[1]))
        reverse.append(max(0.0, b[0] - a[1]))
    lower = max(math.fsum(forward), math.fsum(reverse))
    upper = 1 - math.fsum(min(boxes[0][k][0], boxes[1][k][0]) for k in set(left) & set(right))
    return lower, max(0.0, min(1.0, upper))


def same_law(left, right, tolerance=2e-3):
    """Bit-identical-ish: same reported tokens, every logprob within tolerance."""
    return bool(left and right) and set(left) == set(right) and all(abs(left[k] - right[k]) <= tolerance for k in left)


def _betting_lower(values, alpha, grid=2000):
    threshold = math.log(1 / alpha)
    lambdas, mean, square = [], 0.5, 0.25
    for t, x in enumerate(values, 1):
        variance = max(square - mean * mean, 1e-6) if t > 1 else 0.25
        lambdas.append(math.sqrt(2 * threshold / (variance * t * math.log(t + 1))))
        mean = (mean * t + x) / (t + 1)
        square = (square * t + x * x) / (t + 1)
    for step in range(1, grid + 1):
        m = step / grid
        cap, wealth = 0.5 / m, 0.0
        for x, lam in zip(values, lambdas):
            wealth += math.log1p(min(lam, cap) * (x - m))
            if wealth >= threshold:
                break
        else:
            return (step - 1) / grid
    return 1.0


def average_interval(lowers, uppers, alpha=0.05):
    """(1-alpha) interval for the dataset average, from per-prompt intervals in [0, 1]."""
    n = len(lowers)
    if n < 2 or n != len(uppers):
        return None
    return dict(prompts=n, mean_lower=math.fsum(lowers) / n, mean_upper=math.fsum(uppers) / n,
                ci_lower=max(0.0, _betting_lower(list(lowers), alpha / 2)),
                ci_upper=min(1.0, 1 - _betting_lower([1 - u for u in uppers], alpha / 2)), alpha=alpha)


def _tail(k, n, p, upper):
    if p <= 0:
        return 1.0 if (not upper or k <= 0) else 0.0
    if p >= 1:
        return 1.0 if (upper or k >= n) else 0.0
    terms = range(k, n + 1) if upper else range(0, k + 1)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
            + (n - i) * math.log1p(-p) for i in terms]
    if not logs:
        return 0.0
    peak = max(logs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(v - peak) for v in logs))


def proportion_interval(successes, n, alpha=0.05):
    if n == 0:
        return dict(successes=0, n=0, fraction=None, lower=0.0, upper=1.0)

    def solve(ok):
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2
            low, high = (middle, high) if ok(middle) else (low, middle)
        return (low + high) / 2
    lower = 0.0 if successes == 0 else solve(lambda p: _tail(successes, n, p, True) < alpha / 2)
    upper = 1.0 if successes == n else solve(lambda p: _tail(successes, n, p, False) >= alpha / 2)
    return dict(successes=successes, n=n, fraction=successes / n, lower=lower, upper=upper)


def bootstrap_ratio_ci(a, b, iterations=2000, seed=0, alpha=0.05):
    if len(a) < 3 or len(b) < 3:
        return None
    generator = random.Random(seed)
    ratios = sorted(statistics.median(generator.choices(a, k=len(a))) / statistics.median(generator.choices(b, k=len(b)))
                    for _ in range(iterations))
    return (statistics.median(a) / statistics.median(b), ratios[int(alpha / 2 * iterations)],
            ratios[int((1 - alpha / 2) * iterations) - 1])


def chi2_survival(statistic, freedom):
    """P(X >= statistic) for X ~ chi-square(freedom): regularized upper incomplete gamma Q(k/2, x/2)."""
    if freedom <= 0:
        return None
    a, x = freedom / 2, statistic / 2
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P, then Q = 1 - P
        term = total = 1 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if term < total * 1e-15:
                break
        return max(0.0, 1 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1 / 1e-300, 1 / (x + 1 - a)  # Lentz continued fraction for Q
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def multinomial_gof(counts, probabilities, min_expected=5.0):
    """Chi-square goodness of fit of observed counts to probabilities; small cells are pooled."""
    n = sum(counts)
    if n == 0:
        return None
    cells, pool_obs, pool_exp = [], 0, 0.0
    for c, p in zip(counts, probabilities):
        if n * p >= min_expected:
            cells.append((c, n * p))
        else:
            pool_obs, pool_exp = pool_obs + c, pool_exp + n * p
    if pool_exp > 0:
        cells.append((pool_obs, pool_exp))
    if len(cells) < 2:
        return None
    statistic = sum((o - e) ** 2 / e for o, e in cells if e > 0)
    return dict(statistic=statistic, freedom=len(cells) - 1, p_value=chi2_survival(statistic, len(cells) - 1))


def temperature_mle(logprobs, counts, low=0.05, high=5.0):
    """Temperature that best explains counts given reference log-probabilities (at T=1) on the same support."""
    observed = [(l, c) for l, c in zip(logprobs, counts)]
    if sum(c for _, c in observed) == 0 or len({l for l, c in observed if c}) < 2:
        return None

    def loglik(t):
        scaled = [l / t for l, _ in observed]
        peak = max(scaled)
        norm = peak + math.log(sum(math.exp(v - peak) for v in scaled))
        return sum(c * (v - norm) for v, (_, c) in zip(scaled, observed))
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if loglik(math.exp(x1)) > loglik(math.exp(x2)):
            b, x2 = x2, x1
            x1 = b - ratio * (b - a)
        else:
            a, x1 = x1, x2
            x2 = a + ratio * (b - a)
    return math.exp((a + b) / 2)
