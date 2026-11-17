"""Finite-sample bounds on dataset averages of per-prompt bounded quantities.

Each prompt i drawn uniformly at random from a dataset contributes an interval
[lower_i, upper_i] in [0, 1] known to contain its quantity (for example
first-token TV under a declared report-error allowance). The dataset average
mu then satisfies E[lower] <= mu <= E[upper]; one-sided confidence bounds on
those two means bound mu. Bounds are distribution-free and finite-sample:
Waudby-Smith-Ramdas betting by default, or the tighter of Hoeffding and
Maurer-Pontil empirical Bernstein. They assume prompts drawn independently and
uniformly from the dataset and are conditional on the per-prompt interval
contract.
They describe that dataset and request format, not other traffic.
"""

import math


def _check(values):
    values = [float(v) for v in values]
    if len(values) < 2 or any(not 0 <= v <= 1 or not math.isfinite(v) for v in values):
        raise ValueError("at least two values in [0, 1] required")
    return values


def hoeffding_radius(n, alpha):
    return math.sqrt(math.log(1 / alpha) / (2 * n))


def bernstein_radius(values, alpha):
    """Maurer and Pontil (2009) Theorem 4, one-sided, for [0, 1] values."""
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    log = math.log(2 / alpha)
    return math.sqrt(2 * variance * log / n) + 7 * log / (3 * (n - 1))


def betting_lower(values, alpha, grid=2000):
    """Waudby-Smith and Ramdas (2023) predictable-plug-in betting lower bound.

    For each candidate mean m, a gambler bets lambda_i * (x_i - m) with
    lambda_i chosen from earlier observations only; under mean <= m the
    wealth is a nonnegative supermartingale, so m is rejected when wealth
    reaches 1/alpha (Ville). The bound is the smallest unrejected grid point.
    Valid in any fixed order; time-uniform, hence conservative at fixed n.
    """
    threshold = math.log(1 / alpha)
    lambdas, mean, square = [], 0.5, 0.25
    for t, x in enumerate(values, 1):
        variance = max(square - mean * mean, 1e-6) if t > 1 else 0.25
        lambdas.append(math.sqrt(2 * threshold / (variance * t * math.log(t + 1))))
        mean = (mean * t + x) / (t + 1)
        square = (square * t + x * x) / (t + 1)
    for step in range(grid + 1):
        m = step / grid
        if m == 0:
            continue
        cap = 0.5 / m
        wealth, rejected = 0.0, False
        for x, lam in zip(values, lambdas):
            wealth += math.log1p(min(lam, cap) * (x - m))
            if wealth >= threshold:
                rejected = True
                break
        if not rejected:
            return (step - 1) / grid
    return 1.0


def mean_bound(values, alpha, side, method="betting"):
    """One-sided (1 - alpha) bound on the mean of [0, 1] values.

    betting: Waudby-Smith-Ramdas; classical: tighter of Hoeffding and
    Maurer-Pontil empirical Bernstein at alpha/2 each.
    """
    values = _check(values)
    if not 0 < alpha < 1 or side not in ("lower", "upper"):
        raise ValueError("alpha in (0, 1) and side lower/upper required")
    if method == "betting":
        if side == "lower":
            return max(0.0, betting_lower(values, alpha))
        return min(1.0, 1 - betting_lower([1 - v for v in values], alpha))
    if method != "classical":
        raise ValueError("method must be betting or classical")
    mean = math.fsum(values) / len(values)
    radius = min(hoeffding_radius(len(values), alpha / 2), bernstein_radius(values, alpha / 2))
    return max(0.0, mean - radius) if side == "lower" else min(1.0, mean + radius)


def average_interval(lowers, uppers, alpha=0.05, method="betting"):
    """Simultaneous (1 - alpha) interval for the dataset average of the bounded quantity."""
    lowers, uppers = _check(lowers), _check(uppers)
    if len(lowers) != len(uppers) or any(a > b + 1e-12 for a, b in zip(lowers, uppers)):
        raise ValueError("aligned intervals with lower <= upper required")
    n = len(lowers)
    return dict(
        prompts=n,
        mean_lower=math.fsum(lowers) / n,
        mean_upper=math.fsum(uppers) / n,
        confidence_lower=mean_bound(lowers, alpha / 2, "lower", method),
        confidence_upper=mean_bound(uppers, alpha / 2, "upper", method),
        alpha=alpha,
        method=method,
    )


def _binomial_tail(k, n, p, upper):
    """P(X >= k) if upper else P(X <= k) for X ~ Binomial(n, p), in log space."""
    if p <= 0:
        return 1.0 if (not upper or k <= 0) else 0.0
    if p >= 1:
        return 1.0 if (upper or k >= n) else 0.0
    terms = range(k, n + 1) if upper else range(0, k + 1)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in terms]
    if not logs:
        return 0.0
    peak = max(logs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(v - peak) for v in logs))


def proportion_interval(successes, n, alpha=0.05):
    """Clopper-Pearson two-sided (1 - alpha) interval by bisection."""
    if type(successes) is not int or type(n) is not int or not 0 <= successes <= n or n < 1:
        raise ValueError("integer 0 <= successes <= n required")

    def solve(target_ok):
        low, high = 0.0, 1.0
        for _ in range(100):
            middle = (low + high) / 2
            low, high = (middle, high) if target_ok(middle) else (low, middle)
        return (low + high) / 2
    lower = 0.0 if successes == 0 else solve(
        lambda p: _binomial_tail(successes, n, p, True) < alpha / 2)
    upper = 1.0 if successes == n else solve(
        lambda p: _binomial_tail(successes, n, p, False) >= alpha / 2)
    return dict(successes=successes, n=n, fraction=successes / n, lower=lower, upper=upper,
                alpha=alpha)
