"""Shared interval helpers for figures: Wilson score interval for proportions, percentile bootstrap for statistics."""
import math
import random


def wilson(hits, n, z=1.96):
    """95% Wilson score interval for hits/n (good at small n and at 0% / 100%)."""
    if not n:
        return None, None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap(values, stat, iterations=2000, seed=0, level=0.95):
    """Percentile bootstrap interval of stat(values)."""
    values = list(values)
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(values) for _ in values]) for _ in range(iterations))
    lo = draws[int((1 - level) / 2 * iterations)]
    hi = draws[int((1 + level) / 2 * iterations) - 1]
    return lo, hi


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
