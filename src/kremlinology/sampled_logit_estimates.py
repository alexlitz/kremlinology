"""Logit-difference estimates from sampled tokens under declared truncation.

Samples identify a sampling law, so logits are identified only up to an
additive constant and, unless the temperature is trusted, a scale. Every
estimate here assumes the provider applies the declared temperature, top-K
support and top-P cutoff exactly, in the order temperature then top-K then
top-P, with no hidden penalties. Those assumptions must be checked with
canaries; nothing here detects silently ignored parameters.
"""
import math
import numpy as np

# argmax of x^2 sigma(x)(1-sigma(x)): Fisher information about a logit gap
# from one Bernoulli sample at temperature T is maximal at gap/T = this value.
_BEST_SCALED_GAP = 2.3994


def _counts(counts):
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or len(values) < 2 or (values < 0).any() or not np.isfinite(values).all():
        raise ValueError('at least two finite nonnegative counts required')
    if (values != np.round(values)).any():
        raise ValueError('integer counts required')
    return values


def _temperature(temperature):
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    return float(temperature)


def truncated_logit_differences(counts, *, temperature, reference=0, pseudocount=0.5):
    """Delta-method logit gaps from one support set, e.g. a top-K=len(counts) run.

    Returns z_i - z_reference estimates and standard errors. Zero counts are
    smoothed by the pseudocount and flagged; their estimates are bounds at
    best, not measurements.
    """
    values = _counts(counts)
    temperature = _temperature(temperature)
    if type(reference) is not int or not 0 <= reference < len(values):
        raise ValueError('reference index out of range')
    if not math.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError('finite nonnegative pseudocount required')
    smoothed = values + pseudocount
    if (smoothed == 0).any():
        raise ValueError('zero counts need a positive pseudocount')
    difference = temperature * (np.log(smoothed) - math.log(smoothed[reference]))
    error = temperature * np.sqrt(1 / smoothed + 1 / smoothed[reference])
    error[reference] = 0.
    return dict(difference=difference, standard_error=error, unobserved=values == 0,
                samples=int(values.sum()))


def pooled_logit_fit(observations, *, tokens=None, iterations=100, tolerance=1e-10):
    """Joint multinomial maximum likelihood over several truncated runs.

    Each observation is (token_ids, counts, temperature): the exact support
    the sampler drew from (for example the reference top-K set) and counts on
    it. Logits are fixed by setting the first token's logit to zero. Tokens
    never co-occurring in a connected support graph are not identified and
    raise. Covariance is the inverse observed information.
    """
    rows = []
    for token_ids, counts, temperature in observations:
        values = _counts(counts)
        ids = list(token_ids)
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError('one count per distinct support token required')
        rows.append((ids, values, _temperature(temperature)))
    if not rows:
        raise ValueError('at least one observation required')
    if tokens is None:
        tokens = list(dict.fromkeys(token for ids, _, _ in rows for token in ids))
    index = {token: i for i, token in enumerate(tokens)}
    size = len(tokens)
    # Union-find connectivity over supports; disconnected gaps are unidentified.
    parent = list(range(size))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for ids, _, _ in rows:
        for token in ids[1:]:
            parent[find(index[token])] = find(index[ids[0]])
    if len({find(i) for i in range(size)}) != 1:
        raise ValueError('support graph is disconnected; gaps are not identified')
    def log_likelihood(values_at):
        total = 0.
        for ids, values, temperature in rows:
            scaled = values_at[[index[token] for token in ids]] / temperature
            total += float(values @ (scaled - np.logaddexp.reduce(scaled)))
        return total

    def derivatives(values_at):
        gradient = np.zeros(size)
        information = np.zeros((size, size))
        for ids, values, temperature in rows:
            where = np.array([index[token] for token in ids])
            scaled = values_at[where] / temperature
            probability = np.exp(scaled - np.logaddexp.reduce(scaled))
            total = values.sum()
            gradient[where] += (values - total * probability) / temperature
            block = total * (np.diag(probability) - np.outer(probability, probability))
            information[np.ix_(where, where)] += block / temperature**2
        return gradient, information

    # Start from smoothed log counts (exact for one run) and damp Newton steps:
    # undamped steps from zero overshoot on long-tailed counts and go singular.
    pooled = np.zeros(size)
    for ids, values, temperature in rows:
        for token, value in zip(ids, values):
            pooled[index[token]] += value
    logits = np.mean([t for _, _, t in rows]) * np.log(pooled + .5)
    logits -= logits[0]
    current = log_likelihood(logits)
    for _ in range(iterations):
        gradient, information = derivatives(logits)
        # Pin token 0; the likelihood is invariant to a common shift.
        step = np.zeros(size)
        if size > 1:
            step[1:] = np.linalg.solve(information[1:, 1:] + 1e-12 * np.eye(size - 1), gradient[1:])
        scale = 1.
        while scale > 1e-6:
            candidate = logits + scale * step
            value = log_likelihood(candidate)
            if np.isfinite(value) and value >= current - 1e-12:
                break
            scale /= 2
        logits, current = candidate, value
        if np.abs(scale * step).max() < tolerance:
            break
    _, information = derivatives(logits)
    covariance = np.zeros((size, size))
    if size > 1:
        covariance[1:, 1:] = np.linalg.inv(information[1:, 1:])
    unobserved = [token for token in tokens
                  if not any(values[ids.index(token)] for ids, values, _ in rows if token in ids)]
    return dict(tokens=list(tokens), logits=logits, standard_error=np.sqrt(np.diag(covariance)),
                covariance=covariance, unobserved=unobserved,
                scope='logits up to a shift under declared sampler semantics')


def pair_design(gap, *, maximum_temperature=2., target_standard_error=None):
    """Temperature and samples for a top-K=2 estimate of a positive logit gap.

    Information per sample about the gap is x^2 s(1-s) / gap^2 with x=gap/T
    and s=sigmoid(x); it peaks near x=2.4 (p ~ 0.92), not at p=0.5, because
    hotter sampling also shrinks sensitivity to the gap.
    """
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError('positive finite gap required')
    maximum_temperature = _temperature(maximum_temperature)
    temperature = min(gap / _BEST_SCALED_GAP, maximum_temperature)
    scaled = gap / temperature
    share = 1 / (1 + math.exp(-scaled))
    information = share * (1 - share) / temperature**2
    result = dict(temperature=temperature, second_token_share=1 - share,
                  information_per_sample=information,
                  temperature_capped=temperature < gap / _BEST_SCALED_GAP)
    if target_standard_error is not None:
        if not math.isfinite(target_standard_error) or target_standard_error <= 0:
            raise ValueError('positive target standard error required')
        result['samples'] = math.ceil(1 / (information * target_standard_error**2))
    return result


def nucleus_cutoff_bracket(probes, *, minimum_share, alpha=0.01):
    """Bracket the higher-ranked tempered mass C before a target token.

    Probes are (top_p, samples, target_seen) at one temperature and top-K.
    Under inclusive nucleus semantics the target is in support iff C < top_p.
    Seeing it proves C < top_p. Not seeing it in m samples excludes it only if
    its in-support share is at least minimum_share, with miss probability
    (1-minimum_share)^m <= alpha per probe (no multiplicity correction).
    """
    if not 0 < minimum_share <= 1 or not 0 < alpha < 1:
        raise ValueError('share in (0,1] and alpha in (0,1) required')
    upper, lower = 1., 0.
    for top_p, samples, seen in probes:
        if not 0 < top_p <= 1 or type(samples) is not int or samples < 0:
            raise ValueError('top_p in (0,1] and integer samples required')
        if seen:
            upper = min(upper, top_p)
        elif samples and (1 - minimum_share)**samples <= alpha:
            lower = max(lower, top_p)
    return dict(lower=lower, upper=upper, consistent=lower < upper,
                scope='requires exact nucleus semantics and a justified share bound')


def clustered_overdispersion(counts, *, permutations=2000, seed=0):
    """Within-request versus between-request spread for n>1 requests.

    Rows are requests, columns tokens, entries counts of the n choices. If
    every request shares one fixed logit vector and its n choices are
    independent, rows are multinomial and the dispersion is about one.
    Excess means varying logits between requests, or dependent choices within
    a request (for example a reused seed); this does not say which. A high
    fraction of single-token rows is the signature to check for the latter.
    """
    table = np.asarray(counts, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2 or (table < 0).any():
        raise ValueError('at least two requests and two tokens required')
    if (table != np.round(table)).any() or (table.sum(1) < 2).any():
        raise ValueError('integer counts with at least two choices per request required')
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        raise ValueError('at least two observed tokens required')
    sizes = table.sum(1).astype(int)

    def statistic(values):
        expected = np.outer(values.sum(1), values.sum(0)) / values.sum()
        return float(((values - expected)**2 / expected).sum())
    observed = statistic(table)
    freedom = (table.shape[0] - 1) * (table.shape[1] - 1)
    dispersion = observed / freedom
    # Permute individual choices across requests, preserving request sizes.
    labels = np.repeat(np.arange(table.shape[1]), table.sum(0).astype(int))
    rows = np.repeat(np.arange(table.shape[0]), sizes)
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = np.zeros_like(table)
        np.add.at(shuffled, (rows, generator.permutation(labels)), 1)
        exceed += statistic(shuffled) >= observed
    mean_size = sizes.mean()
    correlation = (dispersion - 1) / (mean_size - 1) if mean_size > 1 else math.nan
    return dict(statistic=observed, degrees_of_freedom=freedom, dispersion=dispersion,
                intraclass_correlation=correlation,
                permutation_pvalue=(exceed + 1) / (permutations + 1),
                single_token_rows=float(((table > 0).sum(1) == 1).mean()))


def temperature_from_reference(reference_logits, counts, *, low=1e-3, high=100.):
    """Maximum-likelihood applied temperature given trusted reference logits.

    Pooled form of Naseh et al. (2023) Theorem 1, tau = log(p_i/p_j) /
    log(p'_i/p'_j): truncation renormalization cancels, so reference logits
    need only cover the observed support. Assumes the reference model is the
    served model; a mismatch is absorbed into a biased temperature.
    """
    logits = np.asarray(reference_logits, dtype=float)
    values = _counts(counts)
    if logits.shape != values.shape or not np.isfinite(logits).all():
        raise ValueError('finite reference logits aligned with counts required')
    observed = logits[values > 0]
    if len(observed) < 2 or np.ptp(observed) == 0:
        raise ValueError('two observed tokens with distinct logits required')

    def score(inverse):
        scaled = logits * inverse
        probability = np.exp(scaled - np.logaddexp.reduce(scaled))
        return float(values @ logits - values.sum() * probability @ logits)
    # The score is decreasing in inverse temperature; bisect in log space.
    left, right = math.log(1 / high), math.log(1 / low)
    if score(math.exp(left)) < 0 or score(math.exp(right)) > 0:
        return dict(temperature=None, at_bound=True)
    for _ in range(200):
        middle = (left + right) / 2
        left, right = (middle, right) if score(math.exp(middle)) > 0 else (left, middle)
    inverse = math.exp((left + right) / 2)
    scaled = logits * inverse
    probability = np.exp(scaled - np.logaddexp.reduce(scaled))
    variance = probability @ logits**2 - (probability @ logits)**2
    information = values.sum() * variance
    return dict(temperature=1 / inverse, at_bound=False,
                standard_error=(1 / inverse**2) / math.sqrt(information) if information > 0 else math.inf)
