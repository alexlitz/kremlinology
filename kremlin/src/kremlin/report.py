"""Markdown helpers."""


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    out += ['| ' + ' | '.join('' if c is None else str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def pct(x, digits=0):
    return '—' if x is None else f'{100 * x:.{digits}f}%'


def interval(p):
    """Clopper-Pearson dict -> '72% [68, 76] (424/586)'."""
    if not p or not p['n']:
        return '—'
    return f"{100 * p['fraction']:.0f}% [{100 * p['lower']:.0f}, {100 * p['upper']:.0f}] ({p['successes']}/{p['n']})"
