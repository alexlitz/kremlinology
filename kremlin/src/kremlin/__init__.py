"""Black-box forensics for LLM inference APIs."""
__version__ = '0.3.2'


def http_client(out, witness=None, **kwargs):
    """httpx.Client that sends every request through a witness relay and records evidence (see kremlin.session)."""
    from .session import http_client as make
    return make(out, witness, **kwargs)


def async_http_client(out, witness=None, **kwargs):
    from .session import async_http_client as make
    return make(out, witness, **kwargs)
