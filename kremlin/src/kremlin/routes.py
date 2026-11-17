"""Route specs: which model, served by whom, with which key.

  openrouter:MODEL@PROVIDER   OpenRouter, pinned to one provider endpoint (no fallbacks)
  openrouter:MODEL            OpenRouter default routing
  openai:BASE_URL|MODEL       any OpenAI-compatible chat completions API
  anthropic:MODEL             Anthropic Messages API (key default ANTHROPIC_API_KEY)

Options follow, separated by ';':
  key=ENV_VAR     environment variable holding the API key
                  (defaults: OPENROUTER_API_KEY, ANTHROPIC_API_KEY; required for openai routes)
  label=NAME      short name used in reports (default: provider tag or model)
  price=IN/OUT    USD per million input/output tokens, used when the API reports no cost

Example: 'openrouter:qwen/qwen3.5-397b-a17b@parasail/fp8;label=parasail'
"""
from dataclasses import dataclass, field

OPENROUTER_URL = 'https://openrouter.ai/api/v1'


@dataclass(frozen=True)
class Route:
    kind: str
    model: str
    base_url: str
    provider: str | None = None
    key_env: str = 'OPENROUTER_API_KEY'
    label: str = ''
    price: tuple | None = None
    options: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_openrouter(self):
        return self.kind == 'openrouter'

    def body(self, messages, *, max_tokens, temperature=0.0, top_logprobs=None, stream=False, extra=None,
             cap_field='max_tokens', require_parameters=False):
        body = {'model': self.model, 'messages': messages, cap_field: max_tokens, 'temperature': temperature}
        if top_logprobs:
            body.update(logprobs=True, top_logprobs=top_logprobs)
        if stream:
            body.update(stream=True, stream_options={'include_usage': True})
        if self.is_openrouter and self.provider:
            body['provider'] = {'only': [self.provider], 'allow_fallbacks': False}
            if require_parameters:
                body['provider']['require_parameters'] = True
        body.update(extra or {})
        return body


def parse_route(spec):
    head, *parts = [p.strip() for p in spec.split(';')]
    options = {}
    for part in parts:
        if '=' not in part:
            raise ValueError(f'route option {part!r} must be key=value')
        key, value = part.split('=', 1)
        options[key.strip()] = value.strip()
    unknown = set(options) - {'key', 'label', 'price'}
    if unknown:
        raise ValueError(f'unknown route options: {sorted(unknown)}')
    if ':' not in head:
        raise ValueError(f'route {spec!r} needs a scheme: openrouter:... or openai:...')
    kind, rest = head.split(':', 1)
    price = None
    if 'price' in options:
        try:
            price = tuple(float(x) for x in options['price'].split('/'))
            assert len(price) == 2 and min(price) >= 0
        except (ValueError, AssertionError):
            raise ValueError('price must be IN/OUT in USD per million tokens') from None
    if kind == 'openrouter':
        model, _, provider = rest.partition('@')
        if not model:
            raise ValueError('openrouter route needs a model')
        label = options.get('label') or provider or model.split('/')[-1]
        return Route('openrouter', model, OPENROUTER_URL, provider or None, options.get('key', 'OPENROUTER_API_KEY'),
                     label, price, options)
    if kind == 'openai':
        base, sep, model = rest.rpartition('|')
        if not sep or not base.startswith(('http://', 'https://')) or not model:
            raise ValueError('openai route must be openai:https://host/v1|model')
        if 'key' not in options:
            raise ValueError('openai routes need key=ENV_VAR')
        label = options.get('label') or f"{base.split('//')[1].split('/')[0]}:{model.split('/')[-1]}"
        return Route('openai', model, base.rstrip('/'), None, options['key'], label, price, options)
    if kind == 'anthropic':
        if not rest:
            raise ValueError('anthropic route needs a model')
        return Route('anthropic', rest, 'https://api.anthropic.com/v1', None, options.get('key', 'ANTHROPIC_API_KEY'),
                     options.get('label') or rest, price, options)
    raise ValueError(f'unknown route scheme {kind!r}')


def unique_labels(routes):
    labels = [r.label for r in routes]
    if len(set(labels)) != len(labels):
        raise ValueError(f'route labels must be unique: {labels} (add ;label=NAME)')
    return routes
