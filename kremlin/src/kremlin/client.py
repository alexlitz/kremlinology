"""One request path for every investigation: budget, retries, evidence, optional witness.

Results are normalised across APIs (OpenRouter / OpenAI-compatible chat completions and
Anthropic Messages) so investigations never parse provider JSON themselves.
"""
import base64
import itertools
import json
import os
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from .evidence import Evidence

RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


class BudgetExceeded(Exception):
    pass


@dataclass
class Result:
    route: str
    ok: bool
    status: int | None
    text: str = ''
    reasoning: str = ''
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: str = 'unknown'
    first_token_logprobs: dict | None = None   # {token: logprob} for the first generated token
    chosen_first: str | None = None             # the sampled first token (string), when logprobs are returned
    choice_texts: list = field(default_factory=list)  # every choice's text (n > 1)
    tokens: list = field(default_factory=list)  # [(token text, logprob)] for every generated token, if reported
    response_id: str | None = None
    served_by: str | None = None               # OpenRouter's reported provider
    model: str | None = None
    elapsed_s: float | None = None
    token_times_s: list = field(default_factory=list)  # stream: arrival times of token-bearing events
    witnessed: bool = False
    error: str | None = None
    capture: str | None = None
    raw: dict | None = None


class Client:
    """Every call has a key. Investigations pass deterministic keys (e.g. 'share:AB:2:read') so calls can run in
    parallel and still be matched exactly by `audit` replay and by resume; unkeyed calls get 'seq:N'."""

    def __init__(self, evidence: Evidence, *, max_usd, max_requests=10_000, witness=None, retries=4, backoff_s=2.0,
                 min_interval_s=0.0, timeout_s=180, transport=None, trust_file=None, disclose_requests=True,
                 concurrency=8, resume_records=None):
        self.evidence = evidence
        self.max_usd, self.max_requests = max_usd, max_requests
        self.witness = witness
        self.disclose = disclose_requests
        self.retries, self.backoff_s, self.min_interval_s = retries, backoff_s, min_interval_s
        self.concurrency = max(1, concurrency)
        self.http = httpx.Client(timeout=timeout_s, transport=transport, trust_env=False,
                                 limits=httpx.Limits(max_connections=self.concurrency * 2),
                                 verify=ssl.create_default_context(cafile=trust_file) if trust_file else True)
        self.spent = 0.0
        self.unknown_cost = 0
        self.requests = 0
        self.resumed = 0
        self._next_slot = 0.0
        self._lock = threading.RLock()
        self._seq = itertools.count()
        self.completed = completed_calls(resume_records or [], self.retries)
        self.prior_attempts = {}  # keys attempted before an interruption but not completed: continue numbering
        for record in resume_records or []:
            if record.get('key') and record['key'] not in self.completed:
                self.prior_attempts[record['key']] = max(self.prior_attempts.get(record['key'], -1),
                                                         record.get('attempt', 0))

    def key(self, prefix='seq'):
        with self._lock:
            return f'{prefix}:{next(self._seq)}'

    def parallel(self, tasks, workers=None):
        """Run zero-argument callables concurrently; results in task order. A budget stop cancels what has not
        started and is re-raised after running tasks finish."""
        workers = workers or self.concurrency
        if workers <= 1 or len(tasks) <= 1:
            return [task() for task in tasks]
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(task) for task in tasks]
            results, error = [], None
            for future in futures:
                try:
                    results.append(future.result())
                except BudgetExceeded as stop:
                    error = error or stop
                    pool.shutdown(wait=True, cancel_futures=True)
                    results.append(None)
            if error:
                raise error
            return results

    # ---- request shaping -----------------------------------------------------------------
    @staticmethod
    def _anthropic(route):
        return route.kind == 'anthropic'

    def _build(self, route, messages, *, max_tokens, temperature, top_logprobs, stream, extra, cap_field,
               require_parameters, key=None):
        if key is None:
            key = os.environ.get(route.key_env)
            if not key:
                raise RuntimeError(f'{route.key_env} is not set (key for route {route.label})')
        if self._anthropic(route):
            body = dict(model=route.model, max_tokens=max_tokens, temperature=temperature, messages=messages,
                        **(dict(stream=True) if stream else {}), **(extra or {}))
            headers = {'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
            return f'{route.base_url}/messages', headers, body, key
        body = route.body(messages, max_tokens=max_tokens, temperature=temperature, top_logprobs=top_logprobs,
                          stream=stream, extra=extra, cap_field=cap_field, require_parameters=require_parameters)
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        return f'{route.base_url}/chat/completions', headers, body, key

    def _admit(self):
        """Budget check and pacing slot, atomically; returns how long to wait before sending."""
        with self._lock:
            if self.spent >= self.max_usd:
                raise BudgetExceeded(f'spent ${self.spent:.4f} of ${self.max_usd} budget')
            if self.requests >= self.max_requests:
                raise BudgetExceeded(f'request limit {self.max_requests} reached')
            self.requests += 1
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval_s
            return slot - now

    # ---- sending ----------------------------------------------------------------------------
    def chat(self, route, messages, *, max_tokens=1, temperature=0.0, top_logprobs=None, stream=False, extra=None,
             cap_field='max_tokens', require_parameters=False, tag=None, key=None):
        call_key = key or self.key()
        url, headers, body, api_key = self._build(route, messages, max_tokens=max_tokens, temperature=temperature,
                                                  top_logprobs=top_logprobs, stream=stream, extra=extra,
                                                  cap_field=cap_field, require_parameters=require_parameters)
        with self._lock:
            if api_key not in self.evidence.secrets:
                self.evidence.secrets.append(api_key)
        if call_key in self.completed:
            return self._resume(route, call_key, body, stream)
        payload = json.dumps(body).encode()
        attempt, offset = 0, self.prior_attempts.get(call_key, -1) + 1
        while True:
            wait = self._admit()
            if wait > 0:
                time.sleep(wait)
            result, raw_body, receipt = self._send_once(route, url, headers, payload, stream)
            with self._lock:
                result = self._account(route, result, body)
            record = dict(kind='chat', key=call_key, route=route.label, model=route.model, provider=route.provider,
                          url=url, request=body, attempt=offset + attempt, tag=tag or {}, sent_at=time.time(),
                          result={k: v for k, v in result.__dict__.items() if k != 'raw'})
            if receipt:
                record['witness_receipt'] = receipt
            result.capture = self.evidence.record(record, raw_body)
            retryable = (not result.ok and (result.status in RETRY_STATUSES or result.status is None)
                         and attempt < self.retries)
            if not retryable:
                return result
            attempt += 1
            time.sleep(self.backoff_s * 2 ** (attempt - 1))

    def _resume(self, route, call_key, body, stream):
        """Serve a call that a previous (interrupted) run already completed, from its evidence."""
        chain = self.completed[call_key]
        final = chain[-1]
        if final.get('request') != json.loads(json.dumps(body)):
            raise RuntimeError(f'resume: call {call_key} would send a different request than the recorded one')
        raw = self.evidence.capture(final['capture'])
        status = final['result'].get('status')
        result = parse(route, status, raw, stream) if status is not None else Result(route.label, False, None)
        saved = final['result']
        result.elapsed_s, result.token_times_s = saved.get('elapsed_s'), saved.get('token_times_s') or []
        result.witnessed, result.capture = saved.get('witnessed', False), final['capture']
        with self._lock:
            self.resumed += 1
            for record in chain:
                cost = (record.get('result') or {}).get('cost_usd')
                if cost is not None:
                    self.spent += cost
            result.cost_usd, result.cost_source = saved.get('cost_usd'), saved.get('cost_source', 'unknown')
        return result

    def _send_once(self, route, url, headers, payload, stream):
        start = time.time()
        receipt = None
        try:
            if self.witness:
                response = self.witness.request('POST', url, headers=headers, body=payload,
                                                disclose_request=self.disclose)
                status, raw_body, receipt = response.status, response.body, response.receipt
                events = response.timings.get('witness_sse_events') or []
                times = [t / 1000 for t, data in events if _token_bearing(data)]
                elapsed = time.time() - start
            elif stream:
                times, chunks = [], []
                with self.http.stream('POST', url, headers=headers, content=payload) as response:
                    status = response.status_code
                    for line in response.iter_lines():
                        chunks.append(line)
                        if line.startswith('data: ') and line != 'data: [DONE]' and _token_bearing(line[6:]):
                            times.append(time.time() - start)
                raw_body = '\n'.join(chunks).encode()
                elapsed = time.time() - start
            else:
                response = self.http.post(url, headers=headers, content=payload)
                status, raw_body, times = response.status_code, response.content, []
                elapsed = time.time() - start
        except (httpx.HTTPError, OSError, Exception) as error:  # witness errors included
            return Result(route.label, False, None, error=f'{type(error).__name__}: {error}',
                          elapsed_s=time.time() - start), b'', None
        result = parse(route, status, raw_body, stream)
        result.elapsed_s = witnessed_latency(receipt) if receipt else elapsed
        result.witnessed = receipt is not None
        if stream and not self.witness:
            result.token_times_s = times
        elif receipt:
            result.token_times_s = times  # witness record arrival times (relative to tunnel open)
        return result, raw_body, receipt

    def _account(self, route, result, body):
        if result.cost_usd is not None:
            result.cost_source = 'reported'
        elif route.price and result.prompt_tokens is not None:
            result.cost_usd = (result.prompt_tokens * route.price[0] + (result.completion_tokens or 0)
                               * route.price[1]) / 1e6
            result.cost_source = 'estimated'
        if result.cost_usd is not None:
            self.spent += result.cost_usd
        elif result.ok:
            self.unknown_cost += 1
        return result

    def request(self, route, method, path, *, json_body=None, files=None, data=None, headers=None, tag=None,
                key=None):
        """Any endpoint under the route's base URL (embeddings, images, audio, responses, ...), witnessed if
        a witness is configured. files/data build a multipart body exactly as httpx would send it."""
        key = os.environ.get(route.key_env)
        if not key:
            raise RuntimeError(f'{route.key_env} is not set (key for route {route.label})')
        if key not in self.evidence.secrets:
            self.evidence.secrets.append(key)
        auth = ({'x-api-key': key, 'anthropic-version': '2023-06-01'} if self._anthropic(route)
                else {'Authorization': f'Bearer {key}'})
        url = f"{route.base_url}/{path.lstrip('/')}"
        built = httpx.Request(method, url, json=json_body, files=files, data=data, headers={**auth, **(headers or {})})
        payload = built.read()
        send_headers = {k: v for k, v in built.headers.items() if k.lower() not in ('host', 'content-length')}
        call_key = key or self.key('request')
        self._admit()
        start, receipt = time.time(), None
        if self.witness:
            response = self.witness.request(method, url, headers=send_headers, body=payload,
                                            disclose_request=self.disclose)
            status, body, receipt = response.status, response.body, response.receipt
        else:
            response = self.http.request(method, url, headers=send_headers, content=payload)
            status, body = response.status_code, response.content
        record = dict(kind='request', key=call_key, route=route.label, url=url, method=method, tag=tag or {},
                      status=status,
                      elapsed_s=time.time() - start, request_json=json_body,
                      request_files=sorted(files) if files else None, request_bytes=len(payload))
        if receipt:
            record['witness_receipt'] = receipt
        self.evidence.record(record, body)
        return status, body, receipt

    def generation(self, route, generation_id, tries=4, delay_s=3.0, key=None):
        """OpenRouter generation record (provider, upstream id, native tokens, cost). Free lookup; recorded,
        and witnessed when a witness is configured, so audits can replay it."""
        if not route.is_openrouter or not generation_id:
            return None
        call_key = key or f'gen:{generation_id}'
        if call_key in self.completed:
            final = self.completed[call_key][-1]
            return extract_generation(self.evidence.capture(final['capture'])) if final.get('status') == 200 else None
        key = os.environ.get(route.key_env)
        url = f'{route.base_url}/generation?id={generation_id}'
        for attempt in range(tries):
            receipt = None
            if self.witness:
                try:
                    response = self.witness.request('GET', url, headers={'Authorization': f'Bearer {key}'},
                                                    disclose_request=self.disclose)
                    status, raw, receipt = response.status, response.body, response.receipt
                except Exception as error:
                    status, raw = None, str(error).encode()
            else:
                response = self.http.get(url, headers={'Authorization': f'Bearer {key}'})
                status, raw = response.status_code, response.content
            record = dict(kind='generation', key=call_key, route=route.label, url=url, generation_id=generation_id,
                          status=status, attempt=attempt)
            if receipt:
                record['witness_receipt'] = receipt
            self.evidence.record(record, raw)
            if status == 200:
                return extract_generation(raw)
            time.sleep(delay_s)
        return None

    def summary(self):
        return dict(spent_usd=round(self.spent, 6), requests=self.requests, unknown_cost_responses=self.unknown_cost,
                    resumed_calls=self.resumed, max_usd=self.max_usd)


def completed_calls(records, retries):
    """key -> attempt chain, for calls whose last attempt was final (success, or a non-retryable failure)."""
    chains = {}
    for record in records:
        if record.get('key'):
            chains.setdefault(record['key'], []).append(record)
    done = {}
    for key, chain in chains.items():
        chain.sort(key=lambda r: r.get('attempt', 0))
        last = chain[-1]
        status = last.get('status') if last.get('kind') in ('generation', 'request') else (last.get('result') or {}).get('status')
        ok = status == 200 and (last.get('kind') != 'chat' or (last.get('result') or {}).get('ok'))
        if ok or (status not in RETRY_STATUSES and status is not None) or last.get('attempt', 0) >= retries:
            done[key] = chain
    return done


GENERATION_FIELDS = ('provider_name', 'upstream_id', 'latency', 'generation_time', 'native_tokens_prompt',
                     'native_tokens_completion', 'native_tokens_reasoning', 'native_tokens_cached', 'finish_reason',
                     'native_finish_reason', 'total_cost', 'upstream_inference_cost')


def extract_generation(raw):
    data = (json.loads(raw).get('data') or {}) if raw else {}
    return {k: data.get(k) for k in GENERATION_FIELDS}


def witnessed_latency(receipt):
    """Time to first response byte as the witness saw it: last request byte -> first response record."""
    records = receipt.get('response_records') or []
    if not records or receipt.get('request_complete_ms') is None:
        return None
    return (records[0]['t_ms'] - receipt['request_complete_ms']) / 1000


def _token_bearing(data):
    try:
        obj = json.loads(data)
    except ValueError:
        return False
    for choice in obj.get('choices') or []:
        delta = choice.get('delta') or {}
        if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')):
            return True
    return obj.get('type') == 'content_block_delta'


def _sse_events(raw):
    for line in raw.decode('utf-8', 'replace').splitlines():
        if line.startswith('data: ') and line[6:].strip() != '[DONE]':
            try:
                yield json.loads(line[6:])
            except ValueError:
                continue


def parse(route, status, raw, stream):
    """Normalise a chat response (plain or SSE) into a Result."""
    result = Result(route.label, status == 200, status)
    try:
        if stream:
            events = list(_sse_events(raw))
            obj = dict(choices=[], usage=None)
            text, reasoning, logprob_first, usage = '', '', None, None
            for event in events:
                if route.kind == 'anthropic':
                    if event.get('type') == 'content_block_delta':
                        text += (event.get('delta') or {}).get('text') or ''
                    if event.get('type') in ('message_start', 'message_delta'):
                        usage = {**(usage or {}), **((event.get('message') or {}).get('usage') or event.get('usage') or {})}
                    continue
                obj.setdefault('id', event.get('id'))
                obj['provider'] = event.get('provider') or obj.get('provider')
                obj['model'] = event.get('model') or obj.get('model')
                usage = event.get('usage') or usage
                for choice in event.get('choices') or []:
                    delta = choice.get('delta') or {}
                    text += delta.get('content') or ''
                    reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                    result.finish_reason = choice.get('finish_reason') or result.finish_reason
                    content = (choice.get('logprobs') or {}).get('content') or []
                    if content and logprob_first is None:
                        logprob_first = content[0]
            obj['usage'] = usage
            result.text, result.reasoning = text, reasoning
            if logprob_first:
                result.first_token_logprobs = _law(logprob_first)
        else:
            obj = json.loads(raw) if raw else {}
            if route.kind == 'anthropic':
                result.text = ''.join(b.get('text', '') for b in obj.get('content') or [] if b.get('type') == 'text')
                result.finish_reason = obj.get('stop_reason')
            elif obj.get('choices'):
                result.choice_texts = [(c.get('message') or {}).get('content') or '' for c in obj['choices']]
                choice = obj['choices'][0]
                message = choice.get('message') or {}
                result.text = message.get('content') or ''
                result.reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
                result.finish_reason = choice.get('finish_reason')
                content = (choice.get('logprobs') or {}).get('content') or []
                if content:
                    result.first_token_logprobs = _law(content[0])
                    result.chosen_first = _key(content[0])
                    result.tokens = [(c.get('token'), c.get('logprob')) for c in content]
        result.raw = obj
        result.response_id = obj.get('id')
        result.served_by = obj.get('provider')
        result.model = obj.get('model')
        usage = obj.get('usage') or {}
        if route.kind == 'anthropic':
            result.prompt_tokens = (usage.get('input_tokens') or 0) + (usage.get('cache_read_input_tokens') or 0) \
                + (usage.get('cache_creation_input_tokens') or 0) if usage else None
            result.completion_tokens = usage.get('output_tokens')
            result.cached_tokens = usage.get('cache_read_input_tokens')
            result.cache_write_tokens = usage.get('cache_creation_input_tokens')
        else:
            details = usage.get('prompt_tokens_details') or {}
            result.prompt_tokens = usage.get('prompt_tokens')
            result.completion_tokens = usage.get('completion_tokens')
            result.reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            result.cached_tokens = details.get('cached_tokens')
            result.cache_write_tokens = details.get('cache_write_tokens')
            if isinstance(usage.get('cost'), (int, float)):
                result.cost_usd = float(usage['cost'])
        if obj.get('error'):
            result.ok = False
            result.error = str(obj['error'])[:500]
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result.ok = False
        result.error = f'unparseable response: {error}'
    return result


def _key(entry):
    raw = entry.get('bytes')
    return bytes(raw).hex() if isinstance(raw, list) and raw else 's:' + str(entry.get('token'))


def _law(entry):
    """First-token alternatives as {key: logprob}; keys are token bytes (hex) when the API gives bytes."""
    out = {}
    for alt in entry.get('top_logprobs') or []:
        if isinstance(alt.get('logprob'), (int, float)):
            out[_key(alt)] = float(alt['logprob'])
    return out or None
