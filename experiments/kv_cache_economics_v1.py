"""KV-cache bytes per token by attention design, and the economics of caching it.

Inputs (no inference, no API calls):
  configs/kv_models/*.json   model configs copied from the local HF cache (snapshot IDs in names)
  reports/kv-cache-economics/openrouter-endpoints-2026-11-11.json  public catalog (cached-read prices)
  data/kremlinology-llama70b-dataset-tv-v2  observed prefix-cache counters (optional)
Outputs in reports/kv-cache-economics/: PNG figures, CSV tables, results JSON.

Model. For context length N, cache bytes per token = c_full + c_window*min(N,W)/N + S/N,
where full-attention layers store every token, sliding-window layers store the last W, and
linear-attention/SSM layers keep a constant per-sequence state S. KV in bf16 (fp8 variant);
recurrent states fp32, conv states bf16. Economics per tier (HBM, host DRAM, NVMe): holding
cost = bytes x $/GB-hour; one-off load = bytes / bandwidth x GPU $/s (conservative: not
overlapped); recompute = prefill FLOPs / (peak x MFU) x GPU $/s. Break-even retention time is
(recompute - load) / holding rate. Hardware numbers are assumptions with low/high bands, not
measurements; active-parameter counts are approximate published figures.
"""
import csv
import json
import math
from pathlib import Path
import os as _os
DATA_ROOT = _os.environ.get('KREMLINOLOGY_DATA', 'data')  # where the release archives are extracted

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False  # labels contain literal dollar signs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/kv-cache-economics'
CONFIGS = ROOT / 'configs/kv_models'

# Reference palette (dataviz skill, light mode); validated for these slot subsets.
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FAMILY = {  # fixed slot order, never cycled
    'Dense GQA': '#2a78d6',
    'MLA (latent)': '#eb6834',
    'Sliding / chunked window + global': '#1baf7a',
    'Linear attention + global': '#eda100',
    'SSM (Mamba) / hybrid': '#e87ba4',
}
TIER_COLOR = {'HBM': '#2a78d6', 'Host DRAM': '#eb6834', 'NVMe SSD': '#1baf7a'}

# Hardware assumptions: (baseline, low, high).
GPU_DOLLARS_PER_HOUR = (2.50, 1.50, 4.00)      # H100 SXM rental
GPU_PEAK_FLOPS = 989e12                         # bf16 dense
MFU = (0.45, 0.60, 0.30)                        # prefill model-FLOP utilization (low cost = high MFU)
HBM_GB = 80
TIERS = {  # $/GB-hour (baseline, low, high), bandwidth to HBM in bytes/s
    'HBM': dict(cost=None, bandwidth=None),
    'Host DRAM': dict(cost=(1e-3, 2e-4, 5e-3), bandwidth=50e9),
    'NVMe SSD': dict(cost=(3e-5, 5e-6, 1.4e-4), bandwidth=10e9),
}

ACTIVE_PARAMS = {  # approximate published active parameters
    'meta-llama--Llama-3.1-8B-Instruct': 8.0e9, 'unsloth--Llama-3.3-70B-Instruct': 70.6e9,
    'Qwen--Qwen3-8B': 8.2e9, 'Qwen--Qwen3-32B': 32.8e9, 'Qwen--Qwen3-235B-A22B': 22e9,
    'Qwen--Qwen3-30B-A3B': 3.3e9, 'Qwen--Qwen2.5-7B-Instruct': 7.6e9,
    'mistralai--Mistral-7B-Instruct-v0.3': 7.2e9, 'mistralai--Mistral-Small-3.1-24B-Instruct-2503': 24e9,
    'zai-org--GLM-4.5': 32e9, 'zai-org--GLM-4.5-Air': 12e9, 'deepseek-ai--DeepSeek-V3': 37e9,
    'deepseek-ai--DeepSeek-V3.2-Exp': 37e9, 'moonshotai--Kimi-K2-Instruct': 32e9,
    'openai--gpt-oss-120b': 5.1e9, 'openai--gpt-oss-20b': 3.6e9, 'google--gemma-3-27b-it': 27e9,
    'google--gemma-3-12b-it': 12e9, 'Qwen--Qwen3-Next-80B-A3B-Instruct': 3e9, 'Qwen--Qwen3.5-9B': 9e9,
    'Qwen--Qwen3.5-27B': 27e9, 'MiniMaxAI--MiniMax-M1-80k': 45.9e9,
    'moonshotai--Kimi-Linear-48B-A3B-Instruct': 3e9, 'ibm-granite--granite-4.0-h-small': 9e9,
    'nvidia--NVIDIA-Nemotron-Nano-9B-v2': 9e9, 'tiiuae--Falcon-H1-7B-Instruct': 7.6e9,
    'state-spaces--mamba-2.8b-hf': 2.8e9,
    'google--gemma-4-31B-it': 30.7e9, 'google--gemma-4-26B-A4B-it': 3.8e9, 'google--gemma-3-4b-it': 4.3e9,
    'Qwen--Qwen3.5-35B-A3B': 3e9, 'Qwen--Qwen3.5-122B-A10B': 10e9, 'Qwen--Qwen3.5-397B-A17B': 17e9,
    'meta-llama--Llama-3.1-70B-Instruct': 70.6e9, 'deepseek-ai--DeepSeek-V3.1': 37e9,
    'deepseek-ai--DeepSeek-V3-0324': 37e9, 'deepseek-ai--DeepSeek-V3.2': 37e9,
    'moonshotai--Kimi-K2-Instruct-0905': 32e9, 'moonshotai--Kimi-K2-Thinking': 32e9, 'moonshotai--Kimi-K2.5': 32e9,
    'ibm-granite--granite-4.2-8b': 8e9, 'Qwen--Qwen3-235B-A22B-Instruct-2507': 22e9,
    'Qwen--Qwen3-30B-A3B-Instruct-2507': 3.3e9, 'meta-llama--Llama-4-Maverick-17B-128E-Instruct': 17e9,
    'meta-llama--Llama-4-Scout-17B-16E-Instruct': 17e9, 'mistralai--Mistral-Small-3.2-24B-Instruct-2506': 24e9,
    'mistralai--Mistral-Nemo-Instruct-2407': 12.2e9, 'Qwen--Qwen2.5-72B-Instruct': 72.7e9,
    'Qwen--Qwen2.5-Coder-32B-Instruct': 32.8e9,
}
PRICE_IDS = {  # config name -> OpenRouter model ID (endpoints fetched 2026-11-11)
    'unsloth--Llama-3.3-70B-Instruct': 'meta-llama/llama-3.3-70b-instruct',
    'meta-llama--Llama-3.1-8B-Instruct': 'meta-llama/llama-3.1-8b-instruct',
    'meta-llama--Llama-3.1-70B-Instruct': 'meta-llama/llama-3.1-70b-instruct',
    'Qwen--Qwen3-235B-A22B-Instruct-2507': 'qwen/qwen3-235b-a22b-2507',
    'Qwen--Qwen3-30B-A3B-Instruct-2507': 'qwen/qwen3-30b-a3b-instruct-2507', 'Qwen--Qwen3-32B': 'qwen/qwen3-32b',
    'deepseek-ai--DeepSeek-V3.2': 'deepseek/deepseek-v3.2', 'deepseek-ai--DeepSeek-V3.2-Exp': 'deepseek/deepseek-v3.2-exp',
    'deepseek-ai--DeepSeek-V3-0324': 'deepseek/deepseek-chat-v3-0324', 'deepseek-ai--DeepSeek-V3.1': 'deepseek/deepseek-chat-v3.1',
    'moonshotai--Kimi-K2-Instruct-0905': 'moonshotai/kimi-k2-0905', 'moonshotai--Kimi-K2-Thinking': 'moonshotai/kimi-k2-thinking',
    'moonshotai--Kimi-K2.5': 'moonshotai/kimi-k2.5', 'openai--gpt-oss-20b': 'openai/gpt-oss-20b',
    'openai--gpt-oss-120b': 'openai/gpt-oss-120b', 'google--gemma-3-4b-it': 'google/gemma-3-4b-it',
    'google--gemma-3-12b-it': 'google/gemma-3-12b-it', 'google--gemma-3-27b-it': 'google/gemma-3-27b-it',
    'google--gemma-4-31B-it': 'google/gemma-4-31b-it', 'google--gemma-4-26B-A4B-it': 'google/gemma-4-26b-a4b-it',
    'meta-llama--Llama-4-Maverick-17B-128E-Instruct': 'meta-llama/llama-4-maverick',
    'meta-llama--Llama-4-Scout-17B-16E-Instruct': 'meta-llama/llama-4-scout',
    'Qwen--Qwen3-Next-80B-A3B-Instruct': 'qwen/qwen3-next-80b-a3b-instruct', 'Qwen--Qwen3.5-27B': 'qwen/qwen3.5-27b',
    'Qwen--Qwen3.5-9B': 'qwen/qwen3.5-9b', 'Qwen--Qwen3.5-35B-A3B': 'qwen/qwen3.5-35b-a3b',
    'Qwen--Qwen3.5-122B-A10B': 'qwen/qwen3.5-122b-a10b', 'Qwen--Qwen3.5-397B-A17B': 'qwen/qwen3.5-397b-a17b',
    'mistralai--Mistral-Small-3.1-24B-Instruct-2503': 'mistralai/mistral-small-3.1-24b-instruct',
    'mistralai--Mistral-Small-3.2-24B-Instruct-2506': 'mistralai/mistral-small-3.2-24b-instruct',
    'mistralai--Mistral-Nemo-Instruct-2407': 'mistralai/mistral-nemo', 'Qwen--Qwen2.5-72B-Instruct': 'qwen/qwen-2.5-72b-instruct',
    'Qwen--Qwen2.5-Coder-32B-Instruct': 'qwen/qwen-2.5-coder-32b-instruct', 'zai-org--GLM-4.5': 'z-ai/glm-4.5',
    'zai-org--GLM-4.5-Air': 'z-ai/glm-4.5-air', 'ibm-granite--granite-4.2-8b': 'ibm-granite/granite-4.2-8b',
}
PRICE_FILE = 'openrouter-endpoints-2026-11-11b.json'


def endpoint_prices(identifier, catalog):
    """Cheapest and median paid uncached and cached-read prices ($/token) across endpoints."""
    entry = catalog.get(identifier, {})
    endpoints = entry.get('body', {}).get('data', {}).get('endpoints', []) if entry.get('status') == 200 else []
    prompt = sorted(float(e['pricing']['prompt']) for e in endpoints if float(e['pricing'].get('prompt') or 0) > 0)
    cached = sorted(float(e['pricing']['input_cache_read']) for e in endpoints
                    if e['pricing'].get('input_cache_read') and float(e['pricing']['input_cache_read']) > 0)
    median = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return dict(endpoints=len(endpoints), prompt_min=prompt[0] if prompt else None, prompt_median=median(prompt),
                cached_min=cached[0] if cached else None, cached_median=median(cached), cached_endpoints=len(cached))


CATALOG_IDS = {  # OpenRouter IDs for models with per-endpoint cached-read prices
    'unsloth--Llama-3.3-70B-Instruct': 'meta-llama/llama-3.3-70b-instruct',
    'meta-llama--Llama-3.1-8B-Instruct': 'meta-llama/llama-3.1-8b-instruct',
    'Qwen--Qwen3-235B-A22B': 'qwen/qwen3-235b-a22b-2507', 'Qwen--Qwen3-32B': 'qwen/qwen3-32b',
    'Qwen--Qwen3-30B-A3B': 'qwen/qwen3-30b-a3b-instruct-2507', 'deepseek-ai--DeepSeek-V3.2-Exp': 'deepseek/deepseek-v3.2',
    'deepseek-ai--DeepSeek-V3': 'deepseek/deepseek-chat-v3-0324', 'openai--gpt-oss-120b': 'openai/gpt-oss-120b',
    'openai--gpt-oss-20b': 'openai/gpt-oss-20b', 'google--gemma-3-27b-it': 'google/gemma-3-27b-it',
    'google--gemma-3-12b-it': 'google/gemma-3-12b-it', 'Qwen--Qwen3-Next-80B-A3B-Instruct': 'qwen/qwen3-next-80b-a3b-instruct',
    'Qwen--Qwen3.5-27B': 'qwen/qwen3.5-27b', 'Qwen--Qwen3.5-9B': 'qwen/qwen3.5-9b',
    'zai-org--GLM-4.5': 'z-ai/glm-4.5', 'zai-org--GLM-4.5-Air': 'z-ai/glm-4.5-air',
}


def load(name):
    path = next(CONFIGS.glob(f'{name}@*.json'))
    config = json.loads(path.read_text())
    if 'text_config' in config:
        config = {**config, **config['text_config']}
    return config, path.name.split('@')[1].removesuffix('.json')


def spec(name, kv_bytes=2):
    """Per-token full-attention bytes, sliding bytes and window, fixed state bytes, attention FLOP terms."""
    c, snapshot = load(name)
    L = c.get('num_hidden_layers') or c.get('n_layer')
    hidden = c.get('hidden_size')
    heads = c.get('num_attention_heads')
    head_dim = c.get('head_dim') or (hidden // heads if heads else None)
    kv_heads = c.get('num_key_value_heads')
    per_layer = lambda h, d: 2 * h * d * kv_bytes  # noqa: E731  (K and V)
    s = dict(model=name, snapshot=snapshot, layers=L, full_bytes=0.0, window_bytes=0.0, window=0,
             state_bytes=0.0, attn_full=0.0, attn_window=0.0, linear_flops=0.0)
    if 'kv_lora_rank' in c and 'linear_attn_config' not in c:  # MLA
        s['family'] = 'MLA (latent)'
        s['full_bytes'] = L * (c['kv_lora_rank'] + c['qk_rope_head_dim']) * kv_bytes
        if 'index_head_dim' in c:  # DeepSeek V3.2 sparse-attention indexer keys (fp8)
            s['full_bytes'] += L * c['index_head_dim'] * 1
        s['attn_full'] = L * heads * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'] + c['v_head_dim'])
    elif 'linear_attn_config' in c:  # Kimi-Linear: MLA global layers + KDA linear layers
        s['family'] = 'Linear attention + global'
        la = c['linear_attn_config']
        full, kda = len(la['full_attn_layers']), len(la['kda_layers'])
        s['full_bytes'] = full * (c['kv_lora_rank'] + c['qk_rope_head_dim']) * kv_bytes
        s['attn_full'] = full * heads * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'] + c['v_head_dim'])
        width = la['num_heads'] * la['head_dim']
        s['state_bytes'] = kda * (la['num_heads'] * la['head_dim'] ** 2 * 4
                                  + 3 * width * (la['short_conv_kernel_size'] - 1) * 2)
        s['linear_flops'] = kda * 6 * la['num_heads'] * la['head_dim'] ** 2
    elif 'full_attention_interval' in c or c.get('model_type', '').startswith('qwen3_5'):  # Gated DeltaNet hybrids
        s['family'] = 'Linear attention + global'
        types = c.get('layer_types') or ['full_attention' if (i + 1) % c['full_attention_interval'] == 0
                                         else 'linear_attention' for i in range(L)]
        full, linear = types.count('full_attention'), types.count('linear_attention')
        s['full_bytes'] = full * per_layer(kv_heads, head_dim)
        s['attn_full'] = full * heads * 2 * head_dim
        dk, dv, hk, hv = (c['linear_key_head_dim'], c['linear_value_head_dim'],
                          c['linear_num_key_heads'], c['linear_num_value_heads'])
        conv = (2 * hk * dk + hv * dv) * (c['linear_conv_kernel_dim'] - 1) * 2
        s['state_bytes'] = linear * (hv * dk * dv * 4 + conv)
        s['linear_flops'] = linear * 6 * hv * dk * dv
    elif 'attn_type_list' in c:  # MiniMax: 0 = lightning (linear), 1 = softmax
        s['family'] = 'Linear attention + global'
        full, linear = c['attn_type_list'].count(1), c['attn_type_list'].count(0)
        s['full_bytes'] = full * per_layer(kv_heads, head_dim)
        s['attn_full'] = full * heads * 2 * head_dim
        s['state_bytes'] = linear * heads * head_dim ** 2 * 4
        s['linear_flops'] = linear * 6 * heads * head_dim ** 2
    elif c.get('model_type') in ('granitemoehybrid', 'nemotron_h', 'falcon_h1') or 'hybrid_override_pattern' in c:
        s['family'] = 'SSM (Mamba) / hybrid'
        if 'hybrid_override_pattern' in c:  # Nemotron-H
            pattern = c['hybrid_override_pattern']
            attention, mamba = pattern.count('*'), pattern.count('M')
            n_heads, d_head, d_state, groups = c['mamba_num_heads'], c['mamba_head_dim'], c['ssm_state_size'], c['n_groups']
            kernel = c['conv_kernel']
        elif c.get('model_type') == 'falcon_h1':  # parallel attention + Mamba in every layer
            attention = mamba = L
            n_heads, d_head, d_state, groups = c['mamba_n_heads'], c['mamba_d_head'], c['mamba_d_state'], c['mamba_n_groups']
            kernel = c['mamba_d_conv']
        else:  # Granite 4.0-H
            attention, mamba = c['layer_types'].count('attention'), c['layer_types'].count('mamba')
            n_heads, d_head, d_state, groups = c['mamba_n_heads'], c['mamba_d_head'], c['mamba_d_state'], c['mamba_n_groups']
            kernel = c['mamba_d_conv']
        s['full_bytes'] = attention * per_layer(kv_heads, head_dim)
        s['attn_full'] = attention * heads * 2 * head_dim
        conv = (n_heads * d_head + 2 * groups * d_state) * (kernel - 1) * 2
        s['state_bytes'] = mamba * (n_heads * d_head * d_state * 4 + conv)
        s['linear_flops'] = mamba * 6 * n_heads * d_head * d_state
    elif c.get('model_type') == 'mamba':
        s['family'] = 'SSM (Mamba) / hybrid'
        inner = c['expand'] * hidden
        s['state_bytes'] = L * (inner * c['state_size'] * 4 + inner * (c['conv_kernel'] - 1) * 2)
        s['linear_flops'] = L * 6 * inner * c['state_size']
    elif c.get('model_type') == 'gemma4_text':
        # Sliding layers: num_key_value_heads x head_dim. Global layers: num_global_key_value_heads x
        # global_head_dim, with values derived from the key projection (attention_k_eq_v); engines
        # still cache K and V separately (different norm/RoPE), so both are counted.
        s['family'] = 'Sliding / chunked window + global'
        types = c['layer_types']
        full, window = types.count('full_attention'), types.count('sliding_attention')
        global_heads, global_dim = c['num_global_key_value_heads'], c['global_head_dim']
        s.update(full_bytes=full * per_layer(global_heads, global_dim),
                 window_bytes=window * per_layer(kv_heads, head_dim), window=c['sliding_window'],
                 attn_full=full * heads * 2 * global_dim, attn_window=window * heads * 2 * head_dim,
                 k_eq_v_global=bool(c.get('attention_k_eq_v')))
    elif c.get('model_type') == 'llama4_text':
        # iRoPE: every 4th layer is global (NoPE); the rest attend within 8192-token chunks, so their
        # cache never exceeds one chunk (treated as a window of attention_chunk_size).
        s['family'] = 'Sliding / chunked window + global'
        full = sum(1 for i in range(L) if (i + 1) % 4 == 0)
        window = L - full
        s.update(full_bytes=full * per_layer(kv_heads, head_dim), window_bytes=window * per_layer(kv_heads, head_dim),
                 window=c['attention_chunk_size'], attn_full=full * heads * 2 * head_dim,
                 attn_window=window * heads * 2 * head_dim)
    elif c.get('layer_types') and 'sliding_attention' in c['layer_types']:  # gpt-oss
        s['family'] = 'Sliding / chunked window + global'
        full, window = c['layer_types'].count('full_attention'), c['layer_types'].count('sliding_attention')
        s.update(full_bytes=full * per_layer(kv_heads, head_dim), window_bytes=window * per_layer(kv_heads, head_dim),
                 window=c['sliding_window'], attn_full=full * heads * 2 * head_dim,
                 attn_window=window * heads * 2 * head_dim)
    elif c.get('model_type', '').startswith('gemma3'):
        s['family'] = 'Sliding / chunked window + global'
        head_dim = c.get('head_dim') or 256  # Gemma3TextConfig defaults (4B omits these)
        heads = c.get('num_attention_heads') or 8
        kv_heads = c.get('num_key_value_heads') or 4
        pattern = c.get('sliding_window_pattern', 6)
        full = sum(1 for i in range(L) if (i + 1) % pattern == 0)
        window = L - full
        s.update(full_bytes=full * per_layer(kv_heads, head_dim), window_bytes=window * per_layer(kv_heads, head_dim),
                 window=c['sliding_window'], attn_full=full * heads * 2 * head_dim,
                 attn_window=window * heads * 2 * head_dim)
    else:
        s['family'] = 'Dense GQA'
        s['full_bytes'] = L * per_layer(kv_heads, head_dim)
        s['attn_full'] = L * heads * 2 * head_dim
    s['active_params'] = ACTIVE_PARAMS[name]
    return s


def bytes_per_token(s, n):
    return s['full_bytes'] + s['window_bytes'] * min(n, s['window']) / n + s['state_bytes'] / n


def prefill_flops_per_token(s, n):
    """Average forward FLOPs per prefix token: weights + causal attention + linear-state updates."""
    attention = 2 * s['attn_full'] * n / 2  # QK^T and AV over the average causal span n/2
    if s['window']:
        span = n / 2 if n <= s['window'] else s['window'] - s['window'] ** 2 / (2 * n)
        attention += 2 * s['attn_window'] * span
    return 2 * s['active_params'] + attention + s['linear_flops']


def economics(s, n, case=0):
    """Dollars per token: recompute, and per tier (holding $/hour, one-off load), break-even hours."""
    gpu = GPU_DOLLARS_PER_HOUR[case] / 3600
    recompute = prefill_flops_per_token(s, n) / (GPU_PEAK_FLOPS * MFU[case]) * gpu
    size = bytes_per_token(s, n)
    result = dict(recompute=recompute, bytes=size, tiers={})
    for tier, t in TIERS.items():
        rate = (GPU_DOLLARS_PER_HOUR[case] / HBM_GB) if t['cost'] is None else t['cost'][case]
        hold = size / 1e9 * rate
        load = 0.0 if t['bandwidth'] is None else size / t['bandwidth'] * gpu
        result['tiers'][tier] = dict(hold_per_hour=hold, load=load,
                                     breakeven_hours=max(0.0, (recompute - load) / hold))
    return result


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', color=INK, fontsize=12, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def human_bytes(value):
    for unit, scale in (('MiB', 2 ** 20), ('KiB', 2 ** 10)):
        if value >= scale:
            return f'{value / scale:.0f} {unit}' if value / scale >= 10 else f'{value / scale:.1f} {unit}'
    return f'{value:.0f} B'


def byte_ticks(ax, axis, low, high):
    ticks = [2 ** k for k in range(0, 31, 2) if low <= 2 ** k <= high]
    labels = [human_bytes(t).replace('.0 ', ' ') for t in ticks]
    (ax.set_xticks if axis == 'x' else ax.set_yticks)(ticks, labels)


def short(name):
    return name.split('--', 1)[1].replace('-Instruct', '').replace('-it', '').replace('-hf', '')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))  # union, first-seen order
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figure_bytes(specs, n=32768):
    rows = sorted(specs, key=lambda s: bytes_per_token(s, n))
    fig, ax = plt.subplots(figsize=(9, 0.29 * len(rows) + 1.6))
    style(ax, f'KV-cache bytes per token at {n // 1024}k context (bf16 KV, fp32 recurrent state)',
          xlabel='bytes per token (log scale)')
    values = [bytes_per_token(s, n) for s in rows]
    ax.barh(range(len(rows)), values, color=[FAMILY[s['family']] for s in rows], height=0.72,
            edgecolor=SURFACE, linewidth=2)
    ax.set_xscale('log')
    ax.set_yticks(range(len(rows)), [short(s['model']) for s in rows], color=INK, fontsize=9)
    for i, v in enumerate(values):
        ax.text(v * 1.08, i, human_bytes(v), va='center', fontsize=8, color=INK2)
    ax.set_xlim(min(values) / 2, max(values) * 3)
    byte_ticks(ax, 'x', min(values) / 2, max(values) * 3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
    ax.legend(handles, FAMILY.keys(), loc='lower right', frameon=False, fontsize=9, labelcolor=INK)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_bytes_per_token_32k.png', dpi=160)
    plt.close(fig)
    write_csv(OUT / 'fig1_bytes_per_token_32k.csv',
              [dict(model=s['model'], snapshot=s['snapshot'], family=s['family'], context=n,
                    bytes_per_token_bf16=round(bytes_per_token(s, n)),
                    full_attention_bytes=round(s['full_bytes']), window_bytes=round(s['window_bytes']),
                    window=s['window'], fixed_state_bytes=round(s['state_bytes'])) for s in rows])


def figure_context(specs):
    picks = ['unsloth--Llama-3.3-70B-Instruct', 'deepseek-ai--DeepSeek-V3', 'google--gemma-3-27b-it',
             'Qwen--Qwen3.5-27B', 'ibm-granite--granite-4.0-h-small']
    by = {s['model']: s for s in specs}
    contexts = [2 ** k for k in range(10, 21)]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    style(ax, 'Bytes per token vs context length (one model per attention family)',
          xlabel='context length (tokens, log scale)', ylabel='bytes per token (log scale)')
    table = []
    for name in picks:
        s = by[name]
        ys = [bytes_per_token(s, n) for n in contexts]
        ax.plot(contexts, ys, color=FAMILY[s['family']], linewidth=2, marker='o', markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{short(name)} — {s['family']}")
        nudge = {'deepseek-ai--DeepSeek-V3': 1.07, 'Qwen--Qwen3.5-27B': 0.93}.get(name, 1.0)
        ax.text(contexts[-1] * 1.15, ys[-1] * nudge, short(name), va='center', fontsize=8, color=INK2)
        table += [dict(model=name, family=s['family'], context=n, bytes_per_token=round(y)) for n, y in zip(contexts, ys)]
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(contexts[0] / 1.3, contexts[-1] * 5)
    ax.set_xticks([2 ** k for k in range(10, 21, 2)],
                  [f'{2 ** k // 1024}k' if k < 20 else '1M' for k in range(10, 21, 2)])
    byte_ticks(ax, 'y', 4096, 2 ** 20)
    ax.set_ylim(6000, 900000)
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.45, -0.14), ncol=2,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_bytes_vs_context.png', dpi=160)
    plt.close(fig)
    write_csv(OUT / 'fig2_bytes_vs_context.csv', table)


def figure_breakeven(specs, n=32768):
    rows = sorted(specs, key=lambda s: economics(s, n)['tiers']['Host DRAM']['breakeven_hours'])
    fig, ax = plt.subplots(figsize=(9.5, 0.3 * len(rows) + 1.6))
    style(ax, f'Break-even retention time for a cached {n // 1024}k-context prefix',
          xlabel='retention time at which holding the cache costs as much as recomputing it (log scale)')
    table = []
    for i, s in enumerate(rows):
        for tier, offset in zip(TIERS, (-0.22, 0, 0.22)):
            base = economics(s, n)['tiers'][tier]['breakeven_hours']
            low = economics(s, n, 1)['tiers'][tier]['breakeven_hours']
            high = economics(s, n, 2)['tiers'][tier]['breakeven_hours']
            lo, hi = sorted((low, high))
            ax.plot([max(lo, 1e-4), max(hi, 1e-4)], [i + offset] * 2, color=TIER_COLOR[tier], linewidth=2, alpha=0.35)
            ax.plot(max(base, 1e-4), i + offset, 'o', color=TIER_COLOR[tier], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=tier if i == 0 else None)
            table.append(dict(model=s['model'], family=s['family'], tier=tier, context=n,
                              breakeven_hours=base, low_cost_case=low, high_cost_case=high))
    ax.set_xscale('log')
    marks = [(1 / 60, '1 min'), (5 / 60, '5 min'), (1, '1 h'), (24, '1 day'), (24 * 30, '30 days')]
    ax.set_xticks([h for h, _ in marks], [label for _, label in marks])
    ax.minorticks_off()
    ax.set_xlim(1 / 200, 24 * 90)
    ax.set_yticks(range(len(rows)), [short(s['model']) for s in rows], color=INK, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK, title='tier (dot = baseline, bar = low/high)',
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_breakeven_retention_32k.png', dpi=160)
    plt.close(fig)
    write_csv(OUT / 'fig3_breakeven_retention_32k.csv', table)


def figure_prices(specs, n=32768):
    catalog = json.loads((OUT / 'openrouter-endpoints-2026-11-11.json').read_text())
    by = {s['model']: s for s in specs}
    rows = []
    for name, identifier in CATALOG_IDS.items():
        endpoints = catalog[identifier]['body'].get('data', {}).get('endpoints', [])
        priced = [(float(e['pricing']['prompt']), float(e['pricing']['input_cache_read']))
                  for e in endpoints if e['pricing'].get('input_cache_read')]
        if not priced:
            continue
        econ = economics(by[name], n)
        hold_hour = econ['tiers']['Host DRAM']['hold_per_hour'] + econ['tiers']['Host DRAM']['load']
        rows.append(dict(model=name, catalog_id=identifier, endpoints_with_cache_price=len(priced),
                         min_input_price=min(p for p, _ in priced) * 1e6, min_cached_read=min(c for _, c in priced) * 1e6,
                         max_cached_read=max(c for _, c in priced) * 1e6,
                         recompute_cost=econ['recompute'] * 1e6, dram_one_hour_cost=hold_hour * 1e6))
    rows.sort(key=lambda r: r['min_cached_read'])
    series = [('min_input_price', 'Cheapest uncached input price', '#2a78d6', 'o'),
              ('min_cached_read', 'Cheapest cached-read price', '#eb6834', 'D'),
              ('recompute_cost', 'Estimated recompute cost (H100)', '#1baf7a', 's'),
              ('dram_one_hour_cost', 'Estimated cost to hold 1 h in DRAM + reload', '#eda100', '^')]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    style(ax, f'Cached-token prices vs estimated costs ({n // 1024}k-context prefix)',
          xlabel='$ per million tokens (log scale)')
    for i, r in enumerate(rows):
        ax.plot([r['min_cached_read'], r['max_cached_read']], [i, i], color='#eb6834', linewidth=3, alpha=0.3,
                solid_capstyle='round', label='Range of cached-read prices across providers' if i == 0 else None)
        for key, label, color, marker in series:
            ax.plot(r[key], i, marker, color=color, markersize=8 if marker == '^' else 7, markeredgecolor=SURFACE,
                    markeredgewidth=1.3, label=label if i == 0 else None)
    ax.set_xscale('log')
    ticks = [0.01, 0.03, 0.1, 0.3, 1]
    ax.set_xticks(ticks, [f'${t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)), [short(r['model']) for r in rows], color=INK, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.42, -0.09), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_cached_price_vs_cost.png', dpi=160)
    plt.close(fig)
    write_csv(OUT / 'fig4_cached_price_vs_cost.csv', rows)
    return rows


def figure_observed():
    run = Path(DATA_ROOT, 'kremlinology-llama70b-dataset-tv-v2/observations.jsonl')
    if not run.is_file():
        return None
    rows = [json.loads(line) for line in run.read_text().splitlines()]
    providers = [('coreweave-fp16', '#2a78d6'), ('akashml-fp8', '#eb6834'), ('parasail-fp8', '#1baf7a')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    summary = {}
    for ax, (target, color) in zip(axes, providers):
        points = [(r['usage']['prompt_tokens'], (r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
                  for r in rows if r['target_id'] == target]
        hits = [h for _, h in points if h]
        block = 0
        for h in hits:
            block = math.gcd(block, h)
        summary[target] = dict(requests=len(points), hits=len(hits), block_tokens=block)
        style(ax, target, xlabel='prompt tokens', ylabel='cached prompt tokens' if ax is axes[0] else None)
        xs = range(40, 201)
        ax.step(xs, [(x - 1) // block * block for x in xs], where='post', color=INK2, linewidth=1,
                linestyle=(0, (3, 2)), label=f'most cacheable with {block}-token blocks')
        ax.scatter([p for p, _ in points], [h for _, h in points], s=16, color=color, edgecolors=SURFACE,
                   linewidths=0.6, label=f'{len(hits)}/{len(points)} requests hit the cache')
        ax.legend(frameon=False, fontsize=8, loc='upper left', labelcolor=INK)
    fig.suptitle('Observed prefix-cache hits, Llama 3.3-70B via OpenRouter (2026-11-10)', x=0.01, ha='left',
                 color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_observed_cache_blocks.png', dpi=160)
    plt.close(fig)
    return summary


def log_fit(points):
    """Least-squares fit of log10(price) on log10(bytes); None without 3 distinct x values."""
    pts = [(math.log10(x), math.log10(y)) for x, y in points if x > 0 and y > 0]
    if len({round(x, 6) for x, _ in pts}) < 3:
        return None
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    intercept = my - slope * mx
    total = sum((y - my) ** 2 for _, y in pts)
    residual = sum((y - intercept - slope * x) ** 2 for x, y in pts)
    return dict(n=n, slope=slope, intercept=intercept, r2=1 - residual / total if total else None,
                x_min=10 ** min(x for x, _ in pts), x_max=10 ** max(x for x, _ in pts))


def figure_size_vs_price(specs, n=32768):
    catalog = json.loads((OUT / PRICE_FILE).read_text())
    by = {s['model']: s for s in specs}
    rows = []
    for name, identifier in PRICE_IDS.items():
        if name not in by:
            continue
        s, prices = by[name], endpoint_prices(identifier, catalog)
        windowed = s['window'] > 0
        rows.append(dict(model=name, catalog_id=identifier, family=s['family'],
                         bytes_with_window=bytes_per_token(s, n),
                         bytes_full_attention_only=s['full_bytes'] + s['state_bytes'] / n if windowed else None,
                         **{k: (v * 1e6 if isinstance(v, float) else v) for k, v in prices.items()}))
    panels = [('prompt_min', 'Cheapest uncached input price'), ('cached_min', 'Cheapest cached-read price')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharex=True, sharey=True)
    fits = []
    for ax, (metric, title) in zip(axes, panels):
        style(ax, title, xlabel=f'KV-cache bytes per token at {n // 1024}k context (log scale)',
              ylabel='$ per million tokens (log scale)' if ax is axes[0] else None)
        for family, color in FAMILY.items():
            members = [r for r in rows if r['family'] == family and r[metric]]
            if not members:
                continue
            for r in members:
                if r['bytes_full_attention_only']:
                    ax.plot([r['bytes_full_attention_only'], r['bytes_with_window']], [r[metric]] * 2,
                            color=color, linewidth=1, alpha=0.5)
                    ax.plot(r['bytes_full_attention_only'], r[metric], 'o', markersize=7, markerfacecolor=SURFACE,
                            markeredgecolor=color, markeredgewidth=1.6)
                ax.plot(r['bytes_with_window'], r[metric], 'o', markersize=7, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            variants = [('all layers', 'bytes_with_window', '-')]
            if family.startswith('Sliding'):
                variants.append(('full-attention layers only', 'bytes_full_attention_only', (0, (4, 3))))
            for variant, key, style_ in variants:
                fit = log_fit([(r[key], r[metric]) for r in members])
                if fit:
                    xs = [fit['x_min'] / 1.15, fit['x_max'] * 1.15]
                    ax.plot(xs, [10 ** (fit['intercept'] + fit['slope'] * math.log10(x)) for x in xs],
                            color=color, linewidth=2, linestyle=style_)
                    fits.append(dict(price=metric, family=family, variant=variant, **fit))
        ax.set_xscale('log')
        ax.set_yscale('log')
        byte_ticks(ax, 'x', 4096, 2 ** 20)
        ticks = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
        ax.set_yticks(ticks, [f'${t:g}' for t in ticks])
        ax.set_ylim(0.002, 1.2)
        ax.minorticks_off()
        for r in rows:  # label a few landmark models only
            if r[metric] and short(r['model']) in ('gemma-4-31B', 'gemma-4-26B-A4B', 'gpt-oss-20b', 'Llama-3.3-70B',
                                                   'DeepSeek-V3.2', 'Qwen3.5-397B-A17B'):
                left = short(r['model']) in ('gemma-4-26B-A4B', 'gpt-oss-20b')  # keep clear of the fit box
                x = r['bytes_full_attention_only'] if left and r['bytes_full_attention_only'] else r['bytes_with_window']
                ax.annotate(short(r['model']), (x, r[metric]), textcoords='offset points',
                            xytext=(-9, -3) if left else (7, -11), ha='right' if left else 'left',
                            fontsize=7.5, color=INK2)
        lines = [f"{f['family'].split(' (')[0].split(' +')[0]}{' (full-attn only)' if f['variant'].startswith('full') else ''}:"
                 f" slope {f['slope']:.2f}, R² {f['r2']:.2f}, n={f['n']}" for f in fits if f['price'] == metric]
        ax.text(0.98, 0.03, '\n'.join(['log-log fits'] + lines), transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, color=INK2, linespacing=1.4)
    present = [f for f in FAMILY if any(r['family'] == f and (r['prompt_min'] or r['cached_min']) for r in rows)]
    fitted = {f['family'] for f in fits}
    handles = [plt.Line2D([], [], color=FAMILY[f], marker='o', linestyle='-' if f in fitted else '', linewidth=2,
                          markeredgecolor=SURFACE) for f in present]
    labels = [f if f in fitted else f'{f} (no fit: all models ~same size)' for f in present]
    handles += [plt.Line2D([], [], color=INK2, marker='o', linestyle='-', linewidth=2, markeredgecolor=SURFACE),
                plt.Line2D([], [], color=INK2, marker='o', linestyle=(0, (4, 3)), linewidth=2, markerfacecolor=SURFACE)]
    labels += ['windowed models incl. window part at 32k (solid fit)',
               'windowed models, full-attention layers only (dashed fit)']
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Price per token vs KV-cache size per token ({n // 1024}k context, OpenRouter endpoints 2026-11-11)',
                 x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / 'fig6_price_vs_bytes_per_token.png', dpi=160)
    plt.close(fig)
    write_csv(OUT / 'fig6_price_vs_bytes_per_token.csv', rows)
    write_csv(OUT / 'fig6_fits.csv', fits)
    return rows, fits


def main():
    names = sorted(ACTIVE_PARAMS)
    specs = [spec(name) for name in names]
    figure_bytes(specs)
    figure_context(specs)
    figure_breakeven(specs)
    prices = figure_prices(specs)
    observed = figure_observed()
    price_rows, fits = [], []  # price-vs-size now produced by kv_cache_survey_v1.py (fig6)
    results = dict(
        assumptions=dict(gpu_dollars_per_hour=GPU_DOLLARS_PER_HOUR, gpu_peak_flops=GPU_PEAK_FLOPS, mfu=MFU,
                         hbm_gb=HBM_GB, tiers=TIERS, kv_element_bytes=2, recurrent_state_bytes=4,
                         note='baseline, low-cost, high-cost cases; assumptions not measurements'),
        models=[dict(s, bytes_per_token_32k=bytes_per_token(s, 32768),
                     bytes_per_token_32k_fp8_kv=bytes_per_token(spec(s['model'], kv_bytes=1), 32768),
                     economics_32k=economics(s, 32768)) for s in specs],
        prices=prices, observed_cache=observed, price_vs_bytes=price_rows, price_vs_bytes_fits=fits)
    (OUT / 'results.json').write_text(json.dumps(results, indent=1, default=str))
    for s in sorted(specs, key=lambda s: bytes_per_token(s, 32768)):
        e = economics(s, 32768)
        print(f"{short(s['model']):32s} {s['family']:26s} {human_bytes(bytes_per_token(s, 32768)):>9s}/tok  "
              f"recompute ${e['recompute'] * 1e6:6.3f}/M  break-even h: "
              + '  '.join(f"{t}={v['breakeven_hours']:.3g}" for t, v in e['tiers'].items()))
    print(json.dumps(observed))


if __name__ == '__main__':
    main()
