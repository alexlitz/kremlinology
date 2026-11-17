"""Survey KV-cache size across the most-used open LLMs, and compare storage with recomputation.

Inputs: configs/kv_models_top (collected by collect_model_configs_v1.py, manifest with download
counts), OpenRouter endpoint prices, and the curated economics in kv_cache_economics_v1.py.
Each config is parsed by kv_from_config into a family and per-token cache bytes, with a status:
  exact        standard fields with known semantics
  approximate  recognised feature with a stated simplification (see notes)
  unparsed     not modelled (reason given), or not a decoder LLM
Outputs in reports/kv-cache-economics/: fig6 (price vs bytes, family fits), fig7 (survey),
fig8 ($ per GB of KV), fig9 (storage tiers vs recomputation), CSVs and survey JSON.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False  # labels contain literal dollar signs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402

OUT = ROOT / 'reports/kv-cache-economics'
TOP = ROOT / 'configs/kv_models_top'
CONTEXT = 32768
SURFACE, INK, INK2, GRID = base.SURFACE, base.INK, base.INK2, base.GRID
# Seven families: fixed slot order from the reference palette, plus a marker per family
# (secondary encoding: seven hues cannot all clear all-pairs CVD separation on a scatter).
FAMILIES = {
    'MHA': ('#2a78d6', 'o'), 'GQA': ('#eb6834', 's'), 'MQA': ('#1baf7a', '^'),
    'MLA (latent)': ('#eda100', 'D'), 'Sliding / chunked + global': ('#e87ba4', 'v'),
    'Linear attention + global': ('#008300', 'P'), 'SSM / conv hybrid': ('#4a3aa7', 'X'),
}
NOT_DECODER = {'t5', 'bart', 'bert', 'xlnet', 'florence2', 't5gemma', 'roberta', 'deberta', 'mt5', 'whisper'}
UNMODELLED = {  # key -> reason
    'compress_ratios': 'compressed KV (ratio-dependent storage) not modelled',
    'kv_source_layer_ids': 'cross-layer KV sourcing not modelled',
    'kv_source_layers': 'cross-layer KV sourcing not modelled',
    'sparse_attention_config': 'sparse-attention KV layout not modelled',
    'sparse_config': 'sparse-attention KV layout not modelled',
}


def text_config(config):
    for key in ('text_config', 'llm_config', 'language_config'):
        if isinstance(config.get(key), dict) and config[key]:
            return {**config, **config[key]}
    return config


def first(c, *keys, default=None):
    for key in keys:
        value = c.get(key)
        if value not in (None, 0, ''):
            return value
    return default


def counts(value):
    if isinstance(value, list):
        out = {}
        for item in value:
            out[str(item)] = out.get(str(item), 0) + 1
        return out
    return {}


# Library defaults for fields some configs omit (transformers config classes).
DEFAULTS = {
    'llama': dict(num_attention_heads=32),
    'mistral': dict(num_attention_heads=32, num_key_value_heads=8),
    'gemma3_text': dict(num_attention_heads=8, num_key_value_heads=4, head_dim=256),
    'gemma3': dict(num_attention_heads=8, num_key_value_heads=4, head_dim=256),
}
DRAFT = ('eagle', 'dflash', 'draft', 'speculator', '-assistant', 'mtp-head')


def kv_from_config(raw, kv_bytes=2, repo=''):
    """Per-token cache bytes (full, windowed, window), fixed state, family, KV type, status, notes."""
    c = text_config(raw)
    model_type = str(c.get('model_type') or raw.get('model_type') or '')
    out = dict(model_type=model_type, full_bytes=0.0, window_bytes=0.0, window=0, state_bytes=0.0, notes=[])
    if model_type in NOT_DECODER or 'encoder_layers' in c or c.get('is_encoder_decoder'):
        return dict(out, status='unparsed', reason='not a decoder-only LLM')
    for key, reason in UNMODELLED.items():
        if c.get(key):
            return dict(out, status='unparsed', reason=reason)
    if any(k in repo.lower() for k in DRAFT) or model_type.endswith('_assistant'):
        return dict(out, status='unparsed', reason='speculative-decoding draft head, not a standalone LLM')
    defaulted = [k for k, v in DEFAULTS.get(model_type, {}).items() if c.get(k) in (None, 0)]
    if defaulted:
        c = {**c, **{k: DEFAULTS[model_type][k] for k in defaulted}}
        out['notes'].append('library defaults used for ' + ', '.join(defaulted))
    blocks = c.get('layers_block_type')
    layers = first(c, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers') or (len(blocks) if isinstance(blocks, list) else None)
    if model_type in ('mamba', 'mamba2') and layers:  # pure SSM: no attention heads at all
        hidden = first(c, 'hidden_size', 'd_model')
        inner = c.get('expand', 2) * hidden
        state = c.get('state_size', 16)
        out['state_bytes'] = layers * (inner * state * 4 + inner * (c.get('conv_kernel', 4) - 1) * kv_bytes)
        return dict(out, family='SSM / conv hybrid', kv_type='none', status='exact')
    heads = first(c, 'num_attention_heads', 'n_head', 'num_heads', 'n_heads')
    hidden = first(c, 'hidden_size', 'n_embd', 'n_embed', 'd_model')
    if not (layers and heads):
        return dict(out, status='unparsed', reason='no layer/head fields')
    kv_heads = first(c, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads', 'multi_query_group_num',
                     'num_query_groups', 'num_attention_groups', 'n_head_kv')
    if kv_heads is None:
        kv_heads = 1 if (c.get('multi_query') or c.get('multi_query_attention')) else heads
    head_dim = first(c, 'head_dim', 'kv_channels', 'attention_head_dim', 'd_head')
    if head_dim is None and hidden:
        head_dim = hidden // heads
    per = lambda h, d: 2 * h * d * kv_bytes  # noqa: E731
    kv_type = 'MHA' if kv_heads == heads else 'MQA' if kv_heads == 1 else 'GQA'
    status = 'approximate' if defaulted else 'exact'
    stored = layers
    if c.get('cla_share_factor'):  # Hunyuan cross-layer attention: one KV per group of layers
        stored = math.ceil(layers / c['cla_share_factor'])
        out['notes'].append(f"cross-layer KV shared by {c['cla_share_factor']}")
    if c.get('num_kv_shared_layers'):  # Gemma 3n: trailing layers reuse earlier KV
        out['notes'].append(f"{c['num_kv_shared_layers']} layers reuse KV")
    types = counts(c.get('layer_types')) if isinstance(c.get('layer_types'), list) else {}
    window = first(c, 'sliding_window', 'sliding_window_size', 'attention_chunk_size')

    # Latent attention (MLA and DSA variants).
    if c.get('kv_lora_rank'):
        latent = c['kv_lora_rank'] + (c.get('qk_rope_head_dim') or 0)
        indexer = c.get('index_head_dim', 0) if c.get('index_n_heads') else 0
        if isinstance(c.get('linear_attn_config'), dict):  # Kimi-Linear, GLM-5-Next: KDA + MLA
            la = c['linear_attn_config']
            full, linear = len(la.get('full_attn_layers', [])), len(la.get('kda_layers', []))
            width = la['num_heads'] * la['head_dim']
            out.update(full_bytes=full * (latent * kv_bytes + indexer),
                       state_bytes=linear * (la['num_heads'] * la['head_dim'] ** 2 * 4
                                             + 3 * width * (la.get('short_conv_kernel_size', 4) - 1) * kv_bytes))
            return dict(out, family='Linear attention + global', kv_type='MLA', status=status)
        if c.get('layer_group_size') and c.get('short_conv_kernel_size'):  # Ling/Bailing hybrid
            full = layers // c['layer_group_size']
            linear = layers - full
            out.update(full_bytes=full * latent * kv_bytes,
                       state_bytes=linear * heads * head_dim * head_dim * 4)
            out['notes'].append('linear-layer state assumed heads x head_dim^2 fp32')
            return dict(out, family='Linear attention + global', kv_type='MLA', status='approximate')
        sliding = types.get('sliding_attention', 0)
        if not sliding and c.get('use_sliding_window') and c.get('sliding_window_period'):  # Motif
            sliding = layers - layers // c['sliding_window_period']
        if sliding and window:
            swa_latent = (c.get('swa_kv_lora_rank') or c['kv_lora_rank']) + (c.get('swa_qk_rope_head_dim')
                                                                             or c.get('qk_rope_head_dim') or 0)
            out.update(full_bytes=(layers - sliding) * (latent * kv_bytes + indexer),
                       window_bytes=sliding * swa_latent * kv_bytes, window=window)
            out['notes'].append('MLA with sliding-window layers')
            return dict(out, family='Sliding / chunked + global', kv_type='MLA', status='approximate')
        if c.get('indexer_types'):  # shared indexers store index keys only on 'full' layers
            idx = counts(c['indexer_types']).get('full', layers) if isinstance(c['indexer_types'], list) else layers
            out['full_bytes'] = layers * latent * kv_bytes + idx * indexer
        else:
            out['full_bytes'] = stored * (latent * kv_bytes + indexer)
        return dict(out, family='MLA (latent)', kv_type='MLA', status=status)

    # Linear-attention hybrids (Gated DeltaNet, lightning).
    if 'linear_attention' in types or c.get('full_attention_interval'):
        linear = types.get('linear_attention') or layers - layers // c['full_attention_interval']
        full = layers - linear
        dk, dv = c.get('linear_key_head_dim', 128), c.get('linear_value_head_dim', 128)
        hk, hv = c.get('linear_num_key_heads', 16), c.get('linear_num_value_heads', 32)
        conv = (2 * hk * dk + hv * dv) * (c.get('linear_conv_kernel_dim', 4) - 1) * kv_bytes
        out.update(full_bytes=full * per(kv_heads, head_dim), state_bytes=linear * (hv * dk * dv * 4 + conv))
        return dict(out, family='Linear attention + global', kv_type=kv_type, status=status)
    if isinstance(c.get('attn_type_list'), list) and 0 in c['attn_type_list']:  # MiniMax lightning
        full, linear = c['attn_type_list'].count(1), c['attn_type_list'].count(0)
        out.update(full_bytes=full * per(kv_heads, head_dim), state_bytes=linear * heads * head_dim ** 2 * 4)
        return dict(out, family='Linear attention + global', kv_type=kv_type, status=status)
    if c.get('lightning_head_dim'):  # MiniCPM-SALA: sparse + lightning hybrid
        return dict(out, status='unparsed', reason='sparse/lightning hybrid layout not modelled')

    # Mamba / conv hybrids.
    if isinstance(blocks, list) and 'mamba' in blocks:  # Nemotron 3: per-layer block list
        attention, mamba = blocks.count('attention'), blocks.count('mamba')
        n, d = c['mamba_num_heads'], c['mamba_head_dim']
        st, g = c.get('ssm_state_size', 128), c.get('n_groups', 8)
        out.update(full_bytes=attention * per(kv_heads, head_dim),
                   state_bytes=mamba * (n * d * st * 4 + (n * d + 2 * g * st) * (c.get('conv_kernel', 4) - 1) * kv_bytes))
        return dict(out, family='SSM / conv hybrid', kv_type=kv_type, status=status)
    pattern = c.get('hybrid_override_pattern')
    if isinstance(pattern, str):  # Nemotron-H
        attention, mamba = pattern.count('*'), pattern.count('M')
        n, d, s, g = c['mamba_num_heads'], c['mamba_head_dim'], c['ssm_state_size'], c['n_groups']
        out.update(full_bytes=attention * per(kv_heads, head_dim),
                   state_bytes=mamba * (n * d * s * 4 + (n * d + 2 * g * s) * (c.get('conv_kernel', 4) - 1) * kv_bytes))
        return dict(out, family='SSM / conv hybrid', kv_type=kv_type, status=status)
    if 'mamba' in types or model_type in ('falcon_h1', 'mamba', 'mamba2'):
        attention = types.get('attention', types.get('full_attention', 0)) if types else (layers if model_type == 'falcon_h1' else 0)
        mamba = types.get('mamba', layers if model_type in ('falcon_h1', 'mamba', 'mamba2') else 0)
        n, d, s = c.get('mamba_n_heads'), c.get('mamba_d_head'), c.get('mamba_d_state') or c.get('state_size')
        if n and d and s:
            groups, kernel = c.get('mamba_n_groups', 1), c.get('mamba_d_conv', 4)
            state = n * d * s * 4 + (n * d + 2 * groups * s) * (kernel - 1) * kv_bytes
        else:  # Mamba-1: expand x hidden channels, state_size each
            state = c.get('expand', 2) * hidden * (s or 16) * 4
        out.update(full_bytes=attention * per(kv_heads, head_dim), state_bytes=mamba * state)
        return dict(out, family='SSM / conv hybrid', kv_type=kv_type, status=status)
    if 'conv' in types:  # LFM2/3: short-conv layers keep (kernel-1) x hidden values
        conv = types['conv'] * (c.get('conv_L_cache', 3)) * hidden * kv_bytes
        sliding = types.get('sliding_attention', 0)
        out.update(full_bytes=types.get('full_attention', 0) * per(kv_heads, head_dim),
                   window_bytes=sliding * per(kv_heads, head_dim), window=window if sliding else 0, state_bytes=conv)
        return dict(out, family='SSM / conv hybrid', kv_type=kv_type, status=status)

    # Windowed attention.
    if model_type.startswith('gemma4') or c.get('global_head_dim'):
        full, sliding = types.get('full_attention', 0), types.get('sliding_attention', 0)
        shared = c.get('num_kv_shared_layers') or 0
        g_heads = c.get('num_global_key_value_heads') or kv_heads
        out.update(full_bytes=full * per(g_heads, c.get('global_head_dim', head_dim)),
                   window_bytes=max(0, sliding - shared) * per(kv_heads, head_dim), window=window)
        if shared:
            status = 'approximate'
        return dict(out, family='Sliding / chunked + global', kv_type=kv_type, status=status)
    if model_type == 'llama4_text' or c.get('attention_chunk_size') and model_type.startswith('llama4'):
        full = sum(1 for i in range(layers) if (i + 1) % 4 == 0)
        out.update(full_bytes=full * per(kv_heads, head_dim), window_bytes=(layers - full) * per(kv_heads, head_dim),
                   window=c['attention_chunk_size'])
        return dict(out, family='Sliding / chunked + global', kv_type=kv_type, status=status)
    sliding = types.get('sliding_attention', 0)
    if isinstance(c.get('hybrid_layer_pattern'), list) and window:  # MiMo: 1 = SWA layer
        sliding = sum(1 for v in c['hybrid_layer_pattern'] if v == 1)
    if c.get('local_layer_ids'):  # Inkling
        sliding = len(c['local_layer_ids'])
    if model_type == 'gemma2':
        sliding = layers // 2
    if model_type.startswith('gemma3') and not c.get('head_dim'):
        head_dim = 256  # Gemma3TextConfig default
    if model_type.startswith('gemma3') and not sliding:
        pattern = c.get('sliding_window_pattern', 6)
        sliding = layers - sum(1 for i in range(layers) if (i + 1) % pattern == 0)
    if sliding and window:
        swa_heads = c.get('swa_num_key_value_heads') or kv_heads
        swa_dim = c.get('swa_head_dim') or head_dim
        swa_v = c.get('swa_v_head_dim') or swa_dim
        other = c.get('attention_other_setting') if isinstance(c.get('attention_other_setting'), dict) else {}
        if other:  # Step: sliding layers use a different group count / head dim
            swa_heads, swa_dim = other.get('num_attention_groups', swa_heads), other.get('head_dim', swa_dim)
            swa_v = swa_dim
        full_v = c.get('v_head_dim') or head_dim
        full_layers = max(0, stored - sliding) if stored != layers else layers - sliding
        out.update(full_bytes=full_layers * kv_heads * (head_dim + full_v) * kv_bytes,
                   window_bytes=sliding * swa_heads * (swa_dim + swa_v) * kv_bytes, window=window)
        if other or c.get('swa_num_key_value_heads') or c.get('local_layer_ids'):
            status = 'approximate'
            out['notes'].append('per-layer-type head counts from config')
        return dict(out, family='Sliding / chunked + global', kv_type=kv_type, status=status)
    # Whole-model sliding window (Mistral v0.1, Phi-3, StarCoder2) when enabled and short.
    if window and c.get('use_sliding_window', True) is not False and window < 32768 and \
            model_type in ('mistral', 'phi3', 'starcoder2', 'mixtral', 'qwen2', 'phi3_v'):
        out.update(window_bytes=stored * per(kv_heads, head_dim), window=window)
        return dict(out, family='Sliding / chunked + global', kv_type=kv_type, status=status)

    v_dim = c.get('v_head_dim') or head_dim
    out['full_bytes'] = stored * kv_heads * (head_dim + v_dim) * kv_bytes
    return dict(out, family=kv_type, kv_type=kv_type, status=status)


def bytes_per_token(spec, n=CONTEXT):
    return spec['full_bytes'] + spec['window_bytes'] * min(n, spec['window'] or n) / n + spec['state_bytes'] / n


def load_survey():
    manifest = json.loads((TOP / 'manifest.json').read_text())
    rows = []
    for entry in manifest['models']:
        row = dict(repo=entry['repo'], downloads=entry.get('downloads') or 0, sources=entry['sources'],
                   openrouter_ids=entry.get('openrouter_ids', []), config_via=entry.get('config_via', 'repo'))
        if 'config' not in entry:
            row.update(status='unparsed', reason=f"no config ({entry.get('config_error', 'unknown')})")
        else:
            try:
                spec = kv_from_config(json.loads((TOP / entry['config']).read_text()), repo=entry['repo'])
            except Exception as error:  # malformed or unexpected structure
                spec = dict(status='unparsed', reason=f'parse error {type(error).__name__}', notes=[])
            row.update({k: v for k, v in spec.items() if k != 'notes'}, notes='; '.join(spec.get('notes', [])))
            if spec['status'] != 'unparsed':
                row['bytes_per_token_32k'] = bytes_per_token(spec)
                row['bytes_full_attention_only'] = spec['full_bytes'] + spec['state_bytes'] / CONTEXT
                if row['bytes_per_token_32k'] <= 0:
                    row.update(status='unparsed', reason='zero cache size parsed')
        rows.append(row)
    return manifest, rows


def coverage(rows):
    top = [r for r in rows if any(s.startswith('hf_top') for s in r['sources'])]
    total = sum(r['downloads'] for r in top) or 1
    out = {}
    for status in ('exact', 'approximate', 'unparsed'):
        members = [r for r in top if r['status'] == status]
        out[status] = dict(models=len(members), download_share=sum(r['downloads'] for r in members) / total)
    reasons = {}
    for r in top:
        if r['status'] == 'unparsed':
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    out['unparsed_reasons'] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return out


def style(ax, title, xlabel=None, ylabel=None):
    base.style(ax, title, xlabel, ylabel)


def figure_survey(rows):
    parsed = [r for r in rows if r['status'] != 'unparsed' and any(s.startswith('hf_top') for s in r['sources'])]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    style(ax, f'KV-cache bytes per token at 32k context, {len(parsed)} of the 1,000 most-downloaded open LLMs',
          xlabel='bytes per token (log scale); dot area ~ 30-day downloads')
    order = list(FAMILIES)
    import random
    jitter = random.Random(0)
    biggest = max(r['downloads'] for r in parsed)
    for i, family in enumerate(order):
        color, marker = FAMILIES[family]
        members = [r for r in parsed if r['family'] == family]
        if not members:
            continue
        xs = [r['bytes_per_token_32k'] for r in members]
        ys = [i + jitter.uniform(-0.28, 0.28) for _ in members]
        sizes = [12 + 500 * math.sqrt(max(r['downloads'], 1) / biggest) for r in members]
        ax.scatter(xs, ys, s=sizes, color=color, marker=marker, alpha=0.6, edgecolors=SURFACE, linewidths=0.6)
        top = max(members, key=lambda r: r['downloads'])
        ax.annotate(f"most downloaded: {top['repo'].split('/')[-1][:30]}", (top['bytes_per_token_32k'], i),
                    textcoords='offset points', xytext=(0, 15), ha='center', fontsize=7.5, color=INK2)
        ax.text(1.005, i, f'n={len(members)}', transform=ax.get_yaxis_transform(), va='center', fontsize=8,
                color=INK2)
    ax.set_xscale('log')
    base.byte_ticks(ax, 'x', 256, 2 ** 21)
    ax.set_xlim(200, 2 ** 21)
    ax.set_yticks(range(len(order)), order, color=INK, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / 'fig7_survey_bytes_per_token.png', dpi=160)
    plt.close(fig)


def price_rows(rows):
    catalog = json.loads((OUT / 'openrouter-endpoints-survey.json').read_text())
    out = []
    for r in rows:
        if r['status'] == 'unparsed':
            continue
        for identifier in r['openrouter_ids']:
            if identifier.endswith(':free') or identifier not in catalog:
                continue
            prices = base.endpoint_prices(identifier, catalog)
            if not prices['prompt_min']:
                continue
            out.append(dict(repo=r['repo'], openrouter_id=identifier, family=r['family'], kv_type=r['kv_type'],
                            status=r['status'], bytes_with_window=r['bytes_per_token_32k'],
                            bytes_full_attention_only=r['bytes_full_attention_only'] if r['window'] else None,
                            prompt_min=prices['prompt_min'] * 1e6,
                            cached_min=prices['cached_min'] * 1e6 if prices['cached_min'] else None,
                            endpoints=prices['endpoints']))
    return out


def fit(points, min_distinct=2):
    pts = [(math.log10(x), math.log10(y)) for x, y in points if x and y and x > 0 and y > 0]
    if len(pts) < 3 or len({round(x, 3) for x, _ in pts}) < min_distinct:
        return None
    result = base.log_fit([(10 ** x, 10 ** y) for x, y in pts]) if len({round(x, 6) for x, _ in pts}) >= 3 else None
    if result is None:  # two x clusters: still a least-squares line, flagged
        n = len(pts)
        mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        if sxx == 0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
        intercept = my - slope * mx
        total = sum((y - my) ** 2 for _, y in pts)
        residual = sum((y - intercept - slope * x) ** 2 for x, y in pts)
        result = dict(n=n, slope=slope, intercept=intercept, r2=1 - residual / total if total else None,
                      x_min=10 ** min(x for x, _ in pts), x_max=10 ** max(x for x, _ in pts), two_clusters=True)
    return result


def figure_price_vs_bytes(prices):
    panels = [('prompt_min', 'Cheapest uncached input price'), ('cached_min', 'Cheapest cached-read price')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.6), sharex=True, sharey=True)
    fits = []
    for ax, (metric, title) in zip(axes, panels):
        style(ax, title, xlabel='KV-cache bytes per token at 32k context (log scale)',
              ylabel='$ per million tokens (log scale)' if ax is axes[0] else None)
        for family, (color, marker) in FAMILIES.items():
            members = [r for r in prices if r['family'] == family and r[metric]]
            for r in members:
                if r['bytes_full_attention_only']:
                    ax.plot([r['bytes_full_attention_only'], r['bytes_with_window']], [r[metric]] * 2, color=color,
                            linewidth=1, alpha=0.45)
                    ax.plot(r['bytes_full_attention_only'], r[metric], marker, markersize=6.5, markerfacecolor=SURFACE,
                            markeredgecolor=color, markeredgewidth=1.5)
                ax.plot(r['bytes_with_window'], r[metric], marker, markersize=6.5, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            variants = [('all layers', 'bytes_with_window', '-')]
            if family.startswith('Sliding'):
                variants.append(('full-attention layers only', 'bytes_full_attention_only', (0, (4, 3))))
            for variant, key, line in variants:
                f = fit([(r[key], r[metric]) for r in members])
                if f:
                    xs = [f['x_min'] / 1.12, f['x_max'] * 1.12]
                    ax.plot(xs, [10 ** (f['intercept'] + f['slope'] * math.log10(x)) for x in xs], color=color,
                            linewidth=2, linestyle=line)
                    fits.append(dict(price=metric, family=family, variant=variant, **f))
        ax.set_xscale('log')
        ax.set_yscale('log')
        base.byte_ticks(ax, 'x', 4096, 2 ** 21)
        ticks = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        ax.set_yticks(ticks, [f'${t:g}' for t in ticks])
        ax.minorticks_off()
        lines = [f"{f['family']}{' (full-attn only)' if f['variant'].startswith('full') else ''}: slope "
                 f"{f['slope']:.2f}, R² {f['r2']:.2f}, n={f['n']}{' *' if f.get('two_clusters') else ''}"
                 for f in fits if f['price'] == metric]
        ax.text(0.0, -0.13, '\n'.join(['log-log fits (* = sizes form two clusters):'] + lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=7.5, color=INK2, linespacing=1.35)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='-', linewidth=2, markeredgecolor=SURFACE)
               for c, m in FAMILIES.values()]
    labels = [f"{f} ({sum(1 for r in prices if r['family'] == f)} priced)" for f in FAMILIES]
    handles += [plt.Line2D([], [], color=INK2, marker='o', linestyle='-', linewidth=2),
                plt.Line2D([], [], color=INK2, marker='o', linestyle=(0, (4, 3)), linewidth=2, markerfacecolor=SURFACE)]
    labels += ['windowed: incl. window part (solid fit)', 'windowed: full-attention layers only (dashed fit)']
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle('Price per token vs KV-cache size per token, all OpenRouter models with parseable configs',
                 x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    fig.savefig(OUT / 'fig6_price_vs_bytes_per_token.png', dpi=160)
    plt.close(fig)
    return fits


def figure_price_per_gb(prices):
    """$ per GB of KV implied by per-token prices: price / bytes per token."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    order = list(FAMILIES)
    for ax, (metric, title) in zip(axes, [('prompt_min', 'Uncached input price per GB of KV it produces'),
                                          ('cached_min', 'Cached-read price per GB of KV read')]):
        style(ax, title, xlabel='$ per GB of KV cache (log scale)')
        for i, family in enumerate(order):
            color, marker = FAMILIES[family]
            members = [r for r in prices if r['family'] == family and r[metric]]
            values = sorted(r[metric] / 1e6 / (r['bytes_with_window'] / 1e9) for r in members)
            if not values:
                continue
            ax.scatter(values, [i] * len(values), color=color, marker=marker, s=34, alpha=0.75, edgecolors=SURFACE,
                       linewidths=0.6)
            median = values[len(values) // 2]
            ax.plot([median] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.5)
            ax.text(median, i - 0.36, f'median ${median:.2g}', ha='center', fontsize=7, color=INK2)
        ax.set_xscale('log')
        ticks = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        ax.set_xticks(ticks, [f'${t:g}' for t in ticks])
        ax.minorticks_off()
        ax.set_yticks(range(len(order)), order, color=INK, fontsize=9)
        ax.grid(axis='y', visible=False)
        for i, family in enumerate(order):
            if not any(r['family'] == family and r[metric] for r in prices):
                ax.text(0.5, i, 'no priced models', transform=ax.get_yaxis_transform(), ha='center', va='center',
                        fontsize=8, color=INK2)
    axes[0].invert_yaxis()  # shared y: invert once
    fig.suptitle('What providers charge per GB of KV cache (32k context, OpenRouter, 2026-11-11)', x=0.01,
                 ha='left', color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'fig8_price_per_gb_kv.png', dpi=160)
    plt.close(fig)


# Storage tiers. Owned hardware amortized over its life; cloud at list price. $/GB-hour and bandwidth to GPU.
HOURS_PER_YEAR = 8766
POWER_DOLLARS_PER_KWH = 0.15


def owned(price_per_gb, years, watts_per_gb=0.0, overhead=1.0):
    return price_per_gb * overhead / (years * HOURS_PER_YEAR) + watts_per_gb / 1000 * POWER_DOLLARS_PER_KWH


TIERS = [  # name, $/GB-hour, bandwidth bytes/s, latency s, basis
    ('HBM (rental opportunity cost)', 2.50 / 80, None, 0.0, 'H100 $2.50/h / 80 GB; upper bound under memory pressure'),
    ('Host DRAM (owned DDR5)', owned(6.0, 4, watts_per_gb=0.35 / 1, overhead=1.3), 50e9, 0.0,
     '$6/GB DDR5 RDIMM, 4 y, ~0.35 W/GB, 1.3x server share; 50 GB/s PCIe 5 to GPU'),
    ('NVMe RAID x4 (owned)', owned(0.075, 5, watts_per_gb=8 / 4000, overhead=1.3), 4 * 6.3e9, 1e-4,
     '$75/TB SN850X-class, 5 y; 4 drives x measured 6.3 GB/s'),
    ('NVMe single (owned)', owned(0.075, 5, watts_per_gb=8 / 4000, overhead=1.3), 6.3e9, 1e-4,
     'measured 6.3 GB/s sequential read (WD SN850X 4 TB, direct I/O)'),
    ('HDD RAID5 x4 (owned)', owned(0.0175 * 4 / 3, 5, watts_per_gb=8 / 16000 * 4 / 3, overhead=1.5), 0.38e9, 0.01,
     '$17.5/TB Exos 16 TB, RAID5 3/4 usable, 5 y; measured 380 MB/s'),
    ('Cloud object storage (S3-class)', 0.023 / 730, 5e9, 0.05, '$0.023/GB-month; ~5 GB/s per node with parallel GETs'),
]


def figure_storage_vs_recompute(prices):
    """Cost per GB of KV to hold for t hours (+ one reload) on each tier, vs recomputing that GB."""
    gpu_per_second = base.GPU_DOLLARS_PER_HOUR[0] / 3600
    specs = {name: base.spec(name) for name in base.ACTIVE_PARAMS}
    reps = [('GQA: Llama-3.3-70B', 'unsloth--Llama-3.3-70B-Instruct'),
            ('MLA: DeepSeek-V3', 'deepseek-ai--DeepSeek-V3'),
            ('Sliding: Gemma-4-31B', 'google--gemma-4-31B-it'),
            ('Sliding: gpt-oss-120b', 'openai--gpt-oss-120b'),
            ('Linear hybrid: Qwen3.5-27B', 'Qwen--Qwen3.5-27B'),
            ('Dense GQA small: Llama-3.1-8B', 'meta-llama--Llama-3.1-8B-Instruct')]
    recompute = {}
    for label, name in reps:
        e = base.economics(specs[name], CONTEXT)
        recompute[label] = e['recompute'] / (e['bytes'] / 1e9)  # $ per GB of KV
    hours = [10 ** (k / 20) for k in range(-60, 86)]  # 0.001 h .. ~2 years
    fig, ax = plt.subplots(figsize=(13, 7.5))
    style(ax, 'Holding 1 GB of KV cache on each storage tier vs recomputing it (32k context)',
          xlabel='retention time (log scale)', ylabel='$ per GB of KV cache (log scale)')
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
    table = []
    for (name, rate, bandwidth, latency, basis), color in zip(TIERS, colors):
        reload = 0.0 if bandwidth is None else (1e9 / bandwidth + latency) * gpu_per_second
        ys = [rate * h + reload for h in hours]
        ax.plot(hours, ys, color=color, linewidth=2, label=name)
        table.append(dict(tier=name, dollars_per_gb_hour=rate, reload_dollars_per_gb=reload,
                          bandwidth_bytes_per_s=bandwidth, basis=basis,
                          **{f'breakeven_hours_{lbl}': (max(0.0, (v - reload) / rate) if v > reload else 0.0)
                             for lbl, v in recompute.items()}))
    levels = [(value, f'recompute {label}: ${value:.2g}/GB', INK2, (0, (4, 3)))
              for label, value in recompute.items()]
    cached = sorted(r['cached_min'] / 1e6 / (r['bytes_with_window'] / 1e9) for r in prices if r['cached_min'])
    if cached:
        median = cached[len(cached) // 2]
        levels.append((median, f'median provider cached-read price: ${median:.2g}/GB', INK, (0, (1, 2))))
    placed = []
    for value, text, color, line in sorted(levels):
        ax.axhline(value, color=color, linewidth=1.1, linestyle=line)
        y = value
        while placed and math.log10(y) - math.log10(placed[-1]) < 0.09:  # keep labels apart
            y *= 10 ** 0.03
        placed.append(y)
        ax.annotate(text, (1.0, y), xycoords=('axes fraction', 'data'), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=7.5, color=color)
    ax.text(0.0, -0.11, 'Reload counted as GPU idle time (not overlapped); owned hardware amortized; '
            'tier assumptions in fig9_storage_tiers_vs_recompute.csv.', transform=ax.transAxes, fontsize=7.5, color=INK2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    marks = [(1 / 60, '1 min'), (1, '1 h'), (24, '1 day'), (24 * 30, '1 month'), (24 * 365, '1 year')]
    ax.set_xticks([h for h, _ in marks], [lbl for _, lbl in marks])
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ax.set_yticks(yticks, [f'${t:.5f}'.rstrip('0').rstrip('.') if t < 1 else f'${t:g}' for t in yticks])
    ax.minorticks_off()
    ax.set_xlim(hours[0], 24 * 365 * 2)
    ax.set_ylim(1e-5, 30)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(OUT / 'fig9_storage_tiers_vs_recompute.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig9_storage_tiers_vs_recompute.csv', table)
    return table, recompute


def price_regression(prices, rows):
    """log10 price ~ log10 active params + log10 KV bytes/token: does cache size matter given model size?"""
    import numpy as np
    manifest = {e['repo']: e for e in json.loads((TOP / 'manifest.json').read_text())['models']}
    data = []
    for p in prices:
        entry = manifest.get(p['repo'])
        if not entry or 'config' not in entry:
            continue
        active = safe_active_params(json.loads((TOP / entry['config']).read_text()))
        if active:
            data.append((p, active))
    out = {}
    generator = np.random.default_rng(0)
    for metric in ('prompt_min', 'cached_min'):
        pts = [(math.log10(a), math.log10(p['bytes_with_window']), math.log10(p[metric])) for p, a in data if p[metric]]
        if len(pts) < 10:
            continue
        x = np.array([[1, a, b] for a, b, _ in pts])
        y = np.array([c for _, _, c in pts])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        boots = []
        for _ in range(2000):
            idx = generator.integers(0, len(pts), len(pts))
            boots.append(np.linalg.lstsq(x[idx], y[idx], rcond=None)[0])
        boots = np.array(boots)
        residual = y - x @ beta
        r2 = 1 - residual.var() / y.var()
        size_only = np.linalg.lstsq(x[:, :2], y, rcond=None)[0]
        r2_size = 1 - (y - x[:, :2] @ size_only).var() / y.var()
        out[metric] = dict(n=len(pts), elasticity_active_params=beta[1], elasticity_kv_bytes=beta[2],
                           kv_ci95=list(np.percentile(boots[:, 2], [2.5, 97.5])),
                           active_ci95=list(np.percentile(boots[:, 1], [2.5, 97.5])), r2=r2,
                           r2_active_params_only=r2_size)
    return out


def figure_compute_per_byte(rows):
    """Prefill FLOPs per token divided by cached bytes per token: how much compute each cached byte saves."""
    manifest = {e['repo']: e for e in json.loads((TOP / 'manifest.json').read_text())['models']}
    parsed = []
    for r in rows:
        if r['status'] == 'unparsed' or not any(x.startswith('hf_top') for x in r['sources']):
            continue
        active = safe_active_params(json.loads((TOP / manifest[r['repo']]['config']).read_text()))
        if active:
            parsed.append((r, 2 * active / r['bytes_per_token_32k']))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    style(ax, 'Prefill compute saved per cached byte (weights only, 32k context; most-downloaded models)',
          xlabel='FLOPs per byte of KV cache (log scale)')
    order = list(FAMILIES)
    table = []
    import random
    jitter = random.Random(1)
    for i, family in enumerate(order):
        color, marker = FAMILIES[family]
        values = sorted(v for r, v in parsed if r['family'] == family)
        if not values:
            continue
        ax.scatter(values, [i + jitter.uniform(-0.25, 0.25) for _ in values], color=color, marker=marker, s=22,
                   alpha=0.6, edgecolors=SURFACE, linewidths=0.5)
        median = values[len(values) // 2]
        ax.plot([median] * 2, [i - 0.35, i + 0.35], color=INK, linewidth=1.6)
        ax.text(median, i - 0.4, f'median {median:,.0f}', ha='center', fontsize=7.5, color=INK2)
        table.append(dict(family=family, models=len(values), median_flops_per_kv_byte=median))
    ax.set_xscale('log')
    ax.set_yticks(range(len(order)), order, color=INK, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.invert_yaxis()
    ax.text(0.99, 0.02, 'attention FLOPs excluded (adds ~30% for 70B dense at 32k); higher = cache worth keeping longer',
            transform=ax.transAxes, ha='right', fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig13_compute_per_cached_byte.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig13_compute_per_cached_byte.csv', table)
    return table


def figure_cache_pricing_all():
    """Cache read/write price as a multiple of uncached input, all OpenRouter models incl. closed providers."""
    models = json.loads((OUT / 'openrouter-models-2026-11-11.json').read_text())['data']
    rows = []
    for m in models:
        p = m.get('pricing', {})
        try:
            prompt = float(p.get('prompt') or 0)
        except ValueError:
            continue
        if prompt <= 0:
            continue
        read = float(p['input_cache_read']) if p.get('input_cache_read') else None
        write = float(p['input_cache_write']) if p.get('input_cache_write') else None
        if read is None and write is None:
            continue
        rows.append(dict(model=m['id'], org=m['id'].split('/')[0], prompt=prompt * 1e6,
                         read_multiple=read / prompt if read is not None else None,
                         write_multiple=write / prompt if write is not None else None,
                         open_weights=bool(m.get('hugging_face_id'))))
    orgs = sorted({r['org'] for r in rows}, key=lambda o: -sum(1 for r in rows if r['org'] == o))
    closed = [o for o in ('anthropic', 'openai', 'google', 'x-ai') if o in orgs]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, key, title in ((axes[0], 'read_multiple', 'Cache-read price ÷ uncached input price'),
                           (axes[1], 'write_multiple', 'Cache-write price ÷ uncached input price')):
        style(ax, title, xlabel='multiple of the uncached input price')
        groups = closed + ['open-weights models']
        for i, group in enumerate(groups):
            members = [r for r in rows if r[key] is not None and
                       (r['org'] == group if group != 'open-weights models' else r['open_weights'])]
            values = sorted(r[key] for r in members)
            if not values:
                ax.text(0.5, i, 'none listed', transform=ax.get_yaxis_transform(), ha='center', va='center',
                        fontsize=8, color=INK2)
                continue
            ax.scatter(values, [i] * len(values), s=26, color='#2a78d6' if group != 'open-weights models' else '#eb6834',
                       alpha=0.6, edgecolors=SURFACE, linewidths=0.5)
            median = values[len(values) // 2]
            ax.plot([median] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.6)
            ax.text(median, i - 0.36, f'median {median:.2f}x (n={len(values)})', ha='center', fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(groups)), groups, color=INK, fontsize=9)
        ax.grid(axis='y', visible=False)
        ax.axvline(1, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].invert_yaxis()
    fig.suptitle('How providers price prompt caching (OpenRouter model-level prices, 2026-11-11)', x=0.01, ha='left',
                 color=INK, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'fig14_cache_price_multiples.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig14_cache_price_multiples.csv', rows)
    return rows


HARDWARE_PRICES = [  # tier, purchase $/GB (or None for cloud), $/GB-month, basis
    ('HBM (H100, rental share)', None, 2.50 / 80 * 730, 'H100 $2.50/h / 80 GB'),
    ('Host DRAM (DDR5 RDIMM)', 6.0, None, '$6/GB, 4 y, 0.35 W/GB, 1.3x server share'),
    ('NVMe SSD (SN850X-class)', 0.075, None, '$75/TB, 5 y'),
    ('HDD RAID5 (Exos 16 TB)', 0.0175 * 4 / 3, None, '$17.5/TB, RAID5 3/4 usable, 5 y'),
    ('Cloud block (gp3-class)', None, 0.08, '$0.08/GB-month list'),
    ('Cloud object (S3-class)', None, 0.023, '$0.023/GB-month list'),
]


def figure_hardware_cost():
    rows = []
    monthly = {name: rate * 730 for name, rate, *_ in TIERS}
    for tier, purchase, month, basis in HARDWARE_PRICES:
        if month is None:
            key = {'Host DRAM (DDR5 RDIMM)': 'Host DRAM (owned DDR5)', 'NVMe SSD (SN850X-class)': 'NVMe single (owned)',
                   'HDD RAID5 (Exos 16 TB)': 'HDD RAID5 x4 (owned)'}[tier]
            month = monthly[key]
        rows.append(dict(tier=tier, purchase_dollars_per_gb=purchase, dollars_per_gb_month=month, basis=basis))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    style(ax, 'Storage cost per GB-month by tier (owned hardware amortized, cloud at list price)',
          xlabel='$ per GB-month (log scale)')
    ys = range(len(rows))
    ax.barh(list(ys), [r['dollars_per_gb_month'] for r in rows], color='#2a78d6', height=0.6, edgecolor=SURFACE)
    for i, r in enumerate(rows):
        label = f"${r['dollars_per_gb_month']:.3g}/GB-month" + (
            f"  (buy ${r['purchase_dollars_per_gb']:.3g}/GB)" if r['purchase_dollars_per_gb'] else '')
        ax.text(r['dollars_per_gb_month'] * 1.15, i, label, va='center', fontsize=8, color=INK2)
    ax.set_xscale('log')
    ax.set_xlim(5e-4, 500)
    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    ax.set_xticks(ticks, [f'${t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(list(ys), [r['tier'] for r in rows], color=INK, fontsize=9)
    ax.grid(axis='y', visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / 'fig15_storage_cost_per_gb.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig15_storage_cost_per_gb.csv', rows)
    return rows


def main():
    manifest, rows = load_survey()
    cover = coverage(rows)
    figure_survey(rows)
    prices = price_rows(rows)
    fits = figure_price_vs_bytes(prices)
    figure_price_per_gb(prices)
    tiers, recompute = figure_storage_vs_recompute(prices)
    regression = price_regression(prices, rows)
    compute_per_byte = figure_compute_per_byte(rows)
    cache_pricing = figure_cache_pricing_all()
    hardware = figure_hardware_cost()
    base.write_csv(OUT / 'fig7_survey.csv', [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                              for k, v in r.items()} for r in rows])
    base.write_csv(OUT / 'fig6_price_vs_bytes_per_token.csv', prices)
    if fits:
        base.write_csv(OUT / 'fig6_fits.csv', fits)
    (OUT / 'survey_results.json').write_text(json.dumps(dict(
        collected_at=manifest['collected_at'], coverage=cover, fits=fits, recompute_dollars_per_gb=recompute,
        tiers=tiers, context=CONTEXT, price_regression=regression, compute_per_byte=compute_per_byte,
        storage_cost=hardware, cache_price_rows=len(cache_pricing)), indent=1, default=str))
    print(json.dumps(regression, indent=1, default=float))
    print(json.dumps(cover, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in recompute.items()}))
    for f in fits:
        print(f['price'], f['family'], f['variant'], 'n', f['n'], 'slope', round(f['slope'], 2), 'R2', round(f['r2'], 2))



def safe_active_params(raw):
    try:
        value = estimate_active_params(raw)
    except (TypeError, KeyError, ValueError, ZeroDivisionError):
        return None
    return value if isinstance(value, (int, float)) and value > 0 else None


def estimate_active_params(raw):
    """Approximate active non-embedding parameters per token from a config (weights used in prefill).

    Attention projections (incl. MLA LoRA factors), dense or MoE MLP (routed experts per token plus
    shared experts; leading dense layers), and linear-attention / Mamba projections. Embedding and
    LM-head weights are excluded (prefill computes logits only for the last token). Returns None
    when required fields are missing. Accuracy is checked against published counts in tests.
    """
    c = text_config(raw)
    model_type = str(c.get('model_type') or '')
    c = {**{k: v for k, v in DEFAULTS.get(model_type, {}).items()}, **{k: v for k, v in c.items() if v not in (None, 0)}}
    blocks = c.get('layers_block_type')
    layers = first(c, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers') or (len(blocks) if isinstance(blocks, list) else None)
    hidden = first(c, 'hidden_size', 'n_embd', 'n_embed', 'd_model')
    if model_type in ('mamba', 'mamba2') and layers and hidden:  # in/out projections dominate
        inner = c.get('expand', 2) * hidden
        return layers * (hidden * 2 * inner + inner * hidden + inner * (c.get('state_size', 16) * 2 + 64))
    heads = first(c, 'num_attention_heads', 'n_head', 'num_heads', 'n_heads')
    if not (layers and hidden and heads):
        return None
    kv_heads = first(c, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads', 'multi_query_group_num',
                     'num_query_groups', 'n_head_kv') or (1 if c.get('multi_query') else heads)
    head_dim = first(c, 'head_dim', 'kv_channels', 'd_head') or hidden // heads
    if c.get('kv_lora_rank'):
        nope, rope, v = c.get('qk_nope_head_dim') or 0, c.get('qk_rope_head_dim') or 0, c.get('v_head_dim') or head_dim
        q = (hidden * c['q_lora_rank'] + c['q_lora_rank'] * heads * (nope + rope)) if c.get('q_lora_rank') \
            else hidden * heads * (nope + rope)
        attention = q + hidden * (c['kv_lora_rank'] + rope) + c['kv_lora_rank'] * heads * (nope + v) + heads * v * hidden
    else:
        attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    # Layer mix: attention layers vs linear-attention / Mamba layers vs MLP-only layers.
    types = c.get('layer_types') if isinstance(c.get('layer_types'), list) else []
    pattern = c.get('hybrid_override_pattern') if isinstance(c.get('hybrid_override_pattern'), str) else ''
    mixer = 0
    if 'linear_attention' in types or c.get('full_attention_interval'):
        linear = types.count('linear_attention') if types else layers - layers // c['full_attention_interval']
        hk, dk = c.get('linear_num_key_heads', 16), c.get('linear_key_head_dim', 128)
        hv, dv = c.get('linear_num_value_heads', 32), c.get('linear_value_head_dim', 128)
        mixer = linear * (hidden * (2 * hk * dk + 2 * hv * dv + 2 * hv) + hv * dv * hidden)
        attention_layers = layers - linear
    elif pattern or (isinstance(blocks, list) and 'mamba' in blocks) or 'mamba' in types:
        names = list(pattern) if pattern else (blocks if isinstance(blocks, list) else types)
        attention_layers = sum(1 for t in names if t in ('*', 'attention'))
        mamba_layers = sum(1 for t in names if t in ('M', 'mamba'))
        n, d = c.get('mamba_num_heads') or c.get('mamba_n_heads') or 0, c.get('mamba_head_dim') or c.get('mamba_d_head') or 0
        st, g = c.get('ssm_state_size') or c.get('mamba_d_state') or 128, c.get('n_groups') or c.get('mamba_n_groups') or 1
        inner = n * d or 2 * hidden
        mixer = mamba_layers * (hidden * (2 * inner + 2 * g * st + n) + inner * hidden)
        if pattern:  # Nemotron-H: '-' layers are MLP-only, 'M' and '*' layers have no MLP
            layers_with_mlp = pattern.count('-') + (pattern.count('E') if 'E' in pattern else 0)
        else:
            layers_with_mlp = layers
    else:
        attention_layers = layers
    layers_with_mlp = locals().get('layers_with_mlp', layers)
    dense_ffn = first(c, 'intermediate_size', 'n_inner', 'ffn_hidden_size') or 4 * hidden
    if isinstance(dense_ffn, list):  # per-layer sizes: use the mean
        dense_ffn = sum(dense_ffn) / len(dense_ffn)
    gated = 3 if c.get('model_type') not in ('gpt2', 'gpt_neox', 'opt', 'bloom', 'falcon', 'gpt_neo', 'phi') else 2
    experts = first(c, 'num_experts_per_tok', 'num_experts_per_token', 'moe_topk', 'top_k_experts')
    if isinstance(experts, (list, dict)):
        experts = None
    moe_ffn = first(c, 'moe_intermediate_size', 'expert_intermediate_size')
    if isinstance(moe_ffn, list):
        moe_ffn = sum(moe_ffn) / len(moe_ffn)
    if model_type.startswith('llama4'):  # routed expert + always-on shared expert, both intermediate_size
        mlp = layers_with_mlp * gated * hidden * (experts or 1) * dense_ffn + layers_with_mlp * gated * hidden * dense_ffn
    elif experts and moe_ffn:
        shared = (c.get('n_shared_experts') or c.get('num_shared_experts') or 0) * moe_ffn
        shared = c.get('shared_expert_intermediate_size') or c.get('moe_shared_expert_intermediate_size') or shared
        if c.get('enable_moe_block'):  # Gemma 4: dense MLP runs alongside the MoE block
            shared += dense_ffn
        dense_layers = c.get('first_k_dense_replace') or c.get('num_dense_layers') or 0
        mlp = (layers_with_mlp - dense_layers) * gated * hidden * (experts * moe_ffn + shared) + \
            dense_layers * gated * hidden * dense_ffn
    elif experts and c.get('num_local_experts'):  # Mixtral-style: experts share intermediate_size
        mlp = layers_with_mlp * gated * hidden * experts * dense_ffn
    else:
        mlp = layers_with_mlp * gated * hidden * dense_ffn
    return attention_layers * attention + mixer + mlp


if __name__ == '__main__':
    main()
