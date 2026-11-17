"""Reuse-driven KV cache economics using Mooncake request traces (no API calls).

Traces (Moonshot/Kimi Mooncake FAST'25 release, downloaded with hashes to
data/kremlinology-mooncake-traces): each request lists the hashes of its 512-token
prompt blocks and a millisecond timestamp, over about one hour.

Policy simulated per (model, tier, TTL): an infinite-capacity cache keeping a block for TTL
after its last use. A block use within TTL is a hit (pay reload from the tier); otherwise a
miss (pay recompute for that block, then write it to the tier). Holding cost is charged for
the time a block is retained. Costs per block:
  bytes   = 512 x KV bytes per token, plus one recurrent-state checkpoint per block for
            linear-attention/SSM hybrids (resuming a hybrid needs the state at that boundary)
  recompute = marginal prefill FLOPs of the block at its position / (peak x MFU) x GPU $/s
  reload  = bytes / tier bandwidth x GPU $/s (stalled) or 0 (overlapped variant)
  write   = SSD endurance: drive price / rated TB written, per byte written
HBM is charged at rental share x pressure factor (1 = fully contended, 0.1 = mostly idle).
Hardware variants: H100 bf16 (989 TF), H100 fp8 (1979 TF), B200 fp8 (4500 TF dense).
The one-hour trace window censors reuse beyond an hour (longer TTLs see no extra hits).
"""
import json
import sys
from bisect import bisect_right
from pathlib import Path
import os as _os
DATA_ROOT = _os.environ.get('KREMLINOLOGY_DATA', 'data')  # where the release archives are extracted

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['text.parse_math'] = False  # labels contain literal dollar signs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import kv_cache_economics_v1 as base  # noqa: E402

TRACES = Path(DATA_ROOT, 'kremlinology-mooncake-traces')
OUT = ROOT / 'reports/kv-cache-economics'
BLOCK = 512
SURFACE, INK, INK2 = base.SURFACE, base.INK, base.INK2

HARDWARE = {  # name: (peak FLOPS, $/hour, HBM GB)
    'H100 bf16': (989e12, 2.50, 80), 'H100 fp8': (1979e12, 2.50, 80), 'B200 fp8': (4500e12, 5.00, 192),
}
MFU = 0.45
TIERS = {  # $/GB-hour, bandwidth B/s, endurance $/GB written, description
    'HBM': dict(rate=None, bandwidth=None, write=0.0),
    'Host DRAM': dict(rate=2.75e-4, bandwidth=50e9, write=0.0),
    'NVMe RAID x4': dict(rate=2.52e-6, bandwidth=4 * 6.3e9, write=300 / 2.4e6),  # $300 / 2,400 TBW (SN850X 4 TB)
    'HDD RAID5 x4': dict(rate=8.99e-7, bandwidth=0.38e9, write=0.0),
    # PUT requests (~$0.005 per 1,000) are negligible per GB at these block sizes.
    'S3-class object store': dict(rate=3.15e-5, bandwidth=5e9, write=0.0),
}
MODELS = [('Llama-3.3-70B (GQA)', 'unsloth--Llama-3.3-70B-Instruct'),
          ('DeepSeek-V3 (MLA)', 'deepseek-ai--DeepSeek-V3'),
          ('gpt-oss-120b (sliding)', 'openai--gpt-oss-120b'),
          ('Qwen3.5-27B (linear hybrid)', 'Qwen--Qwen3.5-27B'),
          ('Llama-3.1-8B (GQA, small)', 'meta-llama--Llama-3.1-8B-Instruct')]
TTLS = [0] + [10 ** (k / 4) for k in range(0, 15)]  # seconds: 1 s .. ~1.8 h


def load_trace(name):
    rows = [json.loads(line) for line in (TRACES / name).read_text().splitlines()]
    rows.sort(key=lambda r: r['timestamp'])
    return rows


def block_events(rows):
    """(time_s, position_index) for every prompt block use, grouped by hash."""
    uses = {}
    for r in rows:
        for position, block in enumerate(r['hash_ids']):
            uses.setdefault(block, []).append((r['timestamp'] / 1000, position))
    return uses


def reuse_gaps(uses):
    gaps = []
    for events in uses.values():
        times = [t for t, _ in events]
        gaps += [b - a for a, b in zip(times, times[1:])]
    return sorted(gaps)


def block_bytes(spec, checkpoint=True):
    """Bytes stored per 512-token block, at long context (window layers keep W tokens of the prefix)."""
    window_share = spec['window_bytes'] * BLOCK * (min(spec['window'], BLOCK) / BLOCK) if spec['window'] else 0
    state = spec['state_bytes'] if checkpoint else 0.0
    return spec['full_bytes'] * BLOCK + window_share + state


def block_flops(spec, position):
    """Marginal prefill FLOPs for the block at index `position` (tokens attend to all earlier ones)."""
    start, end = position * BLOCK, (position + 1) * BLOCK
    span_sum = (start + end - 1) * BLOCK / 2  # sum of causal spans over the block's tokens
    attention = 2 * spec['attn_full'] * span_sum
    if spec['window']:
        attention += 2 * spec['attn_window'] * sum(min(p, spec['window']) for p in range(start, end, 16)) * 16
    return 2 * spec['active_params'] * BLOCK + attention + spec['linear_flops'] * BLOCK


def simulate(uses, spec, tier, ttl, hardware='H100 bf16', pressure=1.0, overlapped=False, kv_fp8=False,
             checkpoint=True):
    peak, gpu_hour, hbm = HARDWARE[hardware]
    gpu_second = gpu_hour / 3600
    t = TIERS[tier]
    size = block_bytes(spec, checkpoint) / 1e9 * (0.5 if kv_fp8 else 1.0)  # GB per block
    rate = gpu_hour / hbm * pressure if t['rate'] is None else t['rate']
    reload = 0.0 if (t['bandwidth'] is None or overlapped) else size * 1e9 / t['bandwidth'] * gpu_second
    total = recompute_all = hold_hours = 0.0
    hits = misses = 0
    for events in uses.values():
        last = None
        for time_s, position in events:
            cost = block_flops(spec, position) / (peak * MFU) * gpu_second
            recompute_all += cost
            if last is not None and ttl > 0 and time_s - last <= ttl:
                hits += 1
                total += reload
                hold_hours += (time_s - last) / 3600
            else:
                misses += 1
                total += cost + (size * t['write'] if ttl > 0 else 0)
                if last is not None and ttl > 0:
                    hold_hours += ttl / 3600  # retained for a full TTL before expiring
            last = time_s
        if last is not None and ttl > 0:
            hold_hours += ttl / 3600  # final retention period
    total += hold_hours * size * rate
    return dict(cost=total, recompute_all=recompute_all, saving=1 - total / recompute_all,
                hit_rate=hits / max(1, hits + misses))


def figure_reuse(traces):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    base.style(ax, 'How soon is a 512-token prompt block reused? (Mooncake traces)',
               xlabel='time since the block was last used (log scale)', ylabel='share of block uses that are reuses')
    colors = ['#2a78d6', '#eb6834', '#1baf7a']
    table = []
    for (name, gaps, total), color in zip(traces, colors):
        xs = [10 ** (k / 10) for k in range(-10, 36)]
        ys = [bisect_right(gaps, x) / total for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, label=f'{name} ({len(gaps) / total:.0%} of block uses are reuses)')
        table += [dict(trace=name, seconds=x, share_reused_within=y) for x, y in zip(xs, ys)]
    marks = [(1, '1 s'), (60, '1 min'), (300, '5 min'), (3600, '1 h')]
    ax.set_xscale('log')
    ax.set_xticks([m for m, _ in marks], [lbl for _, lbl in marks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)
    ax.text(0.99, 0.02, 'traces span ~1 hour; longer reuse is censored', transform=ax.transAxes, ha='right',
            fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig10_reuse_distance.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig10_reuse_distance.csv', table)


def figure_savings(uses, specs):
    tiers = list(TIERS)
    colors = dict(zip(tiers, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4.6), sharey=True)
    table = []
    for ax, (label, name) in zip(axes, MODELS):
        spec = specs[name]
        base.style(ax, label, xlabel='cache TTL', ylabel='saving vs always recomputing' if ax is axes[0] else None)
        for tier in tiers:
            ys = [simulate(uses, spec, tier, ttl)['saving'] for ttl in TTLS[1:]]
            ax.plot(TTLS[1:], ys, color=colors[tier], linewidth=2, label=tier)
            best = max(range(len(ys)), key=lambda i: ys[i])
            ax.plot(TTLS[1 + best], ys[best], 'o', color=colors[tier], markersize=6, markeredgecolor=SURFACE)
            table += [dict(model=label, tier=tier, ttl_s=t, saving=y) for t, y in zip(TTLS[1:], ys)]
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xticks([1, 60, 3600], ['1 s', '1 min', '1 h'])
        ax.minorticks_off()
        ax.set_ylim(-0.6, 0.8)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('Cost saving from caching on each tier vs TTL (Mooncake conversation trace, H100 bf16, stalled reload; '
                 'dot = best TTL)', x=0.01, ha='left', color=INK, fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / 'fig11_tier_savings_vs_ttl.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig11_tier_savings_vs_ttl.csv', table)


def figure_checkpoints(specs):
    hybrids = [('Qwen3.5-27B', 'Qwen--Qwen3.5-27B'), ('Qwen3-Next-80B', 'Qwen--Qwen3-Next-80B-A3B-Instruct'),
               ('Granite-4.0-H-small', 'ibm-granite--granite-4.0-h-small'),
               ('Nemotron-Nano-9B-v2', 'nvidia--NVIDIA-Nemotron-Nano-9B-v2')]
    intervals = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    base.style(ax, 'Hybrid models: cache bytes per token if a state checkpoint is stored every N tokens',
               xlabel='checkpoint interval N (tokens, log scale)', ylabel='bytes per cached token (log scale)')
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    table = []
    for (label, name), color in zip(hybrids, colors):
        s = specs[name]
        ys = [s['full_bytes'] + s['state_bytes'] / n for n in intervals]
        ax.plot(intervals, ys, color=color, linewidth=2, marker='o', markersize=4, markeredgecolor=SURFACE,
                label=f"{label} (state {s['state_bytes'] / 2 ** 20:.0f} MiB)")
        table += [dict(model=label, interval=n, bytes_per_token=y) for n, y in zip(intervals, ys)]
    llama = specs['unsloth--Llama-3.3-70B-Instruct']['full_bytes']
    gqa8 = specs['meta-llama--Llama-3.1-8B-Instruct']['full_bytes']
    for value, text in ((llama, 'Llama-3.3-70B (GQA, no state)'), (gqa8, 'Llama-3.1-8B (GQA, no state)')):
        ax.axhline(value, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(intervals[-1], value * 1.08, text, ha='right', fontsize=8, color=INK2)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(intervals, [f'{n // 1024}k' if n >= 1024 else str(n) for n in intervals])
    base.byte_ticks(ax, 'y', 2 ** 12, 2 ** 22)
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig12_hybrid_checkpoint_bytes.png', dpi=160)
    plt.close(fig)
    base.write_csv(OUT / 'fig12_hybrid_checkpoint_bytes.csv', table)


def main():
    specs = {name: base.spec(name) for name in base.ACTIVE_PARAMS}
    traces, uses_by = [], {}
    for name in ('conversation_trace.jsonl', 'toolagent_trace.jsonl', 'synthetic_trace.jsonl'):
        uses = block_events(load_trace(name))
        gaps = reuse_gaps(uses)
        total = sum(len(v) for v in uses.values())
        traces.append((name.split('_')[0], gaps, total))
        uses_by[name.split('_')[0]] = uses
    figure_reuse(traces)
    figure_checkpoints(specs)
    figure_savings(uses_by['conversation'], specs)
    results = []

    def best(trace, label, name, tier, **variant):
        runs = [simulate(uses_by[trace], specs[name], tier, ttl, **variant) | dict(ttl_s=ttl) for ttl in TTLS[1:]]
        top = max(runs, key=lambda run: run['saving'])
        if top['saving'] < 0:  # not caching at all (TTL 0) saves exactly zero
            top = dict(saving=0.0, hit_rate=0.0, ttl_s=0)
        defaults = dict(hardware='H100 bf16', pressure=1.0, overlapped=False, kv_fp8=False, checkpoint=True)
        results.append(dict(trace=trace, model=label, tier=tier, **(defaults | variant), best_ttl_s=top['ttl_s'],
                            best_saving=top['saving'], hit_rate=top['hit_rate']))

    tier_configs = [(tier, {}) for tier in TIERS] + [('HBM', dict(pressure=0.1))]
    for trace in uses_by:  # baseline on every trace
        for label, name in MODELS:
            for tier, extra in tier_configs:
                best(trace, label, name, tier, **extra)
    variants = [dict(hardware='H100 fp8'), dict(hardware='B200 fp8'), dict(kv_fp8=True), dict(overlapped=True)]
    for label, name in MODELS:  # sensitivity variants on the conversation trace
        for variant in variants + ([dict(checkpoint=False)] if specs[name]['state_bytes'] else []):
            for tier, extra in tier_configs:
                best('conversation', label, name, tier, **(extra | variant))
    base.write_csv(OUT / 'reuse_tier_results.csv', results)
    for r in results:
        if r['trace'] == 'conversation' and r['hardware'] == 'H100 bf16' and not r['kv_fp8'] and \
                not r['overlapped'] and r['checkpoint'] and r['pressure'] == 1.0:
            print(f"{r['model']:30s} {r['tier']:22s} best TTL {r['best_ttl_s']:8.0f}s saving {r['best_saving']:6.1%} "
                  f"hit {r['hit_rate']:.1%}")


if __name__ == '__main__':
    main()
