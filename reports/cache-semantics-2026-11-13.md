# Prompt-cache semantics: breakpoint caches vs automatic prefix caches (2026-11-13)

**How it was run:** three paid probes through OpenRouter, pinned providers, $0.645 reported in total.
- v3 (`experiments/cache_semantics_probe_v3.py`, $0.41) repeats v2's key cells, sweeps Claude's default lifetime at 240–330 s, and adds Claude Sonnet 5.5 and Opus 5.5 and GPT-6.1 Terra and Sol.
- Figure: `reports/cache-semantics-2026-11-13.png` (`experiments/cache_semantics_figure_v1.py`, pools v1–v3).
- v1 used random word filler; v2 used wikitext prose, because v1's word salad tripped Claude's content filter on 9 requests.
- Scripts: `experiments/cache_semantics_probe_v1.py`, `experiments/cache_semantics_probe_v2.py`, `experiments/cache_semantics_analysis_v1.py`.
- Raw evidence: `data/kremlinology-cache-semantics-v{1,2,3}`.
- Each test uses fresh text (random nonce first). Cache reads and writes come from OpenRouter's usage fields.

## Results

| Test | Claude Haiku 5.5 / 4.5 (Anthropic, Vertex) | GPT-5.6 Luna, GPT-6 Luna, GPT-6 Sol | GPT-5 nano, GPT-5.4 nano | DeepSeek-V4 Flash (DeepInfra) |
| --- | --- | --- | --- | --- |
| repeat without breakpoint | **no cache** | whole prompt cached (written first) | cached in 128-token units | cached in 256-token units |
| repeat with breakpoint | whole prefix cached | — | — | — |
| first half shared, then diverges | **0**, full rewrite | **0**, full rewrite | **half cached** (2,816–3,200 = 22–25 × 128) | half cached (3,328 = 13 × 256) |
| follow-up chat turn (previous prompt + reply + new message) | previous prompt cached | previous prompt cached exactly (+23 written) | cached (128-token units) | — |
| cached prefix + new block, breakpoint at the end | earlier prefix read (look-back to a block boundary) | 0 | — | prefix cached |
| cached prefix + new block, no breakpoint | **nothing looked up** (0 read, 0 written) | 0 | — | prefix cached |
| edit shortly before the breakpoint | 0 | 0 | — | prefix up to the edit cached |
| charges cache writes | yes (1.25×) | yes (catalog lists write prices) | no | no |
| minimum cached length | Haiku 5.5 ≤ 2.1k tokens; Haiku 4.5 > 2.4k (caches at 4.8k) | ≤ 1.8k (0.9k not cached) | 3.6k cached; 1.8k missed once | 0.8k cached |
| lifetime | default 4 min hit, 5.5 and 7 min miss (~5 min); `ttl: 1h` 7 min hit | ≥ 7 min (GPT-6 Luna) | — | — |

## Findings

1. **Anthropic only resumes at breakpoints.**
   - A request that shares the first half of a cached prompt gets nothing, though the cached tokens up to the divergence are identical.
   - A request with no breakpoint does no lookup at all.
   - Edits anywhere before the breakpoint invalidate everything.
   - The only flexibility is a look-back to earlier block boundaries where a previous breakpoint was written.

   With plain full-attention, block-paged KV (vLLM-style, as GPT-5.x nano and DeepSeek show here), any shared prefix can be reused in fixed blocks. So requiring breakpoints means the cache can't be resumed at arbitrary positions. That is consistent with state that has to be saved at chosen boundaries (sliding-window or recurrent layers, compressed KV), or with caches persisted to another tier at those points. It does not identify which.
2. **OpenAI changed its caching between model generations.**
   - GPT-5 nano and GPT-5.4 nano reuse arbitrary prefixes in 128-token units and charge no writes.
   - GPT-5.6 Luna, GPT-6 Luna and GPT-6 Sol charge cache writes, reuse nothing from a half-shared prompt, and resume exactly where an earlier request ended (a follow-up turn reuses the whole previous prompt).

   This is the same boundary-only behaviour as Anthropic, with the previous request's end acting as an implicit breakpoint.
3. **The same model behaves the same on different infrastructure.** Claude Haiku 5.5 on Anthropic and on Google Vertex matched on every test that completed on both.
4. **Tokenizers.** The same text was about 16k tokens on Claude Haiku 5.5, against about 9k on Haiku 4.5 and about 6.7k on OpenAI models. The Claude 5.5 tokenizer is roughly 1.8× more verbose than 4.5's on this text.
5. **Reporting glitches.** Two OpenAI responses reported 0 prompt tokens. Nine Claude requests ended with `content_filter` on random word-salad text.

## v3 replication (2026-11-13)

**Shared first half, then new text:**

| model | prefix reused |
|---|---|
| Claude Haiku 5.5 | 0 in 6 of 6 trials (v1–v3) |
| Sonnet 5.5 / Opus 5.5 | 0 in 2 of 2 each |
| GPT-5.6 / GPT-6 / GPT-6.1 | 0 in 12 of 12 |
| GPT-5 / 5.4 nano | about half in 3 of 4; one complete miss |
| DeepSeek-V4 Flash | half in 1 of 2; one complete miss |

**Other tests:**
- **Sonnet and Opus 5.5 behave like Haiku:** no cache without a breakpoint, a full hit on a breakpoint repeat, and the previous prompt read on a follow-up turn. GPT-6.1 Terra and Sol behave like GPT-6.
- **Claude Haiku 5.5's default lifetime:**

  | delay | hits |
  |---|---|
  | 240 s | 3 of 3 |
  | 270 s | 2 of 2 |
  | 300 s | 2 of 2 |
  | 330 s | 2 of 3 (v2 missed) |
  | 420 s | 0 of 2 |

  Sonnet 5.5 hit at 300 s. This corrects the "~5 min" reading above: the cache survives at least 5.5 minutes in most trials and is gone by 7 minutes.
- **0-prompt-token reports:** 5 of 96 GPT-5.x/6.x responses across v1–v3, all on GPT-6 or GPT-6.1.

## Caveats

- One to six trials per cell.
- Usage counters are the providers' reports, relayed by OpenRouter.
- The architectural reading is an inference consistent with the behaviour, not an identification.
