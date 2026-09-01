# Triton-FlashAttention
GQA/MQA-extended FlashAttention-2 in Triton, with a KV-cache decode kernel — built on a public tutorial baseline, hardened for correctness (4 bugs found/fixed), and benchmarked against PyTorch SDPA on NVIDIA L4.

## Objective

This project extends a reference FlashAttention-2 implementation in Triton toward the kernel shapes actually used in LLM inference serving: grouped-query attention (GQA) and multi-query attention (MQA) support in prefill, and a separate decode-time (KV-cache) attention kernel for autoregressive generation. The baseline reference implements only standard multi-head attention (MHA) prefill; this repo hardens it into a correctness-verified, GQA-aware prefill-and-decode pair usable as groundwork for inference-optimization work (this repo, and separate research into speculative decoding, share that goal).

**Baseline / attribution:** built on top of the public tutorial implementation at [hkproj/triton-flash-attention](https://github.com/hkproj/triton-flash-attention) (Umar Jamil), itself based on the FlashAttention-2 algorithm (Tri Dao) and OpenAI's original Triton fused-attention tutorial. All correctness fixes and feature additions described below are on top of that baseline — see the changelog for what changed and why.

**Hardware:** developed and verified on an NVIDIA L4 GPU (Google Colab). L4 (Ada Lovelace) was chosen deliberately over a training-oriented card like A100 — it's positioned as an inference-serving GPU, with a bandwidth-to-compute ratio closer to what decode-time attention (the eventual target of this project) actually runs on.

## Design choices

- **GQA/MQA via KV-head indexing, not kernel duplication.** Rather than writing a   separate kernel for GQA, the existing MHA kernel was extended with a `NUM_KV_HEADS` parameter and `index_kv_head = index_head // group_size` mapping. MHA is the special case `NUM_KV_HEADS == NUM_HEADS`, and MQA is the extreme case `NUM_KV_HEADS == 1` — both fall out of the same code path rather than being handled as separate branches, which keeps the kernel a strict superset of the original.
- **Explicit boundary handling over implicit block-pointer padding.** The original implementation relied on `SEQ_LEN` being an exact multiple of the block sizes. Every load/store now carries an explicit mask or `boundary_check` tied to `SEQ_LEN`, so arbitrary sequence lengths are supported rather than assumed away.
- **Scope decisions are stated explicitly rather than left implicit.** Where a feature is deliberately deferred (GQA backward, decode-time batching across requests with differing cache lengths), that's documented as a stated scope boundary — see "Deferred / explicitly out of scope" below — rather than left as a silent gap someone has to discover.

## Phase 0 — Harden the baseline

**Goal:** fix correctness/robustness issues in the reference implementation before
building features on top of it.

**Achieved:**
- Fixed `_attn_bwd_preprocess` to take explicit stride arguments instead of assuming a specific contiguous memory layout.
- Added explicit boundary masking to the forward kernel's non-causal and off-diagonal code paths (previously only the causal diagonal block was masked; non-causal attention and off-diagonal causal blocks would silently include zero-padded "phantom" key positions in the softmax denominator when `SEQ_LEN` wasn't divisible by the KV block size).
- Added explicit masking to every load and store in the backward kernels (`_attn_bwd_dq`, `_attn_bwd_dk_dv`) for non-divisible `SEQ_LEN`.
- Added a `HEAD_DIM` power-of-2 assertion (required by `tl.arange`).
- Verified forward and backward outputs against two independent references: an eager PyTorch implementation and `F.scaled_dot_product_attention`.

## Phase 1 — GQA / MQA (forward)

**Goal:** support grouped-query and multi-query attention in the forward kernel, matching the KV-head-sharing schemes used by essentially all current serving-scale LLMs (Llama 3, Mistral, Qwen, etc.).

**Achieved:**
- `NUM_KV_HEADS` parameter and head-group indexing added to the forward kernel. 
- Verified against both an eager-PyTorch reference (K/V expanded via `repeat_interleave`) and `F.scaled_dot_product_attention`, across group ratios {1, 2, 4, 8, 16} at `NUM_HEADS=16`, both causal and non-causal, plus a dedicated extreme-MQA case (`NUM_KV_HEADS=1`).
- Backward pass is explicitly *not* supported for GQA/MQA yet: `backward()` raises `NotImplementedError` when `NUM_HEADS != NUM_KV_HEADS`, rather than silently running the single-head gradient kernels and producing invalid gradients. This guard itself is covered by a test that confirms it actually fires. Standard MHA backward is unaffected and fully verified.

## Phase 2 — Decode kernel (single-request, KV cache, GQA-aware)

**Goal:** a structurally separate kernel for decode-time attention — one new query token attending over a KV cache that grows across generation steps — rather than reusing the prefill kernel with `SEQ_LEN=1`, since decode is memory-bandwidth-bound rather than compute-bound and has no causal-masking loop to run (every cached position is valid by construction).

**Achieved:**
- New `_attn_decode` kernel: one program per (batch, head), online-softmax   accumulation over the cache in `BLOCK_SIZE_KV`-sized chunks, GQA indexing reused   identically from the forward kernel (`index_kv_head = index_head // group_size`).
- `cache_len` passed as a runtime scalar, not `tl.constexpr` — required, since it   changes every decode step and baking it into the constexpr signature would force a   kernel recompile per token generated. Autotuning is keyed only on `(NUM_KV_HEADS, HEAD_DIM)`, explicitly excluding `cache_len`, for the same reason.
- Simple preallocated-cache write helper (`update_kv_cache`), kept separate from the attention kernel so cache bookkeeping and attention math can be tested independently.
- Verified via autoregressive equivalence: for a full sequence, prefill's output is compared against the concatenated outputs of calling the decode kernel once per token (writing each token into the cache immediately before decoding it). This is a stronger test than an isolated per-step check against a naive reference — it specifically exercises whether the online-softmax state carries correctly across cache-write/decode-call boundaries, not just whether one step's math is right.
- Test matrix: full GQA/MQA ratio sweep (`NUM_KV_HEADS` from 1 to `NUM_HEADS`) at a block-aligned length, plus a separate sweep of non-block-aligned total-token counts (1, 31, 33, 63, 65, 127, 257) at a fixed mid-range GQA ratio — the latter stresses the partial-final-chunk masking path, since almost every step in an autoregressive loop has a `cache_len` that isn't a multiple of `BLOCK_SIZE_KV`.

## Phase 3 — Benchmarking

**Goal:** quantify prefill and decode performance against honest baselines, on inference-relevant hardware, rather than asserting correctness alone is enough for a portfolio deliverable.

**Setup:** NVIDIA L4 (Google Colab), fp16, `NUM_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128` (a 4:1 GQA ratio matching real serving-scale models), `BATCH_SIZE=2`. Median latency over 30 timed repetitions (10 untimed warmup iterations excluded, to discard autotune/compile overhead), reported with standard deviation. Peak memory measured as the incremental delta over each function's own call (pre-existing resident tensors, e.g. Q/K/V or a shared KV cache, excluded from the measurement so it reflects only what that call itself allocates), with `torch.cuda.synchronize()` around every timed region.

### Prefill: `_attn_fwd` vs `F.scaled_dot_product_attention` (causal)

| Seq Len | Triton Latency (ms) | SDPA Latency (ms) | Peak Incremental Memory |
|---|---|---|---|
| 2048 | 1.42 ± 0.04 | 1.34 ± 0.03 | 32.50 MB (identical, both kernels) |
| 4096 | 5.53 ± 0.08 | 5.53 ± 0.07 | 65.00 MB (identical, both kernels) |
| 8192 | 20.83 ± 0.18 | 21.53 ± 0.21 | 130.00 MB (identical, both kernels) |

**Result:** comparable performance to SDPA, with no consistent winner across shapes — SDPA modestly faster at 2048 (~6%, just outside the combined error bars), tied at 4096, Triton modestly faster at 8192 (~3%, non-overlapping error bars). Memory is identical at every shape, which is expected: with `requires_grad=False` neither kernel needs backward-pass buffers, so both are paying only for the output tensor
(and the measured value matches that tensor's exact byte size, confirming the measurement methodology). This is not a claim that the from-scratch kernel beats a mature, heavily-optimized fused CUDA implementation — it's evidence that a correctness-hardened, GQA-extended kernel built from a public tutorial baseline reaches performance parity with PyTorch's native implementation at these shapes, while supporting a feature (GQA) the baseline didn't have.

### Decode: `_attn_decode` vs naive PyTorch eager baselines

Two eager baselines, to separate two independent effects: a fused-streaming-kernel vs. materializing-intermediate-tensors comparison (`Eager Slice`, using the same static preallocated cache as Triton), and a preallocated-cache vs. `torch.cat`-growth comparison (`Eager Dynamic`).

| Cache Len | Triton (ms) | Eager Slice (ms) | Eager Dynamic (ms) | Triton Mem | Eager Slice Mem | Eager Dynamic Mem |
|---|---|---|---|---|---|---|
| 128 | 0.087 | 0.269 | 0.283 | 0.02 MB | 4.05 MB | 4.50 MB |
| 512 | 0.094 | 0.272 | 0.284 | 0.02 MB | 16.14 MB | 18.00 MB |
| 2048 | 0.147 | 0.615 | 0.704 | 0.02 MB | 64.52 MB | 72.01 MB |
| 8192 | 0.515 | 2.928 | 3.429 | 0.02 MB | 258.02 MB | 288.01 MB |

**Result:** Triton's decode latency is ~3-7x lower than either eager baseline, growing from ~1.5x memory-bandwidth-consistent (peak memory constant, latency growing roughly with cache length — matching the expected memory-bandwidth-bound regime for decode) to a >5x gap at the longest cache length tested. Memory is the starker result: Triton's decode kernel uses a constant ~0.02MB (matching the size of the single-token output tensor — genuinely O(1) relative to cache length), while both eager baselines grow linearly, reaching 258-288MB at `cache_len=8192`.

**Scope of the comparison, stated explicitly:** this is a speedup over a *naive* eager PyTorch implementation (full GQA head expansion via `repeat_interleave`, materializing scores/probs tensors) — not a claim of beating the best possible PyTorch decode implementation. A hand-optimized eager version (e.g. using grouped batched matmul instead of `repeat_interleave`) would likely close some of this gap, though the fused-single-kernel-vs-multiple-eager-ops overhead would remain. The two eager variants (`Eager Slice` vs `Eager Dynamic`) differ only in whether the KV cache is a preallocated static buffer or grown via `torch.cat` each step; the small latency and memory gap between them is attributable specifically to the reallocation cost, isolated from the larger fused-kernel-vs-eager-ops effect.

**Reproducibility:** fixed seed (`torch.manual_seed(42)`), `torch.cuda.empty_cache()` + `reset_peak_memory_stats()` before each measured function's memory reading, `torch.cuda.synchronize()` around every timed call.

- **GQA/MQA backward pass** — planned as a later phase; requires accumulating gradients from multiple Q-head groups into shared dK/dV, which the current backward kernels' per-head indexing doesn't do. 
- **Batched/paged decode across requests with differing cache lengths** (à la PagedAttention) — meaningfully larger scope than single-request decode; noted here explicitly as a known gap rather than an oversight.
- **Fused rotary embeddings, sliding-window attention, attention bias (e.g. ALiBi)** — not implemented; the baseline this builds on doesn't have them either.

## Verification methodology

All correctness claims in this README are backed by tests that check the Triton kernel's output against an independent reference, not just "the code ran without crashing." Prefill (Phases 0-1) is checked against two independent references — an eager-mode PyTorch implementation and `F.scaled_dot_product_attention`. Decode (Phase 2) has no separate PyTorch reference implementation; it's instead checked for
autoregressive equivalence against this repo's own already-verified prefill kernel (see Phase 2 above), which is the strongest available reference short of writing a second independent decode implementation. Tolerance: `atol=1e-2, rtol=1e-2` (fp16 compute, fp32 softmax accumulation).
