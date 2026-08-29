# Triton-FlashAttention
GQA/MQA-extended FlashAttention-2 in Triton, with a KV-cache decode kernel — built on a public tutorial baseline, hardened for correctness (4 bugs found/fixed), and benchmarked against PyTorch SDPA on NVIDIA L4.

## Objective

This project extends a reference FlashAttention-2 implementation in Triton toward the kernel shapes actually used in LLM inference serving: grouped-query attention (GQA) and multi-query attention (MQA) support in prefill, and a separate decode-time (KV-cache) attention kernel for autoregressive generation. The baseline reference implements only standard multi-head attention (MHA) prefill; this repo hardens it into a correctness-verified, GQA-aware prefill-and-decode pair usable as groundwork for inference-optimization work (this repo, and separate research into speculative decoding, share that goal).

**Baseline / attribution:** built on top of the public tutorial implementation at [hkproj/triton-flash-attention](https://github.com/hkproj/triton-flash-attention) (Umar Jamil), itself based on the FlashAttention-2 algorithm (Tri Dao) and OpenAI's original Triton fused-attention tutorial. All correctness fixes and feature additions described below are on top of that baseline — see the changelog for what changed and why.

**Hardware:** developed and verified on an NVIDIA L4 GPU (Google Colab). L4 (Ada Lovelace) was chosen deliberately over a training-oriented card like A100 — it's positioned as an inference-serving GPU, with a bandwidth-to-compute ratio closer to what decode-time attention (the eventual target of this project) actually runs on.

## Design choices

- **GQA/MQA via KV-head indexing, not kernel duplication.** Rather than writing a
  separate kernel for GQA, the existing MHA kernel was extended with a
  `NUM_KV_HEADS` parameter and `index_kv_head = index_head // group_size` mapping.
  MHA is the special case `NUM_KV_HEADS == NUM_HEADS`, and MQA is the extreme case
  `NUM_KV_HEADS == 1` — both fall out of the same code path rather than being handled
  as separate branches, which keeps the kernel a strict superset of the original.
- **Explicit boundary handling over implicit block-pointer padding.** The original
  implementation relied on `SEQ_LEN` being an exact multiple of the block sizes.
  Every load/store now carries an explicit mask or `boundary_check` tied to `SEQ_LEN`,
  so arbitrary sequence lengths are supported rather than assumed away.
- **Scope decisions are stated explicitly rather than left implicit.** Where a
  feature is deliberately deferred (GQA backward, decode-time batching across
  requests with differing cache lengths), that's documented as a stated scope
  boundary — see "Deferred / explicitly out of scope" below — rather than left as a
  silent gap someone has to discover.

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


