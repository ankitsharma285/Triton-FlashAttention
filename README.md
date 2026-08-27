# Triton-FlashAttention
GQA/MQA-extended FlashAttention-2 in Triton, with a KV-cache decode kernel — built on a public tutorial baseline, hardened for correctness (4 bugs found/fixed), and benchmarked against PyTorch SDPA on NVIDIA L4.

## Objective

This project extends a reference FlashAttention-2 implementation in Triton toward the kernel shapes actually used in LLM inference serving: grouped-query attention (GQA) and multi-query attention (MQA) support in prefill, and a separate decode-time (KV-cache) attention kernel for autoregressive generation. The baseline reference implements only standard multi-head attention (MHA) prefill; this repo hardens it into a correctness-verified, GQA-aware prefill-and-decode pair usable as groundwork for inference-optimization work (this repo, and separate research into speculative decoding, share that goal).

**Baseline / attribution:** built on top of the public tutorial implementation at [hkproj/triton-flash-attention](https://github.com/hkproj/triton-flash-attention) (Umar Jamil), itself based on the FlashAttention-2 algorithm (Tri Dao) and OpenAI's original Triton fused-attention tutorial. All correctness fixes and feature additions described below are on top of that baseline — see the changelog for what changed and why.

**Hardware:** developed and verified on an NVIDIA L4 GPU (Google Colab). L4 (Ada Lovelace) was chosen deliberately over a training-oriented card like A100 — it's positioned as an inference-serving GPU, with a bandwidth-to-compute ratio closer to what decode-time attention (the eventual target of this project) actually runs on.

