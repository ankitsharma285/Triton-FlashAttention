import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# =====================================================================
# Phase 0 & 1: Prefill & Backward Kernels
# =====================================================================

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr, boundary_check=(1,))
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            causal_mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            boundary_mask = (start_kv + offs_kv[None, :]) < SEQ_LEN
            mask = causal_mask & boundary_mask
            QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            boundary_mask = (start_kv + offs_kv[None, :]) < SEQ_LEN
            QK_block = QK_block * softmax_scale + tl.where(boundary_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr, boundary_check=(0,))
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in [3, 4, 7]
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q, K, V, softmax_scale, M, O,
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of 2")

    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    group_size = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // group_size

    q_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    k_offset = index_batch.to(tl.int64) * stride_K_batch + index_kv_head.to(tl.int64) * stride_K_head
    v_offset = index_batch.to(tl.int64) * stride_V_batch + index_kv_head.to(tl.int64) * stride_V_head
    o_offset = index_batch.to(tl.int64) * stride_O_batch + index_head.to(tl.int64) * stride_O_head

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(SEQ_LEN, HEAD_DIM), strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(SEQ_LEN, HEAD_DIM), strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_KV, HEAD_DIM), order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(HEAD_DIM, SEQ_LEN), strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_SIZE_KV), order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset, shape=(SEQ_LEN, HEAD_DIM), strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr, boundary_check=(0,))

    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            4 - STAGE, offs_q, offs_kv, SEQ_LEN,
        )

    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            2, offs_q, offs_kv, SEQ_LEN,
        )

    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        NUM_KV_HEADS = K.shape[1]

        assert (HEAD_DIM & (HEAD_DIM - 1)) == 0, f"HEAD_DIM must be a power of 2, got {HEAD_DIM}"
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert NUM_HEADS % NUM_KV_HEADS == 0, "NUM_HEADS must be divisible by NUM_KV_HEADS"

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q=Q, K=K, V=V,
            softmax_scale=softmax_scale,
            M=M, O=O,
            stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1), stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0), stride_K_head=K.stride(1), stride_K_seq=K.stride(2), stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0), stride_V_head=V.stride(1), stride_V_seq=V.stride(2), stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0), stride_O_head=O.stride(1), stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            NUM_KV_HEADS=NUM_KV_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        return O


# =====================================================================
# Phase 2: Decode Kernel & Cache Management
# =====================================================================

def update_kv_cache(K_cache, V_cache, K_new, V_new, cache_len):
    """
    Writes K_new and V_new into the preallocated cache at sequence offset cache_len.
    Inputs:
        K_cache, V_cache: (BATCH_SIZE, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
        K_new, V_new:     (BATCH_SIZE, NUM_KV_HEADS, 1, HEAD_DIM)
        cache_len:        int, position to write to
    """
    K_cache[:, :, cache_len : cache_len + 1, :] = K_new
    V_cache[:, :, cache_len : cache_len + 1, :] = V_new


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_KV": BLOCK_SIZE_KV}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in [2, 3, 4]
        for num_warps in [2, 4]
    ],
    key=["NUM_KV_HEADS", "HEAD_DIM"],  # Explicitly excluding cache_len to avoid re-autotuning per step
)
@triton.jit
def _attn_decode(
    Q, K_cache, V_cache, softmax_scale, O,
    cache_len,
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    tl.static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be a power of 2")

    index_batch_head = tl.program_id(0)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    group_size = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // group_size

    q_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    k_offset = index_batch.to(tl.int64) * stride_K_batch + index_kv_head.to(tl.int64) * stride_K_head
    v_offset = index_batch.to(tl.int64) * stride_V_batch + index_kv_head.to(tl.int64) * stride_V_head
    o_offset = index_batch.to(tl.int64) * stride_O_batch + index_head.to(tl.int64) * stride_O_head

    offs_dim = tl.arange(0, HEAD_DIM)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Load 1D single-token Query vector: (HEAD_DIM,)
    Q_ptr = Q + q_offset + offs_dim * stride_Q_dim
    q = tl.load(Q_ptr)

    m_i = -float("inf")
    l_i = 1.0
    o = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Base pointers for K and V cache
    k_base_ptr = K_cache + k_offset
    v_base_ptr = V_cache + v_offset

    # Dynamic loop over cache_len
    for start_kv in range(0, cache_len, BLOCK_SIZE_KV):
        cols = start_kv + offs_kv
        mask_kv = cols < cache_len

        # Load K block: (BLOCK_SIZE_KV, HEAD_DIM)
        k_ptrs = k_base_ptr + cols[:, None] * stride_K_seq + offs_dim[None, :] * stride_K_dim
        k = tl.load(k_ptrs, mask=mask_kv[:, None], other=0.0)

        # q * K^T -> qk: (BLOCK_SIZE_KV,)
        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        qk = tl.where(mask_kv, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.math.exp(qk - m_ij)
        l_ij = tl.sum(p, 0)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # Load V block: (BLOCK_SIZE_KV, HEAD_DIM)
        v_ptrs = v_base_ptr + cols[:, None] * stride_V_seq + offs_dim[None, :] * stride_V_dim
        v = tl.load(v_ptrs, mask=mask_kv[:, None], other=0.0)

        o = o * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    o = o / l_i
    o_ptr = O + o_offset + offs_dim * stride_O_dim
    tl.store(o_ptr, o.to(O.type.element_ty))


def triton_decode(Q, K_cache, V_cache, cache_len, softmax_scale=None):
    """
    Python wrapper for _attn_decode kernel.
    Q: (BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)
    K_cache, V_cache: (BATCH_SIZE, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
    cache_len: int, total current valid tokens in cache (including newly appended token)
    """
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    NUM_KV_HEADS = K_cache.shape[1]
    assert SEQ_LEN == 1, f"Decode kernel expects single query token, got SEQ_LEN={SEQ_LEN}"

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    O = torch.empty_like(Q)

    grid = (BATCH_SIZE * NUM_HEADS,)

    _attn_decode[grid](
        Q=Q, K_cache=K_cache, V_cache=V_cache,
        softmax_scale=softmax_scale, O=O,
        cache_len=cache_len,
        stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1), stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
        stride_K_batch=K_cache.stride(0), stride_K_head=K_cache.stride(1), stride_K_seq=K_cache.stride(2), stride_K_dim=K_cache.stride(3),
        stride_V_batch=V_cache.stride(0), stride_V_head=V_cache.stride(1), stride_V_seq=V_cache.stride(2), stride_V_dim=V_cache.stride(3),
        stride_O_batch=O.stride(0), stride_O_head=O.stride(1), stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
        NUM_HEADS=NUM_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
    )
    return O


# =====================================================================
# Phase 2: Autoregressive Equivalence Testing
# =====================================================================

def test_autoregressive_equivalence(BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, TOTAL_TOKENS, HEAD_DIM, dtype=torch.float16):
    """
    Simulates sequence generation token-by-token using _attn_decode and compares
    the concatenated per-step outputs against full causal prefill execution.
    """
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    # 1. Generate full sequence data for Prefill Reference
    Q_full = torch.empty((BATCH_SIZE, NUM_HEADS, TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5)
    K_full = torch.empty((BATCH_SIZE, NUM_KV_HEADS, TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5)
    V_full = torch.empty((BATCH_SIZE, NUM_KV_HEADS, TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5)

    # Run Causal Prefill on the complete sequence
    ref_O_prefill = TritonAttention.apply(Q_full, K_full, V_full, True, softmax_scale)

    # 2. Simulate step-by-step decoding
    # Note: For production use, MAX_SEQ_LEN is over-provisioned (MAX_SEQ_LEN > cache_len).
    # Here MAX_SEQ_LEN = TOTAL_TOKENS for exact test boundary validation.
    K_cache = torch.zeros((BATCH_SIZE, NUM_KV_HEADS, TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")
    V_cache = torch.zeros((BATCH_SIZE, NUM_KV_HEADS, TOTAL_TOKENS, HEAD_DIM), dtype=dtype, device="cuda")

    decode_outputs = []

    for t in range(TOTAL_TOKENS):
        q_t = Q_full[:, :, t : t + 1, :]
        k_t = K_full[:, :, t : t + 1, :]
        v_t = V_full[:, :, t : t + 1, :]

        # Cache Update Helper
        update_kv_cache(K_cache, V_cache, k_t, v_t, cache_len=t)

        # Run Decode Kernel
        o_t = triton_decode(q_t, K_cache, V_cache, cache_len=t + 1, softmax_scale=softmax_scale)
        decode_outputs.append(o_t)

    decode_O_full = torch.cat(decode_outputs, dim=2)

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(ref_O_prefill, decode_O_full, atol=atol, rtol=rtol), (
        f"Autoregressive mismatch! (NUM_HEADS={NUM_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, "
        f"TOTAL_TOKENS={TOTAL_TOKENS}, HEAD_DIM={HEAD_DIM})"
    )
    print(f"Passed Autoregressive Equivalence: Heads={NUM_HEADS}, KV_Heads={NUM_KV_HEADS}, Tokens={TOTAL_TOKENS}")


if __name__ == "__main__":
    print("--- Running Phase 2 Decode Kernel Verification ---\n")

    # 1. Sweep GQA / MQA Ratios (g_size = 1, 2, 4, 8, 16)
    for g_size in [1, 2, 4, 8, 16]:
        num_heads = 16
        num_kv_heads = num_heads // g_size
        test_autoregressive_equivalence(
            BATCH_SIZE=2, NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, TOTAL_TOKENS=128, HEAD_DIM=64
        )

    # 2. Test Non-Block-Aligned Token Lengths (stress partial-chunk masking)
    for token_len in [1, 31, 33, 63, 65, 127, 257]:
        test_autoregressive_equivalence(
            BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=2, TOTAL_TOKENS=token_len, HEAD_DIM=128
        )

    print("\nPHASE 2 DECODE KERNEL VERIFIED AND PASSED SUCCESSFULLY")