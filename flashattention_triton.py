import torch
import triton
import triton.language as tl
import torch.nn.functional as F


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


@triton.jit
def _attn_bwd_preprocess(
    O, dO, D,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    o_offset = index_batch.to(tl.int64) * stride_O_batch + index_head.to(tl.int64) * stride_O_head

    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset, shape=(SEQ_LEN, HEAD_DIM), strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO + o_offset, shape=(SEQ_LEN, HEAD_DIM), strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0),
    )

    O_block = tl.load(O_block_ptr, boundary_check=(0,))
    dO_block = tl.load(dO_block_ptr, boundary_check=(0,)).to(tl.float32)

    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block, mask=offs_q < SEQ_LEN)


@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr, STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_q = tl.program_id(0)

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q[:, None] < SEQ_LEN

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim, mask=mask_q, other=0.0)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim, mask=mask_q, other=0.0)

    M_block = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)[:, None]
    Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0)

    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    curr_kv = 0
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_KV)
    for blk_idx in range(num_steps):
        offs_kv_curr = curr_kv + offs_kv
        mask_kv = offs_kv_curr[None, :] < SEQ_LEN
        

        K_T_block = tl.load(kT_ptrs, mask=mask_kv, other=0.0)
        # FIX: Reuse mask_kv (shape: 1, BLOCK_KV) to match vT_ptrs shape (HEAD_DIM, BLOCK_KV)
        V_T_block = tl.load(vT_ptrs, mask=mask_kv, other=0.0)
        
        
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            mask_causal = offs_q[:, None] >= offs_kv_curr[None, :]
            P_block = tl.where(mask_causal & mask_kv, P_block, 0.0)
        else:
            P_block = tl.where(mask_kv, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block, mask=mask_q)


@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr, STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    mask_kv = offs_kv[:, None] < SEQ_LEN

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim, mask=mask_kv, other=0.0)
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim, mask=mask_kv, other=0.0)

    offs_q = tl.arange(0, BLOCK_Q)
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    curr_q = 0
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_Q)
    for blk_idx in range(num_steps):
        offs_q_curr = curr_q + offs_q
        mask_q = offs_q_curr[None, :] < SEQ_LEN

        qT_block = tl.load(qT_ptrs, mask=offs_q_curr[None, :] < SEQ_LEN, other=0.0)
        m = tl.load(M + offs_q_curr, mask=offs_q_curr < SEQ_LEN, other=0.0)

        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            mask_causal = (offs_q_curr[None, :] >= offs_kv[:, None])
            P_T_block = tl.where(mask_causal & mask_q, P_T_block, 0.0)
        else:
            P_T_block = tl.where(mask_q, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs, mask=offs_q_curr[:, None] < SEQ_LEN, other=0.0)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offs_q_curr, mask=offs_q_curr < SEQ_LEN, other=0.0)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block, mask=mask_kv)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block, mask=mask_kv)

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

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        NUM_HEADS = Q.shape[1]
        NUM_KV_HEADS = K.shape[1]

        if NUM_HEADS != NUM_KV_HEADS:
            raise NotImplementedError("Backward for GQA/MQA is scheduled for Phase 4.")

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (triton.cdiv(SEQ_LEN, BLOCK_SIZE_MACRO), BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)

        _attn_bwd_preprocess[preprocess_grid](
            O=O, dO=dO, D=D,
            stride_O_batch=O.stride(0), stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (triton.cdiv(SEQ_LEN, BLOCK_SIZE_MACRO), 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1),
            stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO, BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        _attn_bwd_dq[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1),
            stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO, BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16, check_bwd=False):
    assert (HEAD_DIM & (HEAD_DIM - 1)) == 0, "HEAD_DIM must be power of 2"

    Q = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5).requires_grad_(check_bwd)
    K = torch.empty((BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5).requires_grad_(check_bwd)
    V = torch.empty((BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(0.0, 0.5).requires_grad_(check_bwd)

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q) if check_bwd else None

    # Reference 1: Eager PyTorch Attention
    group_size = NUM_HEADS // NUM_KV_HEADS
    K_ref = K.repeat_interleave(group_size, dim=1)
    V_ref = V.repeat_interleave(group_size, dim=1)

    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K_ref.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK[:SEQ_LEN, :SEQ_LEN] == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).to(dtype)
    ref_O_eager = torch.matmul(P, V_ref)

    if check_bwd:
        ref_O_eager.backward(dO)
        ref_dQ, Q.grad = Q.grad.clone(), None
        ref_dK, K.grad = K.grad.clone(), None
        ref_dV, V.grad = V.grad.clone(), None

    # Reference 2: SDPA
    ref_O_sdpa = F.scaled_dot_product_attention(
        Q, K_ref, V_ref, is_causal=causal, scale=softmax_scale
    )

    # Triton Execution
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
    if check_bwd:
        tri_out.backward(dO)
        tri_dQ, Q.grad = Q.grad.clone(), None
        tri_dK, K.grad = K.grad.clone(), None
        tri_dV, V.grad = V.grad.clone(), None

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(ref_O_eager, tri_out, atol=atol, rtol=rtol), f"Mismatch Forward Eager (causal={causal}, SEQ_LEN={SEQ_LEN})"
    assert torch.allclose(ref_O_sdpa, tri_out, atol=atol, rtol=rtol), f"Mismatch Forward SDPA (causal={causal}, SEQ_LEN={SEQ_LEN})"

    if check_bwd:
        assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol), "Mismatch Backward dQ"
        assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol), "Mismatch Backward dK"
        assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol), "Mismatch Backward dV"


if __name__ == "__main__":
    # Phase 0: Non-causal & Causal MHA Backward on non-divisible SEQ_LEN
    test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=8, SEQ_LEN=4000, HEAD_DIM=64, causal=False, check_bwd=True)
    test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=8, SEQ_LEN=4000, HEAD_DIM=64, causal=True, check_bwd=True)
    
    # Non-power-of-2 / non-block-aligned odd sequence length
    test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=8, SEQ_LEN=3557, HEAD_DIM=64, causal=False, check_bwd=False)

    # Phase 1: GQA & MQA Forward tests across group ratios (1, 2, 4, 8, 16)
    for g_size in [1, 2, 4, 8, 16]:
        num_heads = 16
        num_kv_heads = num_heads // g_size
        test_op(BATCH_SIZE=2, NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, SEQ_LEN=2048, HEAD_DIM=128, causal=True)
        test_op(BATCH_SIZE=2, NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, SEQ_LEN=2048, HEAD_DIM=128, causal=False)

    # Explicit MQA (NUM_KV_HEADS=1) verification
    test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=1, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=1, SEQ_LEN=1024, HEAD_DIM=64, causal=False)

    # Phase 1 Guard Check: Verify GQA backward raises NotImplementedError
    try:
        test_op(BATCH_SIZE=2, NUM_HEADS=8, NUM_KV_HEADS=2, SEQ_LEN=512, HEAD_DIM=64, causal=True, check_bwd=True)
        assert False, "Expected NotImplementedError for GQA backward pass, but it ran without throwing."
    except NotImplementedError:
        print("Verified GQA backward guard (NotImplementedError raised as expected).")

    print("\nPHASE 0 & PHASE 1 VERIFIED AND PASSED SUCCESSFULLY")