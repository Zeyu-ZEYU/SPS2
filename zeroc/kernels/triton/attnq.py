import math

import pycuda.autoinit
import pycuda.driver as drv
import torch
import triton
import triton.language as tl
from triton.runtime import driver

qk_configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in ([3, 4, 7])
    for w in [4, 8]
]


def qk_conf_filter(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


pv_configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [128]
    for s in ([3, 4, 7])
    for w in [4, 8]
]


def pv_conf_filter(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(qk_conf_filter, qk_configs)), key=["M", "N", "HEAD_DIM"])
@triton.jit
def kernel_attnq_qk(
    Q,
    K,
    S,
    Qa,
    Qb,
    Ka,
    Kb,
    # SMAX,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,  #
    stride_qabz,
    stride_qabh,
    stride_qabm,
    stride_kabz,
    stride_kabh,
    stride_kabn,
    # stride_smaxz,
    # stride_smaxh,
    # stride_smaxm,
    Z,
    H,
    M,
    N,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    s_offset = off_z.to(tl.int64) * stride_sz + off_h.to(tl.int64) * stride_sh
    qab_offset = off_z.to(tl.int64) * stride_qabz + off_h.to(tl.int64) * stride_qabh
    kab_offset = off_z.to(tl.int64) * stride_kabz + off_h.to(tl.int64) * stride_kabh
    # smax_offset = off_z.to(tl.int64) * stride_smaxz + off_h.to(tl.int64) * stride_smaxh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + s_offset,
        shape=(M, N),
        strides=(stride_sm, stride_sn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    Qa_block_ptr = tl.make_block_ptr(
        base=Qa + qab_offset,
        shape=(M,),
        strides=(stride_qabm,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    Qb_block_ptr = tl.make_block_ptr(
        base=Qb + qab_offset,
        shape=(M,),
        strides=(stride_qabm,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    Ka_block_ptr = tl.make_block_ptr(
        base=Ka + kab_offset,
        shape=(N,),
        strides=(stride_kabn,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    Kb_block_ptr = tl.make_block_ptr(
        base=Kb + kab_offset,
        shape=(N,),
        strides=(stride_kabn,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    # SMAX_block_ptr = tl.make_block_ptr(
    #     base=SMAX + smax_offset,
    #     shape=(M,),
    #     strides=(stride_smaxm,),
    #     offsets=(start_m * BLOCK_M,),
    #     block_shape=(BLOCK_M,),
    #     order=(0,),
    # )

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    qsum = tl.sum(q.to(tl.float16), 1)
    qa = tl.load(Qa_block_ptr)
    qb = tl.load(Qb_block_ptr)
    # smax = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float16)

    # loop over k and update result
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        s = tl.dot(q, k, out_dtype=tl.int32)
        s_fp16 = s.to(tl.float16)
        ka = tl.load(Ka_block_ptr)
        kb = tl.load(Kb_block_ptr)
        ksum = tl.sum(k.to(tl.float16), 0)
        real = (
            qa[:, None] * ka[None, :] * s_fp16
            + (qa * qsum)[:, None] * kb[None, :]
            + qb[:, None] * (ka * ksum)[None, :]
            + (qb * HEAD_DIM)[:, None] * kb[None, :]
        )
        # smax = tl.maximum(smax, tl.max(s, 1).to(tl.float16))
        tl.store(S_block_ptr, real, boundary_check=(0, 1))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        S_block_ptr = tl.advance(S_block_ptr, (0, BLOCK_N))
        Ka_block_ptr = tl.advance(Ka_block_ptr, (BLOCK_N,))
        Kb_block_ptr = tl.advance(Kb_block_ptr, (BLOCK_N,))

    # tl.store(SMAX_block_ptr, smax)


def attnq_qk(q, k, qa, qb, ka, kb):
    Z = q.shape[0]
    H = q.shape[1]
    M = q.shape[2]
    N = k.shape[2]
    HEAD_DIM = q.shape[3]
    attn_score = torch.empty([Z, H, M, N], dtype=torch.float16, device=q.device)
    qk_grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    kernel_attnq_qk[qk_grid](
        q,
        k,
        attn_score,
        qa,
        qb,
        ka,
        kb,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        attn_score.stride(0),
        attn_score.stride(1),
        attn_score.stride(2),
        attn_score.stride(3),
        qa.stride(0),
        qa.stride(1),
        qa.stride(2),
        ka.stride(0),
        ka.stride(1),
        ka.stride(2),
        Z,
        H,
        M,
        N,
        HEAD_DIM=HEAD_DIM,
    )
    return attn_score


@triton.jit
def softmaxq_kernel(
    output_ptr,
    input_ptr,
    scale,
    causal,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    Z,
    H,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    barrier = n_cols

    for row_idx in range(row_start, n_rows, row_step):
        if causal:
            barrier = row_idx % M + 1

        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < barrier
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row *= scale
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        quantized_output = ((softmax_output - 0.00195312) * 256.0 - 128.0).to(tl.int8)
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, quantized_output, mask=mask)


properties = driver.utils.get_device_properties(0)
attributes = drv.Device(0).get_attributes()
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = attributes[drv.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR]
SIZE_SMEM = attributes[drv.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR]
WARP_SIZE = attributes[drv.device_attribute.WARP_SIZE]
kernels = {}


def softmaxq(x, scale, causal):
    Z, H, M, N = x.shape
    n_rows = Z * H * M
    n_cols = N

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.zeros_like(x, dtype=torch.int8) - 128

    x = x.view(n_rows, n_cols)
    y = y.view(n_rows, n_cols)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmaxq_kernel.warmup(
            y,
            x,
            scale,
            causal,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            Z,
            H,
            M,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.shared  # could be zero
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem) if size_smem else occupancy
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        scale,
        causal,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        Z,
        H,
        M,
        N,
    )
    return y.view(Z, H, M, N)


@triton.autotune(list(filter(pv_conf_filter, pv_configs)), key=["M", "N", "HEAD_DIM"])
@triton.jit
def kernel_attnq_pv(
    P,
    V,
    O,
    Va,
    Vb,
    # SMAX,
    stride_pz,
    stride_ph,
    stride_pm,
    stride_pn,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,  #
    stride_vabz,
    stride_vabh,
    stride_vabg,
    stride_vabhd,
    # stride_smaxz,
    # stride_smaxh,
    # stride_smaxm,
    Z,
    H,
    M,
    N,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    p_offset = off_z.to(tl.int64) * stride_pz + off_h.to(tl.int64) * stride_ph
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    vab_offset = off_z.to(tl.int64) * stride_vabz + off_h.to(tl.int64) * stride_vabh
    # smax_offset = off_z.to(tl.int64) * stride_smaxz + off_h.to(tl.int64) * stride_smaxh

    # block pointers
    P_block_ptr = tl.make_block_ptr(
        base=P + p_offset,
        shape=(M, N),
        strides=(stride_pm, stride_pn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(M, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    groupsz = (N + HEAD_DIM - 1) // HEAD_DIM
    Va_block_ptr = tl.make_block_ptr(
        base=Va + vab_offset,
        shape=(groupsz * HEAD_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HEAD_DIM,),
        order=(0,),
    )
    Vb_block_ptr = tl.make_block_ptr(
        base=Vb + vab_offset,
        shape=(groupsz * HEAD_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HEAD_DIM,),
        order=(0,),
    )
    # SMAX_block_ptr = tl.make_block_ptr(
    #     base=SMAX + smax_offset,
    #     shape=(M,),
    #     strides=(stride_smaxm,),
    #     offsets=(start_m * BLOCK_M,),
    #     block_shape=(BLOCK_M,),
    #     order=(0,),
    # )

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)

    # loop over k and update result
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p = tl.load(P_block_ptr)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p, v, out_dtype=tl.int32)
        pv_fp16 = pv.to(tl.float16)
        psum = tl.sum(p.to(tl.float16), 1)
        vsum = tl.sum(v.to(tl.float16), 0)
        va = tl.load(Va_block_ptr)
        vb = tl.load(Vb_block_ptr)
        # real = (0.00390625 * va)[None, :] * pv_fp16
        # tl.static_print(real.shape)
        real = (
            ((0.00390625 * va).to(tl.float16))[None, :] * pv_fp16
            + ((0.00390625 * psum).to(tl.float16))[:, None] * vb[None, :]
            + ((0.5 * ((va * vsum) + (HEAD_DIM * vb))).to(tl.float16))[None, :]
        )
        acc += real
        P_block_ptr = tl.advance(P_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Va_block_ptr = tl.advance(Va_block_ptr, (HEAD_DIM,))
        Vb_block_ptr = tl.advance(Vb_block_ptr, (HEAD_DIM,))

    tl.store(O_block_ptr, acc, boundary_check=(0,))


def attnq_pv(p, v, va, vb):
    Z = p.shape[0]
    H = p.shape[1]
    M = p.shape[2]
    N = p.shape[3]
    HEAD_DIM = v.shape[3]
    attn_out = torch.empty([Z, H, M, HEAD_DIM], dtype=torch.float16, device=p.device)
    pv_grid = lambda args: (triton.cdiv(M, args["BLOCK_M"]), Z * H, 1)
    kernel_attnq_pv[pv_grid](
        p,
        v,
        attn_out,
        va,
        vb,
        p.stride(0),
        p.stride(1),
        p.stride(2),
        p.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        attn_out.stride(3),
        va.stride(0),
        va.stride(1),
        va.stride(2),
        va.stride(3),
        Z,
        H,
        M,
        N,
        HEAD_DIM=HEAD_DIM,
    )
    return attn_out


# def self_attention(q, k, v, causal=True):
#     HEAD_DIM = q.shape[3]
#     score = attnq_qk(q, k)
#     prob = softmaxq(score, 1 / math.sqrt(HEAD_DIM), causal)
#     attn_out = attnq_pv(prob, v)
#     return attn_out


# def quantize_v(v):
#     window = v.unfold(2, 128, 128)
#     max_values = window.max(dim=4).values
#     min_values = window.min(dim=4).values
#     if v.size(2) % 128 != 0:
#         remaining = v[:, :, -v.size(2) % 128 :, :]
#         max_values_remaining = remaining.max(dim=2).values.unsqueeze(2)
#         min_values_remaining = remaining.min(dim=2).values.unsqueeze(2)
#         max_values = torch.cat((max_values, max_values_remaining), dim=2)
#         min_values = torch.cat((min_values, min_values_remaining), dim=2)
#     return max_values, min_values


# torch.manual_seed(0)
# q = torch.ones(1, 2, 1282, 128, dtype=torch.int8, device="cuda")
# k = torch.ones(1, 2, 1282, 128, dtype=torch.int8, device="cuda")
# v = torch.ones(1, 2, 1282, 128, dtype=torch.int8, device="cuda")
# qa = torch.ones(1, 2, 1282, dtype=torch.float16, device="cuda")
# qb = torch.ones(1, 2, 1282, dtype=torch.float16, device="cuda")
# ka = torch.ones(1, 2, 1282, dtype=torch.float16, device="cuda")
# kb = torch.ones(1, 2, 1282, dtype=torch.float16, device="cuda")

# v = torch.randn([2, 2, 1285, 128], dtype=torch.float16, device="cuda")
# va, vb = quantize_v(v)
# v = torch.ones([2, 2, 1285, 128], dtype=torch.int8, device="cuda")
# p = torch.ones([2, 2, 1285, 1285], dtype=torch.int8, device="cuda")
# # s = attnq_qk(q, k, qa, qb, ka, kb)
# # print(s)
# # print(softmaxq(s, 1 / math.sqrt(128), True))
# print(attnq_pv(p, v, va, vb))
