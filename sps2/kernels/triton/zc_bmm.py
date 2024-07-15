import time

import torch
import triton
import triton.language as tl

# For A100
NUM_SM: tl.constexpr = 108


def _singleton_timer(cls):
    timers = {}

    def get_timer(name):
        if name not in timers:
            timers[name] = cls(name)
        return timers[name]

    return get_timer


@_singleton_timer
class _Timer:
    def __init__(self, name) -> None:
        self.name = name
        self.total_time = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        interval = 1000 * (time.time() - self.start)
        self.total_time += interval
        print(f"{self.name}: {interval}ms out of {self.total_time}ms.")


autotune_config_half = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    # Good config for fp8 inputs.
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
    ),
    triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
    ),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
]


autotune_config_int8 = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    # Good config for fp8 inputs.
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
    ),
    triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
    ),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
]


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=autotune_config_half,
    key=["M", "N", "K"],
)
@triton.jit
def bmm_kernel_half(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    B,
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,  #
    stride_bb,
    stride_bk,
    stride_bn,  #
    stride_cb,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    num_pids_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pids = num_pids_m * num_pids_n
    num_pids_in_group = GROUP_SIZE_M * num_pids_n

    for _ in range(B):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)

        while pid < num_pids:
            group_id = pid // num_pids_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pids_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pids_in_group) % group_size_m)
            pid_n = (pid % num_pids_in_group) // group_size_m

            # ----------------------------------------------------------
            # Create pointers for the first blocks of A and B.
            # We will advance this pointer as we move in the K direction
            # and accumulate
            # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
            # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
            # See above `Pointer Arithmetic` section for details
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            # -----------------------------------------------------------
            # Iterate to compute a block of the C matrix.
            # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
            # of fp32 values for higher accuracy.
            # `accumulator` will be converted back to fp16 after the loop.
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                # We accumulate along the K dimension.
                accumulator = tl.dot(a, b, accumulator)
                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            c = accumulator.to(tl.float16)

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

            pid += NUM_SM

        a_ptr += stride_ab
        b_ptr += stride_bb
        c_ptr += stride_cb


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=autotune_config_int8,
    key=["M", "N", "K"],
)
@triton.jit
def bmm_kernel_int8(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    B,
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,  #
    stride_bb,
    stride_bk,
    stride_bn,  #
    stride_cb,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    num_pids_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pids = num_pids_m * num_pids_n
    num_pids_in_group = GROUP_SIZE_M * num_pids_n

    for _ in range(B):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)

        while pid < num_pids:
            group_id = pid // num_pids_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pids_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pids_in_group) % group_size_m)
            pid_n = (pid % num_pids_in_group) // group_size_m

            # ----------------------------------------------------------
            # Create pointers for the first blocks of A and B.
            # We will advance this pointer as we move in the K direction
            # and accumulate
            # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
            # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
            # See above `Pointer Arithmetic` section for details
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            # -----------------------------------------------------------
            # Iterate to compute a block of the C matrix.
            # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
            # of fp32 values for higher accuracy.
            # `accumulator` will be converted back to fp16 after the loop.
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                # We accumulate along the K dimension.
                accumulator = tl.dot(a, b, accumulator)
                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            c = accumulator.to(tl.float16)

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)

            pid += NUM_SM

        a_ptr += stride_ab
        b_ptr += stride_bb
        c_ptr += stride_cb


def bmm_half(a, b):
    # Check constraints.
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    B, M, K = a.shape
    B, K, N = b.shape
    # Allocates output.
    c = torch.empty((B, M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    bmm_kernel_half[(NUM_SM,)](
        a,
        b,
        c,  #
        B,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),
        a.stride(2),  #
        b.stride(0),
        b.stride(1),
        b.stride(2),  #
        c.stride(0),
        c.stride(1),
        c.stride(2),  #
    )
    return c


def bmm_int8(a, b):
    # Check constraints.
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    B, M, K = a.shape
    B, K, N = b.shape
    # Allocates output.
    c = torch.empty((B, M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    bmm_kernel_int8[(NUM_SM,)](
        a,
        b,
        c,  #
        B,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),
        a.stride(2),  #
        b.stride(0),
        b.stride(1),
        b.stride(2),  #
        c.stride(0),
        c.stride(1),
        c.stride(2),  #
    )
    return c


# torch.manual_seed(0)
a = torch.randint(0, 4, (2, 1024, 1024), device="cuda", dtype=torch.int8)
b = torch.randint(0, 4, (2, 1024, 1024), device="cuda", dtype=torch.int8)

triton_output = torch.bmm(a, b)
for _ in range(10):
    with _Timer("test"):
        triton_output = torch.bmm(a, b)
