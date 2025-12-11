import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple


"""
simplified explanation of the scaled int8 matmul algorithm
adopted from deepseek scaled FP8 matmul and jetfire paper
https://arxiv.org/abs/2403.12422
https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

                                                     N dimension →  
                                               INT8 weights                 scaler per block
                                               ┌-----┬-----┬─────┬─────┐    ┌-----┬-----┬─────┬─────┐
                                               : b00 : b01 : b02 | b03 |    :     :     :     |     |
                                               ├-----┼-----┼─────┼─────┤    :b_s00:b_s10:b_s20|b_s30|
                                           K   : b10 : b11 : b12 | b13 |    :     :     :     |     |
                                          dim  ├-----┼-----┼─────┼─────┤    ├-----┼-----┼─────┼─────┤
                                           ↓   | b20 | b21 | b22 | b23 |    |     |     |     |     |
                                               ├─────┼─────┼─────┼─────┤    |b_s01|b_s11|b_s21|b_s31|
                                               | b30 | b31 | b32 | b33 |    |     |     |     |     |
                                               └─────┴─────┴─────┴─────┘    └─────┴─────┴─────┴─────┘
                                               ┌-----┬-----┐
                                               : b00 : b01 :
     ├─── blk ───┤                             ├-----┼-----┤
                                               : b10 : b11 :
            K dimension →                      └-----┴-----┘                                
     INT8 activations
     ┌-----┬-----┬─────┬─────┐   ┌-----┬-----┐ ┌-----┬-----┐   ┌-----------┐   ┌-----┬-----┐   ┌-----┬-----┐
     : a00 : a01 : a02 | a03 |   : a00 : a01 : :  @  :  @  :   :   a_s00   :   :     :     :   :acc00:acc01:
     ├-----┼-----┼─────┼─────┤   ├-----┼-----┤ ├-----┼-----┤ * ├-----------┤ * :b_s00:b_s10: = ├-----┼-----┤ 
 M   : a10 : a11 : a12 | a13 |   : a10 : a11 : :  @  :  @  :   :   a_s10   :   :     :     :   :acc10:acc11:
dim  ├-----┼-----┼─────┼─────┤   └-----┴-----┘ └-----┴-----┘   └-----------┘   └-----┴-----┘   └-----┴-----┘
 ↓   | a20 | a21 | a22 | a23 |   INT8 matmul acc in INT32      rescale the FP32 intermediate   accumulate
     ├─────┼─────┼─────┼─────┤   then cast to FP32             "rank 1" hadamard scaler        intermediate
     | a30 | a31 | a32 | a33 |  
     └─────┴─────┴─────┴─────┘  
     scaler per block
     ┌-----------┬───────────┐
     :   a_s00   :   a_s01   |
     ├-----------┼───────────┤
     :   a_s10   :   a_s11   |
     ├-----------┼───────────┤
     |   a_s20   |   a_s21   |
     ├───────────┼───────────┤
     |   a_s30   |   a_s31   |
     └───────────┴───────────┘
"""


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))  # reduction
    # amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    s = amax / 127.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def act_dequant_kernel(x_ptr, s_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes the input tensor using the provided scaling factors.

    Args:
        x_ptr: Pointer to the quantized input tensor.
        s_ptr: Pointer to the scaling factors.
        y_ptr: Pointer to the output tensor where dequantized values will be stored.
        BLOCK_SIZE: The size of the block processed by each program instance.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.load(s_ptr + pid)
    y = x * s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)


def act_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Dequantizes the activation tensor ``x`` using the provided scale tensor.

    Args:
        x (torch.Tensor): Quantized activation tensor. Must be contiguous and divisible by block_size on the last dim.
        s (torch.Tensor): Scale tensor shaped like ``(*batch_dims, last_dim // block_size)``.
        block_size (int): Block size used for quantization. Defaults to 128.
        output_dtype (torch.dtype, optional): Target dtype for output. Defaults to torch.get_default_dtype().

    Returns:
        torch.Tensor: Dequantized tensor matching ``x`` shape.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    if output_dtype is None:
        output_dtype = torch.get_default_dtype()

    y = torch.empty_like(x, dtype=output_dtype)
    num_programs = s.numel()  # one program per block
    grid = lambda meta: (num_programs,)
    act_dequant_kernel[grid](x, s, y, BLOCK_SIZE=block_size)
    return y


@triton.jit
def weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes weights using block-wise scales.

    Args:
        x_ptr: Pointer to input weights.
        y_ptr: Pointer to output quantized weights.
        s_ptr: Pointer to output scales.
        M (int): Rows.
        N (int): Cols.
        BLOCK_SIZE: Tiling block size.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x))
    s = amax / 127.0

    y = x / s
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def weight_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a weight matrix block-wise.

    Args:
        x (torch.Tensor): Weight tensor of shape (M, N).
        block_size (int): Block size for quantization. Defaults to 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (int8 weights, float scales) where scales shape is (M//block_size, N//block_size).
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must be 2D"
    M, N = x.size()
    assert M % block_size == 0 and N % block_size == 0, (
        f"Dimensions must be divisible by block_size={block_size}, got shape {x.shape}"
    )

    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(M // block_size, N // block_size, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()

    if output_dtype is None:
        output_dtype = torch.get_default_dtype()

    y = torch.empty_like(x, dtype=output_dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# matmul intermediate block size is hardcoded to 128
int8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.jit
def int8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on INT8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + offs_n * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        # Cast to float32 before multiplying with scaling factors.
        dot_prod = tl.dot(a, b)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def int8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using INT8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


# ======================================================================
# Extended INT8 GEMM family (block_size >= 128) for AB testing and fusion
# ======================================================================

# Larger tiles to match activation block quant (out_block_size=128)
int8_gemm_configs_ext = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [128, 256]
    for block_n in [128, 256]
    for num_stages in [3, 4, 5]
]


@triton.jit
def int8_gemm_kernel_v2(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    INT8 matmul with corrected 2D weight-scale indexing and int32 accumulation.

    This variant mirrors the version in int8_kernels while leaving the legacy
    int8_gemm untouched for comparison/AB testing.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def int8_gemm_addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused INT8 matmul + bias using block>=128 tiles."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def int8_gemm_v2(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
):
    """
    Alternative INT8 GEMM using block>=128 tiles and corrected weight-scale indexing.
    Leaves the legacy int8_gemm intact for AB testing.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"

    c = a.new_empty(*a.size()[:-1], N, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_kernel_v2[grid](
        a, b, c, a_s, b_s, M, N, K,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )
    return c


def int8_addmm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    bias: torch.Tensor = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
):
    """
    Fused INT8 matmul + bias. Uses v2 kernel, leaves legacy matmul untouched.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"

    c = a.new_empty(*a.size()[:-1], N, dtype=torch.float16)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous(), "Bias tensor must be contiguous"
        assert bias.dim() == 1 and bias.size(0) == N, (
            f"Bias must be 1D with length {N}, got shape {bias.shape}"
        )
        bias_ptr = bias
    else:
        bias_ptr = c  # dummy

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_addmm_kernel[grid](
        a, b, c, bias_ptr, a_s, b_s, M, N, K,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )
    return c


# ======================================================================
# Fused INT8 GEMM + Quantization (per-row, block-wise along N)
# ======================================================================


@triton.heuristics({
    'NUM_BLOCKS': lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_quant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_s_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)
    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = tl.reshape(quantized.to(c_ptr.dtype.element_ty), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)

    n_scale_stride = N // out_block_size
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


@triton.heuristics({
    'NUM_BLOCKS': lambda args: args["BLOCK_SIZE_N"] // args["out_block_size"],
})
@triton.jit
def int8_gemm_addmm_quant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_s_ptr,
    bias_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_block_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k

    k_blocks = k
    b_s_base = b_s_ptr + pid_n * k_blocks

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_blocks):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_base + i)
        dot_prod = tl.dot(a, b, out_dtype=tl.int32)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=offs_n[None, :] < N, other=0.0)
        accumulator += bias

    accumulator_reshaped = tl.reshape(accumulator, (BLOCK_SIZE_M, NUM_BLOCKS, out_block_size))
    block_max = tl.max(tl.abs(accumulator_reshaped), axis=2)
    block_scale = tl.maximum(block_max / 127.0, 1e-8)
    block_scale_broadcast = tl.reshape(block_scale, (BLOCK_SIZE_M, NUM_BLOCKS, 1))
    quantized = accumulator_reshaped / block_scale_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = tl.reshape(quantized.to(c_ptr.dtype.element_ty), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m_actual = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m_actual[:, None] < M) & (offs_n_actual[None, :] < N)
    c_ptrs = c_ptr + offs_m_actual[:, None] * N + offs_n_actual[None, :]
    tl.store(c_ptrs, quantized_int8, mask=mask)

    n_scale_stride = N // out_block_size
    offs_m_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_scale = pid_n * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    scale_ptrs = c_s_ptr + offs_m_scale[:, None] * n_scale_stride + offs_n_scale[None, :]
    scale_mask = (offs_m_scale[:, None] < M) & (offs_n_scale[None, :] < n_scale_stride)
    tl.store(scale_ptrs, block_scale, mask=scale_mask)


def int8_gemm_quant(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    out_block_size: int = 128,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused INT8 GEMM that outputs quantized activations + scales."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    batch_shape = a.size()[:-1]
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    assert N % out_block_size == 0, f"N={N} must be divisible by out_block_size={out_block_size}"

    c = a.new_empty(*batch_shape, N, dtype=torch.int8)
    n_blocks = N // out_block_size
    c_s = a.new_empty(M, n_blocks, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_quant_kernel[grid](
        a, b, c, c_s, a_s, b_s, M, N, K, out_block_size,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )

    if len(batch_shape) > 0:
        c_s = c_s.reshape(*batch_shape, n_blocks)

    return c, c_s


def int8_addmm_quant(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    bias: torch.Tensor = None,
    out_block_size: int = 128,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused INT8 addmm (matmul + bias) with output quantization."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling tensors must be contiguous"
    assert b.dim() == 2, f"Expected b to be 2D, got shape {b.shape}"

    K = a.size(-1)
    M = a.numel() // K
    N = b.shape[0]
    batch_shape = a.size()[:-1]
    assert b.size(1) == K, f"Shape mismatch: b.shape={b.shape}, expected [..., {K}]"
    assert N % out_block_size == 0, f"N={N} must be divisible by out_block_size={out_block_size}"

    c = a.new_empty(*batch_shape, N, dtype=torch.int8)
    n_blocks = N // out_block_size
    c_s = a.new_empty(M, n_blocks, dtype=torch.float32)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_contiguous(), "Bias tensor must be contiguous"
        assert bias.dim() == 1 and bias.size(0) == N, (
            f"Bias must be 1D with length {N}, got shape {bias.shape}"
        )
        bias_ptr = bias
    else:
        bias_ptr = c  # dummy

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    int8_gemm_addmm_quant_kernel[grid](
        a, b, c, c_s, bias_ptr, a_s, b_s, M, N, K, out_block_size,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )

    if len(batch_shape) > 0:
        c_s = c_s.reshape(*batch_shape, n_blocks)

    return c, c_s


# ======================================================================
# INT8 GELU (fused dequant + activation + requant)
# ======================================================================

# Note: BLOCK_N must be >= quantization block_size (typically 128) and divisible by it
int8_gelu_configs = [
    Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [64, 128, 256]
    for block_n in [128, 256]
    for num_stages in [2, 3, 4]
    for num_warps in [4, 8]
]


@triton.heuristics({
    'BLOCK_SM': lambda args: args["BLOCK_M"],
    'BLOCK_SN': lambda args: args["BLOCK_N"] // args["BLOCK_SIZE"],
})
@triton.jit
def int8_gelu_kernel(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    input_scale_ptr,
    M,
    N: tl.constexpr,
    SM,
    SN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_m = pid // NUM_BLOCK_N
    pid_n = pid % NUM_BLOCK_N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    input_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    input_data = tl.load(input_ptrs, mask=mask, other=0).to(tl.int32)

    offs_sm = pid_m * BLOCK_SM + tl.arange(0, BLOCK_SM)
    offs_sn = pid_n * BLOCK_SN + tl.arange(0, BLOCK_SN)
    scale_ptrs = input_scale_ptr + offs_sm[:, None] * SN + offs_sn[None, :]
    scale_mask = (offs_sm[:, None] < SM) & (offs_sn[None, :] < SN)
    input_scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

    input_data = tl.reshape(input_data, (BLOCK_M, BLOCK_SN, BLOCK_SIZE))
    input_scales = tl.reshape(input_scales, (BLOCK_M, BLOCK_SN, 1))

    input_fp32 = input_data.to(tl.float32) * input_scales

    sqrt_2 = 1.41421356237
    erf_input = input_fp32 / sqrt_2
    erf_val = tl.math.erf(erf_input)
    gelu_output = input_fp32 * 0.5 * (1.0 + erf_val)

    abs_output = tl.abs(gelu_output)
    max_val = tl.max(abs_output, axis=2)
    output_scales = tl.maximum(max_val / 127.0, 1e-8)

    output_scales_broadcast = tl.reshape(output_scales, (BLOCK_M, BLOCK_SN, 1))

    quantized = gelu_output / output_scales_broadcast
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    quantized_int8 = tl.reshape(quantized.to(tl.int8), (BLOCK_M, BLOCK_N))

    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, quantized_int8, mask=mask)

    output_scale_ptrs = output_scale_ptr + offs_sm[:, None] * SN + offs_sn[None, :]
    tl.store(output_scale_ptrs, output_scales, mask=scale_mask)


def int8_gelu(
    x: torch.Tensor,
    s_x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused INT8 GELU with block-wise quantization.
    Leaves caller free to use legacy paths for AB testing.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert s_x.is_contiguous(), "Scale tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension must be divisible by block_size={block_size}"
    )
    assert block_size == 128, "Only block_size=128 supported in current configs"

    original_shape = x.shape
    batch_shape = original_shape[:-1]
    N = original_shape[-1]

    if x.dim() > 2:
        x = x.reshape(-1, N)
        s_x = s_x.reshape(-1, s_x.size(-1))

    M = x.size(0)
    SM = M
    SN = N // block_size

    y = torch.empty_like(x, dtype=torch.int8)
    s_y = torch.empty_like(s_x, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    int8_gelu_kernel[grid](
        y, s_y, x, s_x,
        M, N, SM, SN,
        BLOCK_SIZE=block_size, BLOCK_M=128, BLOCK_N=128, BLOCK_SM=128,
    )

    if len(batch_shape) > 0:
        y = y.reshape(*batch_shape, N)
        s_y = s_y.reshape(*batch_shape, SN)

    return y, s_y