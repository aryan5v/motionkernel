"""Fused BF16 preprocessing and shared-activation projection kernels.

Repaired from the r4 candidate (fingerprint 2c92e356aa34bc0d3c49522bd1365c1b),
which was bit-exact on the corpus case, deterministic, and 2.557x faster, but
failed full isolated correctness on every numerical-stability case with

    AssertionError: expected size 4680==4680, stride 4096==16384 at dim=1

Two defects produced that, and they compound:

1. The kernel baked its input layout into ``tl.constexpr`` parameters --
   ``SOURCE_ROW_STRIDE=16384``, ``CONTEXT_ROW_STRIDE=8192``, plus the row and
   width extents. Those values are correct for this workload, where input_2 is
   a 4096-wide slice of a 1x4680x16384 timestep tensor and input_7 a 2048-wide
   slice of a 1x101x8192 one, but they are properties of *one* call, not of the
   operation. Handed a contiguous tensor of the same shape (row stride 4096),
   the Triton kernel would read at the wrong addresses.

2. ``kernel_fn`` called ``torch.compile`` once with a backend that captured
   Inductor's raw compiled entry into ``_COMPILED_ENTRY``, then invoked that
   entry directly on every later call -- explicitly, per its own comment, to
   avoid "a guard/cache lookup before every GPU schedule". That bypasses the
   entire Dynamo guard system: no shape check, no stride check, no dtype
   check. The stability stage's contiguous inputs were caught only because
   Inductor happens to emit an internal stride assertion. Nothing else would
   have caught them, and the Triton kernels have no such assertion at all.

Defect 2 also made the recorded 2.557x an unfair comparison: the candidate
skipped guard evaluation that the compiled baseline paid.

The repair reads shapes and strides from the tensors at run time and dispatches
through ``torch.compile`` with its guards intact. Nothing about the fusion
changes, and the arithmetic is untouched -- no approximate intrinsics are
introduced, so the kernel remains capable of bitwise parity.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _round_to_bf16(value):
    """Materialize a BF16 rounding boundary and widen back to FP32."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 rounded;
            cvt.rn.bf16.f32 rounded, $1;
            cvt.f32.bf16 $0, rounded;
        }
        """,
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _main_preprocess(
    source_ptr,
    scale_ptr,
    residual_ptr,
    shift_source_ptr,
    shift_bias_ptr,
    output_ptr,
    shifted_ptr,
    context_source_ptr,
    context_scale_ptr,
    context_residual_ptr,
    context_output_ptr,
    N,
    WIDTH,
    SOURCE_ROW_STRIDE,
    CONTEXT_N,
    CONTEXT_WIDTH,
    CONTEXT_ROW_STRIDE,
    BLOCK: tl.constexpr,
):
    # N, WIDTH and the two row strides are runtime arguments. They describe the
    # caller's tensors, not this kernel, and specializing on them is what made
    # the r4 version silently wrong for any other layout. BLOCK stays constexpr
    # because it shapes the generated code.
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    rows = offsets // WIDTH
    columns = offsets % WIDTH
    strided_offsets = rows * SOURCE_ROW_STRIDE + columns

    source = tl.load(source_ptr + strided_offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + strided_offsets, mask=mask)
    value = _round_to_bf16(source.to(tl.float32) + 1.0)
    value = _round_to_bf16(scale.to(tl.float32) * value)
    value += residual.to(tl.float32)
    tl.store(output_ptr + offsets, value, mask=mask)

    shift_source = tl.load(shift_source_ptr + offsets, mask=mask)
    shift_bias = tl.load(shift_bias_ptr + columns, mask=mask)
    tl.store(shifted_ptr + offsets, shift_source + shift_bias, mask=mask)

    if program < tl.cdiv(CONTEXT_N, BLOCK):
        context_mask = offsets < CONTEXT_N
        context_rows = offsets // CONTEXT_WIDTH
        context_columns = offsets % CONTEXT_WIDTH
        context_strided_offsets = (
            context_rows * CONTEXT_ROW_STRIDE + context_columns
        )
        context_source = tl.load(
            context_source_ptr + context_strided_offsets, mask=context_mask
        )
        context_scale = tl.load(
            context_scale_ptr + offsets, mask=context_mask
        )
        context_residual = tl.load(
            context_residual_ptr + context_strided_offsets, mask=context_mask
        )
        context_value = _round_to_bf16(
            context_source.to(tl.float32) + 1.0
        )
        context_value = _round_to_bf16(
            context_scale.to(tl.float32) * context_value
        )
        context_value += context_residual.to(tl.float32)
        tl.store(context_output_ptr + offsets, context_value, mask=context_mask)


@triton.jit
def _dual_context_linear(
    input_ptr,
    weight_a_ptr,
    bias_a_ptr,
    weight_b_ptr,
    bias_b_ptr,
    output_a_ptr,
    output_b_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator_b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_offsets
        k_mask = k < K
        activations = tl.load(
            input_ptr + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        weight_a = tl.load(
            weight_a_ptr + columns[:, None] * K + k[None, :],
            mask=(columns[:, None] < N) & k_mask[None, :],
            other=0.0,
        )
        weight_b = tl.load(
            weight_b_ptr + columns[:, None] * K + k[None, :],
            mask=(columns[:, None] < N) & k_mask[None, :],
            other=0.0,
        )
        accumulator_a += tl.dot(activations, tl.trans(weight_a))
        accumulator_b += tl.dot(activations, tl.trans(weight_b))

    accumulator_a += tl.load(bias_a_ptr + columns, mask=columns < N, other=0.0)[None, :]
    accumulator_b += tl.load(bias_b_ptr + columns, mask=columns < N, other=0.0)[None, :]
    output_offsets = rows[:, None] * N + columns[None, :]
    output_mask = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(output_a_ptr + output_offsets, accumulator_a, mask=output_mask)
    tl.store(output_b_ptr + output_offsets, accumulator_b, mask=output_mask)


def _row_stride(tensor, width):
    """Elements between consecutive rows of a 2D view of ``tensor``.

    The r4 kernel indexed ``rows * SOURCE_ROW_STRIDE + columns`` with a literal
    16384, which is this workload's value because input_2 is a 4096-wide slice
    of a 16384-wide timestep tensor. Reading it from the tensor keeps the same
    fast path for that case and stays correct for a contiguous one, where the
    stride is simply the width.
    """
    flattened = tensor.reshape(-1, width) if tensor.is_contiguous() else tensor
    if flattened.dim() >= 2:
        return flattened.stride(-2)
    return width


def _kernel_impl(
    input_0,
    input_1,
    input_2,
    input_3,
    input_4,
    input_5,
    input_6,
    input_7,
    input_8,
    input_9,
    input_10,
    input_11,
    input_12,
    input_13,
    input_14,
    input_15,
):
    # Extents come from the tensors. The r4 version hardcoded 4680x4096 and
    # 101x2048 as module constants, so any other size read out of bounds.
    main_width = input_0.shape[-1]
    main_rows = input_0.numel() // main_width
    main_elements = main_rows * main_width
    context_width = input_8.shape[-1]
    context_rows = input_8.numel() // context_width
    context_elements = context_rows * context_width

    source_row_stride = _row_stride(input_2, main_width)
    context_row_stride = _row_stride(input_7, context_width)

    # The kernel stores to these with contiguous indexing, so they must be
    # contiguous regardless of how the inputs are laid out. empty_like
    # preserves the source layout, which silently breaks that assumption for a
    # non-contiguous input.
    residual = torch.empty(
        input_0.shape, dtype=input_0.dtype, device=input_0.device
    )
    shifted = torch.empty(
        input_0.shape, dtype=input_0.dtype, device=input_0.device
    )
    context = torch.empty(
        input_8.shape, dtype=input_8.dtype, device=input_8.device
    )
    context_a = torch.empty_like(context)
    context_b = torch.empty_like(context)

    block = 1024
    _main_preprocess[(triton.cdiv(main_elements, block),)](
        input_2,
        input_6,
        input_3,
        input_0,
        input_1,
        residual,
        shifted,
        input_7,
        input_8,
        input_9,
        context,
        main_elements,
        main_width,
        source_row_stride,
        context_elements,
        context_width,
        context_row_stride,
        BLOCK=block,
        num_warps=4,
    )

    projected = F.linear(residual, input_10, input_11)
    _dual_context_linear[
        (triton.cdiv(context_rows, 64), triton.cdiv(context_width, 32))
    ](
        context,
        input_12,
        input_13,
        input_14,
        input_15,
        context_a,
        context_b,
        context_rows,
        context_width,
        context_width,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=128,
        num_warps=8,
        num_stages=4,
    )
    return projected, context_a, context_b, shifted, input_4, input_5


# Dispatch through torch.compile with its guards intact.
#
# The r4 version captured Inductor's compiled entry through a custom backend
# and then called it directly, bypassing Dynamo entirely on every call after
# the first. That is why a stride change reached the kernel at all, and why its
# measured latency excluded guard evaluation the compiled baseline paid for.
# Keeping the standard dispatch means a layout change recompiles or falls back
# instead of silently reading the wrong addresses.
_COMPILE_DISPATCH = torch.compile(_kernel_impl, fullgraph=True, dynamic=False)


def kernel_fn(
    input_0,
    input_1,
    input_2,
    input_3,
    input_4,
    input_5,
    input_6,
    input_7,
    input_8,
    input_9,
    input_10,
    input_11,
    input_12,
    input_13,
    input_14,
    input_15,
):
    return _COMPILE_DISPATCH(
        input_0,
        input_1,
        input_2,
        input_3,
        input_4,
        input_5,
        input_6,
        input_7,
        input_8,
        input_9,
        input_10,
        input_11,
        input_12,
        input_13,
        input_14,
        input_15,
    )
