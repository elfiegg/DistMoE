# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backward of input-scaled RMSNorm fused with router-score top-k reduction.

The forward is the CuTe fused combine with an input-scale weight
(``dist_moe.kernels.norm.fused_rmsnorm_combine``). For every active route row
``x`` of token ``t`` with router score ``s``, it computes::

    g = round(weight + gain_center)          # FP32 add, one round to x.dtype
    u = round(g * x)                         # FP32 multiply, one round
    n = u * rsqrt(mean(u * u) + eps)         # FP32
    y[t] += s * n                            # FP32 accumulation

This kernel differentiates that forward. Each program owns a contiguous run of
route rows, recomputes the reciprocal RMS from the row it already loads, and
writes the route-input and score gradients. It accumulates the weight
gradient for its rows in registers and stores one FP32 partial row, so the
weight reduction needs ``O(programs * D)`` workspace instead of one partial row
per route. The two rounding steps are treated as identity in backward.
"""

import torch
import triton
import triton.language as tl

from ..._activation_buffer_planner_kernel import MEMORY_ALIGNMENT_TL_CONSTEXPR
from .._environment import num_sms_per_device
from .broadcast_n_reduction import _validate_activation_buffer_args

# Two resident programs per SM hide the per-row reduction latency while keeping
# the weight-gradient partials to a few megabytes.
_PROGRAMS_PER_SM = 2


def _num_warps(block_d: int) -> int:
    """Return the warp count for one register-resident row of ``block_d``."""
    if block_d >= 8192:
        return 16
    if block_d >= 2048:
        return 8
    return 4


def input_scaled_rmsnorm_combine_bwd(
    grad_y_TD: torch.Tensor,
    x_MD: torch.Tensor,
    weight_D: torch.Tensor,
    scores_TK: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    *,
    eps: float,
    gain_center: float,
    grad_x_MD: torch.Tensor | None = None,
    x_buffer: torch.Tensor | None = None,
    x_buffer_offset: torch.Tensor | None = None,
    x_buffer_condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate input-scaled RMSNorm plus score-weighted top-k reduction.

    Args:
        grad_y_TD: Contiguous BF16 or FP32 gradient of the reduced output with
            shape ``[T, D]``.
        x_MD: Contiguous BF16 route-wise pre-norm input with shape
            ``[T * K, D]``. When ``x_buffer`` is given, this tensor is read
            only where ``x_buffer_condition`` is nonzero.
        weight_D: Contiguous input-scale weight with shape ``[D]`` and the
            dtype of ``x_MD``.
        scores_TK: Contiguous FP32 router scores with shape ``[T, K]``.
        expert_ids_TK: Contiguous int32 or int64 expert IDs with shape
            ``[T, K]``. Negative routes receive zero gradients and do not
            contribute to the weight gradient.
        eps: Positive epsilon added to the mean square.
        gain_center: Constant added to ``weight_D`` in forward.
        grad_x_MD: Optional contiguous destination for the route gradient. It
            must be disjoint from ``x_MD`` or alias it exactly.
        x_buffer: Optional uint8 activation buffer holding the saved input.
        x_buffer_offset: Byte offset of the saved input in ``x_buffer``.
        x_buffer_condition: Device predicate selecting ``x_MD`` when nonzero
            and the activation-buffer copy otherwise.

    Returns:
        Route-input gradient ``[T * K, D]`` in ``x_MD.dtype``, router-score
        gradient ``[T, K]`` in FP32, and weight gradient ``[D]`` in
        ``weight_D.dtype``.

    Raises:
        ValueError: If a shape, dtype, device, layout, or alias is unsupported.
    """
    num_tokens, dim = grad_y_TD.shape
    if scores_TK.ndim != 2 or scores_TK.shape[0] != num_tokens:
        raise ValueError("scores_TK must have shape [T, K]")
    topk = scores_TK.shape[1]
    num_rows = num_tokens * topk
    device = x_MD.device
    if (
        x_MD.shape != (num_rows, dim)
        or weight_D.shape != (dim,)
        or expert_ids_TK.shape != (num_tokens, topk)
    ):
        raise ValueError(
            "expected x [T * K, D], weight [D], and expert_ids [T, K]; got "
            f"x {tuple(x_MD.shape)}, weight {tuple(weight_D.shape)}, "
            f"expert_ids {tuple(expert_ids_TK.shape)} for T={num_tokens}, "
            f"K={topk}, D={dim}"
        )
    if (
        x_MD.dtype != torch.bfloat16
        or weight_D.dtype != x_MD.dtype
        or grad_y_TD.dtype not in (torch.bfloat16, torch.float32)
        or scores_TK.dtype != torch.float32
        or expert_ids_TK.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "expected BF16 x and weight, BF16/FP32 grad_y, FP32 scores, and "
            "int32/int64 expert IDs"
        )
    tensors = (grad_y_TD, x_MD, weight_D, scores_TK, expert_ids_TK)
    if any(t.device != device or not t.is_contiguous() for t in tensors):
        raise ValueError("inputs must be contiguous tensors on one CUDA device")
    read_x_from_buffer = _validate_activation_buffer_args(
        activation_buffer=x_buffer,
        activation_offset=x_buffer_offset,
        condition=x_buffer_condition,
        device=device,
    )

    x_may_alias_grad_x = False
    if grad_x_MD is None:
        grad_x_MD = torch.empty_like(x_MD)
    else:
        if (
            grad_x_MD.shape != x_MD.shape
            or grad_x_MD.dtype != x_MD.dtype
            or grad_x_MD.device != device
            or not grad_x_MD.is_contiguous()
        ):
            raise ValueError("grad_x_MD must be a contiguous tensor matching x_MD")
        nbytes = x_MD.numel() * x_MD.element_size()
        x_start, grad_start = x_MD.data_ptr(), grad_x_MD.data_ptr()
        x_may_alias_grad_x = x_start == grad_start
        if not x_may_alias_grad_x and (
            x_start < grad_start + nbytes and grad_start < x_start + nbytes
        ):
            raise ValueError("grad_x_MD must be disjoint from x_MD or alias it exactly")

    grad_scores_TK = torch.empty_like(scores_TK)
    num_programs = max(1, min(num_sms_per_device(device) * _PROGRAMS_PER_SM, num_rows))
    rows_per_program = triton.cdiv(num_rows, num_programs)
    grad_weight_partials_PD = torch.empty(
        (num_programs, dim), dtype=torch.float32, device=device
    )
    block_d = triton.next_power_of_2(dim)
    _input_scaled_rmsnorm_combine_bwd_kernel[(num_programs,)](
        grad_y_TD,
        x_MD,
        weight_D,
        scores_TK,
        expert_ids_TK,
        grad_x_MD,
        grad_scores_TK,
        grad_weight_partials_PD,
        x_buffer,
        x_buffer_offset,
        x_buffer_condition,
        num_rows,
        rows_per_program,
        eps,
        gain_center,
        K=topk,
        D=dim,
        BLOCK_D=block_d,
        READ_X_FROM_BUFFER=read_x_from_buffer,
        X_GRAD_X_MAY_ALIAS=x_may_alias_grad_x,
        num_warps=_num_warps(block_d),
    )
    # A fixed program count and row assignment keep this reduction
    # deterministic for a given device.
    grad_weight_D = grad_weight_partials_PD.sum(dim=0).to(weight_D.dtype)
    return grad_x_MD, grad_scores_TK, grad_weight_D


@triton.jit(do_not_specialize=["num_rows", "rows_per_program"])
def _input_scaled_rmsnorm_combine_bwd_kernel(
    grad_y_ptr,
    x_ptr,
    weight_ptr,
    scores_ptr,
    expert_ids_ptr,
    grad_x_ptr,
    grad_scores_ptr,
    grad_weight_partials_ptr,
    x_buffer_ptr,
    x_buffer_offset_ptr,
    x_buffer_condition_ptr,
    num_rows,
    rows_per_program,
    eps,
    gain_center,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    READ_X_FROM_BUFFER: tl.constexpr,
    X_GRAD_X_MAY_ALIAS: tl.constexpr,
) -> None:
    """Differentiate a contiguous run of route rows and one weight partial."""
    x_dtype = x_ptr.dtype.element_ty
    if READ_X_FROM_BUFFER:
        use_tensor_x = tl.load(x_buffer_condition_ptr, eviction_policy="evict_last").to(
            tl.int1
        )
        x_buffer_byte_offset = tl.load(
            x_buffer_offset_ptr, eviction_policy="evict_last"
        ).to(tl.int64)
        x_align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // (
            x_dtype.primitive_bitwidth // 8
        )
        x_buffer_base = tl.multiple_of(
            (x_buffer_ptr + x_buffer_byte_offset).to(tl.pointer_type(x_dtype)),
            x_align,
        )
        x_ptr = tl.where(use_tensor_x, x_ptr, x_buffer_base)

    pid = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    # Match the forward's scaled weight: FP32 add, then one round to x.dtype.
    gain = (
        (
            tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            + gain_center
        )
        .to(x_dtype)
        .to(tl.float32)
    )
    grad_weight_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, num_rows)
    for row in tl.range(row_start, row_end):
        token = row // K
        active = tl.load(expert_ids_ptr + row) >= 0
        mask = col_mask & active
        x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + token * D + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        score = tl.load(scores_ptr + row, mask=active, other=0.0).to(tl.float32)

        # Inactive rows load zeros, so every gradient below is exactly zero.
        scaled = (x * gain).to(x_dtype).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(scaled * scaled, axis=0) / D + eps)
        normalized = scaled * rstd
        grad_score = tl.sum(normalized * grad_y, axis=0)
        grad_normalized = grad_y * score
        projection = tl.sum(grad_normalized * normalized, axis=0) / D
        grad_scaled = (grad_normalized - normalized * projection) * rstd
        grad_weight_acc += grad_scaled * x

        if X_GRAD_X_MAY_ALIAS:
            # grad_x may be the recomputed x itself; finish reading the row
            # before any thread overwrites it.
            tl.debug_barrier()
        tl.store(grad_x_ptr + row * D + cols, grad_scaled * gain, mask=col_mask)
        tl.store(grad_scores_ptr + row, grad_score)

    tl.store(grad_weight_partials_ptr + pid * D + cols, grad_weight_acc, mask=col_mask)
