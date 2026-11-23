# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training postprocess for learned input-scaled expert RMSNorm."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

_NUM_TOKENS = 29_184
_HIDDEN_DIM = 4_096
_TOP_K = 8
_ROWS_PER_CTA = 4
_PARTIALS_PER_TILE = 256
_TILE_WIDTH = 32
_ROUTES = _NUM_TOKENS * _TOP_K
_PARTIALS = _ROUTES // _ROWS_PER_CTA
_REDUCTION_TILES = math.ceil(_PARTIALS / _PARTIALS_PER_TILE)
# A grad-H2 row contains D BF16 values. The reduction needs one FP32 row
# for each partial and final reduction tile, so each row needs two BF16 rows.
_MIN_RECV_ROWS = 2 * (_PARTIALS + _REDUCTION_TILES)


@dataclass(frozen=True)
class InputScaledRMSNormPostprocess:
    """Opt in to fused, learned input-scale RMSNorm during BF16 training.

    The operator implements rmsnorm(x * (weight + gain_center)) for
    every route before weighting and reducing over the selected experts.
    It supports the fixed-shape path with 29,184 tokens, 4,096 hidden and
    intermediate dimensions, top-8 routing, 256 experts and EP64. Routing
    must guarantee at least 117,192 capacity-padded receive rows per rank
    for reuse of the planner's grad-H2 workspace. Each rank must specify the
    same execution policy. Inference and unsupported shapes use the fallback
    callback. Like the fallback, the fused path supports eager orchestration
    and CUDA graph capture, with gradients accumulated through nested autograd.

    Args:
        weight_getter: Returns the live contiguous local CUDA BF16 gain of
            shape [4096]. Called in both passes to honor a framework's
            parameter materialization or sharding lifecycle. The backward
            call must produce a differentiable tensor.
        eps: Positive epsilon added to the mean square.
        gain_center: Constant added to the weight before input scaling.
        fallback: Equivalent shape-preserving eager postprocess for other
            execution modes, including inference.
    """

    weight_getter: Callable[[], torch.Tensor]
    eps: float
    gain_center: float
    fallback: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self) -> None:
        """Validate the static parameters before selecting an execution path."""
        if not callable(self.weight_getter) or not callable(self.fallback):
            raise TypeError(
                "input-scaled RMSNorm requires a weight getter and fallback"
            )
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("input-scaled RMSNorm epsilon must be positive and finite")
        if not math.isfinite(self.gain_center):
            raise ValueError("input-scaled RMSNorm gain center must be finite")

    def __call__(self, input_MD: torch.Tensor) -> torch.Tensor:
        """Apply the shape-preserving fallback outside the fused training path."""
        return self.fallback(input_MD)


@dataclass(frozen=True)
class _ResolvedInputScaledRMSNorm:
    """Bind router IDs and live gain access to native BF16 orchestration."""

    config: InputScaledRMSNormPostprocess
    expert_ids_TK: torch.Tensor

    def _weight(self) -> torch.Tensor:
        """Resolve the live local gain and validate its storage contract."""
        weight_D = self.config.weight_getter()
        if (
            weight_D.shape != (_HIDDEN_DIM,)
            or weight_D.dtype != torch.bfloat16
            or weight_D.device != self.expert_ids_TK.device
            or not weight_D.is_contiguous()
        ):
            raise ValueError("expected a contiguous local CUDA BF16 input-scale gain")
        return weight_D

    def forward(
        self,
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        save: bool,
        save_buffer: torch.Tensor | None,
        save_offset: torch.Tensor | None,
        save_condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, None, torch.dtype]:
        """Save H3 in the activation slot when requested and reduce scaled routes."""
        from ._triton_ops import conditional_copy_activations
        from .kernels.triton.input_scaled_rmsnorm import route_forward

        if (
            not save
            or output_dtype != torch.bfloat16
            or save_buffer is None
            or save_offset is None
            or save_condition is None
        ):
            raise ValueError(
                "input-scaled RMSNorm requires a BF16 training activation slot"
            )
        weight_D = self._weight()
        conditional_copy_activations(
            condition=save_condition,
            lhs=None,
            lhs_offset=None,
            rhs=h3_MD,
            rhs_offset=save_offset,
            activation_buffer=save_buffer,
            copy_to_buffer=True,
        )
        output_TD = torch.empty(
            (_NUM_TOKENS, _HIDDEN_DIM), device=h3_MD.device, dtype=output_dtype
        )
        route_forward[(_NUM_TOKENS,)](
            h3_MD,
            weight_D,
            scores_TK,
            self.expert_ids_TK,
            output_TD,
            _HIDDEN_DIM,
            _TOP_K,
            self.config.eps,
            self.config.gain_center,
            num_warps=32,
            enable_fp_fusion=False,
        )
        return output_TD, None, None, output_dtype

    def backward(
        self,
        grad_output_TD: torch.Tensor,
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        published_MD: torch.Tensor,
        *,
        activation_buffer: torch.Tensor,
        h3_offset_1: torch.Tensor,
        need_recompute_1: torch.Tensor,
        grad_h2_offset_1: torch.Tensor,
        num_recv_rows_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Differentiate the normalized routes and publish H3 gradients in place."""
        from .kernels.triton.input_scaled_rmsnorm import (
            finish_gain,
            reduce_gain_tiles,
            route_backward,
        )

        if h3_MD.data_ptr() != published_MD.data_ptr():
            raise ValueError("expected W2 output in the local combine buffer")
        # Device-resident counts may change on graph replay. Verify that both
        # stages of gain-gradient reduction fit the planner's grad-H2 region.
        torch._assert_async(
            num_recv_rows_1.ge(_MIN_RECV_ROWS),
            "input-scaled RMSNorm has insufficient grad-H2 workspace",
        )
        grad_output_TD = grad_output_TD.to(torch.bfloat16).contiguous()
        # Resolve the weight under grad mode so a framework-provided local view
        # keeps its edge to the live parameter.
        with torch.enable_grad():
            weight_D = self._weight()
        if not weight_D.requires_grad:
            raise ValueError("input-scaled RMSNorm requires a differentiable gain")
        grad_scores_TK = torch.empty_like(scores_TK)
        grad_weight_D = torch.empty_like(weight_D)
        route_backward[(_PARTIALS,)](
            h3_MD,
            activation_buffer,
            h3_offset_1,
            need_recompute_1,
            grad_output_TD,
            weight_D,
            scores_TK,
            self.expert_ids_TK,
            published_MD,
            grad_scores_TK,
            activation_buffer,
            grad_h2_offset_1,
            _ROUTES,
            _HIDDEN_DIM,
            _TOP_K,
            _ROWS_PER_CTA,
            self.config.eps,
            self.config.gain_center,
            num_warps=16,
            enable_fp_fusion=False,
        )
        reduce_gain_tiles[(_REDUCTION_TILES, _HIDDEN_DIM // _TILE_WIDTH)](
            activation_buffer,
            grad_h2_offset_1,
            _PARTIALS,
            _HIDDEN_DIM,
            _PARTIALS_PER_TILE,
            _TILE_WIDTH,
            num_warps=8,
        )
        finish_gain[(_HIDDEN_DIM // _TILE_WIDTH,)](
            activation_buffer,
            grad_h2_offset_1,
            grad_weight_D,
            _PARTIALS,
            _REDUCTION_TILES,
            _HIDDEN_DIM,
            _PARTIALS_PER_TILE,
            _TILE_WIDTH,
            num_warps=8,
        )
        torch.autograd.backward(weight_D, grad_weight_D)
        return published_MD, grad_scores_TK, True


def _resolve_input_scaled_rmsnorm(
    postprocess: InputScaledRMSNormPostprocess,
    *,
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    intermediate_dim: int,
    num_experts: int,
    ep_size: int,
) -> _ResolvedInputScaledRMSNorm | None:
    """Select the reviewed BF16 shape, leaving other cases to the fallback."""
    if (
        x_TD.shape == (_NUM_TOKENS, _HIDDEN_DIM)
        and intermediate_dim == _HIDDEN_DIM
        and num_experts == 256
        and ep_size == 64
        and x_TD.is_cuda
        and x_TD.dtype == torch.bfloat16
        and topk_scores_TK.dtype == torch.float32
        and topk_expert_ids_TK.shape == (_NUM_TOKENS, _TOP_K)
        and topk_scores_TK.shape == topk_expert_ids_TK.shape
        and topk_expert_ids_TK.is_contiguous()
        and topk_scores_TK.is_contiguous()
    ):
        return _ResolvedInputScaledRMSNorm(postprocess, topk_expert_ids_TK)
    return None
