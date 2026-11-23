# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import dist_moe
from dist_moe._input_scaled_rmsnorm import (
    _MIN_RECV_ROWS,
    _ResolvedInputScaledRMSNorm,
    _resolve_input_scaled_rmsnorm,
)


def test_input_scaled_rmsnorm_policy_supports_eager_fallback():
    """The public policy preserves caller output on fallback and rejects bad epsilon."""
    gain = torch.ones(16)
    rows = torch.ones((2, 16))
    policy = dist_moe.InputScaledRMSNormPostprocess(
        weight_getter=lambda: gain,
        eps=1e-8,
        gain_center=1.0,
        fallback=lambda x: x + gain,
    )
    options = dist_moe.ExecutionOptions(experts_output_postprocess=policy)
    assert options.requires_eager
    torch.testing.assert_close(policy(rows), rows + gain, rtol=0, atol=0)
    with pytest.raises(ValueError, match="epsilon"):
        dist_moe.InputScaledRMSNormPostprocess(
            weight_getter=policy.weight_getter,
            eps=0.0,
            gain_center=1.0,
            fallback=policy.fallback,
        )


def test_input_scaled_rmsnorm_resolves_only_supported_geometry():
    """Metadata selection accepts the exact CUDA training shape without allocating it."""
    tokens = 29_184
    with FakeTensorMode():
        gain = torch.empty((4_096,), device="cuda", dtype=torch.bfloat16)
        x_TD = torch.empty((tokens, 4_096), device="cuda", dtype=torch.bfloat16)
        expert_ids_TK = torch.empty((tokens, 8), device="cuda", dtype=torch.int64)
        scores_TK = torch.empty((tokens, 8), device="cuda", dtype=torch.float32)

    policy = dist_moe.InputScaledRMSNormPostprocess(
        weight_getter=lambda: gain, eps=1e-8, gain_center=1.0, fallback=lambda x: x
    )
    kwargs = dict(
        x_TD=x_TD,
        topk_expert_ids_TK=expert_ids_TK,
        topk_scores_TK=scores_TK,
        intermediate_dim=4_096,
        num_experts=256,
        ep_size=64,
    )
    selected = _resolve_input_scaled_rmsnorm(policy, **kwargs)
    assert selected is not None
    assert selected.expert_ids_TK is expert_ids_TK
    assert selected.config is policy
    assert _resolve_input_scaled_rmsnorm(policy, **(kwargs | {"ep_size": 32})) is None
    assert (
        _resolve_input_scaled_rmsnorm(policy, **(kwargs | {"intermediate_dim": 2048}))
        is None
    )


def test_input_scaled_rmsnorm_backward_gradients_reach_live_gain(monkeypatch):
    """Validate workspace rejection and gain autograd ownership with launch stubs."""
    gain = torch.ones(4_096, dtype=torch.bfloat16, requires_grad=True)
    routes = _ResolvedInputScaledRMSNorm(
        dist_moe.InputScaledRMSNormPostprocess(
            weight_getter=lambda: gain.view_as(gain),
            eps=1e-8,
            gain_center=1.0,
            fallback=lambda x: x,
        ),
        expert_ids_TK=torch.zeros((2, 8), dtype=torch.int64),
    )

    class Launch:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return self.fn

    kernels = ModuleType("dist_moe.kernels.triton.input_scaled_rmsnorm")
    kernels.route_backward = Launch(lambda *args, **kwargs: None)
    kernels.reduce_gain_tiles = Launch(lambda *args, **kwargs: None)
    kernels.finish_gain = Launch(
        lambda base, offset, result, *args, **kwargs: result.fill_(2)
    )
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)

    h3_MD = torch.empty((16, 4_096), dtype=torch.bfloat16)
    grad_output_TD = torch.empty((2, 4_096), dtype=torch.bfloat16)
    scores_TK = torch.empty((2, 8), dtype=torch.float32)
    params = dict(
        activation_buffer=torch.empty(0, dtype=torch.uint8),
        h3_offset_1=torch.tensor([0]),
        need_recompute_1=torch.tensor([True]),
        grad_h2_offset_1=torch.tensor([0]),
    )
    with pytest.raises(RuntimeError, match="insufficient grad-H2 workspace"):
        routes.backward(
            grad_output_TD,
            h3_MD,
            scores_TK,
            h3_MD,
            num_recv_rows_1=torch.tensor([_MIN_RECV_ROWS - 1]),
            **params,
        )
    with torch.no_grad():
        published, grad_scores, written_in_place = routes.backward(
            grad_output_TD,
            h3_MD,
            scores_TK,
            h3_MD,
            num_recv_rows_1=torch.tensor([_MIN_RECV_ROWS]),
            **params,
        )
    assert published is h3_MD and written_in_place
    assert grad_scores.shape == scores_TK.shape
    torch.testing.assert_close(gain.grad, torch.full_like(gain, 2), rtol=0, atol=0)
