# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-owned host snapshots for BF16 down-projection replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._activation_buffer import _align_bytes


def _h3_stage_offsets(
    *,
    scratch_base: int,
    scratch_bytes: int,
    max_receive_rows: int,
    hidden_dim: int,
    intermediate_dim: int,
    route_bytes: int,
) -> tuple[int, int]:
    """Return two aligned staging banks in the forward-unused scratch prefix.

    The BF16 forward planner allocates gathered input, H1, and H2 downward
    from scratch end when replay is selected. Routing traps on more than
    max_receive_rows before any forward activation write.
    """
    forward_bytes = (
        _align_bytes(max_receive_rows * hidden_dim * 2)
        + _align_bytes(max_receive_rows * intermediate_dim * 4)
        + _align_bytes(max_receive_rows * intermediate_dim * 2)
    )
    bank_bytes = _align_bytes(route_bytes)
    if scratch_base != _align_bytes(scratch_base) or (
        scratch_bytes - forward_bytes < 2 * bank_bytes
    ):
        raise ValueError(
            "BF16 H3 offload requires room for two staging banks below "
            "the worst-case forward activation scratch"
        )
    return scratch_base, scratch_base + bank_bytes


@dataclass(eq=False, slots=True)
class _H3Frame:
    """One training forward and its corresponding local combine storage."""

    layer: int
    need_recompute: torch.Tensor
    combine_MD: torch.Tensor
    snapshot: torch.Tensor | None
    snapshot_done: torch.cuda.Event | None = None


class _BF16H3Offload:
    """Order two forward staging banks and one LIFO backward schedule."""

    def __init__(
        self,
        buffer: torch.Tensor,
        stage_offsets: tuple[int, int],
        *,
        route_bytes: int,
        layers: int,
        selected_layers: tuple[int, ...],
    ) -> None:
        """Allocate persistent host snapshots and CUDA streams before capture."""
        self._device = buffer.device
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Create the BF16 H3 context before CUDA graph capture"
                )
            self._stage_stream = torch.cuda.Stream(device=self._device)
            self._snapshot_stream = torch.cuda.Stream(device=self._device)
            self._restore_stream = torch.cuda.Stream(device=self._device)
        self._stages = tuple(
            buffer.narrow(0, offset, route_bytes) for offset in stage_offsets
        )
        self._snapshots = {
            layer: torch.empty(
                route_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            for layer in selected_layers
        }
        self._layers = layers
        self._probe_layer = selected_layers[0]
        self._probed = False
        self._stack: list[_H3Frame] = []
        self._reversing = False
        self._capturing = False
        self._backward: _H3Frame | None = None
        self._prefetched: tuple[_H3Frame, torch.cuda.Event] | None = None
        self._combine_done: torch.cuda.Event | None = None
        self._stage_done: list[torch.cuda.Event | None] = [None, None]
        self._bank = 0

    def _record_compute(self) -> torch.cuda.Event:
        """Record a producer or release event on the caller's CUDA stream."""
        return torch.cuda.current_stream(self._device).record_event()

    def _wait_compute(self, event: torch.cuda.Event) -> None:
        """Order scratch or combine use after the relevant transfer."""
        torch.cuda.current_stream(self._device).wait_event(event)

    def _copy(
        self,
        stream: torch.cuda.Stream,
        dst: torch.Tensor,
        src: torch.Tensor,
        *predecessors: torch.cuda.Event | None,
    ) -> torch.cuda.Event:
        """Copy the full local route extent as bytes on one transfer stream."""
        with torch.cuda.stream(stream):
            for predecessor in predecessors:
                if predecessor is not None:
                    stream.wait_event(predecessor)
            dst.view(-1).view(torch.uint8).copy_(
                src.view(-1).view(torch.uint8), non_blocking=True
            )
            return stream.record_event()

    def require_drained(self) -> None:
        """Prevent reset or scratch-only execution while H3 state is live."""
        if self._stack:
            raise RuntimeError(
                "BF16 H3 offload requires one complete forward/backward schedule"
            )

    def before_forward(self) -> None:
        """Protect the previous combine read before the next EP publication."""
        if self._reversing or len(self._stack) == self._layers:
            raise RuntimeError(
                "BF16 H3 offload requires all forwards before all backwards"
            )
        capturing = torch.cuda.is_current_stream_capturing()
        if not self._stack:
            self._capturing = capturing
            if capturing and not self._probed:
                raise RuntimeError(
                    "Complete an eager BF16 H3 parity probe before CUDA graph capture"
                )
        elif capturing != self._capturing:
            raise RuntimeError("BF16 H3 forward and backward must use the same capture")
        if self._combine_done is not None:
            self._wait_compute(self._combine_done)
            self._combine_done = None

    def saves_next_forward(self) -> bool:
        """Return whether the current layer owns a pinned H3 snapshot."""
        return len(self._stack) in self._snapshots

    def after_forward(
        self, need_recompute: torch.Tensor, combine_MD: torch.Tensor
    ) -> _H3Frame:
        """Retain the forward identity and stage a selected H3 snapshot."""
        layer = len(self._stack)
        frame = _H3Frame(layer, need_recompute, combine_MD, self._snapshots.get(layer))
        if frame.snapshot is not None:
            bank = self._bank
            self._bank = 1 - bank
            stage = self._stages[bank]
            self._combine_done = self._copy(
                self._stage_stream,
                stage,
                combine_MD,
                self._record_compute(),
                self._stage_done[bank],
            )
            frame.snapshot_done = self._copy(
                self._snapshot_stream, frame.snapshot, stage, self._combine_done
            )
            self._stage_done[bank] = frame.snapshot_done
        self._stack.append(frame)
        return frame

    def before_backward(self, frame: _H3Frame, need_recompute: torch.Tensor) -> None:
        """Match reverse order and release both borrowed banks to backward."""
        if (
            self._backward is not None
            or not self._stack
            or self._stack[-1] is not frame
            or frame.need_recompute.data_ptr() != need_recompute.data_ptr()
        ):
            raise RuntimeError("BF16 H3 backward must match the most recent forward")
        if not self._reversing:
            if len(self._stack) != self._layers:
                raise RuntimeError("BF16 H3 backward requires a complete forward slot")
            if torch.cuda.is_current_stream_capturing() != self._capturing:
                raise RuntimeError(
                    "BF16 H3 forward and backward must use the same capture"
                )
            for event in self._stage_done:
                if event is not None:
                    self._wait_compute(event)
            self._stage_done = [None, None]
            self._combine_done = None
            self._bank = 0
            self._reversing = True
        self._backward = frame

    def probes(self, frame: _H3Frame) -> bool:
        """Return whether this is the first eager verification of H3 replay."""
        return frame.layer == self._probe_layer and not self._probed

    def check_replay(
        self, frame: _H3Frame, replayed_MD: torch.Tensor, expert_ids_TK: torch.Tensor
    ) -> None:
        """Compare valid H3 rows bitwise against the pinned forward snapshot."""
        if self._backward is not frame or not self.probes(frame):
            raise RuntimeError("BF16 H3 parity probe must match the selected backward")
        assert frame.snapshot is not None and frame.snapshot_done is not None
        frame.snapshot_done.synchronize()
        valid_M = expert_ids_TK.reshape(-1).ge(0).cpu()
        expected_MD = frame.snapshot.view(torch.int16).view(replayed_MD.shape)
        replayed_MD = replayed_MD.view(torch.int16)
        if not bool(valid_M.any()):
            raise RuntimeError("BF16 H3 parity probe requires at least one valid route")
        for start in range(0, valid_M.numel(), 4096):
            rows = slice(start, start + 4096)
            different_MD = replayed_MD[rows].cpu() != expected_MD[rows]
            if bool((different_MD & valid_M[rows, None]).any()):
                raise RuntimeError("BF16 H3 replay differs from the forward snapshot")
        self._probed = True

    def restore(self, frame: _H3Frame) -> torch.Tensor:
        """Wait for the immediately preceding EP release to restore local H3."""
        prefetched = self._prefetched
        if (
            self._backward is not frame
            or prefetched is None
            or prefetched[0] is not frame
        ):
            raise RuntimeError(
                "BF16 H3 backward reached a selected layer without its restore"
            )
        self._wait_compute(prefetched[1])
        self._prefetched = None
        return frame.combine_MD

    def after_backward_barrier(self, frame: _H3Frame) -> None:
        """Prefetch the next selected H3 when all peers have released combine."""
        if (
            self._backward is not frame
            or self._prefetched is not None
            or self.probes(frame)
        ):
            raise RuntimeError("BF16 H3 backward advanced before its restore or probe")
        self._stack.pop()
        self._backward = None
        self._reversing = bool(self._stack)
        if self._stack:
            next_frame = self._stack[-1]
            if next_frame.snapshot is not None and not self.probes(next_frame):
                event = self._copy(
                    self._restore_stream,
                    next_frame.combine_MD,
                    next_frame.snapshot,
                    self._record_compute(),
                    next_frame.snapshot_done,
                )
                self._prefetched = next_frame, event
