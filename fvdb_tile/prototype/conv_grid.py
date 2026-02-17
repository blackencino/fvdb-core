# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid prototype pipeline with ephemeral CIG materialization.

Semantics are defined over coordinate values:
  1) expand candidate output coordinates from active input coordinates
  2) sort + deduplicate coordinates (collective stage)
  3) materialize CIG3 as an execution artifact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cig import CompressedCIG3, build_compressed_cig3_from_unique
from .dsl_pipeline import PipelineExecutable, PipelinePlan, compile_source, plan_source
from .ops import Value
from .types import ScalarType, Shape, Static, Type


CONV_GRID_COLLECTIVE_PROGRAM = """
sorted = Sort(Input("coords"))
unique = Unique(sorted)
unique
"""
CONV_GRID_COLLECTIVE_PIPELINE: PipelineExecutable = compile_source(CONV_GRID_COLLECTIVE_PROGRAM)


@dataclass(frozen=True)
class ConvGridResult:
    active_coords: torch.Tensor
    cig3: CompressedCIG3
    collective_plan: PipelinePlan


def _as_vec3_int(value: int | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if len(value) != 3:
        raise ValueError(f"Expected scalar or length-3 value, got {value}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_offsets(kernel_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    kx, ky, kz = kernel_size
    if min(kx, ky, kz) <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    xs = torch.arange(kx, device=device, dtype=torch.int32)
    ys = torch.arange(ky, device=device, dtype=torch.int32)
    zs = torch.arange(kz, device=device, dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


def _collective_dedup_with_dsl(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N, 3), got {tuple(coords.shape)}")
    coords_np = coords.detach().cpu().numpy().astype(np.int32, copy=False)
    coords_val = Value(
        Type(Shape(Static(coords_np.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        coords_np.copy(),
    )
    out = CONV_GRID_COLLECTIVE_PIPELINE.run({"coords": coords_val}).output
    out_np = out.data.astype(np.int32, copy=False)
    return torch.from_numpy(out_np).to(device=coords.device, dtype=torch.int32)


def conv_grid_ephemeral_cig(
    active_coords: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
) -> ConvGridResult:
    """Compute a conv-grid topology and materialize it into CIG3 ephemerally.

    Correctness-first prototype semantics:
      y is an output coordinate if there exists active x and kernel offset k
      such that x = y * stride + k (component-wise).
    """
    if active_coords.ndim != 2 or active_coords.shape[1] != 3:
        raise ValueError(f"Expected active_coords shape (N, 3), got {tuple(active_coords.shape)}")
    if active_coords.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Expected integer active_coords dtype, got {active_coords.dtype}")

    device = active_coords.device
    active_i32 = active_coords.to(torch.int32)
    ks = _as_vec3_int(kernel_size)
    st = _as_vec3_int(stride)
    if min(st) <= 0:
        raise ValueError(f"stride must be positive, got {st}")

    offsets = _kernel_offsets(ks, device=device)  # (K, 3)

    # Expand candidates from the value-semantic equation x = y*stride + k.
    cand = active_i32[:, None, :] - offsets[None, :, :]  # (N, K, 3)
    if st != (1, 1, 1):
        stride_vec = torch.tensor(st, device=device, dtype=torch.int32)
        divisible = ((cand % stride_vec) == 0).all(dim=2)
        cand = cand[divisible]
        if cand.numel() == 0:
            unique_coords = cand.reshape(0, 3)
        else:
            unique_coords = _collective_dedup_with_dsl(cand.reshape(-1, 3) // stride_vec)
    else:
        unique_coords = _collective_dedup_with_dsl(cand.reshape(-1, 3))

    cig3 = build_compressed_cig3_from_unique(unique_coords)
    return ConvGridResult(
        active_coords=unique_coords,
        cig3=cig3,
        collective_plan=plan_source(CONV_GRID_COLLECTIVE_PROGRAM),
    )
