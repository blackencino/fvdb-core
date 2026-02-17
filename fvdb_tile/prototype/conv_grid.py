# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid: topology expansion for sparse convolution.

Given active input voxel coordinates, kernel size, and stride, compute the
set of unique output coordinates.  The semantic equation:

    output y is active iff there exists active x and kernel offset k
    such that x = y * stride + k  (component-wise).

The implementation uses the pipeline executor for the Sort + Unique
deduplication step, so collective ops dispatch to torch when a device is
specified.
"""

from __future__ import annotations

import numpy as np
import torch

from .dsl_pipeline import PipelineExecutable, compile_source
from .ops import Value
from .types import Dynamic, ScalarType, Shape, Static, Type


DEDUP_PIPELINE: PipelineExecutable = compile_source("""
sorted = Sort(Input("coords"))
unique = Unique(sorted)
unique
""")


def _as_vec3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected scalar or length-3, got {value}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_offsets(kernel_size: tuple[int, int, int]) -> np.ndarray:
    """Generate (K, 3) i32 kernel offset array from kernel_size."""
    kx, ky, kz = kernel_size
    if min(kx, ky, kz) <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    xs = np.arange(kx, dtype=np.int32)
    ys = np.arange(ky, dtype=np.int32)
    zs = np.arange(kz, dtype=np.int32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def conv_grid(
    active_coords: np.ndarray | torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    device: str = "cpu",
) -> np.ndarray:
    """Compute unique output coordinates for a sparse convolution topology.

    Args:
        active_coords: (N, 3) i32 active input voxel coordinates.
        kernel_size: scalar or (kx, ky, kz) kernel dimensions.
        stride: scalar or (sx, sy, sz) stride.
        device: torch device for the dedup pipeline ("cpu" or "cuda").

    Returns:
        (M, 3) i32 numpy array of unique output coordinates, sorted
        lexicographically.
    """
    if isinstance(active_coords, torch.Tensor):
        active_np = active_coords.detach().cpu().numpy().astype(np.int32, copy=False)
    else:
        active_np = np.asarray(active_coords, dtype=np.int32)
    if active_np.ndim != 2 or active_np.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coords, got {active_np.shape}")

    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)
    if min(st) <= 0:
        raise ValueError(f"stride must be positive, got {st}")

    offsets = _kernel_offsets(ks)

    # Broadcast expansion: candidates = active[:, None, :] - offsets[None, :, :]
    cand = active_np[:, None, :] - offsets[None, :, :]  # (N, K, 3)

    if st != (1, 1, 1):
        stride_arr = np.array(st, dtype=np.int32)
        divisible = np.all(cand % stride_arr == 0, axis=2)
        cand = cand[divisible]
        if cand.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        cand = cand.reshape(-1, 3) // stride_arr
    else:
        cand = cand.reshape(-1, 3)

    cand = cand.astype(np.int32, copy=False)

    # Dedup via the pipeline executor (Sort + Unique, dispatched to torch)
    coords_val = Value(
        Type(Shape(Static(cand.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        cand.copy(),
    )
    result = DEDUP_PIPELINE.run({"coords": coords_val}, device=device)
    return result.output.data.astype(np.int32, copy=False)
