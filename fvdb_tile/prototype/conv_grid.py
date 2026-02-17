# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
conv_grid: topology expansion for sparse convolution.

Given active input voxel coordinates, kernel size, and stride, compute the
set of unique output coordinates.  The semantic equation:

    output y is active iff there exists active x and kernel offset k
    such that x = y * stride + k  (component-wise).

**Adverb decomposition (stride=1):**

The algorithm is expressed as a composition of leading-shape adverbs:

    expanded  = EachLeft(EachRight(EachBoth(Add)), coords, offsets)
    flat      = fuse(expanded)
    reshaped  = reshape(flat, [-1])
    codes     = Morton3dSigned(reshaped)
    sorted    = Sort(codes)
    unique    = Unique(sorted)
    result    = MortonDecode3d(unique)

Type derivation:

    coords:   (*,) / (3,) i32     -- N active voxel coordinates
    offsets:  (K,) / (3,) i32     -- K centered kernel offsets
    expanded: (*,) / (K,) / (3,) i32  -- outer product via adverb composition
    flat:     (*, K) / (3,) i32   -- fuse merges two nesting levels
    reshaped: (*,) / (3,) i32     -- flatten rank-2 leading shape to rank-1
    codes:    (*,) / i64          -- signed morton encoding (trivial 1D from here)
    sorted:   (*,) / i64          -- 1D sort
    unique:   (*,) / i64          -- 1D dedup
    result:   (*,) / (3,) i32     -- decode back to coordinates

EachBoth(Add) makes the component-wise (3,) i32 + (3,) i32 zip-iteration
explicit rather than relying on implicit broadcasting in the verb.

Morton3dSigned maps signed (3,) i32 coordinates to a single i64 code via
bit-interleaving (21 bits per axis, offset by 2^20 for sign handling).
This replaces multi-key lexicographic sort with trivial 1D sort.

**Leaf-level optimisation (future work):**

The voxel-level expansion creates O(N * kernel_volume) candidates before
dedup.  A two-phase leaf-level approach reduces this:

Phase 1 -- Coarse (leaf expansion):
  - Extract source leaf coords: unique(active_coords >> 3)
  - Leaf-level kernel offsets per axis i:
      from  -ceil(kernel_half[i] / 8)  to  floor((7 + kernel_half[i]) / 8)
    For k=3,5,7: always {-1, 0, 1} per axis (27 candidates per source leaf)
  - EachLeft(EachRight(EachBoth(Add))) + Morton3dSigned + Sort + Unique +
    MortonDecode3d on leaf coords
  - Memory: O(L_src * 27) leaf candidates vs O(N_voxels * K) voxel candidates

Phase 2 -- Fine (voxel fill per destination leaf):
  - For each destination leaf D, check which of its 512 positions (8x8x8)
    are reachable from any source voxel via the kernel
  - Requires source grid lookup (hash set or CIG)
  - Each thread block handles one destination leaf (GPU-friendly)
  - Total lookups: O(unique_dst_leaves * 512 * K_volume) -- comparable to
    voxel-level but with bounded memory per leaf
"""

from __future__ import annotations

import numpy as np
import torch

from .dsl_pipeline import PipelineExecutable, compile_source
from .ops import Value, morton3d_signed, morton3d_decode
from .types import Dynamic, ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# DSL pipelines (compiled once at import time)
# ---------------------------------------------------------------------------

CONV_GRID_PIPELINE: PipelineExecutable = compile_source("""
expanded = EachLeft(EachRight(EachBoth(Add)), Input("coords"), Input("offsets"))
flat = fuse(expanded)
reshaped = reshape(flat, Const([-1]))
codes = Morton3dSigned(reshaped)
sorted_codes = Sort(codes)
unique_codes = Unique(sorted_codes)
result = MortonDecode3d(unique_codes)
result
""")

MORTON_DEDUP_PIPELINE: PipelineExecutable = compile_source("""
codes = Morton3dSigned(Input("coords"))
sorted_codes = Sort(codes)
unique_codes = Unique(sorted_codes)
result = MortonDecode3d(unique_codes)
result
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_vec3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected scalar or length-3, got {value}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_offsets_centered(kernel_size: tuple[int, int, int]) -> np.ndarray:
    """Generate (K, 3) i32 kernel offset array matching fVDB convention.

    For odd kernel sizes: centered offsets [-(k-1)/2, (k-1)/2].
    For even kernel sizes: offsets [0, k-1].

    This matches fVDB's BuildGridForConv.cu, where the destination
    coordinate is computed as ``src + offset``.
    """
    ranges = []
    for k in kernel_size:
        if k <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if k % 2 == 0:
            ranges.append(np.arange(0, k, dtype=np.int32))
        else:
            half = (k - 1) // 2
            ranges.append(np.arange(-half, half + 1, dtype=np.int32))
    gx, gy, gz = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        (M, 3) i32 numpy array of unique output coordinates, sorted by
        morton code.
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

    offsets = _kernel_offsets_centered(ks)

    if st == (1, 1, 1):
        return _conv_grid_stride1(active_np, offsets, device)
    else:
        return _conv_grid_strided(active_np, offsets, st, device)


def _conv_grid_stride1(
    active_np: np.ndarray,
    offsets: np.ndarray,
    device: str,
) -> np.ndarray:
    """Full DSL pipeline path for stride=1.

    Uses EachLeft(EachRight(EachBoth(Add))) for the outer-product expansion,
    fuse + reshape to flatten, Morton3d for encoding, 1D Sort + Unique for
    dedup, and MortonDecode3d to recover coordinates.
    """
    n = active_np.shape[0]
    k = offsets.shape[0]

    coords_val = Value(
        Type(Shape(Static(n)), Type(Shape(Static(3)), ScalarType.I32)),
        active_np.copy(),
    )
    offsets_val = Value(
        Type(Shape(Static(k)), Type(Shape(Static(3)), ScalarType.I32)),
        offsets.copy(),
    )

    result = CONV_GRID_PIPELINE.run(
        {"coords": coords_val, "offsets": offsets_val},
        device=device,
    )
    return result.output.data.astype(np.int32, copy=False)


def _conv_grid_strided(
    active_np: np.ndarray,
    offsets: np.ndarray,
    stride: tuple[int, int, int],
    device: str,
) -> np.ndarray:
    """Hybrid path for stride > 1.

    Numpy handles the expansion and stride filtering (the DSL lacks
    Mod/IntDiv primitives).  Morton3d + Sort + Unique + MortonDecode3d
    handles dedup via the pipeline.
    """
    stride_arr = np.array(stride, dtype=np.int32)

    # Adverb expansion: EachLeft(EachRight(EachBoth(Add)))
    # Physical: broadcasting equivalent
    cand = active_np[:, None, :] + offsets[None, :, :]  # (N, K, 3)
    cand = cand.reshape(-1, 3)

    # Stride filtering
    divisible = np.all(cand % stride_arr == 0, axis=1)
    cand = cand[divisible]
    if cand.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    cand = cand // stride_arr
    cand = cand.astype(np.int32, copy=False)

    # Morton-based dedup via pipeline
    coords_val = Value(
        Type(Shape(Static(cand.shape[0])), Type(Shape(Static(3)), ScalarType.I32)),
        cand.copy(),
    )
    result = MORTON_DEDUP_PIPELINE.run({"coords": coords_val}, device=device)
    return result.output.data.astype(np.int32, copy=False)
