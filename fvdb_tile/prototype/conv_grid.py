# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# DSL status: fully DSL-driven
"""
conv_grid: topology expansion for sparse convolution.

Given active input voxel coordinates, kernel size, and stride, compute the
set of unique output coordinates.  The semantic equation:

    output y is active iff there exists active x and kernel offset k
    such that (x + k) is divisible by stride and y = (x + k) / stride.

**Execution is fully DSL-driven.**  The algorithm is expressed as a DSL
program string, parsed into an AST, planned by the barrier-aware pipeline
planner, and executed via the pipeline executor.  This demonstrates:

1. The barrier-based pipeline (cutile + collective segments) works for a
   real multi-step topology operation.
2. The DSL can express non-trivial grid operations as small compositions
   of reusable primitives.

**Stride=1 DSL program** (5 bindings, 3 barriers)::

    flat    = reshape(fuse(EachLeft(EachRight(EachBoth(Add)),
                 Input("coords"), Input("offsets"))), Const([-1]))
    codes   = HierarchicalKey(flat, Const([3, 4, 5]))
    sorted  = Sort(codes)                          # barrier
    unique  = Unique(sorted)                       # barrier
    result  = HierarchicalKeyDecode(unique, Const([3, 4, 5]))

**Stride>1 DSL program** (11 bindings, 4 barriers)::

    flat      = reshape(fuse(EachLeft(EachRight(EachBoth(Add)),
                    Input("coords"), Input("offsets"))), Const([-1]))
    remainder = Mod(flat, Input("stride"))
    eq_zero   = Eq(remainder, Const([0, 0, 0]))
    all_div   = All(eq_zero)
    div_idx   = Where(all_div)                     # barrier
    filtered  = Gather(flat, div_idx)
    scaled    = FloorDiv(filtered, Input("stride"))
    codes     = HierarchicalKey(scaled, Const([3, 4, 5]))
    sorted    = Sort(codes)                        # barrier
    unique    = Unique(sorted)                     # barrier
    result    = HierarchicalKeyDecode(unique, Const([3, 4, 5]))

The pipeline planner partitions these into alternating cutile/collective
segments.  The executor dispatches collectives to ``torch.sort``,
``torch.unique``, ``torch.nonzero`` and runs everything else through the
tree-walk evaluator with vectorized torch ops on the target device.
"""

from __future__ import annotations

import torch

from .dsl_pipeline import compile_source, PipelineExecutable
from .ops import Value
from .types import Dynamic, ScalarType, Shape, Static, Type


# ---------------------------------------------------------------------------
# Default CIG bit-widths (leaf=8^3, lower=16^3, upper=32^3)
# ---------------------------------------------------------------------------

DEFAULT_BIT_WIDTHS: list[int] = [3, 4, 5]


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: dict[str, PipelineExecutable] = {}


# ---------------------------------------------------------------------------
# DSL program builders
# ---------------------------------------------------------------------------


def _conv_grid_source(stride_1: bool, bit_widths: list[int]) -> str:
    """Build the DSL source string for conv_grid."""
    bw = repr(bit_widths)
    lines = [
        'flat = reshape(fuse(EachLeft(EachRight(EachBoth(Add)), Input("coords"), Input("offsets"))), Const([-1]))',
    ]
    if not stride_1:
        lines += [
            'remainder = Mod(flat, Input("stride"))',
            "eq_zero = Eq(remainder, Const([0, 0, 0]))",
            "all_div = All(eq_zero)",
            "div_idx = Where(all_div)",
            "filtered = Gather(flat, div_idx)",
            'scaled = FloorDiv(filtered, Input("stride"))',
            f"codes = HierarchicalKey(scaled, Const({bw}))",
        ]
    else:
        lines += [
            f"codes = HierarchicalKey(flat, Const({bw}))",
        ]
    lines += [
        "sorted = Sort(codes)",
        "unique = Unique(sorted)",
        f"result = HierarchicalKeyDecode(unique, Const({bw}))",
        "result",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_vec3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"Expected scalar or length-3, got {value}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_offsets_centered(kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Generate (K, 3) i32 kernel offset tensor matching fVDB convention.

    For odd kernel sizes: centered offsets [-(k-1)/2, (k-1)/2].
    For even kernel sizes: offsets [0, k-1].
    """
    ranges = []
    for k in kernel_size:
        if k <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if k % 2 == 0:
            ranges.append(torch.arange(0, k, dtype=torch.int32))
        else:
            half = (k - 1) // 2
            ranges.append(torch.arange(-half, half + 1, dtype=torch.int32))
    gx, gy, gz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_COORD_ELEM = Type(Shape(Static(3)), ScalarType.I32)


def conv_grid(
    active_coords: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    device: str = "cpu",
    bit_widths: list[int] | None = None,
) -> torch.Tensor:
    """Compute unique output coordinates for a sparse convolution topology.

    Execution is fully DSL-driven: the algorithm is expressed as a DSL
    program string, parsed, planned by the barrier-aware pipeline planner,
    and executed via the pipeline executor.

    Args:
        active_coords: (N, 3) i32 active input voxel coordinates.
        kernel_size: scalar or (kx, ky, kz) kernel dimensions.
        stride: scalar or (sx, sy, sz) stride.
        device: torch device for computation ("cpu" or "cuda").
        bit_widths: CIG level bit-widths, leaf-first.  Default [3, 4, 5].

    Returns:
        (M, 3) i32 torch tensor of unique output coordinates, sorted by
        hierarchical key (CIG-compatible order).
    """
    if bit_widths is None:
        bit_widths = DEFAULT_BIT_WIDTHS

    dev = torch.device(device)

    if isinstance(active_coords, torch.Tensor):
        active_t = active_coords.to(device=dev, dtype=torch.int32)
    else:
        raise TypeError(f"Expected Tensor, got {type(active_coords)}")

    if active_t.ndim != 2 or active_t.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coords, got {active_t.shape}")
    if active_t.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=dev)

    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)
    if min(st) <= 0:
        raise ValueError(f"stride must be positive, got {st}")

    offsets = _kernel_offsets_centered(ks).to(dev)
    stride_1 = st == (1, 1, 1)

    # --- Compile (cached) ---
    cache_key = f"{stride_1}_{bit_widths}"
    if cache_key not in _PIPELINE_CACHE:
        source = _conv_grid_source(stride_1, bit_widths)
        _PIPELINE_CACHE[cache_key] = compile_source(source)
    pipeline = _PIPELINE_CACHE[cache_key]

    # --- Build inputs ---
    n_offsets = offsets.shape[0]
    inputs: dict[str, Value] = {
        "coords": Value(Type(Shape(Dynamic()), _COORD_ELEM), active_t),
        "offsets": Value(Type(Shape(Static(n_offsets)), _COORD_ELEM), offsets),
    }
    if not stride_1:
        stride_t = torch.tensor(st, dtype=torch.int32, device=dev)
        inputs["stride"] = Value(Type(Shape(Static(3)), ScalarType.I32), stride_t)

    # --- Execute via pipeline ---
    # Use device=None: the evaluator's torch ops are device-agnostic and
    # run on CUDA when inputs are on CUDA.  cuTile compilation is not
    # needed here -- the collectives (Sort, Unique) dominate the runtime.
    result = pipeline.run(inputs, device=None)
    output = result.output.data

    if isinstance(output, torch.Tensor):
        return output.to(device=dev, dtype=torch.int32)
    return torch.empty((0, 3), dtype=torch.int32, device=dev)
