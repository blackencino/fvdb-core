# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# DSL status: fully DSL-driven
"""
conv_grid_leafwise: topology expansion via leaf-level bitmask dilation.

Given a CIG (with leaf masks and leaf coords) and kernel size, compute
the set of unique output voxel coordinates **without** the N*K dense
expansion of the original conv_grid.

Instead of expanding every active voxel by every kernel offset (N*K
candidates), this algorithm shifts entire 8x8x8 leaf masks and
accumulates them into output leaf slots via hash-map scatter-reduce
with OR.  The work is O(L * K_voxel * 8) word-level ops, where L is
the number of input leaves -- typically 50-100x fewer than N.

**Execution is fully DSL-driven.**  The algorithm is expressed as a
single DSL program string (13 bindings, 4 inputs), parsed into an AST,
planned by the barrier-aware pipeline planner, and executed via the
pipeline executor.  The pipeline segments it into alternating
cutile/collective phases automatically, with DilateLeafMasks dispatched
to the fused CUDA kernel via hook on GPU.

**DSL program** (single unified pipeline)::

    expanded_lc   = reshape(fuse(EachLeft(EachRight(EachBoth(Add)),
                        leaf_coords, leaf_deltas)), [-1])
    expanded_keys = HierarchicalKey(expanded_lc, [4, 5])
    unique_keys   = Unique(expanded_keys)
    hash_map      = HashMapBuild(unique_keys)
    output_masks  = DilateLeafMasks(leaf_masks, leaf_coords, offsets, hash_map)
    occupied      = HashMapOccupied(hash_map)
    active_keys   = Gather(hash_map, occupied)
    active_masks  = Gather(output_masks, occupied)
    active_coords = HierarchicalKeyDecode(active_keys, [4, 5])
    result        = MaskToCoords(active_masks, active_coords)
    voxel_keys    = HierarchicalKey(result, [3, 4, 5])
    sorted_keys   = Sort(voxel_keys)
    sorted_result = HierarchicalKeyDecode(sorted_keys, [3, 4, 5])

**CPU path** (reference): ``_conv_grid_leafwise_cpu`` is a standalone
Python-level loop over kernel offsets for correctness verification.
"""

from __future__ import annotations

import torch

from .cig import CompressedCIG3
from .conv_grid import _as_vec3, _kernel_offsets_centered
from .dsl_pipeline import PipelineExecutable, compile_source
from .ops import (
    HASH_MAP_EMPTY_KEY,
    Value,
    hash_map_build,
    hash_map_scatter_reduce,
    mask_to_coords,
    shift_leaf_masks,
    hierarchical_key,
    hierarchical_key_decode,
)
from .types import Dynamic, ScalarType, Shape, Static, Type

# Leaf-level bit widths: [4, 5] covers lower(16^3) + upper(32^3).
# The leaf level (3 bits) is consumed by the leaf mask itself.
_LEAF_BIT_WIDTHS = [4, 5]
_VOXEL_BIT_WIDTHS = [3, 4, 5]

# ---------------------------------------------------------------------------
# Unified DSL program: full leafwise conv_grid pipeline.
#
# 13 bindings, 4 inputs, zero imperative torch in the hot path.
# The pipeline planner segments this into alternating cutile/collective
# phases automatically.  DilateLeafMasks dispatches to the fused CUDA
# kernel on GPU via hook; all other collectives use torch ops.
# ---------------------------------------------------------------------------

_LEAFWISE_SOURCE = """\
expanded_lc = reshape(fuse(EachLeft(EachRight(EachBoth(Add)), Input("leaf_coords"), Input("leaf_deltas"))), Const([-1]))
expanded_keys = HierarchicalKey(expanded_lc, Const([4, 5]))
unique_keys = Unique(expanded_keys)
hash_map = HashMapBuild(unique_keys)
output_masks = DilateLeafMasks(Input("leaf_masks"), Input("leaf_coords"), Input("offsets"), hash_map)
occupied = HashMapOccupied(hash_map)
active_keys = Gather(hash_map, occupied)
active_masks = Gather(output_masks, occupied)
active_coords = HierarchicalKeyDecode(active_keys, Const([4, 5]))
result = MaskToCoords(active_masks, active_coords)
voxel_keys = HierarchicalKey(result, Const([3, 4, 5]))
sorted_keys = Sort(voxel_keys)
sorted_result = HierarchicalKeyDecode(sorted_keys, Const([3, 4, 5]))
sorted_result
"""

_LEAFWISE_PIPELINE: PipelineExecutable | None = None


def _get_leafwise_pipeline() -> PipelineExecutable:
    """Return (and cache) the compiled leafwise conv_grid pipeline."""
    global _LEAFWISE_PIPELINE
    if _LEAFWISE_PIPELINE is None:
        _LEAFWISE_PIPELINE = compile_source(_LEAFWISE_SOURCE)
    return _LEAFWISE_PIPELINE


def _leaf_level_offsets(kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Compute the set of leaf-level offset deltas for a voxel kernel.

    For a voxel kernel of size k, a single leaf can contribute to
    target leaves at delta -1, 0, or +1 per axis (for k <= 8).
    Returns (K_leaf, 3) i32.
    """
    ranges = []
    for k in kernel_size:
        half = (k - 1) // 2
        lo = -(half + 7) // 8
        hi = (half + 7) // 8
        ranges.append(torch.arange(lo, hi + 1, dtype=torch.int32))
    gx, gy, gz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


def conv_grid_leafwise(
    cig: CompressedCIG3,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    device: str | None = None,
) -> torch.Tensor:
    """Compute unique output coordinates for sparse convolution via bitmask dilation.

    Uses the CIG's leaf masks directly -- no N*K expansion.

    Args:
        cig: CompressedCIG3 with leaf_masks and leaf_coords.
        kernel_size: scalar or (kx, ky, kz) kernel dimensions.
        stride: scalar or (sx, sy, sz) stride.  Only stride=1 supported initially.
        device: "cpu", "cuda", or None (use CIG's device).

    Returns:
        (M, 3) i32 torch tensor of unique output coordinates, sorted by
        hierarchical key (CIG-compatible order).
    """
    ks = _as_vec3(kernel_size)
    st = _as_vec3(stride)

    if st != (1, 1, 1):
        raise NotImplementedError("Leafwise conv_grid currently supports stride=1 only")

    target_device = torch.device(device) if device is not None else cig.leaf_masks.device
    use_gpu = target_device.type == "cuda"

    if cig.leaf_masks.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=target_device)

    offsets = _kernel_offsets_centered(ks)  # (K_voxel, 3) i32

    return _conv_grid_leafwise_pipeline(cig, ks, offsets, target_device)


# ---------------------------------------------------------------------------
# DSL-pipeline-driven implementation
#
# The full algorithm is a single DSL program (see _LEAFWISE_SOURCE above).
# The pipeline planner segments it automatically into alternating
# cutile/collective phases.  The Python caller just builds inputs and
# calls pipeline.run().
# ---------------------------------------------------------------------------

_COORD_ELEM = Type(Shape(Static(3)), ScalarType.I32)


def _conv_grid_leafwise_pipeline(
    cig: CompressedCIG3,
    ks: tuple[int, int, int],
    offsets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    dev = str(device)

    leaf_masks = cig.leaf_masks.to(device)
    leaf_coords = cig.leaf_coords.to(device)
    offsets_dev = offsets.to(device=device, dtype=torch.int32)
    leaf_deltas = _leaf_level_offsets(ks).to(device)

    pipeline = _get_leafwise_pipeline()
    inputs = {
        "leaf_masks": Value(Type(Shape(Dynamic()), Type(Shape(Static(8)), ScalarType.I64)), leaf_masks),
        "leaf_coords": Value(Type(Shape(Dynamic()), _COORD_ELEM), leaf_coords),
        "offsets": Value(Type(Shape(Dynamic()), _COORD_ELEM), offsets_dev),
        "leaf_deltas": Value(Type(Shape(Static(leaf_deltas.shape[0])), _COORD_ELEM), leaf_deltas),
    }
    result = pipeline.run(inputs, device=dev)
    output = result.output.data

    if isinstance(output, torch.Tensor):
        return output.to(device=device, dtype=torch.int32)
    return torch.empty((0, 3), dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# CPU path: reference implementation (Python loop, correct but slow)
# ---------------------------------------------------------------------------


def _conv_grid_leafwise_cpu(
    cig: CompressedCIG3,
    ks: tuple[int, int, int],
    offsets: torch.Tensor,
) -> torch.Tensor:
    leaf_masks = cig.leaf_masks.cpu()
    leaf_coords = cig.leaf_coords.cpu()

    all_keys = []
    all_masks = []

    for i in range(offsets.shape[0]):
        ox, oy, oz = int(offsets[i, 0]), int(offsets[i, 1]), int(offsets[i, 2])
        pairs = shift_leaf_masks(leaf_masks, ox, oy, oz)
        for shifted_mask, (dx, dy, dz) in pairs:
            delta = torch.tensor([[dx, dy, dz]], dtype=torch.int32)
            target_lc = leaf_coords + delta
            keys = hierarchical_key(target_lc, _LEAF_BIT_WIDTHS)
            all_keys.append(keys)
            all_masks.append(shifted_mask)

    cat_keys = torch.cat(all_keys, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    unique_keys = torch.unique(cat_keys)

    key_arr = hash_map_build(unique_keys)
    storage_size = key_arr.shape[0]
    output_masks = torch.zeros(storage_size, 8, dtype=torch.int64)
    for w in range(8):
        output_masks[:, w] = hash_map_scatter_reduce(
            key_arr, cat_keys, cat_masks[:, w], reduce_fn="or"
        )

    active_leaf_mask = key_arr != HASH_MAP_EMPTY_KEY
    active_masks = output_masks[active_leaf_mask]
    active_keys = key_arr[active_leaf_mask]
    output_leaf_coords = hierarchical_key_decode(active_keys, _LEAF_BIT_WIDTHS)
    result = mask_to_coords(active_masks, output_leaf_coords)

    if result.shape[0] > 0:
        voxel_keys = hierarchical_key(result, _VOXEL_BIT_WIDTHS)
        order = torch.argsort(voxel_keys, stable=True)
        result = result[order]

    return result
