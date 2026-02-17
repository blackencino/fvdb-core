# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# DSL status: out-of-DSL (hand-written CUDA kernels; future fusion/codegen target)
"""
GPU hash map operations via CUDA kernels.

Provides GPU-native hash map build, lookup, and scatter-reduce using
real CUDA kernels (atomicCAS for build, probe loop for lookup, atomicOr/Add
for scatter-reduce).  Kernels are JIT-compiled via NVRTC and launched on
torch's CUDA stream so they participate in the same synchronization chain
as cuTile segments.

Ported from the cuda_hashmap.h C++ header, using MurmurHash3 64-bit
finalizer instead of pcg4d.

The hash function is an implementation detail -- not exposed in the DSL.
Deterministic within a build+lookup pair; no ordering guarantees.
"""

from __future__ import annotations

import torch

from .cuda_launch import compile_and_get_function, launch_kernel
from .ops import HASH_MAP_EMPTY_KEY, HASH_MAP_NO_SLOT, hash_map_compute_storage_size

# ---------------------------------------------------------------------------
# CUDA C++ kernel source
# ---------------------------------------------------------------------------

_HASHMAP_CUDA_SOURCE = r"""
// No standard library headers in NVRTC -- use built-in types only.
typedef int                  int32_t;
typedef long long            int64_t;
typedef unsigned long long   uint64_t;
typedef unsigned long        size_t;

static __device__ const int64_t EMPTY_KEY = -1LL;
static __device__ const int64_t NO_SLOT   = -1LL;

// ---------------------------------------------------------------------------
// MurmurHash3 64-bit finalizer
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t murmurhash3_fmix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return h;
}

// ---------------------------------------------------------------------------
// Build: insert keys into hash map via atomicCAS
// ---------------------------------------------------------------------------
extern "C" __global__ void hashmap_build_kernel(
    const int64_t* __restrict__ d_input_keys,
    int64_t*       __restrict__ d_hash_map_keys,
    int64_t*       __restrict__ d_slots,
    size_t input_count,
    size_t map_storage_size
) {
    const size_t mask = map_storage_size - 1;
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < input_count;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        int64_t key = d_input_keys[idx];
        uint64_t ukey = static_cast<uint64_t>(key);
        size_t slot = murmurhash3_fmix64(ukey) & mask;

        for (size_t probe = 0; probe < map_storage_size; ++probe) {
            // atomicCAS on the key slot
            uint64_t old = atomicCAS(
                reinterpret_cast<unsigned long long*>(d_hash_map_keys + slot),
                static_cast<unsigned long long>(EMPTY_KEY),
                static_cast<unsigned long long>(key));
            int64_t old_signed = static_cast<int64_t>(old);
            if (old_signed == EMPTY_KEY || old_signed == key) {
                d_slots[idx] = static_cast<int64_t>(slot);
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
}

// ---------------------------------------------------------------------------
// Lookup: read-only probe
// ---------------------------------------------------------------------------
extern "C" __global__ void hashmap_lookup_kernel(
    const int64_t* __restrict__ d_hash_map_keys,
    const int64_t* __restrict__ d_query_keys,
    int64_t*       __restrict__ d_result_slots,
    size_t query_count,
    size_t map_storage_size
) {
    const size_t mask = map_storage_size - 1;
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < query_count;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        int64_t key = d_query_keys[idx];
        uint64_t ukey = static_cast<uint64_t>(key);
        size_t slot = murmurhash3_fmix64(ukey) & mask;
        int64_t result = NO_SLOT;

        for (size_t probe = 0; probe < map_storage_size; ++probe) {
            int64_t k = d_hash_map_keys[slot];
            if (k == key) {
                result = static_cast<int64_t>(slot);
                break;
            }
            if (k == EMPTY_KEY) {
                break;
            }
            slot = (slot + 1) & mask;
        }
        d_result_slots[idx] = result;
    }
}

// ---------------------------------------------------------------------------
// Scatter-reduce: lookup slot then atomic-reduce into value array
// ---------------------------------------------------------------------------
extern "C" __global__ void hashmap_scatter_or_kernel(
    const int64_t* __restrict__ d_hash_map_keys,
    const int64_t* __restrict__ d_input_keys,
    const int64_t* __restrict__ d_input_values,
    int64_t*       __restrict__ d_value_arr,
    size_t input_count,
    size_t map_storage_size
) {
    const size_t mask = map_storage_size - 1;
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < input_count;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        int64_t key = d_input_keys[idx];
        uint64_t ukey = static_cast<uint64_t>(key);
        size_t slot = murmurhash3_fmix64(ukey) & mask;

        // Probe to find the slot
        for (size_t probe = 0; probe < map_storage_size; ++probe) {
            int64_t k = d_hash_map_keys[slot];
            if (k == key) {
                atomicOr(
                    reinterpret_cast<unsigned long long*>(d_value_arr + slot),
                    static_cast<unsigned long long>(d_input_values[idx]));
                break;
            }
            if (k == EMPTY_KEY) {
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
}

extern "C" __global__ void hashmap_scatter_add_kernel(
    const int64_t* __restrict__ d_hash_map_keys,
    const int64_t* __restrict__ d_input_keys,
    const int64_t* __restrict__ d_input_values,
    int64_t*       __restrict__ d_value_arr,
    size_t input_count,
    size_t map_storage_size
) {
    const size_t mask = map_storage_size - 1;
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < input_count;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        int64_t key = d_input_keys[idx];
        uint64_t ukey = static_cast<uint64_t>(key);
        size_t slot = murmurhash3_fmix64(ukey) & mask;

        for (size_t probe = 0; probe < map_storage_size; ++probe) {
            int64_t k = d_hash_map_keys[slot];
            if (k == key) {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(d_value_arr + slot),
                    static_cast<unsigned long long>(d_input_values[idx]));
                break;
            }
            if (k == EMPTY_KEY) {
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
}


// ---------------------------------------------------------------------------
// Inline hash map probe (read-only) -- returns slot or -1
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t hashmap_probe(
    const int64_t* __restrict__ keys,
    int64_t query,
    size_t storage_size
) {
    if (storage_size == 0) return NO_SLOT;
    const size_t mask = storage_size - 1;
    uint64_t uq = static_cast<uint64_t>(query);
    size_t slot = murmurhash3_fmix64(uq) & mask;
    for (size_t i = 0; i < storage_size; ++i) {
        int64_t k = keys[slot];
        if (k == query) return static_cast<int64_t>(slot);
        if (k == EMPTY_KEY) return NO_SLOT;
        slot = (slot + 1) & mask;
    }
    return NO_SLOT;
}

// ---------------------------------------------------------------------------
// Inline hierarchical key for leaf coordinates (bit_widths [4, 5])
// Leaf coord -> key encoding matching ops.hierarchical_key with bw=[4,5]
//
// Parameters (from _hkey_params([4,5])):
//   tree_key_bits = 3*4 + 3*5 = 27
//   remaining = 63 - 27 = 36
//   root_bits_per_axis = 12
//   total_bits_per_axis = 4 + 5 + 12 = 21
//   offset = 1 << 20 = 1048576
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t leaf_hierarchical_key(
    int32_t lx, int32_t ly, int32_t lz
) {
    const int64_t offset = 1048576LL;  // 1 << 20
    int64_t cx = static_cast<int64_t>(lx) + offset;
    int64_t cy = static_cast<int64_t>(ly) + offset;
    int64_t cz = static_cast<int64_t>(lz) + offset;

    // level 0: bw=4, dim=16, n_bits=12
    int64_t l0x = cx & 15, l0y = cy & 15, l0z = cz & 15;
    int64_t lin0 = l0x * 256 + l0y * 16 + l0z;  // x*16*16 + y*16 + z

    // level 1: bw=5, dim=32, n_bits=15
    int64_t l1x = (cx >> 4) & 31, l1y = (cy >> 4) & 31, l1z = (cz >> 4) & 31;
    int64_t lin1 = l1x * 1024 + l1y * 32 + l1z;  // x*32*32 + y*32 + z

    // root: 12 bits per axis, above level 1 (coord >> 9)
    int64_t rx = cx >> 9, ry = cy >> 9, rz = cz >> 9;
    int64_t root_lin = rx * (1LL << 24) + ry * (1LL << 12) + rz;

    // key_shift after level 0: 12, after level 1: 12+15=27
    return lin0 | (lin1 << 12) | (root_lin << 27);
}

// ---------------------------------------------------------------------------
// Fused conv_grid dilation kernel
//
// For each (leaf, voxel_offset) pair: shift the leaf mask, decompose
// boundary crossings, compute the target leaf key, probe the hash map,
// and atomicOr the shifted mask words into the output.
//
// One launch for all L*K pairs.  No Python loops.
//
// NOTE: This is a performance-proving fused kernel.  The algorithm is
// correct and matches the DSL-level description, but it is NOT yet
// expressed through the DSL/AST pipeline.  Future work: express the
// same computation as a DSL program with idiom-recognition lowering
// that emits this fused kernel.  See conv_grid_leafwise.py docstring.
// ---------------------------------------------------------------------------
extern "C" __global__ void conv_grid_dilate_kernel(
    const int64_t* __restrict__ d_leaf_masks,    // (L, 8) row-major
    const int32_t* __restrict__ d_leaf_coords,   // (L, 3) row-major, leaf space
    const int32_t* __restrict__ d_offsets,        // (K, 3) row-major, voxel offsets
    const int64_t* __restrict__ d_hash_map_keys, // (S,) hash map key array
    int64_t*       __restrict__ d_output_masks,  // (S, 8) row-major, output
    size_t n_leaves,
    size_t n_offsets,
    size_t map_storage_size
) {
    const size_t total = n_leaves * n_offsets;
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < total;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t leaf_idx = idx / n_offsets;
        const size_t off_idx  = idx % n_offsets;

        const int32_t ox = d_offsets[off_idx * 3 + 0];
        const int32_t oy = d_offsets[off_idx * 3 + 1];
        const int32_t oz = d_offsets[off_idx * 3 + 2];

        const int32_t base_lx = d_leaf_coords[leaf_idx * 3 + 0];
        const int32_t base_ly = d_leaf_coords[leaf_idx * 3 + 1];
        const int32_t base_lz = d_leaf_coords[leaf_idx * 3 + 2];

        // Process each source x-plane (word) independently.
        // For each word, determine the target leaf delta and shifted bits.
        for (int src_x = 0; src_x < 8; ++src_x) {
            int64_t word = d_leaf_masks[leaf_idx * 8 + src_x];
            if (word == 0) continue;

            // X-axis boundary: where does src_x + ox land?
            int dst_x = src_x + ox;
            int dx = 0;
            if (dst_x >= 8) { dx = 1; dst_x -= 8; }
            else if (dst_x < 0) { dx = -1; dst_x += 8; }

            // Y and Z axes: each bit position is y*8 + z within the word.
            // We need to handle boundary crossings for y and z independently.
            // For each (dy, dz) sub-case, extract the valid source bits,
            // shift them to destination positions, and atomicOr into output.

            // Determine y sub-cases: source y positions that stay vs cross
            // oy can shift y into [0..7] (dy=0) or out of range (dy=+1 or dy=-1)
            struct YZCase { int dy; int y_lo; int y_hi; int dz; int z_lo; int z_hi; };
            YZCase cases[4];
            int n_cases = 0;

            // Y axis cases
            int y_stay_lo = max(0, -oy);
            int y_stay_hi = min(7, 7 - oy);
            int y_cross_lo = -1, y_cross_hi = -1, y_cross_dy = 0;
            if (oy > 0 && (8 - oy) <= 7) {
                y_cross_lo = 8 - oy; y_cross_hi = 7; y_cross_dy = 1;
            } else if (oy < 0 && (-oy - 1) >= 0) {
                y_cross_lo = 0; y_cross_hi = -oy - 1; y_cross_dy = -1;
            }

            // Z axis cases
            int z_stay_lo = max(0, -oz);
            int z_stay_hi = min(7, 7 - oz);
            int z_cross_lo = -1, z_cross_hi = -1, z_cross_dz = 0;
            if (oz > 0 && (8 - oz) <= 7) {
                z_cross_lo = 8 - oz; z_cross_hi = 7; z_cross_dz = 1;
            } else if (oz < 0 && (-oz - 1) >= 0) {
                z_cross_lo = 0; z_cross_hi = -oz - 1; z_cross_dz = -1;
            }

            // Enumerate up to 4 (dy, dz) sub-cases
            if (y_stay_lo <= y_stay_hi && z_stay_lo <= z_stay_hi)
                cases[n_cases++] = {0, y_stay_lo, y_stay_hi, 0, z_stay_lo, z_stay_hi};
            if (y_stay_lo <= y_stay_hi && z_cross_lo >= 0)
                cases[n_cases++] = {0, y_stay_lo, y_stay_hi, z_cross_dz, z_cross_lo, z_cross_hi};
            if (y_cross_lo >= 0 && z_stay_lo <= z_stay_hi)
                cases[n_cases++] = {y_cross_dy, y_cross_lo, y_cross_hi, 0, z_stay_lo, z_stay_hi};
            if (y_cross_lo >= 0 && z_cross_lo >= 0)
                cases[n_cases++] = {y_cross_dy, y_cross_lo, y_cross_hi, z_cross_dz, z_cross_lo, z_cross_hi};

            for (int c = 0; c < n_cases; ++c) {
                int dy = cases[c].dy;
                int dz = cases[c].dz;

                // Build source bit mask for valid (y, z) positions
                uint64_t src_mask = 0;
                for (int y = cases[c].y_lo; y <= cases[c].y_hi; ++y)
                    for (int z = cases[c].z_lo; z <= cases[c].z_hi; ++z)
                        src_mask |= 1ULL << (y * 8 + z);

                uint64_t bits = static_cast<uint64_t>(word) & src_mask;
                if (bits == 0) continue;

                // Compute bit shift for y, z destination
                int local_oy = oy - dy * 8;
                int local_oz = oz - dz * 8;
                int bit_shift = local_oy * 8 + local_oz;

                uint64_t shifted;
                if (bit_shift > 0)
                    shifted = bits << bit_shift;
                else if (bit_shift < 0)
                    shifted = bits >> (-bit_shift);
                else
                    shifted = bits;

                if (shifted == 0) continue;

                // Target leaf coord
                int32_t tlx = base_lx + dx;
                int32_t tly = base_ly + dy;
                int32_t tlz = base_lz + dz;

                // Compute hierarchical key and probe hash map
                int64_t target_key = leaf_hierarchical_key(tlx, tly, tlz);
                int64_t slot = hashmap_probe(d_hash_map_keys, target_key,
                                             map_storage_size);
                if (slot < 0) continue;

                // atomicOr the shifted word into the output mask
                atomicOr(
                    reinterpret_cast<unsigned long long*>(
                        d_output_masks + slot * 8 + dst_x),
                    shifted);
            }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Compiled kernel cache (per-kernel-name)
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[str, object] = {}


def _get_kernel(name: str):
    """Get a compiled kernel function by name, compiling on first use."""
    if name not in _KERNEL_CACHE:
        _KERNEL_CACHE[name] = compile_and_get_function(_HASHMAP_CUDA_SOURCE, name)
    return _KERNEL_CACHE[name]


# ---------------------------------------------------------------------------
# Grid/block sizing
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 256
_SM_MULTIPLIER = 32  # blocks per SM for grid-stride coverage


def _grid_size(n: int) -> int:
    """Compute grid size for grid-stride kernel launch."""
    return min(max(1, (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE), _SM_MULTIPLIER * 128)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def gpu_hash_map_build(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build hash map on GPU.  Returns (key_arr, slots).

    Args:
        keys: (N,) i64 CUDA tensor of keys.

    Returns:
        key_arr: (storage_size,) i64 CUDA tensor -- the hash map key array.
        slots:   (N,) i64 CUDA tensor -- slot index for each input key.
    """
    assert keys.is_cuda, "gpu_hash_map_build requires CUDA tensors"
    assert keys.dtype == torch.int64, "Keys must be int64"

    n = keys.shape[0]
    storage_size = hash_map_compute_storage_size(n)

    if storage_size == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=keys.device),
            torch.empty(0, dtype=torch.int64, device=keys.device),
        )

    key_arr = torch.full(
        (storage_size,), HASH_MAP_EMPTY_KEY, dtype=torch.int64, device=keys.device
    )
    slots = torch.full((n,), HASH_MAP_NO_SLOT, dtype=torch.int64, device=keys.device)

    func = _get_kernel("hashmap_build_kernel")
    grid = (_grid_size(n),)
    block = (_BLOCK_SIZE,)

    launch_kernel(func, grid, block, [keys, key_arr, slots, n, storage_size])

    return key_arr, slots


def gpu_hash_map_lookup(key_arr: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Lookup keys in GPU hash map.  Returns slot indices.

    Args:
        key_arr: (storage_size,) i64 CUDA tensor from gpu_hash_map_build.
        queries: (M,) i64 CUDA tensor of query keys.

    Returns:
        (M,) i64 CUDA tensor of slot indices.  NO_SLOT (-1) for misses.
    """
    assert key_arr.is_cuda and queries.is_cuda, "GPU lookup requires CUDA tensors"

    storage_size = key_arr.shape[0]
    m = queries.shape[0]

    if storage_size == 0 or m == 0:
        return torch.full((m,), HASH_MAP_NO_SLOT, dtype=torch.int64, device=queries.device)

    result = torch.full((m,), HASH_MAP_NO_SLOT, dtype=torch.int64, device=queries.device)

    func = _get_kernel("hashmap_lookup_kernel")
    grid = (_grid_size(m),)
    block = (_BLOCK_SIZE,)

    launch_kernel(func, grid, block, [key_arr, queries, result, m, storage_size])

    return result


def gpu_hash_map_scatter_reduce(
    key_arr: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    reduce_fn: str = "or",
) -> torch.Tensor:
    """Scatter values into hash map slots with atomic reduction on GPU.

    Args:
        key_arr: (storage_size,) i64 CUDA tensor from gpu_hash_map_build.
        keys: (N,) i64 CUDA tensor of keys.
        values: (N,) i64 CUDA tensor of values.
        reduce_fn: "or" or "add".

    Returns:
        (storage_size,) i64 CUDA tensor of reduced values at each slot.
    """
    assert key_arr.is_cuda and keys.is_cuda and values.is_cuda
    assert values.dtype == torch.int64, "Values must be int64 for atomic ops"

    storage_size = key_arr.shape[0]
    n = keys.shape[0]

    result = torch.zeros(storage_size, dtype=torch.int64, device=keys.device)

    if n == 0 or storage_size == 0:
        return result

    if reduce_fn == "or":
        kernel_name = "hashmap_scatter_or_kernel"
    elif reduce_fn == "add":
        kernel_name = "hashmap_scatter_add_kernel"
    else:
        raise ValueError(f"Unsupported reduce_fn for GPU: {reduce_fn!r}")

    func = _get_kernel(kernel_name)
    grid = (_grid_size(n),)
    block = (_BLOCK_SIZE,)

    launch_kernel(func, grid, block, [key_arr, keys, values, result, n, storage_size])

    return result


def gpu_conv_grid_dilate(  # OUT_OF_DSL: hand-fused kernel, future fusion target
    leaf_masks: torch.Tensor,
    leaf_coords: torch.Tensor,
    offsets: torch.Tensor,
    hash_map_keys: torch.Tensor,
    storage_size: int,
) -> torch.Tensor:
    """Fused dilation: shift masks + hash probe + atomicOr in one launch.

    For each (leaf, voxel_offset) pair, shifts the leaf mask, handles
    boundary crossings, computes the target leaf key, probes the hash
    map, and atomicOr's the shifted bits into the output.

    OUT_OF_DSL: This is a performance-proving fused kernel.  It is
    correct and matches the DSL-level algorithm description, but is NOT
    yet expressed through the DSL/AST pipeline.  Future work should
    express this as a DSL program with idiom-recognition lowering.

    Args:
        leaf_masks:    (L, 8) i64 CUDA -- input leaf occupancy masks.
        leaf_coords:   (L, 3) i32 CUDA -- leaf coords in leaf space.
        offsets:       (K, 3) i32 CUDA -- voxel-space kernel offsets.
        hash_map_keys: (S,) i64 CUDA   -- hash map key array for output leaves.
        storage_size:  int              -- hash map storage size.

    Returns:
        (S, 8) i64 CUDA -- accumulated output leaf masks.
    """
    assert leaf_masks.is_cuda and leaf_coords.is_cuda and offsets.is_cuda
    assert hash_map_keys.is_cuda

    L = leaf_masks.shape[0]
    K = offsets.shape[0]
    total = L * K

    output_masks = torch.zeros(storage_size, 8, dtype=torch.int64, device=leaf_masks.device)

    if total == 0 or storage_size == 0:
        return output_masks

    func = _get_kernel("conv_grid_dilate_kernel")
    grid = (_grid_size(total),)
    block = (_BLOCK_SIZE,)

    launch_kernel(func, grid, block, [
        leaf_masks, leaf_coords, offsets, hash_map_keys,
        output_masks, L, K, storage_size,
    ])

    return output_masks
