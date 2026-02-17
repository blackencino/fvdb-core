# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
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
// long long = int64, unsigned long long = uint64, both guaranteed 64-bit in CUDA.
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
