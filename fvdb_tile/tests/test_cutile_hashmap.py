# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
GPU tests for hash map CUDA kernels and cuTile bitwise ops.

All tests skip if CUDA is not available.  Tests verify that GPU
implementations match the CPU reference in ops.py.

Groups:
  1. CUDA kernel infrastructure: compile + launch a trivial kernel
  2. GPU hash map build: atomicCAS-based insertion
  3. GPU hash map lookup: probe-based lookup
  4. GPU scatter-reduce: atomicOr/Add
  5. Pipeline GPU: end-to-end with device="cuda"
  6. Cross-validation: GPU vs CPU reference
  7. Bitwise ops via cuTile pipeline
"""

import torch

HAS_CUDA = torch.cuda.is_available()


def _skip_no_cuda():
    if not HAS_CUDA:
        print("  SKIP: no CUDA device")
        return True
    return False


# =========================================================================
# 1. CUDA kernel infrastructure
# =========================================================================


def test_cuda_launch_trivial():
    """Compile and launch a trivial CUDA kernel that doubles an array."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.cuda_launch import compile_and_get_function, launch_kernel

    source = r"""
    extern "C" __global__ void double_arr(
        const long long* __restrict__ input,
        long long* __restrict__ output,
        size_t n
    ) {
        for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
             i < n; i += blockDim.x * gridDim.x) {
            output[i] = input[i] * 2;
        }
    }
    """
    func = compile_and_get_function(source, "double_arr")

    n = 256
    inp = torch.arange(n, dtype=torch.int64, device="cuda")
    out = torch.zeros(n, dtype=torch.int64, device="cuda")

    launch_kernel(func, grid=(1,), block=(256,), args=[inp, out, n])
    torch.cuda.synchronize()

    expected = inp * 2
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


# =========================================================================
# 2. GPU hash map build
# =========================================================================


def test_gpu_build_basic():
    """GPU build produces valid key array and slots."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build
    from fvdb_tile.prototype.ops import HASH_MAP_EMPTY_KEY, HASH_MAP_NO_SLOT

    keys = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device="cuda")
    key_arr, slots = gpu_hash_map_build(keys)

    assert key_arr.is_cuda
    assert slots.is_cuda
    assert key_arr.shape[0] > 0

    slots_cpu = slots.cpu()
    key_arr_cpu = key_arr.cpu()

    for i in range(len(keys)):
        s = int(slots_cpu[i].item())
        assert s != HASH_MAP_NO_SLOT, f"Key {keys[i]} not inserted"
        assert int(key_arr_cpu[s].item()) == int(keys[i].item())


def test_gpu_build_scale():
    """GPU build works at larger scale."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT

    keys = torch.arange(10000, dtype=torch.int64, device="cuda")
    key_arr, slots = gpu_hash_map_build(keys)

    slots_cpu = slots.cpu()
    miss_count = (slots_cpu == HASH_MAP_NO_SLOT).sum().item()
    assert miss_count == 0, f"{miss_count} keys failed to insert"

    unique_slots = torch.unique(slots_cpu)
    assert len(unique_slots) == len(keys), "Duplicate slots detected"


def test_gpu_build_duplicates():
    """GPU build handles duplicate keys correctly."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build, gpu_hash_map_lookup
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT

    keys = torch.tensor([10, 20, 10, 30, 20], dtype=torch.int64, device="cuda")
    key_arr, _slots = gpu_hash_map_build(keys)

    queries = torch.tensor([10, 20, 30], dtype=torch.int64, device="cuda")
    result = gpu_hash_map_lookup(key_arr, queries)
    result_cpu = result.cpu()

    for i in range(len(queries)):
        assert int(result_cpu[i].item()) != HASH_MAP_NO_SLOT


# =========================================================================
# 3. GPU hash map lookup
# =========================================================================


def test_gpu_lookup_all_hit():
    """All inserted keys are found."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build, gpu_hash_map_lookup
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT

    keys = torch.arange(1000, dtype=torch.int64, device="cuda")
    key_arr, _slots = gpu_hash_map_build(keys)
    result = gpu_hash_map_lookup(key_arr, keys)

    result_cpu = result.cpu()
    miss_count = (result_cpu == HASH_MAP_NO_SLOT).sum().item()
    assert miss_count == 0, f"{miss_count} misses out of {len(keys)}"


def test_gpu_lookup_all_miss():
    """Absent keys return NO_SLOT."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build, gpu_hash_map_lookup
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT

    keys = torch.arange(100, dtype=torch.int64, device="cuda")
    key_arr, _ = gpu_hash_map_build(keys)

    absent = torch.arange(1000, 1100, dtype=torch.int64, device="cuda")
    result = gpu_hash_map_lookup(key_arr, absent)

    result_cpu = result.cpu()
    hit_count = (result_cpu != HASH_MAP_NO_SLOT).sum().item()
    assert hit_count == 0, f"{hit_count} false hits"


# =========================================================================
# 4. GPU scatter-reduce
# =========================================================================


def test_gpu_scatter_or():
    """Scatter-reduce with OR matches CPU reference."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import (
        gpu_hash_map_build,
        gpu_hash_map_lookup,
        gpu_hash_map_scatter_reduce,
    )
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT

    keys = torch.tensor([1, 2, 1, 2, 3], dtype=torch.int64, device="cuda")
    values = torch.tensor([0b0001, 0b0010, 0b0100, 0b1000, 0b1111], dtype=torch.int64, device="cuda")

    key_arr, _ = gpu_hash_map_build(keys)
    result = gpu_hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="or")

    unique_keys = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    slots = gpu_hash_map_lookup(key_arr, unique_keys)
    slots_cpu = slots.cpu()
    result_cpu = result.cpu()

    assert int(result_cpu[slots_cpu[0]].item()) == 0b0001 | 0b0100
    assert int(result_cpu[slots_cpu[1]].item()) == 0b0010 | 0b1000
    assert int(result_cpu[slots_cpu[2]].item()) == 0b1111


def test_gpu_scatter_add():
    """Scatter-reduce with Add matches expected sums."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import (
        gpu_hash_map_build,
        gpu_hash_map_lookup,
        gpu_hash_map_scatter_reduce,
    )

    keys = torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64, device="cuda")
    values = torch.tensor([10, 20, 3, 4, 5], dtype=torch.int64, device="cuda")

    key_arr, _ = gpu_hash_map_build(keys)
    result = gpu_hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="add")

    unique_keys = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    slots = gpu_hash_map_lookup(key_arr, unique_keys)
    slots_cpu = slots.cpu()
    result_cpu = result.cpu()

    assert int(result_cpu[slots_cpu[0]].item()) == 30
    assert int(result_cpu[slots_cpu[1]].item()) == 12


# =========================================================================
# 5. Pipeline GPU test
# =========================================================================


def test_pipeline_gpu_hashmap():
    """Pipeline with HashMapBuild + HashMapLookup runs on GPU."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.dsl_pipeline import compile_source
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT, Value
    from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Type

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("keys"))\n'
        'result'
    )
    pipeline = compile_source(source)

    keys = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    inputs = {
        "keys": Value(Type(Shape(Dynamic()), ScalarType.I64), keys),
    }
    result = pipeline.run(inputs, device="cuda")
    slots = result.output.data

    for i in range(len(keys)):
        assert int(slots[i].item()) != HASH_MAP_NO_SLOT, f"Key {keys[i]} not found on GPU"


# =========================================================================
# 6. Cross-validation: GPU vs CPU
# =========================================================================


def test_gpu_vs_cpu_cross_validation():
    """GPU build+lookup matches CPU build+lookup for 10K random keys."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.hashmap_cuda import gpu_hash_map_build, gpu_hash_map_lookup
    from fvdb_tile.prototype.ops import HASH_MAP_NO_SLOT, hash_map_build, hash_map_lookup

    gen = torch.Generator().manual_seed(42)
    keys = torch.randint(0, 100000, size=(10000,), generator=gen, dtype=torch.int64)
    keys = torch.unique(keys)

    cpu_key_arr = hash_map_build(keys)
    cpu_slots = hash_map_lookup(cpu_key_arr, keys)
    cpu_hits = (cpu_slots != HASH_MAP_NO_SLOT).sum().item()

    keys_gpu = keys.cuda()
    gpu_key_arr, _ = gpu_hash_map_build(keys_gpu)
    gpu_slots = gpu_hash_map_lookup(gpu_key_arr, keys_gpu)
    torch.cuda.synchronize()
    gpu_hits = (gpu_slots.cpu() != HASH_MAP_NO_SLOT).sum().item()

    assert cpu_hits == len(keys), f"CPU: {cpu_hits} hits out of {len(keys)}"
    assert gpu_hits == len(keys), f"GPU: {gpu_hits} hits out of {len(keys)}"

    absent = torch.arange(200000, 201000, dtype=torch.int64)
    cpu_absent_slots = hash_map_lookup(cpu_key_arr, absent)
    gpu_absent_slots = gpu_hash_map_lookup(gpu_key_arr, absent.cuda())
    torch.cuda.synchronize()

    cpu_absent_hits = (cpu_absent_slots != HASH_MAP_NO_SLOT).sum().item()
    gpu_absent_hits = (gpu_absent_slots.cpu() != HASH_MAP_NO_SLOT).sum().item()
    assert cpu_absent_hits == 0, f"CPU false hits: {cpu_absent_hits}"
    assert gpu_absent_hits == 0, f"GPU false hits: {gpu_absent_hits}"


# =========================================================================
# 7. Bitwise ops via cuTile pipeline
# =========================================================================


def test_pipeline_gpu_bitwise():
    """Pipeline with bitwise ops runs on GPU and matches CPU reference."""
    if _skip_no_cuda():
        return

    from fvdb_tile.prototype.dsl_pipeline import compile_source
    from fvdb_tile.prototype.ops import Value
    from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Type

    source = (
        'shifted = ShiftLeft(Input("x"), Const(2))\n'
        'result = ShiftRight(shifted, Const(1))\n'
        'result'
    )
    pipeline = compile_source(source)

    x = torch.tensor([1, 2, 4, 8, 16], dtype=torch.int64)
    inputs = {"x": Value(Type(Shape(Dynamic()), ScalarType.I64), x)}

    result_cpu = pipeline.run(inputs, device=None)
    result_gpu = pipeline.run(inputs, device="cuda")

    # cuTile segments output int32; CPU eval preserves int64. Compare as int64.
    gpu_data = result_gpu.output.data.to(torch.int64)
    cpu_data = result_cpu.output.data.to(torch.int64)
    torch.testing.assert_close(gpu_data, cpu_data, atol=0, rtol=0)


# =========================================================================

if __name__ == "__main__":
    print("=== CUDA kernel infrastructure ===")
    test_cuda_launch_trivial()
    print("  test_cuda_launch_trivial: PASS")

    print("\n=== GPU hash map build ===")
    test_gpu_build_basic()
    print("  test_gpu_build_basic: PASS")
    test_gpu_build_scale()
    print("  test_gpu_build_scale: PASS")
    test_gpu_build_duplicates()
    print("  test_gpu_build_duplicates: PASS")

    print("\n=== GPU hash map lookup ===")
    test_gpu_lookup_all_hit()
    print("  test_gpu_lookup_all_hit: PASS")
    test_gpu_lookup_all_miss()
    print("  test_gpu_lookup_all_miss: PASS")

    print("\n=== GPU scatter-reduce ===")
    test_gpu_scatter_or()
    print("  test_gpu_scatter_or: PASS")
    test_gpu_scatter_add()
    print("  test_gpu_scatter_add: PASS")

    print("\n=== Pipeline GPU ===")
    test_pipeline_gpu_hashmap()
    print("  test_pipeline_gpu_hashmap: PASS")

    print("\n=== Cross-validation ===")
    test_gpu_vs_cpu_cross_validation()
    print("  test_gpu_vs_cpu_cross_validation: PASS")

    print("\n=== Bitwise cuTile ===")
    test_pipeline_gpu_bitwise()
    print("  test_pipeline_gpu_bitwise: PASS")

    print("\nAll GPU hashmap tests passed.")
