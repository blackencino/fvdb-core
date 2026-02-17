# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Tests for hash map primitives, bitwise ops, and dialect lowering.

Groups:
  1. Hash function: murmurhash3 basic properties
  2. Hash map build/lookup: round-trip correctness
  3. Bitwise ops: ShiftLeft, ShiftRight, BitXor
  4. DSL integration: parse + eval hash map programs
  5. Pipeline integration: barrier placement
  6. Dialect lowering: ScatterReduce rewrite
"""

import pytest
import torch

from fvdb_tile.prototype.ops import (
    HASH_MAP_EMPTY_KEY,
    HASH_MAP_NO_SLOT,
    Value,
    _murmurhash3_fmix64,
    hash_map_build,
    hash_map_compute_storage_size,
    hash_map_lookup,
    hash_map_scatter_reduce,
)
from fvdb_tile.prototype.types import Dynamic, ScalarType, Shape, Static, Type


# =========================================================================
# 1. Hash function tests
# =========================================================================


def test_murmurhash3_deterministic():
    """Same input always produces same output."""
    for seed in [0, 1, 42, 2**32, 2**63 - 1]:
        assert _murmurhash3_fmix64(seed) == _murmurhash3_fmix64(seed)


def test_murmurhash3_no_trivial_fixpoints():
    """Hash of small positive integers is not the identity (0 is a known fixpoint)."""
    for seed in range(1, 100):
        h = _murmurhash3_fmix64(seed)
        assert h != seed, f"Fixpoint at {seed}"


def test_murmurhash3_distinct_outputs():
    """First 1000 integers produce 1000 distinct hashes."""
    hashes = set()
    for i in range(1000):
        hashes.add(_murmurhash3_fmix64(i))
    assert len(hashes) == 1000


def test_murmurhash3_bit_spread():
    """Adjacent inputs produce well-separated outputs (basic avalanche check)."""
    h0 = _murmurhash3_fmix64(0)
    h1 = _murmurhash3_fmix64(1)
    diff_bits = bin(h0 ^ h1).count("1")
    assert diff_bits >= 16, f"Only {diff_bits} bits differ between hash(0) and hash(1)"


# =========================================================================
# 2. Hash map build/lookup tests
# =========================================================================


def test_storage_size_power_of_two():
    """Storage size is always a power of two (or zero)."""
    for n in [0, 1, 2, 7, 100, 1000]:
        size = hash_map_compute_storage_size(n)
        if n == 0:
            assert size == 0
        else:
            assert size > 0
            assert (size & (size - 1)) == 0, f"size={size} not power of two"
            assert size >= n * 4  # load factor


def test_build_lookup_roundtrip():
    """All inserted keys can be looked up; absent keys return NO_SLOT."""
    keys = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    key_arr = hash_map_build(keys)

    slots = hash_map_lookup(key_arr, keys)
    for i in range(len(keys)):
        s = int(slots[i].item())
        assert s != HASH_MAP_NO_SLOT, f"Key {keys[i]} not found"
        assert int(key_arr[s].item()) == int(keys[i].item())

    absent = torch.tensor([99, 100, 200], dtype=torch.int64)
    absent_slots = hash_map_lookup(key_arr, absent)
    for i in range(len(absent)):
        assert int(absent_slots[i].item()) == HASH_MAP_NO_SLOT


def test_build_lookup_unique_slots():
    """Each unique key gets a distinct slot."""
    keys = torch.arange(100, dtype=torch.int64)
    key_arr = hash_map_build(keys)
    slots = hash_map_lookup(key_arr, keys)

    slot_set = set()
    for i in range(len(keys)):
        s = int(slots[i].item())
        assert s != HASH_MAP_NO_SLOT
        assert s not in slot_set, f"Duplicate slot {s}"
        slot_set.add(s)


def test_build_with_duplicates():
    """Duplicate keys map to the same slot."""
    keys = torch.tensor([10, 20, 10, 30, 20], dtype=torch.int64)
    key_arr = hash_map_build(keys)

    queries = torch.tensor([10, 20, 30], dtype=torch.int64)
    slots = hash_map_lookup(key_arr, queries)
    for i in range(len(queries)):
        assert int(slots[i].item()) != HASH_MAP_NO_SLOT

    dup_queries = torch.tensor([10, 10, 20, 20], dtype=torch.int64)
    dup_slots = hash_map_lookup(key_arr, dup_queries)
    assert int(dup_slots[0].item()) == int(dup_slots[1].item())
    assert int(dup_slots[2].item()) == int(dup_slots[3].item())


def test_build_lookup_scale():
    """Round-trip correctness at larger scale."""
    for n in [1000, 10000]:
        keys = torch.arange(n, dtype=torch.int64)
        key_arr = hash_map_build(keys)
        slots = hash_map_lookup(key_arr, keys)
        miss_count = (slots == HASH_MAP_NO_SLOT).sum().item()
        assert miss_count == 0, f"n={n}: {miss_count} misses out of {n}"


def test_build_empty():
    """Empty key set produces empty map."""
    keys = torch.empty(0, dtype=torch.int64)
    key_arr = hash_map_build(keys)
    assert key_arr.shape[0] == 0


def test_scatter_reduce_or():
    """scatter_reduce with OR combines values correctly."""
    keys = torch.tensor([1, 2, 1, 2, 3], dtype=torch.int64)
    values = torch.tensor([0b0001, 0b0010, 0b0100, 0b1000, 0b1111], dtype=torch.int64)

    key_arr = hash_map_build(keys)
    result = hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="or")

    slots = hash_map_lookup(key_arr, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert int(result[slots[0]].item()) == 0b0001 | 0b0100  # key=1
    assert int(result[slots[1]].item()) == 0b0010 | 0b1000  # key=2
    assert int(result[slots[2]].item()) == 0b1111            # key=3


def test_scatter_reduce_add():
    """scatter_reduce with Add sums values correctly."""
    keys = torch.tensor([1, 1, 2, 2, 2], dtype=torch.int64)
    values = torch.tensor([10, 20, 3, 4, 5], dtype=torch.int64)

    key_arr = hash_map_build(keys)
    result = hash_map_scatter_reduce(key_arr, keys, values, reduce_fn="add")

    slots = hash_map_lookup(key_arr, torch.tensor([1, 2], dtype=torch.int64))
    assert int(result[slots[0]].item()) == 30  # 10 + 20
    assert int(result[slots[1]].item()) == 12  # 3 + 4 + 5


# =========================================================================
# 3. Bitwise op tests
# =========================================================================


def test_bitwise_shift_left():
    """ShiftLeft via DSL eval matches torch."""
    from fvdb_tile.prototype.dsl_eval import run

    source = 'result = ShiftLeft(Input("x"), Const(2))\nresult'
    x = torch.tensor([1, 2, 4, 8], dtype=torch.int64)
    inputs = {"x": Value(Type(Shape(Dynamic()), ScalarType.I64), x)}
    _, result = run(source, inputs)
    expected = x << 2
    torch.testing.assert_close(result.data, expected)


def test_bitwise_shift_right():
    """ShiftRight via DSL eval matches torch."""
    from fvdb_tile.prototype.dsl_eval import run

    source = 'result = ShiftRight(Input("x"), Const(3))\nresult'
    x = torch.tensor([8, 16, 64, 128], dtype=torch.int64)
    inputs = {"x": Value(Type(Shape(Dynamic()), ScalarType.I64), x)}
    _, result = run(source, inputs)
    expected = x >> 3
    torch.testing.assert_close(result.data, expected)


def test_bitwise_xor():
    """BitXor via DSL eval matches torch."""
    from fvdb_tile.prototype.dsl_eval import run

    source = 'result = BitXor(Input("a"), Input("b"))\nresult'
    a = torch.tensor([0xFF, 0xAA, 0x55], dtype=torch.int64)
    b = torch.tensor([0x0F, 0x55, 0xAA], dtype=torch.int64)
    inputs = {
        "a": Value(Type(Shape(Dynamic()), ScalarType.I64), a),
        "b": Value(Type(Shape(Dynamic()), ScalarType.I64), b),
    }
    _, result = run(source, inputs)
    expected = a ^ b
    torch.testing.assert_close(result.data, expected)


# =========================================================================
# 4. DSL integration tests
# =========================================================================


def test_dsl_hashmap_build_lookup():
    """HashMapBuild + HashMapLookup via DSL eval round-trips correctly."""
    from fvdb_tile.prototype.dsl_eval import run

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("queries"))\n'
        'result'
    )
    keys = torch.tensor([100, 200, 300], dtype=torch.int64)
    queries = torch.tensor([200, 100, 300], dtype=torch.int64)

    inputs = {
        "keys": Value(Type(Shape(Dynamic()), ScalarType.I64), keys),
        "queries": Value(Type(Shape(Dynamic()), ScalarType.I64), queries),
    }
    _, result = run(source, inputs)

    for i in range(len(queries)):
        assert int(result.data[i].item()) != HASH_MAP_NO_SLOT


def test_dsl_hashmap_miss():
    """HashMapLookup returns NO_SLOT for absent keys."""
    from fvdb_tile.prototype.dsl_eval import run

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("queries"))\n'
        'result'
    )
    keys = torch.tensor([100, 200], dtype=torch.int64)
    queries = torch.tensor([999], dtype=torch.int64)

    inputs = {
        "keys": Value(Type(Shape(Dynamic()), ScalarType.I64), keys),
        "queries": Value(Type(Shape(Dynamic()), ScalarType.I64), queries),
    }
    _, result = run(source, inputs)
    assert int(result.data[0].item()) == HASH_MAP_NO_SLOT


# =========================================================================
# 5. Pipeline integration tests
# =========================================================================


def test_pipeline_hashmap_barrier():
    """HashMapBuild is classified as a barrier in the pipeline planner."""
    from fvdb_tile.prototype.dsl_pipeline import plan_source

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("queries"))\n'
        'result'
    )
    plan = plan_source(source)

    kinds = [seg.kind for seg in plan.segments]
    assert "collective" in kinds, f"Expected a collective segment, got {kinds}"

    reasons = [seg.reason for seg in plan.segments if seg.kind == "collective"]
    assert "hashmap_build_collective" in reasons, f"Expected hashmap_build_collective, got {reasons}"
    assert "hashmap_lookup_collective" in reasons, f"Expected hashmap_lookup_collective, got {reasons}"


def test_pipeline_hashmap_exec():
    """Full pipeline execution with HashMapBuild + Lookup produces correct results."""
    from fvdb_tile.prototype.dsl_pipeline import compile_source
    from fvdb_tile.prototype.ops import Value

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
    result = pipeline.run(inputs)
    slots = result.output.data

    for i in range(len(keys)):
        assert int(slots[i].item()) != HASH_MAP_NO_SLOT


# =========================================================================
# 6. Dialect lowering tests
# =========================================================================


def test_lowering_scatter_reduce_structure():
    """ScatterReduce is correctly lowered into HashMapBuild + HashMapLookup + Gather."""
    from fvdb_tile.prototype.dsl_ast import (
        GatherNode,
        HashMapBuildNode,
        HashMapLookupNode,
    )
    from fvdb_tile.prototype.dsl_lower import lower_program
    from fvdb_tile.prototype.dsl_parse import parse
    from fvdb_tile.prototype.dialect_hashmap import HASHMAP_DIALECT

    source = (
        'result = ScatterReduce(Input("keys"), Input("values"), Or)\n'
        'result'
    )
    program = parse(source)
    lowered = lower_program(program, [HASHMAP_DIALECT])

    assert len(lowered.bindings) == 3, f"Expected 3 bindings, got {len(lowered.bindings)}"
    assert lowered.output == "result"

    _, node0 = lowered.bindings[0]
    _, node1 = lowered.bindings[1]
    name2, node2 = lowered.bindings[2]

    assert isinstance(node0, HashMapBuildNode), f"Expected HashMapBuildNode, got {type(node0)}"
    assert isinstance(node1, HashMapLookupNode), f"Expected HashMapLookupNode, got {type(node1)}"
    assert isinstance(node2, GatherNode), f"Expected GatherNode, got {type(node2)}"
    assert name2 == "result"


def test_lowering_passthrough_non_dialect():
    """Non-dialect nodes are not modified by the lowering pass."""
    from fvdb_tile.prototype.dsl_lower import lower_program
    from fvdb_tile.prototype.dsl_parse import parse
    from fvdb_tile.prototype.dialect_hashmap import HASHMAP_DIALECT

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("queries"))\n'
        'result'
    )
    program = parse(source)
    lowered = lower_program(program, [HASHMAP_DIALECT])

    assert len(lowered.bindings) == len(program.bindings)
    for (n1, _), (n2, _) in zip(program.bindings, lowered.bindings):
        assert n1 == n2


def test_lowering_empty_dialects():
    """Empty dialect list returns program unchanged."""
    from fvdb_tile.prototype.dsl_lower import lower_program
    from fvdb_tile.prototype.dsl_parse import parse

    source = 'result = HashMapBuild(Input("keys"))\nresult'
    program = parse(source)
    lowered = lower_program(program, [])

    assert len(lowered.bindings) == len(program.bindings)


def test_scatter_reduce_direct_eval():
    """ScatterReduce via direct eval produces correct reduced output."""
    from fvdb_tile.prototype.dsl_eval import run

    source = (
        'result = ScatterReduce(Input("keys"), Input("values"), Or)\n'
        'result'
    )
    keys = torch.tensor([1, 2, 1, 2, 3], dtype=torch.int64)
    values = torch.tensor([0b0001, 0b0010, 0b0100, 0b1000, 0b1111], dtype=torch.int64)

    inputs = {
        "keys": Value(Type(Shape(Dynamic()), ScalarType.I64), keys),
        "values": Value(Type(Shape(Dynamic()), ScalarType.I64), values),
    }
    _, result = run(source, inputs)
    # 3 unique keys -> 3 output values
    assert result.data.shape[0] == 3
    # Check that OR reduction was applied (values should be non-zero)
    assert result.data.sum().item() > 0


def test_compile_with_dialect():
    """compile_source with dialect parameter works for hash map programs."""
    from fvdb_tile.prototype.dsl_pipeline import compile_source
    from fvdb_tile.prototype.dialect_hashmap import HASHMAP_DIALECT

    source = (
        'map = HashMapBuild(Input("keys"))\n'
        'result = HashMapLookup(map, Input("keys"))\n'
        'result'
    )
    # Dialect param is accepted even if no lowering applies
    pipeline = compile_source(source, dialects=[HASHMAP_DIALECT])

    keys = torch.tensor([10, 20, 30], dtype=torch.int64)
    inputs = {
        "keys": Value(Type(Shape(Dynamic()), ScalarType.I64), keys),
    }
    result = pipeline.run(inputs)
    assert result.output.data is not None
    for i in range(len(keys)):
        assert int(result.output.data[i].item()) != HASH_MAP_NO_SLOT


# =========================================================================

if __name__ == "__main__":
    print("=== Hash function tests ===")
    test_murmurhash3_deterministic()
    print("  test_murmurhash3_deterministic: PASS")
    test_murmurhash3_no_trivial_fixpoints()
    print("  test_murmurhash3_no_trivial_fixpoints: PASS")
    test_murmurhash3_distinct_outputs()
    print("  test_murmurhash3_distinct_outputs: PASS")
    test_murmurhash3_bit_spread()
    print("  test_murmurhash3_bit_spread: PASS")

    print("\n=== Hash map build/lookup tests ===")
    test_storage_size_power_of_two()
    print("  test_storage_size_power_of_two: PASS")
    test_build_lookup_roundtrip()
    print("  test_build_lookup_roundtrip: PASS")
    test_build_lookup_unique_slots()
    print("  test_build_lookup_unique_slots: PASS")
    test_build_with_duplicates()
    print("  test_build_with_duplicates: PASS")
    test_build_lookup_scale()
    print("  test_build_lookup_scale: PASS")
    test_build_empty()
    print("  test_build_empty: PASS")
    test_scatter_reduce_or()
    print("  test_scatter_reduce_or: PASS")
    test_scatter_reduce_add()
    print("  test_scatter_reduce_add: PASS")

    print("\n=== Bitwise op tests ===")
    test_bitwise_shift_left()
    print("  test_bitwise_shift_left: PASS")
    test_bitwise_shift_right()
    print("  test_bitwise_shift_right: PASS")
    test_bitwise_xor()
    print("  test_bitwise_xor: PASS")

    print("\n=== DSL integration tests ===")
    test_dsl_hashmap_build_lookup()
    print("  test_dsl_hashmap_build_lookup: PASS")
    test_dsl_hashmap_miss()
    print("  test_dsl_hashmap_miss: PASS")

    print("\n=== Pipeline integration tests ===")
    test_pipeline_hashmap_barrier()
    print("  test_pipeline_hashmap_barrier: PASS")
    test_pipeline_hashmap_exec()
    print("  test_pipeline_hashmap_exec: PASS")

    print("\n=== Dialect lowering tests ===")
    test_lowering_scatter_reduce_structure()
    print("  test_lowering_scatter_reduce_structure: PASS")
    test_lowering_passthrough_non_dialect()
    print("  test_lowering_passthrough_non_dialect: PASS")
    test_lowering_empty_dialects()
    print("  test_lowering_empty_dialects: PASS")
    test_scatter_reduce_direct_eval()
    print("  test_scatter_reduce_direct_eval: PASS")
    test_compile_with_dialect()
    print("  test_compile_with_dialect: PASS")

    print("\nAll hashmap tests passed.")
