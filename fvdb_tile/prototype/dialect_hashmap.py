# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Hash map dialect: lowering rules for high-level keyed operations.

Provides ``HASHMAP_DIALECT`` which lowers:

  ``ScatterReduce(keys, values, reduce_fn)``

into a composition of hash map primitives::

  _map   = HashMapBuild(keys)           # barrier
  _slots = HashMapLookup(_map, keys)    # tile-parallel
  result = Gather(value_arr, _slots)    # ... (see below)

For the initial torch-eval path, the lowering emits HashMapBuild +
HashMapLookup + a simple Gather-based scatter (sufficient for the
sequential reference evaluator).  A future cuTile-targeted lowering
could emit atomic-reduce intrinsics instead.
"""

from __future__ import annotations

from .dsl_ast import (
    GatherNode,
    HashMapBuildNode,
    HashMapLookupNode,
    Node,
    RefNode,
    ScatterReduceNode,
)
from .dsl_lower import Dialect, LoweringRule, _FreshNames


# ---------------------------------------------------------------------------
# ScatterReduce lowering
# ---------------------------------------------------------------------------


def _lower_scatter_reduce(
    node: ScatterReduceNode, name: str, fresh: _FreshNames
) -> list[tuple[str, Node]]:
    """Lower ScatterReduce into HashMapBuild + HashMapLookup + Gather.

    ScatterReduce(keys, values, reduce_fn) becomes:

        _map   = HashMapBuild(keys)
        _slots = HashMapLookup(_map, keys)
        <name> = Gather(values, _slots)

    The Gather here is a placeholder: it pairs each value with its
    slot index.  The actual scatter-reduce semantics (combining
    duplicate keys with the reduce function) are handled by the
    evaluator for HashMapBuild (which deduplicates keys) and by
    downstream consumers who use the slots to write into value arrays.

    For full scatter-reduce semantics in the evaluator, the eval case
    for ScatterReduceNode is kept as a fallback that uses
    ops.hash_map_scatter_reduce directly.  The lowered form here
    demonstrates the dialect mechanism and is correct for the common
    case of unique keys.
    """
    map_name = fresh("map")
    slots_name = fresh("slots")

    return [
        (map_name, HashMapBuildNode(keys=node.keys)),
        (slots_name, HashMapLookupNode(key_arr=RefNode(map_name), queries=node.keys)),
        (name, GatherNode(target=node.values, indexer=RefNode(slots_name))),
    ]


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------

HASHMAP_DIALECT = Dialect(
    name="hashmap",
    rules={
        ScatterReduceNode: LoweringRule(
            node_type=ScatterReduceNode,
            rewrite=_lower_scatter_reduce,
        ),
    },
)
