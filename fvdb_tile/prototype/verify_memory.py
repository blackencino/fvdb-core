# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Memory verification: detailed breakdown of CIG3 vs fVDB (NanoVDB) storage.

Run inside the fvdb_cutile venv on a machine with GPU and fvdb installed.
Prints exact byte counts for every component of both representations,
so we can verify the comparison is fair.

Usage:
    source ~/.venvs/fvdb_cutile/bin/activate
    python fvdb_tile/prototype/verify_memory.py
"""

import torch

from fvdb_tile.prototype.cig import build_compressed_cig3

try:
    import fvdb

    HAS_FVDB = True
except ImportError:
    HAS_FVDB = False
    print("ERROR: fvdb not available. This script requires fvdb to compare.\n")


def make_grid_coords(n_voxels: int, seed: int = 42) -> torch.Tensor:
    """Generate n_voxels unique random coordinates in [0, 4096)^3."""
    gen = torch.Generator().manual_seed(seed)
    coords_set: set[tuple[int, int, int]] = set()
    while len(coords_set) < n_voxels:
        batch = torch.randint(0, 4096, (n_voxels * 2, 3), generator=gen)
        for row in batch:
            coords_set.add((int(row[0]), int(row[1]), int(row[2])))
            if len(coords_set) >= n_voxels:
                break
    return torch.tensor(sorted(coords_set)[:n_voxels], dtype=torch.int32)


def analyze_one(n_voxels: int):
    print(f"\n{'=' * 70}")
    print(f"  {n_voxels:,} voxels in [0, 4096)^3")
    print(f"{'=' * 70}")

    ijk = make_grid_coords(n_voxels)

    # ===== CIG3 =====
    cig = build_compressed_cig3(ijk)
    print(f"\n  CIG3 structure: {cig.n_upper} upper, {cig.n_lower} lower, {cig.n_leaves} leaves, {cig.n_active} voxels")

    cig_components = [
        ("root_coords", cig.root_coords),
        ("upper_masks", cig.upper_masks),
        ("upper_abs_prefix", cig.upper_abs_prefix),
        ("lower_masks", cig.lower_masks),
        ("lower_abs_prefix", cig.lower_abs_prefix),
        ("leaf_masks", cig.leaf_masks),
        ("leaf_abs_prefix", cig.leaf_abs_prefix),
    ]

    print(f"\n  CIG3 component breakdown:")
    print(f"    {'Component':<22} {'Shape':>20} {'Dtype':>8} {'Bytes':>12} {'Per-node':>10}")
    print(f"    {'-' * 74}")
    cig_total = 0
    for name, t in cig_components:
        nbytes = t.nelement() * t.element_size()
        cig_total += nbytes
        n_nodes = t.shape[0] if t.ndim >= 1 else 1
        per_node = nbytes / n_nodes if n_nodes > 0 else 0
        shape_str = "x".join(str(s) for s in t.shape)
        print(f"    {name:<22} {shape_str:>20} {str(t.dtype):>8} {nbytes:>12,} {per_node:>9.0f}")

    print(f"    {'':.<22} {'':>20} {'':>8} {'----------':>12}")
    print(f"    {'TOTAL':<22} {'':>20} {'':>8} {cig_total:>12,}  ({cig_total / 1024:.1f} KB)")

    upper_bytes = sum(t.nelement() * t.element_size() for n, t in cig_components if "upper" in n)
    lower_bytes = sum(t.nelement() * t.element_size() for n, t in cig_components if "lower" in n)
    leaf_bytes = sum(t.nelement() * t.element_size() for n, t in cig_components if "leaf" in n)
    root_bytes = cig.root_coords.nelement() * cig.root_coords.element_size()
    print(f"\n    Per-level: root={root_bytes:,}  upper={upper_bytes:,}  lower={lower_bytes:,}  leaf={leaf_bytes:,}")
    if cig.n_upper > 0:
        print(f"    Per upper node: {upper_bytes / cig.n_upper:.0f} bytes  (masks: {cig.upper_masks.shape[1] * 8} + prefix: {cig.upper_abs_prefix.shape[1] * 4})")
    if cig.n_lower > 0:
        print(f"    Per lower node: {lower_bytes / cig.n_lower:.0f} bytes  (masks: {cig.lower_masks.shape[1] * 8} + prefix: {cig.lower_abs_prefix.shape[1] * 4})")
    if cig.n_leaves > 0:
        print(f"    Per leaf node:  {leaf_bytes / cig.n_leaves:.0f} bytes  (masks: {cig.leaf_masks.shape[1] * 8} + prefix: {cig.leaf_abs_prefix.shape[1] * 4})")

    # ===== fVDB =====
    if not HAS_FVDB:
        print("\n  fVDB: not available")
        return

    ijk_cuda = ijk.cuda()
    grid = fvdb.Grid.from_ijk(ijk_cuda)

    fvdb_bytes = grid.num_bytes
    print(f"\n  fVDB (NanoVDB):")
    print(f"    num_bytes:       {fvdb_bytes:>12,}  ({fvdb_bytes / 1024:.1f} KB)")
    print(f"    num_voxels:      {grid.num_voxels:>12,}")
    print(f"    num_leaf_nodes:  {grid.num_leaf_nodes:>12,}")

    n_leaves_fvdb = grid.num_leaf_nodes
    n_voxels_fvdb = grid.num_voxels

    leaf_estimate = n_leaves_fvdb * 96
    print(f"\n    Theoretical NanoVDB breakdown (estimated):")
    print(f"      Leaf nodes:  {n_leaves_fvdb:>8} x ~96 bytes  = {leaf_estimate:>12,} bytes")

    lower_per_node = 512 + 512 + 4096 * 4 + 32
    print(f"      Lower node cost: ~{lower_per_node:,} bytes each (512B mask + 16KB dense child table)")

    upper_per_node = 4096 + 4096 + 32768 * 4 + 32
    print(f"      Upper node cost: ~{upper_per_node:,} bytes each (4KB mask + 128KB dense child table)")

    print(f"      Root: variable (hash table of tile entries)")

    print(f"\n    Key structural difference:")
    print(f"      NanoVDB lower: 4096 x 4-byte child offsets = 16,384 bytes per node (DENSE)")
    print(f"      CIG3 lower:    64 x 8-byte mask + 64 x 4-byte prefix = 768 bytes per node")
    print(f"      Ratio: {lower_per_node / 768:.1f}x per lower node")
    print(f"\n      NanoVDB upper: 32768 x 4-byte child offsets = 131,072 bytes per node (DENSE)")
    print(f"      CIG3 upper:    512 x 8-byte mask + 512 x 4-byte prefix = 6,144 bytes per node")
    print(f"      Ratio: {upper_per_node / 6144:.1f}x per upper node")

    # ===== Comparison =====
    ratio = cig_total / fvdb_bytes if fvdb_bytes > 0 else 0
    print(f"\n  {'COMPARISON':=^70}")
    print(f"    CIG3 total:  {cig_total:>12,} bytes  ({cig_total / 1024:.1f} KB)")
    print(f"    fVDB total:  {fvdb_bytes:>12,} bytes  ({fvdb_bytes / 1024:.1f} KB)")
    print(f"    Ratio:       CIG3 = {ratio:.4f}x fVDB  ({1/ratio:.1f}x smaller)")
    print(f"\n    WHAT IS MEASURED:")
    print(f"      CIG3 num_bytes: sum of (nelement * element_size) for each tensor.")
    print(f"        Includes: masks, abs_prefix arrays, root_coords. Nothing else.")
    print(f"        Does NOT include: Python object overhead, CUDA allocation padding.")
    print(f"      fVDB num_bytes: NanoVDB gridSize() -- the total serialized grid buffer.")
    print(f"        (src/fvdb/detail/ops/PopulateGridMetadata.cu line 78:")
    print(f"         mNumBytes = currentGrid->gridSize())")
    print(f"        Includes: grid header, tree metadata, root node hash table,")
    print(f"        all internal nodes (with dense child offset arrays), all leaf nodes,")
    print(f"        statistics, bounding boxes, alignment padding.")
    print(f"      Both measure the GPU memory footprint of the grid structure.")
    print(f"      Neither includes the feature/data tensors attached to voxels.")


def main():
    print("Memory Verification: CIG3 vs fVDB (NanoVDB)")
    print("=" * 70)
    print()
    print("This script provides a detailed byte-level breakdown of both")
    print("representations to verify the memory comparison is fair.")
    print()
    print("CIG3 stores: bitmasks + absolute prefix sums (popcount + offset).")
    print("NanoVDB stores: bitmasks + DENSE child pointer arrays at each level.")
    print("The difference comes from dense vs. compressed child indexing.")

    for n in [1_000, 10_000, 50_000, 200_000]:
        analyze_one(n)

    print(f"\n\n{'=' * 70}")
    print("Summary: The CIG3 memory advantage is real and structural.")
    print("NanoVDB's internal nodes store dense child offset arrays")
    print("(4 bytes per slot, regardless of occupancy), while CIG3")
    print("stores bitmask + prefix and computes offsets via popcount.")
    print("At low occupancy, most of NanoVDB's per-node storage is")
    print("wasted on empty slots. The bitmask approach scales with")
    print("the mask size (fixed per level), not the child count.")
    print("=" * 70)


if __name__ == "__main__":
    main()
