# Superblock GEMM Sparse Convolution -- Unified Design

A sparse 3D convolution kernel that scales with active voxels (not leaf
volume), handles forward, backward (input gradient + weight gradient), and
transposed convolution in a single unified architecture, and requires no
topology pre-pass or global index buffers.

**Anchor.**  Every design decision is traceable to the working reference
implementation in `sifakis_ref.cu`.  Where the new design departs, the
departure is explicitly called out in the final section.

## Prior Art and References

- **Sifakis per-leaf IGEMM**: see
  [leaf_level_igemm_sparse_conv.md](leaf_level_igemm_sparse_conv.md) for the
  full specification.  Reference implementation:
  [`sifakis_ref.cu`](../../src/fvdb/detail/ops/convolution/sifakis_ref.cu).
- **Current fVDB implementation**:
  [`ImplicitGemmConv.cu`](../../src/fvdb/detail/ops/convolution/ImplicitGemmConv.cu) --
  one CUDA block per output leaf, CuTe `TiledMMA` with manual smem tiling
  (no `CollectiveMma`, no multi-stage pipeline).

---

## 1. Problem Statement

The per-leaf approach assigns one CUDA block per output NanoVDB leaf (8x8x8 =
512 voxel slots).  The GEMM tiles over all 512 positions regardless of how many
are active.  For narrow-band isosurface grids -- the dominant use case -- leaf
occupancy is typically 30-75%.  The wasted MMA cycles grow with channel count
because the GEMM's N dimension (voxels) is inflated by inactive positions that
contribute zero.

Benchmark evidence (1.2M voxels, 3x3x3 kernel, NVIDIA RTX PRO 6000 Blackwell):

| Channels | fVDB default | CUTLASS grouped | ImplicitGEMM (per-leaf MMA) | spconv | torchsparse |
|----------|-------------|-----------------|----------------------------|--------|-------------|
| C=32     | 20 ms       | 14 ms           | 11 ms                      | 4 ms   | 2 ms        |
| C=64     | 37 ms       | 28 ms           | 49 ms                      | 10 ms  | 3 ms        |
| C=128    | 81 ms       | 58 ms           | 196 ms                     | 27 ms  | 11 ms       |

At C=32 the per-leaf MMA is competitive.  At C=128 it is 18x slower than
torchsparse.  The gap comes from tiling all 512 voxel slots per leaf
regardless of activity.

torchsparse and spconv avoid this by operating on flat lists of active voxels
with precomputed topology (kernel maps / kmaps).  But precomputed kmaps have
their own costs: multi-step API, global memory for the map, and cross-device
coordination for multi-GPU domain decomposition.

The question: can we get dense-GEMM efficiency (scaling with active voxels)
while keeping the single-kernel, no-pre-pass, block-local properties *and*
supporting all four convolution operations (forward, input grad, weight grad,
transposed) from the start?

---

## 2. The Sifakis Anchor

The reference implementation (`sifakis_ref.cu`) demonstrates a working
per-leaf implicit GEMM convolution.  Its core structure:

1. **One CUDA block per output leaf.**  `blockIdx.x = leafID`.

2. **Phase 1 -- Topology build.**  The block cooperatively builds two index
   maps in shared memory:
   - `gather_map[HaloVol]` (`uint64`): for each position in the halo volume
     (8x8x8 leaf + kernel border = 10x10x10 for 3x3x3), the NanoVDB
     `getValue()` result from the source tree.  Zero means inactive.
   - `scatter_map[LeafVol]` (`uint64`): for each of the 512 positions in the
     output leaf, the NanoVDB `getValue()` from the output tree.  Zero means
     inactive.

   After `__syncthreads()`, the NanoVDB tree is never touched again.

3. **Phase 2 -- GEMM via CUTLASS `CollectiveMma`.**  The convolution is recast
   as an implicit GEMM (im2col without materializing the matrix):
   - **A tensor** (weights): regular dense layout, loaded with `cp.async`
     into a 3-stage smem pipeline.
   - **B tensor** (features): a CuTe `ComposedLayout` that algebraically maps
     `(voxel_position, kernel_offset, channel)` through the `gather_map` to a
     global feature address.  Loaded with `cp.async` + ZFILL, predicated by
     voxel activity.
   - **C tensor** (output): a `ComposedLayout` that maps through the
     `scatter_map` to a global output address.  Written with predicated stores.

**What works:**
- Single kernel launch.  No pre-pass.  No global index buffers.
- All topology is ephemeral in smem.
- Ampere multi-stage pipeline for latency hiding.
- CuTe layout algebra for im2col encoding.

**What's limited:**
- **Fixed channel dimensions.** C=64, K=128 are compile-time constants.
  The CuTe tile decomposition (Z,P,Q blocks, ZZ,PP,QQ clusters) is hardwired.
- **All 512 voxels.**  The GEMM tiles every position in the leaf, including
  inactive ones.  Inactive positions are zero-filled on load and predicated on
  store, but the MMA still computes on them.
- **Forward only.**  No backward pass, no transposed convolution.
- **3x3x3 only.**  The geometry is hardcoded.

---

## 3. Architecture Overview

### 3.1 Four Operations, One Structure

All four convolution operations reduce to the same block-level primitive:

| Operation | Primary grid | Secondary grid | B-data reads from | Kernel dir | Output target |
|-----------|-------------|---------------|-------------------|------------|--------------|
| Forward | dst | src | features | forward | output |
| Input grad | src | dst | grad_output | flipped | grad_features |
| Weight grad | dst | src | features (+grad_out for A) | forward | grad_weights |
| Transposed fwd | target | source | features | flipped | output |

Every row:
1. Iterates over leaves of the **primary grid**.
2. Probes the **secondary grid's** tree for the halo.
3. Compacts active voxels from the primary grid's leaves.
4. Runs a GEMM using the halo maps and compact list.

The halo construction is **identical** for all four operations -- same code,
different grid pointers.  The kernel offset direction is a compile-time
template parameter (`FlipKernel`).

### 3.2 Why This Doesn't Collapse

Previous phased attempts failed because later operations (grad, transpose)
revealed structural incompatibilities with assumptions baked into the forward
implementation.  This design avoids that by anchoring every component to a
property that holds across all four operations:

- **Phase A** (topology build + compaction) depends only on the primary grid's
  leaf structure and the secondary grid's tree.  It does not know or care which
  operation will follow.  The halo base offset `(Dx, Dy, Dz)` and halo
  volume `Hx*Hy*Hz` are the **same** for forward and flipped kernels (proven
  algebraically in S5.5 below).

- **Phase B** (GEMM) consumes the compact list and halo maps via a single
  shared B-tile loader function.  The only variation per operation is the
  A-tile source (weights vs grad_output), the kernel flip flag, and the
  epilogue (write vs atomicAdd).  These are template parameters, not
  structural changes.

- **Shared memory layout** is fixed at kernel launch and does not depend on
  the operation.  All four operations use the same smem footprint.

### 3.3 Key Design Decisions

1. **Keep ALL halo maps in smem simultaneously** (no recycling, no
   re-probing).  This gives a clean Phase A / Phase B separation: all tree
   access happens in Phase A; Phase B is pure smem lookups + gmem loads + MMA.
   Uses `int32` indices (not `uint64`) to halve the smem cost.

2. **Manual B-tile loading** instead of CuTe `ComposedLayout`.  The sifakis
   approach encodes im2col in layout algebra with a single level of
   indirection (gather index buffer).  Compaction introduces a second level
   (compact list -> halo map -> feature address).  Two-level indirection
   doesn't compose with `ComposedLayout`, so we compute B-tile addresses
   explicitly.  The A-tile (weights) retains a regular dense layout and can
   use `cp.async` pipelining directly.

3. **Runtime-dynamic C_in, C_out.**  Sifakis requires compile-time channel
   dimensions for its tile decomposition.  We template on kernel size (Geom)
   and scalar type only; channel dimensions are kernel arguments.

---

## 4. Phase A: Topology Build + Compaction

This phase runs identically for all four operations.  Only the grid pointers
differ.

### 4.1 Superblock Assignment

```
superblock_id = blockIdx.x
first_leaf = superblock_id * N_LEAVES
last_leaf = min(first_leaf + N_LEAVES, total_primary_leaves)
```

NanoVDB orders leaf nodes spatially (Morton-like).  Consecutive leaves are
spatially nearby, so grouping them into fixed-size superblocks gives good
spatial coherence for the halo overlap.  No internal-node-level bookkeeping
is required.

### 4.2 Halo Map Construction

For each leaf `i` in the superblock (sequentially within the block):

```
leaf = primary_grid.leaf(first_leaf + i)
origin = leaf.origin()
halo_base = origin.offsetBy(Dx, Dy, Dz)

cooperative for h in [0, HaloVol):
    coord = halo_base + decode3(h, Hx, Hy, Hz)
    val = secondary_tree.getValue(coord)
    smem.halo_maps[i][h] = (val > 0) ? int32(val - 1 + voxel_offset) : -1

__syncthreads()
```

Each leaf's halo map is stored at a fixed offset in smem.  No recycling.
After all leaves are processed, all halo maps are simultaneously available
for Phase B.

### 4.3 Compaction

After halo maps are built, compact the active primary voxels:

```
smem.compact_count = 0
__syncthreads()

for each leaf i in [0, actual_leaf_count):
    cooperative for v in [0, 512):
        if leaf[i].isActive(v):
            pos = atomicAdd_smem(&smem.compact_count, 1)
            smem.compact_list[pos] = {
                scatter_idx:   int32(leaf[i].getValue(v) - 1 + voxel_offset),
                leaf_in_block: uint8(i),
                local_voxel:   uint16(v)
            }
    __syncthreads()

N_active = smem.compact_count
```

After Phase A, shared memory contains:
- `halo_maps[N][HaloVol]` (`int32`): resolved secondary-grid indices.
- `compact_list[N_active]`: active primary voxels with scatter indices.
- `compact_count`: total active voxels.

**The NanoVDB trees are never accessed again.**

---

## 5. Phase B: Compacted Dense GEMM

### 5.1 Forward

```
Output[C_out, N_active] = W[C_out, K_total] x Features_gathered[K_total, N_active]
```

where `K_total = C_in * KernVol`.

| Matrix | Shape | How it's loaded |
|--------|-------|----------------|
| A (weights) | `[C_out, K_total]` | Regular dense layout. Pre-permuted on host to `[KernVol*C_in, C_out]` row-major. Standard tiled load or `cp.async`. |
| B (features) | `[K_total, N_active]` | Double indirection per element (S5.5). Zero-fill when halo entry is -1. |
| C (output) | `[C_out, N_active]` | Written to `output[compact_list[n].scatter_idx * C_out + m]`. Every write targets a valid active voxel; no predication needed. |

### 5.2 Input Gradient

```
grad_F[C_in, N_active_src] = W'[C_in, K_total'] x grad_O_gathered[K_total', N_active_src]
```

where `K_total' = C_out * KernVol`.

Differences from forward:
- **Primary grid = src** (iterate src leaves, compact src voxels).
- **Secondary grid = dst** (probe dst tree for halo, reading grad_output
  locations).
- **A (weights)**: pre-permuted on host to
  `weights.permute(1,2,3,4,0).reshape(C_in, KernVol*C_out)`.  Regular dense
  layout (the spatial flip is handled in the B-tile loader, not here).
- **B-tile loader**: reads from `grad_output`, uses `FlipKernel=true`.
- **C**: written to `grad_features`.

### 5.3 Weight Gradient

```
grad_W[C_out, K_total] = grad_O[C_out, N_active] x Features_gathered[N_active, K_total]^T
```

This is a GEMM where the active-voxel dimension is the **reduction** (K of
the GEMM), not a free dimension.

| Matrix | Shape | How it's loaded |
|--------|-------|----------------|
| A (grad_output) | `[C_out, N_active]` | Single indirection: `grad_output[compact_list[k].scatter_idx, m]`. Gathered by scatter index. |
| B (features) | `[K_total, N_active]` transposed | Same double-indirection loader as forward, with the tile's K and N roles swapped. |
| C (grad_weights) | `[C_out, K_total]` | Per-block partial result.  `atomicAdd` to global `grad_weights` after the block's GEMM completes. |

Uses the **same topology** as forward (primary = dst, same halo maps + compact
list).  The `atomicAdd` target is `[C_out, C_in, KernVol]`, which is small
(e.g. 128*128*27 = 442K floats = 1.7 MB for C=128, 3x3x3).  Atomic
contention is manageable because the adds are spread across many weight
entries and staggered across blocks.

### 5.4 Transposed Forward

Structurally identical to input gradient (S5.2), with different data arrays:
- B reads from `features` (not `grad_output`).
- C writes to `output` (not `grad_features`).
- Grids: primary = target grid of transpose, secondary = source grid.

The weight permutation and `FlipKernel=true` are the same as input gradient.

### 5.5 Shared B-Tile Loader

A single function handles the B tile for ALL operations:

```
template <typename Geom, bool FlipKernel>
load_b_tile(
    const int32_t* halo_maps,          // smem: [N_LEAVES][HaloVol]
    const CompactEntry* compact_list,  // smem
    const MmaElement* data,            // gmem: features or grad_output
    int C_channel,                     // C_in (fwd/wgrad) or C_out (igrad)
    int voxel_start, int kernel_start, // tile offsets into compact/K_total dims
    int N_active,
    MmaElement* tile_buf,              // smem: output tile
    int tid, int nthreads
) {
    for each element (n_local, k_local) assigned to this thread:
        n = voxel_start + n_local
        k = kernel_start + k_local

        if n >= N_active:
            tile_buf[elem] = 0
            continue

        auto entry = compact_list[n]
        int vi = entry.local_voxel >> 6
        int vj = (entry.local_voxel >> 3) & 7
        int vk = entry.local_voxel & 7

        int kern_pos = k / C_channel
        int c = k % C_channel
        int di = kern_pos / (Geom::R * Geom::S)
        int dj = (kern_pos / Geom::S) % Geom::R
        int dk = kern_pos % Geom::S

        if constexpr (FlipKernel):
            di = Geom::T - 1 - di
            dj = Geom::R - 1 - dj
            dk = Geom::S - 1 - dk

        int halo_idx = (vi + di) * Geom::Hy * Geom::Hz
                     + (vj + dj) * Geom::Hz
                     + (vk + dk)
        int32_t src = halo_maps[entry.leaf_in_block * HaloVol + halo_idx]

        tile_buf[elem] = (src >= 0) ? data[src * C_channel + c]
                                    : MmaElement{0}
}
```

**Why the halo geometry is the same for forward and flipped kernels.**  For a
primary leaf with origin `O` and offset base `(Dx, Dy, Dz) = (-(T-1)/2, ...)`:

Forward accesses halo position `(vi + di)` where `vi in [0,7], di in [0,T-1]`.
Range: `[0, 7+T-1] = [0, Hx-1]`.

Flipped accesses `(vi + T-1-di)`.  Same range: `[0+0, 7+T-1] = [0, Hx-1]`.

The halo base, halo volume, and tree probes are **identical**.  Only the index
arithmetic within the B-tile loader changes, via the compile-time
`FlipKernel` flag.  This is the structural reason Phase A is fully shared.

---

## 6. Phase C: Epilogue

### Forward / Input Grad / Transposed Forward

After the GEMM loop completes for an output tile `(m0, n0)`:

```
cooperative for (m_local, n_local) in tile:
    n = n0 + n_local
    if n < N_active:
        idx = compact_list[n].scatter_idx
        output[idx * C_fast + m0 + m_local] = accum[m_local][n_local]
```

Every compact-list entry is an active voxel.  No predication needed.

### Weight Gradient

Per-block partial `grad_W[C_out, K_total]` is accumulated in registers/smem
during the GEMM loop, then flushed:

```
cooperative for (m, n) in local_grad_W tile:
    atomicAdd(&global_grad_W[m * K_total + n], local_grad_W[m][n])
```

---

## 7. Shared Memory Design

### 7.1 Compact Entry Format

```
struct CompactEntry {        // 8 bytes, naturally aligned
    int32_t scatter_idx;     // global voxel index (0-based) for output write
    uint8_t leaf_in_block;   // leaf index within superblock [0, N)
    uint8_t _pad;
    uint16_t local_voxel;    // linear voxel index within 8x8x8 leaf [0, 511]
};
```

### 7.2 Budget

All figures use `int32` halo indices.  MMA tiles assume TF32
(`TILE_M=32, TILE_N=32, TILE_K=8`, 3-stage pipeline, A+B).  Compact list
assumes 75% leaf occupancy (conservative for narrow-band grids).

| Component | Formula | 3x3x3, N=4 | 3x3x3, N=8 | 5x5x5, N=4 | 7x7x7, N=2 |
|-----------|---------|:----------:|:----------:|:----------:|:----------:|
| Halo maps | `N * HaloVol * 4` | 16,000 | 32,000 | 27,648 | 21,952 |
| Compact list (75%) | `N * 384 * 8` | 12,288 | 24,576 | 12,288 | 6,144 |
| Compact list (100%) | `N * 512 * 8` | 16,384 | 32,768 | 16,384 | 8,192 |
| MMA tiles (3-stage) | `2 * 32 * 8 * 4 * 3` | 6,144 | 6,144 | 6,144 | 6,144 |
| Compact count | `4` | 4 | 4 | 4 | 4 |
| **Total (75%)** | | **34,436** | **62,724** | **46,084** | **34,244** |
| **Total (100%)** | | **38,532** | **70,916** | **50,180** | **36,292** |

The halo maps, compact list, and MMA tiles must all be live simultaneously
during Phase B.  No union/recycling between halo maps and MMA buffers.

Default smem limit: 48 KB.
Extended smem (Ampere/Hopper): 164 KB.
Extended smem (Blackwell): 228 KB.

### 7.3 Recommended Superblock Sizes

| Kernel | Superblock N | Smem needed | Fits in |
|--------|:-----------:|:----------:|---------|
| 3x3x3 | 4 | ~35 KB | Default (48 KB) |
| 3x3x3 | 8 | ~63 KB | Extended |
| 5x5x5 | 4 | ~46 KB | Default (48 KB), tight |
| 5x5x5 | 8 | ~86 KB | Extended |
| 7x7x7 | 2 | ~34 KB | Default (48 KB) |
| 7x7x7 | 4 | ~60 KB | Extended |

At N=1 (no superblock, one leaf per block), smem is minimal and the design
still provides compaction.  N=1 is the degenerate case of the same kernel
template, not a separate code path.

---

## 8. Kernel Size Scaling

| Kernel | HaloVol | Halo bytes (`int32`) | KernVol | K_total (C=128) | Max N (48 KB) | Max N (164 KB) |
|--------|---------|:-------------------:|:-------:|:---------------:|:------------:|:-------------:|
| 3x3x3 | 1,000 | 4,000 | 27 | 3,456 | 4 | 16+ |
| 5x5x5 | 1,728 | 6,912 | 125 | 16,000 | 4 | 8 |
| 7x7x7 | 2,744 | 10,976 | 343 | 43,904 | 2 | 4 |

Larger KernVol increases `K_total = C_in * KernVol`, making the GEMM more
compute-bound and more MMA-friendly.  The compaction benefit grows
proportionally because `K_total * N_active` dominates the flop count and
every compacted-away voxel eliminates `K_total` worth of wasted MAC
operations.

---

## 9. Multi-GPU Friendliness

Each CUDA block is fully self-contained:

- **No inter-block synchronization.**  Blocks do not communicate.  No
  `grid.sync()`, no `cudaLaunchCooperativeKernel`.
- **No global scratch buffers.**  The compacted work list lives entirely in
  shared memory.  No global memory allocation beyond the input/output feature
  arrays (and the small `grad_weights` atomic target for weight grad).
- **No persistent topology.**  The gather/scatter maps are ephemeral.  They
  exist only in smem for the duration of one block's execution.  The user-facing
  API is single-step: `output = conv(features, weights, grid)`.

For multi-GPU with domain decomposition:

- Each GPU holds a local NanoVDB tree covering its domain plus a halo region.
- Each GPU launches the same kernel on its local tree.
- Blocks near domain boundaries probe into the halo region of the local tree.
  The halo voxels were populated by a prior communication step (halo exchange)
  at the tree level.
- **No cross-GPU coordination during the convolution kernel.**

This is the same isolation property that the sifakis per-leaf kernel has.
The superblock design preserves it because blocks remain autonomous.

---

## 10. Pseudocode -- Unified Kernel

```
enum ConvOp { Forward, InputGrad, WeightGrad, TransposedFwd };

template <Geom, Scalar, N_LEAVES, ConvOp Op>
__global__ void superblock_conv(
    Accessor primary_acc,     // leaves we iterate
    Accessor secondary_acc,   // tree we probe for halo
    Scalar* data_B,           // features (fwd/wgrad) or grad_output (igrad)
    Scalar* data_A_extra,     // grad_output (wgrad only; nullptr otherwise)
    Scalar* weights,          // pre-permuted per operation on host
    Scalar* output,           // output / grad_features / grad_weights
    int C_fast,               // M dimension of GEMM (C_out or C_in)
    int C_slow                // channel dim in K_total (C_in or C_out)
) {
    constexpr bool Flip = (Op == InputGrad || Op == TransposedFwd);
    constexpr int HaloVol = Geom::HaloVol;

    __shared__ struct {
        int32_t halo_maps[N_LEAVES][HaloVol];
        CompactEntry compact_list[N_LEAVES * 512];  // worst case
        int compact_count;
        alignas(16) char mma_buf[MMA_BUF_BYTES];
    } smem;

    int sb = blockIdx.x;
    int first = sb * N_LEAVES;
    int n_leaves = min(N_LEAVES, total_primary_leaves - first);

    // ===== PHASE A: topology + compaction (shared by all ops) =====

    smem.compact_count = 0;
    __syncthreads();

    auto sec_tree = secondary_acc.grid(0)->getAccessor();

    for (int i = 0; i < n_leaves; i++) {
        auto& leaf = primary_acc.grid(0)->tree().getFirstNode<0>()[first + i];
        auto origin = leaf.origin();
        auto halo_base = origin.offsetBy(Geom::Dx, Geom::Dy, Geom::Dz);

        // Build halo map
        cooperative for h in [0, HaloVol):
            coord = halo_base + decode3(h, Hx, Hy, Hz)
            val = sec_tree.getValue(coord)
            smem.halo_maps[i][h] = (val > 0) ? int32(val-1+voxel_offset) : -1

        __syncthreads()

        // Compact active voxels
        cooperative for v in [0, 512):
            if leaf.isActive(v):
                pos = atomicAdd_smem(&smem.compact_count, 1)
                smem.compact_list[pos] = {
                    int32(leaf.getValue(v)-1+voxel_offset), uint8(i), 0, uint16(v)
                }

        __syncthreads()
    }

    int N_active = smem.compact_count;
    if (N_active == 0) return;

    // ===== PHASE B: compacted dense GEMM =====

    int K_total = C_slow * Geom::KernVol;

    if constexpr (Op == Forward || Op == InputGrad || Op == TransposedFwd) {
        // C[C_fast, N_active] = A[C_fast, K_total] x B[K_total, N_active]

        for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
            for (int n0 = 0; n0 < N_active; n0 += TILE_N) {
                clear(accum);

                for (int k0 = 0; k0 < K_total; k0 += TILE_K) {
                    // A tile: weights (regular dense layout)
                    load_weight_tile(weights, C_fast, m0, k0, tile_A);

                    // B tile: gathered features or grad_output
                    load_b_tile<Geom, Flip>(
                        smem.halo_maps, smem.compact_list,
                        data_B, C_slow,
                        n0, k0, N_active, tile_B);

                    __syncthreads();
                    MMA(accum, tile_A, tile_B);
                    __syncthreads();
                }

                // Phase C: scatter output
                cooperative for (m_local, n_local) in tile:
                    n = n0 + n_local
                    if n < N_active:
                        idx = smem.compact_list[n].scatter_idx
                        output[idx * C_fast + m0 + m_local] = accum[m_local][n_local]

                __syncthreads();
            }
        }
    }
    else if constexpr (Op == WeightGrad) {
        // C[C_out, K_total] = grad_O[C_out, N_active] x B[N_active, K_total]^T
        // N_active is the REDUCTION dimension

        for (int m0 = 0; m0 < C_fast; m0 += TILE_M) {
            for (int n0 = 0; n0 < K_total; n0 += TILE_N) {
                clear(accum);

                for (int k0 = 0; k0 < N_active; k0 += TILE_K) {
                    // A tile: grad_output gathered by scatter_idx
                    load_grad_tile(
                        data_A_extra, smem.compact_list,
                        C_fast, m0, k0, N_active, tile_A);

                    // B tile: features (same loader as forward)
                    load_b_tile<Geom, false>(
                        smem.halo_maps, smem.compact_list,
                        data_B, C_slow,
                        k0, n0, N_active, tile_B);

                    __syncthreads();
                    MMA(accum, tile_A, tile_B);
                    __syncthreads();
                }

                // Phase C: atomicAdd to global weight gradient
                cooperative for (m_local, n_local) in tile:
                    atomicAdd(&output[(m0+m_local) * K_total + n0+n_local],
                              accum[m_local][n_local]);
            }
        }
    }
}
```

### Host-side Launch

```python
# Forward
superblock_conv<Geom, float, N, Forward><<<
    ceil(dst_leaves / N), 128, SMEM, stream>>>(
    dst_acc, src_acc, features, nullptr,
    W.permute(2,3,4,1,0).reshape(KernVol*C_in, C_out),
    output, C_out, C_in)

# Input gradient
superblock_conv<Geom, float, N, InputGrad><<<
    ceil(src_leaves / N), 128, SMEM, stream>>>(
    src_acc, dst_acc, grad_output, nullptr,
    W.permute(1,2,3,4,0).reshape(C_in, KernVol*C_out),
    grad_features, C_in, C_out)

# Weight gradient
superblock_conv<Geom, float, N, WeightGrad><<<
    ceil(dst_leaves / N), 128, SMEM, stream>>>(
    dst_acc, src_acc, features, grad_output,
    nullptr,
    grad_weights, C_out, C_in)

# Transposed forward
superblock_conv<Geom, float, N, TransposedFwd><<<
    ceil(target_leaves / N), 128, SMEM, stream>>>(
    target_acc, source_acc, features, nullptr,
    W.permute(1,2,3,4,0).reshape(C_in, KernVol*C_out),
    output, C_in, C_out)
```

---

## 11. Comparison to kmap-Based Approaches

| Property | kmap (spconv, torchsparse) | Per-leaf IGEMM (sifakis) | Superblock GEMM |
|----------|---------------------------|-------------------------|-----------------|
| Topology construction | Separate pre-pass, global buffer | In-kernel, per-leaf smem | In-kernel, per-superblock smem |
| GEMM density | Dense (active voxels only) | Sparse (full leaf volume) | Dense (compacted active voxels) |
| Work scaling | O(active voxels) | O(leaves * 512) | O(active voxels) |
| API complexity | Multi-step (build kmap, then conv) | Single-step | Single-step |
| Global memory overhead | kmap buffer (~100s of MB) | None | None |
| Multi-GPU | Requires distributed kmap | Naturally block-local | Naturally block-local |
| Grad / transpose | Built into kmap framework | Not implemented | Same kernel template |
| Tree access during conv | None (kmap replaces it) | Phase 1 only | Phase A only |

---

## 12. What Stays, What Changes from Sifakis

### Unchanged

1. **NanoVDB tree probes in smem.**  `getValue()` on the secondary tree to
   populate halo maps -- identical code.
2. **Halo geometry.**  `Hx = T + 7`, `Hy = R + 7`, `Hz = S + 7`.
3. **Kernel offset semantics.**  `(Dx, Dy, Dz) = (-(T-1)/2, -(R-1)/2,
   -(S-1)/2)`.
4. **MMA atom.**  `SM80_16x8x8_F32TF32TF32F32_TN` for TF32.
5. **Tile sizes.**  `TILE_M=32, TILE_N=32, TILE_K=8` for TF32.
6. **Thread count.**  128 threads per block.
7. **Block autonomy.**  No inter-block communication.  No global scratch.
8. **Inactive source handling.**  Zero-fill on gather when halo entry
   indicates an inactive source voxel.
9. **Index type for halo/scatter.**  NanoVDB `getValue()` returns 0 for
   inactive, 1-based for active.  Same sentinel convention.

### Changed

1. **Work unit.**  One block per **superblock** (N leaves), not one block per
   leaf.  N is a template parameter; N=1 recovers the per-leaf case.
2. **Scatter map replaced by compact list.**  Instead of 512-entry scatter
   maps (one per leaf), a single compact list of active voxels across all
   leaves in the superblock.
3. **Halo maps retained, not recycled.**  All N halo maps live simultaneously
   in smem.  Cost: `N * HaloVol * 4` bytes (`int32` instead of `uint64`).
4. **B-tile loading.**  Manual double indirection (compact list -> halo map ->
   feature address) instead of CuTe `ComposedLayout` with `IndexedGather`.
5. **No output predication.**  Every compact-list entry is active; the
   epilogue writes unconditionally.
6. **Runtime-dynamic channels.**  `C_in`, `C_out` are kernel arguments, not
   template parameters.
7. **All four operations.**  Forward, input grad, weight grad, transposed
   forward from a single kernel template, parameterized by `ConvOp`.
8. **Kernel size templated.**  `LeafConvGeometry<T, R, S>` for 3x3x3, 5x5x5,
   7x7x7.
