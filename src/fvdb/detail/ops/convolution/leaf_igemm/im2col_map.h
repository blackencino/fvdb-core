// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// im2col_map.h -- Virtual im2col address computation.
//
// Maps a logical GEMM coordinate (contraction_idx, spatial_idx) to a
// physical feature-memory address via the gather_map in shared memory,
// without materialising the unrolled activation matrix.
//
// This is the heart of the implicit GEMM: the function that replaces
// an explicit im2col buffer with on-the-fly index arithmetic.
//
// No CUTLASS dependency. Pure address arithmetic parameterized by traits.
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_IM2COL_MAP_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_IM2COL_MAP_H

#include "conv_geometry.h"
#include "conv_tiling.h"
#include "conv_variant.h"
#include "core_types.h"

namespace fvdb {
namespace detail {
namespace leaf_igemm {

// =============================================================================
// Device-side coord3 arithmetic
// =============================================================================
//
// The consteval coord3 from core_types.h cannot be used in __device__ code
// (consteval is host-only in CUDA). These are __device__ __forceinline__
// equivalents for runtime computation on the GPU.

struct dcoord3 {
    int x, y, z;
};

__device__ __forceinline__ dcoord3
operator+(dcoord3 a, dcoord3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ dcoord3
operator-(dcoord3 a, dcoord3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ dcoord3
operator*(dcoord3 a, dcoord3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ int
dprod(dcoord3 v) {
    return v.x * v.y * v.z;
}

__device__ __forceinline__ int
dencode3(dcoord3 pos, dcoord3 dims) {
    return pos.x * dims.y * dims.z + pos.y * dims.z + pos.z;
}

__device__ __forceinline__ dcoord3
ddecode3(int i, dcoord3 dims) {
    return {i / (dims.y * dims.z), (i / dims.z) % dims.y, i % dims.z};
}

// Convert compile-time coord3 to device-side dcoord3.
__device__ __forceinline__ dcoord3
to_dcoord3(coord3 c) {
    return {c.x, c.y, c.z};
}

// =============================================================================
// im2col_result: return type of the im2col map
// =============================================================================

template <typename ElemIdx>
struct im2col_result {
    ElemIdx voxel_idx;
    int channel;
};

// =============================================================================
// im2col_map: the virtual im2col address computation
// =============================================================================
//
// Template parameters:
//   Geom    : geometry_like  (provides K, S, D, halo, kern_vol)
//   Tiling  : tiling_like    (provides B, blk_vol)
//   Variant : conv_variant   (controls flip via variant_traits)
//
// The map takes:
//   ck         : contraction index in [0, C_in * KERN_VOL)
//   n          : spatial index in [0, BLK_VOL)
//   block_orig : leaf-local origin of the current output block
//   gather_map : pointer to shared-memory halo index buffer
//
// Returns: (voxel_index, channel)
//   Caller uses: if voxel_idx != 0 then features[voxel_idx * C_in + channel]
//                else 0.0

template <geometry_like Geom, tiling_like Tiling, conv_variant Variant>
struct im2col_map {

    static constexpr bool flip = variant_traits<Variant>::flip();

    template <typename ElemIdx>
    __device__ __forceinline__ static im2col_result<ElemIdx>
    apply(int ck, int n, dcoord3 block_orig, ElemIdx const *gather_map, int kern_vol) {

        // 1. Decode contraction index into channel + filter offset.
        int const c     = ck / kern_vol;
        int const k_lin = ck % kern_vol;

        dcoord3 const K = to_dcoord3(Geom::K());
        dcoord3 delta   = ddecode3(k_lin, K);

        // 1b. Flip the kernel offset for backward-direction operations.
        if constexpr (flip) {
            dcoord3 const Km1 = {K.x - 1, K.y - 1, K.z - 1};
            delta             = Km1 - delta;
        }

        // 2. Decode spatial index into position within block.
        dcoord3 const B = to_dcoord3(Tiling::B());
        dcoord3 const v = ddecode3(n, B);

        // 3. Compute halo-relative input position.
        //    h = (block_orig + v) * S + delta * D
        dcoord3 const S    = to_dcoord3(Geom::S());
        dcoord3 const D    = to_dcoord3(Geom::D());
        dcoord3 const h    = (block_orig + v) * S + delta * D;
        dcoord3 const halo = to_dcoord3(Geom::halo());

        // 4. Look up the global feature index from shared memory.
        int const h_lin         = dencode3(h, halo);
        ElemIdx const voxel_idx = gather_map[h_lin];

        return {voxel_idx, c};
    }

    // Convenience: compute only the halo position (for predicate construction).
    __device__ __forceinline__ static int
    halo_position(int ck, int n, dcoord3 block_orig, int kern_vol) {

        int const k_lin = ck % kern_vol;
        dcoord3 const K = to_dcoord3(Geom::K());
        dcoord3 delta   = ddecode3(k_lin, K);

        if constexpr (flip) {
            dcoord3 const Km1 = {K.x - 1, K.y - 1, K.z - 1};
            delta             = Km1 - delta;
        }

        dcoord3 const B    = to_dcoord3(Tiling::B());
        dcoord3 const v    = ddecode3(n, B);
        dcoord3 const S    = to_dcoord3(Geom::S());
        dcoord3 const D    = to_dcoord3(Geom::D());
        dcoord3 const h    = (block_orig + v) * S + delta * D;
        dcoord3 const halo = to_dcoord3(Geom::halo());

        return dencode3(h, halo);
    }
};

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_IM2COL_MAP_H
