// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// gather_tensor.h -- CuTe indirect tensor via ComposedLayout.
//
// Implements the spec's IndirectTensor abstraction using CuTe's
// ComposedLayout + IndexedGather + CustomStride. This makes the
// shared-memory indirection through gather_map invisible to the GEMM
// template: the GEMM sees a standard tensor whose "stride" happens
// to perform an indirect lookup.
//
// Adapted from sifakis/openvdb gather_tensor.hpp, parameterized through
// our conv_geometry / conv_types traits rather than hardcoded constants.
//
// Target architecture: SM80 (Ampere).
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_GATHER_TENSOR_H
#define FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_GATHER_TENSOR_H

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace fvdb {
namespace detail {
namespace leaf_igemm {

using namespace cute;

// =============================================================================
// IndexedGather: function object that looks up an index table
// =============================================================================
//
// operator()(i) returns indices_[i].
// This is the shared-memory indirection: the gather_map lookup.

template <typename Index>
struct IndexedGather {
    CUTE_HOST_DEVICE constexpr IndexedGather(Index const *indices = {}) : indices_(indices) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr Index
    operator()(I i) const {
        return indices_[i];
    }

    CUTE_HOST_DEVICE friend void
    print(IndexedGather const &) {
        cute::print("Indexed");
    }

    Index const *indices_;
};

// =============================================================================
// CustomStride: stride that applies a function then multiplies
// =============================================================================
//
// i * CustomStride = func_(i) * stride_
//
// In this algorithm:
//   func_   = IndexedGather (shared-memory lookup)
//   stride_ = C_in (channel stride)
// So: i * CustomStride = gather_map[i] * C_in

template <typename Func, typename Stride>
struct CustomStride {
    CUTE_HOST_DEVICE constexpr CustomStride(Func const &func, Stride const &stride)
        : func_(func), stride_(stride) {}

    template <typename I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(I i, CustomStride const &s) {
        return s.func_(i) * s.stride_;
    }

    template <typename I>
    CUTE_HOST_DEVICE constexpr friend auto
    operator*(CustomStride const &s, I i) {
        return s.func_(i) * s.stride_;
    }

    CUTE_HOST_DEVICE friend void
    print(CustomStride const &s) {
        cute::print("Custom{");
        print(s.func_);
        cute::print(",");
        print(s.stride_);
        cute::print("}");
    }

    template <typename Div>
    CUTE_HOST_DEVICE constexpr friend auto
    safe_div(CustomStride const &s, Div const &div) {
        return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_,
                                                                      safe_div(s.stride_, div));
    }

    template <typename Shape>
    CUTE_HOST_DEVICE constexpr friend auto
    make_layout(Shape const &shape, CustomStride const &stride) {
        return Layout<Shape, CustomStride>(shape, stride);
    }

    Func func_;
    Stride stride_;
};

// =============================================================================
// make_custom_stride_layout: inject a CustomStride into a stride tuple
// =============================================================================
//
// Finds the first non-unit stride in the stride tuple and replaces it
// with a CustomStride wrapping the gather function.

template <typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_custom_stride_layout(Stride const &stride, Func &&func) {
    auto idx = find_if(stride, [](auto x) { return not is_constant<1, decltype(x)>{}; });
    constexpr int I = decltype(idx)::value;
    return make_layout(
        repeat_like(stride, _1{}),
        replace<I>(stride, CustomStride{static_cast<Func &&>(func), get<I>(stride)}));
}

// =============================================================================
// make_gather_tensor: create a tensor with ComposedLayout for indirect access
// =============================================================================
//
// The returned tensor has a ComposedLayout:
//   OuterLayout  = custom stride layout (with IndexedGather)
//   Offset       = (0, 0, ...)
//   InnerLayout  = identity layout (logical coordinate -> (halo_idx, channel))
//
// Accessing element [ck, n] of this tensor:
//   1. InnerLayout maps (ck, n) to (halo_table_index, channel_offset)
//   2. OuterLayout maps halo_table_index through gather_map[i] * C_in + channel_offset
//   3. The result is the global memory offset into the feature array

template <typename Iterator, typename Shape, typename Stride, typename Func>
CUTLASS_HOST_DEVICE auto
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func) {
    Layout<Shape, Stride> matrix_layout = make_identity_layout(shape);
    auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
    auto gather_layout =
        make_custom_stride_layout(stride, static_cast<Func &&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
}

} // namespace leaf_igemm
} // namespace detail
} // namespace fvdb

// CuTe v4.x provides built-in upcast for standard layouts. If a
// ComposedLayout-specific upcast overload is needed for vectorized copy
// operations with the CustomStride, it can be added here following the
// sifakis gather_tensor.hpp pattern. For now, the built-in suffices.

#endif // FVDB_DETAIL_OPS_CONVOLUTION_LEAF_IGEMM_GATHER_TENSOR_H
