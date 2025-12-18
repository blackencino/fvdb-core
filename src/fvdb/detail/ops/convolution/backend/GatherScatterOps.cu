// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/convolution/backend/ConvPragmaMessage.h>
#include <fvdb/detail/ops/convolution/backend/GatherScatterOps.h>
#include <fvdb/detail/utils/MultiDispatch.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <algorithm>
#include <variant>

namespace fvdb {
namespace detail {
namespace ops {

// using detail::CpuTag;
// using detail::CudaTag;
// using detail::DeviceTag;

namespace {

template <typename T>
__global__ void
sigmoidKernel(T const *x, T *y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = T(1) / (T(1) + exp(-x[idx]));
    }
}

template <auto Device, typename...> struct SigmoidOpImpl;

template <typename T> struct SigmoidOpImpl<c10::kCUDA, T> {
    static void
    call(void const *x, void *y, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto *yp       = static_cast<T *>(y);
        sigmoidKernel<T><<<GET_BLOCKS(n, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM>>>(xp, yp, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

template <typename T> struct SigmoidOpImpl<c10::kCPU, T> {
    static void
    call(void const *x, void *y, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto *yp       = static_cast<T *>(y);
        for (int64_t i = 0; i < n; ++i) {
            yp[i] = T(1) / (T(1) + exp(-xp[i]));
        }
    }
};

using SigmoidOp = multi_dispatch::
    MultiDispatchOp<SigmoidOpImpl, multi_dispatch::CpuCudaDevices, multi_dispatch::AllFloatTypes>;

inline torch::Tensor
sigmoid(torch::Tensor x) {
    auto y                  = torch::empty_like(x);
    c10::Device const &dev  = x.device();
    c10::ScalarType const t = x.scalar_type();
    void const *x_ptr       = x.data_ptr<void>();
    void *y_ptr             = y.data_ptr<void>();
    int64_t n               = x.numel();
    SigmoidOp op;
    op(dev, t, x_ptr, y_ptr, n);
    return y;
}

template <typename T, typename S>
__global__ void
multiplyKernel(T const *x, S const *y, std::common_type_t<T, S> *z, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] * y[idx];
    }
}

template <auto Device, typename...> struct MultiplyOpImpl;

template <typename T, typename S> struct MultiplyOpImpl<c10::kCPU, T, S> {
    static void
    call(void const *x, void const *y, void *z, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto const *yp = static_cast<S const *>(y);
        auto *zp       = static_cast<std::common_type_t<T, S> *>(z);
        for (int64_t i = 0; i < n; ++i) {
            zp[i] = xp[i] * yp[i];
        }
    }
};

template <typename T, typename S> struct MultiplyOpImpl<c10::kCUDA, T, S> {
    static void
    call(void const *x, void const *y, void *z, int64_t n) {
        auto const *xp = static_cast<T const *>(x);
        auto const *yp = static_cast<S const *>(y);
        auto *zp       = static_cast<std::common_type_t<T, S> *>(z);
        multiplyKernel<T, S>
            <<<GET_BLOCKS(n, DEFAULT_BLOCK_DIM), DEFAULT_BLOCK_DIM>>>(xp, yp, zp, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

using MultiplyOp = multi_dispatch::MultiDispatchOp<MultiplyOpImpl,
                                                   multi_dispatch::CpuCudaDevices,
                                                   multi_dispatch::AllFloatTypes,
                                                   multi_dispatch::AllFloatTypes>;

inline torch::Tensor
multiply(torch::Tensor x, torch::Tensor y) {
    auto out = c10::promoteTypes(x.scalar_type(), y.scalar_type());
    auto z   = torch::empty(x.sizes(), x.options().dtype(out));

    c10::Device const &dev  = x.device();
    c10::ScalarType const t = x.scalar_type();
    c10::ScalarType const s = y.scalar_type();
    void const *x_ptr       = x.data_ptr<void>();
    void const *y_ptr       = y.data_ptr<void>();
    void *z_ptr             = z.data_ptr<void>();
    int64_t n               = x.numel();
    MultiplyOp op;
    op(dev, t, s, x_ptr, y_ptr, z_ptr, n);
    return z;
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
gatherKernelRegular(int const n_k,
                    int const n_in,
                    int const c,
                    scalar_t const *in_feat,
                    scalar_t *out_feat,
                    int const *kmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i     = index / c;
    int j     = index % c;
    if (i >= n_k)
        return;
    int in_pos = kmap[2 * i];
    if (in_pos < 0)
        return;
    out_feat[i * c + j] = in_feat[in_pos * c + j];
}

struct GatherRegularOp : public DeviceDispatchOp<GatherRegularOp> {};

template <typename scalar_t>
static void
gatherCpuRegular(int const n_k,
                 int const n_in,
                 int const c,
                 scalar_t const *in_feat,
                 scalar_t *out_feat,
                 int const *kmap) {
    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i];
        if (in_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
scatterKernelRegular(int const n_in,
                     int const n_out,
                     int const c,
                     scalar_t const *in_feat,
                     scalar_t *out_feat,
                     int const *kmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i     = index / c;
    int j     = index % c;
    if (i >= n_in)
        return;
    int out_pos = kmap[2 * i + 1];
    if (out_pos < 0)
        return;
    out_feat[out_pos * c + j] += in_feat[i * c + j];
}

template <typename scalar_t>
static void
scatterCpuRegular(int const n_in,
                  int const n_out,
                  int const c,
                  scalar_t const *in_feat,
                  scalar_t *out_feat,
                  int const *kmap) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1];
        if (out_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[out_pos * c + j] += in_feat[i * c + j];
        }
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
gatherKernelTransposed(int const n_k,
                       int const n_in,
                       int const c,
                       scalar_t const *in_feat,
                       scalar_t *out_feat,
                       int const *kmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i     = index / c;
    int j     = index % c;
    if (i >= n_k)
        return;
    int in_pos = kmap[2 * i + 1];
    if (in_pos < 0)
        return;
    out_feat[i * c + j] = in_feat[in_pos * c + j];
}

template <typename scalar_t>
static void
gatherCpuTransposed(int const n_k,
                    int const n_in,
                    int const c,
                    scalar_t const *in_feat,
                    scalar_t *out_feat,
                    int const *kmap) {
    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i + 1];
        if (in_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(DEFAULT_BLOCK_DIM) void
scatterKernelTransposed(int const n_in,
                        int const n_out,
                        int const c,
                        scalar_t const *in_feat,
                        scalar_t *out_feat,
                        int const *kmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i     = index / c;
    int j     = index % c;
    if (i >= n_in)
        return;
    int out_pos = kmap[2 * i];
    if (out_pos < 0)
        return;
    out_feat[out_pos * c + j] += in_feat[i * c + j];
}

template <typename scalar_t>
static void
scatterCpuTransposed(int const n_in,
                     int const n_out,
                     int const c,
                     scalar_t const *in_feat,
                     scalar_t *out_feat,
                     int const *kmap) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i];
        if (out_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[out_pos * c + j] += in_feat[i * c + j];
        }
    }
}

// Type list for scalar dispatch (replaces AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf, c10::kBFloat16)
using ConvScalarTypes = std::tuple<float, double, at::Half, at::BFloat16>;

// Convert a tuple of types to a variant
template <typename Tuple> struct TupleToVariant;

template <typename... Ts> struct TupleToVariant<std::tuple<Ts...>> {
    using type = std::variant<Ts...>;
};

template <typename Tuple> using TupleToVariantT = typename TupleToVariant<Tuple>::type;

// Map scalar type enum to actual type and invoke functor
template <typename Tuple, typename Func> struct ScalarDispatcher;

template <typename First, typename... Rest, typename Func>
struct ScalarDispatcher<std::tuple<First, Rest...>, Func> {
    static void
    dispatch(at::ScalarType scalarType, Func &&func) {
        if (scalarType == c10::CppTypeToScalarType<First>::value) {
            std::forward<Func>(func).template operator()<First>();
        } else if constexpr (sizeof...(Rest) > 0) {
            ScalarDispatcher<std::tuple<Rest...>, Func>::dispatch(scalarType,
                                                                  std::forward<Func>(func));
        } else {
            TORCH_CHECK(false,
                        "Unsupported scalar type: ",
                        toString(scalarType),
                        " in convolution operation");
        }
    }
};

// User-facing dispatch function
template <typename TypeList, typename Func>
void
dispatchScalarType(at::ScalarType scalarType, Func &&func) {
    ScalarDispatcher<TypeList, Func>::dispatch(scalarType, std::forward<Func>(func));
}

} // namespace

// in_feat: (N, c) N=# of input points, c = input channels
// out_feat: (M, o) M=# of output points, o = output channels
//                  for stride=1, M=N. For stride>1, the N input coords
//                  are requantized to M points with grid size (stride *
//                  cur_stride)
// kernel: (k^3, c, o) for a 3D convolution of length k
// neighbor_map: (a, 2) the hash table query results from out_coords to
// in_coords
//                      where neighbor_map[:,0] is the index of the output
//                      feature and neighbor_map[:,1] is the index of the input
//                      feature
// neighbor_offset: (k^3) count of active weights based on neighbor_map
//                      with unused weights having 0 and neighbor_offset[k^3/2]
//                      holding w[0,0].
void
GatherScatterOp::execute(CudaTag,
                         torch::Tensor in_feat,
                         torch::Tensor out_feat,
                         torch::Tensor kernel,
                         torch::Tensor neighbor_map,
                         torch::Tensor neighbor_offset,
                         bool middleAcceleration) {
    torch::Device const device =
        checkDevicesAndGetFirst(in_feat, out_feat, kernel, neighbor_map, neighbor_offset);
    c10::cuda::CUDAGuard deviceGuard(device);

    if (in_feat.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    bool is_half = in_feat.scalar_type() == at::ScalarType::Half;

    int n_in_feats     = in_feat.size(0);
    int n_in_channels  = in_feat.size(1);
    int n_out_feats    = out_feat.size(0);
    int n_out_channels = out_feat.size(1);

    int kernel_volume = kernel.size(0);

    // memory optimization
    bool precompute_mid = false;
    int mid_kernel      = kernel_volume / 2;
    int in_buffer_size  = 1;
    // we can precompute features for w[0,0] which avoids gather/scatter
    if (kernel_volume % 2 == 1 && n_in_feats == n_out_feats && middleAcceleration) {
        precompute_mid = true;
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + mid_kernel);
        in_buffer_size =
            std::max(in_buffer_size,
                     *std::max_element(neighbor_offset.data_ptr<int>() + mid_kernel + 1,
                                       neighbor_offset.data_ptr<int>() + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        // (N, c) X (c, o) = (N, o)
        torch::mm_out(out_feat, in_feat, kernel[mid_kernel]);
    } else {
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume);
    }

    auto options    = torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer  = torch::zeros({in_buffer_size, n_in_channels}, options);
    auto out_buffer = torch::zeros({in_buffer_size, n_out_channels}, options);
    int cur_offset  = 0;
    // gather/gemm/scatter on each weight
    for (int i = 0; i < kernel_volume; i++) {
        int n_active_feats = neighbor_offset.data_ptr<int>()[i];
        // if there's no active features for this weight, skip it
        if (n_active_feats == 0) {
            continue;
        }

        // if w[0,0] was precomputed above, skip it
        if ((i == mid_kernel) && precompute_mid) {
            cur_offset += 2 * n_active_feats;
            continue;
        }

        // in_buffer_activated (i, c) holds the dense input features from gather
        // for i = n_active_feats (# of features in the activated kernel from
        // neighbor_offset) out_buffer_activated (i, o) holds the dense output
        // features to scatter
        torch::Tensor out_buffer_activated;
        torch::Tensor in_buffer_activated;
        if (is_half) {
            out_buffer_activated = torch::from_blob(
                out_buffer.data_ptr<at::Half>(), {n_active_feats, n_out_channels}, options);
            in_buffer_activated = torch::from_blob(
                in_buffer.data_ptr<at::Half>(), {n_active_feats, n_in_channels}, options);
        } else {
            out_buffer_activated =
                torch::from_blob(out_buffer.data_ptr(), {n_active_feats, n_out_channels}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr(), {n_active_feats, n_in_channels}, options);
        }

        // gather n_active_feats dense features from N sparse input features with c
        // feature dimensions
        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "convolution_forward_cuda",
                       AT_WRAP([&] {
                           gatherKernelRegular<scalar_t>
                               <<<GET_BLOCKS(n_active_feats * n_in_channels, DEFAULT_BLOCK_DIM),
                                  DEFAULT_BLOCK_DIM>>>(n_active_feats,
                                                       n_in_feats,
                                                       n_in_channels,
                                                       in_feat.data_ptr<scalar_t>(),
                                                       in_buffer_activated.data_ptr<scalar_t>(),
                                                       neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        // gemm: (i, c) X (c, o) = (i, o)
        torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);

        // scatter n_active_feats dense features into n_out_feats output features of
        // dimension n_out_channels
        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "convolution_forward_cuda",
                       AT_WRAP([&] {
                           scatterKernelRegular<scalar_t>
                               <<<GET_BLOCKS(n_active_feats * n_out_channels, DEFAULT_BLOCK_DIM),
                                  DEFAULT_BLOCK_DIM>>>(n_active_feats,
                                                       n_out_feats,
                                                       n_out_channels,
                                                       out_buffer_activated.data_ptr<scalar_t>(),
                                                       out_feat.data_ptr<scalar_t>(),
                                                       neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        cur_offset += 2 * n_active_feats;
    }
}

template <>
void
dispatchGatherScatterGrad<torch::kCUDA>(torch::Tensor in_feat,
                                        torch::Tensor grad_in_feat,
                                        torch::Tensor grad_out_feat,
                                        torch::Tensor kernel,
                                        torch::Tensor grad_kernel,
                                        torch::Tensor neighbor_map,
                                        torch::Tensor neighbor_offset) {
    TORCH_CHECK(in_feat.device().is_cuda(), "in_feat must be a CUDA tensor");
    TORCH_CHECK(in_feat.device().has_index(), "in_feat must have a device index");
    TORCH_CHECK(in_feat.device() == grad_in_feat.device(),
                "All tensors must be on the same device");
    TORCH_CHECK(in_feat.device() == grad_out_feat.device(),
                "All tensors must be on the same device");
    TORCH_CHECK(in_feat.device() == kernel.device(), "All tensors must be on the same device");
    TORCH_CHECK(in_feat.device() == grad_kernel.device(), "All tensors must be on the same device");
    TORCH_CHECK(in_feat.device() == neighbor_map.device(),
                "All tensors must be on the same device");
    TORCH_CHECK(neighbor_offset.device().is_cpu(),
                "neighborhood_offset must be on the CPU because torch_sparse conv is wack");

    c10::cuda::CUDAGuard deviceGuard(in_feat.device());

    grad_in_feat.resize_as_(in_feat);
    grad_in_feat.zero_();
    grad_kernel.resize_as_(kernel);
    grad_kernel.zero_();

    bool is_half       = in_feat.scalar_type() == at::ScalarType::Half;
    bool is_bfloat16   = in_feat.scalar_type() == at::ScalarType::BFloat16;
    int n_in_feats     = in_feat.size(0);
    int n_in_channels  = in_feat.size(1);
    int n_out_feats    = grad_out_feat.size(0);
    int n_out_channels = kernel.size(-1);

    int kernel_volume = kernel.size(0);
    bool flag         = false;
    int in_buffer_size;
    in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                       neighbor_offset.data_ptr<int>() + kernel_volume);

    auto options         = torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer       = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto in_grad_buffer  = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto out_grad_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);

    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++) {
        auto kernel_grad_buffer = grad_kernel[i];
        int n_active_feats      = neighbor_offset.data_ptr<int>()[i];
        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * n_active_feats;
            continue;
        }

        if (n_active_feats == 0) {
            continue;
        }

        // Can't figure out a cleaner way to do this
        torch::Tensor out_grad_buffer_activated;
        torch::Tensor in_grad_buffer_activated;
        torch::Tensor in_buffer_activated;
        if (is_half) {
            out_grad_buffer_activated = torch::from_blob(
                out_grad_buffer.data_ptr<at::Half>(), {n_active_feats, kernel.size(2)}, options);
            in_grad_buffer_activated = torch::from_blob(
                in_grad_buffer.data_ptr<at::Half>(), {n_active_feats, in_feat.size(1)}, options);
            in_buffer_activated = torch::from_blob(
                in_buffer.data_ptr<at::Half>(), {n_active_feats, in_feat.size(1)}, options);
        } else if (is_bfloat16) {
            out_grad_buffer_activated = torch::from_blob(out_grad_buffer.data_ptr<at::BFloat16>(),
                                                         {n_active_feats, kernel.size(2)},
                                                         options);
            in_grad_buffer_activated  = torch::from_blob(in_grad_buffer.data_ptr<at::BFloat16>(),
                                                         {n_active_feats, in_feat.size(1)},
                                                        options);
            in_buffer_activated       = torch::from_blob(
                in_buffer.data_ptr<at::BFloat16>(), {n_active_feats, in_feat.size(1)}, options);
        } else {
            out_grad_buffer_activated = torch::from_blob(
                out_grad_buffer.data_ptr(), {n_active_feats, kernel.size(2)}, options);
            in_grad_buffer_activated = torch::from_blob(
                in_grad_buffer.data_ptr(), {n_active_feats, in_feat.size(1)}, options);
            in_buffer_activated =
                torch::from_blob(in_buffer.data_ptr(), {n_active_feats, in_feat.size(1)}, options);
        }

        // gather
        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "convolution_forward_cuda",
                       AT_WRAP([&] {
                           gatherKernelTransposed<scalar_t>
                               <<<ceil((double)(n_active_feats * n_out_channels) / 256), 256>>>(
                                   n_active_feats,
                                   n_out_feats,
                                   n_out_channels,
                                   grad_out_feat.data_ptr<scalar_t>(),
                                   out_grad_buffer_activated.data_ptr<scalar_t>(),
                                   neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "convolution_forward_cuda",
                       AT_WRAP([&] {
                           gatherKernelRegular<scalar_t>
                               <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                                   n_active_feats,
                                   n_in_feats,
                                   n_in_channels,
                                   in_feat.data_ptr<scalar_t>(),
                                   in_buffer_activated.data_ptr<scalar_t>(),
                                   neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        // gemm
        torch::mm_out(
            in_grad_buffer_activated, out_grad_buffer_activated, torch::transpose(kernel[i], 0, 1));
        torch::mm_out(kernel_grad_buffer,
                      torch::transpose(in_buffer_activated, 0, 1),
                      out_grad_buffer_activated);

        // scatter
        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "convolution_forward_cuda",
                       AT_WRAP([&] {
                           scatterKernelTransposed<scalar_t>
                               <<<ceil((double)(n_active_feats * n_in_channels) / 256), 256>>>(
                                   n_active_feats,
                                   n_in_feats,
                                   n_in_channels,
                                   in_grad_buffer_activated.data_ptr<scalar_t>(),
                                   grad_in_feat.data_ptr<scalar_t>(),
                                   neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        cur_offset += 2 * n_active_feats;
    }
}

template <>
void
GatherScatterOp::operator()<CpuTag>(CpuTag,
                                    torch::Tensor in_feat,
                                    torch::Tensor out_feat,
                                    torch::Tensor kernel,
                                    torch::Tensor neighbor_map,
                                    torch::Tensor neighbor_offset,
                                    bool middleAcceleration) const {
    if (in_feat.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    int out_nrows = out_feat.size(0);
    out_feat.resize_({out_nrows, kernel.size(2)});
    out_feat.zero_();

    int kernel_volume  = kernel.size(0);
    int in_buffer_size = 1;
    bool flag          = false;
    // memory optimization
    if (kernel_volume % 2 && out_nrows == in_feat.size(0) && middleAcceleration) {
        flag           = true;
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume / 2);
        in_buffer_size =
            std::max(in_buffer_size,
                     *std::max_element(neighbor_offset.data_ptr<int>() + kernel_volume / 2 + 1,
                                       neighbor_offset.data_ptr<int>() + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        torch::mm_out(out_feat, in_feat, kernel[kernel_volume / 2]);
    } else {
        in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                           neighbor_offset.data_ptr<int>() + kernel_volume);
    }

    auto options    = torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer  = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto out_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);
    int cur_offset  = 0;
    for (int i = 0; i < kernel_volume; i++) {
        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
            continue;
        }

        if (neighbor_offset.data_ptr<int>()[i] == 0) {
            continue;
        }

        auto out_buffer_activated = torch::from_blob(
            out_buffer.data_ptr(), {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
        auto in_buffer_activated = torch::from_blob(
            in_buffer.data_ptr(), {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);

        // gather
        AT_DISPATCH_V2(in_feat.scalar_type(),
                       "gatherCpu",
                       AT_WRAP([&]() {
                           gatherCpuRegular(in_buffer_activated.size(0),
                                            in_feat.size(0),
                                            kernel.size(1),
                                            in_feat.data_ptr<scalar_t>(),
                                            in_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        // matmul
        torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);

        // scatter
        AT_DISPATCH_V2(out_feat.scalar_type(),
                       "scatterCpu",
                       AT_WRAP([&]() {
                           scatterCpuRegular(neighbor_offset.data_ptr<int>()[i],
                                             out_nrows,
                                             kernel.size(2),
                                             out_buffer_activated.data_ptr<scalar_t>(),
                                             out_feat.data_ptr<scalar_t>(),
                                             neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);
        cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
    }
}

template <>
void
dispatchGatherScatterGrad<torch::kCPU>(torch::Tensor in_feat,
                                       torch::Tensor grad_in_feat,
                                       torch::Tensor grad_out_feat,
                                       torch::Tensor kernel,
                                       torch::Tensor grad_kernel,
                                       torch::Tensor neighbor_map,
                                       torch::Tensor neighbor_offset) {
    grad_in_feat.resize_as_(in_feat);
    grad_in_feat.zero_();
    grad_kernel.resize_as_(kernel);
    grad_kernel.zero_();

    int kernel_volume = kernel.size(0);
    bool flag         = false;
    int in_buffer_size;
    in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(),
                                       neighbor_offset.data_ptr<int>() + kernel_volume);

    auto options         = torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
    auto in_buffer       = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto in_grad_buffer  = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
    auto out_grad_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);

    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++) {
        auto kernel_grad_buffer = grad_kernel[i];
        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
            continue;
        }

        if (neighbor_offset.data_ptr<int>()[i] == 0) {
            continue;
        }

        auto out_grad_buffer_activated =
            torch::from_blob(out_grad_buffer.data_ptr(),
                             {neighbor_offset.data_ptr<int>()[i], kernel.size(2)},
                             options);
        auto in_grad_buffer_activated =
            torch::from_blob(in_grad_buffer.data_ptr(),
                             {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)},
                             options);
        auto in_buffer_activated = torch::from_blob(
            in_buffer.data_ptr(), {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);

        // gather
        AT_DISPATCH_V2(grad_out_feat.scalar_type(),
                       "gatherCpu",
                       AT_WRAP([&]() {
                           gatherCpuTransposed(out_grad_buffer_activated.size(0),
                                               grad_out_feat.size(0),
                                               kernel.size(2),
                                               grad_out_feat.data_ptr<scalar_t>(),
                                               out_grad_buffer_activated.data_ptr<scalar_t>(),
                                               neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);
        AT_DISPATCH_V2(grad_out_feat.scalar_type(),
                       "gatherCpu",
                       AT_WRAP([&]() {
                           gatherCpuRegular(in_buffer_activated.size(0),
                                            in_feat.size(0),
                                            kernel.size(1),
                                            in_feat.data_ptr<scalar_t>(),
                                            in_buffer_activated.data_ptr<scalar_t>(),
                                            neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        // matmul
        torch::mm_out(
            in_grad_buffer_activated, out_grad_buffer_activated, torch::transpose(kernel[i], 0, 1));
        torch::mm_out(kernel_grad_buffer,
                      torch::transpose(in_buffer_activated, 0, 1),
                      out_grad_buffer_activated);

        // scatter
        AT_DISPATCH_V2(grad_out_feat.scalar_type(),
                       "scatterCpu",
                       AT_WRAP([&]() {
                           scatterCpuTransposed(neighbor_offset.data_ptr<int>()[i],
                                                in_feat.size(0),
                                                kernel.size(1),
                                                in_grad_buffer_activated.data_ptr<scalar_t>(),
                                                grad_in_feat.data_ptr<scalar_t>(),
                                                neighbor_map.data_ptr<int>() + cur_offset);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES),
                       c10::kHalf,
                       c10::kBFloat16);

        cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
