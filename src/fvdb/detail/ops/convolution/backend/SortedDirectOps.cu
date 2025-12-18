// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// SortedDirectOps.cu
//
// Sorted Direct Sparse Convolution - CUDA Implementation
//
// Key design decisions:
// - One CUDA block per kernel position (K blocks total)
// - Kernel weights loaded into shared memory, reused for all edges in segment
// - Channel tiling to handle arbitrary c and o sizes
// - Atomic scatter to output (unavoidable for sparse conv)
// - No CPU loop - segment boundaries read from GPU memory

#include <fvdb/detail/ops/convolution/backend/SortedDirectOps.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {
namespace detail {
namespace ops {

// Default block dimension for the kernel
constexpr int SORTED_DIRECT_BLOCK_DIM = 256;

// Tile sizes for shared memory - adjust based on available smem
constexpr int TILE_C = 32; // Input channel tile
constexpr int TILE_O = 32; // Output channel tile

/// Sorted Direct Convolution Forward Kernel
///
/// Each block handles all edges for ONE kernel position.
/// Kernel weights are loaded into shared memory and reused for all edges.
/// Computation is tiled over input and output channels for large c, o.
///
template <typename scalar_t, int BLOCK_DIM, int TILE_C_SIZE, int TILE_O_SIZE>
__global__ void
sortedDirectConvForwardKernel(const scalar_t *__restrict__ in_feat,   // [N, c]
                              scalar_t *__restrict__ out_feat,        // [M, o]
                              const scalar_t *__restrict__ kernel,    // [K, c, o]
                              const int *__restrict__ in_indices,     // [P] = neighbor_map[:, 0]
                              const int *__restrict__ out_indices,    // [P] = neighbor_map[:, 1]
                              const int *__restrict__ segment_starts, // [K+1]
                              const int K,
                              const int c,
                              const int o,
                              const bool transpose) {
    // Each block handles one kernel position
    const int kernel_idx = blockIdx.x;
    if (kernel_idx >= K)
        return;

    // Get segment boundaries for this kernel position
    const int seg_start = segment_starts[kernel_idx];
    const int seg_end   = segment_starts[kernel_idx + 1];
    const int seg_size  = seg_end - seg_start;

    if (seg_size == 0)
        return;

    // Pointer to this kernel position's weights
    const scalar_t *k_weights = kernel + kernel_idx * c * o;

    // Shared memory for kernel weight tile
    extern __shared__ char smem[];
    scalar_t *shared_kernel = reinterpret_cast<scalar_t *>(smem);

    // Determine which index is source vs destination based on transpose flag
    const int *src_indices = transpose ? out_indices : in_indices;
    const int *dst_indices = transpose ? in_indices : out_indices;

    // Tile over output channels
    for (int o_tile = 0; o_tile < o; o_tile += TILE_O_SIZE) {
        const int o_tile_end  = min(o_tile + TILE_O_SIZE, o);
        const int o_tile_size = o_tile_end - o_tile;

        // Tile over input channels
        for (int c_tile = 0; c_tile < c; c_tile += TILE_C_SIZE) {
            const int c_tile_end  = min(c_tile + TILE_C_SIZE, c);
            const int c_tile_size = c_tile_end - c_tile;

            // Collaboratively load kernel weight tile into shared memory
            // Layout: shared_kernel[ci][oi] for ci in [0, TILE_C_SIZE), oi in [0, TILE_O_SIZE)
            for (int i = threadIdx.x; i < TILE_C_SIZE * TILE_O_SIZE; i += BLOCK_DIM) {
                const int ci = i / TILE_O_SIZE;
                const int oi = i % TILE_O_SIZE;

                if (ci < c_tile_size && oi < o_tile_size) {
                    // k_weights layout: [c, o] row-major
                    shared_kernel[i] = k_weights[(c_tile + ci) * o + (o_tile + oi)];
                } else {
                    shared_kernel[i] = static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

            // Process edges: each thread handles one or more edges
            for (int edge = seg_start + threadIdx.x; edge < seg_end; edge += BLOCK_DIM) {
                const int src_idx = src_indices[edge];
                const int dst_idx = dst_indices[edge];

                // Compute partial dot products for this tile
                // For each output channel in tile, compute sum over input channel tile
                for (int oi = 0; oi < o_tile_size; ++oi) {
                    scalar_t sum = static_cast<scalar_t>(0);

                    for (int ci = 0; ci < c_tile_size; ++ci) {
                        const scalar_t in_val = in_feat[src_idx * c + c_tile + ci];
                        const scalar_t k_val  = shared_kernel[ci * TILE_O_SIZE + oi];
                        sum += in_val * k_val;
                    }

                    // Atomic add to output
                    atomicAdd(&out_feat[dst_idx * o + o_tile + oi], sum);
                }
            }
            __syncthreads(); // Ensure all threads done before loading next tile
        }
    }
}

/// Forward dispatch for CUDA
template <>
void
dispatchSortedDirectConv<torch::kCUDA>(torch::Tensor in_feat,
                                       torch::Tensor out_feat,
                                       torch::Tensor kernel,
                                       torch::Tensor neighbor_map,
                                       torch::Tensor segment_starts,
                                       bool transpose) {
    TORCH_CHECK(in_feat.device().is_cuda(), "in_feat must be a CUDA tensor");
    TORCH_CHECK(out_feat.device().is_cuda(), "out_feat must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(neighbor_map.device().is_cuda(), "neighbor_map must be a CUDA tensor");
    TORCH_CHECK(segment_starts.device().is_cuda(), "segment_starts must be a CUDA tensor");

    TORCH_CHECK(in_feat.is_contiguous(), "in_feat must be contiguous");
    TORCH_CHECK(out_feat.is_contiguous(), "out_feat must be contiguous");
    TORCH_CHECK(kernel.is_contiguous(), "kernel must be contiguous");
    TORCH_CHECK(neighbor_map.is_contiguous(), "neighbor_map must be contiguous");
    TORCH_CHECK(segment_starts.is_contiguous(), "segment_starts must be contiguous");

    c10::cuda::CUDAGuard device_guard(in_feat.device());

    const int K = kernel.size(0); // kernel volume
    const int c = kernel.size(1); // input channels
    const int o = kernel.size(2); // output channels

    // Verify dimensions
    TORCH_CHECK(in_feat.size(1) == c, "Input feature channels must match kernel input channels");
    TORCH_CHECK(out_feat.size(1) == o, "Output feature channels must match kernel output channels");
    TORCH_CHECK(neighbor_map.size(1) == 2, "neighbor_map must have shape [P, 2]");
    TORCH_CHECK(segment_starts.size(0) == K + 1, "segment_starts must have shape [K+1]");

    // Extract index pointers from neighbor_map
    // neighbor_map[:, 0] = in_indices, neighbor_map[:, 1] = out_indices
    auto in_indices  = neighbor_map.select(1, 0).contiguous();
    auto out_indices = neighbor_map.select(1, 1).contiguous();

    // Launch configuration: one block per kernel position
    dim3 grid(K);
    dim3 block(SORTED_DIRECT_BLOCK_DIM);
    size_t smem_size = TILE_C * TILE_O * sizeof(float); // Will be adjusted per dtype

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in_feat.scalar_type(),
        "sortedDirectConvForward",
        [&] {
            smem_size = TILE_C * TILE_O * sizeof(scalar_t);

            sortedDirectConvForwardKernel<scalar_t, SORTED_DIRECT_BLOCK_DIM, TILE_C, TILE_O>
                <<<grid, block, smem_size>>>(in_feat.data_ptr<scalar_t>(),
                                             out_feat.data_ptr<scalar_t>(),
                                             kernel.data_ptr<scalar_t>(),
                                             in_indices.data_ptr<int>(),
                                             out_indices.data_ptr<int>(),
                                             segment_starts.data_ptr<int>(),
                                             K,
                                             c,
                                             o,
                                             transpose);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Backward kernel for gradient w.r.t. input features
///
/// For d(Loss)/d(in_feat), we need to propagate gradients backward through:
///   out_feat[dst] += in_feat[src] @ kernel[k]
///
/// The gradient is:
///   d(Loss)/d(in_feat[src]) += d(Loss)/d(out_feat[dst]) @ kernel[k].T
///
template <typename scalar_t, int BLOCK_DIM, int TILE_C_SIZE, int TILE_O_SIZE>
__global__ void
sortedDirectConvBackwardInputKernel(const scalar_t *__restrict__ grad_out_feat, // [M, o]
                                    scalar_t *__restrict__ grad_in_feat,        // [N, c]
                                    const scalar_t *__restrict__ kernel,        // [K, c, o]
                                    const int *__restrict__ in_indices,         // [P]
                                    const int *__restrict__ out_indices,        // [P]
                                    const int *__restrict__ segment_starts,     // [K+1]
                                    const int K,
                                    const int c,
                                    const int o,
                                    const bool transpose) {
    const int kernel_idx = blockIdx.x;
    if (kernel_idx >= K)
        return;

    const int seg_start = segment_starts[kernel_idx];
    const int seg_end   = segment_starts[kernel_idx + 1];
    const int seg_size  = seg_end - seg_start;

    if (seg_size == 0)
        return;

    const scalar_t *k_weights = kernel + kernel_idx * c * o;

    extern __shared__ char smem[];
    scalar_t *shared_kernel = reinterpret_cast<scalar_t *>(smem);

    // For backward pass, roles are swapped: grad flows from out to in
    const int *grad_src_indices = transpose ? in_indices : out_indices;
    const int *grad_dst_indices = transpose ? out_indices : in_indices;

    // Tile over input channels (destination for this backward pass)
    for (int c_tile = 0; c_tile < c; c_tile += TILE_C_SIZE) {
        const int c_tile_end  = min(c_tile + TILE_C_SIZE, c);
        const int c_tile_size = c_tile_end - c_tile;

        // Tile over output channels (source for this backward pass)
        for (int o_tile = 0; o_tile < o; o_tile += TILE_O_SIZE) {
            const int o_tile_end  = min(o_tile + TILE_O_SIZE, o);
            const int o_tile_size = o_tile_end - o_tile;

            // Load kernel tile (transposed access pattern for backward)
            for (int i = threadIdx.x; i < TILE_C_SIZE * TILE_O_SIZE; i += BLOCK_DIM) {
                const int ci = i / TILE_O_SIZE;
                const int oi = i % TILE_O_SIZE;

                if (ci < c_tile_size && oi < o_tile_size) {
                    shared_kernel[i] = k_weights[(c_tile + ci) * o + (o_tile + oi)];
                } else {
                    shared_kernel[i] = static_cast<scalar_t>(0);
                }
            }
            __syncthreads();

            // Process edges
            for (int edge = seg_start + threadIdx.x; edge < seg_end; edge += BLOCK_DIM) {
                const int grad_src_idx = grad_src_indices[edge]; // Where grad comes from
                const int grad_dst_idx = grad_dst_indices[edge]; // Where grad goes to

                // For each input channel, sum over output channels
                for (int ci = 0; ci < c_tile_size; ++ci) {
                    scalar_t sum = static_cast<scalar_t>(0);

                    for (int oi = 0; oi < o_tile_size; ++oi) {
                        const scalar_t grad_val = grad_out_feat[grad_src_idx * o + o_tile + oi];
                        const scalar_t k_val    = shared_kernel[ci * TILE_O_SIZE + oi];
                        sum += grad_val * k_val;
                    }

                    atomicAdd(&grad_in_feat[grad_dst_idx * c + c_tile + ci], sum);
                }
            }
            __syncthreads();
        }
    }
}

/// Backward kernel for gradient w.r.t. kernel weights
///
/// For d(Loss)/d(kernel[k]), the gradient is:
///   d(Loss)/d(kernel[k]) = sum over edges in segment k of:
///       in_feat[src].T @ d(Loss)/d(out_feat[dst])
///
template <typename scalar_t, int BLOCK_DIM>
__global__ void
sortedDirectConvBackwardKernelWeightsKernel(const scalar_t *__restrict__ in_feat,       // [N, c]
                                            const scalar_t *__restrict__ grad_out_feat, // [M, o]
                                            scalar_t *__restrict__ grad_kernel,         // [K, c, o]
                                            const int *__restrict__ in_indices,         // [P]
                                            const int *__restrict__ out_indices,        // [P]
                                            const int *__restrict__ segment_starts,     // [K+1]
                                            const int K,
                                            const int c,
                                            const int o,
                                            const bool transpose) {
    const int kernel_idx = blockIdx.x;
    if (kernel_idx >= K)
        return;

    const int seg_start = segment_starts[kernel_idx];
    const int seg_end   = segment_starts[kernel_idx + 1];
    const int seg_size  = seg_end - seg_start;

    if (seg_size == 0)
        return;

    scalar_t *k_grad = grad_kernel + kernel_idx * c * o;

    const int *src_indices = transpose ? out_indices : in_indices;
    const int *dst_indices = transpose ? in_indices : out_indices;

    // Each thread computes gradient for a subset of (c, o) pairs
    // This is a reduction over edges
    for (int weight_idx = threadIdx.x; weight_idx < c * o; weight_idx += BLOCK_DIM) {
        const int ci = weight_idx / o;
        const int oi = weight_idx % o;

        scalar_t grad_sum = static_cast<scalar_t>(0);

        for (int edge = seg_start; edge < seg_end; ++edge) {
            const int src_idx = src_indices[edge];
            const int dst_idx = dst_indices[edge];

            const scalar_t in_val   = in_feat[src_idx * c + ci];
            const scalar_t grad_val = grad_out_feat[dst_idx * o + oi];

            grad_sum += in_val * grad_val;
        }

        // Accumulate gradient (kernel gradients don't conflict across blocks)
        k_grad[ci * o + oi] += grad_sum;
    }
}

/// Backward dispatch for CUDA
template <>
void
dispatchSortedDirectConvGrad<torch::kCUDA>(torch::Tensor in_feat,
                                           torch::Tensor grad_in_feat,
                                           torch::Tensor grad_out_feat,
                                           torch::Tensor kernel,
                                           torch::Tensor grad_kernel,
                                           torch::Tensor neighbor_map,
                                           torch::Tensor segment_starts,
                                           bool transpose) {
    TORCH_CHECK(in_feat.device().is_cuda(), "in_feat must be a CUDA tensor");
    TORCH_CHECK(grad_out_feat.device().is_cuda(), "grad_out_feat must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "kernel must be a CUDA tensor");
    TORCH_CHECK(neighbor_map.device().is_cuda(), "neighbor_map must be a CUDA tensor");
    TORCH_CHECK(segment_starts.device().is_cuda(), "segment_starts must be a CUDA tensor");

    c10::cuda::CUDAGuard device_guard(in_feat.device());

    const int K = kernel.size(0);
    const int c = kernel.size(1);
    const int o = kernel.size(2);
    const int N = in_feat.size(0);

    // Initialize gradients to zero
    grad_in_feat.resize_as_(in_feat);
    grad_in_feat.zero_();
    grad_kernel.resize_as_(kernel);
    grad_kernel.zero_();

    auto in_indices  = neighbor_map.select(1, 0).contiguous();
    auto out_indices = neighbor_map.select(1, 1).contiguous();

    dim3 grid(K);
    dim3 block(SORTED_DIRECT_BLOCK_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in_feat.scalar_type(),
        "sortedDirectConvBackward",
        [&] {
            size_t smem_size = TILE_C * TILE_O * sizeof(scalar_t);

            // Backward pass for input gradients
            sortedDirectConvBackwardInputKernel<scalar_t, SORTED_DIRECT_BLOCK_DIM, TILE_C, TILE_O>
                <<<grid, block, smem_size>>>(grad_out_feat.data_ptr<scalar_t>(),
                                             grad_in_feat.data_ptr<scalar_t>(),
                                             kernel.data_ptr<scalar_t>(),
                                             in_indices.data_ptr<int>(),
                                             out_indices.data_ptr<int>(),
                                             segment_starts.data_ptr<int>(),
                                             K,
                                             c,
                                             o,
                                             transpose);

            // Backward pass for kernel gradients
            sortedDirectConvBackwardKernelWeightsKernel<scalar_t, SORTED_DIRECT_BLOCK_DIM>
                <<<grid, block>>>(in_feat.data_ptr<scalar_t>(),
                                  grad_out_feat.data_ptr<scalar_t>(),
                                  grad_kernel.data_ptr<scalar_t>(),
                                  in_indices.data_ptr<int>(),
                                  out_indices.data_ptr<int>(),
                                  segment_starts.data_ptr<int>(),
                                  K,
                                  c,
                                  o,
                                  transpose);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// CPU fallback - uses the Python-style loop internally
template <>
void
dispatchSortedDirectConv<torch::kCPU>(torch::Tensor in_feat,
                                      torch::Tensor out_feat,
                                      torch::Tensor kernel,
                                      torch::Tensor neighbor_map,
                                      torch::Tensor segment_starts,
                                      bool transpose) {
    const int K = kernel.size(0);
    const int c = kernel.size(1);
    const int o = kernel.size(2);

    // CPU implementation following the Python fallback pattern
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in_feat.scalar_type(),
        "sortedDirectConvCPU",
        [&] {
            const scalar_t *in_ptr = in_feat.data_ptr<scalar_t>();
            scalar_t *out_ptr      = out_feat.data_ptr<scalar_t>();
            const scalar_t *k_ptr  = kernel.data_ptr<scalar_t>();
            const int *in_idx_ptr  = neighbor_map.data_ptr<int>();
            const int *out_idx_ptr = neighbor_map.data_ptr<int>() + neighbor_map.size(0);
            const int *seg_ptr     = segment_starts.data_ptr<int>();

            for (int k = 0; k < K; ++k) {
                const int seg_start = seg_ptr[k];
                const int seg_end   = seg_ptr[k + 1];
                const int n_k       = seg_end - seg_start;

                if (n_k == 0)
                    continue;

                const scalar_t *k_weights = k_ptr + k * c * o;

#pragma omp parallel for
                for (int e = 0; e < n_k; ++e) {
                    const int edge    = seg_start + e;
                    const int src_idx = transpose ? out_idx_ptr[edge] : in_idx_ptr[edge];
                    const int dst_idx = transpose ? in_idx_ptr[edge] : out_idx_ptr[edge];

                    for (int oi = 0; oi < o; ++oi) {
                        scalar_t sum = 0;
                        for (int ci = 0; ci < c; ++ci) {
                            sum += in_ptr[src_idx * c + ci] * k_weights[ci * o + oi];
                        }
#pragma omp atomic
                        out_ptr[dst_idx * o + oi] += sum;
                    }
                }
            }
        });
}

/// CPU backward fallback
template <>
void
dispatchSortedDirectConvGrad<torch::kCPU>(torch::Tensor in_feat,
                                          torch::Tensor grad_in_feat,
                                          torch::Tensor grad_out_feat,
                                          torch::Tensor kernel,
                                          torch::Tensor grad_kernel,
                                          torch::Tensor neighbor_map,
                                          torch::Tensor segment_starts,
                                          bool transpose) {
    grad_in_feat.resize_as_(in_feat);
    grad_in_feat.zero_();
    grad_kernel.resize_as_(kernel);
    grad_kernel.zero_();

    const int K = kernel.size(0);
    const int c = kernel.size(1);
    const int o = kernel.size(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        in_feat.scalar_type(),
        "sortedDirectConvGradCPU",
        [&] {
            const scalar_t *in_ptr       = in_feat.data_ptr<scalar_t>();
            scalar_t *grad_in_ptr        = grad_in_feat.data_ptr<scalar_t>();
            const scalar_t *grad_out_ptr = grad_out_feat.data_ptr<scalar_t>();
            const scalar_t *k_ptr        = kernel.data_ptr<scalar_t>();
            scalar_t *grad_k_ptr         = grad_kernel.data_ptr<scalar_t>();
            const int *in_idx_ptr        = neighbor_map.data_ptr<int>();
            const int *out_idx_ptr       = neighbor_map.data_ptr<int>() + neighbor_map.size(0);
            const int *seg_ptr           = segment_starts.data_ptr<int>();

            for (int k = 0; k < K; ++k) {
                const int seg_start = seg_ptr[k];
                const int seg_end   = seg_ptr[k + 1];
                const int n_k       = seg_end - seg_start;

                if (n_k == 0)
                    continue;

                const scalar_t *k_weights = k_ptr + k * c * o;
                scalar_t *k_grad          = grad_k_ptr + k * c * o;

                // Gradient w.r.t. input features
                for (int e = 0; e < n_k; ++e) {
                    const int edge    = seg_start + e;
                    const int src_idx = transpose ? in_idx_ptr[edge] : out_idx_ptr[edge];
                    const int dst_idx = transpose ? out_idx_ptr[edge] : in_idx_ptr[edge];

                    for (int ci = 0; ci < c; ++ci) {
                        scalar_t sum = 0;
                        for (int oi = 0; oi < o; ++oi) {
                            sum += grad_out_ptr[src_idx * o + oi] * k_weights[ci * o + oi];
                        }
#pragma omp atomic
                        grad_in_ptr[dst_idx * c + ci] += sum;
                    }
                }

                // Gradient w.r.t. kernel weights
                for (int e = 0; e < n_k; ++e) {
                    const int edge    = seg_start + e;
                    const int src_idx = transpose ? out_idx_ptr[edge] : in_idx_ptr[edge];
                    const int dst_idx = transpose ? in_idx_ptr[edge] : out_idx_ptr[edge];

                    for (int ci = 0; ci < c; ++ci) {
                        for (int oi = 0; oi < o; ++oi) {
                            k_grad[ci * o + oi] +=
                                in_ptr[src_idx * c + ci] * grad_out_ptr[dst_idx * o + oi];
                        }
                    }
                }
            }
        });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
