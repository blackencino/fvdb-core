// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// PredGatherIGemmBackward.cu -- Leaf-local no-kmap sparse convolution backward.
//
// Backward pass for the PredGatherIGemm forward backend. Unlike the existing
// GatherScatterDefault fallback, this implementation does not require a global
// k-map build. Instead, it walks the NanoVDB topology directly inside CUDA
// kernels, using one output leaf as the locality unit and atomically
// accumulating:
//
//   dX[i, c]                  += dY[o, k] * W[k, c, t, r, s]
//   dW[k, c, t, r, s]         += X[i, c]  * dY[o, k]
//
// where i is the input voxel reached from output voxel o by kernel offset
// (t, r, s). The implementation is specialized to the same constraints as
// PredGatherIGemm forward:
//   - CUDA only
//   - float32 only
//   - input / output channels must be multiples of 32
//   - uniform kernel sizes 3 / 5 / 7
//   - uniform strides 1 / 2
//   - batch size 1
//
// The algorithm is intentionally leaf-local and dense-tile oriented, so its
// sweet spot matches the forward kernel: leaves that are moderately to highly
// active. It avoids any global pre-pass or compacted kernel map construction.
//
// ============================================================================
// PERFORMANCE STATUS AND OPTIMIZATION ROADMAP
// ============================================================================
//
// This implementation is functionally correct (validated by adjoint identity
// tests, cross-backend comparison with GatherScatterDefault, and autograd
// integration tests) but has severe performance problems. Benchmarks on A100
// (Cin=64, Cout=128, kernel 3x3x3, stride 1):
//
//   Config              IGEMM bwd   GS bwd (no topo)  Ratio        IGEMM fwd
//   1M dense  (75%)     153 ms      48 ms             3.2x slower  5.3 ms
//   2M dense  (75%)     307 ms      115 ms            2.7x slower  10.2 ms
//   4M sparse (25%)     590 ms      23 ms             25x slower   24 ms
//   8M sparse (10%)     1141 ms     8 ms              143x slower  41 ms
//
// The backward is 29x slower than the forward for the 1M dense case, which
// is physically unreasonable -- a well-written backward should be 2-3x the
// forward cost, not 29x.
//
// ROOT CAUSE: Scalar FP32 GEMM
//
//   The GEMM inner loops (dgrad: lines ~235-246, wgrad: lines ~377-387) use
//   scalar `for (kk)` multiply-accumulate instead of tensor core MMA. The
//   forward uses SM80 TF32 tensor cores via CUTLASS (SM80_16x8x8_F32TF32TF32
//   F32_TN with 2x2 tiling). GatherScatterDefault backward calls cuBLAS
//   torch::mm which also uses tensor cores. On A100, FP32 scalar peak is
//   ~19.5 TFLOPS while TF32 tensor core peak is ~156 TFLOPS -- an 8x
//   theoretical gap that directly maps to the observed cost.
//
// SECONDARY BOTTLENECKS:
//
//   1. Tiny GEMM tiles: each (cluster, kernel_offset, k_tile) does a
//      [64,32] x [32,32] GEMM = 65K FMAs. cuBLAS in GS processes the entire
//      active-pair set (100K+ rows) per kernel offset -- much higher
//      arithmetic intensity and utilization.
//
//   2. Sequential cluster x kernel_offset loop: each dgrad CTA processes
//      8 clusters x kernel_volume offsets sequentially (e.g. 216 iterations
//      for 3x3x3). Creates long-running CTAs with poor tail effects at low
//      leaf occupancy.
//
//   3. Per-offset NanoVDB tree probes: for each kernel offset within each
//      cluster, act_tree.getValue(probe) is called for all 64 voxels. The
//      forward avoids this by loading the entire halo region once per
//      cluster into shared memory.
//
//   4. wgrad atomic contention: grid is (leaves, C/32, K/32), so ALL leaf
//      CTAs for the same (c_tile, k_tile, kernel_offset) atomically reduce
//      into the same weight element. For 2048+ leaves this is severe and
//      explains the super-linear performance degradation with leaf count.
//
// RECOMMENDED OPTIMIZATION TIERS:
//
//   Tier 1 (highest impact, ~5-8x improvement expected):
//     Replace scalar GEMM with CuTe MMA atoms. Use SM80_16x8x8_F32TF32TF32
//     F32_TN with 2x2 tiling (matching the forward). Requires restructuring
//     shared memory with swizzled layouts for LDSM, loading data into MMA
//     fragments via smem-to-register copy atoms, and replacing the for(kk)
//     loop with cute::gemm(tiled_mma, ...). Note: this changes backward
//     precision from FP32 to TF32, matching the forward and cuBLAS behavior.
//
//   Tier 2 (moderate impact):
//     Pre-load the full halo region into shared memory once per cluster
//     (like the forward does), then index into it for each kernel offset.
//     Replaces kernel_volume x 64 tree traversals per cluster with one bulk
//     load of CHx x CHy x CHz voxels (e.g. 6x6x6=216 for ks=3).
//
//   Tier 3 (moderate impact):
//     Restructure wgrad to reduce atomic contention. Options: workspace
//     buffer for per-leaf partial weight gradients with a final reduction
//     kernel, or increase the spatial tile per wgrad CTA.
//
//   Tier 4 (long-term ideal, largest scope):
//     Adapt the forward's MainloopSm80CpAsyncPredGatherB for the transposed
//     GEMM directions. Gives pipelined cp.async loads + tensor core compute
//     matching the forward architecture exactly.
// ============================================================================

#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/PredGatherIGemmBackward.h>

#include <nanovdb/NanoVDB.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <tuple>

namespace fvdb {
namespace detail {
namespace ops {
namespace pred_gather_igemm_backward {

constexpr int kLeafX      = 8;
constexpr int kLeafY      = 8;
constexpr int kLeafZ      = 8;
constexpr int kLeafVoxels = kLeafX * kLeafY * kLeafZ; // 512

constexpr int kClusterX      = 4;
constexpr int kClusterY      = 4;
constexpr int kClusterZ      = 4;
constexpr int kClusterVoxels = kClusterX * kClusterY * kClusterZ;                            // 64

constexpr int kClustersPerLeafX = kLeafX / kClusterX;                                        // 2
constexpr int kClustersPerLeafY = kLeafY / kClusterY;                                        // 2
constexpr int kClustersPerLeafZ = kLeafZ / kClusterZ;                                        // 2
constexpr int kClustersPerLeaf  = kClustersPerLeafX * kClustersPerLeafY * kClustersPerLeafZ; // 8

constexpr int kTileC        = 32;
constexpr int kTileK        = 32;
constexpr int kBlockThreads = 256;
__device__ __forceinline__ int
leafLinearIndex(int x, int y, int z) {
    return x * 64 + y * 8 + z;
}

__device__ __forceinline__ void
clusterIdToCoord(int cluster_id, int &cx, int &cy, int &cz) {
    constexpr int yz = kClustersPerLeafY * kClustersPerLeafZ; // 4
    cx               = cluster_id / yz;
    int rem          = cluster_id % yz;
    cy               = rem / kClustersPerLeafZ;
    cz               = rem % kClustersPerLeafZ;
}

__device__ __forceinline__ void
clusterLinearToCoord(int n, int &x, int &y, int &z) {
    x       = n / (kClusterY * kClusterZ); // /16
    int rem = n % (kClusterY * kClusterZ);
    y       = rem / kClusterZ;             // /4
    z       = rem % kClusterZ;
}

template <int KernelSize>
__device__ __forceinline__ void
kernelOffsetToCoord(int kernel_offset, int &t, int &r, int &s) {
    constexpr int ks2 = KernelSize * KernelSize;
    t                 = kernel_offset / ks2;
    int rem           = kernel_offset % ks2;
    r                 = rem / KernelSize;
    s                 = rem % KernelSize;
}

template <int KernelSize>
__device__ __forceinline__ int64_t
weightIndex(int oc, int ic, int t, int r, int s, int C) {
    return (
        ((((static_cast<int64_t>(oc) * C + ic) * KernelSize + t) * KernelSize + r) * KernelSize) +
        s);
}

struct DgradSharedStorage {
    uint64_t out_leaf_idx[kLeafVoxels];
    uint64_t out_cluster_rows[kClusterVoxels];
    uint64_t in_cluster_rows[kClusterVoxels];
    float grad_out_tile[kClusterVoxels][kTileK];
    float weight_tile[kTileC][kTileK + 1];
    float accum_tile[kClusterVoxels][kTileC];
};

struct WgradSharedStorage {
    uint64_t out_leaf_idx[kLeafVoxels];
    uint64_t out_cluster_rows[kClusterVoxels];
    uint64_t in_cluster_rows[kClusterVoxels];
    float feat_tile[kClusterVoxels][kTileC];
    float grad_out_tile[kClusterVoxels][kTileK];
    float accum_tile[kTileC][kTileK];
};

template <int KernelSize, int Stride, typename BuildT>
__global__ void
predGatherIGemmDgradKernel(const nanovdb::NanoGrid<BuildT> *act_grid,
                           const nanovdb::NanoGrid<BuildT> *out_grid,
                           const float *__restrict__ grad_out,
                           const float *__restrict__ weights,
                           float *__restrict__ grad_features,
                           int C,
                           int K) {
    __shared__ DgradSharedStorage smem;

    int const leaf_id = static_cast<int>(blockIdx.x);
    int const c_tile  = static_cast<int>(blockIdx.y);
    int const c0      = c_tile * kTileC;

    if (c0 >= C) {
        return;
    }

    auto const &out_leaf = out_grid->tree().template getFirstNode<0>()[leaf_id];

    for (int v = threadIdx.x; v < kLeafVoxels; v += blockDim.x) {
        smem.out_leaf_idx[v] = out_leaf.getValue(v);
    }
    __syncthreads();

    auto const &act_tree = act_grid->tree();

    auto const out_leaf_origin = out_leaf.origin();
    auto const act_leaf_origin = nanovdb::Coord(
        out_leaf_origin[0] * Stride, out_leaf_origin[1] * Stride, out_leaf_origin[2] * Stride);

    constexpr int radius = KernelSize / 2;

    for (int cluster_id = 0; cluster_id < kClustersPerLeaf; ++cluster_id) {
        int cx, cy, cz;
        clusterIdToCoord(cluster_id, cx, cy, cz);

        int const base_x = cx * kClusterX;
        int const base_y = cy * kClusterY;
        int const base_z = cz * kClusterZ;

        // Cache the 64 output rows for this cluster once.
        for (int n = threadIdx.x; n < kClusterVoxels; n += blockDim.x) {
            int lx, ly, lz;
            clusterLinearToCoord(n, lx, ly, lz);

            int const x = base_x + lx;
            int const y = base_y + ly;
            int const z = base_z + lz;

            smem.out_cluster_rows[n] = smem.out_leaf_idx[leafLinearIndex(x, y, z)];
        }
        __syncthreads();

        constexpr int kernel_volume = KernelSize * KernelSize * KernelSize;
        for (int kernel_offset = 0; kernel_offset < kernel_volume; ++kernel_offset) {
            int t, r, s;
            kernelOffsetToCoord<KernelSize>(kernel_offset, t, r, s);

            // Build the current 64-entry input row indirection vector.
            for (int n = threadIdx.x; n < kClusterVoxels; n += blockDim.x) {
                uint64_t in_row = 0;

                if (smem.out_cluster_rows[n] != 0) {
                    int lx, ly, lz;
                    clusterLinearToCoord(n, lx, ly, lz);

                    int const x = base_x + lx;
                    int const y = base_y + ly;
                    int const z = base_z + lz;

                    nanovdb::Coord probe = act_leaf_origin.offsetBy(x * Stride + (t - radius),
                                                                    y * Stride + (r - radius),
                                                                    z * Stride + (s - radius));

                    in_row = act_tree.getValue(probe);
                }

                smem.in_cluster_rows[n] = in_row;
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < kClusterVoxels * kTileC; idx += blockDim.x) {
                int const n            = idx / kTileC;
                int const ic           = idx % kTileC;
                smem.accum_tile[n][ic] = 0.0f;
            }
            __syncthreads();

            for (int k0 = 0; k0 < K; k0 += kTileK) {
                // Load one [32 x 32] weight tile W[k0:k0+32, c0:c0+32, t, r, s].
                for (int idx = threadIdx.x; idx < kTileC * kTileK; idx += blockDim.x) {
                    int const ic_local = idx / kTileK;
                    int const oc_local = idx % kTileK;
                    int const ic       = c0 + ic_local;
                    int const oc       = k0 + oc_local;

                    smem.weight_tile[ic_local][oc_local] =
                        weights[weightIndex<KernelSize>(oc, ic, t, r, s, C)];
                }

                // Load one [64 x 32] grad-output tile for this cluster.
                for (int idx = threadIdx.x; idx < kClusterVoxels * kTileK; idx += blockDim.x) {
                    int const n        = idx / kTileK;
                    int const oc_local = idx % kTileK;
                    uint64_t const row = smem.out_cluster_rows[n];

                    smem.grad_out_tile[n][oc_local] =
                        row ? grad_out[static_cast<int64_t>(row - 1) * K + (k0 + oc_local)] : 0.0f;
                }
                __syncthreads();

                // accum[n, ic] += sum_k grad_out[n, k] * W[k, ic]
                for (int idx = threadIdx.x; idx < kClusterVoxels * kTileC; idx += blockDim.x) {
                    int const n        = idx / kTileC;
                    int const ic_local = idx % kTileC;

                    float acc = smem.accum_tile[n][ic_local];
#pragma unroll
                    for (int kk = 0; kk < kTileK; ++kk) {
                        acc += smem.grad_out_tile[n][kk] * smem.weight_tile[ic_local][kk];
                    }
                    smem.accum_tile[n][ic_local] = acc;
                }
                __syncthreads();
            }

            // Atomic scatter-add into dX using the current 64-entry indirection vector.
            for (int idx = threadIdx.x; idx < kClusterVoxels * kTileC; idx += blockDim.x) {
                int const n        = idx / kTileC;
                int const ic_local = idx % kTileC;
                uint64_t const row = smem.in_cluster_rows[n];

                if (row != 0) {
                    atomicAdd(&grad_features[static_cast<int64_t>(row - 1) * C + (c0 + ic_local)],
                              smem.accum_tile[n][ic_local]);
                }
            }
            __syncthreads();
        }
    }
}

template <int KernelSize, int Stride, typename BuildT>
__global__ void
predGatherIGemmWgradKernel(const nanovdb::NanoGrid<BuildT> *act_grid,
                           const nanovdb::NanoGrid<BuildT> *out_grid,
                           const float *__restrict__ features,
                           const float *__restrict__ grad_out,
                           float *__restrict__ grad_weights,
                           int C,
                           int K) {
    __shared__ WgradSharedStorage smem;

    int const leaf_id = static_cast<int>(blockIdx.x);
    int const c_tile  = static_cast<int>(blockIdx.y);
    int const k_tile  = static_cast<int>(blockIdx.z);
    int const c0      = c_tile * kTileC;
    int const k0      = k_tile * kTileK;

    if (c0 >= C || k0 >= K) {
        return;
    }

    auto const &out_leaf = out_grid->tree().template getFirstNode<0>()[leaf_id];

    for (int v = threadIdx.x; v < kLeafVoxels; v += blockDim.x) {
        smem.out_leaf_idx[v] = out_leaf.getValue(v);
    }
    __syncthreads();

    auto const &act_tree = act_grid->tree();

    auto const out_leaf_origin = out_leaf.origin();
    auto const act_leaf_origin = nanovdb::Coord(
        out_leaf_origin[0] * Stride, out_leaf_origin[1] * Stride, out_leaf_origin[2] * Stride);

    constexpr int radius        = KernelSize / 2;
    constexpr int kernel_volume = KernelSize * KernelSize * KernelSize;

    for (int kernel_offset = 0; kernel_offset < kernel_volume; ++kernel_offset) {
        int t, r, s;
        kernelOffsetToCoord<KernelSize>(kernel_offset, t, r, s);

        for (int idx = threadIdx.x; idx < kTileC * kTileK; idx += blockDim.x) {
            int const ic_local                  = idx / kTileK;
            int const oc_local                  = idx % kTileK;
            smem.accum_tile[ic_local][oc_local] = 0.0f;
        }
        __syncthreads();

        for (int cluster_id = 0; cluster_id < kClustersPerLeaf; ++cluster_id) {
            int cx, cy, cz;
            clusterIdToCoord(cluster_id, cx, cy, cz);

            int const base_x = cx * kClusterX;
            int const base_y = cy * kClusterY;
            int const base_z = cz * kClusterZ;

            // Cache the 64 output rows for this cluster.
            for (int n = threadIdx.x; n < kClusterVoxels; n += blockDim.x) {
                int lx, ly, lz;
                clusterLinearToCoord(n, lx, ly, lz);

                int const x = base_x + lx;
                int const y = base_y + ly;
                int const z = base_z + lz;

                smem.out_cluster_rows[n] = smem.out_leaf_idx[leafLinearIndex(x, y, z)];
            }
            __syncthreads();

            // Build the 64-entry input row indirection vector for this offset.
            for (int n = threadIdx.x; n < kClusterVoxels; n += blockDim.x) {
                uint64_t in_row = 0;

                if (smem.out_cluster_rows[n] != 0) {
                    int lx, ly, lz;
                    clusterLinearToCoord(n, lx, ly, lz);

                    int const x = base_x + lx;
                    int const y = base_y + ly;
                    int const z = base_z + lz;

                    nanovdb::Coord probe = act_leaf_origin.offsetBy(x * Stride + (t - radius),
                                                                    y * Stride + (r - radius),
                                                                    z * Stride + (s - radius));

                    in_row = act_tree.getValue(probe);
                }

                smem.in_cluster_rows[n] = in_row;
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < kClusterVoxels * kTileC; idx += blockDim.x) {
                int const n        = idx / kTileC;
                int const ic_local = idx % kTileC;
                uint64_t const row = smem.in_cluster_rows[n];

                smem.feat_tile[n][ic_local] =
                    row ? features[static_cast<int64_t>(row - 1) * C + (c0 + ic_local)] : 0.0f;
            }

            for (int idx = threadIdx.x; idx < kClusterVoxels * kTileK; idx += blockDim.x) {
                int const n        = idx / kTileK;
                int const oc_local = idx % kTileK;
                uint64_t const row = smem.out_cluster_rows[n];

                smem.grad_out_tile[n][oc_local] =
                    row ? grad_out[static_cast<int64_t>(row - 1) * K + (k0 + oc_local)] : 0.0f;
            }
            __syncthreads();

            // accum[ic, oc] += sum_n X[n, ic] * dY[n, oc]
            for (int idx = threadIdx.x; idx < kTileC * kTileK; idx += blockDim.x) {
                int const ic_local = idx / kTileK;
                int const oc_local = idx % kTileK;

                float acc = smem.accum_tile[ic_local][oc_local];
#pragma unroll
                for (int n = 0; n < kClusterVoxels; ++n) {
                    acc += smem.feat_tile[n][ic_local] * smem.grad_out_tile[n][oc_local];
                }
                smem.accum_tile[ic_local][oc_local] = acc;
            }
            __syncthreads();
        }

        // Atomic reduction across leaves into dW.
        for (int idx = threadIdx.x; idx < kTileC * kTileK; idx += blockDim.x) {
            int const ic_local = idx / kTileK;
            int const oc_local = idx % kTileK;
            int const ic       = c0 + ic_local;
            int const oc       = k0 + oc_local;

            atomicAdd(&grad_weights[weightIndex<KernelSize>(oc, ic, t, r, s, C)],
                      smem.accum_tile[ic_local][oc_local]);
        }
        __syncthreads();
    }
}

template <int KernelSize, int Stride>
std::tuple<torch::Tensor, torch::Tensor>
launchPredGatherIGemmBackward(torch::Tensor grad_output,
                              torch::Tensor features,
                              torch::Tensor weights,
                              GridBatchImpl const &feature_grid,
                              GridBatchImpl const &output_grid) {
    auto *nano_input_grid =
        feature_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();
    auto *nano_output_grid =
        output_grid.nanoGridHandle().template deviceGrid<nanovdb::ValueOnIndex>();

    TORCH_CHECK(nano_input_grid != nullptr, "Failed to get device input grid");
    TORCH_CHECK(nano_output_grid != nullptr, "Failed to get device output grid");

    int64_t const N_in  = features.size(0);
    int64_t const N_out = grad_output.size(0);
    int64_t const C     = features.size(1);
    int64_t const K     = weights.size(0);

    auto opts = torch::dtype(torch::kFloat32).device(features.device());

    auto grad_features = torch::zeros({N_in, C}, opts);
    auto grad_weights  = torch::zeros_like(weights);

    uint32_t const output_leaf_count = output_grid.numLeavesAt(0);

    if (N_in == 0 || N_out == 0 || C == 0 || K == 0 || output_leaf_count == 0) {
        return {grad_features, grad_weights};
    }

    dim3 const block(kBlockThreads);
    dim3 const dgrad_grid(
        static_cast<unsigned int>(output_leaf_count), static_cast<unsigned int>(C / kTileC), 1u);
    dim3 const wgrad_grid(static_cast<unsigned int>(output_leaf_count),
                          static_cast<unsigned int>(C / kTileC),
                          static_cast<unsigned int>(K / kTileK));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    predGatherIGemmDgradKernel<KernelSize, Stride, nanovdb::ValueOnIndex>
        <<<dgrad_grid, block, 0, stream>>>(nano_input_grid,
                                           nano_output_grid,
                                           grad_output.data_ptr<float>(),
                                           weights.data_ptr<float>(),
                                           grad_features.data_ptr<float>(),
                                           static_cast<int>(C),
                                           static_cast<int>(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    predGatherIGemmWgradKernel<KernelSize, Stride, nanovdb::ValueOnIndex>
        <<<wgrad_grid, block, 0, stream>>>(nano_input_grid,
                                           nano_output_grid,
                                           features.data_ptr<float>(),
                                           grad_output.data_ptr<float>(),
                                           grad_weights.data_ptr<float>(),
                                           static_cast<int>(C),
                                           static_cast<int>(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_features, grad_weights};
}

static void
checkPredGatherIGemmBackwardPreconditions(torch::Tensor grad_output,
                                          torch::Tensor features,
                                          torch::Tensor weights,
                                          GridBatchImpl const &feature_grid,
                                          GridBatchImpl const &output_grid,
                                          int kernel_size,
                                          int stride) {
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    TORCH_CHECK(features.scalar_type() == torch::kFloat32, "features must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");

    TORCH_CHECK(features.dim() == 2, "features must be 2-D [N_in, C]");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2-D [N_out, K]");
    TORCH_CHECK(weights.dim() == 5, "weights must be 5-D [K, C, T, R, S]");

    TORCH_CHECK(kernel_size == 3 || kernel_size == 5 || kernel_size == 7,
                "PredGatherIGemm backward supports kernel sizes 3, 5, 7; got ",
                kernel_size);
    TORCH_CHECK(
        stride == 1 || stride == 2, "PredGatherIGemm backward supports strides 1, 2; got ", stride);

    int64_t const C = features.size(1);
    int64_t const K = weights.size(0);

    TORCH_CHECK(weights.size(1) == C, "weights C dimension must match features");
    TORCH_CHECK(grad_output.size(1) == K, "grad_output channel dimension must match weights K");
    TORCH_CHECK(weights.size(2) == kernel_size && weights.size(3) == kernel_size &&
                    weights.size(4) == kernel_size,
                "weights spatial dimensions must be ",
                kernel_size,
                "x",
                kernel_size,
                "x",
                kernel_size);
    TORCH_CHECK(C % 32 == 0, "Input channels must be a multiple of 32, got ", C);
    TORCH_CHECK(K % 32 == 0, "Output channels must be a multiple of 32, got ", K);

    TORCH_CHECK(feature_grid.batchSize() == 1,
                "PredGatherIGemm backward currently supports batch size 1");
    TORCH_CHECK(output_grid.batchSize() == 1,
                "PredGatherIGemm backward currently supports batch size 1");

    TORCH_CHECK(feature_grid.totalVoxels() == features.size(0),
                "feature_grid voxel count (",
                feature_grid.totalVoxels(),
                ") must match features row count (",
                features.size(0),
                ")");
    TORCH_CHECK(output_grid.totalVoxels() == grad_output.size(0),
                "output_grid voxel count (",
                output_grid.totalVoxels(),
                ") must match grad_output row count (",
                grad_output.size(0),
                ")");
    TORCH_CHECK(features.device() == weights.device() && features.device() == grad_output.device(),
                "features, weights, and grad_output must be on the same CUDA device");
}

} // namespace pred_gather_igemm_backward

std::tuple<torch::Tensor, torch::Tensor>
predGatherIGemmSparseConvBackward(torch::Tensor grad_output,
                                  torch::Tensor features,
                                  torch::Tensor weights,
                                  GridBatchImpl const &feature_grid,
                                  GridBatchImpl const &output_grid,
                                  int kernel_size,
                                  int stride) {
    using namespace pred_gather_igemm_backward;

    checkPredGatherIGemmBackwardPreconditions(
        grad_output, features, weights, feature_grid, output_grid, kernel_size, stride);

    // Raw pointer kernels below assume contiguous storage in the standard fVDB
    // layout [N, C], [N, K], [K, C, T, R, S].
    if (!features.is_contiguous())
        features = features.contiguous();
    if (!grad_output.is_contiguous())
        grad_output = grad_output.contiguous();
    if (!weights.is_contiguous())
        weights = weights.contiguous();

    switch (kernel_size) {
    case 3:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<3, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<3, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    case 5:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<5, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<5, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    case 7:
        switch (stride) {
        case 1:
            return launchPredGatherIGemmBackward<7, 1>(
                grad_output, features, weights, feature_grid, output_grid);
        case 2:
            return launchPredGatherIGemmBackward<7, 2>(
                grad_output, features, weights, feature_grid, output_grid);
        default: break;
        }
        break;
    default: break;
    }

    TORCH_CHECK(false,
                "Unsupported PredGatherIGemm backward configuration: kernel_size=",
                kernel_size,
                ", stride=",
                stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
