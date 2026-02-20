// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// superblock_conv_benchmark.cu -- Performance benchmarks for the
// Superblock GEMM sparse convolution operator.
//
// Benchmark groups:
//
//   Forward         Superblock forward at varying grid sizes and channel widths.
//   Backward        Superblock backward (input grad + weight grad).
//   Comparison      Side-by-side with GatherScatterDefault at matching configs.
//   Sparsity        Forward at varying occupancy levels.
//
// All tensors are pre-allocated outside the timing loop.
// CUDA benchmarks synchronize after warmup and after every iteration.
//

#ifdef __NVCC__
#pragma nv_diag_suppress 177
#endif

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/Superblock.h>

#include <torch/torch.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using namespace fvdb;
using namespace fvdb::detail;

// ============================================================================
// Grid factory helpers
// ============================================================================

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

static c10::intrusive_ptr<GridBatchImpl>
makeDenseGrid(int dim, torch::Device device) {
    std::vector<int32_t> flat;
    flat.reserve(dim * dim * dim * 3);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            for (int k = 0; k < dim; ++k) {
                flat.push_back(i);
                flat.push_back(j);
                flat.push_back(k);
            }
    int64_t N = static_cast<int64_t>(dim) * dim * dim;
    auto ijk  = torch::from_blob(flat.data(), {N, 3}, torch::kInt32).clone();
    return makeGrid(ijk, device);
}

static c10::intrusive_ptr<GridBatchImpl>
makeSparseGrid(int bbox_dim, int occupancy_pct, torch::Device device) {
    int64_t total = static_cast<int64_t>(bbox_dim) * bbox_dim * bbox_dim;
    int64_t N     = std::max<int64_t>(1, total * occupancy_pct / 100);

    torch::manual_seed(42);
    auto perm     = torch::randperm(total, torch::kInt64);
    auto selected = perm.slice(0, 0, N);

    auto ijk     = torch::zeros({N, 3}, torch::kInt32);
    auto sel_acc = selected.accessor<int64_t, 1>();
    auto ijk_acc = ijk.accessor<int32_t, 2>();
    for (int64_t i = 0; i < N; ++i) {
        int64_t idx   = sel_acc[i];
        ijk_acc[i][0] = static_cast<int32_t>(idx / (bbox_dim * bbox_dim));
        ijk_acc[i][1] = static_cast<int32_t>((idx / bbox_dim) % bbox_dim);
        ijk_acc[i][2] = static_cast<int32_t>(idx % bbox_dim);
    }
    return makeGrid(ijk, device);
}

static torch::TensorOptions
topts(torch::Device device) {
    return torch::dtype(torch::kFloat32).device(device);
}

static bool
sm80Available() {
    if (!torch::cuda::is_available())
        return false;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    return major >= 8;
}

// ============================================================================
// A. Forward benchmark
// ============================================================================

static void
BM_SuperblockConv_Forward_CUDA(benchmark::State &state) {
    if (!sm80Available()) {
        state.SkipWithError("Requires Sm80+ CUDA");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    int64_t C   = state.range(1);
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["channels"] = static_cast<double>(C);
}

// ============================================================================
// B. Backward benchmark
// ============================================================================

static void
BM_SuperblockConv_Backward_CUDA(benchmark::State &state) {
    if (!sm80Available()) {
        state.SkipWithError("Requires Sm80+ CUDA");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    int64_t C   = state.range(1);
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    int64_t N        = grid->totalVoxels();
    auto features    = torch::randn({N, C}, topts(device));
    auto weights     = torch::randn({C, C, 3, 3, 3}, topts(device));
    auto grad_output = torch::randn({N, C}, topts(device));

    auto [gf, gw] =
        ops::superblockConvBackward(grad_output, features, weights, *grid, *grid, ks, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto [g1, g2] =
            ops::superblockConvBackward(grad_output, features, weights, *grid, *grid, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(g1.data_ptr<float>());
        benchmark::DoNotOptimize(g2.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["channels"] = static_cast<double>(C);
}

// ============================================================================
// C. Side-by-side comparison with GatherScatterDefault
// ============================================================================

static void
BM_Comparison_GatherScatter_Forward_CUDA(benchmark::State &state) {
    if (!torch::cuda::is_available()) {
        state.SkipWithError("CUDA not available");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    int64_t C   = state.range(1);
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::gatherScatterDefaultSparseConv(features, weights, topo);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["channels"] = static_cast<double>(C);
}

static void
BM_Comparison_Superblock_Forward_CUDA(benchmark::State &state) {
    if (!sm80Available()) {
        state.SkipWithError("Requires Sm80+ CUDA");
        return;
    }
    int dim     = static_cast<int>(state.range(0));
    int64_t C   = state.range(1);
    auto device = torch::Device(torch::kCUDA, 0);
    auto grid   = makeDenseGrid(dim, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"]   = static_cast<double>(N);
    state.counters["channels"] = static_cast<double>(C);
}

// ============================================================================
// D. Sparsity scaling
// ============================================================================

static void
BM_SuperblockConv_Sparsity_CUDA(benchmark::State &state) {
    if (!sm80Available()) {
        state.SkipWithError("Requires Sm80+ CUDA");
        return;
    }
    int occupancy_pct = static_cast<int>(state.range(0));
    int64_t C         = 32;
    auto device       = torch::Device(torch::kCUDA, 0);
    auto grid         = makeSparseGrid(32, occupancy_pct, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    int64_t N     = grid->totalVoxels();
    auto features = torch::randn({N, C}, topts(device));
    auto weights  = torch::randn({C, C, 3, 3, 3}, topts(device));

    auto out = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
    torch::cuda::synchronize();

    for (auto _: state) {
        auto o = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
        torch::cuda::synchronize();
        benchmark::DoNotOptimize(o.data_ptr<float>());
    }
    state.SetItemsProcessed(state.iterations() * N);
    state.counters["voxels"]    = static_cast<double>(N);
    state.counters["occupancy"] = static_cast<double>(occupancy_pct);
}

// ============================================================================
// Register benchmarks
// ============================================================================

BENCHMARK(BM_SuperblockConv_Forward_CUDA)
    ->Args({8, 32})
    ->Args({8, 64})
    ->Args({8, 128})
    ->Args({16, 32})
    ->Args({16, 64})
    ->Args({16, 128})
    ->Args({32, 32})
    ->Args({32, 64})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SuperblockConv_Backward_CUDA)
    ->Args({8, 32})
    ->Args({8, 64})
    ->Args({16, 32})
    ->Args({16, 64})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Comparison_GatherScatter_Forward_CUDA)
    ->Args({16, 32})
    ->Args({16, 64})
    ->Args({16, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Comparison_Superblock_Forward_CUDA)
    ->Args({16, 32})
    ->Args({16, 64})
    ->Args({16, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SuperblockConv_Sparsity_CUDA)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(75)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond);

} // anonymous namespace
