// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// ImplicitGemmConvTest.cu -- Tests for the leaf-fused implicit GEMM convolution.
//
// Compares implicitGemmConv output against gatherScatterDefaultSparseConv
// (the validated reference) to verify correctness of the leaf-fused kernel.
//
// Requires Sm80+ (Ampere or newer). Tests are skipped on older hardware.
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/ImplicitGemmConv.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Helpers (shared with GatherScatterDefaultConvTest)
// =============================================================================

static bool
cudaIsAvailable() {
    int count = 0;
    auto err  = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

static bool
deviceSupportsSm80() {
    if (!cudaIsAvailable())
        return false;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    return major >= 8;
}

static torch::Device
makeDevice() {
    return torch::Device(torch::kCUDA, 0);
}

static c10::intrusive_ptr<GridBatchImpl>
makeGrid(torch::Tensor ijk_2d, torch::Device device) {
    auto ijk_dev = ijk_2d.to(device);
    JaggedTensor jt(ijk_dev);
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    return GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);
}

static c10::intrusive_ptr<GridBatchImpl>
makeDenseTestGrid(int dim, torch::Device device) {
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
makeSparseTestGrid(int bbox_dim, int occupancy_pct, torch::Device device) {
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
topts(torch::Device device, torch::ScalarType dtype) {
    return torch::dtype(dtype).device(device);
}

static void
assertNoNanInf(torch::Tensor t, char const *label) {
    auto t_cpu = t.cpu().to(torch::kFloat64);
    EXPECT_FALSE(t_cpu.isnan().any().item<bool>()) << label << " contains NaN";
    EXPECT_FALSE(t_cpu.isinf().any().item<bool>()) << label << " contains Inf";
}

// =============================================================================
// Test parameter: (scalar_type, channel_count, kernel_size)
// =============================================================================

using ImplicitGemmParam = std::tuple<torch::ScalarType, int64_t, int>;

static std::string
igParamName(::testing::TestParamInfo<ImplicitGemmParam> const &info) {
    auto dtype = std::get<0>(info.param);
    auto C     = std::get<1>(info.param);
    auto K     = std::get<2>(info.param);

    std::string dtype_str = (dtype == torch::kFloat16) ? "f16" : "f32";
    return dtype_str + "_C" + std::to_string(C) + "_K" + std::to_string(K);
}

// =============================================================================
// Test fixture
// =============================================================================

class ImplicitGemmConvTest : public ::testing::TestWithParam<ImplicitGemmParam> {
  protected:
    void
    SetUp() override {
        if (!cudaIsAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (!deviceSupportsSm80()) {
            GTEST_SKIP() << "Implicit GEMM requires Sm80+ (Ampere or newer)";
        }
    }
};

// =============================================================================
// Tests: compare implicit GEMM against gather-scatter default reference
// =============================================================================

TEST_P(ImplicitGemmConvTest, DenseGridMatchesReference) {
    auto [dtype, C, K] = GetParam();
    auto device        = makeDevice();
    int dim            = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(123);
    auto features = torch::randn({N, C}, topts(device, dtype));
    auto weights  = torch::randn({C, C, K, K, K}, topts(device, dtype));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::implicitGemmConv(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "ImplicitGemm dense output");

    EXPECT_EQ(test.dim(), 2);
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/rtol, /*atol=*/atol))
        << "Mismatch on dense " << dim << "^3 grid, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST_P(ImplicitGemmConvTest, SparseGridMatchesReference) {
    auto [dtype, C, K] = GetParam();
    auto device        = makeDevice();

    auto grid = makeSparseTestGrid(32, 10, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(123);
    auto features = torch::randn({N, C}, topts(device, dtype));
    auto weights  = torch::randn({C, C, K, K, K}, topts(device, dtype));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::implicitGemmConv(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "ImplicitGemm sparse output");

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/rtol, /*atol=*/atol))
        << "Mismatch on sparse grid, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST_P(ImplicitGemmConvTest, AsymmetricChannels) {
    auto [dtype, C, K] = GetParam();
    auto device        = makeDevice();
    int dim            = 8;

    int64_t C_in  = C;
    int64_t C_out = C * 2;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(123);
    auto features = torch::randn({N, C_in}, topts(device, dtype));
    auto weights  = torch::randn({C_out, C_in, K, K, K}, topts(device, dtype));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::implicitGemmConv(features, weights, *grid, *grid, ks, stride);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/rtol, /*atol=*/atol))
        << "Mismatch on asymmetric channels C_in=" << C_in << " C_out=" << C_out
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Empty grid edge case
// =============================================================================

TEST(ImplicitGemmConvEdge, EmptyGrid) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    JaggedTensor jt(torch::zeros({0, 3}, torch::dtype(torch::kInt32).device(device)));
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    auto grid = GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    auto features = torch::empty({0, 32}, topts(device, torch::kFloat16));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat16));

    auto out = ops::implicitGemmConv(features, weights, *grid, *grid, ks, stride);
    EXPECT_EQ(out.size(0), 0);
    EXPECT_EQ(out.size(1), 32);
}

// =============================================================================
// Single voxel edge case
// =============================================================================

TEST(ImplicitGemmConvEdge, SingleVoxel) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto ijk  = torch::tensor({{50, 50, 50}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(123);
    auto features = torch::randn({1, 32}, topts(device, torch::kFloat16));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat16));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::implicitGemmConv(features, weights, *grid, *grid, ks, stride);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/1e-1, /*atol=*/1e-1))
        << "Single voxel mismatch, max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Instantiate parameterized tests
// =============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

INSTANTIATE_TEST_SUITE_P(ImplicitGemm,
                         ImplicitGemmConvTest,
                         ::testing::Values(
                             // fp16 configs (channels must be multiples of 32)
                             ImplicitGemmParam{torch::kFloat16, 32, 3},
                             ImplicitGemmParam{torch::kFloat16, 64, 3},
                             ImplicitGemmParam{torch::kFloat16, 32, 5},
                             // fp32 configs
                             ImplicitGemmParam{torch::kFloat32, 32, 3},
                             ImplicitGemmParam{torch::kFloat32, 64, 3}),
                         igParamName);

#pragma GCC diagnostic pop
