// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmConvTest.cu -- Tests for the leaf-level implicit GEMM convolution.
//
// Compares leafIGemmConv output against gatherScatterDefaultSparseConv
// (the validated reference) to verify correctness of the fused
// per-leaf topology-densification + GEMM kernel.
//
// Current restrictions (hardcoded in LeafIGemmConv):
//   - 3x3x3 kernel, stride 1, dilation 1
//   - SM80+ (Ampere or newer)
//   - fp32 or fp16
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/LeafIGemmConv.h>

#include <torch/types.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using namespace fvdb;
using namespace fvdb::detail;

// =============================================================================
// Helpers
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
makeSparseTestGrid(int bbox_dim, int occupancy_pct, torch::Device device, int64_t seed = 42) {
    int64_t total = static_cast<int64_t>(bbox_dim) * bbox_dim * bbox_dim;
    int64_t N     = std::max<int64_t>(1, total * occupancy_pct / 100);

    torch::manual_seed(seed);
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

// Run the reference (gather-scatter default) for a 3x3x3 stride-1 conv.
static torch::Tensor
referenceConv(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &grid) {
    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(grid, grid, ks, stride);
    return ops::gatherScatterDefaultSparseConv(features, weights, topo);
}

// =============================================================================
// Test parameter: (scalar_type, C_in, C_out)
// =============================================================================

using LeafIGemmParam = std::tuple<torch::ScalarType, int64_t, int64_t>;

static std::string
leafIGemmParamName(::testing::TestParamInfo<LeafIGemmParam> const &info) {
    auto dtype = std::get<0>(info.param);
    auto C_in  = std::get<1>(info.param);
    auto C_out = std::get<2>(info.param);

    std::string dtype_str = (dtype == torch::kFloat16) ? "f16" : "f32";
    return dtype_str + "_Cin" + std::to_string(C_in) + "_Cout" + std::to_string(C_out);
}

// =============================================================================
// Test fixture
// =============================================================================

class LeafIGemmConvTest : public ::testing::TestWithParam<LeafIGemmParam> {
  protected:
    void
    SetUp() override {
        if (!cudaIsAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (!deviceSupportsSm80()) {
            GTEST_SKIP() << "Leaf iGEMM requires Sm80+ (Ampere or newer)";
        }
    }

    // Tolerance depends on dtype and accumulation path differences.
    void
    checkClose(torch::Tensor test, torch::Tensor ref, char const *context) {
        auto dtype = std::get<0>(GetParam());

        assertNoNanInf(test, context);

        auto ref_f64  = ref.cpu().to(torch::kFloat64);
        auto test_f64 = test.cpu().to(torch::kFloat64);

        double atol = (dtype == torch::kFloat16) ? 0.5 : 1e-1;
        double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

        auto diff       = (test_f64 - ref_f64).abs();
        double max_diff = diff.max().item<double>();

        EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
            << context << ": max diff=" << max_diff << ", mean diff=" << diff.mean().item<double>();
    }
};

// =============================================================================
// Parameterized tests
// =============================================================================

TEST_P(LeafIGemmConvTest, DenseSmallGrid) {
    auto [dtype, C_in, C_out] = GetParam();
    auto device               = makeDevice();
    int dim                   = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(123);
    auto features = torch::randn({N, C_in}, topts(device, dtype));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device, dtype));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.dim(), 2);
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "DenseSmallGrid");
}

TEST_P(LeafIGemmConvTest, DenseMultiLeafGrid) {
    auto [dtype, C_in, C_out] = GetParam();
    auto device               = makeDevice();
    int dim                   = 16;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    EXPECT_GT(grid->totalLeaves(), 1) << "16^3 grid should span multiple 8^3 leaves";

    torch::manual_seed(456);
    auto features = torch::randn({N, C_in}, topts(device, dtype));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device, dtype));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "DenseMultiLeafGrid");
}

TEST_P(LeafIGemmConvTest, SparseGrid) {
    auto [dtype, C_in, C_out] = GetParam();
    auto device               = makeDevice();

    auto grid = makeSparseTestGrid(32, 10, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(789);
    auto features = torch::randn({N, C_in}, topts(device, dtype));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device, dtype));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "SparseGrid_10pct");
}

TEST_P(LeafIGemmConvTest, HighOccupancySparse) {
    auto [dtype, C_in, C_out] = GetParam();
    auto device               = makeDevice();

    auto grid = makeSparseTestGrid(32, 75, device, 99);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(321);
    auto features = torch::randn({N, C_in}, topts(device, dtype));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device, dtype));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "HighOccupancySparse_75pct");
}

// =============================================================================
// Edge cases (non-parameterized)
// =============================================================================

TEST(LeafIGemmConvEdge, EmptyGrid) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    JaggedTensor jt(torch::zeros({0, 3}, torch::dtype(torch::kInt32).device(device)));
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    auto grid = GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);

    auto features = torch::empty({0, 32}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({64, 32, 3, 3, 3}, topts(device, torch::kFloat32));

    auto out = ops::leafIGemmConv(features, weights, *grid, *grid);
    EXPECT_EQ(out.size(0), 0);
    EXPECT_EQ(out.size(1), 64);
}

TEST(LeafIGemmConvEdge, SingleVoxel) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto ijk  = torch::tensor({{50, 50, 50}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);

    torch::manual_seed(123);
    auto features = torch::randn({1, 32}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat32));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), 1);
    EXPECT_EQ(test.size(1), 32);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 1e-1, 1e-1))
        << "Single voxel mismatch, max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST(LeafIGemmConvEdge, SingleLeafPartiallyActive) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto grid = makeSparseTestGrid(8, 50, device, 77);
    int64_t N = grid->totalVoxels();

    EXPECT_EQ(grid->totalLeaves(), 1) << "8^3 bbox should be a single leaf";
    EXPECT_LT(N, 512) << "50% of 512 should be < 512 active voxels";
    EXPECT_GT(N, 0);

    torch::manual_seed(555);
    auto features = torch::randn({N, 16}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({16, 16, 3, 3, 3}, topts(device, torch::kFloat32));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), 16);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 1e-1, 1e-1))
        << "Partially active leaf mismatch, max diff="
        << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST(LeafIGemmConvEdge, SingleVoxelF16) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto ijk  = torch::tensor({{10, 10, 10}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);

    torch::manual_seed(42);
    auto features = torch::randn({1, 8}, topts(device, torch::kFloat16));
    auto weights  = torch::randn({8, 8, 3, 3, 3}, topts(device, torch::kFloat16));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), 1);
    EXPECT_EQ(test.size(1), 8);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 5e-2, 0.5))
        << "Single voxel f16 mismatch, max diff="
        << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Instantiate parameterized tests
// =============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

INSTANTIATE_TEST_SUITE_P(LeafIGemm,
                         LeafIGemmConvTest,
                         ::testing::Values(
                             // fp32 configs (alignment: C must be multiple of 4)
                             LeafIGemmParam{torch::kFloat32, 4, 4},
                             LeafIGemmParam{torch::kFloat32, 32, 32},
                             LeafIGemmParam{torch::kFloat32, 32, 64},
                             LeafIGemmParam{torch::kFloat32, 64, 32},
                             // fp16 configs (alignment: C must be multiple of 8)
                             LeafIGemmParam{torch::kFloat16, 8, 8},
                             LeafIGemmParam{torch::kFloat16, 32, 32},
                             LeafIGemmParam{torch::kFloat16, 32, 64},
                             LeafIGemmParam{torch::kFloat16, 128, 128}),
                         leafIGemmParamName);

#pragma GCC diagnostic pop
