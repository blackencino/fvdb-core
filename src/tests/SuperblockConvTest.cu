// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// SuperblockConvTest.cu -- Tests for the superblock GEMM sparse convolution.
//
// Compares superblockConv* output against gatherScatterDefaultSparseConv*
// (the validated reference) to verify correctness for forward, backward,
// and transposed forward operations.
//
// Requires Sm80+ (Ampere or newer). Tests are skipped on older hardware.
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/Superblock.h>

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

using SuperblockParam = std::tuple<torch::ScalarType, int64_t, int>;

static std::string
sbParamName(::testing::TestParamInfo<SuperblockParam> const &info) {
    auto dtype = std::get<0>(info.param);
    auto C     = std::get<1>(info.param);
    auto K     = std::get<2>(info.param);

    std::string dtype_str = (dtype == torch::kFloat16) ? "f16" : "f32";
    return dtype_str + "_C" + std::to_string(C) + "_K" + std::to_string(K);
}

// =============================================================================
// Test fixture
// =============================================================================

class SuperblockConvTest : public ::testing::TestWithParam<SuperblockParam> {
  protected:
    void
    SetUp() override {
        if (!cudaIsAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (!deviceSupportsSm80()) {
            GTEST_SKIP() << "Superblock GEMM requires Sm80+ (Ampere or newer)";
        }
    }
};

// =============================================================================
// Forward tests: compare superblock against gather-scatter default reference
// =============================================================================

TEST_P(SuperblockConvTest, DenseGridForwardMatchesReference) {
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
    auto test = ops::superblockConv(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "Superblock dense forward output");

    EXPECT_EQ(test.dim(), 2);
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Forward mismatch on dense " << dim << "^3 grid, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST_P(SuperblockConvTest, SparseGridForwardMatchesReference) {
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
    auto test = ops::superblockConv(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "Superblock sparse forward output");

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Forward mismatch on sparse grid, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST_P(SuperblockConvTest, AsymmetricChannelsForward) {
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
    auto test = ops::superblockConv(features, weights, *grid, *grid, ks, stride);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(0.1, K_vol * 0.01) : 1e-1;
    double rtol = (dtype == torch::kFloat16) ? 5e-2 : 1e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Forward mismatch on asymmetric channels C_in=" << C_in << " C_out=" << C_out
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Backward tests: compare superblock backward against gather-scatter default
// =============================================================================

TEST_P(SuperblockConvTest, BackwardGradFeaturesMatchesReference) {
    auto [dtype, C, K] = GetParam();
    auto device = makeDevice();
    int dim     = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(200);
    auto features    = torch::randn({N, C}, topts(device, dtype));
    auto weights     = torch::randn({C, C, K, K, K}, topts(device, dtype));
    auto grad_output = torch::randn({N, C}, topts(device, dtype));

    auto [ref_gf, ref_gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
    auto [test_gf, test_gw] =
        ops::superblockConvBackward(grad_output, features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test_gf, "Superblock backward grad_features");

    EXPECT_EQ(test_gf.sizes(), features.sizes());

    auto ref_f64  = ref_gf.cpu().to(torch::kFloat64);
    auto test_f64 = test_gf.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(1.0, K_vol * 0.05) : 2e-1;
    double rtol = (dtype == torch::kFloat16) ? 1e-1 : 2e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Backward grad_features mismatch, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST_P(SuperblockConvTest, BackwardGradWeightsMatchesReference) {
    auto [dtype, C, K] = GetParam();
    auto device = makeDevice();
    int dim     = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(201);
    auto features    = torch::randn({N, C}, topts(device, dtype));
    auto weights     = torch::randn({C, C, K, K, K}, topts(device, dtype));
    auto grad_output = torch::randn({N, C}, topts(device, dtype));

    auto [ref_gf, ref_gw] =
        ops::gatherScatterDefaultSparseConvBackward(grad_output, features, weights, topo);
    auto [test_gf, test_gw] =
        ops::superblockConvBackward(grad_output, features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test_gw, "Superblock backward grad_weights");

    EXPECT_EQ(test_gw.sizes(), weights.sizes());

    auto ref_f64  = ref_gw.cpu().to(torch::kFloat64);
    auto test_f64 = test_gw.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(2.0, K_vol * 0.1) : 5e-1;
    double rtol = (dtype == torch::kFloat16) ? 2e-1 : 2e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Backward grad_weights mismatch, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Adjoint identity: <gy, forward(x)> == <backward(gy).grad_features, x>
// =============================================================================

TEST(SuperblockConvAdjoint, ForwardAdjointIdentity) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto grid           = makeDenseTestGrid(8, device);
    int64_t const N     = grid->totalVoxels();
    int64_t const C_in  = 32;
    int64_t const C_out = 64;

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);

    torch::manual_seed(300);
    auto x  = torch::randn({N, C_in}, topts(device, torch::kFloat32));
    auto W  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device, torch::kFloat32));
    auto gy = torch::randn({N, C_out}, topts(device, torch::kFloat32));

    auto y        = ops::superblockConv(x, W, *grid, *grid, ks, stride);
    auto [gx, gW] = ops::superblockConvBackward(gy, x, W, *grid, *grid, ks, stride);

    auto lhs = (gy.cpu().to(torch::kFloat64).flatten() *
                y.cpu().to(torch::kFloat64).flatten())
                   .sum()
                   .item<double>();
    auto rhs = (gx.cpu().to(torch::kFloat64).flatten() *
                x.cpu().to(torch::kFloat64).flatten())
                   .sum()
                   .item<double>();

    double rel_err = std::abs(lhs - rhs) / (std::abs(lhs) + 1e-12);
    EXPECT_LT(rel_err, 1e-3)
        << "Feature adjoint identity violated: <gy, L(x)>=" << lhs
        << " vs <L*(gy), x>=" << rhs << ", rel_err=" << rel_err;
}

// =============================================================================
// Transposed forward: compare against gather-scatter default transpose
// =============================================================================

TEST_P(SuperblockConvTest, TransposedForwardMatchesReference) {
    auto [dtype, C, K] = GetParam();
    auto device        = makeDevice();
    int dim            = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo =
        ops::gatherScatterDefaultSparseConvTransposeTopology(*grid, *grid, ks, stride);

    torch::manual_seed(400);
    auto features = torch::randn({N, C}, topts(device, dtype));
    auto weights  = torch::randn({C, C, K, K, K}, topts(device, dtype));

    auto ref  = ops::gatherScatterDefaultSparseConvTranspose(features, weights, topo);
    auto test = ops::superblockConvTranspose(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "Superblock transpose forward output");

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    int K_vol   = K * K * K;
    double atol = (dtype == torch::kFloat16) ? std::max(1.0, K_vol * 0.05) : 2e-1;
    double rtol = (dtype == torch::kFloat16) ? 1e-1 : 2e-1;

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
        << "Transposed forward mismatch, C=" << C << ", K=" << K
        << ", max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Edge cases
// =============================================================================

TEST(SuperblockConvEdge, EmptyGrid) {
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

    auto features = torch::empty({0, 32}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat32));

    auto out = ops::superblockConv(features, weights, *grid, *grid, ks, stride);
    EXPECT_EQ(out.size(0), 0);
    EXPECT_EQ(out.size(1), 32);
}

TEST(SuperblockConvEdge, SingleVoxel) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto ijk  = torch::tensor({{50, 50, 50}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(500);
    auto features = torch::randn({1, 32}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat32));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::superblockConv(features, weights, *grid, *grid, ks, stride);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 1e-1, 1e-1))
        << "Single voxel mismatch, max diff="
        << (test_f64 - ref_f64).abs().max().item<double>();
}

TEST(SuperblockConvEdge, MultiLeafSparse) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto grid = makeSparseTestGrid(32, 10, device);
    int64_t N = grid->totalVoxels();

    nanovdb::Coord ks(3, 3, 3);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(*grid, *grid, ks, stride);

    torch::manual_seed(600);
    auto features = torch::randn({N, 32}, topts(device, torch::kFloat32));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device, torch::kFloat32));

    auto ref  = ops::gatherScatterDefaultSparseConv(features, weights, topo);
    auto test = ops::superblockConv(features, weights, *grid, *grid, ks, stride);

    assertNoNanInf(test, "MultiLeafSparse forward output");

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 1e-1, 1e-1))
        << "MultiLeafSparse mismatch, max diff="
        << (test_f64 - ref_f64).abs().max().item<double>();
}

// =============================================================================
// Instantiate parameterized tests
// =============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

INSTANTIATE_TEST_SUITE_P(Superblock,
                         SuperblockConvTest,
                         ::testing::Values(
                             // fp32 configs
                             SuperblockParam{torch::kFloat32, 32, 3},
                             SuperblockParam{torch::kFloat32, 64, 3},
                             SuperblockParam{torch::kFloat32, 32, 5},
                             // fp16 configs (forward only, backward skipped)
                             SuperblockParam{torch::kFloat16, 32, 3},
                             SuperblockParam{torch::kFloat16, 64, 3}),
                         sbParamName);

#pragma GCC diagnostic pop
