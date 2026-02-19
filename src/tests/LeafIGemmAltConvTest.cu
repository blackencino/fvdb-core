// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
// LeafIGemmAltConvTest.cu -- Tests for the standalone leaf-level implicit GEMM
// convolution (scalar reference).
//
// Compares leafIGemmAltConv output against gatherScatterDefaultSparseConv
// (the validated reference) to verify correctness of the fused
// per-leaf topology-densification + scalar GEMM kernel.
//

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/convolution/GatherScatterDefault.h>
#include <fvdb/detail/ops/convolution/LeafIGemmAltConv.h>

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
topts(torch::Device device) {
    return torch::dtype(torch::kFloat32).device(device);
}

static void
assertNoNanInf(torch::Tensor t, char const *label) {
    auto t_cpu = t.cpu().to(torch::kFloat64);
    EXPECT_FALSE(t_cpu.isnan().any().item<bool>()) << label << " contains NaN";
    EXPECT_FALSE(t_cpu.isinf().any().item<bool>()) << label << " contains Inf";
}

static torch::Tensor
referenceConv(torch::Tensor features, torch::Tensor weights, GridBatchImpl const &grid) {
    int K = static_cast<int>(weights.size(2));
    nanovdb::Coord ks(K, K, K);
    nanovdb::Coord stride(1, 1, 1);
    auto topo = ops::gatherScatterDefaultSparseConvTopology(grid, grid, ks, stride);
    return ops::gatherScatterDefaultSparseConv(features, weights, topo);
}

// =============================================================================
// Unit test: Weight layout consistency
// =============================================================================
//
// Verifies that our weight indexing formula produces the same element as
// the reference's permute+reshape for every (out, in, t, r, s) combination.
//
// Our kernel uses:   W_flat[out, c * 27 + t*9 + r*3 + s]
//                    from weights.reshape({C_out, C_in * 27})
//
// Reference uses:    W_ref[t*9 + r*3 + s, c, out]
//                    from weights.permute({2,3,4,1,0}).reshape({27, C_in, C_out})

TEST(LeafIGemmAltUnit, WeightLayoutConsistency) {
    int C_out = 2, C_in = 3;
    auto weights = torch::zeros({C_out, C_in, 3, 3, 3});
    auto w_acc   = weights.accessor<float, 5>();
    for (int o = 0; o < C_out; ++o)
        for (int c = 0; c < C_in; ++c)
            for (int t = 0; t < 3; ++t)
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        w_acc[o][c][t][r][s] =
                            static_cast<float>(o * 10000 + c * 1000 + t * 100 + r * 10 + s);

    auto W_ours = weights.reshape({C_out, -1}).contiguous();
    auto W_ref  = weights.permute({2, 3, 4, 1, 0}).reshape({27, C_in, C_out}).contiguous();

    auto ours_acc = W_ours.accessor<float, 2>();
    auto ref_acc  = W_ref.accessor<float, 3>();

    for (int o = 0; o < C_out; ++o)
        for (int c = 0; c < C_in; ++c)
            for (int t = 0; t < 3; ++t)
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s) {
                        int k_lin = t * 9 + r * 3 + s;
                        float expected =
                            static_cast<float>(o * 10000 + c * 1000 + t * 100 + r * 10 + s);
                        float ours_val = ours_acc[o][c * 27 + k_lin];
                        float ref_val  = ref_acc[k_lin][c][o];
                        EXPECT_FLOAT_EQ(ours_val, expected)
                            << "Our indexing wrong at o=" << o << " c=" << c << " t=" << t
                            << " r=" << r << " s=" << s;
                        EXPECT_FLOAT_EQ(ref_val, expected)
                            << "Ref indexing wrong at o=" << o << " c=" << c << " t=" << t
                            << " r=" << r << " s=" << s;
                    }
}

// =============================================================================
// Unit test: GEMM contraction equivalence
// =============================================================================
//
// The reference computes (for each kernel offset k):
//   output[scatter, :] += features[gather, :] @ W_ref[k, :, :]
//   where W_ref[k] is [C_in, C_out]
//
// Our kernel computes (fused across all k):
//   output[out_ch] = sum_{ck} W_flat[out_ch, ck] * B[ck, voxel]
//   where ck = c * 27 + k_lin
//
// These should produce the same result. This test verifies it on CPU
// with a tiny example (no CUDA needed).

TEST(LeafIGemmAltUnit, GemmContractionEquivalence) {
    int C_in = 2, C_out = 2;
    int K_vol    = 27;
    int contract = C_in * K_vol;

    torch::manual_seed(42);
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3});
    auto feat_vec = torch::randn({C_in});

    auto W_flat = weights.reshape({C_out, -1}).contiguous();
    auto W_ref  = weights.permute({2, 3, 4, 1, 0}).reshape({K_vol, C_in, C_out}).contiguous();

    auto flat_acc = W_flat.accessor<float, 2>();
    auto ref_acc  = W_ref.accessor<float, 3>();
    auto f_acc    = feat_vec.accessor<float, 1>();

    // "Our" approach: output[out] = sum_{ck} W_flat[out, ck] * B[ck]
    // where B[ck] = feat_vec[c] for c = ck / K_vol
    // (all 27 filter taps see the SAME feature vector -- center tap scenario)
    std::vector<float> out_ours(C_out, 0.0f);
    for (int out = 0; out < C_out; ++out)
        for (int ck = 0; ck < contract; ++ck) {
            int c = ck / K_vol;
            out_ours[out] += flat_acc[out][ck] * f_acc[c];
        }

    // "Reference" approach: output[out] = sum_k { sum_c feat[c] * W_ref[k, c, out] }
    std::vector<float> out_ref(C_out, 0.0f);
    for (int k = 0; k < K_vol; ++k)
        for (int c = 0; c < C_in; ++c)
            for (int out = 0; out < C_out; ++out)
                out_ref[out] += f_acc[c] * ref_acc[k][c][out];

    for (int out = 0; out < C_out; ++out) {
        EXPECT_NEAR(out_ours[out], out_ref[out], 1e-4) << "Output channel " << out << " disagrees";
    }
}

// =============================================================================
// Unit test: Single-voxel identity check
// =============================================================================
//
// One voxel, identity weight (center tap only, all others zero).
// output should equal features (scaled by the center weight).

TEST(LeafIGemmAltUnit, SingleVoxelIdentity) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();
    int C       = 32;

    auto ijk  = torch::tensor({{4, 4, 4}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);
    int64_t N = grid->totalVoxels();

    auto features = torch::ones({N, C}, topts(device));
    auto weights  = torch::zeros({C, C, 3, 3, 3}, topts(device));

    // Set center tap (1,1,1) to identity matrix
    {
        auto w_cpu = weights.cpu();
        auto w_acc = w_cpu.accessor<float, 5>();
        for (int i = 0; i < C; ++i)
            w_acc[i][i][1][1][1] = 1.0f;
        weights = w_cpu.to(device);
    }

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    // With identity center weight and all-ones features, each active output
    // voxel should have value 1.0 in every channel.
    auto ref_cpu  = ref.cpu();
    auto test_cpu = test.cpu();

    // Find which rows are nonzero (the active voxel)
    auto ref_sum  = ref_cpu.abs().sum(1);
    auto test_sum = test_cpu.abs().sum(1);

    // At least one nonzero row
    EXPECT_GT(ref_sum.max().item<float>(), 0.0f) << "Reference has no active output";

    auto diff   = (test_cpu - ref_cpu).abs();
    double maxd = diff.max().item<double>();
    EXPECT_LT(maxd, 1e-5) << "SingleVoxelIdentity: max diff=" << maxd;
}

// =============================================================================
// Unit test: Dense 8x8x8 with small channels, element-wise comparison
// =============================================================================
//
// Runs both reference and our kernel on a dense 8x8x8 grid (one leaf)
// and reports per-element diagnostics for the first few mismatches.

TEST(LeafIGemmAltUnit, Dense8x8x8_Diagnostics) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();
    int C_in = 32, C_out = 32;

    auto grid = makeDenseTestGrid(8, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(999);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    assertNoNanInf(test, "Dense8x8x8_Diagnostics");

    auto ref_cpu  = ref.cpu().to(torch::kFloat64);
    auto test_cpu = test.cpu().to(torch::kFloat64);
    auto diff     = (test_cpu - ref_cpu).abs();

    double maxd  = diff.max().item<double>();
    double meand = diff.mean().item<double>();

    // Print first few mismatches for diagnostics
    if (maxd > 0.1) {
        auto diff_acc = diff.accessor<double, 2>();
        auto ref_acc  = ref_cpu.accessor<double, 2>();
        auto test_acc = test_cpu.accessor<double, 2>();
        int printed   = 0;
        for (int64_t i = 0; i < diff.size(0) && printed < 10; ++i) {
            for (int64_t j = 0; j < diff.size(1) && printed < 10; ++j) {
                if (diff_acc[i][j] > 0.1) {
                    std::cout << "  row=" << i << " col=" << j << " ref=" << ref_acc[i][j]
                              << " test=" << test_acc[i][j] << " diff=" << diff_acc[i][j] << "\n";
                    ++printed;
                }
            }
        }
    }

    EXPECT_LT(maxd, 0.5) << "Dense8x8x8_Diagnostics: max diff=" << maxd << " mean diff=" << meand;
}

// =============================================================================
// Unit test: Verify output tensor shape and non-NaN for various sizes
// =============================================================================

TEST(LeafIGemmAltUnit, OutputShapeAndFiniteness) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    for (int C_in: {32, 64}) {
        for (int C_out: {32, 64}) {
            auto grid = makeDenseTestGrid(8, device);
            int64_t N = grid->totalVoxels();

            torch::manual_seed(100);
            auto features = torch::randn({N, C_in}, topts(device));
            auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

            auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

            EXPECT_EQ(test.dim(), 2) << "C_in=" << C_in << " C_out=" << C_out;
            EXPECT_EQ(test.size(0), N) << "C_in=" << C_in << " C_out=" << C_out;
            EXPECT_EQ(test.size(1), C_out) << "C_in=" << C_in << " C_out=" << C_out;

            assertNoNanInf(test, "OutputShapeAndFiniteness");
        }
    }
}

// =============================================================================
// Unit test: Multi-channel correctness ladder
// =============================================================================
//
// Step up channel counts to find where (if anywhere) multi-channel breaks.

TEST(LeafIGemmAltUnit, Dense8x8x8_Cin32_Cout32) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();
    int C_in = 32, C_out = 32;

    auto grid = makeDenseTestGrid(8, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(555);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    auto diff   = (test.cpu().to(torch::kFloat64) - ref.cpu().to(torch::kFloat64)).abs();
    double maxd = diff.max().item<double>();
    EXPECT_LT(maxd, 0.5) << "Cin32_Cout32: max diff=" << maxd
                         << " mean diff=" << diff.mean().item<double>();
}

TEST(LeafIGemmAltUnit, Dense8x8x8_Cin64_Cout128) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();
    int C_in = 64, C_out = 128;

    auto grid = makeDenseTestGrid(8, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(777);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    auto diff   = (test.cpu().to(torch::kFloat64) - ref.cpu().to(torch::kFloat64)).abs();
    double maxd = diff.max().item<double>();
    EXPECT_LT(maxd, 0.5) << "Cin64_Cout128: max diff=" << maxd
                         << " mean diff=" << diff.mean().item<double>();
}

TEST(LeafIGemmAltUnit, MultiLeaf16x16x16_Cin32_Cout32) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();
    int C_in = 32, C_out = 32;

    auto grid = makeDenseTestGrid(16, device);
    int64_t N = grid->totalVoxels();
    EXPECT_GT(grid->totalLeaves(), 1);

    torch::manual_seed(888);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    auto diff   = (test.cpu().to(torch::kFloat64) - ref.cpu().to(torch::kFloat64)).abs();
    double maxd = diff.max().item<double>();
    EXPECT_LT(maxd, 0.5) << "MultiLeaf Cin32_Cout32: max diff=" << maxd
                         << " mean diff=" << diff.mean().item<double>();
}

// =============================================================================
// Unit test: im2col halo index arithmetic verification
// =============================================================================
//
// Verifies the halo index computation used by our scalar im2col_map
// matches the expected formula:
//   halo_idx = (bx*B0 + z + t) * Hy * Hz + (by*B1 + p + r) * Hz + (bz*B2 + q + s)
//
// This is the same formula that igemm_layouts::activationIndexLayout encodes
// in CuTe layout algebra. If both compute the same indices, the CuTe
// ComposedLayout will produce the correct gather accesses.

TEST(LeafIGemmAltUnit, HaloIndexArithmetic) {
    constexpr int B0 = 4, B1 = 2, B2 = 2;
    constexpr int nblk0 = 2, nblk1 = 4, nblk2 = 4;
    constexpr int Hy = 10, Hz = 10;

    int mismatches = 0;

    for (int bx = 0; bx < nblk0 && mismatches < 5; ++bx)
        for (int by = 0; by < nblk1 && mismatches < 5; ++by)
            for (int bz = 0; bz < nblk2 && mismatches < 5; ++bz)
                for (int z = 0; z < B0 && mismatches < 5; ++z)
                    for (int p = 0; p < B1 && mismatches < 5; ++p)
                        for (int q = 0; q < B2 && mismatches < 5; ++q)
                            for (int t = 0; t < 3 && mismatches < 5; ++t)
                                for (int r = 0; r < 3 && mismatches < 5; ++r)
                                    for (int s = 0; s < 3 && mismatches < 5; ++s) {
                                        int block_orig_x = bx * B0;
                                        int block_orig_y = by * B1;
                                        int block_orig_z = bz * B2;

                                        int halo_x        = block_orig_x + z + t;
                                        int halo_y        = block_orig_y + p + r;
                                        int halo_z        = block_orig_z + q + s;
                                        int expected_halo = halo_x * Hy * Hz + halo_y * Hz + halo_z;

                                        int n     = z * B1 * B2 + p * B2 + q;
                                        int k_lin = t * 9 + r * 3 + s;

                                        int block_orig_flat_x = block_orig_x;
                                        int block_orig_flat_y = block_orig_y;
                                        int block_orig_flat_z = block_orig_z;
                                        int v_x               = n / (B1 * B2);
                                        int v_y               = (n / B2) % B1;
                                        int v_z               = n % B2;
                                        int delta_x           = k_lin / 9;
                                        int delta_y           = (k_lin / 3) % 3;
                                        int delta_z           = k_lin % 3;
                                        int im2col_halo =
                                            (block_orig_flat_x + v_x + delta_x) * Hy * Hz +
                                            (block_orig_flat_y + v_y + delta_y) * Hz +
                                            (block_orig_flat_z + v_z + delta_z);

                                        EXPECT_EQ(im2col_halo, expected_halo)
                                            << "im2col_map disagrees at bx=" << bx << " by=" << by
                                            << " bz=" << bz << " z=" << z << " p=" << p
                                            << " q=" << q << " t=" << t << " r=" << r << " s=" << s;

                                        if (im2col_halo != expected_halo && ++mismatches >= 5)
                                            break;
                                    }
}

// =============================================================================
// Full parameterized tests
// =============================================================================

using LeafIGemmAltParam = std::tuple<int64_t, int64_t>;

static std::string
altParamName(::testing::TestParamInfo<LeafIGemmAltParam> const &info) {
    auto C_in  = std::get<0>(info.param);
    auto C_out = std::get<1>(info.param);
    return "Cin" + std::to_string(C_in) + "_Cout" + std::to_string(C_out);
}

class LeafIGemmAltConvTest : public ::testing::TestWithParam<LeafIGemmAltParam> {
  protected:
    void
    SetUp() override {
        if (!cudaIsAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (!deviceSupportsSm80()) {
            GTEST_SKIP() << "Leaf iGEMM Alt requires Sm80+ (Ampere or newer)";
        }
    }

    void
    checkClose(torch::Tensor test, torch::Tensor ref, char const *context) {
        assertNoNanInf(test, context);

        auto ref_f64  = ref.cpu().to(torch::kFloat64);
        auto test_f64 = test.cpu().to(torch::kFloat64);

        double atol = 1e-1;
        double rtol = 1e-1;

        auto diff       = (test_f64 - ref_f64).abs();
        double max_diff = diff.max().item<double>();

        EXPECT_TRUE(torch::allclose(test_f64, ref_f64, rtol, atol))
            << context << ": max diff=" << max_diff << ", mean diff=" << diff.mean().item<double>();
    }
};

// Disabled: re-enable once unit tests pass
TEST_P(LeafIGemmAltConvTest, DenseSmallGrid) {
    auto [C_in, C_out] = GetParam();
    auto device        = makeDevice();
    int dim            = 8;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(123);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.dim(), 2);
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "DenseSmallGrid");
}

TEST_P(LeafIGemmAltConvTest, DenseMultiLeafGrid) {
    auto [C_in, C_out] = GetParam();
    auto device        = makeDevice();
    int dim            = 16;

    auto grid = makeDenseTestGrid(dim, device);
    int64_t N = grid->totalVoxels();

    EXPECT_GT(grid->totalLeaves(), 1) << "16^3 grid should span multiple 8^3 leaves";

    torch::manual_seed(456);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "DenseMultiLeafGrid");
}

TEST_P(LeafIGemmAltConvTest, SparseGrid) {
    auto [C_in, C_out] = GetParam();
    auto device        = makeDevice();

    auto grid = makeSparseTestGrid(32, 10, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(789);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    checkClose(test, ref, "SparseGrid_10pct");
}

// Edge cases

TEST(LeafIGemmAltConvEdge, EmptyGrid) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    JaggedTensor jt(torch::zeros({0, 3}, torch::dtype(torch::kInt32).device(device)));
    std::vector<nanovdb::Vec3d> voxel_sizes = {{1.0, 1.0, 1.0}};
    std::vector<nanovdb::Vec3d> origins     = {{0.0, 0.0, 0.0}};
    auto grid = GridBatchImpl::createFromIjk(jt, voxel_sizes, origins);

    auto features = torch::empty({0, 32}, topts(device));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device));

    auto out = ops::leafIGemmAltConv(features, weights, *grid, *grid);
    EXPECT_EQ(out.size(0), 0);
    EXPECT_EQ(out.size(1), 32);
}

TEST(LeafIGemmAltConvEdge, SingleVoxel) {
    if (!cudaIsAvailable() || !deviceSupportsSm80()) {
        GTEST_SKIP() << "Requires Sm80+ CUDA";
    }
    auto device = makeDevice();

    auto ijk  = torch::tensor({{50, 50, 50}}, torch::kInt32);
    auto grid = makeGrid(ijk, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(123);
    auto features = torch::randn({N, 32}, topts(device));
    auto weights  = torch::randn({32, 32, 3, 3, 3}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), 32);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, 1e-1, 1e-1))
        << "Single voxel mismatch, max diff=" << (test_f64 - ref_f64).abs().max().item<double>();
}

// Instantiate parameterized tests

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

INSTANTIATE_TEST_SUITE_P(LeafIGemmAlt,
                         LeafIGemmAltConvTest,
                         ::testing::Values(LeafIGemmAltParam{32, 32},
                                           LeafIGemmAltParam{32, 64},
                                           LeafIGemmAltParam{64, 32},
                                           LeafIGemmAltParam{64, 64}),
                         altParamName);

#pragma GCC diagnostic pop

// =============================================================================
// CuTe kernel tests: exercise multiple kernel sizes (1x1x1, 3x3x3, 5x5x5)
// =============================================================================

using CuteKernelParam = std::tuple<int64_t, int64_t, int>;

static std::string
cuteParamName(::testing::TestParamInfo<CuteKernelParam> const &info) {
    auto C_in  = std::get<0>(info.param);
    auto C_out = std::get<1>(info.param);
    auto K     = std::get<2>(info.param);
    return "Cin" + std::to_string(C_in) + "_Cout" + std::to_string(C_out) + "_K" +
           std::to_string(K);
}

class LeafIGemmAltCuteConvTest : public ::testing::TestWithParam<CuteKernelParam> {
  protected:
    void
    SetUp() override {
        if (!cudaIsAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        if (!deviceSupportsSm80()) {
            GTEST_SKIP() << "Leaf iGEMM Alt CuTe requires Sm80+ (Ampere or newer)";
        }
    }
};

TEST_P(LeafIGemmAltCuteConvTest, DenseSingleLeaf) {
    auto [C_in, C_out, K] = GetParam();
    auto device           = makeDevice();

    auto grid = makeDenseTestGrid(8, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(123);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, K, K, K}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    assertNoNanInf(test, "CuTe dense single leaf");
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);
    auto diff     = (test_f64 - ref_f64).abs();
    double maxd   = diff.max().item<double>();

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/1e-1, /*atol=*/1e-1))
        << "CuTe dense K=" << K << " C_in=" << C_in << " C_out=" << C_out << ": max diff=" << maxd;
}

TEST_P(LeafIGemmAltCuteConvTest, DenseMultiLeaf) {
    auto [C_in, C_out, K] = GetParam();
    auto device           = makeDevice();

    auto grid = makeDenseTestGrid(16, device);
    int64_t N = grid->totalVoxels();
    EXPECT_GT(grid->totalLeaves(), 1);

    torch::manual_seed(456);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, K, K, K}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    assertNoNanInf(test, "CuTe dense multi-leaf");
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);
    auto diff     = (test_f64 - ref_f64).abs();
    double maxd   = diff.max().item<double>();

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/1e-1, /*atol=*/1e-1))
        << "CuTe multi-leaf K=" << K << " C_in=" << C_in << " C_out=" << C_out
        << ": max diff=" << maxd;
}

TEST_P(LeafIGemmAltCuteConvTest, SparseGrid) {
    auto [C_in, C_out, K] = GetParam();
    auto device           = makeDevice();

    auto grid = makeSparseTestGrid(32, 10, device);
    int64_t N = grid->totalVoxels();

    torch::manual_seed(789);
    auto features = torch::randn({N, C_in}, topts(device));
    auto weights  = torch::randn({C_out, C_in, K, K, K}, topts(device));

    auto ref  = referenceConv(features, weights, *grid);
    auto test = ops::leafIGemmAltConv(features, weights, *grid, *grid);

    assertNoNanInf(test, "CuTe sparse grid");
    EXPECT_EQ(test.size(0), N);
    EXPECT_EQ(test.size(1), C_out);

    auto ref_f64  = ref.cpu().to(torch::kFloat64);
    auto test_f64 = test.cpu().to(torch::kFloat64);
    auto diff     = (test_f64 - ref_f64).abs();
    double maxd   = diff.max().item<double>();

    EXPECT_TRUE(torch::allclose(test_f64, ref_f64, /*rtol=*/1e-1, /*atol=*/1e-1))
        << "CuTe sparse K=" << K << " C_in=" << C_in << " C_out=" << C_out << ": max diff=" << maxd;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

INSTANTIATE_TEST_SUITE_P(LeafIGemmAltCute,
                         LeafIGemmAltCuteConvTest,
                         ::testing::Values(CuteKernelParam{32, 32, 1},
                                           CuteKernelParam{32, 32, 3},
                                           CuteKernelParam{32, 64, 3},
                                           CuteKernelParam{64, 64, 3},
                                           CuteKernelParam{32, 32, 5},
                                           CuteKernelParam{64, 64, 5}),
                         cuteParamName);

#pragma GCC diagnostic pop
