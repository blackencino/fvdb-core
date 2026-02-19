# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Test the CUTLASS 2.x implicit GEMM sparse convolution backend.

Compares implicit GEMM results against the default gather-scatter backend
to verify that the fused gather-GEMM-scatter kernel produces correct output.

Requires Sm80+ (Ampere or newer) GPU. Tests are skipped on older hardware.
"""

import unittest

import torch
from parameterized import parameterized

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

REQUIRES_SM80 = unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    "Implicit GEMM backend requires Sm80+ (Ampere or newer)",
)


def _generate_dense_coords(dim: int) -> torch.Tensor:
    r = torch.arange(dim, dtype=torch.int32)
    g = torch.meshgrid(r, r, r, indexing="ij")
    return torch.stack(g, dim=-1).reshape(-1, 3)


def _generate_sparse_coords(bbox_dim: int, occupancy_pct: int, seed: int = 42) -> torch.Tensor:
    total = bbox_dim**3
    n = max(1, total * occupancy_pct // 100)
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen)[:n]
    ijk = torch.zeros(n, 3, dtype=torch.int32)
    ijk[:, 0] = (perm // (bbox_dim * bbox_dim)).to(torch.int32)
    ijk[:, 1] = ((perm // bbox_dim) % bbox_dim).to(torch.int32)
    ijk[:, 2] = (perm % bbox_dim).to(torch.int32)
    return ijk


def _run_conv(ijk: torch.Tensor, C_in: int, C_out: int, kernel_size: int, backend: str, dtype: torch.dtype):
    """Run a sparse convolution with the given backend and return the output tensor."""
    device = torch.device("cuda")
    ijk_dev = ijk.to(device)
    jt = JaggedTensor(ijk_dev)
    grid = GridBatch.from_ijk(jt, voxel_sizes=1, origins=0, device=device)

    plan = ConvolutionPlan.from_grid_batch(
        kernel_size=kernel_size,
        stride=1,
        source_grid=grid,
        target_grid=grid,
        expert_config={"backend": backend},
    )

    torch.manual_seed(123)
    n = grid.total_voxels
    features = torch.randn(n, C_in, device=device, dtype=dtype)
    weights = torch.randn(C_out, C_in, kernel_size, kernel_size, kernel_size, device=device, dtype=dtype)

    output = plan.execute(features, weights)
    return output


# (label, grid_dim, C_in, C_out, kernel_size, dtype)
TEST_CONFIGS = [
    ("dense_8_C32_K3_fp16", 8, 32, 32, 3, torch.float16),
    ("dense_8_C64_K3_fp16", 8, 64, 64, 3, torch.float16),
    ("dense_16_C32_K3_fp16", 16, 32, 32, 3, torch.float16),
    ("dense_8_C32_K3_fp32", 8, 32, 32, 3, torch.float32),
    ("dense_8_C32_K5_fp16", 8, 32, 32, 5, torch.float16),
    ("dense_8_C32_asym_fp16", 8, 32, 64, 3, torch.float16),
]


@REQUIRES_SM80
class TestImplicitGemmConv(unittest.TestCase):

    @parameterized.expand(TEST_CONFIGS)
    def test_matches_default_dense(self, _name, grid_dim, C_in, C_out, kernel_size, dtype):
        """Implicit GEMM output should match the default gather-scatter backend on a dense grid."""
        ijk = _generate_dense_coords(grid_dim)

        ref = _run_conv(ijk, C_in, C_out, kernel_size, "default", dtype)
        test = _run_conv(ijk, C_in, C_out, kernel_size, "implicit_gemm", dtype)

        K_vol = kernel_size**3
        if dtype == torch.float16:
            atol, rtol = max(0.1, K_vol * 0.01), 5e-2
        else:
            atol, rtol = 1e-1, 1e-1

        torch.testing.assert_close(
            test,
            ref,
            atol=atol,
            rtol=rtol,
            msg=f"Mismatch on dense {grid_dim}^3, C={C_in}->{C_out}, K={kernel_size}, {dtype}",
        )

    def test_matches_default_sparse(self):
        """Implicit GEMM should match default on a sparse grid (bbox=64, 10% occupancy)."""
        ijk = _generate_sparse_coords(64, 10)
        ref = _run_conv(ijk, 32, 32, 3, "default", torch.float16)
        test = _run_conv(ijk, 32, 32, 3, "implicit_gemm", torch.float16)
        torch.testing.assert_close(test, ref, atol=0.1, rtol=5e-2)

    def test_matches_default_sparse_large(self):
        """Implicit GEMM should match default on a larger sparse grid (bbox=128, 5%)."""
        ijk = _generate_sparse_coords(128, 5)
        ref = _run_conv(ijk, 32, 32, 3, "default", torch.float16)
        test = _run_conv(ijk, 32, 32, 3, "implicit_gemm", torch.float16)
        torch.testing.assert_close(test, ref, atol=0.1, rtol=5e-2)

    def test_zero_output_with_no_neighbors(self):
        """A single isolated voxel with K=3 should produce output only from the center weight."""
        ijk = torch.tensor([[100, 100, 100]], dtype=torch.int32)
        ref = _run_conv(ijk, 32, 32, 3, "default", torch.float16)
        test = _run_conv(ijk, 32, 32, 3, "implicit_gemm", torch.float16)
        torch.testing.assert_close(test, ref, atol=0.1, rtol=5e-2)

    def test_empty_grid(self):
        """Convolution on an empty grid should return an empty tensor."""
        device = torch.device("cuda")
        grid = GridBatch.from_zero_voxels(device=device, voxel_sizes=1.0, origins=0.0)
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=3,
            stride=1,
            source_grid=grid,
            target_grid=grid,
            expert_config={"backend": "implicit_gemm"},
        )
        features = torch.empty(0, 32, device=device, dtype=torch.float16)
        weights = torch.randn(32, 32, 3, 3, 3, device=device, dtype=torch.float16)
        output = plan.execute(features, weights)
        self.assertEqual(output.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
