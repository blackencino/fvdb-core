# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Sparse convolution benchmark on realistic narrow-band isosurface grids.
#
# Unlike benchmark_sparse_conv_comparison.py (which sweeps small dense grids
# for apples-to-apples comparison with dense conv3d), this script exercises
# problem sizes representative of real sparse-voxel workloads: narrow bands
# of +-3 voxels around procedural isosurfaces, including configurations with
# huge spatial extents where dense storage is impossible.
#
# Only sparse backends are tested -- no dense baseline.
#
# Usage:
#   python benchmark_sparse_conv_narrowband.py
#   python benchmark_sparse_conv_narrowband.py --output narrowband_results.json
#

from __future__ import annotations

import argparse
import math
import sys

import torch

from benchmark_sparse_conv_comparison import (
    ALL_ADAPTERS,
    BenchmarkResult,
    DenseAdapter,
    benchmark_one,
    save_results,
)

# ---------------------------------------------------------------------------
# Sparse-only adapter list
# ---------------------------------------------------------------------------

SPARSE_ADAPTERS = [a for a in ALL_ADAPTERS if a is not DenseAdapter]


def get_available_sparse_adapters():
    return [a for a in SPARSE_ADAPTERS if a.available()]


# ---------------------------------------------------------------------------
# Narrow-band isosurface grid generation
# ---------------------------------------------------------------------------


def generate_sphere_narrowband(
    radius: float,
    half_width: int = 3,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return (N, 3) int32 ijk coordinates for a narrow band around a sphere.

    Keeps all integer grid points where |distance_to_surface| <= half_width.
    """
    lo = [int(math.floor(center[d] - radius - half_width)) for d in range(3)]
    hi = [int(math.ceil(center[d] + radius + half_width)) for d in range(3)]

    axes = [torch.arange(lo[d], hi[d] + 1, dtype=torch.float32) for d in range(3)]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    dist = torch.sqrt(
        (coords[:, 0] - center[0]) ** 2 + (coords[:, 1] - center[1]) ** 2 + (coords[:, 2] - center[2]) ** 2
    )
    mask = (dist - radius).abs() <= half_width
    ijk = coords[mask].to(torch.int32)

    ijk -= ijk.min(dim=0).values
    return ijk


def generate_two_spheres_narrowband(
    radius: float,
    separation: float,
    half_width: int = 3,
) -> torch.Tensor:
    """Narrow band around two spheres separated along the X axis."""
    s1 = generate_sphere_narrowband(radius, half_width, center=(0.0, 0.0, 0.0))
    s2 = generate_sphere_narrowband(radius, half_width, center=(separation, 0.0, 0.0))

    ijk = torch.cat([s1, s2], dim=0)
    ijk -= ijk.min(dim=0).values
    return ijk.unique(dim=0)


def bbox_dim(ijk: torch.Tensor) -> int:
    """Axis-aligned bounding box side length (covers all coordinates)."""
    return int((ijk.max() - ijk.min()).item()) + 1


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def suite_scale(
    adapters: list[type],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep sphere radius at fixed channels to show scaling behavior."""
    results: list[BenchmarkResult] = []
    radii = [32, 64, 128, 256]
    C = 32
    K = 3
    for r in radii:
        ijk = generate_sphere_narrowband(r)
        n = ijk.shape[0]
        bd = bbox_dim(ijk)
        params = {"radius": r, "voxels": n, "bbox_dim": bd, "channels": C, "kernel_size": K}
        print(f"\n[scale] radius={r}, {n:,} voxels, bbox={bd}, C={C}, K={K}")
        for acls in adapters:
            res = benchmark_one(acls, ijk, C, C, K, device, bd, "scale", params, warmup, num_iters)
            if res:
                print(f"  {res.library:20s}  mean={res.mean_ms:8.3f} ms  std={res.std_ms:6.3f}")
                results.append(res)
    return results


def suite_channels(
    adapters: list[type],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep channel width on a fixed ~1.4M voxel narrow-band sphere."""
    results: list[BenchmarkResult] = []
    r = 128
    ijk = generate_sphere_narrowband(r)
    n = ijk.shape[0]
    bd = bbox_dim(ijk)
    channels = [16, 32, 64, 128]
    K = 3
    for C in channels:
        params = {"radius": r, "voxels": n, "bbox_dim": bd, "channels": C, "kernel_size": K}
        print(f"\n[channels] radius={r} ({n:,} voxels), C={C}, K={K}")
        for acls in adapters:
            res = benchmark_one(acls, ijk, C, C, K, device, bd, "channels", params, warmup, num_iters)
            if res:
                print(f"  {res.library:20s}  mean={res.mean_ms:8.3f} ms  std={res.std_ms:6.3f}")
                results.append(res)
    return results


def suite_extent(
    adapters: list[type],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Two spheres at increasing separation -- huge bounding box, same voxel count."""
    results: list[BenchmarkResult] = []
    r = 64
    separations = [256, 1024, 4096]
    C = 32
    K = 3
    for sep in separations:
        ijk = generate_two_spheres_narrowband(r, sep)
        n = ijk.shape[0]
        bd = bbox_dim(ijk)
        params = {
            "radius": r,
            "separation": sep,
            "voxels": n,
            "bbox_dim": bd,
            "channels": C,
            "kernel_size": K,
        }
        print(f"\n[extent] 2 x sphere r={r}, sep={sep}, {n:,} voxels, bbox={bd}, C={C}, K={K}")
        for acls in adapters:
            res = benchmark_one(acls, ijk, C, C, K, device, bd, "extent", params, warmup, num_iters)
            if res:
                print(f"  {res.library:20s}  mean={res.mean_ms:8.3f} ms  std={res.std_ms:6.3f}")
                results.append(res)
    return results


SUITES = {
    "scale": suite_scale,
    "channels": suite_channels,
    "extent": suite_extent,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse conv benchmark on narrow-band isosurface grids.")
    parser.add_argument("--output", "-o", default="narrowband_results.json", help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    available = get_available_sparse_adapters()
    if not available:
        print("No sparse backends available.")
        sys.exit(1)

    print("Backends under test:")
    for a in available:
        print(f"  - {a.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results: list[BenchmarkResult] = []
    for name, fn in SUITES.items():
        print(f"\n{'=' * 60}")
        print(f"  Suite: {name}")
        print(f"{'=' * 60}")
        all_results.extend(fn(available, device, args.warmup, args.iters))

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
