# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Python-level benchmark comparing fVDB sparse convolution against other sparse
# convolution libraries (spconv, torchsparse, MinkowskiEngine).  All libraries
# receive identical GridBatch / JaggedTensor inputs so the comparison reflects
# the production pipeline where data lives in fVDB grids.
#
# The benchmark measures four phases per iteration:
#   1. build_topology  - create convolution plan / kernel map / native structures
#   2. pre_execute     - convert features/weights to library-native format
#   3. execute         - run the actual convolution kernel
#   4. post_execute    - convert output back to common format
#
# Primary metric:   e2e          (all 4 phases)
# Secondary metric: all_execute  (phases 2-4, topology excluded)
#
# Usage:
#   python benchmark_sparse_conv_comparison.py --output comparison_results.json
#   python benchmark_sparse_conv_comparison.py --suite sparsity --output sparsity_results.json
#   python benchmark_sparse_conv_comparison.py --list-backends
#

from __future__ import annotations

import abc
import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------

_SPCONV_CUDA_VARIANTS = {
    "12.0": "spconv-cu120",
    "12.4": "spconv-cu124",
    "12.6": "spconv-cu126",
}


def _detect_spconv_package() -> str:
    """Pick the best spconv CUDA variant for the current environment."""
    if not torch.cuda.is_available():
        return "spconv"
    cuda_ver = torch.version.cuda or ""
    major_minor = ".".join(cuda_ver.split(".")[:2])
    best = "spconv-cu126"
    for ver, pkg in sorted(_SPCONV_CUDA_VARIANTS.items()):
        if major_minor >= ver:
            best = pkg
    return best


def install_deps(include_minkowski: bool = False) -> None:
    """pip-install optional competitor libraries into the current environment."""
    spconv_pkg = _detect_spconv_package()
    packages = [spconv_pkg]
    if include_minkowski:
        packages.append("git+https://github.com/NVIDIA/MinkowskiEngine")

    for pkg in packages:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])


# ---------------------------------------------------------------------------
# Data generation helpers (deterministic, shared across all backends)
# ---------------------------------------------------------------------------


def generate_dense_coords(dim: int) -> torch.Tensor:
    """Return (N, 3) int32 ijk coordinates for a dense dim^3 grid."""
    r = torch.arange(dim, dtype=torch.int32)
    g = torch.meshgrid(r, r, r, indexing="ij")
    return torch.stack(g, dim=-1).reshape(-1, 3)


def generate_sparse_coords(bbox_dim: int, occupancy_pct: int, seed: int = 42) -> torch.Tensor:
    """Return (N, 3) int32 ijk coordinates at the given occupancy percentage."""
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


def prepare_benchmark_inputs(
    ijk: torch.Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    device: torch.device,
) -> tuple:
    """Build GridBatch, JaggedTensor features, and weight tensor from raw ijk.

    The GridBatch construction happens *outside* the timed region -- it is a
    given input that all adapters receive identically.

    Returns ``(grid, features, weights)`` where *grid* serves as both src and
    dst for sub-manifold (stride=1) convolution.
    """
    import fvdb

    ijk_dev = ijk.to(device)
    grid = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor(ijk_dev), voxel_sizes=1, origins=0, device=device)
    torch.manual_seed(0)
    features = grid.jagged_like(torch.randn(grid.total_voxels, in_channels, device=device))
    weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device=device)
    return grid, features, weights


# ---------------------------------------------------------------------------
# Abstract backend adapter
# ---------------------------------------------------------------------------


class BackendAdapter(abc.ABC):
    """Phased adapter for library-specific sparse convolution.

    All adapters receive the same ``(GridBatch, JaggedTensor, Tensor)`` inputs,
    ensuring the comparison reflects the production pipeline.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @staticmethod
    @abc.abstractmethod
    def available() -> bool: ...

    @abc.abstractmethod
    def build_topology(
        self,
        kernel_size: int,
        stride: int,
        src: Any,  # fvdb.GridBatch
        dst: Any,  # fvdb.GridBatch
        channels_in: int,
        channels_out: int,
    ) -> None:
        """Phase 1: build internal structures from GridBatch topology."""
        ...

    def pre_execute(self, features: Any, weights: torch.Tensor) -> tuple[Any, Any]:
        """Phase 2a: convert features/weights to library-native format."""
        return features, weights

    @abc.abstractmethod
    def execute(self, features: Any, weights: Any) -> Any:
        """Phase 2b: run the convolution kernel."""
        ...

    def post_execute(self, output: Any) -> Any:
        """Phase 2c: convert output back (default: identity)."""
        return output

    def teardown(self) -> None:
        """Release resources (optional)."""
        pass


# ---------------------------------------------------------------------------
# fVDB plan-based adapter (base for all ConvolutionPlan backends)
# ---------------------------------------------------------------------------


class _FVDBPlanAdapter(BackendAdapter):
    """Shared implementation for all fVDB ConvolutionPlan-based backends.

    Subclasses only need to set ``name``, ``available()``, and the three
    class-level knobs: ``_expert_config``, ``_dtype``, ``_channel_alignment``.
    """

    _expert_config: dict[str, Any] = {"backend": "default"}
    _dtype: torch.dtype = torch.float32
    _channel_alignment: int = 1

    def build_topology(self, kernel_size, stride, src, dst, channels_in, channels_out):
        import fvdb

        a = self._channel_alignment
        if a > 1 and (channels_in % a != 0 or channels_out % a != 0):
            raise ValueError(f"{self.name} requires channels divisible by {a}")
        backend_name = self._expert_config.get("backend", "default")
        target = None if backend_name == "dense" else dst
        self._plan = fvdb.ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size,
            stride=stride,
            source_grid=src,
            target_grid=target,
            expert_config=self._expert_config,
        )

    def pre_execute(self, features, weights):
        if self._dtype == torch.float32:
            return features, weights
        import fvdb

        feat_data = features.jdata
        if feat_data.dtype != self._dtype:
            features = fvdb.JaggedTensor(feat_data.to(self._dtype))
        if weights.dtype != self._dtype:
            weights = weights.to(self._dtype)
        return features, weights

    def execute(self, features, weights):
        return self._plan.execute(features, weights)

    def teardown(self):
        if hasattr(self, "_plan"):
            del self._plan


# ---------------------------------------------------------------------------
# Concrete fVDB adapters
# ---------------------------------------------------------------------------


class FVDBAdapter(_FVDBPlanAdapter):
    name = "fVDB"

    @staticmethod
    def available() -> bool:
        try:
            import fvdb  # noqa: F401

            return True
        except ImportError:
            return False


class FVDBCutlassAdapter(_FVDBPlanAdapter):
    name = "fVDB (CUTLASS)"
    _expert_config = {"backend": "cutlass"}
    _dtype = torch.float16
    _channel_alignment = 32

    @staticmethod
    def available() -> bool:
        try:
            import fvdb

            fvdb._fvdb_cpp.cutlass_grouped_gemm_conv
            return torch.cuda.is_available()
        except (ImportError, AttributeError):
            return False


class FVDBImplicitGemmAdapter(_FVDBPlanAdapter):
    name = "fVDB (ImplicitGEMM)"
    _expert_config = {"backend": "implicit_gemm"}
    _dtype = torch.float16
    _channel_alignment = 8

    @staticmethod
    def available() -> bool:
        try:
            import fvdb

            fvdb._fvdb_cpp.implicit_gemm_conv
            if not torch.cuda.is_available():
                return False
            return torch.cuda.get_device_capability()[0] >= 9
        except (ImportError, AttributeError):
            return False


class FVDBSuperblockAdapter(_FVDBPlanAdapter):
    name = "fVDB (Superblock)"
    _expert_config = {"backend": "superblock"}
    _channel_alignment = 32

    @staticmethod
    def available() -> bool:
        try:
            import fvdb

            fvdb._fvdb_cpp.superblock_conv
            if not torch.cuda.is_available():
                return False
            return torch.cuda.get_device_capability()[0] >= 8
        except (ImportError, AttributeError):
            return False


class FVDBDenseAdapter(_FVDBPlanAdapter):
    name = "fVDB (Dense)"
    _expert_config = {"backend": "dense"}

    @staticmethod
    def available() -> bool:
        try:
            import fvdb  # noqa: F401

            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# spconv adapter
# ---------------------------------------------------------------------------


class SpconvAdapter(BackendAdapter):
    name = "spconv"

    @staticmethod
    def available() -> bool:
        try:
            import spconv.pytorch  # noqa: F401

            return True
        except ImportError:
            return False

    def build_topology(self, kernel_size, stride, src, dst, channels_in, channels_out):
        import spconv.pytorch as spconv

        ijk_flat = src.ijk.jdata
        batch_idx = src.jidx.int()

        # Shift coordinates to non-negative (spconv requirement)
        ijk_min = ijk_flat.min(dim=0).values
        shifted_ijk = (ijk_flat - ijk_min).int()

        self._indices = torch.cat([batch_idx.unsqueeze(1), shifted_ijk], dim=1)
        spatial_max = shifted_ijk.max(dim=0).values + 1
        self._spatial_shape = spatial_max.cpu().tolist()
        self._batch_size = src.grid_count
        self._conv = spconv.SubMConv3d(channels_in, channels_out, kernel_size, bias=False).to(ijk_flat.device)

    def pre_execute(self, features, weights):
        import spconv.pytorch as spconv

        inp = spconv.SparseConvTensor(features.jdata, self._indices, self._spatial_shape, batch_size=self._batch_size)
        return inp, weights

    def execute(self, sparse_input, weights):
        return self._conv(sparse_input)

    def post_execute(self, output):
        return output.features

    def teardown(self):
        for attr in ("_conv", "_indices"):
            if hasattr(self, attr):
                delattr(self, attr)


# ---------------------------------------------------------------------------
# torchsparse adapter
# ---------------------------------------------------------------------------


class TorchSparseAdapter(BackendAdapter):
    name = "torchsparse"

    @staticmethod
    def available() -> bool:
        try:
            import torchsparse  # noqa: F401

            return True
        except (ImportError, RuntimeError):
            return False

    @staticmethod
    def _clear_caches() -> None:
        """Best-effort clearing of all torchsparse kmap / hash caches."""
        try:
            import torchsparse.backends

            for attr_name in dir(torchsparse.backends):
                backend = getattr(torchsparse.backends, attr_name, None)
                if backend is not None and hasattr(backend, "kmap_cache"):
                    backend.kmap_cache.clear()
        except Exception:
            pass

    def build_topology(self, kernel_size, stride, src, dst, channels_in, channels_out):
        import torchsparse.nn as spnn

        self._clear_caches()
        ijk_flat = src.ijk.jdata
        batch_idx = src.jidx.int()

        # torchsparse coords: [i, j, k, batch_idx]
        self._coords = torch.cat([ijk_flat.int(), batch_idx.unsqueeze(1)], dim=1)
        self._conv = spnn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, bias=False).to(
            ijk_flat.device
        )

    def pre_execute(self, features, weights):
        import torchsparse

        self._clear_caches()
        inp = torchsparse.SparseTensor(feats=features.jdata, coords=self._coords)
        return inp, weights

    def execute(self, sparse_input, weights):
        return self._conv(sparse_input)

    def post_execute(self, output):
        return output.feats

    def teardown(self):
        for attr in ("_conv", "_coords"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._clear_caches()


# ---------------------------------------------------------------------------
# MinkowskiEngine adapter
# ---------------------------------------------------------------------------


class MinkowskiAdapter(BackendAdapter):
    name = "MinkowskiEngine"

    @staticmethod
    def available() -> bool:
        try:
            import MinkowskiEngine  # noqa: F401

            return True
        except ImportError:
            return False

    def build_topology(self, kernel_size, stride, src, dst, channels_in, channels_out):
        import MinkowskiEngine as ME

        ijk_flat = src.ijk.jdata
        batch_idx = src.jidx.int()

        # ME coords: [batch_idx, i, j, k]
        self._coords = torch.cat([batch_idx.unsqueeze(1), ijk_flat.int()], dim=1)
        self._device = ijk_flat.device
        self._conv = ME.MinkowskiConvolution(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            dimension=3,
        ).to(self._device)

    def pre_execute(self, features, weights):
        import MinkowskiEngine as ME

        try:
            ME.clear_global_coords_man()
        except Exception:
            pass
        inp = ME.SparseTensor(features=features.jdata, coordinates=self._coords, device=self._device)
        return inp, weights

    def execute(self, sparse_input, weights):
        return self._conv(sparse_input)

    def post_execute(self, output):
        return output.F

    def teardown(self):
        for attr in ("_conv", "_coords"):
            if hasattr(self, attr):
                delattr(self, attr)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ADAPTERS: list[type[BackendAdapter]] = [
    FVDBAdapter,
    FVDBCutlassAdapter,
    FVDBImplicitGemmAdapter,
    FVDBSuperblockAdapter,
    FVDBDenseAdapter,
    SpconvAdapter,
    TorchSparseAdapter,
    MinkowskiAdapter,
]


def get_available_adapters() -> list[type[BackendAdapter]]:
    return [a for a in ALL_ADAPTERS if a.available()]


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    library: str
    suite: str
    params: dict[str, Any]
    num_voxels: int
    num_iters: int
    # Per-phase means
    topology_mean_ms: float
    pre_execute_mean_ms: float
    execute_mean_ms: float
    post_execute_mean_ms: float
    # Aggregate: all_execute = pre + execute + post (secondary metric)
    all_execute_mean_ms: float
    all_execute_std_ms: float
    all_execute_min_ms: float
    all_execute_max_ms: float
    # Aggregate: e2e = topology + all_execute (primary metric)
    e2e_mean_ms: float
    e2e_std_ms: float
    e2e_min_ms: float
    e2e_max_ms: float
    # Per-iteration raw data
    times_topology_ms: list[float] = field(default_factory=list)
    times_pre_execute_ms: list[float] = field(default_factory=list)
    times_execute_ms: list[float] = field(default_factory=list)
    times_post_execute_ms: list[float] = field(default_factory=list)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_one(
    adapter_cls: type[BackendAdapter],
    kernel_size: int,
    stride: int,
    src: Any,
    dst: Any,
    features: Any,
    weights: torch.Tensor,
    suite_name: str,
    params: dict[str, Any],
    warmup: int = 3,
    num_iters: int = 20,
) -> BenchmarkResult | None:
    """Run a full phased benchmark for one adapter on one configuration.

    Each timed iteration executes all four phases (build_topology,
    pre_execute, execute, post_execute) with per-phase timing calipers.
    """
    adapter = adapter_cls()
    c_in = weights.shape[1]
    c_out = weights.shape[0]

    # Verify the adapter can handle this configuration
    try:
        _sync()
        adapter.build_topology(kernel_size, stride, src, dst, c_in, c_out)
        _sync()
    except Exception as e:
        print(f"  [{adapter.name}] build_topology failed: {e}")
        return None

    try:
        # Warmup: run all phases
        for _ in range(warmup):
            adapter.build_topology(kernel_size, stride, src, dst, c_in, c_out)
            native_feat, native_w = adapter.pre_execute(features, weights)
            output = adapter.execute(native_feat, native_w)
            adapter.post_execute(output)
            _sync()

        # Timed iterations with per-phase calipers
        t_topo: list[float] = []
        t_pre: list[float] = []
        t_exec: list[float] = []
        t_post: list[float] = []

        for _ in range(num_iters):
            _sync()

            t0 = time.perf_counter()
            adapter.build_topology(kernel_size, stride, src, dst, c_in, c_out)
            _sync()
            t1 = time.perf_counter()

            native_feat, native_w = adapter.pre_execute(features, weights)
            _sync()
            t2 = time.perf_counter()

            output = adapter.execute(native_feat, native_w)
            _sync()
            t3 = time.perf_counter()

            adapter.post_execute(output)
            _sync()
            t4 = time.perf_counter()

            t_topo.append((t1 - t0) * 1e3)
            t_pre.append((t2 - t1) * 1e3)
            t_exec.append((t3 - t2) * 1e3)
            t_post.append((t4 - t3) * 1e3)

        import numpy as np

        a_topo = np.array(t_topo)
        a_pre = np.array(t_pre)
        a_exec = np.array(t_exec)
        a_post = np.array(t_post)
        a_all_exec = a_pre + a_exec + a_post
        a_e2e = a_topo + a_all_exec

        result = BenchmarkResult(
            library=adapter.name,
            suite=suite_name,
            params=params,
            num_voxels=src.total_voxels,
            num_iters=num_iters,
            topology_mean_ms=float(a_topo.mean()),
            pre_execute_mean_ms=float(a_pre.mean()),
            execute_mean_ms=float(a_exec.mean()),
            post_execute_mean_ms=float(a_post.mean()),
            all_execute_mean_ms=float(a_all_exec.mean()),
            all_execute_std_ms=float(a_all_exec.std()),
            all_execute_min_ms=float(a_all_exec.min()),
            all_execute_max_ms=float(a_all_exec.max()),
            e2e_mean_ms=float(a_e2e.mean()),
            e2e_std_ms=float(a_e2e.std()),
            e2e_min_ms=float(a_e2e.min()),
            e2e_max_ms=float(a_e2e.max()),
            times_topology_ms=[float(t) for t in t_topo],
            times_pre_execute_ms=[float(t) for t in t_pre],
            times_execute_ms=[float(t) for t in t_exec],
            times_post_execute_ms=[float(t) for t in t_post],
        )
    except Exception as e:
        print(f"  [{adapter.name}] failed: {e}")
        return None
    finally:
        adapter.teardown()
        _sync()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def suite_grid_size(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep dense grid sizes at fixed C=32, K=3."""
    results: list[BenchmarkResult] = []
    dims = [8, 16, 24, 32]
    C = 32
    K = 3
    for dim in dims:
        ijk = generate_dense_coords(dim)
        grid, features, weights = prepare_benchmark_inputs(ijk, C, C, K, device)
        params = {"grid_dim": dim, "voxels": dim**3, "channels": C, "kernel_size": K}
        print(f"\n[grid_size] dim={dim} ({dim**3} voxels), C={C}, K={K}")
        for acls in adapters:
            r = benchmark_one(acls, K, 1, grid, grid, features, weights, "grid_size", params, warmup, num_iters)
            if r:
                print(
                    f"  {r.library:20s}  e2e={r.e2e_mean_ms:8.3f} ms"
                    f"  exec={r.all_execute_mean_ms:8.3f} ms"
                    f"  topo={r.topology_mean_ms:8.3f} ms"
                )
                results.append(r)
    return results


def suite_sparsity(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep occupancy at fixed channel width, multiple bbox sizes."""
    results: list[BenchmarkResult] = []
    configs = [
        (64, [1, 5, 10, 25, 50, 100]),
        (128, [1, 5, 10, 25, 50]),
        (256, [1, 5, 10, 25]),
    ]
    C = 32
    K = 3
    for bbox_dim, occupancies in configs:
        for occ in occupancies:
            ijk = generate_dense_coords(bbox_dim) if occ >= 100 else generate_sparse_coords(bbox_dim, occ)
            grid, features, weights = prepare_benchmark_inputs(ijk, C, C, K, device)
            n_voxels = ijk.shape[0]
            params = {
                "bbox_dim": bbox_dim,
                "occupancy_pct": occ,
                "voxels": n_voxels,
                "channels": C,
                "kernel_size": K,
            }
            print(f"\n[sparsity] bbox={bbox_dim}, occ={occ}% ({n_voxels} voxels), C={C}, K={K}")
            for acls in adapters:
                r = benchmark_one(acls, K, 1, grid, grid, features, weights, "sparsity", params, warmup, num_iters)
                if r:
                    print(
                        f"  {r.library:20s}  e2e={r.e2e_mean_ms:8.3f} ms"
                        f"  exec={r.all_execute_mean_ms:8.3f} ms"
                    )
                    results.append(r)
    return results


def suite_channels(
    adapters: list[type[BackendAdapter]],
    device: torch.device,
    warmup: int,
    num_iters: int,
) -> list[BenchmarkResult]:
    """Sweep channel width at fixed 16^3 dense grid, K=3."""
    results: list[BenchmarkResult] = []
    dim = 16
    K = 3
    channels = [4, 16, 32, 64, 128, 256]
    for C in channels:
        ijk = generate_dense_coords(dim)
        grid, features, weights = prepare_benchmark_inputs(ijk, C, C, K, device)
        params = {"grid_dim": dim, "voxels": dim**3, "channels": C, "kernel_size": K}
        print(f"\n[channels] dim={dim}, C={C}, K={K}")
        for acls in adapters:
            r = benchmark_one(acls, K, 1, grid, grid, features, weights, "channels", params, warmup, num_iters)
            if r:
                print(
                    f"  {r.library:20s}  e2e={r.e2e_mean_ms:8.3f} ms"
                    f"  exec={r.all_execute_mean_ms:8.3f} ms"
                )
                results.append(r)
    return results


SUITES = {
    "grid_size": suite_grid_size,
    "sparsity": suite_sparsity,
    "channels": suite_channels,
}

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _env_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return meta


def save_results(results: list[BenchmarkResult], path: str) -> None:
    data = {
        "metadata": _env_metadata(),
        "benchmarks": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fVDB sparse convolution against alternative libraries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--install-deps", action="store_true", help="pip-install optional competitor libraries")
    parser.add_argument(
        "--install-minkowski", action="store_true", help="Also attempt to install MinkowskiEngine (fragile)"
    )
    parser.add_argument("--list-backends", action="store_true", help="List available backends and exit")
    parser.add_argument(
        "--suite",
        nargs="*",
        choices=list(SUITES.keys()),
        default=None,
        help="Benchmark suites to run (default: all)",
    )
    parser.add_argument("--output", "-o", type=str, default="comparison_results.json", help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Restrict to these backends (by name substring, e.g. 'fVDB spconv')",
    )
    args = parser.parse_args()

    if args.install_deps:
        install_deps(include_minkowski=args.install_minkowski)
        print("Done installing dependencies.")
        return

    available = get_available_adapters()

    if args.list_backends:
        print("Available backends:")
        for a in available:
            print(f"  - {a.name}")  # type: ignore[attr-defined]
        missing = [a for a in ALL_ADAPTERS if not a.available()]
        if missing:
            print("Not available (install with --install-deps):")
            for a in missing:
                print(f"  - {a.name}")  # type: ignore[attr-defined]
        return

    if args.backends:
        filtered = []
        for a in available:
            if any(b.lower() in a.name.lower() for b in args.backends):  # type: ignore[attr-defined]
                filtered.append(a)
        available = filtered

    if not available:
        print("No backends available. Run with --install-deps or check your fVDB installation.")
        sys.exit(1)

    print("Backends under test:")
    for a in available:
        print(f"  - {a.name}")  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    suites_to_run = args.suite if args.suite else list(SUITES.keys())
    all_results: list[BenchmarkResult] = []

    for suite_name in suites_to_run:
        print(f"\n{'='*60}")
        print(f"  Suite: {suite_name}")
        print(f"{'='*60}")
        suite_fn = SUITES[suite_name]
        results = suite_fn(available, device, args.warmup, args.iters)
        all_results.extend(results)

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
