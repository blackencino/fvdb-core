# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Visualization for sparse convolution benchmarks.
#
# Reads JSON output from benchmark_sparse_conv_comparison.py or
# benchmark_sparse_conv_narrowband.py and produces publication-quality
# plots.  Handles all suite types from both benchmarks.
#
# The benchmark results contain per-phase timing (build_topology,
# pre_execute, execute, post_execute) with two aggregate metrics:
#   - e2e:          all 4 phases (primary)
#   - all_execute:  pre + execute + post (secondary, topology excluded)
#
# Usage:
#   python visualize_comparison.py --file comparison_results.json
#   python visualize_comparison.py --file results.json --filter "fVDB spconv"
#

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# =============================================================================
# 1. Loading
# =============================================================================

LIBRARY_STYLE = {
    "fVDB": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},  # blue
    "fVDB (CUTLASS)": {"color": "#9467bd", "marker": "p", "linestyle": "--"},  # purple
    "fVDB (ImplicitGEMM)": {"color": "#17becf", "marker": "*", "linestyle": "-."},  # cyan
    "fVDB (Superblock)": {"color": "#e377c2", "marker": "h", "linestyle": "-"},  # pink
    "fVDB (Dense)": {"color": "#d62728", "marker": "x", "linestyle": ":"},  # red
    "spconv": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},  # orange
    "torchsparse": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},  # green
    "MinkowskiEngine": {"color": "#8c564b", "marker": "D", "linestyle": ":"},  # brown
    # Legacy name kept for backward compatibility with old result files
    "Dense (conv3d)": {"color": "#d62728", "marker": "x", "linestyle": ":"},  # red
}


def _style_for(library: str) -> dict:
    return LIBRARY_STYLE.get(library, {"color": "gray", "marker": ".", "linestyle": "-"})


def load_results(path: str) -> pd.DataFrame:
    """Load benchmark JSON into a DataFrame.

    Handles both the new phased format (topology_mean_ms, e2e_mean_ms, ...)
    and the legacy format (setup_ms, mean_ms, std_ms, ...) for backward
    compatibility with older result files.
    """
    with open(path, "r") as f:
        data = json.load(f)

    records = []
    for b in data.get("benchmarks", []):
        rec = {
            "library": b["library"],
            "suite": b["suite"],
            "num_voxels": b["num_voxels"],
        }

        # New phased format
        if "e2e_mean_ms" in b:
            rec["topology_mean_ms"] = b["topology_mean_ms"]
            rec["pre_execute_mean_ms"] = b["pre_execute_mean_ms"]
            rec["execute_mean_ms"] = b["execute_mean_ms"]
            rec["post_execute_mean_ms"] = b["post_execute_mean_ms"]
            rec["all_execute_mean_ms"] = b["all_execute_mean_ms"]
            rec["all_execute_std_ms"] = b["all_execute_std_ms"]
            rec["e2e_mean_ms"] = b["e2e_mean_ms"]
            rec["e2e_std_ms"] = b["e2e_std_ms"]
        else:
            # Legacy format: map old fields into new column names
            rec["topology_mean_ms"] = b.get("setup_ms", 0.0)
            rec["pre_execute_mean_ms"] = 0.0
            rec["execute_mean_ms"] = b.get("mean_ms", 0.0)
            rec["post_execute_mean_ms"] = 0.0
            rec["all_execute_mean_ms"] = b.get("mean_ms", 0.0)
            rec["all_execute_std_ms"] = b.get("std_ms", 0.0)
            rec["e2e_mean_ms"] = b.get("setup_ms", 0.0) + b.get("mean_ms", 0.0)
            rec["e2e_std_ms"] = b.get("std_ms", 0.0)

        rec.update(b.get("params", {}))
        records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# 2. Axis formatters
# =============================================================================


def format_voxels(x, _pos):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def format_ms(x, _pos):
    if x >= 1000:
        return f"{x / 1000:.1f}s"
    elif x >= 1:
        return f"{x:.1f}ms"
    elif x >= 0.001:
        return f"{x * 1000:.0f}us"
    return f"{x:.4f}ms"


# =============================================================================
# 3. Plot functions
# =============================================================================


def plot_grid_size_scaling(
    df: pd.DataFrame, metric: str = "e2e_mean_ms", title_suffix: str = "", file_suffix: str = ""
) -> None:
    """Time vs voxel count, one line per library.  Log-log axes.

    Handles both the ``grid_size`` suite (dense grids from the comparison
    benchmark) and the ``scale`` suite (narrow-band spheres from the
    narrowband benchmark).
    """
    suite_df = df[df["suite"].isin(["grid_size", "scale"])].copy()
    if suite_df.empty:
        print("No grid_size / scale data to plot.")
        return

    is_narrowband = "scale" in suite_df["suite"].values
    std_col = metric.replace("_mean_ms", "_std_ms")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in suite_df["library"].unique():
        sub = suite_df[suite_df["library"] == lib].sort_values("num_voxels")
        s = _style_for(lib)
        yerr = sub[std_col] if std_col in sub.columns else None
        ax.errorbar(
            sub["num_voxels"],
            sub[metric],
            yerr=yerr,
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=8,
            capsize=3,
            label=lib,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_voxels))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_xlabel("Voxels")
    metric_label = "E2E Time" if "e2e" in metric else "Execute Time"
    ax.set_ylabel(metric_label)
    base = "Narrow-Band Sphere Scaling (C=32, K=3x3x3)" if is_narrowband else "Grid-Size Scaling (C=32, K=3x3x3)"
    ax.set_title(f"{base}{title_suffix}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    tag = metric.split("_")[0]
    out = f"comparison_grid_size_{tag}{file_suffix}.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_sparsity_breakeven(
    df: pd.DataFrame, metric: str = "e2e_mean_ms", title_suffix: str = "", file_suffix: str = ""
) -> None:
    """Time vs occupancy at each bbox size -- the key comparison plot.

    The fVDB (Dense) backend appears as a reference line showing the cost
    of processing via the dense convolution path.
    """
    suite_df = df[df["suite"] == "sparsity"].copy()
    if suite_df.empty:
        print("No sparsity data to plot.")
        return

    std_col = metric.replace("_mean_ms", "_std_ms")
    bbox_sizes = sorted(suite_df["bbox_dim"].unique())
    sns.set_theme(style="whitegrid", context="talk")
    n_panels = len(bbox_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), squeeze=False)

    dense_names = {"fVDB (Dense)", "Dense (conv3d)"}

    for col, bbox_val in enumerate(bbox_sizes):
        ax = axes[0][col]
        bbox_df = suite_df[suite_df["bbox_dim"] == bbox_val]

        dense_df = bbox_df[bbox_df["library"].isin(dense_names)]
        if not dense_df.empty:
            dense_time = dense_df[metric].iloc[0]
            lib_name = dense_df["library"].iloc[0]
            ax.axhline(
                y=dense_time,
                color=_style_for(lib_name)["color"],
                linestyle="--",
                linewidth=2,
                label=f"{lib_name} ({dense_time:.2f} ms)",
            )

        sparse_libs = [lib for lib in bbox_df["library"].unique() if lib not in dense_names]
        for lib in sparse_libs:
            sub = bbox_df[bbox_df["library"] == lib].sort_values("occupancy_pct")
            s = _style_for(lib)
            yerr = sub[std_col] if std_col in sub.columns else None
            ax.errorbar(
                sub["occupancy_pct"],
                sub[metric],
                yerr=yerr,
                marker=s["marker"],
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=2.5,
                markersize=8,
                capsize=3,
                label=lib,
            )

        total = int(bbox_val) ** 3
        ax.set_xlabel("Occupancy (%)")
        metric_label = "E2E Time" if "e2e" in metric else "Execute Time"
        ax.set_ylabel(metric_label)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
        ax.set_title(f"bbox={bbox_val}^3 ({total:,} cells)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    metric_label = "E2E" if "e2e" in metric else "Execute"
    fig.suptitle(f"Sparsity Breakeven -- {metric_label} (C=32, K=3x3x3){title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    tag = metric.split("_")[0]
    out = f"comparison_sparsity_breakeven_{tag}{file_suffix}.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_channel_scaling(
    df: pd.DataFrame, metric: str = "e2e_mean_ms", title_suffix: str = "", file_suffix: str = ""
) -> None:
    """Time vs channel count, one line per library.  Log-log axes."""
    suite_df = df[df["suite"] == "channels"].copy()
    if suite_df.empty:
        print("No channel data to plot.")
        return

    std_col = metric.replace("_mean_ms", "_std_ms")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in suite_df["library"].unique():
        sub = suite_df[suite_df["library"] == lib].sort_values("channels")
        s = _style_for(lib)
        yerr = sub[std_col] if std_col in sub.columns else None
        ax.errorbar(
            sub["channels"],
            sub[metric],
            yerr=yerr,
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=8,
            capsize=3,
            label=lib,
        )

    chan_vals = np.sort(suite_df["channels"].unique())
    if len(chan_vals) >= 2:
        ref_t = suite_df[suite_df["channels"] == chan_vals[0]][metric].mean()
        y_lin = ref_t * (chan_vals / chan_vals[0])
        y_quad = ref_t * (chan_vals / chan_vals[0]) ** 2
        ax.plot(chan_vals, y_lin, ":", alpha=0.4, color="gray", label="O(C) ref")
        ax.plot(chan_vals, y_quad, "-.", alpha=0.4, color="gray", label="O(C^2) ref")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    chan_vals_sorted = np.sort(suite_df["channels"].unique())
    ax.set_xticks(chan_vals_sorted)
    ax.set_xticklabels([str(int(c)) for c in chan_vals_sorted])
    ax.minorticks_off()
    ax.set_xlabel("Channels (C)")
    metric_label = "E2E Time" if "e2e" in metric else "Execute Time"
    ax.set_ylabel(metric_label)
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    n_vox = suite_df["num_voxels"].iloc[0] if not suite_df.empty else 0
    if "grid_dim" in suite_df.columns and suite_df["grid_dim"].notna().any():
        dim = int(suite_df["grid_dim"].iloc[0])
        subtitle = f"{dim}^3 grid"
    else:
        subtitle = f"~{n_vox:,} voxels"
    ax.set_title(f"Channel Scaling ({subtitle}, K=3x3x3){title_suffix}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    tag = metric.split("_")[0]
    out = f"comparison_channel_scaling_{tag}{file_suffix}.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_phase_breakdown(df: pd.DataFrame, title_suffix: str = "", file_suffix: str = "") -> None:
    """Stacked bar chart showing the 4-phase breakdown per library.

    Uses the largest grid size from the grid_size suite.
    """
    suite_df = df[df["suite"] == "grid_size"].copy()
    if suite_df.empty:
        print("No grid_size data for phase breakdown plot.")
        return

    phase_cols = ["topology_mean_ms", "pre_execute_mean_ms", "execute_mean_ms", "post_execute_mean_ms"]
    if not all(c in suite_df.columns for c in phase_cols):
        print("Phase breakdown columns not found (legacy data?). Skipping.")
        return

    max_dim = suite_df["grid_dim"].max()
    subset = suite_df[suite_df["grid_dim"] == max_dim].copy()
    if subset.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 5))

    libs = list(subset["library"])
    topo = list(subset["topology_mean_ms"])
    pre = list(subset["pre_execute_mean_ms"])
    exe = list(subset["execute_mean_ms"])
    post = list(subset["post_execute_mean_ms"])

    x = np.arange(len(libs))
    bar_width = 0.5

    bottom = np.zeros(len(libs))
    colors = ["#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    labels = ["build_topology", "pre_execute", "execute", "post_execute"]
    for vals, color, label in zip([topo, pre, exe, post], colors, labels):
        arr = np.array(vals)
        ax.bar(x, arr, bar_width, bottom=bottom, label=label, color=color)
        bottom += arr

    for i, total in enumerate(bottom):
        ax.text(i, total + max(bottom) * 0.02, f"{total:.1f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(libs, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Phase Breakdown ({int(max_dim)}^3 grid, C=32, K=3x3x3){title_suffix}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"comparison_phase_breakdown{file_suffix}.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


def plot_extent(
    df: pd.DataFrame, metric: str = "e2e_mean_ms", title_suffix: str = "", file_suffix: str = ""
) -> None:
    """Time vs sphere separation -- shows behavior at huge spatial extents."""
    suite_df = df[df["suite"] == "extent"].copy()
    if suite_df.empty:
        return

    std_col = metric.replace("_mean_ms", "_std_ms")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in suite_df["library"].unique():
        sub = suite_df[suite_df["library"] == lib].sort_values("separation")
        s = _style_for(lib)
        yerr = sub[std_col] if std_col in sub.columns else None
        ax.errorbar(
            sub["separation"],
            sub[metric],
            yerr=yerr,
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=8,
            capsize=3,
            label=lib,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.set_xlabel("Sphere Separation (voxels)")
    metric_label = "E2E Time" if "e2e" in metric else "Execute Time"
    ax.set_ylabel(metric_label)
    n_vox = suite_df["num_voxels"].iloc[0] if not suite_df.empty else 0
    ax.set_title(f"Spatial Extent Scaling (~{n_vox:,} voxels, C=32, K=3x3x3){title_suffix}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    tag = metric.split("_")[0]
    out = f"comparison_extent_{tag}{file_suffix}.png"
    print(f"Saving {out}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()


# =============================================================================
# 4. Summary
# =============================================================================


def print_summary(df: pd.DataFrame, title_suffix: str = "") -> None:
    """Print a text summary of benchmark results."""
    print("\n" + "=" * 72)
    print(f"  Sparse Convolution Comparison Summary{title_suffix}")
    print("=" * 72)

    libs = sorted(df["library"].unique())
    has_phases = "topology_mean_ms" in df.columns

    # Grid-size results
    grid_df = df[df["suite"] == "grid_size"]
    if not grid_df.empty:
        print("\n--- Grid-Size Scaling (C=32, K=3) ---")
        header = f"  {'Grid':>6s}  {'Voxels':>8s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for dim in sorted(grid_df["grid_dim"].unique()):
            vox = int(dim) ** 3
            row = f"  {int(dim):>4d}^3  {vox:>8d}"
            for lib in libs:
                sub = grid_df[(grid_df["grid_dim"] == dim) & (grid_df["library"] == lib)]
                if not sub.empty:
                    row += f"  {sub['e2e_mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    # Sparsity results
    sparse_df = df[df["suite"] == "sparsity"]
    if not sparse_df.empty:
        for bbox_val in sorted(sparse_df["bbox_dim"].unique()):
            bbox_sub = sparse_df[sparse_df["bbox_dim"] == bbox_val]
            total = int(bbox_val) ** 3
            print(f"\n--- Sparsity (bbox={int(bbox_val)}, {total:,} cells, C=32) ---")
            header = f"  {'Occ%':>6s}  {'Voxels':>10s}"
            for lib in libs:
                header += f"  {lib:>16s}"
            print(header)
            for occ in sorted(bbox_sub["occupancy_pct"].unique()):
                occ_sub = bbox_sub[bbox_sub["occupancy_pct"] == occ]
                vox = occ_sub["num_voxels"].iloc[0] if not occ_sub.empty else 0
                row = f"  {int(occ):>5d}%  {int(vox):>10,d}"
                for lib in libs:
                    lsub = occ_sub[occ_sub["library"] == lib]
                    if not lsub.empty:
                        row += f"  {lsub['e2e_mean_ms'].values[0]:>13.3f}ms"
                    else:
                        row += f"  {'--':>16s}"
                print(row)

    # Scale suite (narrowband spheres)
    scale_df = df[df["suite"] == "scale"]
    if not scale_df.empty:
        print("\n--- Narrow-Band Sphere Scaling (C=32, K=3) ---")
        header = f"  {'Radius':>6s}  {'Voxels':>10s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for r in sorted(scale_df["radius"].unique()):
            r_sub = scale_df[scale_df["radius"] == r]
            vox = int(r_sub["num_voxels"].iloc[0])
            row = f"  {int(r):>6d}  {vox:>10,d}"
            for lib in libs:
                sub = r_sub[r_sub["library"] == lib]
                if not sub.empty:
                    row += f"  {sub['e2e_mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    # Channel scaling
    chan_df = df[df["suite"] == "channels"]
    if not chan_df.empty:
        n_vox = int(chan_df["num_voxels"].iloc[0])
        print(f"\n--- Channel Scaling ({n_vox:,} voxels, K=3) ---")
        header = f"  {'C':>6s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for c in sorted(chan_df["channels"].unique()):
            row = f"  {int(c):>6d}"
            for lib in libs:
                sub = chan_df[(chan_df["channels"] == c) & (chan_df["library"] == lib)]
                if not sub.empty:
                    row += f"  {sub['e2e_mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    # Extent suite (two separated spheres)
    extent_df = df[df["suite"] == "extent"]
    if not extent_df.empty:
        print("\n--- Spatial Extent (2 spheres, C=32, K=3) ---")
        header = f"  {'Sep':>6s}  {'Voxels':>10s}  {'BBox':>6s}"
        for lib in libs:
            header += f"  {lib:>16s}"
        print(header)
        for sep in sorted(extent_df["separation"].unique()):
            sep_sub = extent_df[extent_df["separation"] == sep]
            vox = int(sep_sub["num_voxels"].iloc[0])
            bd = int(sep_sub["bbox_dim"].iloc[0])
            row = f"  {int(sep):>6d}  {vox:>10,d}  {bd:>6d}"
            for lib in libs:
                sub = sep_sub[sep_sub["library"] == lib]
                if not sub.empty:
                    row += f"  {sub['e2e_mean_ms'].values[0]:>13.3f}ms"
                else:
                    row += f"  {'--':>16s}"
            print(row)

    # Phase breakdown for the largest grid (if phase data available)
    if has_phases and not grid_df.empty:
        max_dim = grid_df["grid_dim"].max()
        phase_sub = grid_df[grid_df["grid_dim"] == max_dim]
        if not phase_sub.empty:
            print(f"\n--- Phase Breakdown ({int(max_dim)}^3 grid, C=32) ---")
            header = f"  {'Library':>20s}  {'Topology':>10s}  {'PreExec':>10s}  {'Execute':>10s}  {'PostExec':>10s}  {'E2E':>10s}"
            print(header)
            for _, row in phase_sub.iterrows():
                print(
                    f"  {row['library']:>20s}"
                    f"  {row['topology_mean_ms']:>8.3f}ms"
                    f"  {row['pre_execute_mean_ms']:>8.3f}ms"
                    f"  {row['execute_mean_ms']:>8.3f}ms"
                    f"  {row['post_execute_mean_ms']:>8.3f}ms"
                    f"  {row['e2e_mean_ms']:>8.3f}ms"
                )

    print()


# =============================================================================
# 5. Main
# =============================================================================


def _run_plots(df: pd.DataFrame, title_suffix: str = "", file_suffix: str = "") -> None:
    """Generate all applicable plots for the given DataFrame."""
    # Primary metric: e2e
    plot_grid_size_scaling(df, "e2e_mean_ms", title_suffix, file_suffix)
    plot_sparsity_breakeven(df, "e2e_mean_ms", title_suffix, file_suffix)
    plot_channel_scaling(df, "e2e_mean_ms", title_suffix, file_suffix)
    plot_extent(df, "e2e_mean_ms", title_suffix, file_suffix)

    # Secondary metric: all_execute (topology excluded)
    plot_grid_size_scaling(df, "all_execute_mean_ms", " (Execute Only)", "_exec" + file_suffix)
    plot_sparsity_breakeven(df, "all_execute_mean_ms", " (Execute Only)", "_exec" + file_suffix)
    plot_channel_scaling(df, "all_execute_mean_ms", " (Execute Only)", "_exec" + file_suffix)
    plot_extent(df, "all_execute_mean_ms", " (Execute Only)", "_exec" + file_suffix)

    # Phase breakdown
    plot_phase_breakdown(df, title_suffix, file_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize sparse convolution comparison benchmark results.",
    )
    parser.add_argument("--file", "-f", required=True, help="Path to comparison JSON results file")
    parser.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help="Only show these libraries (name substrings, e.g. 'fVDB spconv')",
    )
    args = parser.parse_args()

    df = load_results(args.file)
    print(f"Loaded {len(df)} benchmark records.")

    if args.filter:
        mask = df["library"].apply(lambda lib: any(f.lower() in lib.lower() for f in args.filter))
        df = df[mask]
        print(f"Filtered to {len(df)} records for: {', '.join(args.filter)}")

    if df.empty:
        print("No data to plot.")
        sys.exit(0)

    print_summary(df)
    _run_plots(df)


if __name__ == "__main__":
    main()
