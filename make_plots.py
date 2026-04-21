#!/usr/bin/env python3
"""
make_plots.py — Clean publication-quality plots for BatteryHypoBench paper.

Generates 4 figures:
  Fig 1: Leaderboard bar chart (CBS + all 6 metrics per system)
  Fig 2: Radar chart (metric profiles per system)
  Fig 3: Error taxonomy heatmap (per system × error category)
  Fig 4: Retrieval gap + rank stability

Usage:
  python make_plots.py --results results/analysis_final/combined_results.csv
  python make_plots.py --results results/analysis_final/combined_results.csv \
                       --output figures/
"""

import argparse, json, pathlib, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

# ── Publication style ────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── System display names and colors ──────────────────────────
SYSTEM_META = {
    "REFERENCE":         ("Reference",          "#2C3E50"),
    "ai-researcher":     ("AI-Researcher",       "#E74C3C"),
    "chemdfm-8b":        ("ChemDFM-8B",          "#E67E22"),
    "open-coscientist":  ("Open Co-Scientist",   "#8E44AD"),
    "gemini-direct":     ("Gemini-Direct",       "#2980B9"),
    "gemini-retrieval":  ("Gemini+Search",       "#27AE60"),
    "gemini-weak":       ("Gemini-Weak",         "#95A5A6"),
}

METRIC_META = {
    "rcf_aggregate": ("RCF",  "Reasoning\nFidelity"),
    "hpa_aggregate": ("HPA",  "Hypothesis\nAlignment"),
    "msi_aggregate": ("MSI",  "Mechanistic\nSpecificity"),
    "sns_aggregate": ("SNS",  "Scientific\nNovelty"),
    "ip_aggregate":  ("IP",   "Intervention\nPlausibility"),
    "pdq_aggregate": ("PDQ",  "Problem\nDecomposition"),
    "cbs_score":     ("CBS",  "Composite\nScore"),
}

ERROR_META = {
    "err_high_alignment_weak_mechanism":  "High Alignment\nWeak Mechanism",
    "err_novel_but_implausible":          "Novel but\nImplausible",
    "err_plausible_wrong_target":         "Plausible\nWrong Target",
    "err_verbose_weak_decomposition":     "Verbose Weak\nDecomposition",
    "err_literature_like_low_novelty":    "Literature-like\nLow Novelty",
}

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results and compute per-system summary stats."""
    df = pd.read_csv(csv_path)

    score_cols = list(METRIC_META.keys())
    score_cols = [c for c in score_cols if c in df.columns]

    summary = (df.groupby("_system")[score_cols]
                 .agg(["mean", "std", "count"])
                 .round(4))
    summary.columns = ["_".join(c) for c in summary.columns]

    # Order systems by CBS mean descending
    if "cbs_score_mean" in summary.columns:
        summary = summary.sort_values("cbs_score_mean", ascending=False)

    return df, summary


def get_ordered_systems(summary: pd.DataFrame) -> list[tuple]:
    """Return [(system_key, display_name, color), ...] in CBS order."""
    ordered = []
    for sys in summary.index:
        if sys in SYSTEM_META:
            ordered.append((sys, *SYSTEM_META[sys]))
        else:
            ordered.append((sys, sys, "#7F8C8D"))
    return ordered

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Grouped bar chart — all metrics per system
# ═══════════════════════════════════════════════════════════════

def plot_metric_bars(df: pd.DataFrame, summary: pd.DataFrame,
                     out_dir: pathlib.Path) -> None:
    metrics = [c for c in ["cbs_score","rcf_aggregate","hpa_aggregate",
                             "msi_aggregate","sns_aggregate",
                             "ip_aggregate","pdq_aggregate"]
               if c in summary.columns.str.replace("_mean","")]
    # Only keep cols that have _mean
    metrics = [m for m in metrics
               if f"{m}_mean" in summary.columns]

    systems = get_ordered_systems(summary)
    n_sys = len(systems)
    n_met = len(metrics)

    fig, ax = plt.subplots(figsize=(10, 3.8))

    x = np.arange(n_met)
    total_width = 0.75
    bar_w = total_width / n_sys

    for i, (sys_key, sys_name, color) in enumerate(systems):
        if sys_key not in summary.index:
            continue
        means = [summary.loc[sys_key, f"{m}_mean"] for m in metrics]
        stds  = [summary.loc[sys_key, f"{m}_std"]  for m in metrics]
        offset = (i - n_sys/2 + 0.5) * bar_w
        bars = ax.bar(x + offset, means, bar_w * 0.88,
                      yerr=stds, capsize=2,
                      color=color, alpha=0.85,
                      error_kw={"linewidth": 0.7, "ecolor": "#555"},
                      label=sys_name)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_META[m][0] for m in metrics],
                       fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=4, loc="upper right",
              framealpha=0.9, edgecolor="#ccc",
              columnspacing=0.8, handlelength=1.2)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    path = out_dir / "fig1_metric_bars.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig1] → {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Radar chart — metric profiles
# ═══════════════════════════════════════════════════════════════

def plot_radar(df: pd.DataFrame, summary: pd.DataFrame,
               out_dir: pathlib.Path) -> None:
    radar_metrics = [m for m in ["rcf_aggregate","hpa_aggregate",
                                   "msi_aggregate","sns_aggregate",
                                   "ip_aggregate","pdq_aggregate"]
                     if f"{m}_mean" in summary.columns]
    labels = [METRIC_META[m][0] for m in radar_metrics]
    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    systems = get_ordered_systems(summary)
    # Skip gemini-weak in radar to reduce clutter
    systems = [(k,n,c) for k,n,c in systems if k != "gemini-weak"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5),
                           subplot_kw=dict(projection="polar"))

    for sys_key, sys_name, color in systems:
        if sys_key not in summary.index:
            continue
        vals = [summary.loc[sys_key, f"{m}_mean"]
                for m in radar_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=1.5,
                label=sys_name)
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"], size=7,
                        color="#888")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.5)
    ax.legend(loc="upper left",
              bbox_to_anchor=(-0.35, 1.25),
              framealpha=0.9, edgecolor="#ccc",
              ncol=1)

    plt.tight_layout(pad=0.3)
    path = out_dir / "fig2_radar.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig2] → {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Error taxonomy heatmap
# ═══════════════════════════════════════════════════════════════

def plot_error_taxonomy(df: pd.DataFrame, summary: pd.DataFrame,
                        out_dir: pathlib.Path) -> None:
    err_cols = [c for c in ERROR_META.keys() if c in df.columns]
    if not err_cols:
        print("[fig3] No error columns found, skipping")
        return

    systems = [s for s in summary.index if s in df["_system"].unique()]
    matrix = np.zeros((len(systems), len(err_cols)))
    for i, sys in enumerate(systems):
        sub = df[df["_system"]==sys]
        for j, col in enumerate(err_cols):
            matrix[i, j] = sub[col].astype(float).mean()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.35,
                   aspect="auto")

    # Annotations
    for i in range(len(systems)):
        for j in range(len(err_cols)):
            val = matrix[i, j]
            color = "white" if val > 0.20 else "#333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    sys_labels = [SYSTEM_META.get(s,(s,))[0] for s in systems]
    err_labels = [ERROR_META[c] for c in err_cols]

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(sys_labels, fontsize=8.5)
    ax.set_xticks(range(len(err_cols)))
    ax.set_xticklabels(err_labels, fontsize=7.5, ha="center")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Failure rate", fontsize=8)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    path = out_dir / "fig3_error_taxonomy.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig3] → {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Two-panel — Retrieval gap + Metric correlations
# ═══════════════════════════════════════════════════════════════

def plot_retrieval_and_correlations(df: pd.DataFrame,
                                    summary: pd.DataFrame,
                                    out_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # ── Panel A: Retrieval gap ────────────────────────────────
    ax = axes[0]
    closed_key   = "gemini-direct"
    retrieval_key = "gemini-retrieval"
    metrics_show = ["cbs_score","rcf_aggregate","hpa_aggregate",
                    "msi_aggregate","sns_aggregate","ip_aggregate",
                    "pdq_aggregate"]
    metrics_show = [m for m in metrics_show
                    if f"{m}_mean" in summary.columns
                    and closed_key in summary.index
                    and retrieval_key in summary.index]

    if metrics_show:
        xlabels = [METRIC_META[m][0] for m in metrics_show]
        closed_vals   = [summary.loc[closed_key,   f"{m}_mean"]
                         for m in metrics_show]
        retrieval_vals = [summary.loc[retrieval_key, f"{m}_mean"]
                          for m in metrics_show]
        deltas = [r - c for r, c in
                  zip(retrieval_vals, closed_vals)]

        x = np.arange(len(metrics_show))
        colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]
        bars = ax.bar(x, deltas, 0.55, color=colors, alpha=0.85)

        ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Δ Score (Retrieval − Closed)", fontsize=8.5)
        ax.set_title("A  Retrieval Gap", fontsize=9, loc="left",
                     fontweight="bold")

        # Value labels
        for bar, d in zip(bars, deltas):
            y = bar.get_height()
            va = "bottom" if d >= 0 else "top"
            offset = 0.003 if d >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width()/2,
                    y + offset, f"{d:+.3f}",
                    ha="center", va=va, fontsize=7,
                    color="#333")
    else:
        ax.text(0.5, 0.5, "Retrieval data\nnot available",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title("A  Retrieval Gap", fontsize=9, loc="left",
                     fontweight="bold")

    # ── Panel B: Spearman correlation heatmap ─────────────────
    ax = axes[1]
    corr_metrics = ["rcf_aggregate","hpa_aggregate","msi_aggregate",
                    "sns_aggregate","ip_aggregate","pdq_aggregate"]
    corr_metrics = [m for m in corr_metrics if m in df.columns]

    if len(corr_metrics) >= 2:
        labels = [METRIC_META[m][0] for m in corr_metrics]
        n = len(corr_metrics)
        corr_matrix = np.zeros((n, n))
        for i, ca in enumerate(corr_metrics):
            for j, cb in enumerate(corr_metrics):
                if i == j:
                    corr_matrix[i,j] = 1.0
                elif i < j:
                    vals_a = df[ca].dropna()
                    vals_b = df[cb].dropna()
                    idx = vals_a.index.intersection(vals_b.index)
                    if len(idx) > 10:
                        rho, _ = spearmanr(vals_a[idx], vals_b[idx])
                        corr_matrix[i,j] = corr_matrix[j,i] = rho

        im = ax.imshow(corr_matrix, cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                val = corr_matrix[i,j]
                color = "white" if abs(val) > 0.30 else "#333"
                ax.text(j, i, f"{val:.2f}", ha="center",
                        va="center", fontsize=7, color=color)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Spearman ρ", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    ax.set_title("B  Metric Correlations", fontsize=9,
                 loc="left", fontweight="bold")

    plt.tight_layout(pad=0.8, w_pad=1.5)
    path = out_dir / "fig4_retrieval_correlations.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig4] → {path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: CBS score distributions (violin)
# ═══════════════════════════════════════════════════════════════

def plot_cbs_distributions(df: pd.DataFrame, summary: pd.DataFrame,
                           out_dir: pathlib.Path) -> None:
    if "cbs_score" not in df.columns:
        return

    systems = get_ordered_systems(summary)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    positions = range(len(systems))
    data_by_sys = []
    labels = []
    colors = []

    for sys_key, sys_name, color in systems:
        vals = df[df["_system"]==sys_key]["cbs_score"].dropna().values
        if len(vals) > 0:
            data_by_sys.append(vals)
            labels.append(sys_name)
            colors.append(color)

    parts = ax.violinplot(data_by_sys,
                          positions=range(len(data_by_sys)),
                          showmedians=True,
                          showextrema=False)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor(color)
    parts["cmedians"].set_color("#333")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CBS Score", fontsize=9)
    ax.set_ylim(0.1, 0.75)
    ax.axhline(df[df["_system"]=="REFERENCE"]["cbs_score"].median()
               if "REFERENCE" in df["_system"].values else 0.47,
               color="#2C3E50", linewidth=1, linestyle="--",
               alpha=0.5, label="Reference median")
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=0.5)
    path = out_dir / "fig5_cbs_distributions.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[fig5] → {path}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True,
                   help="Path to combined_results.csv")
    p.add_argument("--output", default="figures/",
                   help="Output directory for figures")
    p.add_argument("--figs", nargs="+",
                   default=["1","2","3","4","5"],
                   help="Which figures to generate")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.results}")
    df, summary = load_data(args.results)
    print(f"[load] {len(df)} rows, {len(summary)} systems")
    print(f"[systems] {summary.index.tolist()}")

    # Classify errors if not already done
    try:
        from full_benchmark import classify_errors
        err_cols = [f"err_{c}" for c in [
            "high_alignment_weak_mechanism","novel_but_implausible",
            "plausible_wrong_target","verbose_weak_decomposition",
            "literature_like_low_novelty"]]
        if not any(c in df.columns for c in err_cols):
            df = classify_errors(df)
    except Exception as e:
        print(f"[warn] Could not classify errors: {e}")

    if "1" in args.figs:
        plot_metric_bars(df, summary, out_dir)
    if "2" in args.figs:
        plot_radar(df, summary, out_dir)
    if "3" in args.figs:
        plot_error_taxonomy(df, summary, out_dir)
    if "4" in args.figs:
        plot_retrieval_and_correlations(df, summary, out_dir)
    if "5" in args.figs:
        plot_cbs_distributions(df, summary, out_dir)

    print(f"\n[done] All figures saved to {out_dir}/")
    print("Files:")
    for f in sorted(out_dir.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
