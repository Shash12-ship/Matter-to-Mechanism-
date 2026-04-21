#!/usr/bin/env python3
"""
run_coscientist_benchmark.py — Full head-to-head co-scientist benchmark.

Generates hypotheses from each system on the same battery problems,
scores them with BatteryHypoBench, and produces a ranked leaderboard.

Usage:
  # List available systems and check which keys you have:
  python run_coscientist_benchmark.py --list-systems

  # Run all systems you have keys for (50 problems):
  python run_coscientist_benchmark.py \\
      --csv /path/to/battery_problem_solution_500.csv \\
      --sample 50 \\
      --output results/coscientist_benchmark/ \\
      --openai-api-key $OPENAI_API_KEY \\
      --anthropic-api-key $ANTHROPIC_API_KEY \\
      --gemini-api-key $GEMINI_API_KEY \\
      --fh-api-key $FH_API_KEY

  # Specific systems only:
  python run_coscientist_benchmark.py \\
      --csv /path/to/battery_problem_solution_500.csv \\
      --systems gpt-4o o3 gemini-2.5-pro sakana-ai-scientist futurehouse-crow \\
      --sample 50 \\
      --openai-api-key $OPENAI_API_KEY \\
      --gemini-api-key $GEMINI_API_KEY \\
      --fh-api-key $FH_API_KEY

  # Include local ChemDFM (no API key needed, needs GPU):
  python run_coscientist_benchmark.py \\
      --csv /path/to/battery_problem_solution_500.csv \\
      --systems chemdfm-8b gpt-4o gemini-2.5-pro \\
      --sample 30

  # Score reference (ground truth) hypotheses only (no API calls):
  python run_coscientist_benchmark.py \\
      --csv /path/to/battery_problem_solution_500.csv \\
      --reference-only --sample 500

SLURM (on NCSA Delta GH200):
  sbatch slurm_benchmark.sh
"""

import argparse
import json
import os
import sys
import re
import time
import pathlib
import statistics
import textwrap
import collections

import pandas as pd
import numpy as np

# ── local imports ──────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from benchmark import (
    compute_rcf, compute_hpa, compute_msi, compute_ip, compute_pdq,
    compute_cbs, compute_sns_corpus, VERSION,
)
from co_scientist_adapters import (
    ADAPTER_REGISTRY, generate_with_adapter, list_systems,
)


# ─────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────

def score_row(row: pd.Series) -> dict:
    """Apply all non-corpus metrics to a generated row."""
    scores = {}
    scores.update(compute_rcf(row))
    scores.update(compute_hpa(row))
    scores.update(compute_msi(row))
    scores.update(compute_ip(row))
    scores.update(compute_pdq(row))
    return scores


AGG_COLS = [
    "rcf_aggregate", "hpa_aggregate", "msi_aggregate",
    "ip_aggregate", "pdq_aggregate",
]


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run(args):
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "per_system"
    logs_dir.mkdir(exist_ok=True)

    # ── load dataset ────────────────────────────────────────
    print(f"\n[load] {args.csv}")
    df = pd.read_csv(args.csv)
    text_cols = [
        "problem_statement", "hypothesis", "reasoning_process",
        "mechanism_or_rationale", "intervention_or_solution",
        "claimed_outcome", "battery_system", "component",
        "failure_mode_or_limitation", "problem_type_broad",
        "problem_type_fine", "problem_core", "target_property",
        "evidence_strength",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    sample_df = df.sample(
        n=min(args.sample, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"[load] {len(df)} total → sampled {len(sample_df)} problems")

    # ── collect API keys ────────────────────────────────────
    api_keys = {
        "OPENAI_API_KEY":   args.openai_api_key  or os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY":   args.gemini_api_key  or os.environ.get("GEMINI_API_KEY", ""),
        "FH_API_KEY":       args.fh_api_key      or os.environ.get("FH_API_KEY", ""),
        "HF_TOKEN":         args.hf_token        or os.environ.get("HF_TOKEN", ""),
    }

    # ── determine which systems to run ──────────────────────
    if args.list_systems:
        list_systems()
        sys.exit(0)

    if args.reference_only:
        systems_to_run = []
    elif args.systems:
        systems_to_run = args.systems
    else:
        # Auto-detect: run all systems where we have the required key
        systems_to_run = []
        for name, (_, _, env_var, _) in ADAPTER_REGISTRY.items():
            if api_keys.get(env_var):
                systems_to_run.append(name)
        # Don't auto-include deep research (slow/expensive) unless explicit
        systems_to_run = [s for s in systems_to_run if "deep-research" not in s
                         and "google-co-scientist" not in s]
        if not systems_to_run:
            print("[warn] No API keys found. Running reference-only mode.")

    print(f"\n[plan] Systems to benchmark: {systems_to_run or ['(reference only)']}")
    print(f"[plan] Problems per system: {len(sample_df)}")
    print()

    all_results = []

    # ── REFERENCE (ground-truth dataset hypotheses) ─────────
    print("─" * 60)
    print("[REFERENCE] Scoring ground-truth hypotheses from dataset")
    print("─" * 60)
    ref_rows = []
    for _, row in sample_df.iterrows():
        d = row.to_dict()
        d["_system"] = "REFERENCE"
        d["_model"] = "ground_truth"
        scores = score_row(pd.Series(d))
        d.update(scores)
        ref_rows.append(d)
    all_results.extend(ref_rows)
    ref_cbs = [compute_cbs({c: r.get(c, 0.5) for c in AGG_COLS if c in r}) for r in ref_rows]
    print(f"  Mean CBS: {statistics.mean(ref_cbs):.4f} ± {statistics.stdev(ref_cbs):.4f}")

    # ── EACH CO-SCIENTIST ────────────────────────────────────
    for system_name in systems_to_run:
        print()
        print("─" * 60)
        _, _, env_var, desc = ADAPTER_REGISTRY[system_name]
        print(f"[{system_name}] {desc}")
        print("─" * 60)

        system_rows = []
        errors = 0

        for i, (_, row) in enumerate(sample_df.iterrows()):
            pid = str(row.get("paper_id", f"row_{i}"))[:30]
            print(f"  [{i+1:3d}/{len(sample_df)}] {pid}...", end=" ", flush=True)

            gen = generate_with_adapter(system_name, row.to_dict(), api_keys)

            if gen.get("error") and not gen.get("hypothesis"):
                errors += 1
                print(f"ERROR: {gen['error'][:60]}")
                d = row.to_dict()
                d.update({"_system": system_name, "_model": ADAPTER_REGISTRY[system_name][1],
                           "hypothesis": "", "error": gen["error"]})
                system_rows.append(d)
                if errors > 5:
                    print(f"  [skip] Too many errors for {system_name}, skipping remaining rows")
                    break
                continue

            # Merge generated fields with original problem context
            d = row.to_dict()
            d.update({k: v for k, v in gen.items() if k not in ("paper_id",)})
            d["_system"] = system_name
            d["_model"] = ADAPTER_REGISTRY[system_name][1]

            # Score
            scores = score_row(pd.Series(d))
            d.update(scores)
            cbs = compute_cbs({c: d.get(c, 0.5) for c in AGG_COLS if c in d})
            d["cbs_score"] = cbs
            system_rows.append(d)
            print(f"CBS={cbs:.3f}")

            # Rate limiting
            time.sleep(args.sleep)

        all_results.extend(system_rows)

        # Save per-system intermediate results
        sys_df = pd.DataFrame(system_rows)
        sys_safe = system_name.replace("/", "-")
        sys_df.to_csv(logs_dir / f"{sys_safe}.csv", index=False)
        valid = [r for r in system_rows if r.get("cbs_score")]
        if valid:
            mean_cbs = statistics.mean([r["cbs_score"] for r in valid])
            print(f"  → {len(valid)}/{len(system_rows)} valid | Mean CBS: {mean_cbs:.4f}")

    # ── Compile full results ─────────────────────────────────
    print("\n[compile] Building full results dataframe...")
    results_df = pd.DataFrame(all_results)

    # Compute CBS for rows missing it
    def safe_cbs(r):
        vals = {c: r.get(c) for c in AGG_COLS if pd.notna(r.get(c, float("nan")))}
        return compute_cbs(vals) if vals else 0.0
    if "cbs_score" not in results_df.columns:
        results_df["cbs_score"] = results_df.apply(safe_cbs, axis=1)
    else:
        missing_mask = results_df["cbs_score"].isna()
        results_df.loc[missing_mask, "cbs_score"] = results_df[missing_mask].apply(safe_cbs, axis=1)

    # ── SNS corpus-level novelty (per system) ───────────────
    print("[sns] Computing corpus novelty per co-scientist...")
    if "_system" in results_df.columns:
        for system in results_df["_system"].unique():
            mask = results_df["_system"] == system
            sys_sub = results_df[mask].copy()
            if len(sys_sub) > 5 and "hypothesis" in sys_sub.columns:
                sns_sub = compute_sns_corpus(sys_sub)
                for col in sns_sub.columns:
                    results_df.loc[mask, col] = sns_sub[col].values
                # Recompute CBS with SNS
                results_df.loc[mask, "cbs_score"] = results_df[mask].apply(
                    lambda r: compute_cbs({
                        c: r.get(c) for c in AGG_COLS + ["sns_aggregate"]
                        if pd.notna(r.get(c, float("nan")))
                    }), axis=1
                )

    # ── LEADERBOARD ──────────────────────────────────────────
    score_cols = ["cbs_score"] + AGG_COLS + ["sns_aggregate"]
    score_cols = [c for c in score_cols if c in results_df.columns]

    if "_system" in results_df.columns:
        lb = (
            results_df.groupby("_system")[score_cols]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        # Flat column names
        lb.columns = ["_".join(c) for c in lb.columns]
        lb = lb.sort_values("cbs_score_mean", ascending=False)

        print("\n" + "=" * 75)
        print("  CO-SCIENTIST LEADERBOARD (BatteryHypoBench CBS Score)")
        print("=" * 75)
        for rank, (system, row_lb) in enumerate(lb.iterrows(), 1):
            n = int(row_lb.get("cbs_score_count", 0))
            mean = row_lb.get("cbs_score_mean", 0.0)
            std = row_lb.get("cbs_score_std", 0.0)
            rcf = row_lb.get("rcf_aggregate_mean", 0.0)
            msi = row_lb.get("msi_aggregate_mean", 0.0)
            sns = row_lb.get("sns_aggregate_mean", 0.0)
            print(f"  {rank:2}. {system:<28}  CBS={mean:.4f}±{std:.4f}  "
                  f"RCF={rcf:.3f}  MSI={msi:.3f}  SNS={sns:.3f}  n={n}")
        print("=" * 75 + "\n")

        lb.to_csv(out_dir / "coscientist_leaderboard.csv")

    # ── SAVE OUTPUTS ─────────────────────────────────────────
    out_csv = out_dir / "all_generated_scored.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[save] Full results → {out_csv}")

    # System × metric pivot
    if "_system" in results_df.columns:
        pivot = results_df.groupby("_system")[score_cols].mean().round(4)
        pivot.to_csv(out_dir / "system_metric_pivot.csv")
        print(f"[save] Metric pivot → {out_dir}/system_metric_pivot.csv")

    # JSON summary
    summary = {}
    if "_system" in results_df.columns:
        for system in results_df["_system"].unique():
            mask = results_df["_system"] == system
            sub = results_df[mask]
            summary[system] = {
                col: {
                    "mean": round(float(sub[col].mean()), 4),
                    "std": round(float(sub[col].std()), 4),
                    "n": int(sub[col].notna().sum()),
                }
                for col in score_cols if col in sub.columns
            }
    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] Summary JSON → {out_dir}/benchmark_summary.json")

    # Markdown report
    _write_leaderboard_report(results_df, score_cols, out_dir)

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench Co-Scientist Benchmark DONE")
    print(f"  {len(systems_to_run)} systems × {len(sample_df)} problems")
    print(f"  Results in: {out_dir}/")
    print(f"{'='*60}\n")


def _write_leaderboard_report(results_df, score_cols, out_dir):
    """Write detailed Markdown report."""
    lines = [
        "# BatteryHypoBench: Co-Scientist Benchmark Report",
        "",
        "## Leaderboard",
        "",
        "| Rank | System | CBS ↓ | RCF | HPA | MSI | SNS | IP | PDQ | n |",
        "|------|--------|-------|-----|-----|-----|-----|----|-----|---|",
    ]

    if "_system" not in results_df.columns:
        with open(out_dir / "coscientist_report.md", "w") as f:
            f.write("\n".join(lines))
        return

    col_map = {
        "cbs_score": "cbs_score", "rcf_aggregate": "rcf_aggregate",
        "hpa_aggregate": "hpa_aggregate", "msi_aggregate": "msi_aggregate",
        "sns_aggregate": "sns_aggregate", "ip_aggregate": "ip_aggregate",
        "pdq_aggregate": "pdq_aggregate",
    }
    avail = {k: v for k, v in col_map.items() if v in results_df.columns}

    sys_means = results_df.groupby("_system")[list(avail.values())].mean().round(4)
    sys_n = results_df.groupby("_system")["cbs_score"].count() if "cbs_score" in results_df.columns else {}
    sys_means = sys_means.sort_values(avail.get("cbs_score", sys_means.columns[0]), ascending=False)

    for rank, (system, row) in enumerate(sys_means.iterrows(), 1):
        cbs  = f"{row.get(avail.get('cbs_score',''), 0.0):.4f}"
        rcf  = f"{row.get(avail.get('rcf_aggregate',''), 0.0):.3f}"
        hpa  = f"{row.get(avail.get('hpa_aggregate',''), 0.0):.3f}"
        msi  = f"{row.get(avail.get('msi_aggregate',''), 0.0):.3f}"
        sns  = f"{row.get(avail.get('sns_aggregate',''), 0.0):.3f}"
        ip   = f"{row.get(avail.get('ip_aggregate',''), 0.0):.3f}"
        pdq  = f"{row.get(avail.get('pdq_aggregate',''), 0.0):.3f}"
        n    = int(sys_n.get(system, 0))
        lines.append(f"| {rank} | `{system}` | **{cbs}** | {rcf} | {hpa} | {msi} | {sns} | {ip} | {pdq} | {n} |")

    lines += [
        "",
        "---",
        "",
        "## System Descriptions",
        "",
        "| System | Description | Access |",
        "|--------|-------------|--------|",
    ]
    from co_scientist_adapters import ADAPTER_REGISTRY
    for name, (_, model, env_var, desc) in ADAPTER_REGISTRY.items():
        lines.append(f"| `{name}` | {desc[:60]} | `{env_var}` |")

    lines += [
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "- **CBS** — Composite Battery Science Score (weighted aggregate)",
        "- **RCF** — Reasoning Chain Fidelity: logical step progression + convergence",
        "- **HPA** — Hypothesis-Problem Alignment: semantic coherence with stated failure mode",
        "- **MSI** — Mechanistic Specificity Index: domain vocabulary depth + quantitative grounding",
        "- **SNS** — Scientific Novelty Score: TF-IDF corpus distinctiveness",
        "- **IP**  — Intervention Plausibility: physical feasibility + scalability",
        "- **PDQ** — Problem Decomposition Quality: root cause specificity",
        "",
        f"*BatteryHypoBench v{VERSION} | NeurIPS 2026 Evaluations & Datasets Track*",
    ]

    with open(out_dir / "coscientist_report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[save] Report → {out_dir}/coscientist_report.md")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BatteryHypoBench: Co-Scientist Head-to-Head Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Check which systems you can run:
          python run_coscientist_benchmark.py --list-systems

          # Run all detectable systems on 50 problems:
          python run_coscientist_benchmark.py \\
              --csv /path/to/battery_problem_solution_500.csv \\
              --sample 50 \\
              --openai-api-key $OPENAI_API_KEY \\
              --gemini-api-key $GEMINI_API_KEY

          # Specific systems:
          python run_coscientist_benchmark.py \\
              --csv /path/to/battery_problem_solution_500.csv \\
              --systems gpt-4o o3 gemini-2.5-pro sakana-ai-scientist \\
              --sample 50 --openai-api-key $OPENAI_API_KEY --gemini-api-key $GEMINI_API_KEY

          # Reference only (score ground-truth, zero API calls):
          python run_coscientist_benchmark.py \\
              --csv /path/to/battery_problem_solution_500.csv \\
              --reference-only --sample 500
        """)
    )
    parser.add_argument("--csv", default=None,
                        help="Path to battery_problem_solution_500.csv")
    parser.add_argument("--output", default="results/coscientist_benchmark/",
                        help="Output directory")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of problems to evaluate per system")
    parser.add_argument("--systems", nargs="+", default=None,
                        help=f"Systems to run. Choices: {list(ADAPTER_REGISTRY.keys())}")
    parser.add_argument("--reference-only", action="store_true",
                        help="Score ground-truth only, no generation")
    parser.add_argument("--list-systems", action="store_true",
                        help="List all systems and API key status, then exit")
    # API keys
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--anthropic-api-key", default=None)
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--fh-api-key", default=None,
                        help="FutureHouse API key")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (for ChemDFM)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds between API calls (rate limiting)")

    args = parser.parse_args()

    if args.list_systems:
        list_systems()
        return

    if not args.csv:
        parser.error("--csv is required (unless --list-systems)")

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench v{VERSION}")
    print(f"  Co-Scientist Benchmark")
    print(f"  NeurIPS 2026 Evaluations & Datasets Track")
    print(f"{'='*60}")

    run(args)


if __name__ == "__main__":
    main()
