#!/usr/bin/env python3
"""
run_agentic_systems.py — Separate benchmark for slower agentic co-scientists.

Systems:
  - open-coscientist : jataware tournament loop (~7 calls/row)
  - ai-researcher     : HKUDS AI-Researcher (NeurIPS 2025 Spotlight)
                        Simulated via multi-step LLM pipeline

Run on 500 samples to keep within time budget.
Results are merged with main full_eval results in post-processing.

Usage:
  export GEMINI_API_KEY="your-key"
  python run_agentic_systems.py \
      --csv results_psm_extraction_20260415_005815.csv \
      --sample 500 \
      --output results/agentic_eval/

SLURM: sbatch run_agentic_eval.sh
"""

import argparse, json, os, re, sys, time, pathlib
import statistics
import pandas as pd
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from benchmark import (
    compute_rcf, compute_hpa, compute_msi, compute_ip,
    compute_pdq, compute_cbs, compute_sns_corpus,
)
from full_benchmark import (
    parse_tagged, litellm_call, score_row,
    STRONG_PROMPT, AGG_COLS, print_leaderboard,
    _manual_tournament,
)

# ═══════════════════════════════════════════════════════════════
# AI-Researcher adapter
# Based on HKUDS/AI-Researcher (NeurIPS 2025 Spotlight)
# Replicates the 3-stage pipeline:
#   Stage 1: Resource analysis (what's known about this problem)
#   Stage 2: Gap identification (what's missing)
#   Stage 3: Hypothesis synthesis (novel direction)
# ═══════════════════════════════════════════════════════════════

AI_RESEARCHER_STAGE1 = """\
You are a scientific literature analyst specializing in battery materials.

Analyze the current state of knowledge for this research problem:
PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}
FAILURE MODE: {failure_mode}

Identify:
1. What approaches have been tried to address this failure mode?
2. What are their key limitations?
3. What physical/chemical mechanisms are known to be relevant?

Be specific and concise. Focus on mechanistic understanding."""

AI_RESEARCHER_STAGE2 = """\
You are a scientific gap analyst.

Given this battery research problem and the current state of knowledge:

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}

CURRENT STATE OF KNOWLEDGE:
{stage1_output}

Identify:
1. The most critical unresolved mechanistic gap
2. What intervention has NOT been tried but is physically motivated
3. What the key enabling insight would be

Be specific. One gap, one untried approach, one insight."""

AI_RESEARCHER_STAGE3 = """\
You are a battery materials scientist synthesizing a novel hypothesis.

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}
FAILURE MODE: {failure_mode}

KNOWLEDGE GAP ANALYSIS:
{stage2_output}

Now synthesize a complete research hypothesis. Use this EXACT format:

[HYPOTHESIS]
A precise, falsifiable hypothesis (2-3 sentences) that directly addresses
the knowledge gap identified above.
[/HYPOTHESIS]

[INTERVENTION]
The specific material, process, or modification proposed.
[/INTERVENTION]

[MECHANISM]
The physical/chemical mechanism explaining why the intervention works,
grounded in the gap analysis above.
[/MECHANISM]

[REASONING]
[Begin Step 1] Connect problem to known failure mechanism [End Step 1]
[Begin Step 2] Identify gap in existing approaches [End Step 2]
[Begin Step 3] Motivate the new intervention physically [End Step 3]
[Begin Step 4] Explain mechanism by which it addresses the failure [End Step 4]
[Begin Step 5] Arrive at testable hypothesis with expected outcome [End Step 5]
[/REASONING]

[TARGET_PROPERTY]
The specific measurable property this hypothesis aims to improve.
[/TARGET_PROPERTY]

[CLAIMED_OUTCOME]
The quantified or specific expected improvement.
[/CLAIMED_OUTCOME]"""


def generate_ai_researcher(row: dict, sleep: float = 1.5) -> dict:
    """
    AI-Researcher style 3-stage pipeline:
    Stage 1: Literature state analysis
    Stage 2: Gap identification
    Stage 3: Hypothesis synthesis

    Replicates the core reasoning pipeline of HKUDS/AI-Researcher
    (NeurIPS 2025 Spotlight) adapted for hypothesis generation
    without requiring the full web-scraping infrastructure.
    """
    problem     = str(row.get("problem_statement",""))[:700]
    battery_sys = str(row.get("battery_system",""))[:100]
    failure     = str(row.get("failure_mode_or_limitation",""))[:200]

    # Stage 1: Resource/knowledge analysis
    s1_raw = litellm_call(
        AI_RESEARCHER_STAGE1.format(
            problem=problem,
            battery_system=battery_sys,
            failure_mode=failure,
        ),
        sleep=sleep,
    )
    time.sleep(sleep)

    # Stage 2: Gap identification
    s2_raw = litellm_call(
        AI_RESEARCHER_STAGE2.format(
            problem=problem,
            battery_system=battery_sys,
            stage1_output=s1_raw[:600],
        ),
        sleep=sleep,
    )
    time.sleep(sleep)

    # Stage 3: Hypothesis synthesis
    s3_raw = litellm_call(
        AI_RESEARCHER_STAGE3.format(
            problem=problem,
            battery_system=battery_sys,
            failure_mode=failure,
            stage2_output=s2_raw[:500],
        ),
        sleep=sleep,
    )
    time.sleep(sleep)

    result = parse_tagged(s3_raw)
    result["_stage1_analysis"] = s1_raw[:300]
    result["_stage2_gaps"]     = s2_raw[:300]
    result["_n_stages"]        = 3
    return result


# ═══════════════════════════════════════════════════════════════
# Open Co-Scientist adapter (same as full_benchmark but isolated)
# ═══════════════════════════════════════════════════════════════

def generate_open_coscientist(row: dict, sleep: float = 1.5) -> dict:
    """Open Co-Scientist tournament loop."""
    try:
        from coscientist.engine import CoscientistEngine
        problem_stmt = str(row.get("problem_statement",""))[:800]
        battery_sys  = str(row.get("battery_system",""))[:100]
        failure_mode = str(row.get("failure_mode_or_limitation",""))[:200]
        research_goal = (
            f"Generate a novel, testable battery materials hypothesis.\n"
            f"PROBLEM: {problem_stmt}\n"
            f"BATTERY SYSTEM: {battery_sys}\n"
            f"FAILURE MODE: {failure_mode}\n"
            f"Include: specific intervention, named mechanism, "
            f"step-by-step reasoning, target property, quantified outcome."
        )
        engine = CoscientistEngine(
            model_name="gemini/gemini-2.5-flash",
            num_hypotheses=3,
            num_review_rounds=1,
            enable_literature=False,
        )
        result = engine.generate_hypotheses(research_goal)
        best = sorted(result,
                      key=lambda h: getattr(h,"elo",0),
                      reverse=True)[0]
        raw = (
            f"[HYPOTHESIS] {getattr(best,'hypothesis','')} [/HYPOTHESIS]\n"
            f"[INTERVENTION] {getattr(best,'intervention','')} [/INTERVENTION]\n"
            f"[MECHANISM] {getattr(best,'mechanism','')} [/MECHANISM]\n"
            f"[REASONING]\n"
            + "\n".join(
                f"[Begin Step {i+1}] {s} [End Step {i+1}]"
                for i,s in enumerate(getattr(best,"reasoning_steps",[]))
            )
            + "\n[/REASONING]\n"
            f"[TARGET_PROPERTY] {getattr(best,'target_property','')} [/TARGET_PROPERTY]\n"
            f"[CLAIMED_OUTCOME] {getattr(best,'claimed_outcome','')} [/CLAIMED_OUTCOME]"
        )
        return parse_tagged(raw)
    except Exception as e:
        print(f"    [open-coscientist] pkg failed ({str(e)[:50]}), "
              f"using manual tournament...")
        return _manual_tournament(row, n_candidates=3, sleep=sleep)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

AGENTIC_SYSTEMS = {
    "open-coscientist": generate_open_coscientist,
    "ai-researcher":    generate_ai_researcher,
}


def run(args):
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_system").mkdir(exist_ok=True)
    pathlib.Path("logs").mkdir(exist_ok=True)

    # Load
    print(f"\n[load] {args.csv}")
    df = pd.read_csv(args.csv)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)

    sample_df = df.sample(
        n=min(args.sample, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"[load] {len(df)} total → {len(sample_df)} sampled")

    os.environ["GEMINI_API_KEY"] = (
        args.gemini_key or os.environ.get("GEMINI_API_KEY",""))

    systems_to_run = args.systems or list(AGENTIC_SYSTEMS.keys())
    all_rows = []

    # Score reference on same sample
    print(f"\n[REFERENCE] Scoring {len(sample_df)} ground-truth rows...")
    for _, row in sample_df.iterrows():
        d = row.to_dict()
        d["_system"] = "REFERENCE"
        d.update(score_row(pd.Series(d)))
        all_rows.append(d)

    # Run agentic systems
    for system_name in systems_to_run:
        gen_fn = AGENTIC_SYSTEMS.get(system_name)
        if gen_fn is None:
            print(f"[skip] Unknown system: {system_name}")
            continue

        print(f"\n{'─'*60}\n[{system_name}]\n{'─'*60}")
        sys_rows = []
        errors = 0

        for i, (_, row) in enumerate(sample_df.iterrows()):
            pid = str(row.get("doi", f"row_{i}"))[:35]
            print(f"  [{i+1:4d}/{len(sample_df)}] {pid}...",
                  end=" ", flush=True)
            try:
                gen = gen_fn(row.to_dict(), sleep=args.sleep)
                d = row.to_dict()
                d.update(gen)
                d["_system"] = system_name
                d.update(score_row(pd.Series(d)))
                d["cbs_score"] = compute_cbs(
                    {c: d.get(c, 0.5) for c in AGG_COLS if c in d})
                sys_rows.append(d)
                print(f"CBS={d['cbs_score']:.3f}")
            except Exception as e:
                errors += 1
                print(f"ERR: {str(e)[:60]}")
                d = row.to_dict()
                d["_system"] = system_name
                d["error"] = str(e)
                for c in AGG_COLS:
                    d[c] = 0.0
                d["cbs_score"] = 0.0
                sys_rows.append(d)
                if errors > 10:
                    print(f"  [skip] >10 errors, stopping {system_name}")
                    break

        all_rows.extend(sys_rows)
        sys_safe = system_name.replace("/","_")
        pd.DataFrame(sys_rows).to_csv(
            out/"per_system"/f"{sys_safe}.csv", index=False)
        valid = [r for r in sys_rows if r.get("cbs_score",0) > 0]
        if valid:
            m = statistics.mean([r["cbs_score"] for r in valid])
            print(f"  → {len(valid)}/{len(sys_rows)} valid | "
                  f"Mean CBS: {m:.4f}")

    # Compile
    results = pd.DataFrame(all_rows)

    # SNS per system
    for sys in results["_system"].unique():
        mask = results["_system"] == sys
        sub = results[mask].copy()
        if len(sub) >= 5 and "hypothesis" in sub.columns:
            try:
                sns_df = compute_sns_corpus(sub)
                for col in sns_df.columns:
                    results.loc[mask, col] = sns_df[col].values
                results.loc[mask, "cbs_score"] = results[mask].apply(
                    lambda r: compute_cbs(
                        {c: r.get(c,0.5) for c in AGG_COLS
                         if pd.notna(r.get(c, float("nan")))}),
                    axis=1)
            except Exception as e:
                print(f"  [sns warn] {sys}: {e}")

    print_leaderboard(results)

    results.to_csv(out/"agentic_results.csv", index=False)
    print(f"\n[save] → {out}/agentic_results.csv")

    # Leaderboard
    if "_system" in results.columns:
        lb = (results.groupby("_system")[["cbs_score"]+AGG_COLS]
                .agg(["mean","std","count"]).round(4))
        lb.columns = ["_".join(c) for c in lb.columns]
        lb.sort_values("cbs_score_mean", ascending=False).to_csv(
            out/"agentic_leaderboard.csv")

    print(f"\n{'='*60}")
    print(f"  Agentic Systems Benchmark DONE")
    print(f"  Systems: {systems_to_run}")
    print(f"  Outputs: {out}/")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(
        description="Agentic Co-Scientist Benchmark (separate job)")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/agentic_eval/")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--systems", nargs="+", default=None,
                   choices=list(AGENTIC_SYSTEMS.keys()))
    p.add_argument("--gemini-key", default=None)
    p.add_argument("--sleep", type=float, default=1.5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
