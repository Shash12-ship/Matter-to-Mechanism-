#!/usr/bin/env python3
"""
full_benchmark.py — BatteryHypoBench Complete Evaluation Pipeline
NeurIPS 2026 Evaluations & Datasets Track

Covers all requirements:
  1. Retrieval-free vs retrieval-enabled comparison
  2. All 6 metrics separately + CBS robustness / alt weights
  3. Error taxonomy (5 failure categories, automated)
  4. Simple baselines (Gemini direct, weak prompt)
  5. Contamination / memorization check
  6. Ablation by battery system / problem type
  7. Qualitative case studies (top-5 examples per failure mode)

Systems evaluated:
  - REFERENCE       : ground-truth from dataset
  - gemini-direct   : plain Gemini 2.5 Flash, same prompt (baseline)
  - gemini-weak     : Gemini with vague/weak prompt (lower baseline)
  - open-coscientist: jataware/open-coscientist, closed-book
  - gemini-retrieval: Gemini 2.5 Flash + Google Search tool (retrieval-enabled)

Usage (login node, no GPU needed):
  export GEMINI_API_KEY="your-key"
  python full_benchmark.py \
      --csv results_psm_extraction_20260415_005815.csv \
      --sample 100 \
      --output results/full_eval/

SLURM (recommended for full 2645 rows):
  sbatch run_full_eval.sh
"""

import argparse, json, os, re, sys, time, math, pathlib
import statistics, collections, textwrap, random
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ── local benchmark metrics ──────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from benchmark import (
    compute_rcf, compute_hpa, compute_msi, compute_ip,
    compute_pdq, compute_cbs, compute_sns_corpus,
)

PYTHON = sys.executable
VERSION = "2.0.0"
AGG_COLS = ["rcf_aggregate","hpa_aggregate","msi_aggregate",
            "sns_aggregate","ip_aggregate","pdq_aggregate"]

# ═══════════════════════════════════════════════════════════════
# GENERATION PROMPTS
# ═══════════════════════════════════════════════════════════════

STRONG_PROMPT = """\
You are an expert battery materials scientist and co-scientist AI.
Given the research problem below, generate a rigorous scientific hypothesis.

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}
COMPONENT: {component}
FAILURE MODE: {failure_mode}

Respond in this EXACT format:
[HYPOTHESIS] ... [/HYPOTHESIS]
[INTERVENTION] ... [/INTERVENTION]
[MECHANISM] ... [/MECHANISM]
[REASONING]
[Begin Step 1] ... [End Step 1]
[Begin Step 2] ... [End Step 2]
[Begin Step 3] ... [End Step 3]
[Begin Step 4] ... [End Step 4]
[Begin Step 5] ... [End Step 5]
[/REASONING]
[TARGET_PROPERTY] ... [/TARGET_PROPERTY]
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]"""

WEAK_PROMPT = """\
Suggest a solution to this battery problem: {problem}
Be brief."""

RETRIEVAL_PROMPT = """\
You are a battery materials scientist with access to current literature.
Search for recent advances in: {battery_system} {failure_mode}
Then generate a hypothesis:

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}
FAILURE MODE: {failure_mode}

Respond with:
[HYPOTHESIS] ... [/HYPOTHESIS]
[INTERVENTION] ... [/INTERVENTION]
[MECHANISM] ... [/MECHANISM]
[REASONING]
[Begin Step 1] ... [End Step 1]
[Begin Step 2] ... [End Step 2]
[Begin Step 3] ... [End Step 3]
[Begin Step 4] ... [End Step 4]
[Begin Step 5] ... [End Step 5]
[/REASONING]
[TARGET_PROPERTY] ... [/TARGET_PROPERTY]
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]"""

# ═══════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════

def parse_tagged(raw: str) -> dict:
    def extract(tag, text):
        m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)
        return m.group(1).strip() if m else ""
    steps = re.findall(r"\[Begin Step \d+\](.*?)\[End Step \d+\]",
                       raw, re.DOTALL)
    hyp = extract("HYPOTHESIS", raw)
    if not hyp:
        # fallback: first substantive paragraph
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
        hyp = paras[0] if paras else raw[:300]
    rp = extract("REASONING", raw)
    if not rp and steps:
        rp = "\n".join(
            f"[Begin Step {i+1}] {s.strip()} [End Step {i+1}]"
            for i, s in enumerate(steps))
    if not rp and not steps:
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 30]
        rp = "\n".join(
            f"[Begin Step {i+1}] {l} [End Step {i+1}]"
            for i, l in enumerate(lines[:5]))
    mech = extract("MECHANISM", raw)
    if not mech:
        mech_sents = [s.strip() for s in re.split(r"[.!?]", raw)
                      if any(k in s.lower() for k in
                             ["because","due to","mechanism","enables",
                              "diffusion","reaction","bond","phase"])]
        mech = ". ".join(mech_sents[:3])
    return {
        "hypothesis": hyp,
        "intervention_or_solution": extract("INTERVENTION", raw),
        "mechanism_or_rationale": mech,
        "reasoning_process": rp,
        "target_property": extract("TARGET_PROPERTY", raw),
        "claimed_outcome": extract("CLAIMED_OUTCOME", raw),
        "num_reasoning_steps": len(steps),
        "raw_output": raw[:1500],
    }

# ═══════════════════════════════════════════════════════════════
# LiteLLM CALL (used by all Gemini-based systems)
# ═══════════════════════════════════════════════════════════════

def litellm_call(prompt: str, model: str = "gemini/gemini-2.5-flash",
                 tools: list = None, sleep: float = 1.0) -> str:
    import litellm
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )
    if tools:
        kwargs["tools"] = tools
    for attempt in range(4):
        try:
            r = litellm.completion(**kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            wait = 2 ** (attempt + 1)
            print(f"    [litellm] {err[:80]} — retry in {wait}s")
            time.sleep(wait)
    return f"ERROR after 4 retries"

# ═══════════════════════════════════════════════════════════════
# SYSTEM 1: Gemini Direct (plain baseline)
# ═══════════════════════════════════════════════════════════════

def generate_gemini_direct(row: dict, sleep: float = 1.0) -> dict:
    prompt = STRONG_PROMPT.format(
        problem=str(row.get("problem_statement",""))[:800],
        battery_system=str(row.get("battery_system",""))[:100],
        component=str(row.get("component",""))[:100],
        failure_mode=str(row.get("failure_mode_or_limitation",""))[:200],
    )
    raw = litellm_call(prompt, sleep=sleep)
    time.sleep(sleep)
    return parse_tagged(raw)

# ═══════════════════════════════════════════════════════════════
# SYSTEM 2: Gemini Weak (deliberately poor prompt — lower baseline)
# ═══════════════════════════════════════════════════════════════

def generate_gemini_weak(row: dict, sleep: float = 1.0) -> dict:
    prompt = WEAK_PROMPT.format(
        problem=str(row.get("problem_statement",""))[:300])
    raw = litellm_call(prompt, sleep=sleep)
    time.sleep(sleep)
    # weak prompt gives unstructured output — parse best-effort
    return parse_tagged(raw)

# ═══════════════════════════════════════════════════════════════
# SYSTEM 3: Open Co-Scientist (jataware, LiteLLM + Gemini)
# Runs the full generate→review→evolve tournament loop
# ═══════════════════════════════════════════════════════════════

def generate_open_coscientist(row: dict, sleep: float = 2.0) -> dict:
    """
    Calls Open Co-Scientist's hypothesis generation pipeline.
    Uses the Python API: CoscientistEngine.generate_hypotheses()
    Falls back to direct LiteLLM if package API is unavailable.
    """
    problem_stmt = str(row.get("problem_statement",""))[:800]
    battery_sys  = str(row.get("battery_system",""))[:100]
    failure_mode = str(row.get("failure_mode_or_limitation",""))[:200]

    research_goal = (
        f"Generate a novel, testable hypothesis for the following "
        f"battery materials research problem.\n\n"
        f"PROBLEM: {problem_stmt}\n"
        f"BATTERY SYSTEM: {battery_sys}\n"
        f"FAILURE MODE: {failure_mode}\n\n"
        f"The hypothesis must: (1) be falsifiable, (2) propose a "
        f"specific intervention with a named mechanism, (3) include "
        f"step-by-step reasoning, (4) specify a target property and "
        f"quantified claimed outcome."
    )

    try:
        from coscientist.engine import CoscientistEngine
        engine = CoscientistEngine(
            model_name="gemini/gemini-2.5-flash",
            num_hypotheses=3,          # generate 3, evolve best
            num_review_rounds=1,       # 1 tournament round (fast)
            enable_literature=False,   # closed-book mode
        )
        result = engine.generate_hypotheses(research_goal)
        # result is a list of Hypothesis objects; take best by Elo
        best = sorted(result, key=lambda h: getattr(h,"elo",0), reverse=True)[0]
        raw = (
            f"[HYPOTHESIS] {getattr(best,'hypothesis','')} [/HYPOTHESIS]\n"
            f"[INTERVENTION] {getattr(best,'intervention','')} [/INTERVENTION]\n"
            f"[MECHANISM] {getattr(best,'mechanism','')} [/MECHANISM]\n"
            f"[REASONING]\n"
            + "\n".join(
                f"[Begin Step {i+1}] {s} [End Step {i+1}]"
                for i, s in enumerate(getattr(best,"reasoning_steps",[]))
            )
            + "\n[/REASONING]\n"
            f"[TARGET_PROPERTY] {getattr(best,'target_property','')} [/TARGET_PROPERTY]\n"
            f"[CLAIMED_OUTCOME] {getattr(best,'claimed_outcome','')} [/CLAIMED_OUTCOME]"
        )
        time.sleep(sleep)
        return parse_tagged(raw)

    except Exception as e:
        # Fallback: replicate tournament manually with 3 LiteLLM calls
        print(f"    [open-coscientist] package API failed ({e}), "
              f"running manual tournament...")
        return _manual_tournament(row, n_candidates=3, sleep=sleep)


def _manual_tournament(row: dict, n_candidates: int = 3,
                        sleep: float = 1.5) -> dict:
    """
    Manual replication of Open Co-Scientist's core loop:
    1. Generate N candidate hypotheses
    2. Pairwise judge picks best
    3. Evolve winner with critique
    """
    problem_stmt = str(row.get("problem_statement",""))[:700]
    battery_sys  = str(row.get("battery_system",""))[:100]
    failure_mode = str(row.get("failure_mode_or_limitation",""))[:200]

    JUDGE_PROMPT = """You are a battery materials expert judge.
Compare these two hypotheses for the problem:
PROBLEM: {problem}

HYPOTHESIS A:
{hyp_a}

HYPOTHESIS B:
{hyp_b}

Which is scientifically stronger? Reply with ONLY 'A' or 'B', then one sentence why."""

    EVOLVE_PROMPT = """You are a battery materials scientist.
Improve this hypothesis by making it more mechanistically specific,
quantitatively grounded, and experimentally testable.

ORIGINAL HYPOTHESIS:
{hyp}

PROBLEM: {problem}
BATTERY SYSTEM: {battery_sys}
FAILURE MODE: {failure_mode}

Respond in the tagged format:
[HYPOTHESIS] ... [/HYPOTHESIS]
[INTERVENTION] ... [/INTERVENTION]
[MECHANISM] ... [/MECHANISM]
[REASONING]
[Begin Step 1] ... [End Step 1]
[Begin Step 2] ... [End Step 2]
[Begin Step 3] ... [End Step 3]
[Begin Step 4] ... [End Step 4]
[Begin Step 5] ... [End Step 5]
[/REASONING]
[TARGET_PROPERTY] ... [/TARGET_PROPERTY]
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]"""

    # Step 1: Generate N candidates
    candidates = []
    for _ in range(n_candidates):
        raw = litellm_call(
            STRONG_PROMPT.format(
                problem=problem_stmt, battery_system=battery_sys,
                component=str(row.get("component",""))[:100],
                failure_mode=failure_mode),
            sleep=sleep)
        parsed = parse_tagged(raw)
        candidates.append(parsed)
        time.sleep(sleep)

    # Step 2: Tournament — compare pairs, count wins
    wins = [0] * n_candidates
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            hyp_a = candidates[i].get("hypothesis","")[:400]
            hyp_b = candidates[j].get("hypothesis","")[:400]
            judge_raw = litellm_call(
                JUDGE_PROMPT.format(
                    problem=problem_stmt[:400],
                    hyp_a=hyp_a, hyp_b=hyp_b),
                sleep=0.5)
            time.sleep(0.5)
            if judge_raw.strip().upper().startswith("A"):
                wins[i] += 1
            else:
                wins[j] += 1

    best_idx = wins.index(max(wins))
    winner = candidates[best_idx]

    # Step 3: Evolve winner
    evolved_raw = litellm_call(
        EVOLVE_PROMPT.format(
            hyp=winner.get("hypothesis","")[:400],
            problem=problem_stmt, battery_sys=battery_sys,
            failure_mode=failure_mode),
        sleep=sleep)
    time.sleep(sleep)
    evolved = parse_tagged(evolved_raw)
    evolved["_tournament_wins"] = wins[best_idx]
    evolved["_n_candidates"] = n_candidates
    return evolved

# ═══════════════════════════════════════════════════════════════
# SYSTEM 4: Gemini + Retrieval (Google Search tool)
# ═══════════════════════════════════════════════════════════════

def generate_gemini_retrieval(row: dict, sleep: float = 2.0) -> dict:
    """
    Gemini 2.5 Flash with Google Search grounding tool enabled.
    This is the retrieval-enabled condition.
    Uses google-generativeai native SDK (not LiteLLM)
    because LiteLLM doesn't expose google_search_retrieval tool.
    """
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))

    prompt = RETRIEVAL_PROMPT.format(
        problem=str(row.get("problem_statement",""))[:700],
        battery_system=str(row.get("battery_system",""))[:100],
        failure_mode=str(row.get("failure_mode_or_limitation",""))[:200],
    )

    try:
        from google.genai import types as genai_types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY",""))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(
                    google_search=genai_types.GoogleSearch()
                )]
            )
        )
        raw = response.text or ""
        # capture grounding metadata if available
        grounding = ""
        if hasattr(response, "candidates"):
            for cand in response.candidates:
                if hasattr(cand, "grounding_metadata"):
                    gm = cand.grounding_metadata
                    if hasattr(gm, "search_entry_point"):
                        grounding = str(gm.search_entry_point)[:200]
        time.sleep(sleep)
        result = parse_tagged(raw)
        result["_retrieval_grounding"] = grounding
        result["_retrieval_enabled"] = True
        return result
    except Exception as e:
        print(f"    [retrieval] {e} — falling back to direct")
        result = generate_gemini_direct(row, sleep=sleep)
        result["_retrieval_enabled"] = False
        return result

# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def score_row(row: pd.Series) -> dict:
    scores = {}
    scores.update(compute_rcf(row))
    scores.update(compute_hpa(row))
    scores.update(compute_msi(row))
    scores.update(compute_ip(row))
    scores.update(compute_pdq(row))
    return scores

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: CBS ROBUSTNESS (alt weight schemes)
# ═══════════════════════════════════════════════════════════════

ALT_WEIGHTS = {
    "default":          [0.20, 0.20, 0.18, 0.15, 0.15, 0.12],
    "uniform":          [1/6,  1/6,  1/6,  1/6,  1/6,  1/6 ],
    "reasoning_heavy":  [0.40, 0.15, 0.15, 0.10, 0.10, 0.10],
    "novelty_heavy":    [0.10, 0.15, 0.15, 0.40, 0.10, 0.10],
    "mechanism_heavy":  [0.15, 0.15, 0.40, 0.10, 0.10, 0.10],
    "plausibility_heavy":[0.15, 0.15, 0.15, 0.10, 0.35, 0.10],
}

def compute_cbs_weighted(row: dict, weights: list) -> float:
    vals = [row.get(c, 0.0) for c in AGG_COLS]
    vals = [v if pd.notna(v) else 0.0 for v in vals]
    total_w = sum(weights)
    return round(sum(w*v for w,v in zip(weights, vals)) / total_w, 4)

def robustness_analysis(df: pd.DataFrame) -> dict:
    """Compute CBS under all weight schemes, check rank stability."""
    results = {}
    for scheme, weights in ALT_WEIGHTS.items():
        col = f"cbs_{scheme}"
        df[col] = df.apply(
            lambda r: compute_cbs_weighted(r.to_dict(), weights), axis=1)

    if "_system" in df.columns:
        system_ranks = {}
        for scheme in ALT_WEIGHTS:
            col = f"cbs_{scheme}"
            ranked = (df.groupby("_system")[col]
                        .mean()
                        .sort_values(ascending=False)
                        .index.tolist())
            system_ranks[scheme] = ranked

        # Kendall tau between default and each alt scheme
        default_ranks = system_ranks.get("default", [])
        tau_scores = {}
        for scheme, ranks in system_ranks.items():
            if scheme == "default":
                continue
            # align
            common = [s for s in default_ranks if s in ranks]
            r1 = [default_ranks.index(s) for s in common]
            r2 = [ranks.index(s) for s in common]
            if len(r1) > 1:
                from scipy.stats import kendalltau
                tau, p = kendalltau(r1, r2)
                tau_scores[scheme] = {"tau": round(tau,4), "p": round(p,4)}

        results["rank_stability"] = {
            "rankings_by_scheme": system_ranks,
            "kendall_tau_vs_default": tau_scores,
        }

    # Metric correlations (Spearman)
    corr_results = {}
    for i, ca in enumerate(AGG_COLS):
        for cb in AGG_COLS[i+1:]:
            if ca in df.columns and cb in df.columns:
                vals_a = df[ca].dropna()
                vals_b = df[cb].dropna()
                idx = vals_a.index.intersection(vals_b.index)
                if len(idx) > 10:
                    rho, p = spearmanr(vals_a[idx], vals_b[idx])
                    corr_results[f"{ca}_x_{cb}"] = {
                        "rho": round(rho,4), "p": round(p,6)}
    results["metric_correlations"] = corr_results
    return results

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: ERROR TAXONOMY
# ═══════════════════════════════════════════════════════════════

ERROR_CATEGORIES = {
    "high_alignment_weak_mechanism": {
        "desc": "High HPA but low MSI — hypothesis addresses the problem "
                "but lacks mechanistic depth",
        "condition": lambda r: (r.get("hpa_aggregate",0) > 0.25 and
                                r.get("msi_aggregate",0) < 0.15),
    },
    "novel_but_implausible": {
        "desc": "High SNS but low IP — novel-sounding but physically implausible",
        "condition": lambda r: (r.get("sns_aggregate",0) > 0.75 and
                                r.get("ip_aggregate",0) < 0.40),
    },
    "plausible_wrong_target": {
        "desc": "High IP but low HPA — plausible intervention but "
                "misaligned with stated failure mode",
        "condition": lambda r: (r.get("ip_aggregate",0) > 0.55 and
                                r.get("hpa_aggregate",0) < 0.10),
    },
    "verbose_weak_decomposition": {
        "desc": "High RCF but low PDQ — verbose reasoning chain but "
                "vague problem decomposition",
        "condition": lambda r: (r.get("rcf_aggregate",0) > 0.65 and
                                r.get("pdq_aggregate",0) < 0.45),
    },
    "literature_like_low_novelty": {
        "desc": "High MSI but low SNS — mechanistically rich but "
                "not corpus-novel (likely literature-like)",
        "condition": lambda r: (r.get("msi_aggregate",0) > 0.25 and
                                r.get("sns_aggregate",0) < 0.65),
    },
}

def classify_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Assign error category to each row (can have multiple)."""
    for cat, spec in ERROR_CATEGORIES.items():
        df[f"err_{cat}"] = df.apply(
            lambda r: spec["condition"](r.to_dict()), axis=1)
    df["error_categories"] = df.apply(
        lambda r: [c for c in ERROR_CATEGORIES
                   if r.get(f"err_{c}", False)],
        axis=1)
    df["n_error_flags"] = df["error_categories"].apply(len)
    return df

def error_taxonomy_report(df: pd.DataFrame) -> dict:
    """Summarise error taxonomy per system."""
    report = {}
    for cat, spec in ERROR_CATEGORIES.items():
        col = f"err_{cat}"
        if col not in df.columns:
            continue
        per_system = {}
        if "_system" in df.columns:
            for sys in df["_system"].unique():
                mask = df["_system"] == sys
                rate = df[mask][col].mean()
                per_system[sys] = round(rate, 4)
        report[cat] = {
            "description": spec["desc"],
            "overall_rate": round(df[col].mean(), 4),
            "per_system": per_system,
        }
    return report

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: CONTAMINATION CHECK
# ═══════════════════════════════════════════════════════════════

def contamination_check(df: pd.DataFrame) -> dict:
    """
    Proxy contamination check:
    1. Year split — 2024+ papers less likely in pretraining
    2. DOI hash check — papers with unusual DOI patterns
    3. Keyword rarity — hypotheses with very rare keyword combos
       (high-specificity terms unlikely to be memorized)
    """
    report = {}

    # Year split if available
    if "year" in df.columns:
        df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
        recent = df[df["year_num"] >= 2024]
        older  = df[df["year_num"] <  2024]
        if len(recent) > 10 and len(older) > 10:
            for col in ["cbs_score"] + AGG_COLS:
                if col in df.columns:
                    r_mean = recent[col].mean()
                    o_mean = older[col].mean()
                    report[f"year_split_{col}"] = {
                        "recent_2024plus_mean": round(r_mean,4),
                        "older_mean":           round(o_mean,4),
                        "delta":                round(r_mean - o_mean, 4),
                    }
            report["recent_n"] = len(recent)
            report["older_n"]  = len(older)

    # Specificity of hypothesis vocabulary as memorization proxy
    # High-specificity unique terms → harder to memorize
    rare_terms = [
        "jahn-teller", "butler-volmer", "operando", "galvanostatic",
        "tortuosity", "wettability", "sei formation", "cryo-tem",
        "neutron diffraction", "aimd", "gitt", "pitt",
    ]
    if "hypothesis" in df.columns:
        df["rare_term_count"] = df["hypothesis"].apply(
            lambda h: sum(1 for t in rare_terms if t in str(h).lower()))
        report["rare_term_stats"] = {
            "mean": round(df["rare_term_count"].mean(), 3),
            "pct_zero": round((df["rare_term_count"]==0).mean(), 3),
            "pct_two_plus": round((df["rare_term_count"]>=2).mean(), 3),
        }
        report["note"] = (
            "High rare_term_count suggests mechanistic specificity "
            "unlikely to arise from simple memorization. "
            "Reference-free metrics further mitigate contamination risk."
        )
    return report

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: ABLATION BY BATTERY SYSTEM / PROBLEM TYPE
# ═══════════════════════════════════════════════════════════════

def ablation_analysis(df: pd.DataFrame) -> dict:
    report = {}
    score_cols = ["cbs_score"] + AGG_COLS
    score_cols = [c for c in score_cols if c in df.columns]

    for group_col in ["battery_system", "problem_type_broad",
                      "component", "problem_type_fine"]:
        if group_col not in df.columns:
            continue
        group_by_cols = ([group_col, "_system"]
                         if "_system" in df.columns else [group_col])
        gb = (df.groupby(group_by_cols)[score_cols]
                .mean().round(4)
                .reset_index())
        # Convert to list of dicts (avoids tuple-key JSON issue)
        report[group_col] = gb.to_dict(orient="records")

    # Best/worst battery systems by CBS gap (model vs reference)
    if "_system" in df.columns and "battery_system" in df.columns:
        ref  = df[df["_system"]=="REFERENCE"]
        gen  = df[df["_system"]!="REFERENCE"]
        if len(ref) > 0 and len(gen) > 0 and "cbs_score" in df.columns:
            ref_sys  = ref.groupby("battery_system")["cbs_score"].mean()
            gen_sys  = gen.groupby("battery_system")["cbs_score"].mean()
            common   = ref_sys.index.intersection(gen_sys.index)
            gap      = (gen_sys[common] - ref_sys[common]).sort_values()
            report["cbs_gap_by_system"] = {
                "worst_gap_systems": gap.head(5).to_dict(),
                "best_gap_systems":  gap.tail(5).to_dict(),
            }
    return report

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: QUALITATIVE CASE STUDIES
# ═══════════════════════════════════════════════════════════════

def build_case_studies(df: pd.DataFrame, n_per_category: int = 2) -> list:
    """
    For each error category, find examples where:
    - REFERENCE has the error flag = False (good reference)
    - generated system has error flag = True (bad generation)
    Returns side-by-side comparison dicts.
    """
    studies = []
    if "_system" not in df.columns:
        return studies

    ref_df = df[df["_system"]=="REFERENCE"].copy()
    gen_df = df[df["_system"]!="REFERENCE"].copy()

    for cat in ERROR_CATEGORIES:
        err_col = f"err_{cat}"
        if err_col not in gen_df.columns:
            continue
        # Find generated rows flagged with this error
        bad_gen = gen_df[gen_df[err_col]==True]
        if len(bad_gen) == 0:
            continue
        # Take up to n_per_category examples
        samples = bad_gen.nsmallest(
            min(n_per_category, len(bad_gen)), "cbs_score")
        for _, gen_row in samples.iterrows():
            # Find matching reference row (same doi if available)
            doi = gen_row.get("doi","")
            if doi and "doi" in ref_df.columns:
                ref_match = ref_df[ref_df["doi"]==doi]
            else:
                ref_match = pd.DataFrame()
            study = {
                "error_category": cat,
                "error_description": ERROR_CATEGORIES[cat]["desc"],
                "system": gen_row.get("_system",""),
                "battery_system": str(gen_row.get("battery_system",""))[:60],
                "problem": str(gen_row.get("problem_statement",""))[:300],
                "generated_hypothesis":
                    str(gen_row.get("hypothesis",""))[:400],
                "generated_scores": {
                    c: round(float(gen_row.get(c,0)),3)
                    for c in AGG_COLS if c in gen_row
                },
                "generated_cbs": round(float(gen_row.get("cbs_score",0)),3),
            }
            if len(ref_match) > 0:
                ref_row = ref_match.iloc[0]
                study["reference_hypothesis"] = \
                    str(ref_row.get("hypothesis",""))[:400]
                study["reference_scores"] = {
                    c: round(float(ref_row.get(c,0)),3)
                    for c in AGG_COLS if c in ref_row
                }
                study["reference_cbs"] = \
                    round(float(ref_row.get("cbs_score",0)),3)
            studies.append(study)
    return studies

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 6: RETRIEVAL GAP
# ═══════════════════════════════════════════════════════════════

def retrieval_gap_report(df: pd.DataFrame) -> dict:
    """Compare closed-book vs retrieval-enabled on same problems."""
    if "_system" not in df.columns:
        return {}
    closed  = df[df["_system"]=="gemini-direct"]
    retriev = df[df["_system"]=="gemini-retrieval"]
    if len(closed)==0 or len(retriev)==0:
        return {"note": "Need both gemini-direct and gemini-retrieval"}
    report = {}
    for col in ["cbs_score"] + AGG_COLS:
        if col in df.columns:
            c_mean = closed[col].mean()
            r_mean = retriev[col].mean()
            report[col] = {
                "closed_book": round(c_mean,4),
                "retrieval_enabled": round(r_mean,4),
                "delta": round(r_mean - c_mean, 4),
                "pct_improvement": round(100*(r_mean-c_mean)/
                                         (c_mean+1e-9), 2),
            }
    return report

# ═══════════════════════════════════════════════════════════════
# LEADERBOARD PRINTER
# ═══════════════════════════════════════════════════════════════

def print_leaderboard(df: pd.DataFrame):
    if "_system" not in df.columns or "cbs_score" not in df.columns:
        return
    print("\n" + "="*85)
    print(f"  {'SYSTEM':<28} {'CBS':>6} {'RCF':>6} {'HPA':>6} "
          f"{'MSI':>6} {'SNS':>6} {'IP':>6} {'PDQ':>6}  N")
    print("="*85)
    lb = (df.groupby("_system")[["cbs_score"]+AGG_COLS]
            .mean().round(4)
            .sort_values("cbs_score", ascending=False))
    ns = df.groupby("_system")["cbs_score"].count()
    for sys, row in lb.iterrows():
        n = ns.get(sys,0)
        print(f"  {sys:<28} "
              f"{row.get('cbs_score',0):>6.4f} "
              f"{row.get('rcf_aggregate',0):>6.3f} "
              f"{row.get('hpa_aggregate',0):>6.3f} "
              f"{row.get('msi_aggregate',0):>6.3f} "
              f"{row.get('sns_aggregate',0):>6.3f} "
              f"{row.get('ip_aggregate',0):>6.3f} "
              f"{row.get('pdq_aggregate',0):>6.3f}  {n}")
    print("="*85)

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

SYSTEMS = {
    "REFERENCE":        None,
    "gemini-direct":    generate_gemini_direct,
    "gemini-weak":      generate_gemini_weak,
    "open-coscientist": generate_open_coscientist,
    "gemini-retrieval": generate_gemini_retrieval,
}

def run(args):
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_system").mkdir(exist_ok=True)
    (out / "analysis").mkdir(exist_ok=True)
    pathlib.Path("logs").mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    print(f"\n[load] {args.csv}")
    df = pd.read_csv(args.csv)
    text_cols = [c for c in df.columns
                 if df[c].dtype == object or c in [
                    "problem_statement","hypothesis","reasoning_process",
                    "mechanism_or_rationale","intervention_or_solution",
                    "claimed_outcome","battery_system","component",
                    "failure_mode_or_limitation","problem_type_broad",
                    "problem_type_fine","problem_core","target_property",
                    "evidence_strength","num_reasoning_steps"]]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "num_reasoning_steps" in df.columns:
        df["num_reasoning_steps"] = pd.to_numeric(
            df["num_reasoning_steps"], errors="coerce").fillna(0).astype(int)

    sample_df = df.sample(
        n=min(args.sample, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"[load] {len(df)} total → {len(sample_df)} sampled")

    api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY","")
    if not api_key:
        print("[warn] No GEMINI_API_KEY — only REFERENCE will run")
    os.environ["GEMINI_API_KEY"] = api_key

    # Determine which systems to run
    if args.systems:
        run_systems = args.systems
    else:
        run_systems = list(SYSTEMS.keys())
        if not api_key:
            run_systems = ["REFERENCE"]

    all_rows = []

    # ── REFERENCE ─────────────────────────────────────────────
    print(f"\n{'─'*60}\n[REFERENCE] Scoring ground-truth hypotheses\n{'─'*60}")
    for _, row in sample_df.iterrows():
        d = row.to_dict()
        d["_system"] = "REFERENCE"
        scores = score_row(pd.Series(d))
        d.update(scores)
        all_rows.append(d)
    ref_cbs = [r.get("cbs_score",
               compute_cbs({c:r.get(c,0.5) for c in AGG_COLS}))
               for r in all_rows]
    # compute CBS for ref rows
    for r in all_rows:
        if "cbs_score" not in r:
            r["cbs_score"] = compute_cbs(
                {c:r.get(c,0.5) for c in AGG_COLS if c in r})
    print(f"  Mean CBS: {statistics.mean([r['cbs_score'] for r in all_rows]):.4f}")

    # ── EACH SYSTEM ───────────────────────────────────────────
    for system_name in run_systems:
        if system_name == "REFERENCE":
            continue
        gen_fn = SYSTEMS.get(system_name)
        if gen_fn is None:
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
                if gen.get("error") and not gen.get("hypothesis"):
                    raise ValueError(gen["error"])
                d = row.to_dict()
                d.update({k:v for k,v in gen.items()})
                d["_system"] = system_name
                scores = score_row(pd.Series(d))
                d.update(scores)
                d["cbs_score"] = compute_cbs(
                    {c:d.get(c,0.5) for c in AGG_COLS if c in d})
                sys_rows.append(d)
                print(f"CBS={d['cbs_score']:.3f}")
            except Exception as e:
                errors += 1
                print(f"ERR: {str(e)[:60]}")
                d = row.to_dict()
                d.update({"_system":system_name,
                           "hypothesis":"","error":str(e)})
                for c in AGG_COLS:
                    d[c] = 0.0
                d["cbs_score"] = 0.0
                sys_rows.append(d)
                if errors > 10:
                    print(f"  [skip] >10 errors, stopping {system_name}")
                    break

        all_rows.extend(sys_rows)
        # Save intermediate
        pd.DataFrame(sys_rows).to_csv(
            out/"per_system"/f"{system_name.replace('/','_')}.csv",
            index=False)
        valid = [r for r in sys_rows if r.get("cbs_score",0) > 0]
        if valid:
            m = statistics.mean([r["cbs_score"] for r in valid])
            print(f"  → {len(valid)}/{len(sys_rows)} valid | "
                  f"Mean CBS: {m:.4f}")

    # ── COMPILE ───────────────────────────────────────────────
    print("\n[compile] Building results dataframe...")
    results = pd.DataFrame(all_rows)

    # SNS corpus-level (per system)
    if "hypothesis" in results.columns and "_system" in results.columns:
        print("[sns] Computing corpus novelty per system...")
        for sys in results["_system"].unique():
            mask = results["_system"]==sys
            sub = results[mask].copy()
            if len(sub) >= 5:
                try:
                    sns_df = compute_sns_corpus(sub)
                    for col in sns_df.columns:
                        results.loc[mask, col] = sns_df[col].values
                    # recompute CBS with SNS
                    results.loc[mask, "cbs_score"] = results[mask].apply(
                        lambda r: compute_cbs(
                            {c:r.get(c,0.5) for c in AGG_COLS
                             if pd.notna(r.get(c,float("nan")))}),
                        axis=1)
                except Exception as e:
                    print(f"  [sns warn] {sys}: {e}")

    # ── ANALYSES ─────────────────────────────────────────────
    print("\n[analysis] Running all analyses...")

    # Error taxonomy
    results = classify_errors(results)
    err_report = error_taxonomy_report(results)

    # Robustness
    rob = robustness_analysis(results)

    # Contamination
    cont = contamination_check(results)

    # Ablation
    abl = ablation_analysis(results)

    # Retrieval gap
    ret_gap = retrieval_gap_report(results)

    # Case studies
    studies = build_case_studies(results, n_per_category=3)

    # ── PRINT LEADERBOARD ─────────────────────────────────────
    print_leaderboard(results)

    # Print retrieval gap summary
    if ret_gap and "cbs_score" in ret_gap:
        print(f"\n[retrieval_gap] CBS: "
              f"closed={ret_gap['cbs_score']['closed_book']:.4f} "
              f"retrieval={ret_gap['cbs_score']['retrieval_enabled']:.4f} "
              f"Δ={ret_gap['cbs_score']['delta']:+.4f}")

    # Print error taxonomy summary
    print("\n[error_taxonomy]")
    for cat, info in err_report.items():
        print(f"  {cat[:40]:<40} overall={info['overall_rate']:.3f}")

    # ── SAVE ─────────────────────────────────────────────────
    results.to_csv(out/"all_results.csv", index=False)
    print(f"\n[save] all_results.csv → {out}/all_results.csv")

    # Leaderboard CSV
    if "_system" in results.columns:
        lb = (results.groupby("_system")[["cbs_score"]+AGG_COLS]
                .agg(["mean","std","count"]).round(4))
        lb.columns = ["_".join(c) for c in lb.columns]
        lb = lb.sort_values("cbs_score_mean", ascending=False)
        lb.to_csv(out/"leaderboard.csv")

    # Alt-weights CBS columns
    results[[c for c in results.columns
             if c.startswith("cbs_")]].describe().round(4).to_csv(
        out/"analysis"/"cbs_alt_weights.csv")

    # Analysis JSONs
    analysis_out = {
        "robustness": rob,
        "contamination": cont,
        "retrieval_gap": ret_gap,
        "error_taxonomy": err_report,
    }
    with open(out/"analysis"/"full_analysis.json","w") as f:
        json.dump(analysis_out, f, indent=2, default=str)

    with open(out/"analysis"/"ablation_by_system.json","w") as f:
        json.dump(abl, f, indent=2, default=str)

    with open(out/"analysis"/"case_studies.json","w") as f:
        json.dump(studies, f, indent=2, default=str)

    # Pretty case studies markdown
    _write_case_studies_md(studies, out/"analysis"/"case_studies.md")

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench v{VERSION} — COMPLETE")
    print(f"  Systems: {run_systems}")
    print(f"  Rows: {len(results)}")
    print(f"  Outputs: {out}/")
    print(f"{'='*60}\n")


def _write_case_studies_md(studies: list, path: pathlib.Path):
    lines = ["# Qualitative Case Studies\n"]
    for i, s in enumerate(studies, 1):
        lines += [
            f"## Case {i}: {s['error_category'].replace('_',' ').title()}",
            f"**System:** `{s['system']}` | "
            f"**Battery:** {s['battery_system']}",
            f"**Error:** {s['error_description']}",
            "",
            f"**Problem:** {s['problem']}",
            "",
            "### Generated Hypothesis",
            s.get("generated_hypothesis",""),
            "",
            f"**Scores:** " + " | ".join(
                f"{k.replace('_aggregate','').upper()}="
                f"{v:.3f}"
                for k,v in s.get("generated_scores",{}).items()),
            f"**CBS:** {s.get('generated_cbs',0):.3f}",
            "",
        ]
        if "reference_hypothesis" in s:
            lines += [
                "### Reference Hypothesis",
                s["reference_hypothesis"],
                "",
                f"**Scores:** " + " | ".join(
                    f"{k.replace('_aggregate','').upper()}="
                    f"{v:.3f}"
                    for k,v in s.get("reference_scores",{}).items()),
                f"**CBS:** {s.get('reference_cbs',0):.3f}",
                "",
            ]
        lines.append("---\n")
    with open(path,"w") as f:
        f.write("\n".join(lines))

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="BatteryHypoBench Full Evaluation Pipeline")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/full_eval/")
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--systems", nargs="+", default=None,
                   choices=list(SYSTEMS.keys()),
                   help="Systems to run (default: all with valid key)")
    p.add_argument("--gemini-key", default=None)
    p.add_argument("--sleep", type=float, default=1.5,
                   help="Seconds between API calls")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
