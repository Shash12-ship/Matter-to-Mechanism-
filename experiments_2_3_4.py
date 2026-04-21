#!/usr/bin/env python3
"""
experiments_2_3_4.py — Validation experiments for BatteryHypoBench

Experiment 2: Generic-metric comparison (BLEU, ROUGE-L, BERTScore, cosine sim)
Experiment 3: Pairwise LLM-judge validation (Gemini Pro + Flash, swap-order)
Experiment 4: Metric-gaming stress test (adversarial outputs)

Usage:
  python experiments_2_3_4.py \
      --results results/analysis_final/combined_results.csv \
      --gemini-key $GEMINI_API_KEY \
      --output results/validation/ \
      --exp 2 3 4

SLURM: sbatch run_experiments.sh
"""

import argparse, json, os, re, sys, time, pathlib, random
import statistics, collections
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from full_benchmark import litellm_call, parse_tagged, AGG_COLS
from benchmark import compute_rcf, compute_hpa, compute_msi, compute_ip, compute_pdq, compute_cbs

# ═══════════════════════════════════════════════════════════════
# INSTALL HELPER — runs once
# ═══════════════════════════════════════════════════════════════

def ensure_packages():
    import subprocess
    pkgs = ["nltk", "rouge-score", "bert-score", "sentence-transformers"]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-","_").split("[")[0])
        except ImportError:
            print(f"[install] {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg,
                           "--break-system-packages", "-q"], check=False)
    # NLTK data
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except Exception:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Generic metric comparison
# ═══════════════════════════════════════════════════════════════

def compute_bleu(hypothesis: str, reference: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        if not ref_tokens or not hyp_tokens:
            return 0.0
        sf = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens,
                              smoothing_function=sf)
    except Exception:
        return 0.0


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_bertscore_batch(hypotheses: list[str],
                             references: list[str],
                             device: str = "cpu") -> list[float]:
    from bert_score import score as bert_score_fn
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        P, R, F = bert_score_fn(
            hypotheses, references,
            lang="en", device=dev,
            model_type="distilbert-base-uncased",
            verbose=False,
            batch_size=32,
        )
        return F.tolist()
    except Exception as e:
        print(f"[bertscore] error: {e}")
        return [0.0] * len(hypotheses)


def compute_cosine_batch(hypotheses: list[str],
                          references: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        hyp_emb = model.encode(hypotheses, show_progress_bar=False,
                                batch_size=64, normalize_embeddings=True)
        ref_emb = model.encode(references, show_progress_bar=False,
                                batch_size=64, normalize_embeddings=True)
        # Row-wise dot product (already normalized = cosine)
        sims = (hyp_emb * ref_emb).sum(axis=1)
        return sims.tolist()
    except Exception as e:
        print(f"[cosine] error: {e}")
        return [0.0] * len(hypotheses)


def run_experiment2(df: pd.DataFrame, out: pathlib.Path,
                    sample_n: int = 500) -> dict:
    """
    Compare CBS ranking vs BLEU/ROUGE-L/BERTScore/cosine rankings.
    Requires ground-truth hypothesis as reference.
    """
    print(f"\n{'='*60}\nEXPERIMENT 2: Generic Metric Comparison\n{'='*60}")
    ensure_packages()

    out.mkdir(parents=True, exist_ok=True)

    # We need generated rows matched to their reference hypothesis.
    # Join generated systems with REFERENCE on doi.
    if "doi" not in df.columns:
        df["doi"] = df.get("paper_id", pd.Series(range(len(df)))).astype(str)

    ref_df = df[df["_system"] == "REFERENCE"][
        ["doi", "hypothesis"]].rename(
        columns={"hypothesis": "ref_hypothesis"})
    gen_df = df[df["_system"] != "REFERENCE"].copy()

    merged = gen_df.merge(ref_df, on="doi", how="inner")
    if len(merged) == 0:
        print("[exp2] No doi overlap between generated and reference — "
              "using position-based matching")
        # Fallback: pair by position within same battery_system
        ref_pool = df[df["_system"]=="REFERENCE"]["hypothesis"].tolist()
        gen_df = df[df["_system"]!="REFERENCE"].copy()
        gen_df["ref_hypothesis"] = [
            ref_pool[i % len(ref_pool)]
            for i in range(len(gen_df))
        ]
        merged = gen_df
    else:
        print(f"[exp2] Matched {len(merged)} generated rows to references")

    # Sample
    if sample_n and len(merged) > sample_n:
        merged = merged.sample(n=sample_n, random_state=42)
    print(f"[exp2] Computing generic metrics on {len(merged)} rows...")

    hypotheses = merged["hypothesis"].fillna("").astype(str).tolist()
    references = merged["ref_hypothesis"].fillna("").astype(str).tolist()

    # BLEU + ROUGE row-by-row (fast)
    print("  [bleu + rouge]...")
    merged["bleu"]    = [compute_bleu(h, r)
                         for h, r in zip(hypotheses, references)]
    merged["rouge_l"] = [compute_rouge_l(h, r)
                         for h, r in zip(hypotheses, references)]

    # BERTScore batch
    print("  [bertscore]...")
    merged["bertscore"] = compute_bertscore_batch(hypotheses, references)

    # Cosine similarity
    print("  [cosine sim]...")
    merged["cosine_sim"] = compute_cosine_batch(hypotheses, references)

    # Per-system means
    metric_cols = ["cbs_score", "bleu", "rouge_l", "bertscore", "cosine_sim"]
    metric_cols = [c for c in metric_cols if c in merged.columns]
    sys_means = merged.groupby("_system")[metric_cols].mean().round(4)

    print("\n[exp2] Per-system means:")
    print(sys_means.to_string())

    # Rankings
    rankings = {}
    for col in metric_cols:
        ranked = sys_means[col].sort_values(ascending=False).index.tolist()
        rankings[col] = ranked

    print("\n[exp2] Rankings by metric:")
    for col, ranked in rankings.items():
        print(f"  {col:<15}: {ranked}")

    # Kendall tau between CBS ranking and each generic metric
    cbs_ranks = rankings.get("cbs_score", [])
    tau_results = {}
    for col in metric_cols:
        if col == "cbs_score":
            continue
        other_ranks = rankings.get(col, [])
        common = [s for s in cbs_ranks if s in other_ranks]
        if len(common) >= 3:
            r1 = [cbs_ranks.index(s) for s in common]
            r2 = [other_ranks.index(s) for s in common]
            tau, p = kendalltau(r1, r2)
            tau_results[col] = {"tau": round(tau,4), "p": round(p,4),
                                 "n_systems": len(common)}
            print(f"  CBS vs {col:<15}: tau={tau:.3f}  p={p:.4f}")

    # Disagreement examples (top 3 where CBS and BLEU disagree most)
    if "bleu" in merged.columns and "cbs_score" in merged.columns:
        merged["cbs_bleu_gap"] = (merged["cbs_score"] - merged["bleu"]).abs()
        top_disagree = merged.nlargest(3, "cbs_bleu_gap")[
            ["doi","_system","hypothesis","ref_hypothesis",
             "cbs_score","bleu","rouge_l","bertscore"]
        ].to_dict(orient="records")
    else:
        top_disagree = []

    # Save
    merged.to_csv(out/"exp2_generic_metrics.csv", index=False)
    sys_means.to_csv(out/"exp2_system_means.csv")

    result = {
        "system_means": sys_means.to_dict(),
        "rankings": rankings,
        "kendall_tau_vs_cbs": tau_results,
        "disagreement_examples": top_disagree[:3],
    }
    with open(out/"exp2_results.json","w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[exp2] Saved → {out}/exp2_*")
    return result


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Pairwise LLM-judge validation
# ═══════════════════════════════════════════════════════════════

JUDGE_PROMPT = """\
You are an expert battery materials scientist evaluating two AI-generated research hypotheses.

PROBLEM:
{problem}

BATTERY SYSTEM: {battery_system}
FAILURE MODE: {failure_mode}

HYPOTHESIS {label_a}:
{hyp_a}

HYPOTHESIS {label_b}:
{hyp_b}

Evaluate which hypothesis is better on THREE criteria:
1. PROBLEM_ADDRESSED: Which better addresses the stated problem and failure mode?
2. MECHANISTIC_DEPTH: Which is more mechanistically grounded (specific mechanisms, physical reasoning)?
3. SCIENTIFIC_UTILITY: Which would be more useful to a battery researcher?

Respond ONLY with this JSON (no other text):
{{"problem_addressed": "A_or_B", "mechanistic_depth": "A_or_B", "scientific_utility": "A_or_B", "overall": "A_or_B", "confidence": "high/medium/low", "reason": "<10 words max>"}}"""


def judge_pair(problem: str, battery_system: str, failure_mode: str,
               hyp_a: str, hyp_b: str,
               label_a: str, label_b: str,
               model: str, sleep: float = 1.0) -> dict:
    prompt = JUDGE_PROMPT.format(
        problem=problem[:600],
        battery_system=battery_system[:80],
        failure_mode=failure_mode[:150],
        hyp_a=hyp_a[:400],
        hyp_b=hyp_b[:400],
        label_a=label_a,
        label_b=label_b,
    )
    raw = litellm_call(prompt, model=model, sleep=sleep)
    time.sleep(sleep)
    try:
        clean = re.sub(r"```json|```","",raw).strip()
        return json.loads(clean)
    except Exception:
        # Try to extract winner from raw text
        if label_a in raw[:50]:
            return {"overall": label_a, "parse_error": True}
        elif label_b in raw[:50]:
            return {"overall": label_b, "parse_error": True}
        return {"overall": "tie", "parse_error": True, "raw": raw[:100]}


PAIRS_TO_COMPARE = [
    ("chemdfm-8b",       "gemini-direct"),
    ("chemdfm-8b",       "ai-researcher"),
    ("gemini-direct",    "open-coscientist"),
    ("ai-researcher",    "REFERENCE"),
    ("gemini-direct",    "REFERENCE"),
]

JUDGE_MODELS = {
    "gemini-pro":   "gemini/gemini-2.5-pro",
    "gemini-flash": "gemini/gemini-2.5-flash",
}


def run_experiment3(df: pd.DataFrame, out: pathlib.Path,
                    sample_n: int = 150,
                    gemini_key: str = "") -> dict:
    """
    Pairwise LLM-judge with swap-order bias correction.
    """
    print(f"\n{'='*60}\nEXPERIMENT 3: Pairwise LLM-Judge Validation\n{'='*60}")
    out.mkdir(parents=True, exist_ok=True)

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Sample problems common to all systems
    if "doi" in df.columns:
        doi_col = "doi"
    else:
        doi_col = "paper_id"
        df["paper_id"] = df.index.astype(str)

    systems_present = df["_system"].unique().tolist()

    # Find DOIs present in multiple systems
    doi_counts = df.groupby(doi_col)["_system"].nunique()
    common_dois = doi_counts[doi_counts >= 2].index.tolist()
    print(f"[exp3] {len(common_dois)} problems with ≥2 systems (DOI match)")

    # If too few common DOIs, use all unique DOIs and match by position
    if len(common_dois) < 20:
        print("[exp3] Few DOI matches — using per-pair position matching")
        common_dois = df[doi_col].unique().tolist()
        if len(common_dois) > sample_n:
            common_dois = random.sample(common_dois, sample_n)
    elif len(common_dois) > sample_n:
        common_dois = random.sample(common_dois, sample_n)

    all_judgments = []
    results_by_pair = {}

    for sys_a, sys_b in PAIRS_TO_COMPARE:
        if sys_a not in systems_present or sys_b not in systems_present:
            print(f"[exp3] Skip {sys_a} vs {sys_b} — not in data")
            continue

        pair_key = f"{sys_a}_vs_{sys_b}"
        print(f"\n[exp3] Judging: {sys_a} vs {sys_b}")

        df_a = df[df["_system"]==sys_a].reset_index(drop=True)
        df_b = df[df["_system"]==sys_b].reset_index(drop=True)
        df_a_idx = df[df["_system"]==sys_a].set_index(doi_col)
        df_b_idx = df[df["_system"]==sys_b].set_index(doi_col)
        # Try DOI match first, fall back to position match
        common = [d for d in common_dois
                  if d in df_a_idx.index and d in df_b_idx.index][:50]
        use_position = False
        if len(common) < 5:
            print(f"  [info] Only {len(common)} DOI matches — using position match")
            n_common = min(50, len(df_a), len(df_b))
            common = list(range(n_common))
            use_position = True

        if len(common) == 0:
            print(f"  [skip] No common problems")
            continue

        print(f"  {len(common)} common problems, "
              f"2 orders × 2 judges = {len(common)*4} API calls")

        pair_results = []
        for doi in common:
            if use_position:
                row_a = df_a.iloc[int(doi)]
                row_b = df_b.iloc[int(doi)]
            else:
                row_a = df_a_idx.loc[doi]
                row_b = df_b_idx.loc[doi]
                # Handle duplicate DOIs returning DataFrame
                if isinstance(row_a, pd.DataFrame):
                    row_a = row_a.iloc[0]
                if isinstance(row_b, pd.DataFrame):
                    row_b = row_b.iloc[0]
            problem = str(row_a.get("problem_statement",""))[:500]
            battery = str(row_a.get("battery_system",""))[:80]
            failure = str(row_a.get("failure_mode_or_limitation",""))[:150]
            hyp_a = str(row_a.get("hypothesis",""))[:400]
            hyp_b = str(row_b.get("hypothesis",""))[:400]

            for judge_name, judge_model in JUDGE_MODELS.items():
                # Order 1: A first
                j1 = judge_pair(problem, battery, failure,
                                 hyp_a, hyp_b, "A", "B",
                                 judge_model, sleep=1.0)
                winner_1 = sys_a if j1.get("overall")=="A" else (
                    sys_b if j1.get("overall")=="B" else "tie")

                # Order 2: B first (swap)
                j2 = judge_pair(problem, battery, failure,
                                 hyp_b, hyp_a, "A", "B",
                                 judge_model, sleep=1.0)
                # In order 2, "A" = sys_b
                winner_2_raw = j2.get("overall","tie")
                winner_2 = sys_b if winner_2_raw=="A" else (
                    sys_a if winner_2_raw=="B" else "tie")

                # Reconcile
                if winner_1 == winner_2:
                    final_winner = winner_1
                    order_flip = False
                else:
                    final_winner = "tie"
                    order_flip = True

                pair_results.append({
                    "doi": doi,
                    "sys_a": sys_a, "sys_b": sys_b,
                    "judge": judge_name,
                    "winner_order1": winner_1,
                    "winner_order2": winner_2,
                    "final_winner": final_winner,
                    "order_flip": order_flip,
                    "confidence": j1.get("confidence",""),
                })
                all_judgments.append(pair_results[-1])

        # Aggregate for this pair
        pdf = pd.DataFrame(pair_results)
        for judge_name in JUDGE_MODELS:
            sub = pdf[pdf["judge"]==judge_name]
            if len(sub) == 0:
                continue
            wins_a = (sub["final_winner"]==sys_a).sum()
            wins_b = (sub["final_winner"]==sys_b).sum()
            ties   = (sub["final_winner"]=="tie").sum()
            flip_rate = sub["order_flip"].mean()

            # CBS says which system wins
            cbs_a = df[df["_system"]==sys_a]["cbs_score"].mean()
            cbs_b = df[df["_system"]==sys_b]["cbs_score"].mean()
            cbs_winner = sys_a if cbs_a > cbs_b else sys_b

            # Agreement: judge winner matches CBS winner
            judge_winner = sys_a if wins_a > wins_b else (
                sys_b if wins_b > wins_a else "tie")
            cbs_judge_agree = (judge_winner == cbs_winner)

            key = f"{pair_key}_{judge_name}"
            results_by_pair[key] = {
                "pair": f"{sys_a} vs {sys_b}",
                "judge": judge_name,
                "wins_a": int(wins_a),
                "wins_b": int(wins_b),
                "ties": int(ties),
                "n": len(sub),
                "flip_rate": round(flip_rate, 3),
                "judge_winner": judge_winner,
                "cbs_winner": cbs_winner,
                "cbs_judge_agree": cbs_judge_agree,
            }
            print(f"  [{judge_name}] {sys_a}={wins_a} {sys_b}={wins_b} "
                  f"ties={ties} flip={flip_rate:.2f} "
                  f"agree_CBS={'✓' if cbs_judge_agree else '✗'}")

    # Overall agreement stats
    if results_by_pair:
        pro_agrees   = [v["cbs_judge_agree"]
                        for v in results_by_pair.values()
                        if v["judge"]=="gemini-pro"]
        flash_agrees = [v["cbs_judge_agree"]
                        for v in results_by_pair.values()
                        if v["judge"]=="gemini-flash"]
        flip_rates   = [v["flip_rate"]
                        for v in results_by_pair.values()]

        # Pro–Flash agreement: do they pick the same winner per pair?
        jdf = pd.DataFrame(all_judgments)
        pro_flash_agrees = []
        if len(jdf) > 0 and "judge" in jdf.columns:
            for (pair_a, pair_b), grp in jdf.groupby(["sys_a","sys_b"]):
                pro_grp   = grp[grp["judge"]=="gemini-pro"]
                flash_grp = grp[grp["judge"]=="gemini-flash"]
                # Match by doi and compare final_winner
                merged_j = pro_grp[["doi","final_winner"]].merge(
                    flash_grp[["doi","final_winner"]],
                    on="doi", suffixes=("_pro","_flash"))
                if len(merged_j) > 0:
                    agree = (merged_j["final_winner_pro"] ==
                             merged_j["final_winner_flash"]).mean()
                    pro_flash_agrees.append(agree)

        summary = {
            "cbs_gemini_pro_agreement":
                round(np.mean(pro_agrees),3)      if pro_agrees      else None,
            "cbs_gemini_flash_agreement":
                round(np.mean(flash_agrees),3)    if flash_agrees    else None,
            "pro_flash_agreement":
                round(np.mean(pro_flash_agrees),3) if pro_flash_agrees else None,
            "mean_order_flip_rate":
                round(np.mean(flip_rates),3)      if flip_rates      else None,
            "pairs_evaluated": len(set(
                v["pair"] for v in results_by_pair.values())),
            "note": ("Judges are two Gemini-family models (Pro and Flash), "
                     "not independent judges. Agreement rates should be "
                     "interpreted as intra-family consistency, not "
                     "cross-architecture validation."),
        }

        print(f"\n[exp3] Summary:")
        print(f"  CBS–GeminiPro agreement:   {summary['cbs_gemini_pro_agreement']}")
        print(f"  CBS–GeminiFlash agreement: {summary['cbs_gemini_flash_agreement']}")
        print(f"  Pro–Flash agreement:       {summary['pro_flash_agreement']}")
        print(f"  Mean order-flip rate:      {summary['mean_order_flip_rate']}")
        print(f"  Note: {summary['note']}")
    else:
        summary = {}

    # Save
    pd.DataFrame(all_judgments).to_csv(out/"exp3_judgments.csv", index=False)
    result = {"pair_results": results_by_pair, "summary": summary}
    with open(out/"exp3_results.json","w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[exp3] Saved → {out}/exp3_*")
    return result


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Metric-gaming stress test
# ═══════════════════════════════════════════════════════════════

ADVERSARIAL_PROMPTS = {
    "jargon_stuffing": """\
Generate a battery materials hypothesis that SOUNDS highly scientific and mechanistic
but is actually vague and contains no real scientific insight.

Pack it with technical jargon: use terms like SEI, Butler-Volmer, tortuosity,
Coulombic efficiency, impedance spectroscopy, DFT, operando — but don't connect
them meaningfully. Use lots of causal phrases like "thereby", "resulting in",
"which enables" — but the logic should not hold up.

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}

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
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]""",

    "problem_mirroring": """\
Generate a battery materials hypothesis that simply mirrors back the problem statement
and failure mode without adding any new scientific insight or intervention.

Restate the problem in different words, mention the failure mode by name,
and claim improvement without specifying how. Use vague interventions.

PROBLEM: {problem}
FAILURE MODE: {failure_mode}
BATTERY SYSTEM: {battery_system}

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
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]""",

    "verbose_fake_reasoning": """\
Generate a battery materials hypothesis with a very long, structured-sounding
reasoning chain that says very little. Each step should be verbose but add
minimal new information. The final hypothesis should be generic.

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}

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
[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]""",
}


def run_experiment4(df: pd.DataFrame, out: pathlib.Path,
                    sample_n: int = 75,
                    gemini_key: str = "",
                    sleep: float = 1.0) -> dict:
    """
    Generate adversarial outputs and score them to check metric robustness.
    """
    print(f"\n{'='*60}\nEXPERIMENT 4: Metric-Gaming Stress Test\n{'='*60}")
    out.mkdir(parents=True, exist_ok=True)

    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Sample problems
    ref_df = df[df["_system"]=="REFERENCE"].sample(
        n=min(sample_n, len(df[df["_system"]=="REFERENCE"])),
        random_state=42)
    print(f"[exp4] Generating adversarial outputs for "
          f"{len(ref_df)} problems × 3 styles")

    all_rows = []

    # Also score the original reference for comparison
    for _, row in ref_df.iterrows():
        d = row.to_dict()
        d["_system"] = "REFERENCE"
        d["_adv_style"] = "none"
        scores = _score_row(pd.Series(d))
        d.update(scores)
        d["cbs_score"] = compute_cbs(
            {c: d.get(c,0.5) for c in AGG_COLS if c in d})
        all_rows.append(d)

    # Generate each adversarial style
    for style, prompt_template in ADVERSARIAL_PROMPTS.items():
        print(f"\n  [{style}] generating {len(ref_df)} rows...")
        style_rows = []
        for i, (_, row) in enumerate(ref_df.iterrows()):
            prompt = prompt_template.format(
                problem=str(row.get("problem_statement",""))[:600],
                battery_system=str(row.get("battery_system",""))[:100],
                failure_mode=str(row.get("failure_mode_or_limitation",""))[:150],
            )
            raw = litellm_call(prompt,
                                model="gemini/gemini-2.5-flash",
                                sleep=sleep)
            time.sleep(sleep)
            parsed = parse_tagged(raw)

            d = row.to_dict()
            d.update(parsed)
            d["_system"] = f"adversarial_{style}"
            d["_adv_style"] = style
            scores = _score_row(pd.Series(d))
            d.update(scores)
            d["cbs_score"] = compute_cbs(
                {c: d.get(c,0.5) for c in AGG_COLS if c in d})
            style_rows.append(d)

            if (i+1) % 10 == 0:
                print(f"    {i+1}/{len(ref_df)}")

        all_rows.extend(style_rows)
        valid = [r for r in style_rows if r.get("cbs_score",0) > 0]
        if valid:
            m = statistics.mean([r["cbs_score"] for r in valid])
            print(f"    Mean CBS: {m:.4f}")

    results_df = pd.DataFrame(all_rows)

    # Per-style metric means
    score_cols = AGG_COLS + ["cbs_score"]
    score_cols = [c for c in score_cols if c in results_df.columns]

    style_means = results_df.groupby("_adv_style")[score_cols].mean().round(4)
    print("\n[exp4] Adversarial style scores:")
    print(style_means.to_string())

    # Which metrics inflate most vs reference?
    if "none" in style_means.index:
        ref_means = style_means.loc["none"]
        inflation = {}
        for style in ADVERSARIAL_PROMPTS.keys():
            if style in style_means.index:
                delta = style_means.loc[style] - ref_means
                inflation[style] = delta.to_dict()
                print(f"\n  [{style}] inflation vs reference:")
                for metric, val in delta.items():
                    bar = "▲" if val > 0.02 else ("▼" if val < -0.02 else "·")
                    print(f"    {bar} {metric:<25}: {val:+.4f}")
    else:
        inflation = {}

    # Save
    results_df.to_csv(out/"exp4_adversarial.csv", index=False)
    style_means.to_csv(out/"exp4_style_means.csv")

    result = {
        "style_means": style_means.to_dict(),
        "inflation_vs_reference": inflation,
    }
    with open(out/"exp4_results.json","w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[exp4] Saved → {out}/exp4_*")
    return result


def _score_row(row: pd.Series) -> dict:
    scores = {}
    scores.update(compute_rcf(row))
    scores.update(compute_hpa(row))
    scores.update(compute_msi(row))
    scores.update(compute_ip(row))
    scores.update(compute_pdq(row))
    return scores


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def make_plots(out: pathlib.Path, exp2: dict, exp3: dict,
               exp4: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })

    figs_dir = out / "figures"
    figs_dir.mkdir(exist_ok=True)

    # ── Plot E2: Ranking comparison heatmap ───────────────────
    if exp2 and "rankings" in exp2:
        rankings = exp2["rankings"]
        metrics  = list(rankings.keys())
        systems  = rankings.get("cbs_score", [])

        rank_matrix = np.zeros((len(systems), len(metrics)))
        for j, metric in enumerate(metrics):
            ranked = rankings.get(metric, [])
            for i, sys in enumerate(systems):
                if sys in ranked:
                    rank_matrix[i, j] = ranked.index(sys) + 1
                else:
                    rank_matrix[i, j] = len(systems)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.imshow(rank_matrix, cmap="RdYlGn_r",
                       vmin=1, vmax=len(systems), aspect="auto")

        sys_labels = [s.replace("gemini-","G-")
                       .replace("open-coscientist","Open-CoSci")
                       .replace("ai-researcher","AI-Res")
                       .replace("REFERENCE","REF")
                       for s in systems]
        met_labels = [m.replace("cbs_score","CBS")
                       .replace("rouge_l","ROUGE-L")
                       .replace("bertscore","BERTScore")
                       .replace("cosine_sim","Cosine")
                       .replace("bleu","BLEU").upper()
                       for m in metrics]

        for i in range(len(systems)):
            for j in range(len(metrics)):
                ax.text(j, i, f"#{int(rank_matrix[i,j])}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold",
                        color="white" if rank_matrix[i,j] > len(systems)//2
                        else "#333")

        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(sys_labels, fontsize=8.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(met_labels, fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Rank", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        plt.tight_layout(pad=0.5)
        path = figs_dir / "exp2_ranking_heatmap.pdf"
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(str(path).replace(".pdf",".png"),
                    bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[plot] → {path}")

    # ── Plot E3: Judge agreement bar chart ────────────────────
    if exp3 and "pair_results" in exp3:
        pairs = exp3["pair_results"]
        if pairs:
            pair_names, pro_agree, flash_agree, flip_rates = [], [], [], []
            seen_pairs = set()
            for key, v in pairs.items():
                pair = v["pair"]
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    short = pair.replace("gemini-direct","G-Dir") \
                                .replace("chemdfm-8b","ChemDFM") \
                                .replace("ai-researcher","AI-Res") \
                                .replace("open-coscientist","OpenCoSci") \
                                .replace("REFERENCE","REF")
                    pair_names.append(short)
                    # Find pro and flash for this pair
                    pro_key   = f"{key.split('_gemini')[0]}_gemini-pro" \
                                if "gemini-pro" not in key else key
                    flash_key = f"{key.split('_gemini')[0]}_gemini-flash" \
                                if "gemini-flash" not in key else key
                    pro_v   = pairs.get(f"{v['pair'].replace(' vs ','_vs_')}_gemini-pro",  {})
                    flash_v = pairs.get(f"{v['pair'].replace(' vs ','_vs_')}_gemini-flash",{})
                    pro_agree.append(1 if pro_v.get("cbs_judge_agree") else 0)
                    flash_agree.append(1 if flash_v.get("cbs_judge_agree") else 0)
                    flip_rates.append(v.get("flip_rate", 0))

            if pair_names:
                x = np.arange(len(pair_names))
                w = 0.28
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.bar(x - w, pro_agree,   w*0.9, label="CBS–GeminiPro agree",
                       color="#E74C3C", alpha=0.8)
                ax.bar(x,     flash_agree, w*0.9, label="CBS–GeminiFlash agree",
                       color="#3498DB", alpha=0.8)
                ax.bar(x + w, flip_rates,  w*0.9, label="Order-flip rate",
                       color="#95A5A6", alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(pair_names, fontsize=8, rotation=15, ha="right")
                ax.set_ylim(0, 1.15)
                ax.set_ylabel("Rate", fontsize=9)
                ax.legend(fontsize=8, ncol=3, loc="upper right")
                ax.axhline(0.5, color="#333", lw=0.8, ls="--", alpha=0.5)
                plt.tight_layout(pad=0.5)
                path = figs_dir / "exp3_judge_agreement.pdf"
                plt.savefig(path, bbox_inches="tight")
                plt.savefig(str(path).replace(".pdf",".png"),
                            bbox_inches="tight", dpi=200)
                plt.close()
                print(f"[plot] → {path}")

    # ── Plot E4: Gaming stress test heatmap ───────────────────
    if exp4 and "style_means" in exp4:
        sm = exp4["style_means"]
        styles  = [k for k in sm.get("cbs_score",{}).keys()
                   if k != "none"]
        metrics = [k for k in sm.keys()
                   if k in AGG_COLS + ["cbs_score"]]
        if styles and metrics:
            ref_vals = {m: sm[m].get("none", 0.0) for m in metrics}
            matrix = np.zeros((len(styles), len(metrics)))
            for i, style in enumerate(styles):
                for j, metric in enumerate(metrics):
                    adv_val = sm[metric].get(style, 0.0)
                    matrix[i, j] = adv_val - ref_vals[metric]

            fig, ax = plt.subplots(figsize=(7, 2.8))
            im = ax.imshow(matrix, cmap="RdBu_r",
                           vmin=-0.3, vmax=0.3, aspect="auto")
            style_labels = [s.replace("_"," ").title() for s in styles]
            met_labels   = [m.replace("_aggregate","").upper()
                             .replace("CBS_SCORE","CBS")
                             for m in metrics]
            for i in range(len(styles)):
                for j in range(len(metrics)):
                    val = matrix[i,j]
                    color = "white" if abs(val) > 0.15 else "#333"
                    ax.text(j, i, f"{val:+.2f}", ha="center",
                            va="center", fontsize=8, color=color)
            ax.set_yticks(range(len(styles)))
            ax.set_yticklabels(style_labels, fontsize=8.5)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(met_labels, fontsize=8.5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("Δ vs Reference", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            plt.tight_layout(pad=0.5)
            path = figs_dir / "exp4_gaming_heatmap.pdf"
            plt.savefig(path, bbox_inches="tight")
            plt.savefig(str(path).replace(".pdf",".png"),
                        bbox_inches="tight", dpi=200)
            plt.close()
            print(f"[plot] → {path}")

    print(f"\n[plots] All figures → {figs_dir}/")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="BatteryHypoBench Validation Experiments 2, 3, 4")
    p.add_argument("--results", required=True,
                   help="Path to combined_results.csv")
    p.add_argument("--output", default="results/validation/")
    p.add_argument("--gemini-key", default=None)
    p.add_argument("--exp", nargs="+", default=["2","3","4"],
                   choices=["2","3","4"])
    p.add_argument("--sample2", type=int, default=500,
                   help="Sample size for Exp 2 generic metrics")
    p.add_argument("--sample3", type=int, default=150,
                   help="Sample size for Exp 3 LLM judge")
    p.add_argument("--sample4", type=int, default=75,
                   help="Sample size for Exp 4 gaming test")
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY","")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench Validation Experiments")
    print(f"  Running: Exp {', '.join(args.exp)}")
    print(f"{'='*60}")

    print(f"\n[load] {args.results}")
    df = pd.read_csv(args.results)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)
    print(f"[load] {len(df)} rows, "
          f"systems: {df['_system'].unique().tolist()}")

    exp2_result, exp3_result, exp4_result = {}, {}, {}

    if "2" in args.exp:
        exp2_result = run_experiment2(df, out/"exp2", args.sample2)

    if "3" in args.exp:
        if not gemini_key:
            print("[exp3] No GEMINI_API_KEY — skipping LLM judge")
        else:
            exp3_result = run_experiment3(
                df, out/"exp3", args.sample3, gemini_key)

    if "4" in args.exp:
        if not gemini_key:
            print("[exp4] No GEMINI_API_KEY — skipping gaming test")
        else:
            exp4_result = run_experiment4(
                df, out/"exp4", args.sample4, gemini_key, args.sleep)

    # Make plots
    make_plots(out, exp2_result, exp3_result, exp4_result)

    print(f"\n{'='*60}")
    print(f"  All experiments done → {out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
