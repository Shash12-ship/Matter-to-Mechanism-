#!/usr/bin/env python3
"""
BatteryHypoBench: A Multi-Dimensional Benchmark for Co-Scientist Hypothesis Generation
in Battery Materials Research

NeurIPS 2026 Evaluations & Datasets Track Submission
Author: Shashwat Sourav (Washington University in St. Louis / ORNL GRO)

Usage:
    python benchmark.py --csv /path/to/battery_problem_solution_500.csv
    python benchmark.py --csv /path/to/data.csv --models gpt-4o claude-3-5-sonnet-20241022
    python benchmark.py --csv /path/to/data.csv --metrics all --output results/
    python benchmark.py --help
"""

import argparse
import json
import os
import sys
import csv
import re
import math
import time
import hashlib
import statistics
import collections
import textwrap
import random
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr, kendalltau

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

VERSION = "1.0.0"

# Battery domain vocabulary for Mechanistic Specificity Index
MECHANISM_KEYWORDS = {
    "high": [
        "diffusion coefficient", "activation energy", "butler-volmer",
        "nernst", "ohmic resistance", "sei formation", "lithiation",
        "delithiation", "intercalation", "dendrite", "tortuosity",
        "solid electrolyte interphase", "charge transfer", "rate constant",
        "overpotential", "exchange current density", "diffusion length",
        "grain boundary", "crystal structure", "lattice parameter",
        "jahn-teller", "spinel", "olivine", "rocksalt", "formation energy",
        "binding energy", "dft", "md simulation", "first principles",
        "galvanostatic", "cyclic voltammetry", "impedance spectroscopy",
        "eis", "xrd", "tem", "xps", "synchrotron", "operando",
        "coulombic efficiency", "capacity retention", "c-rate",
        "voltage hysteresis", "polarization", "wettability",
        "ionic conductivity", "electronic conductivity",
    ],
    "medium": [
        "coating", "doping", "electrolyte", "cathode", "anode",
        "separator", "binder", "conductive additive", "porosity",
        "morphology", "nanostructure", "composite", "interface",
        "degradation", "capacity fade", "cycle life", "thermal stability",
        "safety", "voltage window", "cutoff", "charge/discharge",
        "electrolyte decomposition", "gas evolution", "volume expansion",
    ],
    "low": [
        "improve", "enhance", "increase", "better", "optimize",
        "novel", "promising", "efficient", "effective", "superior",
        "advanced", "innovative", "approach", "strategy", "method",
    ],
}

PROBLEM_TYPE_WEIGHTS = {
    "Mass transport and manufacturing control": 1.2,
    "Electrolyte degradation and interfacial chemistry": 1.3,
    "Structural instability and phase transformation": 1.25,
    "Thermal management and safety": 1.1,
    "Electrochemical kinetics and rate capability": 1.2,
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate the battery problem-solution dataset."""
    print(f"[load] Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[load] Loaded {len(df)} rows, {len(df.columns)} columns")

    required_cols = [
        "paper_id", "problem_statement", "hypothesis",
        "reasoning_process", "num_reasoning_steps",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[warn] Missing expected columns: {missing}")

    # Fill NAs with empty string for text fields
    text_cols = [
        "problem_statement", "hypothesis", "reasoning_process",
        "mechanism_or_rationale", "intervention_or_solution",
        "claimed_outcome", "novelty_axis", "battery_system",
        "component", "failure_mode_or_limitation", "keywords",
        "evidence_strength", "problem_type_broad", "problem_type_fine",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "num_reasoning_steps" in df.columns:
        df["num_reasoning_steps"] = pd.to_numeric(
            df["num_reasoning_steps"], errors="coerce"
        ).fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────
# METRIC 1: REASONING CHAIN FIDELITY (RCF)
# ─────────────────────────────────────────────

def extract_reasoning_steps(reasoning_text: str) -> list[str]:
    """Parse [Begin Step N] ... [End Step N] blocks."""
    pattern = r"\[Begin Step \d+\](.*?)\[End Step \d+\]"
    steps = re.findall(pattern, reasoning_text, re.DOTALL)
    return [s.strip() for s in steps if s.strip()]


def simple_overlap(text_a: str, text_b: str) -> float:
    """Token-level Jaccard overlap as a lightweight similarity proxy."""
    def tokenize(t):
        return set(re.findall(r"\b\w+\b", t.lower()))
    a, b = tokenize(text_a), tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_rcf(row: pd.Series) -> dict:
    """
    Reasoning Chain Fidelity (RCF) — measures how well the reasoning steps:
      1. Progress (forward momentum: each step extends prior)
      2. Converge to hypothesis (final step ~ hypothesis)
      3. Are non-redundant (steps are distinct)
      4. Are logically dense (information per step token)

    Returns a dict of sub-scores and an aggregate RCF ∈ [0,1].
    """
    steps = extract_reasoning_steps(str(row.get("reasoning_process", "")))
    n = len(steps)

    if n == 0:
        return {"rcf_progression": 0.0, "rcf_convergence": 0.0,
                "rcf_nonredundancy": 0.0, "rcf_density": 0.0, "rcf_aggregate": 0.0,
                "rcf_num_steps": 0}

    hypothesis = str(row.get("hypothesis", ""))

    # 1. Progression: average overlap between consecutive steps (want: moderate, not 0 or 1)
    if n > 1:
        overlaps = [simple_overlap(steps[i], steps[i+1]) for i in range(n-1)]
        # Ideal overlap ~0.15-0.35 (builds on prior but adds new content)
        # Score peaks at 0.25, penalizes both 0 and 1
        progression_scores = [
            1.0 - abs(o - 0.25) / 0.75 for o in overlaps
        ]
        rcf_progression = max(0.0, statistics.mean(progression_scores))
    else:
        rcf_progression = 0.5  # single step: neutral

    # 2. Convergence: overlap of last step with hypothesis
    rcf_convergence = simple_overlap(steps[-1], hypothesis)

    # 3. Non-redundancy: 1 - mean pairwise overlap among all steps
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append(simple_overlap(steps[i], steps[j]))
        rcf_nonredundancy = max(0.0, 1.0 - statistics.mean(pairs))
    else:
        rcf_nonredundancy = 1.0

    # 4. Density: words per step normalized (prefer 50-150 words/step)
    word_counts = [len(s.split()) for s in steps]
    avg_words = statistics.mean(word_counts)
    # Sigmoid-like score: peaks around 80 words, decays for very short or very long
    rcf_density = 1.0 / (1.0 + math.exp(-0.05 * (avg_words - 40)))
    rcf_density = min(rcf_density, 1.0 - 1.0 / (1.0 + math.exp(-0.02 * (avg_words - 150))))
    rcf_density = max(0.0, rcf_density)

    rcf_aggregate = 0.3 * rcf_progression + 0.3 * rcf_convergence + \
                    0.25 * rcf_nonredundancy + 0.15 * rcf_density

    return {
        "rcf_progression": round(rcf_progression, 4),
        "rcf_convergence": round(rcf_convergence, 4),
        "rcf_nonredundancy": round(rcf_nonredundancy, 4),
        "rcf_density": round(rcf_density, 4),
        "rcf_aggregate": round(rcf_aggregate, 4),
        "rcf_num_steps": n,
    }


# ─────────────────────────────────────────────
# METRIC 2: HYPOTHESIS-PROBLEM ALIGNMENT (HPA)
# ─────────────────────────────────────────────

def compute_hpa(row: pd.Series) -> dict:
    """
    Hypothesis-Problem Alignment (HPA) — measures semantic coherence between:
      1. Problem scope coverage: does hypothesis address the stated failure mode?
      2. Solution specificity: does intervention match the problem type?
      3. Target property coverage: does hypothesis mention target properties?
      4. Causal directionality: presence of causal language in hypothesis

    All computed without external models — using structural and lexical features.
    """
    problem = str(row.get("problem_statement", "")).lower()
    hypothesis = str(row.get("hypothesis", "")).lower()
    failure_mode = str(row.get("failure_mode_or_limitation", "")).lower()
    intervention = str(row.get("intervention_or_solution", "")).lower()
    target_property = str(row.get("target_property", "")).lower()
    mechanism = str(row.get("mechanism_or_rationale", "")).lower()

    # 1. Scope coverage: hypothesis overlaps with problem tokens
    hpa_scope = simple_overlap(problem, hypothesis)

    # 2. Failure mode addressal: hypothesis mentions failure mode tokens
    if failure_mode:
        hpa_failure_addressal = simple_overlap(failure_mode, hypothesis)
    else:
        hpa_failure_addressal = hpa_scope * 0.8

    # 3. Target property coverage: hypothesis mentions what property improves
    if target_property:
        hpa_target_coverage = simple_overlap(target_property, hypothesis)
    else:
        hpa_target_coverage = 0.5

    # 4. Causal language — does hypothesis use causal connectors?
    causal_patterns = [
        r"\bthereby\b", r"\bthus\b", r"\bleading to\b", r"\bresulting in\b",
        r"\bwhich enables\b", r"\bby\b.*\bincreas", r"\benhancing\b",
        r"\breducing\b", r"\bimproving\b", r"\bwill\b.*\benable\b",
        r"\bprovides\b", r"\ballows\b", r"\bfacilitates\b",
    ]
    causal_hits = sum(1 for p in causal_patterns if re.search(p, hypothesis))
    hpa_causal = min(1.0, causal_hits / 4.0)

    # 5. Intervention-mechanism coherence
    if intervention and mechanism:
        hpa_intervention_coherence = simple_overlap(intervention, mechanism)
    else:
        hpa_intervention_coherence = 0.3

    hpa_aggregate = (
        0.25 * hpa_scope +
        0.25 * hpa_failure_addressal +
        0.15 * hpa_target_coverage +
        0.2 * hpa_causal +
        0.15 * hpa_intervention_coherence
    )

    return {
        "hpa_scope_coverage": round(hpa_scope, 4),
        "hpa_failure_addressal": round(hpa_failure_addressal, 4),
        "hpa_target_coverage": round(hpa_target_coverage, 4),
        "hpa_causal_language": round(hpa_causal, 4),
        "hpa_intervention_coherence": round(hpa_intervention_coherence, 4),
        "hpa_aggregate": round(hpa_aggregate, 4),
    }


# ─────────────────────────────────────────────
# METRIC 3: MECHANISTIC SPECIFICITY INDEX (MSI)
# ─────────────────────────────────────────────

def compute_msi(row: pd.Series) -> dict:
    """
    Mechanistic Specificity Index (MSI) — how mechanistically grounded is the hypothesis?

    Scores:
      1. Vocabulary tier: presence of high/medium/low specificity terms
      2. Quantitative grounding: presence of numbers, units, formulas
      3. Characterization anchoring: reference to experimental techniques
      4. Physical mechanism depth: ratio of mechanism text to hypothesis text
    """
    hypothesis = str(row.get("hypothesis", "")).lower()
    mechanism = str(row.get("mechanism_or_rationale", "")).lower()
    reasoning = str(row.get("reasoning_process", "")).lower()
    combined = hypothesis + " " + mechanism + " " + reasoning

    # 1. Vocabulary tier scoring
    high_hits = sum(1 for kw in MECHANISM_KEYWORDS["high"] if kw in combined)
    med_hits = sum(1 for kw in MECHANISM_KEYWORDS["medium"] if kw in combined)
    low_hits = sum(1 for kw in MECHANISM_KEYWORDS["low"] if kw in combined)
    total_kw = high_hits + med_hits + low_hits + 1e-9
    msi_vocab = (3 * high_hits + 1.5 * med_hits + 0.5 * low_hits) / (
        3 * len(MECHANISM_KEYWORDS["high"]) + 
        1.5 * len(MECHANISM_KEYWORDS["medium"]) + 
        0.5 * len(MECHANISM_KEYWORDS["low"])
    )
    msi_vocab = min(1.0, msi_vocab * 3)  # normalize generously

    # 2. Quantitative grounding: numbers with units
    quant_patterns = [
        r"\d+\.?\d*\s*(nm|µm|mm|cm|m)\b",
        r"\d+\.?\d*\s*(°c|k|kelvin)\b",
        r"\d+\.?\d*\s*(ev|kj|kcal)\b",
        r"\d+\.?\d*\s*(mah|mwh|wh)\b",
        r"\d+\.?\d*\s*(mol|mmol|µmol)\b",
        r"\d+\.?\d*\s*(v|mv|µv)\b",
        r"\d+\.?\d*\s*(ma|µa|a)\s*(cm|g|kg)",
        r"\d+\.?\d*\s*%",
        r"\d+\.?\d*\s*(mg|g|kg)\s*(cm|m|l)",
        r"[×x]\s*10[⁻\-]?\d",
    ]
    quant_hits = sum(1 for p in quant_patterns if re.search(p, combined))
    msi_quantitative = min(1.0, quant_hits / 5.0)

    # 3. Characterization anchoring
    char_techniques = [
        "xrd", "tem", "sem", "xps", "eis", "nmr", "raman", "ftir",
        "dft", "aimd", "synchrotron", "neutron diffraction", "operando",
        "cryo-tem", "saxs", "waxs", "dsc", "tga", "gitt", "pitt",
        "cyclic voltammetry", "galvanostatic", "impedance",
    ]
    char_hits = sum(1 for t in char_techniques if t in combined)
    msi_characterization = min(1.0, char_hits / 4.0)

    # 4. Mechanism depth ratio
    hyp_words = len(hypothesis.split()) + 1
    mech_words = len(mechanism.split())
    msi_depth = min(1.0, mech_words / (hyp_words * 3))

    msi_aggregate = (
        0.35 * msi_vocab +
        0.25 * msi_quantitative +
        0.2 * msi_characterization +
        0.2 * msi_depth
    )

    return {
        "msi_vocab_tier": round(msi_vocab, 4),
        "msi_quantitative": round(msi_quantitative, 4),
        "msi_characterization": round(msi_characterization, 4),
        "msi_mechanism_depth": round(msi_depth, 4),
        "msi_aggregate": round(msi_aggregate, 4),
        "msi_high_kw_count": high_hits,
        "msi_medium_kw_count": med_hits,
    }


# ─────────────────────────────────────────────
# METRIC 4: SCIENTIFIC NOVELTY SCORE (SNS)
# ─────────────────────────────────────────────

def build_tfidf_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build a simple TF-IDF matrix from a list of texts."""
    # Tokenize
    tokenized = [re.findall(r"\b[a-z]{3,}\b", t.lower()) for t in texts]

    # Build vocabulary (top 2000 terms by df)
    df_counts = collections.Counter()
    for toks in tokenized:
        df_counts.update(set(toks))

    vocab = [w for w, _ in df_counts.most_common(2000)]
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    N = len(texts)
    V = len(vocab)

    idf = np.zeros(V)
    for w, cnt in df_counts.items():
        if w in vocab_idx:
            idf[vocab_idx[w]] = math.log((N + 1) / (cnt + 1)) + 1.0

    matrix = np.zeros((N, V))
    for i, toks in enumerate(tokenized):
        tf = collections.Counter(toks)
        total = sum(tf.values()) + 1e-9
        for w, cnt in tf.items():
            if w in vocab_idx:
                matrix[i, vocab_idx[w]] = (cnt / total) * idf[vocab_idx[w]]

    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms

    return matrix, vocab


def compute_sns_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scientific Novelty Score (SNS) — measures distinctiveness of each hypothesis
    relative to the corpus using:
      1. Within-corpus TF-IDF novelty: 1 - max cosine sim to other hypotheses
      2. Problem-type normalized novelty: novelty relative to same battery system
      3. Intervention novelty: distinctiveness of proposed solution
      4. Cross-domain novelty bonus: hypothesis borrows from non-battery fields

    This is a corpus-level metric requiring all rows together.
    """
    print("[sns] Building TF-IDF matrix over hypothesis corpus...")
    hypotheses = df["hypothesis"].tolist()
    hyp_matrix, _ = build_tfidf_matrix(hypotheses)

    # Within-corpus novelty: 1 - mean of top-5 similarities (excluding self)
    N = len(df)
    sim_matrix = hyp_matrix @ hyp_matrix.T  # (N, N)
    np.fill_diagonal(sim_matrix, -1)  # exclude self

    # Sort and take top-5
    top5_sims = np.sort(sim_matrix, axis=1)[:, -5:]
    mean_top5 = np.mean(np.maximum(top5_sims, 0), axis=1)
    sns_corpus_novelty = 1.0 - mean_top5

    # Within-battery-system novelty
    battery_systems = df["battery_system"].tolist() if "battery_system" in df.columns else ["unknown"] * N
    sns_within_system = np.ones(N)
    unique_systems = set(battery_systems)
    for sys in unique_systems:
        idxs = [i for i, s in enumerate(battery_systems) if s == sys]
        if len(idxs) > 1:
            sub_matrix = hyp_matrix[idxs]
            sub_sim = sub_matrix @ sub_matrix.T
            np.fill_diagonal(sub_sim, -1)
            sub_top3 = np.sort(sub_sim, axis=1)[:, -3:]
            sub_mean = np.mean(np.maximum(sub_top3, 0), axis=1)
            for i, orig_i in enumerate(idxs):
                sns_within_system[orig_i] = 1.0 - sub_mean[i]

    # Cross-domain novelty bonus: does the hypothesis contain cross-domain terms?
    cross_domain_terms = [
        "biological", "biomimetic", "geological", "photovoltaic", "neuromorphic",
        "origami", "metamaterial", "topology", "fractal", "quantum", "plasma",
        "textile", "aerogel", "zeolite", "mof", "cof", "bot", "machine learning",
        "deep learning", "neural network", "gasification", "zeolitic",
        "wood-derived", "bio-inspired", "biomass", "silk", "cellulose", "chitin",
    ]
    def cross_domain_score(hyp: str) -> float:
        hyp_lower = hyp.lower()
        hits = sum(1 for t in cross_domain_terms if t in hyp_lower)
        return min(1.0, hits / 2.0)

    sns_cross_domain = df["hypothesis"].apply(cross_domain_score).values

    # Intervention novelty: TF-IDF over intervention field
    if "intervention_or_solution" in df.columns:
        interventions = df["intervention_or_solution"].tolist()
        int_matrix, _ = build_tfidf_matrix(interventions)
        int_sim = int_matrix @ int_matrix.T
        np.fill_diagonal(int_sim, -1)
        int_top3 = np.sort(int_sim, axis=1)[:, -3:]
        int_mean = np.mean(np.maximum(int_top3, 0), axis=1)
        sns_intervention = 1.0 - int_mean
    else:
        sns_intervention = np.ones(N) * 0.5

    sns_aggregate = (
        0.4 * sns_corpus_novelty +
        0.25 * sns_within_system +
        0.2 * sns_intervention +
        0.15 * sns_cross_domain
    )

    result_df = pd.DataFrame({
        "sns_corpus_novelty": np.round(sns_corpus_novelty, 4),
        "sns_within_system_novelty": np.round(sns_within_system, 4),
        "sns_intervention_novelty": np.round(sns_intervention, 4),
        "sns_cross_domain_bonus": np.round(sns_cross_domain, 4),
        "sns_aggregate": np.round(sns_aggregate, 4),
    })
    return result_df


# ─────────────────────────────────────────────
# METRIC 5: INTERVENTION PLAUSIBILITY (IP)
# ─────────────────────────────────────────────

def compute_ip(row: pd.Series) -> dict:
    """
    Intervention Plausibility (IP) — assesses whether the proposed solution is
    scientifically and physically plausible based on:
      1. Physical constraint consistency: no contradictory claims
      2. Material compatibility: intervention compatible with battery system
      3. Scalability signal: presence of practical/scalable language
      4. Evidence grounding: strength of cited evidence
      5. Claimed outcome specificity: quantitative vs. qualitative claims
    """
    intervention = str(row.get("intervention_or_solution", "")).lower()
    mechanism = str(row.get("mechanism_or_rationale", "")).lower()
    hypothesis = str(row.get("hypothesis", "")).lower()
    claimed_outcome = str(row.get("claimed_outcome", "")).lower()
    evidence_strength = str(row.get("evidence_strength", "")).lower()
    battery_system = str(row.get("battery_system", "")).lower()

    # 1. Physical constraint consistency — detect common contradictions
    contradiction_pairs = [
        ("reduce", "increase"),  # shouldn't claim both for same property
        ("stable", "unstable"),
        ("low resistance", "high resistance"),
    ]
    combined = intervention + " " + mechanism + " " + hypothesis
    # Simple: penalize very short interventions (under-specified)
    ip_consistency = 1.0 if len(intervention.split()) > 5 else 0.4

    # 2. Material compatibility heuristic
    system_material_map = {
        "nmc": ["li", "ni", "mn", "co", "oxide", "layered"],
        "lfp": ["fe", "phosphate", "olivine", "iron"],
        "nca": ["ni", "co", "al", "layered"],
        "silicon": ["si", "silicon", "expansion", "volume"],
        "solid state": ["solid", "ceramic", "sulfide", "oxide", "garnet"],
        "lithium metal": ["li metal", "dendrite", "plating"],
    }
    ip_compatibility = 0.5  # default neutral
    for system_key, materials in system_material_map.items():
        if system_key in battery_system:
            hits = sum(1 for m in materials if m in combined)
            ip_compatibility = min(1.0, 0.5 + hits * 0.1)
            break

    # 3. Scalability signal
    scalable_terms = [
        "scalable", "cost-effective", "low-cost", "roll-to-roll", "industrial",
        "commercializ", "mass produc", "pilot", "kg-scale", "ton-scale",
        "solution process", "spray coat", "simple", "facile",
    ]
    non_scalable_terms = [
        "atomic layer deposition", "cvd without scalab", "extremely expensive",
        "requires ultra-high vacuum",
    ]
    scale_hits = sum(1 for t in scalable_terms if t in combined)
    non_scale_hits = sum(1 for t in non_scalable_terms if t in combined)
    ip_scalability = min(1.0, 0.3 + scale_hits * 0.15) - non_scale_hits * 0.2
    ip_scalability = max(0.0, ip_scalability)

    # 4. Evidence grounding
    evidence_map = {
        "strong": 1.0, "high": 1.0, "moderate": 0.65,
        "preliminary": 0.4, "weak": 0.25, "unknown": 0.35, "theoretical": 0.5,
    }
    ip_evidence = 0.35  # default
    for key, val in evidence_map.items():
        if key in evidence_strength:
            ip_evidence = val
            break

    # 5. Claimed outcome specificity
    quant_outcome = bool(re.search(r"\d+\.?\d*\s*(%|×|fold|mah|wh|v\b|°c)", claimed_outcome))
    ip_outcome_specificity = 0.8 if quant_outcome else (
        0.6 if len(claimed_outcome.split()) > 10 else 0.3
    )

    ip_aggregate = (
        0.2 * ip_consistency +
        0.2 * ip_compatibility +
        0.15 * ip_scalability +
        0.25 * ip_evidence +
        0.2 * ip_outcome_specificity
    )

    return {
        "ip_consistency": round(ip_consistency, 4),
        "ip_material_compatibility": round(ip_compatibility, 4),
        "ip_scalability": round(ip_scalability, 4),
        "ip_evidence_grounding": round(ip_evidence, 4),
        "ip_outcome_specificity": round(ip_outcome_specificity, 4),
        "ip_aggregate": round(ip_aggregate, 4),
    }


# ─────────────────────────────────────────────
# METRIC 6: PROBLEM DECOMPOSITION QUALITY (PDQ)
# ─────────────────────────────────────────────

def compute_pdq(row: pd.Series) -> dict:
    """
    Problem Decomposition Quality (PDQ) — evaluates how well the problem is
    identified and decomposed before the hypothesis is formed:
      1. Root cause specificity: does problem_core isolate the failure mechanism?
      2. Scope precision: is the failure mode clearly bounded?
      3. Abstraction consistency: does problem_type align with problem_core?
      4. Component specificity: electrode/material level vs. system level
    """
    problem = str(row.get("problem_statement", "")).lower()
    problem_core = str(row.get("problem_core", "")).lower()
    failure_mode = str(row.get("failure_mode_or_limitation", "")).lower()
    component = str(row.get("component", "")).lower()
    problem_type_broad = str(row.get("problem_type_broad", "")).lower()
    problem_type_fine = str(row.get("problem_type_fine", "")).lower()

    # 1. Root cause specificity: problem_core should be shorter but denser than problem
    prob_words = len(problem.split()) + 1
    core_words = len(problem_core.split())
    # Core should be ~20-40% the length of problem (distilled, not padded)
    ratio = core_words / prob_words
    if 0.15 <= ratio <= 0.5:
        pdq_core_specificity = 1.0
    elif ratio < 0.05:
        pdq_core_specificity = 0.3  # too sparse
    elif ratio > 0.8:
        pdq_core_specificity = 0.4  # not distilled
    else:
        pdq_core_specificity = 0.7

    # 2. Failure mode bounded
    # Good failure modes: specific mechanism (e.g. "lithium plating at high C-rate")
    # Poor: vague ("poor performance")
    vague_failure = ["poor", "bad", "issue", "problem", "challenge", "difficulty", "limitation"]
    specific_failure = [kw for kw in MECHANISM_KEYWORDS["high"] if kw in failure_mode]
    vague_count = sum(1 for v in vague_failure if v in failure_mode)
    pdq_failure_specificity = min(1.0, 0.3 + len(specific_failure) * 0.2 - vague_count * 0.1)
    pdq_failure_specificity = max(0.0, pdq_failure_specificity)

    # 3. Abstraction consistency: fine type should be a specialization of broad type
    # Heuristic: overlap between broad and fine type tokens
    pdq_type_consistency = simple_overlap(problem_type_broad, problem_type_fine)
    pdq_type_consistency = max(0.2, pdq_type_consistency)  # floor at 0.2

    # 4. Component specificity
    specific_components = [
        "cathode", "anode", "electrolyte", "separator", "current collector",
        "sei", "cei", "interface", "grain boundary", "particle", "electrode",
        "binder", "carbon black", "nlp", "active material",
    ]
    component_hits = sum(1 for c in specific_components if c in component)
    pdq_component = min(1.0, component_hits * 0.4 + (0.2 if len(component) > 5 else 0))

    pdq_aggregate = (
        0.3 * pdq_core_specificity +
        0.3 * pdq_failure_specificity +
        0.2 * pdq_type_consistency +
        0.2 * pdq_component
    )

    return {
        "pdq_core_specificity": round(pdq_core_specificity, 4),
        "pdq_failure_specificity": round(pdq_failure_specificity, 4),
        "pdq_type_consistency": round(pdq_type_consistency, 4),
        "pdq_component_granularity": round(pdq_component, 4),
        "pdq_aggregate": round(pdq_aggregate, 4),
    }


# ─────────────────────────────────────────────
# METRIC 7: CO-SCIENTIST GENERATION EVALUATION (CGE)
# LLM-as-judge — optional, requires API key
# ─────────────────────────────────────────────

def call_openai(prompt: str, model: str = "gpt-4o", api_key: str = None) -> str:
    """Call OpenAI API with retry."""
    import urllib.request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', '')}",
    }
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body, headers=headers, method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def call_anthropic(prompt: str, model: str = "claude-opus-4-5", api_key: str = None) -> str:
    """Call Anthropic API with retry."""
    import urllib.request
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body, headers=headers, method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


CGE_JUDGE_PROMPT = """You are a scientific peer reviewer specializing in battery materials and electrochemistry.

Evaluate the following hypothesis generated for a battery materials research problem.

PROBLEM STATEMENT:
{problem}

PROPOSED HYPOTHESIS:
{hypothesis}

PROPOSED INTERVENTION:
{intervention}

Rate the hypothesis on these 5 dimensions (each 0-10):
1. SCIENTIFIC_SOUNDNESS: Is the proposed mechanism physically/chemically valid?
2. FALSIFIABILITY: Can this hypothesis be experimentally tested and falsified?
3. IMPACT_POTENTIAL: If true, would this significantly advance the field?
4. REASONING_QUALITY: Is the logic from problem to hypothesis sound and complete?
5. ORIGINALITY: Is this a genuinely novel approach not previously explored?

Respond ONLY with a JSON object, no other text:
{{"scientific_soundness": <int>, "falsifiability": <int>, "impact_potential": <int>, "reasoning_quality": <int>, "originality": <int>, "brief_justification": "<15 words max>"}}"""


def compute_cge_row(row: pd.Series, model: str, api_type: str, api_key: str) -> dict:
    """Run LLM judge on a single row."""
    prompt = CGE_JUDGE_PROMPT.format(
        problem=str(row.get("problem_statement", ""))[:800],
        hypothesis=str(row.get("hypothesis", ""))[:600],
        intervention=str(row.get("intervention_or_solution", ""))[:400],
    )

    if api_type == "openai":
        response = call_openai(prompt, model=model, api_key=api_key)
    elif api_type == "anthropic":
        response = call_anthropic(prompt, model=model, api_key=api_key)
    else:
        return {"cge_error": f"Unknown API type: {api_type}"}

    try:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", response).strip()
        scores = json.loads(clean)
        cge_aggregate = statistics.mean([
            scores.get("scientific_soundness", 5),
            scores.get("falsifiability", 5),
            scores.get("impact_potential", 5),
            scores.get("reasoning_quality", 5),
            scores.get("originality", 5),
        ]) / 10.0

        return {
            "cge_scientific_soundness": scores.get("scientific_soundness", -1) / 10.0,
            "cge_falsifiability": scores.get("falsifiability", -1) / 10.0,
            "cge_impact_potential": scores.get("impact_potential", -1) / 10.0,
            "cge_reasoning_quality": scores.get("reasoning_quality", -1) / 10.0,
            "cge_originality": scores.get("originality", -1) / 10.0,
            "cge_aggregate": round(cge_aggregate, 4),
            "cge_justification": scores.get("brief_justification", ""),
        }
    except Exception as e:
        return {"cge_error": str(e), "cge_raw": response[:200]}


# ─────────────────────────────────────────────
# METRIC 8: COMPOSITE BATTERY SCIENCE SCORE (CBS)
# ─────────────────────────────────────────────

def compute_cbs(row_metrics: dict) -> float:
    """
    Composite Battery Science Score (CBS) — weighted aggregate of all metrics.
    Weights tuned for NeurIPS scientific contribution framing.
    """
    weights = {
        "rcf_aggregate": 0.20,
        "hpa_aggregate": 0.20,
        "msi_aggregate": 0.18,
        "sns_aggregate": 0.15,
        "ip_aggregate": 0.15,
        "pdq_aggregate": 0.12,
    }
    # Optional CGE component
    if "cge_aggregate" in row_metrics:
        # Rescale CGE to 0-1
        cge_val = row_metrics.get("cge_aggregate", 0.5)
        weights["cge_aggregate"] = 0.10
        # Renormalize
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

    score = 0.0
    total_w = 0.0
    for key, w in weights.items():
        if key in row_metrics:
            score += w * float(row_metrics[key])
            total_w += w

    return round(score / total_w if total_w > 0 else 0.0, 4)


# ─────────────────────────────────────────────
# CROSS-MODEL CONTRASTIVE ANALYSIS
# ─────────────────────────────────────────────

def compute_contrastive_stats(results_df: pd.DataFrame) -> dict:
    """
    Cross-model contrastive analysis — if results contain multiple model outputs,
    compute inter-model agreement, rank correlation, and performance gaps.
    For single-model (reference) dataset, compute intra-dataset statistics.
    """
    stats = {}

    aggregate_cols = [c for c in results_df.columns if c.endswith("_aggregate")]
    
    if len(aggregate_cols) == 0:
        return stats

    # Distribution stats for each metric
    for col in aggregate_cols:
        vals = results_df[col].dropna().values
        if len(vals) > 0:
            stats[col] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "q25": round(float(np.percentile(vals, 25)), 4),
                "q75": round(float(np.percentile(vals, 75)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    # Pairwise Spearman correlation between metrics
    metric_pairs = []
    for i, col_a in enumerate(aggregate_cols):
        for col_b in aggregate_cols[i+1:]:
            vals_a = results_df[col_a].dropna()
            vals_b = results_df[col_b].dropna()
            common_idx = vals_a.index.intersection(vals_b.index)
            if len(common_idx) > 10:
                rho, pval = spearmanr(vals_a.loc[common_idx], vals_b.loc[common_idx])
                metric_pairs.append({
                    "metric_a": col_a, "metric_b": col_b,
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(pval), 4),
                })
    stats["metric_correlations"] = metric_pairs

    # Top/bottom performers
    if "cbs_score" in results_df.columns:
        top10 = results_df.nlargest(10, "cbs_score")[
            ["paper_id", "cbs_score"] + aggregate_cols[:4]
        ].to_dict(orient="records")
        bottom10 = results_df.nsmallest(10, "cbs_score")[
            ["paper_id", "cbs_score"] + aggregate_cols[:4]
        ].to_dict(orient="records")
        stats["top10_papers"] = top10
        stats["bottom10_papers"] = bottom10

    # Battery system breakdown
    if "battery_system" in results_df.columns and "cbs_score" in results_df.columns:
        sys_breakdown = (
            results_df.groupby("battery_system")["cbs_score"]
            .agg(["mean", "std", "count"])
            .round(4)
            .to_dict(orient="index")
        )
        stats["battery_system_breakdown"] = sys_breakdown

    # Problem type breakdown
    if "problem_type_broad" in results_df.columns and "cbs_score" in results_df.columns:
        pt_breakdown = (
            results_df.groupby("problem_type_broad")["cbs_score"]
            .agg(["mean", "std", "count"])
            .round(4)
            .to_dict(orient="index")
        )
        stats["problem_type_breakdown"] = pt_breakdown

    return stats


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_benchmark(
    csv_path: str,
    output_dir: str = "results",
    metrics: list[str] = None,
    sample_n: Optional[int] = None,
    cge_model: Optional[str] = None,
    cge_api_type: Optional[str] = None,
    cge_api_key: Optional[str] = None,
    cge_sample: int = 50,
    verbose: bool = True,
) -> None:
    """Run the full BatteryHypoBench benchmark pipeline."""

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_metrics = ["rcf", "hpa", "msi", "sns", "ip", "pdq"]
    if metrics is None or "all" in metrics:
        metrics = all_metrics

    df = load_dataset(csv_path)

    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
        print(f"[sample] Subsampled to {len(df)} rows")

    results = df.copy()
    n = len(df)

    # ── RCF ──────────────────────────────────
    if "rcf" in metrics:
        print(f"[rcf] Computing Reasoning Chain Fidelity for {n} rows...")
        rcf_rows = [compute_rcf(row) for _, row in df.iterrows()]
        rcf_df = pd.DataFrame(rcf_rows)
        results = pd.concat([results.reset_index(drop=True), rcf_df.reset_index(drop=True)], axis=1)
        print(f"[rcf] Mean RCF aggregate: {rcf_df['rcf_aggregate'].mean():.4f}")

    # ── HPA ──────────────────────────────────
    if "hpa" in metrics:
        print(f"[hpa] Computing Hypothesis-Problem Alignment for {n} rows...")
        hpa_rows = [compute_hpa(row) for _, row in df.iterrows()]
        hpa_df = pd.DataFrame(hpa_rows)
        results = pd.concat([results.reset_index(drop=True), hpa_df.reset_index(drop=True)], axis=1)
        print(f"[hpa] Mean HPA aggregate: {hpa_df['hpa_aggregate'].mean():.4f}")

    # ── MSI ──────────────────────────────────
    if "msi" in metrics:
        print(f"[msi] Computing Mechanistic Specificity Index for {n} rows...")
        msi_rows = [compute_msi(row) for _, row in df.iterrows()]
        msi_df = pd.DataFrame(msi_rows)
        results = pd.concat([results.reset_index(drop=True), msi_df.reset_index(drop=True)], axis=1)
        print(f"[msi] Mean MSI aggregate: {msi_df['msi_aggregate'].mean():.4f}")

    # ── SNS (corpus-level) ───────────────────
    if "sns" in metrics:
        sns_df = compute_sns_corpus(df)
        results = pd.concat([results.reset_index(drop=True), sns_df.reset_index(drop=True)], axis=1)
        print(f"[sns] Mean SNS aggregate: {sns_df['sns_aggregate'].mean():.4f}")

    # ── IP ───────────────────────────────────
    if "ip" in metrics:
        print(f"[ip] Computing Intervention Plausibility for {n} rows...")
        ip_rows = [compute_ip(row) for _, row in df.iterrows()]
        ip_df = pd.DataFrame(ip_rows)
        results = pd.concat([results.reset_index(drop=True), ip_df.reset_index(drop=True)], axis=1)
        print(f"[ip] Mean IP aggregate: {ip_df['ip_aggregate'].mean():.4f}")

    # ── PDQ ──────────────────────────────────
    if "pdq" in metrics:
        print(f"[pdq] Computing Problem Decomposition Quality for {n} rows...")
        pdq_rows = [compute_pdq(row) for _, row in df.iterrows()]
        pdq_df = pd.DataFrame(pdq_rows)
        results = pd.concat([results.reset_index(drop=True), pdq_df.reset_index(drop=True)], axis=1)
        print(f"[pdq] Mean PDQ aggregate: {pdq_df['pdq_aggregate'].mean():.4f}")

    # ── CGE (LLM-judge, optional) ─────────────
    if cge_model and cge_api_type and cge_api_key:
        print(f"[cge] Running LLM judge ({cge_model}) on {min(cge_sample, n)} rows...")
        cge_sample_idx = df.sample(n=min(cge_sample, n), random_state=42).index
        cge_rows = []
        for i, (idx, row) in enumerate(df.iloc[cge_sample_idx.tolist()].iterrows()):
            print(f"  [cge] {i+1}/{min(cge_sample, n)}: {str(row.get('paper_id',''))[:30]}...")
            cge_result = compute_cge_row(row, cge_model, cge_api_type, cge_api_key)
            cge_result["paper_id"] = row.get("paper_id", str(idx))
            cge_rows.append(cge_result)
            time.sleep(0.3)
        cge_df = pd.DataFrame(cge_rows)
        cge_out_path = pathlib.Path(output_dir) / "cge_llm_judge.csv"
        cge_df.to_csv(cge_out_path, index=False)
        print(f"[cge] LLM judge results saved to {cge_out_path}")

    # ── CBS (composite score) ─────────────────
    print("[cbs] Computing Composite Battery Science Score...")
    metric_agg_cols = [c for c in results.columns if c.endswith("_aggregate") and not c.startswith("cbs")]
    cbs_scores = []
    for _, row in results.iterrows():
        row_dict = {c: row[c] for c in metric_agg_cols if c in row and pd.notna(row[c])}
        cbs_scores.append(compute_cbs(row_dict))
    results["cbs_score"] = cbs_scores
    print(f"[cbs] Mean CBS: {statistics.mean(cbs_scores):.4f} ± {statistics.stdev(cbs_scores):.4f}")

    # ── CONTRASTIVE ANALYSIS ─────────────────
    print("[stats] Running contrastive/summary statistics...")
    stats = compute_contrastive_stats(results)

    # ── SAVE OUTPUTS ─────────────────────────
    # Full scored dataset
    out_csv = pathlib.Path(output_dir) / "battery_benchmark_scored.csv"
    results.to_csv(out_csv, index=False)
    print(f"[save] Scored dataset → {out_csv}")

    # Summary statistics JSON
    out_stats = pathlib.Path(output_dir) / "benchmark_summary_stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[save] Summary stats → {out_stats}")

    # Leaderboard (top/bottom 20 by CBS)
    lb_cols = ["paper_id", "title", "battery_system", "cbs_score"] + metric_agg_cols[:6]
    lb_cols = [c for c in lb_cols if c in results.columns]
    leaderboard = results[lb_cols].sort_values("cbs_score", ascending=False)
    out_lb = pathlib.Path(output_dir) / "leaderboard.csv"
    leaderboard.to_csv(out_lb, index=False)
    print(f"[save] Leaderboard → {out_lb}")

    # Human-readable report
    _write_report(results, stats, output_dir, metrics)

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench v{VERSION} — DONE")
    print(f"  Evaluated {n} hypotheses across {len(metrics)} metric families")
    print(f"  Results in: {output_dir}/")
    print(f"{'='*60}\n")


def _write_report(results, stats, output_dir, metrics):
    """Write a human-readable Markdown benchmark report."""
    n = len(results)
    report_lines = [
        "# BatteryHypoBench: Evaluation Report",
        f"\n**Dataset size:** {n} problem-hypothesis pairs",
        f"**Metrics evaluated:** {', '.join(m.upper() for m in metrics)}",
        f"**Version:** {VERSION}",
        "\n---\n",
        "## Metric Summary",
        "",
        "| Metric | Full Name | Mean | Std | Median |",
        "|--------|-----------|------|-----|--------|",
    ]

    metric_names = {
        "rcf_aggregate": "Reasoning Chain Fidelity",
        "hpa_aggregate": "Hypothesis-Problem Alignment",
        "msi_aggregate": "Mechanistic Specificity Index",
        "sns_aggregate": "Scientific Novelty Score",
        "ip_aggregate": "Intervention Plausibility",
        "pdq_aggregate": "Problem Decomposition Quality",
        "cbs_score": "Composite Battery Science Score",
    }

    for col, name in metric_names.items():
        if col in results.columns:
            vals = results[col].dropna().values
            row_str = (
                f"| `{col}` | {name} | "
                f"{np.mean(vals):.4f} | {np.std(vals):.4f} | {np.median(vals):.4f} |"
            )
            report_lines.append(row_str)

    report_lines += [
        "\n---\n",
        "## Top 10 Papers by CBS Score",
        "",
        "| Rank | Paper ID | Battery System | CBS Score |",
        "|------|----------|----------------|-----------|",
    ]

    if "cbs_score" in results.columns:
        top10 = results.nlargest(10, "cbs_score")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            pid = str(row.get("paper_id", ""))[:30]
            bsys = str(row.get("battery_system", "N/A"))[:25]
            cbs = row.get("cbs_score", 0.0)
            report_lines.append(f"| {rank} | {pid} | {bsys} | {cbs:.4f} |")

    # Battery system breakdown
    if "battery_system_breakdown" in stats:
        report_lines += [
            "\n---\n",
            "## CBS Score by Battery System",
            "",
            "| Battery System | Mean CBS | Std | N |",
            "|----------------|----------|-----|---|",
        ]
        for sys, agg in sorted(stats["battery_system_breakdown"].items(),
                                key=lambda x: -x[1]["mean"])[:15]:
            report_lines.append(
                f"| {sys[:30]} | {agg['mean']:.4f} | {agg['std']:.4f} | {int(agg['count'])} |"
            )

    # Metric correlations
    if "metric_correlations" in stats and stats["metric_correlations"]:
        report_lines += [
            "\n---\n",
            "## Inter-Metric Spearman Correlations",
            "",
            "| Metric A | Metric B | ρ | p-value |",
            "|----------|----------|---|---------|",
        ]
        for pair in sorted(stats["metric_correlations"],
                           key=lambda x: -abs(x["spearman_rho"]))[:10]:
            report_lines.append(
                f"| {pair['metric_a']} | {pair['metric_b']} | "
                f"{pair['spearman_rho']:.4f} | {pair['p_value']:.4f} |"
            )

    report_lines += [
        "\n---\n",
        "## Metric Definitions",
        "",
        "- **RCF** (Reasoning Chain Fidelity): Evaluates step-wise logical progression, "
        "convergence to hypothesis, non-redundancy, and information density of reasoning chains.",
        "- **HPA** (Hypothesis-Problem Alignment): Measures semantic coherence between "
        "problem statement and proposed hypothesis via scope coverage, causal language, "
        "and intervention-mechanism coherence.",
        "- **MSI** (Mechanistic Specificity Index): Quantifies the level of mechanistic "
        "detail using domain vocabulary tiers, quantitative grounding, and characterization "
        "technique anchoring.",
        "- **SNS** (Scientific Novelty Score): Corpus-level TF-IDF distinctiveness of "
        "hypotheses within and across battery systems, with cross-domain novelty bonus.",
        "- **IP** (Intervention Plausibility): Assesses physical feasibility, material "
        "compatibility, scalability signals, evidence strength, and outcome specificity.",
        "- **PDQ** (Problem Decomposition Quality): Evaluates root cause specificity, "
        "failure mode boundedness, abstraction consistency, and component granularity.",
        "- **CBS** (Composite Battery Science Score): Weighted aggregate of all six metrics.",
        "",
        "---",
        "",
        "*Generated by BatteryHypoBench v{} — NeurIPS 2026 E&D Track Submission*".format(VERSION),
    ]

    out_report = pathlib.Path(output_dir) / "benchmark_report.md"
    with open(out_report, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[save] Markdown report → {out_report}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BatteryHypoBench — Multi-Dimensional Co-Scientist Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full benchmark on all rows
          python benchmark.py --csv /projects/bfir/ssourav/battery_problem_solution_500.csv

          # Quick test on 50 rows
          python benchmark.py --csv /projects/bfir/ssourav/battery_problem_solution_500.csv --sample 50

          # Specific metrics only
          python benchmark.py --csv data.csv --metrics rcf hpa msi

          # With LLM judge (requires API key)
          python benchmark.py --csv data.csv \\
              --cge-model gpt-4o --cge-api-type openai --cge-sample 100

          # Custom output directory
          python benchmark.py --csv data.csv --output /projects/bfir/ssourav/bench_results/
        """)
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to battery problem-solution CSV dataset"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["all"],
        choices=["all", "rcf", "hpa", "msi", "sns", "ip", "pdq"],
        help="Metrics to compute (default: all)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Subsample N rows for quick testing"
    )
    parser.add_argument(
        "--cge-model", default=None,
        help="LLM model for CGE judge (e.g., gpt-4o, claude-opus-4-5)"
    )
    parser.add_argument(
        "--cge-api-type", default=None, choices=["openai", "anthropic"],
        help="API type for LLM judge"
    )
    parser.add_argument(
        "--cge-api-key", default=None,
        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env vars)"
    )
    parser.add_argument(
        "--cge-sample", type=int, default=50,
        help="Number of rows for LLM judge (default: 50, can be expensive)"
    )
    parser.add_argument(
        "--version", action="version", version=f"BatteryHypoBench v{VERSION}"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  BatteryHypoBench v{VERSION}")
    print(f"  NeurIPS 2026 Evaluations & Datasets Track")
    print(f"{'='*60}\n")

    run_benchmark(
        csv_path=args.csv,
        output_dir=args.output,
        metrics=args.metrics,
        sample_n=args.sample,
        cge_model=args.cge_model,
        cge_api_type=args.cge_api_type,
        cge_api_key=args.cge_api_key,
        cge_sample=args.cge_sample,
    )


if __name__ == "__main__":
    main()
