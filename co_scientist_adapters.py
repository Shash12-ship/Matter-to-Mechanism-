#!/usr/bin/env python3
"""
co_scientist_adapters.py — Unified adapter registry for all real co-scientist systems.

Supported systems and their ACCESS METHOD:
  ┌────────────────────────────────┬──────────────────────────────────────┬───────────────┐
  │ System                         │ Access                               │ Needs API Key │
  ├────────────────────────────────┼──────────────────────────────────────┼───────────────┤
  │ Sakana AI Scientist v2         │ Local run (open source, OpenAI calls)│ OPENAI_API_KEY│
  │ FutureHouse Crow               │ futurehouse-platform REST API        │ FH_API_KEY    │
  │ FutureHouse Falcon             │ futurehouse-platform REST api        │ FH_API_KEY    │
  │ ChemDFM (OpenDFM)              │ HuggingFace local inference          │ HF_TOKEN      │
  │ Google Gemini 2.5 Pro          │ Gemini generativeai API              │ GEMINI_API_KEY│
  │ Google Deep Research Agent     │ Gemini Interactions API (preview)    │ GEMINI_API_KEY│
  │ OpenAI o3                      │ OpenAI chat completions API          │ OPENAI_API_KEY│
  │ OpenAI o4-mini                 │ OpenAI chat completions API          │ OPENAI_API_KEY│
  │ OpenAI GPT-4o                  │ OpenAI chat completions API          │ OPENAI_API_KEY│
  │ Claude opus-4-5                │ Anthropic messages API               │ ANTHROPIC_KEY │
  │ Claude sonnet-4-5              │ Anthropic messages API               │ ANTHROPIC_KEY │
  └────────────────────────────────┴──────────────────────────────────────┴───────────────┘

Usage (imported by run_coscientist_benchmark.py):
  from co_scientist_adapters import ADAPTER_REGISTRY, generate_with_adapter
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


# ─────────────────────────────────────────────────────────────
# GENERATION PROMPT — identical across all systems for fair comparison
# ─────────────────────────────────────────────────────────────

GENERATION_PROMPT_TEMPLATE = """You are an expert battery materials scientist. 
Given the research problem below, generate a rigorous scientific hypothesis.

PROBLEM:
{problem}

BATTERY SYSTEM: {battery_system}
COMPONENT: {component}  
FAILURE MODE: {failure_mode}

Respond in this EXACT format (use the tags literally):

[HYPOTHESIS]
One precise, falsifiable hypothesis in 2-3 sentences.
[/HYPOTHESIS]

[INTERVENTION]
Specific proposed material/process/modification.
[/INTERVENTION]

[MECHANISM]
Physical or chemical mechanism explaining why it works.
[/MECHANISM]

[REASONING]
[Begin Step 1] First logical step from problem context. [End Step 1]
[Begin Step 2] Second step building on step 1. [End Step 2]
[Begin Step 3] Third step. [End Step 3]
[Begin Step 4] Fourth step. [End Step 4]
[Begin Step 5] Final step arriving at the hypothesis. [End Step 5]
[/REASONING]

[TARGET_PROPERTY]
The specific property this hypothesis aims to improve.
[/TARGET_PROPERTY]

[CLAIMED_OUTCOME]
Expected improvement, ideally quantified.
[/CLAIMED_OUTCOME]"""


def build_prompt(row: dict) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(
        problem=str(row.get("problem_statement", ""))[:1000],
        battery_system=str(row.get("battery_system", "Unknown"))[:100],
        component=str(row.get("component", "Unknown"))[:100],
        failure_mode=str(row.get("failure_mode_or_limitation", "Unknown"))[:200],
    )


def parse_response(raw: str) -> dict:
    """Parse structured response into dict fields."""
    def extract(tag, text):
        m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    hyp = extract("HYPOTHESIS", raw)
    steps = re.findall(r"\[Begin Step \d+\](.*?)\[End Step \d+\]", raw, re.DOTALL)
    return {
        "hypothesis": hyp,
        "intervention_or_solution": extract("INTERVENTION", raw),
        "mechanism_or_rationale": extract("MECHANISM", raw),
        "reasoning_process": extract("REASONING", raw),
        "target_property": extract("TARGET_PROPERTY", raw),
        "claimed_outcome": extract("CLAIMED_OUTCOME", raw),
        "num_reasoning_steps": len(steps),
        "raw_output": raw[:2000],
    }


# ─────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────

def _http_post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            if e.code == 429:
                wait = 2 ** (attempt + 2)
                print(f"    [rate_limit] waiting {wait}s...")
                time.sleep(wait)
            elif e.code in (500, 502, 503):
                time.sleep(2 ** attempt)
            else:
                return {"error": f"HTTP {e.code}: {err_body[:300]}"}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e)}
    return {"error": "Max retries exceeded"}


def _http_get(url: str, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# ADAPTER: OpenAI (GPT-4o, o3, o4-mini)
# ─────────────────────────────────────────────────────────────

def adapter_openai(row: dict, model: str, api_key: str) -> dict:
    """Standard OpenAI chat completions."""
    prompt = build_prompt(row)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    resp = _http_post("https://api.openai.com/v1/chat/completions", headers, body)
    if "error" in resp:
        return {"error": resp["error"], "hypothesis": ""}
    raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return parse_response(raw)


# ─────────────────────────────────────────────────────────────
# ADAPTER: Anthropic Claude
# ─────────────────────────────────────────────────────────────

def adapter_anthropic(row: dict, model: str, api_key: str) -> dict:
    """Anthropic messages API."""
    prompt = build_prompt(row)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
    }
    resp = _http_post("https://api.anthropic.com/v1/messages", headers, body)
    if "error" in resp:
        return {"error": resp["error"], "hypothesis": ""}
    raw = resp.get("content", [{}])[0].get("text", "")
    return parse_response(raw)


# ─────────────────────────────────────────────────────────────
# ADAPTER: Google Gemini (2.5 Pro / Flash)
# ─────────────────────────────────────────────────────────────

def adapter_gemini(row: dict, model: str, api_key: str) -> dict:
    """Google Gemini generateContent API."""
    prompt = build_prompt(row)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200},
    }
    resp = _http_post(url, headers, body, timeout=90)
    if "error" in resp:
        return {"error": str(resp.get("error", resp)), "hypothesis": ""}
    try:
        raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return {"error": f"Unexpected Gemini response: {str(resp)[:200]}", "hypothesis": ""}
    return parse_response(raw)


# ─────────────────────────────────────────────────────────────
# ADAPTER: Google Gemini Deep Research Agent (Interactions API)
# NOTE: This is async / polling — may take 2-10 min per query
# ─────────────────────────────────────────────────────────────

def adapter_gemini_deep_research(row: dict, model: str, api_key: str) -> dict:
    """
    Google Gemini Deep Research via Interactions API (preview).
    Uses background=true + polling.
    """
    prompt = (
        f"You are a battery materials co-scientist. Generate a rigorous, falsifiable "
        f"scientific hypothesis for the following research problem.\n\n"
        f"PROBLEM: {str(row.get('problem_statement',''))[:800]}\n"
        f"BATTERY SYSTEM: {row.get('battery_system','')}\n"
        f"FAILURE MODE: {row.get('failure_mode_or_limitation','')}\n\n"
        f"Structure your response with clearly labeled sections: "
        f"HYPOTHESIS, INTERVENTION, MECHANISM, STEP-BY-STEP REASONING (5 steps), "
        f"TARGET PROPERTY, CLAIMED OUTCOME."
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    # Start interaction
    body = {"input": prompt, "agent": model, "background": True}
    resp = _http_post(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        headers, body, timeout=30
    )
    if "error" in resp:
        return {"error": str(resp["error"]), "hypothesis": ""}

    interaction_id = resp.get("name", "")
    if not interaction_id:
        return {"error": "No interaction_id returned", "hypothesis": ""}

    # Poll for completion (max 10 min)
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{interaction_id}"
    for _ in range(60):
        time.sleep(10)
        result = _http_get(poll_url, headers, timeout=30)
        state = result.get("state", "")
        if state == "COMPLETED":
            try:
                raw = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                raw = str(result.get("response", ""))[:1000]
            parsed = parse_response(raw)
            # Deep research gives narrative not tagged — try to extract best-effort
            if not parsed["hypothesis"]:
                # Fallback: first non-empty paragraph as hypothesis
                paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 50]
                parsed["hypothesis"] = paragraphs[0] if paragraphs else raw[:300]
            return parsed
        elif state in ("FAILED", "CANCELLED"):
            return {"error": f"Interaction {state}", "hypothesis": ""}
    return {"error": "Deep research timed out after 10 min", "hypothesis": ""}


# ─────────────────────────────────────────────────────────────
# ADAPTER: FutureHouse Crow/Falcon
# API: https://api.futurehouse.org  (requires FH_API_KEY)
# ─────────────────────────────────────────────────────────────

def adapter_futurehouse(row: dict, model: str, api_key: str) -> dict:
    """
    FutureHouse Platform API (Crow = concise Q&A, Falcon = deep literature review).
    model should be 'crow' or 'falcon'.
    Docs: https://platform.futurehouse.org/docs
    """
    agent = model.lower()  # 'crow' or 'falcon'
    question = (
        f"For the following battery research problem, generate a scientifically rigorous "
        f"hypothesis with: (1) a falsifiable hypothesis statement, (2) a specific intervention, "
        f"(3) the physical mechanism, (4) step-by-step reasoning (5 steps), "
        f"(5) target property, (6) claimed quantitative outcome.\n\n"
        f"Problem: {str(row.get('problem_statement',''))[:800]}\n"
        f"Battery system: {row.get('battery_system','')}\n"
        f"Failure mode: {row.get('failure_mode_or_limitation','')}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {"agent": agent, "query": question}
    resp = _http_post("https://api.futurehouse.org/v1/query", headers, body, timeout=300)

    if "error" in resp and not resp.get("answer"):
        return {"error": str(resp.get("error", resp)), "hypothesis": ""}

    raw = resp.get("answer", resp.get("response", str(resp)))
    parsed = parse_response(raw)
    # FutureHouse gives narrative prose — extract best-effort
    if not parsed["hypothesis"]:
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 40]
        parsed["hypothesis"] = lines[0] if lines else raw[:300]
    return parsed


# ─────────────────────────────────────────────────────────────
# ADAPTER: ChemDFM (local HuggingFace inference)
# Model: OpenDFM/ChemDFM-13B-v1.0 or ChemDFM-v1.5-8B
# Requires: transformers, torch, ~16GB VRAM (A100/GH200)
# ─────────────────────────────────────────────────────────────

_chemdfm_model = None
_chemdfm_tokenizer = None


def adapter_chemdfm(row: dict, model: str, api_key: str = None) -> dict:
    """
    ChemDFM local inference using AutoModelForCausalLM + chat template.
    Uses input_ids approach to cleanly strip prompt from output.

    model = 'OpenDFM/ChemDFM-v1.5-8B' (recommended for GH200)
          = 'OpenDFM/ChemDFM-13B-v1.0' (original, ~26GB VRAM)
    """
    global _chemdfm_model, _chemdfm_tokenizer
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        return {
            "error": "transformers/torch not installed. "
                     "Run: pip install transformers accelerate --break-system-packages",
            "hypothesis": "",
        }

    if _chemdfm_model is None:
        model_id = model if "/" in model else "OpenDFM/ChemDFM-v1.5-8B"
        hf_token = api_key or os.environ.get("HF_TOKEN", None)
        print(f"    [chemdfm] Loading {model_id}...")

        _chemdfm_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )
        _chemdfm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        _chemdfm_model.eval()
        print(f"    [chemdfm] Loaded on {next(_chemdfm_model.parameters()).device}")

    # Build prompt — use chat template if tokenizer supports it, else raw text
    user_content = (
        f"You are an expert battery materials scientist.\n"
        f"Given this research problem, generate a rigorous scientific hypothesis.\n\n"
        f"PROBLEM: {str(row.get('problem_statement', ''))[:800]}\n"
        f"BATTERY SYSTEM: {str(row.get('battery_system', ''))[:100]}\n"
        f"COMPONENT: {str(row.get('component', ''))[:100]}\n"
        f"FAILURE MODE: {str(row.get('failure_mode_or_limitation', ''))[:200]}\n\n"
        f"Structure your response with these clearly labeled sections:\n"
        f"[HYPOTHESIS] ... [/HYPOTHESIS]\n"
        f"[INTERVENTION] ... [/INTERVENTION]\n"
        f"[MECHANISM] ... [/MECHANISM]\n"
        f"[REASONING]\n"
        f"[Begin Step 1] ... [End Step 1]\n"
        f"[Begin Step 2] ... [End Step 2]\n"
        f"[Begin Step 3] ... [End Step 3]\n"
        f"[Begin Step 4] ... [End Step 4]\n"
        f"[Begin Step 5] ... [End Step 5]\n"
        f"[/REASONING]\n"
        f"[TARGET_PROPERTY] ... [/TARGET_PROPERTY]\n"
        f"[CLAIMED_OUTCOME] ... [/CLAIMED_OUTCOME]"
    )

    import torch

    # Try chat template first (ChemDFM-v1.5 supports it)
    try:
        messages = [{"role": "user", "content": user_content}]
        input_ids = _chemdfm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(_chemdfm_model.device)
    except Exception:
        # Fallback: plain tokenization
        input_ids = _chemdfm_tokenizer(
            user_content, return_tensors="pt"
        ).input_ids.to(_chemdfm_model.device)

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = _chemdfm_model.generate(
            input_ids,
            max_new_tokens=700,
            temperature=0.3,
            do_sample=True,
            pad_token_id=_chemdfm_tokenizer.eos_token_id,
            eos_token_id=_chemdfm_tokenizer.eos_token_id,
        )

    # Decode ONLY the newly generated tokens (strip prompt)
    new_tokens = output_ids[0][prompt_len:]
    raw = _chemdfm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not raw:
        return {"error": "ChemDFM generated empty output", "hypothesis": "", "raw_output": ""}

    parsed = parse_response(raw)
    parsed["raw_output"] = raw[:1000]

    # Fallback extraction if tags not found — parse free-form scientific text
    if not parsed["hypothesis"]:
        paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
        parsed["hypothesis"] = paragraphs[0] if paragraphs else raw[:400]

    if not parsed["mechanism_or_rationale"]:
        # Look for sentences with mechanistic language
        mech_sentences = [
            s.strip() for s in re.split(r"[.!?]", raw)
            if any(kw in s.lower() for kw in [
                "because", "due to", "mechanism", "enables", "results in",
                "diffusion", "reaction", "bond", "structure", "phase"
            ])
        ]
        parsed["mechanism_or_rationale"] = ". ".join(mech_sentences[:3])

    if not parsed["reasoning_process"]:
        # Reconstruct reasoning from numbered lines or paragraphs
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 30]
        steps = [f"[Begin Step {i+1}] {l} [End Step {i+1}]"
                 for i, l in enumerate(lines[:5])]
        parsed["reasoning_process"] = "\n".join(steps)
        parsed["num_reasoning_steps"] = len(steps)

    return parsed


# ─────────────────────────────────────────────────────────────
# ADAPTER: Sakana AI Scientist v2 (hypothesis idea generation only)
# Uses AI-Scientist-v2's idea generation module directly,
# stripped of the experimental execution loop (we only need hypotheses).
# Calls OpenAI API internally (o1/gpt-4o).
# ─────────────────────────────────────────────────────────────

SAKANA_IDEA_PROMPT = """You are an AI scientist doing autonomous research on battery materials.
Given this problem, generate 1 novel research idea/hypothesis using the AI Scientist format.

PROBLEM: {problem}
BATTERY SYSTEM: {battery_system}
FAILURE MODE: {failure_mode}

Generate a JSON with these fields:
{{
  "Name": "short_idea_name",
  "Title": "Full hypothesis title",
  "Experiment": "What experiment to run",
  "Interestingness": 8,
  "Feasibility": 7,
  "Novelty": 8,
  "novel": true,
  "hypothesis": "The full hypothesis statement (2-3 sentences)",
  "mechanism": "Physical/chemical mechanism",
  "intervention": "Specific material or process",
  "reasoning_steps": ["step1", "step2", "step3", "step4", "step5"],
  "target_property": "Property to improve",
  "claimed_outcome": "Expected result"
}}
Return only the JSON, no other text."""


def adapter_sakana_ai_scientist(row: dict, model: str, api_key: str) -> dict:
    """
    Sakana AI Scientist v2 — uses AI Scientist's idea generation prompt format.
    Calls through OpenAI API (o1-preview or gpt-4o, as AI Scientist does).
    model = 'sakana-ai-scientist-v2' → internally routes to o1-preview
    """
    # Map sakana model to underlying LLM
    underlying_model = "o1-preview-2024-09-12"  # as used in AI Scientist v2

    prompt = SAKANA_IDEA_PROMPT.format(
        problem=str(row.get("problem_statement", ""))[:800],
        battery_system=str(row.get("battery_system", ""))[:100],
        failure_mode=str(row.get("failure_mode_or_limitation", ""))[:200],
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": underlying_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
    }
    resp = _http_post("https://api.openai.com/v1/chat/completions", headers, body, timeout=120)
    if "error" in resp:
        return {"error": resp["error"], "hypothesis": ""}

    raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Parse JSON response
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(clean)
        steps = data.get("reasoning_steps", [])
        # Reconstruct reasoning_process in expected format
        rp_parts = [f"[Begin Step {i+1}] {s} [End Step {i+1}]" for i, s in enumerate(steps)]
        return {
            "hypothesis": data.get("hypothesis", ""),
            "intervention_or_solution": data.get("intervention", ""),
            "mechanism_or_rationale": data.get("mechanism", ""),
            "reasoning_process": "\n".join(rp_parts),
            "target_property": data.get("target_property", ""),
            "claimed_outcome": data.get("claimed_outcome", ""),
            "num_reasoning_steps": len(steps),
            "sakana_title": data.get("Title", ""),
            "sakana_interestingness": data.get("Interestingness", 0),
            "sakana_feasibility": data.get("Feasibility", 0),
            "sakana_novelty": data.get("Novelty", 0),
            "raw_output": raw[:500],
        }
    except json.JSONDecodeError:
        # Fallback: treat as plain text
        return parse_response(raw)


# ─────────────────────────────────────────────────────────────
# ADAPTER REGISTRY
# Maps system_name → (adapter_fn, default_model_string, api_key_env_var)
# ─────────────────────────────────────────────────────────────

ADAPTER_REGISTRY = {
    # ── Closed-source / API ──────────────────────────────────
    "gpt-4o": (
        adapter_openai, "gpt-4o", "OPENAI_API_KEY",
        "OpenAI GPT-4o (baseline frontier model)"
    ),
    "o3": (
        adapter_openai, "o3", "OPENAI_API_KEY",
        "OpenAI o3 (reasoning model)"
    ),
    "o4-mini": (
        adapter_openai, "o4-mini", "OPENAI_API_KEY",
        "OpenAI o4-mini (fast reasoning)"
    ),
    "claude-opus": (
        adapter_anthropic, "claude-opus-4-5", "ANTHROPIC_API_KEY",
        "Anthropic Claude Opus 4.5"
    ),
    "claude-sonnet": (
        adapter_anthropic, "claude-sonnet-4-5", "ANTHROPIC_API_KEY",
        "Anthropic Claude Sonnet 4.5"
    ),
    "gemini-2.5-pro": (
        adapter_gemini, "gemini-2.5-pro", "GEMINI_API_KEY",
        "Google Gemini 2.5 Pro (Gemini API)"
    ),
    "gemini-2.5-flash": (
        adapter_gemini, "gemini-2.5-flash", "GEMINI_API_KEY",
        "Google Gemini 2.5 Flash (fast, cheap)"
    ),
    # ── Google Co-Scientist (Deep Research Agent) ────────────
    "google-co-scientist": (
        adapter_gemini_deep_research,
        "deep-research-pro-preview-12-2025",
        "GEMINI_API_KEY",
        "Google AI Co-Scientist (Gemini Deep Research Agent, ~2-10 min/query)"
    ),
    # ── FutureHouse ──────────────────────────────────────────
    "futurehouse-crow": (
        adapter_futurehouse, "crow", "FH_API_KEY",
        "FutureHouse Crow (PaperQA2, concise literature Q&A)"
    ),
    "futurehouse-falcon": (
        adapter_futurehouse, "falcon", "FH_API_KEY",
        "FutureHouse Falcon (deep literature synthesis)"
    ),
    # ── Sakana AI Scientist v2 ───────────────────────────────
    "sakana-ai-scientist": (
        adapter_sakana_ai_scientist, "sakana-ai-scientist-v2", "OPENAI_API_KEY",
        "Sakana AI Scientist v2 (idea generation via o1-preview, open-source format)"
    ),
    # ── ChemDFM (open-source, local) ─────────────────────────
    "chemdfm-8b": (
        adapter_chemdfm, "OpenDFM/ChemDFM-v1.5-8B", "HF_TOKEN",
        "ChemDFM v1.5-8B (open-source chemistry LLM, local inference on GH200)"
    ),
    "chemdfm-13b": (
        adapter_chemdfm, "OpenDFM/ChemDFM-13B-v1.0", "HF_TOKEN",
        "ChemDFM 13B (open-source chemistry LLM, local inference on GH200)"
    ),
}


def list_systems():
    """Print all available systems."""
    print("\n" + "="*75)
    print("  Available Co-Scientist Systems")
    print("="*75)
    for name, (_, model, env_var, desc) in ADAPTER_REGISTRY.items():
        key_set = "✓" if os.environ.get(env_var) else "✗"
        print(f"  [{key_set}] {name:<28} {desc[:55]}")
        print(f"       model={model:<30} key_env={env_var}")
    print("="*75 + "\n")


def generate_with_adapter(
    system_name: str,
    row: dict,
    api_keys: dict,
) -> dict:
    """
    Generate hypothesis from a co-scientist system.
    
    Args:
        system_name: key from ADAPTER_REGISTRY
        row: dict with problem fields
        api_keys: dict of env_var_name → key_value
    
    Returns:
        dict with hypothesis, reasoning_process, etc.
    """
    if system_name not in ADAPTER_REGISTRY:
        return {"error": f"Unknown system: {system_name}. "
                         f"Valid: {list(ADAPTER_REGISTRY.keys())}"}

    adapter_fn, default_model, env_var, _ = ADAPTER_REGISTRY[system_name]
    api_key = api_keys.get(env_var) or os.environ.get(env_var, "")

    if not api_key and system_name not in ("chemdfm-8b", "chemdfm-13b"):
        return {
            "error": f"Missing API key. Set env var {env_var} or pass via --{env_var.lower().replace('_','-')}",
            "hypothesis": "",
        }

    result = adapter_fn(row, default_model, api_key)
    result["_system"] = system_name
    result["_model"] = default_model
    return result


if __name__ == "__main__":
    list_systems()
