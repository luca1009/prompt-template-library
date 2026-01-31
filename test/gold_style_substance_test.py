#!/usr/bin/env python3
"""
Gold level evaluation script for the prompt‑pattern library.

This script exercises style‑sensitive rewrite templates and attempts to
measure "style" and "substance" independently.  For each test case it
invokes the local FastAPI service to render and execute a template,
parses the resulting JSON, and then computes a handful of metrics:

* Style metrics
  - An OpenAI‑based rating (1–5) on how well the rewritten text matches
    the requested style.  This call is optional and only performed when
    the ``OPENAI_API_KEY`` environment variable is defined.  You can
    override the OpenAI model via ``--judge-model``.
  - Basic linguistic features: average sentence length (in tokens) and
    lexical diversity (unique tokens / total tokens).  These are
    computed using a simple tokenizer and do not require external
    libraries.

* Substance metrics
  - Fact preservation: The rewrite template returns an object with
    ``preserved_facts`` and ``changed_or_missing_facts`` fields.  The
    script counts how many facts were preserved relative to the number
    of facts supplied in the test case.  If the output cannot be
    parsed, these values default to zero.
  - An OpenAI‑based rating (1–5) on how faithfully the rewritten text
    preserves the original content and facts.  This call is optional
    and controlled by the same environment variable as above.

All results are written to ``runs.jsonl`` and ``summary.csv`` in the
specified output directory.  A top‑level summary (mean scores) is
printed at the end of the run to aid quick inspection.

Example usage:

    python gold_style_substance_test.py \
        --api http://127.0.0.1:8000 \
        --config test/test_data/hero_rewrite.json \
        --model lmstudio-model \
        --outdir gold_results

You may specify multiple ``--config`` files; the script will merge all
cases and evaluate them in one run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    # OpenAI SDK is optional.  Import lazily to avoid mandatory dependency.
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


###############################################################################
# Helper functions
###############################################################################

def _simple_tokenize(text: str) -> List[str]:
    """Very basic word tokenizer used for lexical metrics.

    Splits on whitespace and strips punctuation.  This is deliberately
    simple to avoid heavy dependencies; for more precise metrics one
    could install nltk or similar libraries.
    """
    # Replace common punctuation with spaces
    text = re.sub(r"[\r\n\t]", " ", text)
    text = re.sub(r"[,.!?;:\-()\[\]{}\"']", " ", text)
    tokens = [tok for tok in text.split() if tok]
    return tokens


def compute_linguistic_metrics(text: str) -> Tuple[float, float]:
    """Compute basic linguistic metrics.

    Returns a tuple of (average_sentence_length, lexical_diversity).

    The average sentence length is computed as the number of tokens divided
    by the number of sentences, where sentences are heuristically split
    on ``. ? !``.  Lexical diversity is the ratio of unique tokens to
    total tokens.  If the input text has no tokens the metrics return 0.0.
    """
    if not text or not text.strip():
        return 0.0, 0.0
    # Simple sentence split
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    tokens = _simple_tokenize(text)
    if not sentences or not tokens:
        return 0.0, 0.0
    avg_len = len(tokens) / len(sentences) if sentences else 0.0
    unique_tokens = set(tokens)
    lex_div = len(unique_tokens) / len(tokens) if tokens else 0.0
    return avg_len, lex_div


def call_openai_score(prompt: str, model: str) -> Optional[float]:
    """Call the OpenAI API with a scoring prompt.

    Expects ``OPENAI_API_KEY`` in the environment.  Returns a float
    representing the score on a 1–5 scale if successful, or ``None``
    otherwise.  This function is resilient to errors and will not raise.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None or not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        # Use chat completion API.  We pass a single message as system
        # instruction because the judge is essentially a self‑contained
        # scoring task.
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.0,
            max_tokens=5
        )
        # Extract the numeric score from the first assistant message
        choice = completion.choices[0].message.content.strip()
        m = re.search(r"([1-5])", choice)
        if m:
            return float(m.group(1))
    except Exception:
        # Suppress all exceptions; treat as unavailable
        return None
    return None


def evaluate_style_openai(style_desc: str, output_text: str, model: str) -> Optional[float]:
    """Ask OpenAI to rate adherence to a style description.

    Returns a number between 1 and 5 or ``None`` if the API call is
    disabled or fails.  The prompt instructs the assistant to return
    only a single digit representing the score.  A higher number
    indicates better adherence.
    """
    prompt = (
        "You are an impartial evaluator of writing style. "
        "On a scale from 1 (poor match) to 5 (excellent match), rate how well "
        "the following output matches the requested style description. "
        "Return only the numeric score.\n"
        f"Requested style: {style_desc}\n"
        f"Output: {output_text}\n"
        "Score:"
    )
    return call_openai_score(prompt, model)


def evaluate_substance_openai(source_text: str, output_text: str, model: str) -> Optional[float]:
    """Ask OpenAI to rate fidelity between source and rewritten text.

    The evaluator is instructed to judge whether all facts and meaning from
    the source are preserved without introducing new information.  The
    returned score is on a 1–5 scale; ``None`` signifies that the API
    is not configured.
    """
    prompt = (
        "You are an impartial evaluator of factual fidelity. "
        "On a scale from 1 (poor fidelity) to 5 (perfect fidelity), rate how well "
        "the rewritten text preserves the meaning and factual content of the "
        "source text.  Ignore stylistic differences.  Return only the numeric score.\n"
        f"Source text: {source_text}\n"
        f"Rewritten text: {output_text}\n"
        "Score:"
    )
    return call_openai_score(prompt, model)


###############################################################################
# Data classes
###############################################################################

@dataclass
class CaseResult:
    experiment: str
    case_id: str
    template_id: str
    model_name: str
    status: str
    latency_ms: int
    raw_response: str
    rewritten_text: str
    preserved_facts: List[str]
    changed_facts: List[str]
    style_score: Optional[float]
    substance_score: Optional[float]
    avg_sentence_length: float
    lexical_diversity: float
    facts_input_count: int


###############################################################################
# Main evaluation logic
###############################################################################

def load_cases(config_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load and merge test cases from one or more JSON config files.

    Each config file should contain a dictionary with a ``cases`` key.  The
    surrounding metadata (``test_name``, ``pattern_id``, ``model_name``) is
    propagated into each case record.  Returns a list of flattened
    dictionaries containing ``experiment``, ``template_id``, ``model_name``
    and ``params`` fields.
    """
    all_cases: List[Dict[str, Any]] = []
    for cfg_path in config_paths:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        test_name = data.get("test_name", cfg_path.stem)
        template_id = data["pattern_id"]
        model_name = data.get("model_name", "lmstudio-model")
        cases = data.get("cases", [])
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id", f"{cfg_path.stem}_{idx}")) if isinstance(case, dict) else f"{cfg_path.stem}_{idx}"
            params = {}
            if isinstance(case, dict) and isinstance(case.get("params"), dict):
                params = dict(case["params"])
            elif isinstance(case, str):
                # Simple string case: treat as input/text
                params = {"input": case, "text": case}
            all_cases.append({
                "experiment": test_name,
                "case_id": case_id,
                "template_id": template_id,
                "model_name": model_name,
                "params": params
            })
    return all_cases


def parse_rewrite_response(resp_text: str) -> Tuple[str, List[str], List[str]]:
    """Parse a JSON response from the rewrite template.

    The ``rewrite_preserve_facts_v1`` template returns a JSON object with
    keys ``rewritten_text``, ``preserved_facts`` and ``changed_or_missing_facts``.
    If the text cannot be parsed as JSON, the function returns the
    raw text and empty lists for the fact arrays.
    """
    try:
        obj = json.loads(resp_text)
        rewritten = obj.get("rewritten_text", "")
        pf = obj.get("preserved_facts", [])
        cf = obj.get("changed_or_missing_facts", [])
        if not isinstance(pf, list):
            pf = []
        if not isinstance(cf, list):
            cf = []
        return rewritten, pf, cf
    except Exception:
        return resp_text, [], []


def count_facts_input(facts_field: str) -> int:
    """Count the number of facts specified in the ``facts`` placeholder.

    Facts are expected to be delimited by semicolons or newlines.  This
    heuristic splits the string on ``;`` or ``\n`` and counts non‑empty
    elements.
    """
    if not facts_field or not isinstance(facts_field, str):
        return 0
    # Split on semicolon or newline
    parts = re.split(r"[;\n]", facts_field)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def execute_case(api: str, template_id: str, params: Dict[str, str], model_name: str,
                 connect_timeout: int, exec_timeout: int) -> Tuple[str, str, int]:
    """Render and execute a template via the API.

    Returns a tuple ``(status, response_text, latency_ms)``.  ``status``
    will be ``OK`` for HTTP 200, ``TIMEOUT`` for a client timeout, or
    ``ERR_<code>`` for any other HTTP status.  ``response_text`` is the
    raw text returned by the server, or an empty string on error.
    ``latency_ms`` measures the wall‑clock time for the HTTP call.
    """
    session = requests.Session()
    url = f"{api.rstrip('/')}/execute"
    payload = {
        "template_id": template_id,
        "params": params,
        "model_name": model_name
    }
    t0 = time.time()
    try:
        r = session.post(url, json=payload, timeout=(connect_timeout, exec_timeout))
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            try:
                j = r.json()
                return "OK", j.get("response_text", ""), ms
            except Exception:
                return f"ERR_JSON", r.text, ms
        else:
            return f"ERR_{r.status_code}", r.text[:200], ms
    except requests.exceptions.Timeout:
        ms = int((time.time() - t0) * 1000)
        return "TIMEOUT", "", ms
    except requests.exceptions.RequestException as e:
        ms = int((time.time() - t0) * 1000)
        return "ERR_REQ", str(e), ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold level style vs. substance evaluation")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of the FastAPI service")
    ap.add_argument("--config", action="append", required=True, help="Path to a JSON config file with test cases")
    ap.add_argument("--outdir", default="gold_results", help="Directory to store results")
    ap.add_argument("--connect-timeout", type=int, default=3, help="TCP connect timeout (seconds)")
    ap.add_argument("--timeout-exec", type=int, default=120, help="/execute read timeout (seconds)")
    ap.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model name for scoring")

    args = ap.parse_args()

    # Prepare output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load cases
    cfg_paths = [Path(p) for p in args.config]
    all_cases = load_cases(cfg_paths)
    if not all_cases:
        print("[FAIL] No cases loaded. Check config files.", file=sys.stderr)
        sys.exit(2)

    # Execute cases
    results: List[CaseResult] = []
    for idx, case in enumerate(all_cases, start=1):
        print(f"[{idx}/{len(all_cases)}] {case['experiment']} case={case['case_id']} template={case['template_id']}")
        status, resp_text, latency_ms = execute_case(
            args.api,
            case["template_id"],
            case["params"],
            case["model_name"],
            args.connect_timeout,
            args.timeout_exec
        )
        rewritten, preserved, changed = parse_rewrite_response(resp_text)
        facts_input_count = count_facts_input(case["params"].get("facts", ""))
        avg_len, lex_div = compute_linguistic_metrics(rewritten)
        # Evaluate style and substance via OpenAI if possible
        style_score = None
        substance_score = None
        if status == "OK":
            style_desc = case["params"].get("style", "")
            src_text = case["params"].get("input", case["params"].get("text", ""))
            if style_desc:
                style_score = evaluate_style_openai(style_desc, rewritten, args.judge_model)
            if src_text:
                substance_score = evaluate_substance_openai(src_text, rewritten, args.judge_model)
        result = CaseResult(
            experiment=case["experiment"],
            case_id=case["case_id"],
            template_id=case["template_id"],
            model_name=case["model_name"],
            status=status,
            latency_ms=latency_ms,
            raw_response=resp_text,
            rewritten_text=rewritten,
            preserved_facts=preserved,
            changed_facts=changed,
            style_score=style_score,
            substance_score=substance_score,
            avg_sentence_length=avg_len,
            lexical_diversity=lex_div,
            facts_input_count=facts_input_count
        )
        results.append(result)

    # Write JSONL
    jsonl_path = run_dir / "runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Write CSV summary
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "case_id",
            "template_id",
            "model_name",
            "status",
            "latency_ms",
            "style_score",
            "substance_score",
            "avg_sentence_length",
            "lexical_diversity",
            "facts_input_count",
            "facts_preserved",
            "facts_changed",
        ])
        for r in results:
            writer.writerow([
                r.experiment,
                r.case_id,
                r.template_id,
                r.model_name,
                r.status,
                r.latency_ms,
                r.style_score if r.style_score is not None else "",
                r.substance_score if r.substance_score is not None else "",
                f"{r.avg_sentence_length:.2f}",
                f"{r.lexical_diversity:.2f}",
                r.facts_input_count,
                len(r.preserved_facts),
                len(r.changed_facts),
            ])

    # Print aggregated statistics
    ok_results = [r for r in results if r.status == "OK"]
    if ok_results:
        # Style and substance scores may be None if OpenAI not configured
        style_scores = [r.style_score for r in ok_results if r.style_score is not None]
        substance_scores = [r.substance_score for r in ok_results if r.substance_score is not None]
        avg_len_values = [r.avg_sentence_length for r in ok_results]
        lex_div_values = [r.lexical_diversity for r in ok_results]
        def mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0
        print("\n[SUMMARY] OK cases =", len(ok_results))
        print("- Avg style score (OpenAI):", f"{mean(style_scores):.2f}" if style_scores else "n/a")
        print("- Avg substance score (OpenAI):", f"{mean(substance_scores):.2f}" if substance_scores else "n/a")
        print("- Avg sentence length:", f"{mean(avg_len_values):.2f}")
        print("- Avg lexical diversity:", f"{mean(lex_div_values):.2f}")
    else:
        print("\n[SUMMARY] No successful cases to summarise.")

    print(f"\nResults written to:\n- {jsonl_path}\n- {csv_path}")


if __name__ == "__main__":
    main()