#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------
# HTTP helpers
# -----------------------------
def get_json(session: requests.Session, url: str, ct: int, rt: int) -> Tuple[int, Any, str]:
    try:
        r = session.get(url, timeout=(ct, rt))
        try:
            return r.status_code, r.json(), r.text
        except Exception:
            return r.status_code, None, r.text
    except requests.exceptions.Timeout:
        return 0, None, "TIMEOUT"
    except requests.exceptions.RequestException as e:
        return 0, None, f"REQUEST_ERROR: {e}"

def post_json(session: requests.Session, url: str, payload: Dict[str, Any], ct: int, rt: int) -> Tuple[int, Any, str]:
    try:
        r = session.post(url, json=payload, timeout=(ct, rt))
        try:
            return r.status_code, r.json(), r.text
        except Exception:
            return r.status_code, None, r.text
    except requests.exceptions.Timeout:
        return 0, None, "TIMEOUT"
    except requests.exceptions.RequestException as e:
        return 0, None, f"REQUEST_ERROR: {e}"


# -----------------------------
# Placeholder discovery
# -----------------------------
_PLACEHOLDER_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")

def extract_placeholders(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for m in _PLACEHOLDER_RE.finditer(text):
        k = m.group(1) or m.group(2)
        if k and k not in out:
            out.append(k)
    return out

def discover_placeholders(template_obj: Dict[str, Any]) -> List[str]:
    keys = set()
    declared = template_obj.get("placeholders", {}) or {}
    if isinstance(declared, dict):
        keys.update(declared.keys())

    for field in ("prompt", "system_prompt", "user_prompt"):
        keys.update(extract_placeholders(template_obj.get(field, "") or ""))

    return sorted(keys)


# -----------------------------
# Defaults (only used if missing)
# -----------------------------
def default_value(k: str) -> str:
    kl = k.lower()
    if kl in ("input", "text", "content"):
        return "Rewrite this text to be clearer while preserving meaning."
    if kl == "topic":
        return "Prompt patterns"
    if kl == "persona":
        return "a beginner developer"
    if kl == "style":
        return "formal"
    if kl == "schema":
        return '{"name": null, "date": null, "amount": null}'
    if kl == "facts":
        return "deadline: Feb 5; budget: 200 EUR; location: Nuremberg"
    if kl == "question":
        return "How can I structure an academic paper about prompt patterns?"
    if kl == "task":
        return "Summarize the input."
    return f"TEST_VALUE_FOR_{k}"


# -----------------------------
# Optional Gemini judge (pairwise)
# Keep it optional so the runner still works without extra deps.
# -----------------------------
def judge_pairwise_gemini(criteria: str, user_input: str, out_a: str, out_b: str) -> Dict[str, Any]:
    """
    Returns dict with winner A|B|TIE and short rationale.
    Requires google-generativeai.
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"winner": "TIE", "rationale": "GEMINI_API_KEY not set"}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a strict A/B judge.

CRITERIA:
{criteria}

INPUT:
{user_input}

OUTPUT A:
{out_a}

OUTPUT B:
{out_b}

Decide which output is better overall with respect to the criteria.
Return EXACTLY this format, one line each:
WINNER: A|B|TIE
RATIONALE: <one sentence>
"""
    res = model.generate_content(prompt)
    text = (res.text or "").strip()

    winner = "TIE"
    m = re.search(r"WINNER:\s*(A|B|TIE)", text)
    if m:
        winner = m.group(1).strip()

    rat = ""
    m2 = re.search(r"RATIONALE:\s*(.+)", text)
    if m2:
        rat = m2.group(1).strip()

    return {"winner": winner, "rationale": rat, "raw": text[:1200]}


# -----------------------------
# Results
# -----------------------------
@dataclass
class RunRow:
    experiment: str
    case_id: str
    model_name: str
    baseline_id: str
    pattern_id: str
    estimated_tokens_a: int
    estimated_tokens_b: int
    latency_ms_a: int
    latency_ms_b: int
    status_a: str
    status_b: str
    out_a: str
    out_b: str
    judge_winner: str
    judge_rationale: str


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--data-dir", default="test_data", help="Directory with test JSON files")
    ap.add_argument("--outdir", default="ab_results")
    ap.add_argument("--connect-timeout", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=10, help="GET + /render read timeout")
    ap.add_argument("--timeout-exec", type=int, default=120, help="/execute read timeout")
    ap.add_argument("--limit-cases", type=int, default=20, help="Hard cap cases per experiment")
    ap.add_argument("--max-judge-calls", type=int, default=0, help="0 disables judge")
    ap.add_argument("--judge", choices=["none", "gemini"], default="none")
    ap.add_argument("--cache", action="store_true", help="Cache /execute outputs to avoid reruns")
    args = ap.parse_args()

    base = args.api.rstrip("/")
    session = requests.Session()

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    run_dir = outroot / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = run_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Healthcheck
    st, _, raw = get_json(session, f"{base}/", args.connect_timeout, args.timeout)
    if st != 200:
        raise SystemExit(f"API not reachable: {st} {raw[:200]}")

    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob("*.json")])
    if not files:
        raise SystemExit(f"No *.json files found in {data_dir}")

    judge_calls = 0
    all_rows: List[RunRow] = []

    for fp in files:
        cfg = json.loads(fp.read_text(encoding="utf-8"))
        test_name = cfg.get("test_name", fp.stem)
        pattern_id = cfg["pattern_id"]
        baseline_id = cfg["baseline_id"]
        model_name = cfg.get("model_name", "lmstudio-model")
        criteria = cfg.get("criteria", "")
        cases = cfg.get("cases", [])
        cases = cases[: args.limit_cases]

        print(f"\n[EXP] {test_name} cases={len(cases)} pattern={pattern_id} baseline={baseline_id}")

        # Fetch templates to discover placeholders
        stp, pat_obj, rawp = get_json(session, f"{base}/templates/{pattern_id}", args.connect_timeout, args.timeout)
        stb, base_obj, rawb = get_json(session, f"{base}/templates/{baseline_id}", args.connect_timeout, args.timeout)
        if stp != 200:
            print(f"[SKIP] cannot fetch pattern template: {stp} {rawp[:120]}")
            continue
        if stb != 200:
            print(f"[SKIP] cannot fetch baseline template: {stb} {rawb[:120]}")
            continue

        pat_keys = discover_placeholders(pat_obj)
        base_keys = discover_placeholders(base_obj)

        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id", f"{fp.stem}_{idx}")) if isinstance(case, dict) else f"{fp.stem}_{idx}"

            # Expect dict case with params (recommended). Fallback: string case.
            if isinstance(case, dict) and isinstance(case.get("params"), dict):
                shared_params = case["params"]
            elif isinstance(case, str):
                shared_params = {"input": case, "text": case}
            else:
                shared_params = {}

            # Ensure all placeholders are filled (pattern/baseline may differ)
            params_a = dict(shared_params)
            for k in base_keys:
                params_a.setdefault(k, default_value(k))

            params_b = dict(shared_params)
            for k in pat_keys:
                params_b.setdefault(k, default_value(k))

            # ---- Render (debug + token estimate) ----
            est_a = 0
            est_b = 0
            st_ra, j_ra, raw_ra = post_json(
                session, f"{base}/render",
                {"template_id": baseline_id, "params": params_a, "model_name": "gpt-4o-mini"},
                args.connect_timeout, args.timeout
            )
            if st_ra == 200 and isinstance(j_ra, dict):
                est_a = int(j_ra.get("estimated_tokens", 0) or 0)

            st_rb, j_rb, raw_rb = post_json(
                session, f"{base}/render",
                {"template_id": pattern_id, "params": params_b, "model_name": "gpt-4o-mini"},
                args.connect_timeout, args.timeout
            )
            if st_rb == 200 and isinstance(j_rb, dict):
                est_b = int(j_rb.get("estimated_tokens", 0) or 0)

            # ---- Execute (with optional caching) ----
            def exec_one(tid: str, params: Dict[str, str]) -> Tuple[str, str, int]:
                cache_key = stable_hash({"tid": tid, "params": params, "model": model_name})
                cache_path = cache_dir / f"{cache_key}.json"
                if args.cache and cache_path.exists():
                    d = json.loads(cache_path.read_text(encoding="utf-8"))
                    return d["status"], d["out"], int(d["latency_ms"])

                t0 = time.time()
                st_e, j_e, raw_e = post_json(
                    session, f"{base}/execute",
                    {"template_id": tid, "params": params, "model_name": model_name},
                    args.connect_timeout, args.timeout_exec
                )
                ms = int((time.time() - t0) * 1000)

                if st_e == 200 and isinstance(j_e, dict):
                    out = j_e.get("response_text", "") or ""
                    status = "OK"
                elif st_e == 0 and raw_e == "TIMEOUT":
                    out = ""
                    status = "TIMEOUT"
                else:
                    out = ""
                    status = f"ERR_{st_e}"

                if args.cache:
                    cache_path.write_text(json.dumps({"status": status, "out": out, "latency_ms": ms}, ensure_ascii=False), encoding="utf-8")

                return status, out, ms

            status_a, out_a, lat_a = exec_one(baseline_id, params_a)
            status_b, out_b, lat_b = exec_one(pattern_id, params_b)

            # ---- Judge (pairwise, optional & capped) ----
            winner = ""
            rationale = ""
            if args.judge == "gemini" and args.max_judge_calls > 0 and criteria and status_a == "OK" and status_b == "OK":
                if judge_calls < args.max_judge_calls:
                    judge_calls += 1
                    j = judge_pairwise_gemini(criteria, shared_params.get("input", "") or json.dumps(shared_params), out_a, out_b)
                    winner = j.get("winner", "TIE")
                    rationale = j.get("rationale", "")
                else:
                    winner = "SKIPPED_CAP"
                    rationale = "judge cap reached"

            row = RunRow(
                experiment=test_name,
                case_id=case_id,
                model_name=model_name,
                baseline_id=baseline_id,
                pattern_id=pattern_id,
                estimated_tokens_a=est_a,
                estimated_tokens_b=est_b,
                latency_ms_a=lat_a,
                latency_ms_b=lat_b,
                status_a=status_a,
                status_b=status_b,
                out_a=out_a,
                out_b=out_b,
                judge_winner=winner,
                judge_rationale=rationale
            )
            all_rows.append(row)

            print(f"  - case={case_id} A={status_a} B={status_b} judge={winner or '-'}")

    # Write JSONL
    jsonl_path = run_dir / "runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Write CSV (lightweight)
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment","case_id","model_name",
            "baseline_id","pattern_id",
            "estimated_tokens_a","estimated_tokens_b",
            "latency_ms_a","latency_ms_b",
            "status_a","status_b",
            "judge_winner","judge_rationale"
        ])
        for r in all_rows:
            w.writerow([
                r.experiment, r.case_id, r.model_name,
                r.baseline_id, r.pattern_id,
                r.estimated_tokens_a, r.estimated_tokens_b,
                r.latency_ms_a, r.latency_ms_b,
                r.status_a, r.status_b,
                r.judge_winner, r.judge_rationale
            ])

    print(f"\n[DONE] rows={len(all_rows)}")
    print(f"- {jsonl_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
