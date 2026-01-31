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
from typing import Any, Dict, List, Tuple

import requests

# Optional: load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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
    if kl == "role":
        return "a helpful assistant"
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
    if kl == "output_language":
        return "en"
    return f"TEST_VALUE_FOR_{k}"


# -----------------------------
# Stable hash for caching
# -----------------------------
def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# Judge helpers
# -----------------------------
def _truncate(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[TRUNCATED]..."


def _parse_judge(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    winner = "TIE"
    m = re.search(r"WINNER:\s*(A|B|TIE)", text)
    if m:
        winner = m.group(1).strip()

    rationale = ""
    m2 = re.search(r"RATIONALE:\s*(.+)", text)
    if m2:
        rationale = m2.group(1).strip()

    return {"winner": winner, "rationale": rationale, "raw": text[:1200]}


def judge_pairwise_gemini(
    criteria: str,
    user_input: str,
    out_a: str,
    out_b: str,
    judge_model: str,
    max_chars: int,
) -> Dict[str, Any]:
    """
    Gemini/Gemma judge via google-genai (new SDK).
    Requires: pip install google-genai
    Uses: GEMINI_API_KEY env var
    """
    try:
        from google import genai
    except ModuleNotFoundError:
        return {"winner": "SKIPPED_NO_DEP", "rationale": "Missing dependency. Install: pip install google-genai"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"winner": "SKIPPED_NO_KEY", "rationale": "GEMINI_API_KEY not set"}

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a strict A/B judge.

CRITERIA:
{criteria}

INPUT:
{_truncate(user_input, max_chars)}

OUTPUT A:
{_truncate(out_a, max_chars)}

OUTPUT B:
{_truncate(out_b, max_chars)}

Decide which output is better overall with respect to the criteria.
Return EXACTLY this format, one line each:
WINNER: A|B|TIE
RATIONALE: <one sentence>
""".strip()

    try:
        resp = client.models.generate_content(model=judge_model, contents=prompt)
        text = getattr(resp, "text", None) or ""
        # Fallback for some SDK response shapes
        if not text:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        return _parse_judge(text)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "429" in msg or "resource_exhausted" in low or "rate" in low:
            return {"winner": "SKIPPED_RATELIMIT", "rationale": msg[:200]}
        return {"winner": "SKIPPED_JUDGE_ERR", "rationale": msg[:200]}


def judge_pairwise_openai(
    criteria: str,
    user_input: str,
    out_a: str,
    out_b: str,
    judge_model: str,
    max_chars: int,
) -> Dict[str, Any]:
    """
    OpenAI judge via official Python SDK.
    Requires: pip install openai
    Uses: OPENAI_API_KEY env var
    """
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return {"winner": "SKIPPED_NO_DEP", "rationale": "Missing dependency. Install: pip install openai"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"winner": "SKIPPED_NO_KEY", "rationale": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a strict A/B judge.

CRITERIA:
{criteria}

INPUT:
{_truncate(user_input, max_chars)}

OUTPUT A:
{_truncate(out_a, max_chars)}

OUTPUT B:
{_truncate(out_b, max_chars)}

Decide which output is better overall with respect to the criteria.
Return EXACTLY this format, one line each:
WINNER: A|B|TIE
RATIONALE: <one sentence>
""".strip()

    try:
        resp = client.responses.create(model=judge_model, input=prompt)
        text = getattr(resp, "output_text", None) or ""
        if not text:
            # Fallback for older response shapes
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = ""
        return _parse_judge(text)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "429" in msg or "rate" in low:
            return {"winner": "SKIPPED_RATELIMIT", "rationale": msg[:200]}
        return {"winner": "SKIPPED_JUDGE_ERR", "rationale": msg[:200]}


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
    judge_provider: str
    judge_model: str
    judge_winner: str
    judge_rationale: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--data-dir", default="test_data", help="Directory with test JSON files")
    ap.add_argument("--outdir", default="ab_results")
    ap.add_argument("--connect-timeout", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=10, help="GET + /render read timeout")
    ap.add_argument("--timeout-exec", type=int, default=120, help="/execute read timeout")
    ap.add_argument("--limit-cases", type=int, default=20, help="Hard cap cases per experiment")

    ap.add_argument("--cache", action="store_true", help="Cache /execute + judge outputs to avoid reruns")
    ap.add_argument("--render-model", default="gpt-4o-mini", help="Model name used only for token estimates via /render")

    # Judge
    ap.add_argument("--judge", choices=["none", "gemini", "openai"], default="none")
    ap.add_argument("--judge-model", default="", help="Judge model name (provider-specific)")
    ap.add_argument("--max-judge-calls", type=int, default=0, help="0 disables judge entirely")
    ap.add_argument("--max-judge-per-exp", type=int, default=0, help="0 means no per-experiment cap")
    ap.add_argument("--judge-delay", type=float, default=0.0, help="Sleep seconds after each judge call (rate limit safety)")
    ap.add_argument("--judge-max-chars", type=int, default=8000, help="Truncate INPUT/A/B for judge prompt (chars)")

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

    judge_calls_global = 0
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

        pat_keys = discover_placeholders(pat_obj if isinstance(pat_obj, dict) else {})
        base_keys = discover_placeholders(base_obj if isinstance(base_obj, dict) else {})

        judge_calls_exp = 0

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
                {"template_id": baseline_id, "params": params_a, "model_name": args.render_model},
                args.connect_timeout, args.timeout
            )
            if st_ra == 200 and isinstance(j_ra, dict):
                est_a = int(j_ra.get("estimated_tokens", 0) or 0)

            st_rb, j_rb, raw_rb = post_json(
                session, f"{base}/render",
                {"template_id": pattern_id, "params": params_b, "model_name": args.render_model},
                args.connect_timeout, args.timeout
            )
            if st_rb == 200 and isinstance(j_rb, dict):
                est_b = int(j_rb.get("estimated_tokens", 0) or 0)

            # ---- Execute (with optional caching) ----
            def exec_one(tid: str, params: Dict[str, str]) -> Tuple[str, str, int]:
                cache_key = stable_hash({"tid": tid, "params": params, "model": model_name})
                cache_path = cache_dir / f"exec_{cache_key}.json"

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
                    cache_path.write_text(
                        json.dumps({"status": status, "out": out, "latency_ms": ms}, ensure_ascii=False),
                        encoding="utf-8"
                    )

                return status, out, ms

            status_a, out_a, lat_a = exec_one(baseline_id, params_a)
            status_b, out_b, lat_b = exec_one(pattern_id, params_b)

            # ---- Judge (pairwise, optional & capped + cached) ----
            judge_provider = args.judge
            judge_model = ""
            winner = ""
            rationale = ""

            can_judge = (
                args.judge != "none"
                and args.max_judge_calls > 0
                and bool(criteria)
                and status_a == "OK"
                and status_b == "OK"
            )

            if can_judge:
                if judge_calls_global >= args.max_judge_calls:
                    winner = "SKIPPED_CAP"
                    rationale = "global judge cap reached"
                elif args.max_judge_per_exp > 0 and judge_calls_exp >= args.max_judge_per_exp:
                    winner = "SKIPPED_CAP"
                    rationale = "per-experiment judge cap reached"
                else:
                    # Decide judge model defaults
                    if args.judge == "gemini":
                        judge_model = args.judge_model or "gemma-3-27b-it"
                    else:
                        judge_model = args.judge_model or "gpt-4o-mini"

                    # Judge cache key (only if --cache)
                    user_input_for_judge = shared_params.get("input", "") or json.dumps(shared_params, ensure_ascii=False)
                    judge_cache_key = stable_hash({
                        "provider": args.judge,
                        "judge_model": judge_model,
                        "criteria": criteria,
                        "user_input": user_input_for_judge,
                        "out_a": out_a,
                        "out_b": out_b,
                        "max_chars": args.judge_max_chars,
                    })
                    judge_cache_path = cache_dir / f"judge_{judge_cache_key}.json"

                    if args.cache and judge_cache_path.exists():
                        d = json.loads(judge_cache_path.read_text(encoding="utf-8"))
                        winner = d.get("winner", "TIE")
                        rationale = d.get("rationale", "")
                    else:
                        judge_calls_global += 1
                        judge_calls_exp += 1

                        if args.judge == "gemini":
                            j = judge_pairwise_gemini(
                                criteria, user_input_for_judge, out_a, out_b,
                                judge_model, args.judge_max_chars
                            )
                        else:
                            j = judge_pairwise_openai(
                                criteria, user_input_for_judge, out_a, out_b,
                                judge_model, args.judge_max_chars
                            )

                        winner = j.get("winner", "TIE")
                        rationale = j.get("rationale", "")

                        if args.cache:
                            judge_cache_path.write_text(
                                json.dumps({"winner": winner, "rationale": rationale}, ensure_ascii=False),
                                encoding="utf-8"
                            )

                        if args.judge_delay and args.judge_delay > 0:
                            time.sleep(args.judge_delay)

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
                judge_provider=judge_provider,
                judge_model=judge_model,
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
            "experiment", "case_id", "model_name",
            "baseline_id", "pattern_id",
            "estimated_tokens_a", "estimated_tokens_b",
            "latency_ms_a", "latency_ms_b",
            "status_a", "status_b",
            "judge_provider", "judge_model",
            "judge_winner", "judge_rationale"
        ])
        for r in all_rows:
            w.writerow([
                r.experiment, r.case_id, r.model_name,
                r.baseline_id, r.pattern_id,
                r.estimated_tokens_a, r.estimated_tokens_b,
                r.latency_ms_a, r.latency_ms_b,
                r.status_a, r.status_b,
                r.judge_provider, r.judge_model,
                r.judge_winner, r.judge_rationale
            ])

    print(f"\n[DONE] rows={len(all_rows)}")
    print(f"- {jsonl_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
