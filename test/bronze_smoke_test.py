#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def snippet(text: str, n: int = 200) -> str:
    t = (text or "").strip().replace("\r\n", "\n")
    return t if len(t) <= n else (t[:n].rstrip() + " …")


# ---- HTTP helpers (no crash on timeout) ----
def get_json(session: requests.Session, url: str, connect_timeout_s: int, read_timeout_s: int) -> Tuple[int, Any, str]:
    try:
        r = session.get(url, timeout=(connect_timeout_s, read_timeout_s))
        try:
            j = r.json()
        except Exception:
            j = None
        return r.status_code, j, r.text
    except requests.exceptions.Timeout:
        return 0, None, "TIMEOUT"
    except requests.exceptions.RequestException as e:
        return 0, None, f"REQUEST_ERROR: {e}"

def post_json(session: requests.Session, url: str, payload: Dict[str, Any], connect_timeout_s: int, read_timeout_s: int) -> Tuple[int, Dict[str, Any], str]:
    try:
        r = session.post(url, json=payload, timeout=(connect_timeout_s, read_timeout_s))
        try:
            j = r.json()
        except Exception:
            j = {}
        return r.status_code, j, r.text
    except requests.exceptions.Timeout:
        return 0, {}, "TIMEOUT"
    except requests.exceptions.RequestException as e:
        return 0, {}, f"REQUEST_ERROR: {e}"


# ---- placeholder discovery ----
_PLACEHOLDER_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")

def extract_placeholders_from_text(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    for m in _PLACEHOLDER_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if name and name not in found:
            found.append(name)
    return found

def discover_placeholders(template_obj: Dict[str, Any]) -> List[str]:
    keys = set()
    declared = template_obj.get("placeholders", {}) or {}
    if isinstance(declared, dict):
        keys.update(declared.keys())
    for field in ("prompt", "system_prompt", "user_prompt"):
        txt = template_obj.get(field, "") or ""
        keys.update(extract_placeholders_from_text(txt))
    return sorted(keys)

def default_value_for_placeholder(name: str) -> str:
    n = name.lower()

    if n in ("text", "input", "content"):
        return (
            "Luca booked a room from Feb 5 to Feb 10 for 350 EUR. "
            "Address: Example Street 12, 90402 Nuremberg. "
            "Contact: luca@example.com. Notes: late check-in."
        )
    if n in ("task", "goal"):
        return "Rewrite the input to be clearer and more concise while preserving meaning."
    if n == "question":
        return "How can I structure a short academic paper about prompt patterns and template libraries?"
    if n in ("problem", "query"):
        return "I want to compare prompt patterns and decide which ones work best for rewriting text without changing meaning."
    if n == "topic":
        return "Prompt patterns for improving LLM reliability"

    if n == "persona":
        return "a meticulous academic writing assistant"
    if n == "style":
        return "formal and concise"
    if n == "facts":
        return "deadline: Feb 5; budget: 200 EUR; location: Nuremberg"
    if n == "steps":
        return "1) collect requirements 2) draft structure 3) write sections 4) revise 5) finalize references"

    if n == "schema":
        return '{"name": null, "date": null, "amount": null, "notes": null}'

    if n in ("answers", "context"):
        return ""
    if n == "num_questions":
        return "3"
    if n == "termination_condition":
        return "You have enough information to produce a complete outline with headings and bullet points."
    if n == "questions_per_turn":
        return "2"
    if n == "deliverable":
        return "Provide a structured outline with headings and bullet points."

    return f"TEST_VALUE_FOR_{name}"

def build_params(keys: List[str]) -> Dict[str, str]:
    return {k: default_value_for_placeholder(k) for k in keys}


@dataclass
class SmokeResult:
    template_id: str
    status: str
    estimated_tokens: int
    placeholders: List[str]
    rendered_prompt: str
    response_text: str
    error: str
    duration_ms: int


def write_html_report(results: List[SmokeResult], out_path: Path) -> None:
    rows = []
    for r in results:
        cls = "ok" if r.status == "OK" else "fail"
        rows.append(f"""
<tr class="{cls}">
<td><code>{html.escape(r.template_id)}</code></td>
<td><b>{html.escape(r.status)}</b><br><small>{r.duration_ms} ms</small></td>
<td>{r.estimated_tokens}</td>
<td><small>{html.escape(", ".join(r.placeholders) if r.placeholders else "-")}</small></td>
<td><details><summary>Prompt</summary><pre>{html.escape(r.rendered_prompt.strip())}</pre></details></td>
<td><details><summary>Response</summary><pre>{html.escape(r.response_text.strip())}</pre></details></td>
<td><small>{html.escape(r.error)}</small></td>
</tr>
""")

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Bronze Smoke Test Report</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial;margin:24px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;vertical-align:top}}
th{{background:#f5f5f5;text-align:left}}
tr.ok{{background:#f3fff3}} tr.fail{{background:#fff3f3}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace}}
pre{{white-space:pre-wrap;word-break:break-word}}
details summary{{cursor:pointer}}
</style></head><body>
<h1>Bronze Smoke Test Report</h1>
<table>
<thead><tr>
<th>Template</th><th>Status</th><th>Tokens(est)</th><th>Placeholders</th>
<th>Prompt</th><th>Response</th><th>Error</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="lmstudio-model")
    ap.add_argument("--max-templates", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--template-ids", default="")
    ap.add_argument("--connect-timeout", type=int, default=3, help="TCP connect timeout (seconds)")
    ap.add_argument("--timeout", type=int, default=10, help="Read timeout for GET + /render (seconds)")
    ap.add_argument("--timeout-exec", type=int, default=120, help="Read timeout for /execute (seconds)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="bronze_results")
    args = ap.parse_args()

    print(f"[START] api={args.api} model={args.model} dry_run={args.dry_run}", flush=True)

    base = args.api.rstrip("/")
    outdir = Path(args.outdir) / now_tag()
    ensure_dir(outdir)

    session = requests.Session()

    # health
    print("[CHECK] GET /", flush=True)
    st_root, _, raw_root = get_json(session, f"{base}/", args.connect_timeout, args.timeout)
    if st_root != 200:
        print(f"[FAIL] API not reachable: {st_root} {raw_root[:200]}", flush=True)
        sys.exit(2)

    print("[CHECK] GET /templates", flush=True)
    st, templates, raw = get_json(session, f"{base}/templates", args.connect_timeout, args.timeout)
    if st != 200 or not isinstance(templates, list):
        print(f"[FAIL] /templates failed: {st} {raw[:200]}", flush=True)
        sys.exit(2)

    wanted: Optional[List[str]] = None
    if args.template_ids.strip():
        wanted = [x.strip() for x in args.template_ids.split(",") if x.strip()]

    selected = []
    for t in templates:
        tid = t.get("id")
        if not tid:
            continue
        if wanted is not None and tid not in wanted:
            continue
        selected.append(tid)
    selected = selected[:args.max_templates]

    if not selected:
        print("[FAIL] No templates selected.", flush=True)
        sys.exit(2)

    print(f"[INFO] Selected {len(selected)} template(s): {', '.join(selected)}", flush=True)

    results: List[SmokeResult] = []
    call_count = 0

    for i, template_id in enumerate(selected, start=1):
        start = time.time()
        print(f"[{i}/{len(selected)}] Template={template_id} ...", flush=True)

        # fetch template
        st_t, tmpl_obj, raw_t = get_json(session, f"{base}/templates/{template_id}", args.connect_timeout, args.timeout)
        if st_t != 200 or not isinstance(tmpl_obj, dict):
            results.append(SmokeResult(
                template_id=template_id,
                status="TEMPLATE_FETCH_FAIL",
                estimated_tokens=0,
                placeholders=[],
                rendered_prompt="",
                response_text="",
                error=f"/templates/{template_id} {st_t}: {raw_t[:200]}",
                duration_ms=int((time.time() - start) * 1000),
            ))
            continue

        placeholder_keys = discover_placeholders(tmpl_obj)
        params = build_params(placeholder_keys)

        # render
        st_r, j_r, raw_r = post_json(
            session,
            f"{base}/render",
            {"template_id": template_id, "params": params, "model_name": "gpt-4o-mini"},
            args.connect_timeout,
            args.timeout
        )
        if st_r != 200:
            results.append(SmokeResult(
                template_id=template_id,
                status="RENDER_FAIL" if st_r != 0 else "RENDER_TIMEOUT",
                estimated_tokens=0,
                placeholders=placeholder_keys,
                rendered_prompt="",
                response_text="",
                error=f"/render {st_r}: {raw_r[:200]}",
                duration_ms=int((time.time() - start) * 1000),
            ))
            continue

        rendered_prompt = j_r.get("rendered_prompt", "") or ""
        estimated_tokens = int(j_r.get("estimated_tokens", 0) or 0)

        # execute
        response_text = ""
        status = "RENDER_OK"
        err = ""

        if args.dry_run:
            status = "DRY_RUN"
        else:
            if call_count >= args.max_calls:
                status = "SKIPPED_CAP"
                err = f"Skipped: max_calls reached ({args.max_calls})"
            else:
                call_count += 1
                st_e, j_e, raw_e = post_json(
                    session,
                    f"{base}/execute",
                    {"template_id": template_id, "params": params, "model_name": args.model},
                    args.connect_timeout,
                    args.timeout_exec
                )
                if st_e == 200:
                    response_text = j_e.get("response_text", "") or ""
                    status = "OK"
                elif st_e == 0 and raw_e == "TIMEOUT":
                    status = "EXEC_TIMEOUT"
                    err = f"/execute timed out after {args.timeout_exec}s (LM Studio slow?)"
                else:
                    status = "EXEC_FAIL"
                    err = f"/execute {st_e}: {raw_e[:200]}"

        results.append(SmokeResult(
            template_id=template_id,
            status=status,
            estimated_tokens=estimated_tokens,
            placeholders=placeholder_keys,
            rendered_prompt=rendered_prompt,
            response_text=response_text,
            error=err,
            duration_ms=int((time.time() - start) * 1000),
        ))

    # write results
    jsonl_path = outdir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    html_path = outdir / "report.html"
    write_html_report(results, html_path)

    ok = sum(1 for r in results if r.status == "OK")
    print(f"[DONE] OK={ok}/{len(results)}  JSONL={jsonl_path}  HTML={html_path}", flush=True)


if __name__ == "__main__":
    main()
