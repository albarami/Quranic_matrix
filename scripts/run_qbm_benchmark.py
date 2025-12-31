#!/usr/bin/env python
"""
Run QBM benchmark datasets against the local FastAPI app via TestClient.

Outputs:
- reports/benchmarks/qbm_legendary_200_<timestamp>.json  (machine)
- reports/benchmarks/qbm_legendary_200_<timestamp>.md    (human)
- reports/benchmarks/qbm_legendary_200_<timestamp>.csv   (triage)
- reports/benchmarks/failures/<timestamp>/*.json         (per-FAIL artifacts)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class DatasetItem:
    id: str
    section: str
    title: str
    question_ar: str
    question_en: str
    expected: Dict[str, Any]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return items


def load_dataset(path: Path) -> List[DatasetItem]:
    raw = _read_jsonl(path)
    items: List[DatasetItem] = []
    for obj in raw:
        for key in ("id", "section", "title", "question_ar", "question_en", "expected"):
            if key not in obj:
                raise ValueError(f"Dataset item missing '{key}': {obj}")
        items.append(
            DatasetItem(
                id=str(obj["id"]),
                section=str(obj["section"]),
                title=str(obj["title"]),
                question_ar=str(obj["question_ar"]),
                question_en=str(obj["question_en"]),
                expected=dict(obj["expected"]),
            )
        )
    return items


def select_items(
    items: Sequence[DatasetItem],
    limit: Optional[int],
    ids: Optional[Sequence[str]],
    sections: Optional[Sequence[str]],
    smoke: bool,
) -> List[DatasetItem]:
    selected = list(items)

    if ids:
        want = {i.strip() for i in ids if i.strip()}
        selected = [it for it in selected if it.id in want]

    if sections:
        want_sections = {s.strip().upper() for s in sections if s.strip()}
        selected = [it for it in selected if it.section.upper() in want_sections]

    if smoke:
        by_section: Dict[str, List[DatasetItem]] = {}
        for it in selected:
            by_section.setdefault(it.section.upper(), []).append(it)
        # Deterministic: first 2 per section (A–J => 20 items).
        smoke_items: List[DatasetItem] = []
        for sec in sorted(by_section.keys()):
            smoke_items.extend(by_section[sec][:2])
        selected = smoke_items

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be >= 0")
        selected = selected[:limit]

    return selected


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _make_repro_snippets(question: str, payload: Dict[str, Any]) -> Dict[str, str]:
    json_payload = json.dumps(payload, ensure_ascii=False)
    curl = (
        "curl -sS -X POST http://127.0.0.1:8000/api/proof/query "
        "-H \"Content-Type: application/json\" "
        f"--data '{json_payload}'"
    )
    pwsh = (
        "$body = @'\n"
        f"{json_payload}\n"
        "'@\n"
        "$resp = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/proof/query "
        "-ContentType 'application/json' -Body $body\n"
        "$resp | ConvertTo-Json -Depth 50\n"
    )
    return {"curl": curl, "powershell": pwsh, "question": question}


def _extract_counts_for_csv(response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not response:
        return {"quran_verses": 0, "tafsir_chunks": 0, "sources_with_tafsir": 0}
    proof = response.get("proof") if isinstance(response.get("proof"), dict) else {}
    quran = proof.get("quran", [])
    tafsir = proof.get("tafsir", {}) if isinstance(proof.get("tafsir"), dict) else {}
    sources_with_tafsir = 0
    tafsir_chunks = 0
    for chunks in tafsir.values():
        if isinstance(chunks, list) and chunks:
            sources_with_tafsir += 1
            tafsir_chunks += len(chunks)
    return {
        "quran_verses": len(quran) if isinstance(quran, list) else 0,
        "tafsir_chunks": tafsir_chunks,
        "sources_with_tafsir": sources_with_tafsir,
    }


def build_markdown_report(run: Dict[str, Any]) -> str:
    totals = run["summary"]["totals"]
    per_section = run["summary"]["per_section"]
    top_fail_reasons = run["summary"]["top_fail_reasons"]

    lines: List[str] = []
    lines.append("# QBM Benchmark Report")
    lines.append("")
    lines.append(f"- Dataset: `{run['meta']['dataset']}`")
    lines.append(f"- Timestamp (UTC): `{run['meta']['timestamp_utc']}`")
    lines.append(f"- Proof-only: `{run['meta']['proof_only']}`")
    lines.append(f"- Items: `{totals['total']}`")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- PASS: `{totals['PASS']}`")
    lines.append(f"- PARTIAL: `{totals['PARTIAL']}`")
    lines.append(f"- FAIL: `{totals['FAIL']}`")
    lines.append(f"- Schema invalid: `{totals['schema_invalid']}`")
    lines.append("")
    lines.append("## Per Section")
    lines.append("")
    lines.append("| Section | Total | PASS | PARTIAL | FAIL |")
    lines.append("|---|---:|---:|---:|---:|")
    for sec in sorted(per_section.keys()):
        row = per_section[sec]
        lines.append(
            f"| {sec} | {row['total']} | {row['PASS']} | {row['PARTIAL']} | {row['FAIL']} |"
        )
    lines.append("")
    lines.append("## Top FAIL Reasons")
    lines.append("")
    if not top_fail_reasons:
        lines.append("- (none)")
    else:
        for reason, count in top_fail_reasons[:15]:
            lines.append(f"- `{count}` × {reason}")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Open `docs/BENCHMARK_REMEDIATION.md` for the remediation loop.")
    return "\n".join(lines) + "\n"


def _summarize_run(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"total": len(results), "PASS": 0, "PARTIAL": 0, "FAIL": 0, "schema_invalid": 0}
    per_section: Dict[str, Dict[str, int]] = {}
    fail_reasons_count: Dict[str, int] = {}

    for r in results:
        sec = r.get("section", "?")
        verdict = r.get("verdict", "FAIL")
        schema_valid = bool(r.get("schema_valid"))

        totals[verdict] = totals.get(verdict, 0) + 1
        if not schema_valid:
            totals["schema_invalid"] += 1

        per_section.setdefault(sec, {"total": 0, "PASS": 0, "PARTIAL": 0, "FAIL": 0})
        per_section[sec]["total"] += 1
        per_section[sec][verdict] += 1

        if verdict == "FAIL":
            for reason in r.get("reasons", []):
                fail_reasons_count[reason] = fail_reasons_count.get(reason, 0) + 1

    top_fail_reasons = sorted(fail_reasons_count.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"totals": totals, "per_section": per_section, "top_fail_reasons": top_fail_reasons}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run QBM benchmark suite (TestClient, proof-first).")
    parser.add_argument(
        "--dataset",
        default="data/benchmarks/qbm_legendary_200.v1.jsonl",
        help="Path to benchmark JSONL dataset",
    )
    parser.add_argument("--outdir", default="reports/benchmarks", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N items")
    parser.add_argument("--id", dest="ids", action="append", default=None, help="Run only specific id (repeatable)")
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        default=None,
        help="Run only a section letter (repeatable)",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a small, balanced subset (2 per section)")
    parser.add_argument(
        "--proof-only",
        dest="proof_only",
        action="store_true",
        default=True,
        help="Use proof_only=true (default)",
    )
    parser.add_argument(
        "--no-proof-only",
        dest="proof_only",
        action="store_false",
        help="Use proof_only=false (FullPower path; may require prebuilt index)",
    )
    parser.add_argument("--default-mode", choices=["summary", "full"], default="summary")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI gate: return non-zero on schema invalid or runtime errors",
    )
    args = parser.parse_args(argv)

    dataset_path = (REPO_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    outdir = (REPO_ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    _mkdir(outdir)

    timestamp = _now_timestamp()

    items = load_dataset(dataset_path)
    selected = select_items(items, args.limit, args.ids, args.sections, args.smoke)

    from fastapi.testclient import TestClient

    from schemas.proof_response_v2 import validate_response_against_contract
    from src.api.main import app
    from src.benchmarks.scoring import score_benchmark_item

    client = TestClient(app)

    results: List[Dict[str, Any]] = []
    failures_dir = outdir / "failures" / timestamp
    _mkdir(failures_dir)

    for item in selected:
        question = item.question_ar.strip() or item.question_en.strip()
        mode = args.default_mode
        # Future: allow expected to request full mode (without per-question hardcoding).
        if "mode" in item.expected and item.expected.get("mode") in ("summary", "full"):
            mode = item.expected["mode"]

        payload: Dict[str, Any] = {
            "question": question,
            "mode": mode,
            "page": 1,
            "page_size": 20,
            "per_ayah": True,
            "max_chunks_per_source": 1,
            "proof_only": bool(args.proof_only),
        }

        http_status: int
        response_json: Optional[Dict[str, Any]] = None
        schema_valid = False
        schema_issues: List[str] = []
        request_error: Optional[str] = None

        try:
            resp = client.post("/api/proof/query", json=payload)
            http_status = int(resp.status_code)
            if http_status == 200:
                response_json = resp.json()
                schema_valid, schema_issues = validate_response_against_contract(response_json)
            else:
                request_error = resp.text
        except Exception as e:
            http_status = 0
            request_error = f"exception: {type(e).__name__}: {e}"

        scoring = score_benchmark_item(
            benchmark_item=asdict(item),
            response=response_json,
            http_status=http_status,
            request_payload=payload,
            schema_valid=schema_valid,
            schema_issues=schema_issues,
            request_error=request_error,
        )

        record: Dict[str, Any] = {
            "id": item.id,
            "section": item.section,
            "title": item.title,
            "question": question,
            "expected": item.expected,
            "request": payload,
            "http_status": http_status,
            "schema_valid": schema_valid,
            "schema_issues": schema_issues,
            "verdict": scoring["verdict"],
            "reasons": scoring.get("reasons", []),
            "tags": scoring.get("tags", []),
            "metrics": scoring.get("metrics", {}),
            "response": response_json,
        }
        results.append(record)

        if scoring["verdict"] == "FAIL":
            fail_artifact = {
                "benchmark_item": asdict(item),
                "request": payload,
                "http_status": http_status,
                "schema_valid": schema_valid,
                "schema_issues": schema_issues,
                "request_error": request_error,
                "scoring": scoring,
                "response": response_json,
                "repro": _make_repro_snippets(question, payload),
            }
            _write_json(failures_dir / f"{_safe_filename(item.id)}.json", fail_artifact)

    summary = _summarize_run(results)

    run = {
        "meta": {
            "dataset": str(dataset_path.as_posix()),
            "timestamp_utc": timestamp,
            "proof_only": bool(args.proof_only),
            "default_mode": args.default_mode,
            "selected_items": len(selected),
            "total_questions": len(selected),  # Explicit count for CI validation
            "smoke_mode": args.smoke,
            "python": sys.version,
            "cwd": os.getcwd(),
        },
        "summary": summary,
        "results": results,
    }

    stem = dataset_path.stem.replace(".v1", "")
    json_path = outdir / f"{stem}_{timestamp}.json"
    md_path = outdir / f"{stem}_{timestamp}.md"
    csv_path = outdir / f"{stem}_{timestamp}.csv"

    _write_json(json_path, run)
    _write_text(md_path, build_markdown_report(run))
    
    # Write to known path for CI validation
    eval_dir = REPO_ROOT / "reports" / "eval"
    _mkdir(eval_dir)
    latest_path = eval_dir / "latest_eval_report.json"
    _write_json(latest_path, run)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "section",
                "title",
                "verdict",
                "schema_valid",
                "http_status",
                "quran_verses",
                "sources_with_tafsir",
                "tafsir_chunks",
                "reasons",
            ],
        )
        writer.writeheader()
        for r in results:
            counts = _extract_counts_for_csv(r.get("response"))
            writer.writerow(
                {
                    "id": r.get("id"),
                    "section": r.get("section"),
                    "title": r.get("title"),
                    "verdict": r.get("verdict"),
                    "schema_valid": r.get("schema_valid"),
                    "http_status": r.get("http_status"),
                    **counts,
                    "reasons": " | ".join(r.get("reasons", [])),
                }
            )

    if args.ci:
        if summary["totals"]["schema_invalid"] > 0:
            return 2
        # Treat non-200 as CI failure too.
        if any(r.get("http_status") != 200 for r in results):
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

