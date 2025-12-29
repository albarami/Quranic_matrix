"""
Metrics Router - /api/metrics/*

Enterprise rule: frontend renders metrics from backend truth JSON only.
This router serves the precomputed artifact deterministically (no LLM, no recomputation).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

TRUTH_METRICS_FILE = DATA_DIR / "metrics" / "truth_metrics_v1.json"


@router.get("/overview")
async def get_metrics_overview():
    """
    Deterministic metrics endpoint - SOURCE OF TRUTH for all statistics.

    Serves the pre-computed truth_metrics_v1.json artifact VERBATIM.
    Frontend MUST use this endpoint for displaying metrics.

    If the truth file is missing/corrupted/not ready, returns HTTP 503 with instructions.
    """
    if not TRUTH_METRICS_FILE.exists():
        return JSONResponse(
            status_code=503,
            content={
                "error": "metrics_truth_missing",
                "status": "no_data",
                "how_to_fix": "run: python scripts/build_truth_metrics_v1.py",
                "expected_file": str(TRUTH_METRICS_FILE),
            },
        )

    try:
        with open(TRUTH_METRICS_FILE, "r", encoding="utf-8") as f:
            truth_metrics = json.load(f)

        if truth_metrics.get("status") != "ready":
            return JSONResponse(
                status_code=503,
                content={
                    "error": "metrics_not_ready",
                    "status": truth_metrics.get("status", "unknown"),
                    "how_to_fix": "rebuild: python scripts/build_truth_metrics_v1.py",
                },
            )

        return truth_metrics

    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": "metrics_corrupted",
                "detail": str(e),
                "how_to_fix": "rebuild: python scripts/build_truth_metrics_v1.py",
            },
        )

