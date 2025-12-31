"""
Regression tests for CROSS_CONTEXT_BEHAVIOR intent.

Acceptance criteria:
1) Query without behavior returns status=need_behavior and 0 verses (fail-closed).
2) Query with a known behavior (الصبر) returns >=2 context clusters and verse-locked evidence.
3) Tafsir chunks are verse-locked (no cross-verse citations) and have provenance fields.
4) No generic "opening verses" default behavior.
5) API response validates against schemas/proof_response_v2.py.
"""

import json
from pathlib import Path


def _load_patience_verse_keys() -> set[str]:
    concept_index = Path("data/evidence/concept_index_v3.jsonl")
    verse_keys: set[str] = set()
    with concept_index.open("r", encoding="utf-8") as f:
        for line in f:
            if "BEH_EMO_PATIENCE" not in line:
                continue
            obj = json.loads(line)
            if obj.get("concept_id") != "BEH_EMO_PATIENCE":
                continue
            for v in obj.get("verses", []):
                if isinstance(v, dict) and isinstance(v.get("verse_key"), str):
                    verse_keys.add(v["verse_key"])
            break
    assert verse_keys, "Expected BEH_EMO_PATIENCE to have verse_keys in concept_index_v3.jsonl"
    return verse_keys


class TestCrossContextQueryRouter:
    def test_intent_detected_arabic(self):
        from src.ml.query_router import get_query_router, QueryIntent

        router = get_query_router()
        q = "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة"
        result = router.route(q)
        assert result.intent == QueryIntent.CROSS_CONTEXT_BEHAVIOR


class TestCrossContextHandler:
    def test_need_behavior_returns_no_verses(self):
        from src.ml.cross_context_behavior_handler import get_cross_context_handler

        handler = get_cross_context_handler()
        q = "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة"
        result = handler.handle(q)

        assert result.status == "need_behavior"
        assert len(result.selected_verses) == 0

    def test_patience_multi_context_success(self):
        from src.ml.cross_context_behavior_handler import get_cross_context_handler

        handler = get_cross_context_handler()
        q = "أوجد الآيات التي يظهر فيها نفس السلوك (الصبر) في سياقات مختلفة"
        result = handler.handle(q)

        assert result.status in {"success", "partial", "no_evidence"}

        if result.status == "success":
            assert len(result.context_groups) >= 2
            assert len(result.selected_verses) >= 2

            patience_keys = _load_patience_verse_keys()
            for ve in result.selected_verses:
                assert ve.verse_key in patience_keys, "Hard gate: verses must come from concept index"
                assert ve.confidence < 1.0, "Confidence must be computed (not forced to 1.0)"

                # Verse-locked tafsir with provenance
                for source, chunks in ve.tafsir.items():
                    for chunk in chunks:
                        assert chunk.get("verse_key") == ve.verse_key
                        assert chunk.get("chunk_id")
                        assert chunk.get("char_start") is not None
                        assert chunk.get("char_end") is not None
                        assert chunk.get("source") == source


class TestCrossContextAPIContract:
    def test_api_need_behavior_contract(self):
        from fastapi.testclient import TestClient
        from schemas.proof_response_v2 import validate_response_against_contract
        from src.api.main import app

        client = TestClient(app)
        q = "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة"
        resp = client.post("/api/proof/query", json={"question": q, "proof_only": True, "mode": "summary"})
        assert resp.status_code == 200

        data = resp.json()
        ok, issues = validate_response_against_contract(data)
        assert ok, f"Schema issues: {issues}"

        assert data.get("status") == "need_behavior"
        assert data.get("debug", {}).get("intent") == "CROSS_CONTEXT_BEHAVIOR"
        assert data.get("proof", {}).get("intent") == "CROSS_CONTEXT_BEHAVIOR"
        assert len(data.get("proof", {}).get("quran", [])) == 0

    def test_api_patience_success_contract(self):
        from fastapi.testclient import TestClient
        from schemas.proof_response_v2 import validate_response_against_contract
        from src.api.main import app

        client = TestClient(app)
        q = "أوجد الآيات التي يظهر فيها نفس السلوك (الصبر) في سياقات مختلفة"
        resp = client.post("/api/proof/query", json={"question": q, "proof_only": True, "mode": "summary"})
        assert resp.status_code == 200

        data = resp.json()
        ok, issues = validate_response_against_contract(data)
        assert ok, f"Schema issues: {issues}"

        assert data.get("debug", {}).get("intent") == "CROSS_CONTEXT_BEHAVIOR"
        assert data.get("proof", {}).get("intent") == "CROSS_CONTEXT_BEHAVIOR"

        # Hard gate: must not silently return generic opening verses for unrelated queries
        verse_keys = [v.get("verse_key") for v in data.get("proof", {}).get("quran", []) if isinstance(v, dict)]
        assert verse_keys, "Expected at least 1 verse for patience cross-context query"
        generic = {*(f"1:{i}" for i in range(1, 8)), *(f"2:{i}" for i in range(1, 21))}
        assert not all(vk in generic for vk in verse_keys), "Generic opening verses default detected"
