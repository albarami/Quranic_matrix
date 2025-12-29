"""
Phase 11 remediation test: CROSS_CONTEXT_BEHAVIOR intent must be deterministic and fail-closed.

Checks:
- Intent classification via QueryRouter
- No generic opening-verse defaults
- Multi-context output (>=2 clusters) for a known behavior (الصبر)
"""

import json
from pathlib import Path


def _concept_verse_keys(concept_id: str) -> set[str]:
    p = Path("data/evidence/concept_index_v2.jsonl")
    keys: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if concept_id not in line:
                continue
            obj = json.loads(line)
            if obj.get("concept_id") != concept_id:
                continue
            for v in obj.get("verses", []):
                if isinstance(v, dict) and isinstance(v.get("verse_key"), str):
                    keys.add(v["verse_key"])
            break
    assert keys, f"Missing verse_keys for {concept_id} in concept_index_v2.jsonl"
    return keys


def test_cross_context_intent_classification():
    from src.ml.query_router import get_query_router, QueryIntent

    router = get_query_router()
    q = "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة"
    r = router.route(q)
    assert r.intent == QueryIntent.CROSS_CONTEXT_BEHAVIOR


def test_cross_context_no_generic_default_and_multi_context():
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

    assert data.get("status") == "success", f"Expected success, got {data.get('status')}"

    verse_keys = [v.get("verse_key") for v in data.get("proof", {}).get("quran", []) if isinstance(v, dict)]
    assert len(verse_keys) >= 2, "Expected >=2 representative verses"

    patience_keys = _concept_verse_keys("BEH_EMO_PATIENCE")
    assert all(vk in patience_keys for vk in verse_keys), "Hard gate violated: verse not in concept index"

    generic = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}
    assert not all(vk in generic for vk in verse_keys), "Generic opening verses default detected"
