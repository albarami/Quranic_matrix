"""
Phase 10.5: External Validity - Manual Scholarly Spot-Check Pack

This module generates a curated set of test cases for manual scholarly review.
Each test case includes:
- Query
- System response (answer + evidence)
- Expected scholarly validation points

Run with: pytest tests/test_scholarly_spotcheck.py -v -s --tb=short
"""
import pytest
import json
from pathlib import Path
from datetime import datetime

# Output directory for spot-check reports
SPOTCHECK_DIR = Path("reports/scholarly_spotcheck")


@pytest.fixture(scope="module")
def client():
    """Create test client for API."""
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")


@pytest.fixture(scope="module")
def spotcheck_queries():
    """Curated queries for scholarly validation."""
    return [
        {
            "id": "SC001",
            "query": "ما هو الحسد وما آثاره؟",
            "query_en": "What is envy and what are its effects?",
            "category": "behavior_definition",
            "validation_points": [
                "Definition matches classical tafsir understanding",
                "Verses cited are relevant to envy (hasad)",
                "Effects mentioned are Quranically grounded",
            ],
        },
        {
            "id": "SC002",
            "query": "ما العلاقة بين الكبر والكفر؟",
            "query_en": "What is the relationship between arrogance and disbelief?",
            "category": "causal_chain",
            "validation_points": [
                "Causal relationship is supported by tafsir",
                "Iblis example is correctly cited",
                "Chain of causation is logically sound",
            ],
        },
        {
            "id": "SC003",
            "query": "كيف يؤثر القلب السليم على السلوك؟",
            "query_en": "How does a sound heart affect behavior?",
            "category": "heart_state",
            "validation_points": [
                "Definition of qalb saleem is accurate",
                "Behavioral effects are Quranically grounded",
                "Tafsir quotes support the claims",
            ],
        },
        {
            "id": "SC004",
            "query": "ما هي صفات المنافقين في القرآن؟",
            "query_en": "What are the characteristics of hypocrites in the Quran?",
            "category": "agent_profile",
            "validation_points": [
                "Characteristics match Quranic descriptions",
                "Verses from Surah Al-Baqarah and Al-Munafiqun cited",
                "Tafsir explanations are accurate",
            ],
        },
        {
            "id": "SC005",
            "query": "ما الفرق بين الصبر والرضا؟",
            "query_en": "What is the difference between patience and contentment?",
            "category": "concept_comparison",
            "validation_points": [
                "Distinction is theologically accurate",
                "Both concepts have Quranic evidence",
                "Scholarly consensus is reflected",
            ],
        },
        {
            "id": "SC006",
            "query": "ما عواقب الربا في الدنيا والآخرة؟",
            "query_en": "What are the consequences of usury in this world and the hereafter?",
            "category": "consequence_analysis",
            "validation_points": [
                "Worldly consequences are mentioned",
                "Afterlife consequences are mentioned",
                "Verses from Surah Al-Baqarah cited",
            ],
        },
        {
            "id": "SC007",
            "query": "كيف يعالج القرآن الغضب؟",
            "query_en": "How does the Quran address anger?",
            "category": "behavioral_remedy",
            "validation_points": [
                "Quranic remedies are accurately cited",
                "Prophetic guidance may be referenced",
                "Tafsir explanations support the remedies",
            ],
        },
        {
            "id": "SC008",
            "query": "ما هي مراتب التوبة؟",
            "query_en": "What are the levels of repentance?",
            "category": "spiritual_stages",
            "validation_points": [
                "Stages are theologically sound",
                "Quranic evidence for each stage",
                "Scholarly understanding is reflected",
            ],
        },
        {
            "id": "SC009",
            "query": "ما دور اللسان في السلوك القرآني؟",
            "query_en": "What is the role of the tongue in Quranic behavior?",
            "category": "organ_behavior",
            "validation_points": [
                "Tongue-related behaviors are identified",
                "Positive and negative uses mentioned",
                "Verses about speech are cited",
            ],
        },
        {
            "id": "SC010",
            "query": "ما هي علامات الإيمان الصادق؟",
            "query_en": "What are the signs of true faith?",
            "category": "faith_indicators",
            "validation_points": [
                "Signs match Quranic descriptions",
                "Both internal and external signs mentioned",
                "Tafsir support is accurate",
            ],
        },
    ]


@pytest.mark.phase10
class TestScholarlySpotcheckGeneration:
    """Generate spot-check pack for scholarly review."""
    
    def test_generate_spotcheck_pack(self, client, spotcheck_queries):
        """Generate complete spot-check pack with system responses."""
        SPOTCHECK_DIR.mkdir(parents=True, exist_ok=True)
        
        results = []
        for sq in spotcheck_queries:
            response = client.post(
                "/api/proof/query",
                json={"question": sq["query"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    "id": sq["id"],
                    "query": sq["query"],
                    "query_en": sq["query_en"],
                    "category": sq["category"],
                    "validation_points": sq["validation_points"],
                    "system_response": {
                        "answer": data.get("answer", "")[:500],  # Truncate for readability
                        "quran_verses": data.get("proof", {}).get("quran", [])[:5],
                        "tafsir_summary": {
                            source: len(data.get("proof", {}).get(source, []))
                            for source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
                        },
                    },
                    "status": "generated",
                }
            else:
                result = {
                    "id": sq["id"],
                    "query": sq["query"],
                    "status": "error",
                    "error": response.text[:200],
                }
            
            results.append(result)
        
        # Write spot-check pack
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = SPOTCHECK_DIR / f"spotcheck_pack_{timestamp}.json"
        
        pack = {
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(spotcheck_queries),
            "successful": sum(1 for r in results if r.get("status") == "generated"),
            "queries": results,
            "instructions": {
                "purpose": "Manual scholarly validation of QBM system outputs",
                "process": [
                    "1. Review each query and system response",
                    "2. Check validation points for accuracy",
                    "3. Mark as PASS/FAIL with notes",
                    "4. Return completed pack for analysis",
                ],
                "rating_scale": {
                    "PASS": "All validation points satisfied",
                    "PARTIAL": "Some validation points satisfied",
                    "FAIL": "Critical errors or fabrications found",
                },
            },
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SPOTCHECK] Generated pack: {output_file}")
        print(f"[SPOTCHECK] Total queries: {len(spotcheck_queries)}")
        print(f"[SPOTCHECK] Successful: {pack['successful']}")
        
        # Test passes if we generated at least 8/10 successfully
        assert pack["successful"] >= 8, f"Only {pack['successful']}/10 queries succeeded"


@pytest.mark.phase10
class TestSpotcheckQualityGates:
    """Quality gates for spot-check queries."""
    
    def test_all_queries_return_quran_verses(self, client, spotcheck_queries):
        """All spot-check queries should return Quran verses."""
        queries_without_verses = []
        
        for sq in spotcheck_queries:
            response = client.post(
                "/api/proof/query",
                json={"question": sq["query"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                quran = data.get("proof", {}).get("quran", [])
                if len(quran) == 0:
                    queries_without_verses.append(sq["id"])
        
        assert len(queries_without_verses) == 0, \
            f"Queries without Quran verses: {queries_without_verses}"
    
    def test_all_queries_return_tafsir(self, client, spotcheck_queries):
        """All spot-check queries should return tafsir quotes."""
        queries_without_tafsir = []
        
        for sq in spotcheck_queries:
            response = client.post(
                "/api/proof/query",
                json={"question": sq["query"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                proof = data.get("proof", {})
                total_tafsir = sum(
                    len(proof.get(s, []))
                    for s in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
                )
                if total_tafsir == 0:
                    queries_without_tafsir.append(sq["id"])
        
        assert len(queries_without_tafsir) == 0, \
            f"Queries without tafsir: {queries_without_tafsir}"
    
    def test_answers_are_substantive(self, client, spotcheck_queries):
        """All answers should be substantive (>100 chars)."""
        short_answers = []
        
        for sq in spotcheck_queries:
            response = client.post(
                "/api/proof/query",
                json={"question": sq["query"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer") or ""
                if len(answer) < 100:
                    short_answers.append((sq["id"], len(answer)))
        
        assert len(short_answers) == 0, \
            f"Queries with short answers: {short_answers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "phase10"])
