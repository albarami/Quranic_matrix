"""
QBM Test Configuration - Phase 0 Instrumentation
Provides fixtures and helpers for no-fallback testing.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# NO-FALLBACK ASSERTION HELPERS
# =============================================================================

class FallbackDetectedError(AssertionError):
    """Raised when a fallback was used during proof generation"""
    
    def __init__(self, debug_info: Dict[str, Any]):
        self.debug_info = debug_info
        reasons = debug_info.get("fallback_reasons", [])
        components = debug_info.get("component_fallbacks", {})
        
        message = "FALLBACK DETECTED - Primary path did not work!\n"
        message += f"  fallback_used: {debug_info.get('fallback_used', 'unknown')}\n"
        message += f"  reasons: {reasons}\n"
        message += f"  component_fallbacks: {components}\n"
        
        super().__init__(message)


def assert_no_fallback(response: Dict[str, Any]) -> None:
    """
    Assert that no fallback was used in the proof response.
    
    This is the core Phase 0 invariant check. If this fails, it means
    the primary retrieval path did not work and a fallback patched it.
    
    Args:
        response: The response from answer_with_full_proof() or /api/proof/query
        
    Raises:
        FallbackDetectedError: If any fallback was used
        KeyError: If debug info is missing from response
    """
    debug = response.get("debug")
    
    if debug is None:
        raise KeyError(
            "Response missing 'debug' field. "
            "Ensure Phase 0 instrumentation is applied."
        )
    
    if debug.get("fallback_used", False):
        raise FallbackDetectedError(debug)


def assert_source_distribution(
    response: Dict[str, Any],
    min_per_source: int = 5,
    required_sources: List[str] = None
) -> None:
    """
    Assert that all required sources have minimum results.
    
    Args:
        response: The response from answer_with_full_proof() or /api/proof/query
        min_per_source: Minimum results required per source
        required_sources: List of required source names
        
    Raises:
        AssertionError: If any source is below minimum
    """
    if required_sources is None:
        required_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    
    debug = response.get("debug", {})
    distribution = debug.get("retrieval_distribution", {})
    
    missing = []
    for source in required_sources:
        count = distribution.get(source, 0)
        if count < min_per_source:
            missing.append(f"{source}: {count} (need {min_per_source})")
    
    if missing:
        raise AssertionError(
            f"Source distribution below minimum:\n  " + 
            "\n  ".join(missing)
        )


# =============================================================================
# FIXTURES
# =============================================================================

def _is_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def qbm_system():
    """
    Initialize the full power QBM system with proof system.
    Session-scoped to avoid repeated initialization.
    Requires GPU - use qbm_system_mini for CPU-only testing.
    """
    if not _is_gpu_available():
        pytest.skip("GPU not available - use qbm_system_mini for CPU-only testing")
    
    from src.ml.full_power_system import FullPowerQBMSystem
    from src.ml.mandatory_proof_system import integrate_with_system
    
    system = FullPowerQBMSystem()
    
    # Build index if not already built
    status = system.get_status()
    if status["vector_search"].get("status") == "not_built":
        system.build_index()
        system.build_graph()
    
    # Add proof system
    system = integrate_with_system(system)
    
    return system


# =============================================================================
# MINI-FIXTURES FOR CPU-ONLY TESTING (Phase 1)
# =============================================================================

@pytest.fixture(scope="session")
def qbm_system_mini():
    """
    Lightweight QBM system for CPU-only testing.
    Uses mock embeddings and minimal data for fast CI runs.
    """
    from unittest.mock import MagicMock
    
    # Create a minimal mock system
    system = MagicMock()
    system.index_source = "disk"
    system.tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
    
    # Mock search to return sample results
    def mock_search(query, top_k=10, ensure_source_diversity=False):
        return [
            {"text": "Sample Quran verse", "score": 0.9, "metadata": {"type": "quran", "surah": 2, "ayah": 255}},
            {"text": "Sample Ibn Kathir", "score": 0.85, "metadata": {"source": "ibn_kathir", "surah": 2, "ayah": 255}},
            {"text": "Sample Tabari", "score": 0.8, "metadata": {"source": "tabari", "surah": 2, "ayah": 255}},
            {"text": "Sample Qurtubi", "score": 0.75, "metadata": {"source": "qurtubi", "surah": 2, "ayah": 255}},
            {"text": "Sample Saadi", "score": 0.7, "metadata": {"source": "saadi", "surah": 2, "ayah": 255}},
            {"text": "Sample Jalalayn", "score": 0.65, "metadata": {"source": "jalalayn", "surah": 2, "ayah": 255}},
        ]
    
    system.search = mock_search
    system.get_status = lambda: {"vector_search": {"status": "ready"}}
    
    return system


@pytest.fixture
def mock_proof_response():
    """
    Sample proof response for testing without GPU.
    """
    return {
        "question": "ما هو الصبر؟",
        "answer": "الصبر هو حبس النفس على ما تكره",
        "validation": {"score": 100.0, "passed": True, "missing": []},
        "processing_time_ms": 150.0,
        "debug": {
            "fallback_used": False,
            "fallback_reasons": [],
            "retrieval_distribution": {
                "ibn_kathir": 10,
                "tabari": 10,
                "qurtubi": 10,
                "saadi": 10,
                "jalalayn": 10,
                "quran": 15
            },
            "primary_path_latency_ms": 150,
            "index_source": "disk",
            "component_fallbacks": {
                "quran": False,
                "graph": False,
                "taxonomy": False,
                "tafsir": {
                    "ibn_kathir": False,
                    "tabari": False,
                    "qurtubi": False,
                    "saadi": False,
                    "jalalayn": False
                }
            }
        }
    }


@pytest.fixture
def mock_fallback_response():
    """
    Sample proof response with fallback for testing.
    """
    return {
        "question": "ما هو الصبر؟",
        "answer": "الصبر هو حبس النفس على ما تكره",
        "validation": {"score": 100.0, "passed": True, "missing": []},
        "processing_time_ms": 200.0,
        "debug": {
            "fallback_used": True,
            "fallback_reasons": ["quran: primary retrieval returned 0 verses, using surah name extraction"],
            "retrieval_distribution": {"ibn_kathir": 5, "tabari": 3},
            "primary_path_latency_ms": 200,
            "index_source": "runtime_build",
            "component_fallbacks": {
                "quran": True,
                "graph": False,
                "taxonomy": False,
                "tafsir": {
                    "ibn_kathir": False,
                    "tabari": True,
                    "qurtubi": False,
                    "saadi": True,
                    "jalalayn": False
                }
            }
        }
    }


@pytest.fixture(scope="session")
def strict_client(qbm_system):
    """
    A test client that enforces no-fallback invariant.
    Returns a wrapper that automatically checks for fallbacks.
    """
    class StrictProofClient:
        def __init__(self, system):
            self.system = system
        
        def query(self, question: str, allow_fallback: bool = False) -> Dict[str, Any]:
            """
            Run a proof query with optional fallback checking.
            
            Args:
                question: The question to ask
                allow_fallback: If False (default), raises error on fallback
                
            Returns:
                The proof response
            """
            response = self.system.answer_with_full_proof(question)
            
            if not allow_fallback:
                assert_no_fallback(response)
            
            return response
    
    return StrictProofClient(qbm_system)


# =============================================================================
# STANDARD TEST QUERIES
# =============================================================================

STANDARD_QUERIES = [
    "ما هو الصبر؟",
    "ما هو الكبر؟", 
    "ما هي التقوى؟",
    "حلل سلوك الكبر",
    "قارن تفسير البقرة:7 عند الخمسة",
]


@pytest.fixture
def standard_queries():
    """Return the standard test queries for no-fallback testing."""
    return STANDARD_QUERIES


# =============================================================================
# BASELINE REPORT GENERATION
# =============================================================================

def generate_baseline_report(system, queries: List[str] = None) -> Dict[str, Any]:
    """
    Generate a baseline report of fallback usage.
    
    Args:
        system: The QBM system with proof capability
        queries: List of queries to test (defaults to STANDARD_QUERIES)
        
    Returns:
        Dictionary with baseline metrics
    """
    if queries is None:
        queries = STANDARD_QUERIES
    
    results = []
    total_fallbacks = 0
    component_fallback_counts = {
        "quran": 0,
        "graph": 0,
        "taxonomy": 0,
        "tafsir": {"ibn_kathir": 0, "tabari": 0, "qurtubi": 0, "saadi": 0, "jalalayn": 0}
    }
    
    for query in queries:
        try:
            response = system.answer_with_full_proof(query)
            debug = response.get("debug", {})
            
            fallback_used = debug.get("fallback_used", False)
            if fallback_used:
                total_fallbacks += 1
            
            # Track component fallbacks
            components = debug.get("component_fallbacks", {})
            if components.get("quran"):
                component_fallback_counts["quran"] += 1
            if components.get("graph"):
                component_fallback_counts["graph"] += 1
            if components.get("taxonomy"):
                component_fallback_counts["taxonomy"] += 1
            
            tafsir_fallbacks = components.get("tafsir", {})
            for source, used in tafsir_fallbacks.items():
                if used and source in component_fallback_counts["tafsir"]:
                    component_fallback_counts["tafsir"][source] += 1
            
            results.append({
                "query": query,
                "fallback_used": fallback_used,
                "fallback_reasons": debug.get("fallback_reasons", []),
                "retrieval_distribution": debug.get("retrieval_distribution", {}),
                "validation_score": response.get("validation", {}).get("score", 0),
            })
            
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "fallback_used": True,  # Treat errors as fallback
            })
            total_fallbacks += 1
    
    return {
        "total_queries": len(queries),
        "total_fallbacks": total_fallbacks,
        "fallback_rate": total_fallbacks / len(queries) if queries else 0,
        "component_fallback_counts": component_fallback_counts,
        "results": results,
    }
