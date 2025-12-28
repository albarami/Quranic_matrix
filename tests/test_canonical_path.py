"""
Tests to enforce the canonical data path.

These tests ensure that:
1. The production API path does not use deprecated stores (Chroma)
2. Quran indexing includes ALL verses (including muqattaʿāt)
3. Canonical entity counts are correct
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestNoChromaInProdPath:
    """Ensure Chroma RAG is not used in the canonical production path."""
    
    def test_proof_router_does_not_import_chroma(self):
        """The proof router should not import chromadb."""
        from api.routers import proof
        
        # Check that chromadb is not in the module's imports
        proof_source = proof.__file__
        with open(proof_source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Should not have chromadb imports
        assert 'import chromadb' not in source_code, \
            "proof.py should not import chromadb - use deterministic retrieval instead"
        assert 'from chromadb' not in source_code, \
            "proof.py should not import from chromadb"
    
    def test_genome_router_does_not_import_chroma(self):
        """The genome router should not import chromadb."""
        from api.routers import genome
        
        genome_source = genome.__file__
        with open(genome_source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        assert 'import chromadb' not in source_code
        assert 'from chromadb' not in source_code
    
    def test_reviews_router_does_not_import_chroma(self):
        """The reviews router should not import chromadb."""
        from api.routers import reviews
        
        reviews_source = reviews.__file__
        with open(reviews_source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        assert 'import chromadb' not in source_code
        assert 'from chromadb' not in source_code


class TestQuranIndexCompleteness:
    """Ensure ALL Quran verses are indexed, including muqattaʿāt."""
    
    TOTAL_QURAN_VERSES = 6236  # Standard count
    
    # Muqattaʿāt verses (opening letters) that must be included
    MUQATTAAT_VERSES = [
        ("2", "1"),   # الم - Al-Baqarah
        ("3", "1"),   # الم - Aal-Imran
        ("7", "1"),   # المص - Al-A'raf
        ("10", "1"),  # الر - Yunus
        ("11", "1"),  # الر - Hud
        ("12", "1"),  # الر - Yusuf
        ("13", "1"),  # المر - Ar-Ra'd
        ("14", "1"),  # الر - Ibrahim
        ("15", "1"),  # الر - Al-Hijr
        ("19", "1"),  # كهيعص - Maryam
        ("20", "1"),  # طه - Ta-Ha
        ("26", "1"),  # طسم - Ash-Shu'ara
        ("27", "1"),  # طس - An-Naml
        ("28", "1"),  # طسم - Al-Qasas
        ("29", "1"),  # الم - Al-Ankabut
        ("30", "1"),  # الم - Ar-Rum
        ("31", "1"),  # الم - Luqman
        ("32", "1"),  # الم - As-Sajdah
        ("36", "1"),  # يس - Ya-Sin
        ("38", "1"),  # ص - Sad
        ("40", "1"),  # حم - Ghafir
        ("41", "1"),  # حم - Fussilat
        ("42", "1"),  # حم - Ash-Shura
        ("43", "1"),  # حم - Az-Zukhruf
        ("44", "1"),  # حم - Ad-Dukhan
        ("45", "1"),  # حم - Al-Jathiyah
        ("46", "1"),  # حم - Al-Ahqaf
        ("50", "1"),  # ق - Qaf
        ("68", "1"),  # ن - Al-Qalam
    ]
    
    def test_full_power_includes_short_verses(self):
        """FullPower indexer should not exclude short verses."""
        # Read the source file directly to verify the fix
        full_power_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'ml', 'full_power_system.py'
        )
        with open(full_power_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Should NOT have the old len > 5 filter for verses
        assert 'len(verse_text) > 5' not in source, \
            "FullPower should not exclude short verses (muqattaʿāt)"
        
        # Should have the fix comment
        assert 'muqatta' in source.lower(), \
            "FullPower should document that muqattaʿāt are included"


class TestCanonicalEntityCounts:
    """Ensure canonical entity counts match the registry."""
    
    def test_canonical_entities_file_exists(self):
        """canonical_entities.json must exist."""
        import json
        canonical_path = os.path.join(
            os.path.dirname(__file__), '..', 'vocab', 'canonical_entities.json'
        )
        assert os.path.exists(canonical_path), \
            "vocab/canonical_entities.json must exist"
        
        with open(canonical_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify structure
        assert 'behaviors' in data, "Must have behaviors"
        assert 'agents' in data, "Must have agents"
    
    def test_behavior_count_is_73(self):
        """Must have exactly 73 canonical behaviors."""
        import json
        canonical_path = os.path.join(
            os.path.dirname(__file__), '..', 'vocab', 'canonical_entities.json'
        )
        with open(canonical_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        behaviors = data.get('behaviors', [])
        assert len(behaviors) == 73, \
            f"Expected 73 behaviors, got {len(behaviors)}"
    
    def test_agent_count_is_14(self):
        """Must have exactly 14 canonical agents."""
        import json
        canonical_path = os.path.join(
            os.path.dirname(__file__), '..', 'vocab', 'canonical_entities.json'
        )
        with open(canonical_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        agents = data.get('agents', [])
        assert len(agents) == 14, \
            f"Expected 14 agents, got {len(agents)}"


class TestTafsirSourceCount:
    """Ensure all 7 tafsir sources are available."""
    
    CANONICAL_TAFSIR_SOURCES = [
        'ibn_kathir',
        'tabari', 
        'qurtubi',
        'saadi',
        'jalalayn',
        'baghawi',
        'muyassar',
    ]
    
    def test_tafsir_files_exist(self):
        """All 7 tafsir JSONL files must exist."""
        tafsir_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'tafsir'
        )
        
        for source in self.CANONICAL_TAFSIR_SOURCES:
            filepath = os.path.join(tafsir_dir, f'{source}.ar.jsonl')
            assert os.path.exists(filepath), \
                f"Missing tafsir file: {source}.ar.jsonl"
    
    def test_proof_router_uses_7_sources(self):
        """Proof router must reference all 7 tafsir sources."""
        from api.routers import proof
        
        proof_source = proof.__file__
        with open(proof_source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        for source in self.CANONICAL_TAFSIR_SOURCES:
            assert source in source_code, \
                f"Proof router missing tafsir source: {source}"


class TestDeprecatedStoresNotInProdPath:
    """Ensure deprecated stores are not used in production endpoints."""
    
    DEPRECATED_PATTERNS = [
        'qbm_silver_20251221',  # Legacy export
        'qbm_ayat',             # Chroma collection
        'qbm_tafsir',           # Chroma collection
    ]
    
    def test_proof_router_no_deprecated_refs(self):
        """Proof router should not reference deprecated stores."""
        from api.routers import proof
        
        proof_source = proof.__file__
        with open(proof_source, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        for pattern in self.DEPRECATED_PATTERNS:
            assert pattern not in source_code, \
                f"Proof router references deprecated store: {pattern}"


class TestProofRouterIsCanonical:
    """Ensure /api/proof/query is served by the canonical router, not legacy inline handlers."""
    
    def test_proof_query_served_by_modular_router(self):
        """
        /api/proof/query must be served by src/api/routers/proof.py, not main.py.
        This prevents the routing conflict bug where legacy inline endpoints shadow modular routers.
        """
        import inspect
        from src.api.main import app
        
        # Find all routes matching /api/proof/query
        hits = [r for r in app.routes if getattr(r, 'path', None) == '/api/proof/query']
        
        assert len(hits) == 1, f"Expected exactly 1 route for /api/proof/query, found {len(hits)}"
        
        endpoint_file = inspect.getsourcefile(hits[0].endpoint)
        assert 'routers/proof.py' in endpoint_file or 'routers\\proof.py' in endpoint_file, \
            f"/api/proof/query must be served by routers/proof.py, not {endpoint_file}"
    
    def test_legacy_endpoints_not_shadowing(self):
        """Legacy endpoints should be at different paths (e.g., /api/proof/query-legacy)."""
        from src.api.main import app
        
        # Check that legacy path exists but doesn't conflict
        legacy_hits = [r for r in app.routes if getattr(r, 'path', None) == '/api/proof/query-legacy']
        canonical_hits = [r for r in app.routes if getattr(r, 'path', None) == '/api/proof/query']
        
        # Canonical must exist
        assert len(canonical_hits) == 1, "Canonical /api/proof/query route missing"
        
        # If legacy exists, it must be at a different path
        if legacy_hits:
            assert len(legacy_hits) == 1, "Multiple legacy routes found"
    
    def test_no_duplicate_route_registrations(self):
        """
        Ensure each canonical proof route is registered exactly once.
        This prevents route shadowing bugs where multiple handlers compete.
        """
        from src.api.main import app
        
        # Critical proof routes that must be unique
        critical_routes = [
            "/api/proof/query",
            "/api/proof/status",
        ]
        
        for route_path in critical_routes:
            hits = [r for r in app.routes if getattr(r, 'path', None) == route_path]
            assert len(hits) == 1, \
                f"Route {route_path} registered {len(hits)} times (expected exactly 1). " \
                f"This causes route shadowing bugs."
    
    def test_proof_routes_served_by_canonical_module(self):
        """All /api/proof/* routes must be served by routers/proof.py."""
        import inspect
        from src.api.main import app
        
        proof_routes = [r for r in app.routes if getattr(r, 'path', '').startswith('/api/proof/')]
        
        for route in proof_routes:
            # Skip legacy routes (they're allowed to be elsewhere)
            if '-legacy' in route.path:
                continue
            
            endpoint_file = inspect.getsourcefile(route.endpoint)
            assert 'routers/proof.py' in endpoint_file or 'routers\\proof.py' in endpoint_file, \
                f"Route {route.path} must be in routers/proof.py, found in {endpoint_file}"
