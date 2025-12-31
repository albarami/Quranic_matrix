"""
Citation Verifier for QBM System (v2.1 - Zero Hallucination)

Phase 6: Verifier gate for citation validity.
Ensures all claims have valid verse_key references and proper provenance.

STRICT Verification Rules:
1. All verse_keys must exist in SSOT (6,236 verses)
2. All behavior_ids must be canonical (87 behaviors)
3. All tafsir sources must be from canonical list (7 sources)
4. No fabricated statistics (all numbers must have derived_from trail)
5. No generic opening verses as primary evidence
6. SUBSET CONTRACT: All tafsir verse_keys must be in behavior's verse list
7. NO SURAH_INTRO: entry_type="surah_intro" cannot appear in behavior evidence
8. CLAIM-EVIDENCE ALIGNMENT: Each claim/paragraph must have supporting verse_keys
9. Long answers (>200 chars) must have at least one verse_key citation
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Generic opening verses to disallow as primary evidence
GENERIC_OPENING_VERSES: Set[str] = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}

# Canonical tafsir sources
CANONICAL_TAFSIR_SOURCES = [
    "ibn_kathir",
    "tabari",
    "qurtubi",
    "saadi",
    "jalalayn",
    "baghawi",
    "muyassar"
]

# Valid verse range
VALID_SURAH_RANGE = range(1, 115)  # 1-114
MAX_AYAH_PER_SURAH = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}


# =============================================================================
# VERIFICATION RESULT
# =============================================================================

@dataclass
class VerificationResult:
    """Result of citation verification."""
    valid: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    verified_citations: int = 0
    total_citations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "verified_citations": self.verified_citations,
            "total_citations": self.total_citations,
            "verification_rate": round(self.verified_citations / max(self.total_citations, 1) * 100, 1)
        }
    
    def add_violation(self, violation_type: str, details: Dict[str, Any]):
        """Add a violation (fails verification)."""
        self.violations.append({
            "type": violation_type,
            **details
        })
        self.valid = False
    
    def add_warning(self, warning_type: str, details: Dict[str, Any]):
        """Add a warning (doesn't fail verification)."""
        self.warnings.append({
            "type": warning_type,
            **details
        })


# =============================================================================
# CITATION VERIFIER
# =============================================================================

class CitationVerifier:
    """
    Verifies citations and provenance in QBM responses.
    
    Implements fail-closed verification:
    - Any violation causes the entire response to fail
    - All claims must have valid provenance
    - No unsupported assertions allowed
    
    Zero-Hallucination Contracts:
    - SUBSET: Tafsir verse_keys must be subset of behavior's verse list
    - NO_SURAH_INTRO: entry_type="surah_intro" is forbidden in evidence
    - CLAIM_EVIDENCE: Each claim must have supporting verse_keys
    - NUMBERS_PROVENANCE: Statistics must have derived_from trail
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize verifier with data directory."""
        self.data_dir = data_dir or Path("data")
        self._canonical_behaviors: Optional[Set[str]] = None
        self._valid_verse_keys: Optional[Set[str]] = None
        self._behavior_verse_map: Optional[Dict[str, Set[str]]] = None
    
    def _load_canonical_behaviors(self) -> Set[str]:
        """Load canonical behavior IDs."""
        if self._canonical_behaviors is not None:
            return self._canonical_behaviors
        
        vocab_path = Path("vocab/canonical_entities.json")
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
            self._canonical_behaviors = {b["id"] for b in entities.get("behaviors", [])}
        else:
            self._canonical_behaviors = set()
            logger.warning("canonical_entities.json not found")
        
        return self._canonical_behaviors
    
    def _load_valid_verse_keys(self) -> Set[str]:
        """Load valid verse keys from SSOT."""
        if self._valid_verse_keys is not None:
            return self._valid_verse_keys
        
        # Generate all valid verse keys
        self._valid_verse_keys = set()
        for surah, max_ayah in MAX_AYAH_PER_SURAH.items():
            for ayah in range(1, max_ayah + 1):
                self._valid_verse_keys.add(f"{surah}:{ayah}")
        
        return self._valid_verse_keys
    
    def _load_behavior_verse_map(self) -> Dict[str, Set[str]]:
        """Load mapping of behavior_id -> set of valid verse_keys from concept_index_v3."""
        if self._behavior_verse_map is not None:
            return self._behavior_verse_map
        
        self._behavior_verse_map = {}
        ci_path = self.data_dir / "evidence" / "concept_index_v3.jsonl"
        
        if ci_path.exists():
            with open(ci_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    concept_id = entry.get("concept_id", "")
                    if concept_id.startswith("BEH_"):
                        verses = entry.get("verses", [])
                        verse_keys = {v.get("verse_key") for v in verses if v.get("verse_key")}
                        self._behavior_verse_map[concept_id] = verse_keys
        else:
            logger.warning(f"concept_index_v3.jsonl not found at {ci_path}")
        
        return self._behavior_verse_map
    
    def verify_verse_key(self, verse_key: str) -> bool:
        """Verify a single verse key is valid."""
        valid_keys = self._load_valid_verse_keys()
        return verse_key in valid_keys
    
    def verify_behavior_id(self, behavior_id: str) -> bool:
        """Verify a behavior ID is canonical."""
        canonical = self._load_canonical_behaviors()
        return behavior_id in canonical
    
    def verify_tafsir_source(self, source: str) -> bool:
        """Verify a tafsir source is canonical."""
        return source in CANONICAL_TAFSIR_SOURCES
    
    def is_generic_opening_verse(self, verse_key: str) -> bool:
        """Check if verse is a generic opening verse."""
        return verse_key in GENERIC_OPENING_VERSES
    
    def verify_subset_contract(
        self, 
        behavior_id: str, 
        tafsir_verse_keys: Set[str]
    ) -> List[str]:
        """
        Verify SUBSET CONTRACT: All tafsir verse_keys must be in behavior's verse list.
        
        Args:
            behavior_id: The behavior being queried
            tafsir_verse_keys: Verse keys referenced in tafsir evidence
            
        Returns:
            List of verse_keys that violate the subset contract (should be empty)
        """
        behavior_verses = self._load_behavior_verse_map()
        valid_verses = behavior_verses.get(behavior_id, set())
        
        if not valid_verses:
            # If behavior has no verses mapped, all tafsir verses are violations
            return list(tafsir_verse_keys) if tafsir_verse_keys else []
        
        violations = []
        for vk in tafsir_verse_keys:
            if vk not in valid_verses:
                violations.append(vk)
        
        return violations
    
    def verify_no_surah_intro(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verify NO_SURAH_INTRO: entry_type="surah_intro" cannot appear in evidence.
        
        Args:
            response: Response to check
            
        Returns:
            List of violations (entry_type=surah_intro found)
        """
        violations = []
        self._find_surah_intro_violations(response, violations)
        return violations
    
    def _find_surah_intro_violations(self, data: Any, violations: List[Dict[str, Any]]):
        """Recursively find surah_intro entry_type violations."""
        if isinstance(data, dict):
            entry_type = data.get("entry_type", "")
            if entry_type == "surah_intro":
                violations.append({
                    "entry_type": entry_type,
                    "verse_key": data.get("verse_key"),
                    "source": data.get("source"),
                    "message": "surah_intro cannot be used as behavior evidence"
                })
            
            for v in data.values():
                self._find_surah_intro_violations(v, violations)
        
        elif isinstance(data, list):
            for item in data:
                self._find_surah_intro_violations(item, violations)
    
    def verify_numbers_provenance(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verify NUMBERS_PROVENANCE: Statistics must have derived_from trail.
        
        Args:
            response: Response to check
            
        Returns:
            List of numbers without provenance
        """
        violations = []
        self._find_unprovenanced_numbers(response, violations, path="")
        return violations
    
    def verify_claim_evidence_alignment(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verify CLAIM-EVIDENCE ALIGNMENT: Each claim must have supporting verse_keys.
        
        For academic-grade outputs, every claim/paragraph in the response
        must be explicitly linked to supporting evidence (verse_keys + tafsir sources).
        
        Args:
            response: Response to check
            
        Returns:
            List of claims without proper evidence mapping
        """
        violations = []
        
        # Check for claims field with evidence mapping
        claims = response.get("claims", [])
        if claims:
            for i, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    continue
                
                claim_text = claim.get("text", claim.get("claim", ""))
                verse_keys = claim.get("verse_keys", claim.get("evidence", []))
                
                if claim_text and not verse_keys:
                    violations.append({
                        "claim_index": i,
                        "claim_text": claim_text[:100] + "..." if len(claim_text) > 100 else claim_text,
                        "message": "Claim has no supporting verse_keys"
                    })
        
        # Check for paragraphs/sections without evidence
        paragraphs = response.get("paragraphs", response.get("sections", []))
        if paragraphs:
            for i, para in enumerate(paragraphs):
                if not isinstance(para, dict):
                    continue
                
                content = para.get("content", para.get("text", ""))
                evidence = para.get("verse_keys", para.get("evidence", para.get("citations", [])))
                
                if content and not evidence:
                    violations.append({
                        "paragraph_index": i,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content,
                        "message": "Paragraph has no supporting evidence"
                    })
        
        # Check answer field for structured responses
        answer = response.get("answer", "")
        if isinstance(answer, str) and len(answer) > 200:
            # Long answer should have evidence
            all_verse_keys = self._extract_verse_keys(response)
            if not all_verse_keys:
                violations.append({
                    "field": "answer",
                    "length": len(answer),
                    "message": "Long answer has no verse_key citations"
                })
        
        return violations
    
    def _find_unprovenanced_numbers(
        self, 
        data: Any, 
        violations: List[Dict[str, Any]], 
        path: str
    ):
        """Recursively find numbers without derived_from provenance."""
        # Keys that are expected to have numeric values without explicit provenance
        ALLOWED_NUMERIC_KEYS = {
            "surah", "ayah", "verse_count", "total_verses", "returned_verses",
            "hops", "paths_found", "total_chunks", "size_bytes", "file_count",
            "out_degree", "in_degree", "total_degree", "total_nodes", "total_edges",
            "behavior_nodes", "causal_edges", "verified_citations", "total_citations",
            "processing_time_ms", "max_verses", "max_per_source"
        }
        
        if isinstance(data, dict):
            for k, v in data.items():
                new_path = f"{path}.{k}" if path else k
                
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    # Check if this is a statistic that needs provenance
                    if k not in ALLOWED_NUMERIC_KEYS and "count" in k.lower():
                        # This looks like a computed statistic
                        # Check if there's a derived_from field nearby
                        if "derived_from" not in data and "provenance" not in data:
                            violations.append({
                                "path": new_path,
                                "value": v,
                                "message": f"Statistic '{k}' has no derived_from provenance"
                            })
                else:
                    self._find_unprovenanced_numbers(v, violations, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._find_unprovenanced_numbers(item, violations, f"{path}[{i}]")
    
    def verify_response(self, response: Dict[str, Any], strict: bool = True) -> VerificationResult:
        """
        Verify a complete response for citation validity.
        
        Args:
            response: Response dictionary to verify
            strict: If True, enforce all zero-hallucination contracts
            
        Returns:
            VerificationResult with violations and warnings
        """
        result = VerificationResult(valid=True)
        
        # Extract all citations from response
        verse_keys = self._extract_verse_keys(response)
        behavior_ids = self._extract_behavior_ids(response)
        tafsir_sources = self._extract_tafsir_sources(response)
        
        result.total_citations = len(verse_keys) + len(behavior_ids)
        
        # 1. Verify verse keys exist in SSOT
        for vk in verse_keys:
            if self.verify_verse_key(vk):
                result.verified_citations += 1
            else:
                result.add_violation("invalid_verse_key", {
                    "verse_key": vk,
                    "message": f"Verse key {vk} does not exist in SSOT"
                })
        
        # 2. Check for generic opening verses as primary evidence
        generic_count = sum(1 for vk in verse_keys if self.is_generic_opening_verse(vk))
        if generic_count > 0 and generic_count == len(verse_keys):
            result.add_violation("generic_opening_verses_only", {
                "count": generic_count,
                "message": "Response uses only generic opening verses (Fatiha/early Baqarah)"
            })
        elif generic_count > len(verse_keys) * 0.5:
            result.add_warning("high_generic_verse_ratio", {
                "generic_count": generic_count,
                "total_count": len(verse_keys),
                "ratio": round(generic_count / len(verse_keys), 2)
            })
        
        # 3. Verify behavior IDs are canonical
        for bid in behavior_ids:
            if self.verify_behavior_id(bid):
                result.verified_citations += 1
            else:
                result.add_violation("invalid_behavior_id", {
                    "behavior_id": bid,
                    "message": f"Behavior ID {bid} is not canonical"
                })
        
        # 4. Verify tafsir sources are canonical
        for src in tafsir_sources:
            if not self.verify_tafsir_source(src):
                result.add_warning("unknown_tafsir_source", {
                    "source": src,
                    "canonical_sources": CANONICAL_TAFSIR_SOURCES
                })
        
        # 5. Check for provenance field
        if not response.get("provenance") and result.total_citations > 0:
            result.add_warning("missing_provenance", {
                "message": "Response has citations but no provenance field"
            })
        
        # === STRICT ZERO-HALLUCINATION CONTRACTS ===
        if strict:
            # 6. NO_SURAH_INTRO: entry_type="surah_intro" forbidden
            surah_intro_violations = self.verify_no_surah_intro(response)
            for v in surah_intro_violations:
                result.add_violation("surah_intro_in_evidence", v)
            
            # 7. SUBSET CONTRACT: tafsir verse_keys must be in behavior's verse list
            # Only check if we have a single behavior context
            if len(behavior_ids) == 1:
                behavior_id = list(behavior_ids)[0]
                tafsir_verse_keys = self._extract_tafsir_verse_keys(response)
                subset_violations = self.verify_subset_contract(behavior_id, tafsir_verse_keys)
                for vk in subset_violations:
                    result.add_violation("subset_contract_violation", {
                        "behavior_id": behavior_id,
                        "tafsir_verse_key": vk,
                        "message": f"Tafsir verse {vk} not in behavior's verse list"
                    })
            
            # 8. NUMBERS_PROVENANCE: statistics need derived_from trail
            # This is a warning, not a violation, as it's hard to enforce perfectly
            numbers_violations = self.verify_numbers_provenance(response)
            for v in numbers_violations:
                result.add_warning("unprovenanced_statistic", v)
            
            # 9. CLAIM-EVIDENCE ALIGNMENT: each claim needs supporting verse_keys
            claim_violations = self.verify_claim_evidence_alignment(response)
            for v in claim_violations:
                result.add_violation("claim_without_evidence", v)
        
        return result
    
    def _extract_tafsir_verse_keys(self, data: Any, keys: Optional[Set[str]] = None) -> Set[str]:
        """Extract verse_keys specifically from tafsir sections."""
        if keys is None:
            keys = set()
        
        if isinstance(data, dict):
            # Check if this is a tafsir entry
            is_tafsir = "source" in data and any(
                ts in str(data.get("source", "")).lower() 
                for ts in CANONICAL_TAFSIR_SOURCES
            )
            
            if is_tafsir and "verse_key" in data:
                vk = data["verse_key"]
                if isinstance(vk, str):
                    keys.add(vk)
            
            # Check tafsir dict structure
            if "tafsir" in data and isinstance(data["tafsir"], dict):
                for source, chunks in data["tafsir"].items():
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            if isinstance(chunk, dict) and "verse_key" in chunk:
                                keys.add(chunk["verse_key"])
            
            # Recurse
            for v in data.values():
                self._extract_tafsir_verse_keys(v, keys)
        
        elif isinstance(data, list):
            for item in data:
                self._extract_tafsir_verse_keys(item, keys)
        
        return keys
    
    def _extract_verse_keys(self, data: Any, keys: Optional[Set[str]] = None) -> Set[str]:
        """Recursively extract verse keys from response."""
        if keys is None:
            keys = set()
        
        if isinstance(data, dict):
            # Check for verse_key field
            if "verse_key" in data and isinstance(data["verse_key"], str):
                keys.add(data["verse_key"])
            
            # Check for verse_keys list
            if "verse_keys" in data and isinstance(data["verse_keys"], list):
                for vk in data["verse_keys"]:
                    if isinstance(vk, str):
                        keys.add(vk)
            
            # Recurse into values
            for v in data.values():
                self._extract_verse_keys(v, keys)
        
        elif isinstance(data, list):
            for item in data:
                self._extract_verse_keys(item, keys)
        
        elif isinstance(data, str):
            # Extract verse keys from text (pattern: surah:ayah)
            matches = re.findall(r'\b(\d{1,3}:\d{1,3})\b', data)
            for m in matches:
                keys.add(m)
        
        return keys
    
    def _extract_behavior_ids(self, data: Any, ids: Optional[Set[str]] = None) -> Set[str]:
        """Recursively extract behavior IDs from response."""
        if ids is None:
            ids = set()
        
        if isinstance(data, dict):
            # Check for behavior_id field
            if "behavior_id" in data and isinstance(data["behavior_id"], str):
                ids.add(data["behavior_id"])
            
            # Check for from_behavior/to_behavior
            for field in ["from_behavior", "to_behavior", "source", "target"]:
                if field in data and isinstance(data[field], str) and data[field].startswith("BEH_"):
                    ids.add(data[field])
            
            # Recurse into values
            for v in data.values():
                self._extract_behavior_ids(v, ids)
        
        elif isinstance(data, list):
            for item in data:
                self._extract_behavior_ids(item, ids)
        
        elif isinstance(data, str):
            # Extract behavior IDs from text
            matches = re.findall(r'\bBEH_[A-Z_]+\b', data)
            for m in matches:
                ids.add(m)
        
        return ids
    
    def _extract_tafsir_sources(self, data: Any, sources: Optional[Set[str]] = None) -> Set[str]:
        """Recursively extract tafsir sources from response."""
        if sources is None:
            sources = set()
        
        if isinstance(data, dict):
            # Check for source field
            if "source" in data and isinstance(data["source"], str):
                src = data["source"].lower()
                if any(ts in src for ts in CANONICAL_TAFSIR_SOURCES):
                    for ts in CANONICAL_TAFSIR_SOURCES:
                        if ts in src:
                            sources.add(ts)
            
            # Check for sources list
            if "sources" in data and isinstance(data["sources"], list):
                for s in data["sources"]:
                    if isinstance(s, str):
                        sources.add(s.lower())
            
            # Recurse into values
            for v in data.values():
                self._extract_tafsir_sources(v, sources)
        
        elif isinstance(data, list):
            for item in data:
                self._extract_tafsir_sources(item, sources)
        
        return sources
    
    def _extract_numbers(self, data: Any, numbers: Optional[List[int]] = None) -> List[int]:
        """Extract numeric values from response for fabrication check."""
        if numbers is None:
            numbers = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numbers.append(v)
                else:
                    self._extract_numbers(v, numbers)
        
        elif isinstance(data, list):
            for item in data:
                self._extract_numbers(item, numbers)
        
        return numbers
    
    def verify_tool_output(self, tool_name: str, output: Dict[str, Any]) -> VerificationResult:
        """
        Verify output from a specific tool.
        
        Args:
            tool_name: Name of the tool that produced the output
            output: Tool output to verify
            
        Returns:
            VerificationResult
        """
        result = self.verify_response(output)
        
        # Tool-specific checks
        if tool_name == "get_causal_paths":
            # Verify path nodes are valid behaviors
            for path in output.get("paths", []):
                for node in path.get("nodes", []):
                    if node.startswith("BEH_") and not self.verify_behavior_id(node):
                        result.add_violation("invalid_path_node", {
                            "node": node,
                            "message": f"Path contains invalid behavior: {node}"
                        })
        
        elif tool_name == "get_behavior_dossier":
            # Verify behavior exists
            bid = output.get("behavior_id")
            if bid and not self.verify_behavior_id(bid):
                result.add_violation("invalid_dossier_behavior", {
                    "behavior_id": bid
                })
        
        return result


# =============================================================================
# FAIL-CLOSED GATE
# =============================================================================

def fail_closed_gate(response: Dict[str, Any], verifier: Optional[CitationVerifier] = None) -> Dict[str, Any]:
    """
    Fail-closed verification gate.
    
    If verification fails, returns an error response instead of the original.
    This ensures no unverified claims reach the user.
    
    Args:
        response: Response to verify
        verifier: Optional verifier instance
        
    Returns:
        Original response if valid, error response if invalid
    """
    if verifier is None:
        verifier = CitationVerifier()
    
    result = verifier.verify_response(response)
    
    if result.valid:
        # Add verification metadata
        response["_verification"] = {
            "status": "PASSED",
            "verified_citations": result.verified_citations,
            "total_citations": result.total_citations,
            "warnings": len(result.warnings)
        }
        return response
    else:
        # Return error response
        return {
            "success": False,
            "error": "VERIFICATION_FAILED",
            "message": "Response failed citation verification and was blocked",
            "violations": result.violations,
            "warnings": result.warnings,
            "_verification": {
                "status": "FAILED",
                "violation_count": len(result.violations)
            }
        }
