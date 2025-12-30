"""
Base classes for Capability Engines

All engines inherit from CapabilityEngine and return CapabilityResult.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class CapabilityResult:
    """Result from a capability engine execution."""
    
    success: bool
    capability: str
    data: Dict[str, Any] = field(default_factory=dict)
    verses: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "capability": self.capability,
            "data": self.data,
            "verses": self.verses,
            "provenance": self.provenance,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
        }
    
    def has_provenance(self) -> bool:
        """Check if result has proper provenance."""
        return len(self.provenance) > 0
    
    def has_verses(self) -> bool:
        """Check if result has verse evidence."""
        return len(self.verses) > 0
    
    def get_verse_keys(self) -> Set[str]:
        """Extract all verse keys from result."""
        keys = set()
        for v in self.verses:
            if "verse_key" in v:
                keys.add(v["verse_key"])
            elif "surah" in v and "ayah" in v:
                keys.add(f"{v['surah']}:{v['ayah']}")
        return keys


class CapabilityEngine(ABC):
    """
    Base class for all capability engines.
    
    Each engine:
    1. Reads from SSOT (Postgres tables, KB files, graph)
    2. Performs deterministic computation
    3. Returns structured result with provenance
    
    LLM is NEVER the source of truth - engines retrieve facts.
    """
    
    # Class-level configuration
    capability_id: str = "BASE"
    capability_name: str = "Base Engine"
    required_sources: List[str] = []
    
    def __init__(self):
        self.kb_dir = Path("data/kb")
        self.graph_dir = Path("data/graph")
        self.vocab_dir = Path("vocab")
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> CapabilityResult:
        """
        Execute the capability engine.
        
        Args:
            query: The query/question to process
            params: Optional parameters for the engine
            
        Returns:
            CapabilityResult with data, verses, and provenance
        """
        pass
    
    def validate_result(self, result: CapabilityResult) -> List[str]:
        """
        Validate result meets minimum requirements.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        if not result.has_provenance():
            errors.append("Result missing provenance")
        
        if not result.has_verses() and self.capability_id not in ["TAXONOMY", "MODEL_REGISTRY"]:
            errors.append("Result missing verse evidence")
        
        return errors
    
    def load_kb_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load a JSONL file from KB directory."""
        path = self.kb_dir / filename
        if not path.exists():
            return []
        
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def load_graph_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from graph directory."""
        path = self.graph_dir / filename
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_vocab_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from vocab directory."""
        path = self.vocab_dir / filename
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_behavior_dossier(self, behavior_id: str) -> Optional[Dict[str, Any]]:
        """Get a behavior dossier from KB."""
        if "dossiers" not in self._cache:
            dossiers = self.load_kb_file("behavior_dossiers.jsonl")
            self._cache["dossiers"] = {d["behavior_id"]: d for d in dossiers}
        
        return self._cache["dossiers"].get(behavior_id)
    
    def get_behavior_by_term(self, term_ar: str) -> Optional[Dict[str, Any]]:
        """Get a behavior dossier by Arabic term."""
        if "dossiers_by_term" not in self._cache:
            dossiers = self.load_kb_file("behavior_dossiers.jsonl")
            self._cache["dossiers_by_term"] = {d["term_ar"]: d for d in dossiers}
        
        return self._cache["dossiers_by_term"].get(term_ar)
