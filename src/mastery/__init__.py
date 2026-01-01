"""
Behavior Mastery Module for QBM.

Phase 2 of Behavior Mastery Plan:
- BehaviorDossier dataclass with full schema
- Dossier assembler merging all sources
- Context assertions with citations
- 87/87 dossier generation
"""

from .dossier import (
    BehaviorDossier,
    VerseEvidence,
    TafsirChunk,
    RelationshipEdge,
    CausalPath,
    BouzidaniContexts,
    ContextAssertion,
    EvidenceStats,
)
from .assembler import (
    DossierAssembler,
    build_all_dossiers,
    load_dossier,
    get_dossier,
)

__all__ = [
    "BehaviorDossier",
    "VerseEvidence",
    "TafsirChunk",
    "RelationshipEdge",
    "CausalPath",
    "BouzidaniContexts",
    "ContextAssertion",
    "EvidenceStats",
    "DossierAssembler",
    "build_all_dossiers",
    "load_dossier",
    "get_dossier",
]
