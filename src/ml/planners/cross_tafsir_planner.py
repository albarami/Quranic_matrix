"""
Phase 3: CROSS_TAFSIR_COMPARISON Planner (3.2)

Thin wrapper around LegendaryPlanner for cross-tafsir comparison.
Computes agreement/disagreement metrics across 7 tafsir sources.

REUSES:
- LegendaryPlanner.get_concept_evidence()
- LegendaryPlanner.resolve_entities()
- CORE_SOURCES (7 canonical tafsir sources)

ADDS:
- Agreement/disagreement metric computation across 7 sources
- Verse enumeration from concept index for target concepts
- Source-by-source evidence comparison
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from src.ml.tafsir_constants import CANONICAL_TAFSIR_SOURCES

logger = logging.getLogger(__name__)


@dataclass
class SourceEvidence:
    """Evidence from a single tafsir source."""
    source: str
    chunk_count: int
    sample_text: str = ""
    verse_keys: List[str] = field(default_factory=list)


@dataclass
class CrossTafsirResult:
    """Result of cross-tafsir comparison."""
    entity_id: str
    entity_label_ar: str
    entity_label_en: str
    total_sources: int
    sources_with_evidence: int
    agreement_ratio: float
    source_evidence: List[SourceEvidence]
    common_verses: List[str]
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_label_ar": self.entity_label_ar,
            "entity_label_en": self.entity_label_en,
            "total_sources": self.total_sources,
            "sources_with_evidence": self.sources_with_evidence,
            "agreement_ratio": self.agreement_ratio,
            "consensus_percentage": round(self.agreement_ratio * 100, 1),
            "source_evidence": [
                {
                    "source": se.source,
                    "chunk_count": se.chunk_count,
                    "sample_text": se.sample_text[:200] if se.sample_text else "",
                    "verse_keys": se.verse_keys[:5],
                }
                for se in self.source_evidence
            ],
            "common_verses": self.common_verses[:10],
            "gaps": self.gaps,
        }


class CrossTafsirPlanner:
    """
    Planner for CROSS_TAFSIR_COMPARISON question class.

    Wraps LegendaryPlanner to provide:
    - Multi-source tafsir comparison
    - Agreement/disagreement metrics
    - Source-level evidence breakdown
    """

    def __init__(self, legendary_planner=None):
        self._planner = legendary_planner

    def _ensure_planner(self):
        if self._planner is None:
            from src.ml.legendary_planner import get_legendary_planner
            self._planner = get_legendary_planner()

    def compare_tafsir_for_entity(
        self,
        entity_id: str,
        entity_label_ar: str = "",
        entity_label_en: str = "",
    ) -> CrossTafsirResult:
        """
        Compare tafsir evidence across all 7 sources for an entity.

        Args:
            entity_id: The canonical entity ID
            entity_label_ar: Arabic label (optional)
            entity_label_en: English label (optional)

        Returns:
            CrossTafsirResult with source-by-source comparison
        """
        self._ensure_planner()

        gaps = []
        evidence = self._planner.get_concept_evidence(entity_id)

        if evidence.get("status") == "not_found":
            gaps.append(f"entity_not_in_index:{entity_id}")
            return CrossTafsirResult(
                entity_id=entity_id,
                entity_label_ar=entity_label_ar,
                entity_label_en=entity_label_en,
                total_sources=len(CANONICAL_TAFSIR_SOURCES),
                sources_with_evidence=0,
                agreement_ratio=0.0,
                source_evidence=[],
                common_verses=[],
                gaps=gaps,
            )

        # Organize evidence by source
        source_data = {src: {"chunks": [], "verses": set()} for src in CANONICAL_TAFSIR_SOURCES}

        for chunk in evidence.get("sample_evidence", []):
            source = chunk.get("source", "")
            if source in source_data:
                source_data[source]["chunks"].append(chunk)
                verse_key = chunk.get("verse_key", "")
                if verse_key:
                    source_data[source]["verses"].add(verse_key)

        # Build SourceEvidence objects
        source_evidence = []
        verse_sets = []

        for src in CANONICAL_TAFSIR_SOURCES:
            data = source_data[src]
            chunks = data["chunks"]
            verses = list(data["verses"])

            sample_text = chunks[0].get("text", "") if chunks else ""

            source_evidence.append(SourceEvidence(
                source=src,
                chunk_count=len(chunks),
                sample_text=sample_text,
                verse_keys=verses[:5],
            ))

            if verses:
                verse_sets.append(set(verses))

        # Compute common verses (intersection of all sources with evidence)
        common_verses = []
        if verse_sets:
            common = verse_sets[0]
            for vs in verse_sets[1:]:
                common = common.intersection(vs)
            common_verses = sorted(list(common))

        # Compute agreement ratio
        sources_with_evidence = sum(1 for se in source_evidence if se.chunk_count > 0)
        agreement_ratio = sources_with_evidence / len(CANONICAL_TAFSIR_SOURCES)

        # Identify gaps
        for src in CANONICAL_TAFSIR_SOURCES:
            if source_data[src]["chunks"] == []:
                gaps.append(f"missing_tafsir_{src}")

        return CrossTafsirResult(
            entity_id=entity_id,
            entity_label_ar=entity_label_ar,
            entity_label_en=entity_label_en,
            total_sources=len(CANONICAL_TAFSIR_SOURCES),
            sources_with_evidence=sources_with_evidence,
            agreement_ratio=agreement_ratio,
            source_evidence=source_evidence,
            common_verses=common_verses,
            gaps=gaps,
        )

    def compare_tafsir_for_query(self, query: str) -> List[CrossTafsirResult]:
        """
        Compare tafsir for all entities in a query.

        Args:
            query: The query text

        Returns:
            List of CrossTafsirResult for each resolved entity
        """
        self._ensure_planner()

        resolution = self._planner.resolve_entities(query)
        results = []

        for entity in resolution.get("entities", []):
            entity_id = entity.get("entity_id", "")
            result = self.compare_tafsir_for_entity(
                entity_id=entity_id,
                entity_label_ar=entity.get("term", ""),
            )
            results.append(result)

        return results

    def get_global_tafsir_coverage(self) -> Dict[str, Any]:
        """
        Get global tafsir coverage statistics (entity-free).

        Returns:
            Dictionary with coverage stats per source
        """
        self._ensure_planner()
        return self._planner.compute_global_tafsir_coverage()
