"""
Unified QBM System - Integrates all AI components.

This module provides a unified interface that links:
- Quran ayat ↔ All 4 tafsirs (Ibn Kathir, Tabari, Qurtubi, Saadi)
- Behaviors ↔ Tafsir mentions ↔ Ayat references
- Knowledge Graph ↔ Vector Store ↔ RAG Pipeline
- Concepts (heart, hand, etc.) ↔ All related data

Query Types:
1. Ayah Query: Get all tafsirs + behaviors + relationships for an ayah
2. Concept Query: Find all mentions of a concept across tafsirs + ayat
3. Behavior Query: Get behavior details + related behaviors + tafsir evidence
4. Consensus Query: Find what all tafsirs agree on for an ayah
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..graph import QBMKnowledgeGraph
from ..vectors import QBMVectorStore
from ..tafsir import CrossTafsirAnalyzer


class QBMUnifiedSystem:
    """
    Unified system integrating all QBM AI components.
    
    Provides smart, interconnected queries across:
    - 4 Tafsir sources (24,944 records)
    - Knowledge Graph (behaviors, relationships, organs, agents)
    - Vector Store (semantic search)
    - Behavior Taxonomy (73+ behaviors across categories)
    """

    def __init__(
        self,
        graph_db_path: str = "data/qbm_graph.db",
        vector_persist_dir: str = "data/chromadb",
        tafsir_data_dir: str = "data/tafsir",
        vocab_dir: str = "vocab",
    ):
        """
        Initialize the unified system.

        Args:
            graph_db_path: Path to SQLite graph database.
            vector_persist_dir: Path to ChromaDB persistence directory.
            tafsir_data_dir: Path to tafsir JSONL files.
            vocab_dir: Path to vocabulary JSON files.
        """
        self.graph = QBMKnowledgeGraph(db_path=graph_db_path)
        self.graph.load()  # Load graph from SQLite DB
        self.vectors = QBMVectorStore(persist_dir=vector_persist_dir)
        self.tafsir = CrossTafsirAnalyzer(data_dir=tafsir_data_dir)
        self.vocab_dir = Path(vocab_dir)

        # Load behavior taxonomy
        self._behaviors: Dict[str, Dict] = {}
        self._behavior_roots: Dict[str, List[str]] = {}  # root → behavior_ids
        self._load_behavior_taxonomy()

        # Load organs and agents
        self._organs: Dict[str, Dict] = {}
        self._agents: Dict[str, Dict] = {}
        self._load_organs()
        self._load_agents()

        # Tafsir index for concept search
        self._concept_index: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
        # {concept: {source: [(surah, ayah), ...]}}

    # =========================================================================
    # Initialization
    # =========================================================================

    def _load_behavior_taxonomy(self):
        """Load behavior concepts from vocabulary."""
        vocab_path = self.vocab_dir / "behavior_concepts.json"
        if not vocab_path.exists():
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for category, behaviors in data.get("categories", {}).items():
            for beh in behaviors:
                beh_id = beh.get("id", "")
                self._behaviors[beh_id] = {
                    "id": beh_id,
                    "category": category,
                    "name_en": beh.get("en", ""),
                    "name_ar": beh.get("ar", ""),
                    "quranic_roots": beh.get("quranic_roots", []),
                }
                # Index by root
                for root in beh.get("quranic_roots", []):
                    if root not in self._behavior_roots:
                        self._behavior_roots[root] = []
                    self._behavior_roots[root].append(beh_id)

    def _load_organs(self):
        """Load organs from vocabulary."""
        vocab_path = self.vocab_dir / "organs.json"
        if not vocab_path.exists():
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for organ in data.get("items", []):
            organ_id = organ.get("id", "")
            self._organs[organ_id] = {
                "id": organ_id,
                "name_ar": organ.get("ar", ""),
                "name_ar_plural": organ.get("ar_plural", ""),
                "semantic_domains": organ.get("semantic_domains", []),
            }

    def _load_agents(self):
        """Load agents from vocabulary."""
        vocab_path = self.vocab_dir / "agents.json"
        if not vocab_path.exists():
            return

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for agent in data.get("items", []):
            agent_id = agent.get("id", "")
            self._agents[agent_id] = {
                "id": agent_id,
                "name_ar": agent.get("ar", ""),
                "name_en": agent.get("en", ""),
                "category": agent.get("category", ""),
            }

    # =========================================================================
    # Ayah Query: Get everything for an ayah
    # =========================================================================

    def query_ayah(
        self,
        surah: int,
        ayah: int,
        include_tafsir: bool = True,
        include_behaviors: bool = True,
        include_related: bool = True,
    ) -> Dict[str, Any]:
        """
        Get all information for a specific ayah.

        Args:
            surah: Surah number (1-114).
            ayah: Ayah number.
            include_tafsir: Include all 4 tafsir texts.
            include_behaviors: Include extracted behaviors.
            include_related: Include related ayat and concepts.

        Returns:
            Comprehensive data for the ayah.
        """
        result = {
            "reference": {"surah": surah, "ayah": ayah},
            "ayah_key": f"{surah}:{ayah}",
        }

        # Get ayah from graph if exists
        ayah_node = self.graph.get_node(f"{surah}:{ayah}")
        if ayah_node:
            result["text_uthmani"] = ayah_node.get("text_uthmani", "")
            result["text_simple"] = ayah_node.get("text_simple", "")

        # Get all tafsirs
        if include_tafsir:
            result["tafsir"] = {}
            for source in self.tafsir.get_available_sources():
                tafsir_data = self.tafsir.get_tafsir(surah, ayah, source)
                if tafsir_data:
                    result["tafsir"][source] = {
                        "source_name": self.tafsir.TAFSIR_SOURCES[source]["name_ar"],
                        "text": tafsir_data.get("text", ""),
                        "text_length": len(tafsir_data.get("text", "")),
                    }

        # Get behaviors mentioned in this ayah
        if include_behaviors:
            result["behaviors"] = self._extract_behaviors_from_ayah(surah, ayah)

        # Get organs mentioned in this ayah
        result["organs"] = self._extract_organs_from_ayah(surah, ayah)

        # Get agents mentioned in this ayah
        result["agents"] = self._extract_agents_from_ayah(surah, ayah)

        # Get related ayat (same behaviors, similar content)
        if include_related:
            result["related"] = self._find_related_ayat(surah, ayah)

        return result

    def _extract_behaviors_from_ayah(
        self, surah: int, ayah: int
    ) -> List[Dict[str, Any]]:
        """Extract behaviors mentioned in an ayah based on tafsir analysis."""
        behaviors = []
        seen_ids = set()

        # Get all tafsirs for this ayah
        all_tafsir = self.tafsir.get_all_tafsir(surah, ayah)

        # Search for behavior Arabic terms in tafsir texts
        for beh_id, beh_data in self._behaviors.items():
            name_ar = beh_data.get("name_ar", "")
            if not name_ar:
                continue

            mentions = {}
            for source, tafsir_data in all_tafsir.items():
                text = tafsir_data.get("text", "")
                if name_ar in text:
                    # Extract context around mention
                    idx = text.find(name_ar)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(name_ar) + 50)
                    mentions[source] = text[start:end]

            if mentions and beh_id not in seen_ids:
                seen_ids.add(beh_id)
                behaviors.append({
                    "behavior_id": beh_id,
                    "name_ar": name_ar,
                    "name_en": beh_data.get("name_en", ""),
                    "category": beh_data.get("category", ""),
                    "tafsir_mentions": mentions,
                    "mention_count": len(mentions),
                })

        # Sort by mention count (more tafsirs agree = higher confidence)
        behaviors.sort(key=lambda x: x["mention_count"], reverse=True)
        return behaviors

    def _extract_organs_from_ayah(
        self, surah: int, ayah: int
    ) -> List[Dict[str, Any]]:
        """Extract organs mentioned in an ayah based on tafsir analysis."""
        organs = []
        seen_ids = set()

        all_tafsir = self.tafsir.get_all_tafsir(surah, ayah)

        for organ_id, organ_data in self._organs.items():
            name_ar = organ_data.get("name_ar", "")
            name_ar_plural = organ_data.get("name_ar_plural", "")
            if not name_ar:
                continue

            mentions = {}
            for source, tafsir_data in all_tafsir.items():
                text = tafsir_data.get("text", "")
                # Check both singular and plural forms
                if name_ar in text or (name_ar_plural and name_ar_plural in text):
                    search_term = name_ar if name_ar in text else name_ar_plural
                    idx = text.find(search_term)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(search_term) + 50)
                    mentions[source] = text[start:end]

            if mentions and organ_id not in seen_ids:
                seen_ids.add(organ_id)
                organs.append({
                    "organ_id": organ_id,
                    "name_ar": name_ar,
                    "name_ar_plural": name_ar_plural,
                    "semantic_domains": organ_data.get("semantic_domains", []),
                    "tafsir_mentions": mentions,
                    "mention_count": len(mentions),
                })

        organs.sort(key=lambda x: x["mention_count"], reverse=True)
        return organs

    def _extract_agents_from_ayah(
        self, surah: int, ayah: int
    ) -> List[Dict[str, Any]]:
        """Extract agents mentioned in an ayah based on tafsir analysis."""
        agents = []
        seen_ids = set()

        all_tafsir = self.tafsir.get_all_tafsir(surah, ayah)

        for agent_id, agent_data in self._agents.items():
            name_ar = agent_data.get("name_ar", "")
            if not name_ar:
                continue

            mentions = {}
            for source, tafsir_data in all_tafsir.items():
                text = tafsir_data.get("text", "")
                if name_ar in text:
                    idx = text.find(name_ar)
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(name_ar) + 50)
                    mentions[source] = text[start:end]

            if mentions and agent_id not in seen_ids:
                seen_ids.add(agent_id)
                agents.append({
                    "agent_id": agent_id,
                    "name_ar": name_ar,
                    "name_en": agent_data.get("name_en", ""),
                    "category": agent_data.get("category", ""),
                    "tafsir_mentions": mentions,
                    "mention_count": len(mentions),
                })

        agents.sort(key=lambda x: x["mention_count"], reverse=True)
        return agents

    def _find_related_ayat(
        self, surah: int, ayah: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find ayat related to this one."""
        related = []

        # Use vector similarity if available
        try:
            ayah_key = f"{surah}:{ayah}"
            similar = self.vectors.search_similar_ayat(ayah_key, limit=limit)
            for item in similar:
                if item.get("ayah_key") != ayah_key:
                    related.append({
                        "ayah_key": item.get("ayah_key"),
                        "similarity": item.get("score", 0),
                        "reason": "semantic_similarity",
                    })
        except Exception:
            pass

        return related

    # =========================================================================
    # Concept Query: Find all mentions of a concept
    # =========================================================================

    def query_concept(
        self,
        concept: str,
        sources: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Find all mentions of a concept across all tafsirs.

        Args:
            concept: Arabic or English concept (e.g., "القلب", "heart", "الكبر").
            sources: List of tafsir sources to search (default: all).
            limit: Maximum results per source.

        Returns:
            All mentions with ayah references and context.
        """
        sources = sources or self.tafsir.get_available_sources()

        result = {
            "concept": concept,
            "sources_searched": sources,
            "mentions": {},
            "ayat": set(),
            "related_behaviors": [],
            "statistics": {},
        }

        # Search each tafsir source
        for source in sources:
            mentions = self.tafsir.search_behavior_in_tafsir(
                concept, source, limit=limit
            )
            result["mentions"][source] = mentions
            result["statistics"][source] = len(mentions)

            # Collect unique ayat
            for m in mentions:
                result["ayat"].add((m.get("surah"), m.get("ayah")))

        # Convert ayat set to sorted list
        result["ayat"] = sorted(list(result["ayat"]))
        result["total_mentions"] = sum(result["statistics"].values())
        result["unique_ayat"] = len(result["ayat"])

        # Find related behaviors
        for beh_id, beh_data in self._behaviors.items():
            if concept in beh_data.get("name_ar", "") or concept.lower() in beh_data.get("name_en", "").lower():
                result["related_behaviors"].append({
                    "behavior_id": beh_id,
                    "name_ar": beh_data.get("name_ar"),
                    "name_en": beh_data.get("name_en"),
                    "category": beh_data.get("category"),
                })

        return result

    # =========================================================================
    # Behavior Query: Get behavior details with tafsir evidence
    # =========================================================================

    def query_behavior(
        self,
        behavior_id: str,
        include_tafsir_evidence: bool = True,
        include_relationships: bool = True,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a behavior.

        Args:
            behavior_id: Behavior ID (e.g., "BEH_EMO_ARROGANCE").
            include_tafsir_evidence: Search all tafsirs for mentions.
            include_relationships: Include graph relationships.
            limit: Max tafsir mentions per source.

        Returns:
            Behavior details with evidence from all sources.
        """
        if behavior_id not in self._behaviors:
            return {"error": f"Unknown behavior: {behavior_id}"}

        beh_data = self._behaviors[behavior_id]
        result = {
            "behavior_id": behavior_id,
            "name_ar": beh_data.get("name_ar"),
            "name_en": beh_data.get("name_en"),
            "category": beh_data.get("category"),
            "quranic_roots": beh_data.get("quranic_roots", []),
        }

        # Get tafsir evidence
        if include_tafsir_evidence:
            name_ar = beh_data.get("name_ar", "")
            if name_ar:
                concept_result = self.query_concept(name_ar, limit=limit)
                result["tafsir_evidence"] = {
                    "total_mentions": concept_result["total_mentions"],
                    "unique_ayat": concept_result["unique_ayat"],
                    "by_source": concept_result["statistics"],
                    "sample_ayat": concept_result["ayat"][:10],
                }

        # Get graph relationships
        if include_relationships:
            result["relationships"] = {
                "causes": [],
                "caused_by": [],
                "results_in": [],
                "opposite_of": [],
                "related_to": [],
            }

            # Get from graph using get_relationships (correct method)
            node = self.graph.get_node(behavior_id)
            if node:
                # Get outgoing edges
                for source, target, edge_data in self.graph.get_relationships(behavior_id, direction="out"):
                    edge_type = edge_data.get("edge_type", "RELATED")
                    if edge_type == "CAUSES":
                        result["relationships"]["causes"].append(target)
                    elif edge_type == "RESULTS_IN":
                        result["relationships"]["results_in"].append(target)
                    elif edge_type == "OPPOSITE_OF":
                        result["relationships"]["opposite_of"].append(target)
                    else:
                        result["relationships"]["related_to"].append(target)

                # Get incoming edges for caused_by
                for source, target, edge_data in self.graph.get_relationships(behavior_id, direction="in"):
                    edge_type = edge_data.get("edge_type", "RELATED")
                    if edge_type == "CAUSES":
                        result["relationships"]["caused_by"].append(source)

        return result

    # =========================================================================
    # Cross-Tafsir Consensus
    # =========================================================================

    def find_tafsir_consensus(
        self,
        surah: int,
        ayah: int,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find what all tafsirs agree/disagree on for an ayah.

        Args:
            surah: Surah number.
            ayah: Ayah number.
            topic: Optional topic to focus on.

        Returns:
            Consensus analysis across all 4 tafsirs.
        """
        all_tafsir = self.tafsir.get_all_tafsir(surah, ayah)

        result = {
            "reference": {"surah": surah, "ayah": ayah},
            "sources": list(all_tafsir.keys()),
            "behaviors_mentioned": defaultdict(list),
            "consensus": [],
            "unique_insights": {},
        }

        # Find behaviors mentioned in each tafsir
        for source, tafsir_data in all_tafsir.items():
            text = tafsir_data.get("text", "")
            for beh_id, beh_data in self._behaviors.items():
                name_ar = beh_data.get("name_ar", "")
                if name_ar and name_ar in text:
                    result["behaviors_mentioned"][beh_id].append(source)

        # Find consensus (behaviors mentioned by 3+ tafsirs)
        for beh_id, sources in result["behaviors_mentioned"].items():
            if len(sources) >= 3:
                result["consensus"].append({
                    "behavior_id": beh_id,
                    "name_ar": self._behaviors[beh_id].get("name_ar"),
                    "name_en": self._behaviors[beh_id].get("name_en"),
                    "agreed_by": sources,
                    "agreement_level": len(sources) / len(all_tafsir) if all_tafsir else 0,
                })

        # Sort by agreement level
        result["consensus"].sort(
            key=lambda x: x["agreement_level"], reverse=True
        )

        # Convert defaultdict to regular dict
        result["behaviors_mentioned"] = dict(result["behaviors_mentioned"])

        return result

    # =========================================================================
    # Build Comprehensive Index
    # =========================================================================

    def build_tafsir_behavior_index(
        self,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Build index of all behaviors mentioned in all tafsirs.

        This creates a comprehensive mapping:
        - behavior → [(source, surah, ayah), ...]
        - (surah, ayah) → [behaviors...]

        Returns:
            Statistics about the index.
        """
        behavior_index = defaultdict(list)  # beh_id → [(source, surah, ayah)]
        ayah_index = defaultdict(list)  # (surah, ayah) → [beh_ids]

        sources = self.tafsir.get_available_sources()
        total_ayat = 6236

        for source in sources:
            # Load source
            self.tafsir._load_source(source)

            for (surah, ayah), data in self.tafsir._index.get(source, {}).items():
                text = data.get("text", "")

                for beh_id, beh_data in self._behaviors.items():
                    name_ar = beh_data.get("name_ar", "")
                    if name_ar and name_ar in text:
                        behavior_index[beh_id].append((source, surah, ayah))
                        ayah_index[(surah, ayah)].append(beh_id)

                if progress_callback:
                    progress_callback(source, surah, ayah)

        # Store indices
        self._behavior_tafsir_index = dict(behavior_index)
        self._ayah_behavior_index = dict(ayah_index)

        return {
            "behaviors_indexed": len(behavior_index),
            "ayat_with_behaviors": len(ayah_index),
            "total_mentions": sum(len(v) for v in behavior_index.values()),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the unified system."""
        stats = {
            "tafsir": self.tafsir.get_statistics(),
            "behaviors": {
                "total": len(self._behaviors),
                "by_category": defaultdict(int),
            },
            "organs": {
                "total": len(self._organs),
            },
            "agents": {
                "total": len(self._agents),
            },
            "graph": {
                "nodes": self.graph.G.number_of_nodes() if hasattr(self.graph, 'G') else 0,
                "edges": self.graph.G.number_of_edges() if hasattr(self.graph, 'G') else 0,
            },
        }

        # Count behaviors by category
        for beh_data in self._behaviors.values():
            cat = beh_data.get("category", "unknown")
            stats["behaviors"]["by_category"][cat] += 1

        stats["behaviors"]["by_category"] = dict(stats["behaviors"]["by_category"])

        return stats
