"""
Tafsir Annotation Extractor - Phase 5 Implementation.

Extracts from ALL 4 tafsir sources:
- Behaviors (using Arabic roots, not just string matching)
- Relationships (CAUSES, RESULTS_IN, OPPOSITE_OF)
- Inner states (قلب states, emotions)
- Speech acts (commands, prohibitions, promises, warnings)
- Agents and organs mentioned

This creates comprehensive annotations linked to ayat.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from ..tafsir import CrossTafsirAnalyzer
from ..graph import QBMKnowledgeGraph


@dataclass
class TafsirAnnotation:
    """Annotation extracted from tafsir."""
    surah: int
    ayah: int
    source: str  # ibn_kathir, tabari, qurtubi, saadi
    annotation_type: str  # behavior, relationship, inner_state, speech_act
    value: str  # The extracted value
    context: str  # Surrounding text
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


class TafsirAnnotationExtractor:
    """
    Extract annotations from all 4 tafsir sources.
    
    Uses:
    - Arabic root matching (not just exact strings)
    - Pattern-based relationship extraction
    - Keyword-based inner state detection
    - Speech act classification
    """

    # Relationship patterns in Arabic tafsir
    CAUSAL_PATTERNS = [
        (r"بسبب\s+(\S+)", "CAUSES"),  # because of
        (r"يؤدي\s+إلى\s+(\S+)", "RESULTS_IN"),  # leads to
        (r"نتيجة\s+(\S+)", "RESULTS_IN"),  # result of
        (r"يسبب\s+(\S+)", "CAUSES"),  # causes
        (r"عاقبة\s+(\S+)", "RESULTS_IN"),  # consequence of
        (r"ثمرة\s+(\S+)", "RESULTS_IN"),  # fruit of
        (r"أثر\s+(\S+)", "RESULTS_IN"),  # effect of
    ]

    # Opposite patterns
    OPPOSITE_PATTERNS = [
        (r"ضد\s+(\S+)", "OPPOSITE_OF"),  # opposite of
        (r"عكس\s+(\S+)", "OPPOSITE_OF"),  # reverse of
        (r"نقيض\s+(\S+)", "OPPOSITE_OF"),  # antithesis of
        (r"خلاف\s+(\S+)", "OPPOSITE_OF"),  # contrary to
    ]

    # Inner state keywords
    INNER_STATE_KEYWORDS = {
        "خوف": "fear",
        "رجاء": "hope",
        "محبة": "love",
        "بغض": "hatred",
        "حسد": "envy",
        "كبر": "arrogance",
        "تواضع": "humility",
        "رضا": "contentment",
        "سخط": "anger",
        "طمأنينة": "tranquility",
        "قلق": "anxiety",
        "يقين": "certainty",
        "شك": "doubt",
        "توكل": "trust",
        "إخلاص": "sincerity",
        "رياء": "showing_off",
        "نفاق": "hypocrisy",
        "صدق": "truthfulness",
    }

    # Speech act markers
    SPEECH_ACT_MARKERS = {
        "command": ["افعل", "عليك", "يجب", "فرض", "واجب", "أمر"],
        "prohibition": ["لا تفعل", "حرام", "نهى", "منع", "لا يجوز"],
        "promise": ["وعد", "جزاء", "ثواب", "جنة", "نعيم"],
        "warning": ["وعيد", "عذاب", "نار", "عقاب", "حذر"],
        "statement": ["أخبر", "قال", "ذكر", "بين"],
    }

    # Heart type keywords
    HEART_TYPES = {
        "قلب سليم": "sound_heart",
        "قلب مريض": "diseased_heart",
        "قلب قاسي": "hard_heart",
        "قلب ميت": "dead_heart",
        "قلب منيب": "repentant_heart",
        "قلب مطمئن": "tranquil_heart",
        "قلب خاشع": "humble_heart",
    }

    def __init__(
        self,
        tafsir_data_dir: str = "data/tafsir",
        vocab_dir: str = "vocab",
    ):
        """Initialize the extractor."""
        self.tafsir = CrossTafsirAnalyzer(data_dir=tafsir_data_dir)
        self.vocab_dir = Path(vocab_dir)
        
        # Load vocabularies
        self._behaviors: Dict[str, Dict] = {}
        self._behavior_roots: Dict[str, List[str]] = {}  # root → [behavior_ids]
        self._organs: Dict[str, Dict] = {}
        self._agents: Dict[str, Dict] = {}
        self._load_vocabularies()

        # Results storage
        self.annotations: List[TafsirAnnotation] = []
        self.behavior_index: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
        self.relationship_index: List[Dict] = []

    def _load_vocabularies(self):
        """Load all vocabulary files."""
        # Load behaviors with roots
        beh_path = self.vocab_dir / "behavior_concepts.json"
        if beh_path.exists():
            with open(beh_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for category, behaviors in data.get("categories", {}).items():
                for beh in behaviors:
                    beh_id = beh.get("id", "")
                    self._behaviors[beh_id] = {
                        "id": beh_id,
                        "category": category,
                        "name_ar": beh.get("ar", ""),
                        "name_en": beh.get("en", ""),
                        "quranic_roots": beh.get("quranic_roots", []),
                    }
                    # Index by root for morphological matching
                    for root in beh.get("quranic_roots", []):
                        if root not in self._behavior_roots:
                            self._behavior_roots[root] = []
                        self._behavior_roots[root].append(beh_id)

        # Load organs
        org_path = self.vocab_dir / "organs.json"
        if org_path.exists():
            with open(org_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for organ in data.get("items", []):
                organ_id = organ.get("id", "")
                self._organs[organ_id] = {
                    "id": organ_id,
                    "name_ar": organ.get("ar", ""),
                    "name_ar_plural": organ.get("ar_plural", ""),
                }

        # Load agents
        agent_path = self.vocab_dir / "agents.json"
        if agent_path.exists():
            with open(agent_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for agent in data.get("items", []):
                agent_id = agent.get("id", "")
                self._agents[agent_id] = {
                    "id": agent_id,
                    "name_ar": agent.get("ar", ""),
                    "quranic_terms": agent.get("quranic_terms", []),
                }

    def extract_all(
        self,
        sources: Optional[List[str]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Extract annotations from all tafsir sources.

        Args:
            sources: List of sources to process (default: all 4).
            progress_callback: Optional callback(source, surah, ayah, count).

        Returns:
            Statistics about extraction.
        """
        sources = sources or ["ibn_kathir", "tabari", "qurtubi", "saadi"]
        
        stats = {
            "sources_processed": 0,
            "ayat_processed": 0,
            "behaviors_found": 0,
            "relationships_found": 0,
            "inner_states_found": 0,
            "speech_acts_found": 0,
            "by_source": {},
        }

        for source in sources:
            source_stats = {
                "behaviors": 0,
                "relationships": 0,
                "inner_states": 0,
                "speech_acts": 0,
            }

            # Load source into memory
            self.tafsir._load_source(source)
            source_index = self.tafsir._index.get(source, {})

            for (surah, ayah), data in source_index.items():
                text = data.get("text", "")
                if not text:
                    continue

                # Extract behaviors (using roots)
                behaviors = self._extract_behaviors_with_roots(text, surah, ayah, source)
                source_stats["behaviors"] += len(behaviors)

                # Extract relationships
                relationships = self._extract_relationships(text, surah, ayah, source)
                source_stats["relationships"] += len(relationships)

                # Extract inner states
                inner_states = self._extract_inner_states(text, surah, ayah, source)
                source_stats["inner_states"] += len(inner_states)

                # Extract speech acts
                speech_acts = self._extract_speech_acts(text, surah, ayah, source)
                source_stats["speech_acts"] += len(speech_acts)

                stats["ayat_processed"] += 1

                if progress_callback:
                    progress_callback(source, surah, ayah, stats["ayat_processed"])

            stats["by_source"][source] = source_stats
            stats["behaviors_found"] += source_stats["behaviors"]
            stats["relationships_found"] += source_stats["relationships"]
            stats["inner_states_found"] += source_stats["inner_states"]
            stats["speech_acts_found"] += source_stats["speech_acts"]
            stats["sources_processed"] += 1

        return stats

    def _extract_behaviors_with_roots(
        self, text: str, surah: int, ayah: int, source: str
    ) -> List[TafsirAnnotation]:
        """Extract behaviors using Arabic root matching."""
        found = []
        seen_behaviors = set()

        # Method 1: Direct Arabic name match
        for beh_id, beh_data in self._behaviors.items():
            name_ar = beh_data.get("name_ar", "")
            if name_ar and name_ar in text:
                if beh_id not in seen_behaviors:
                    seen_behaviors.add(beh_id)
                    idx = text.find(name_ar)
                    context = text[max(0, idx-50):min(len(text), idx+len(name_ar)+50)]
                    
                    ann = TafsirAnnotation(
                        surah=surah,
                        ayah=ayah,
                        source=source,
                        annotation_type="behavior",
                        value=beh_id,
                        context=context,
                        confidence=0.9,  # High confidence for exact match
                        metadata={"match_type": "exact", "name_ar": name_ar},
                    )
                    found.append(ann)
                    self.annotations.append(ann)
                    self.behavior_index[beh_id].append((source, surah, ayah))

        # Method 2: Root-based matching (finds inflected forms)
        for root, beh_ids in self._behavior_roots.items():
            # Convert root format "ك-ب-ر" to pattern
            root_letters = root.replace("-", "")
            if len(root_letters) >= 3:
                # Create pattern that matches words containing these root letters
                # This is a simplified approach - proper Arabic morphology would be better
                pattern = self._create_root_pattern(root_letters)
                matches = re.findall(pattern, text)
                
                if matches:
                    for beh_id in beh_ids:
                        if beh_id not in seen_behaviors:
                            seen_behaviors.add(beh_id)
                            # Find first match context
                            match = matches[0]
                            idx = text.find(match)
                            context = text[max(0, idx-50):min(len(text), idx+len(match)+50)]
                            
                            ann = TafsirAnnotation(
                                surah=surah,
                                ayah=ayah,
                                source=source,
                                annotation_type="behavior",
                                value=beh_id,
                                context=context,
                                confidence=0.7,  # Lower confidence for root match
                                metadata={"match_type": "root", "root": root, "matched_word": match},
                            )
                            found.append(ann)
                            self.annotations.append(ann)
                            self.behavior_index[beh_id].append((source, surah, ayah))

        return found

    def _create_root_pattern(self, root_letters: str) -> str:
        """Create regex pattern for Arabic root matching."""
        if len(root_letters) < 3:
            return ""
        # Match words that contain the root letters in order
        # Allow Arabic letters between root letters
        arabic_char = r"[\u0600-\u06FF]"
        pattern = arabic_char + "*"
        for letter in root_letters:
            pattern += re.escape(letter) + arabic_char + "*"
        return r"\b" + pattern + r"\b"

    def _extract_relationships(
        self, text: str, surah: int, ayah: int, source: str
    ) -> List[TafsirAnnotation]:
        """Extract causal and opposite relationships."""
        found = []

        # Check causal patterns
        for pattern, rel_type in self.CAUSAL_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                ann = TafsirAnnotation(
                    surah=surah,
                    ayah=ayah,
                    source=source,
                    annotation_type="relationship",
                    value=rel_type,
                    context=match,
                    confidence=0.6,
                    metadata={"pattern": pattern, "target": match},
                )
                found.append(ann)
                self.annotations.append(ann)
                self.relationship_index.append({
                    "source": source,
                    "surah": surah,
                    "ayah": ayah,
                    "rel_type": rel_type,
                    "target": match,
                })

        # Check opposite patterns
        for pattern, rel_type in self.OPPOSITE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                ann = TafsirAnnotation(
                    surah=surah,
                    ayah=ayah,
                    source=source,
                    annotation_type="relationship",
                    value=rel_type,
                    context=match,
                    confidence=0.6,
                    metadata={"pattern": pattern, "target": match},
                )
                found.append(ann)
                self.annotations.append(ann)

        return found

    def _extract_inner_states(
        self, text: str, surah: int, ayah: int, source: str
    ) -> List[TafsirAnnotation]:
        """Extract inner states (emotions, heart conditions)."""
        found = []

        # Check inner state keywords
        for ar_term, en_term in self.INNER_STATE_KEYWORDS.items():
            if ar_term in text:
                idx = text.find(ar_term)
                context = text[max(0, idx-50):min(len(text), idx+len(ar_term)+50)]
                
                ann = TafsirAnnotation(
                    surah=surah,
                    ayah=ayah,
                    source=source,
                    annotation_type="inner_state",
                    value=en_term,
                    context=context,
                    confidence=0.8,
                    metadata={"arabic": ar_term},
                )
                found.append(ann)
                self.annotations.append(ann)

        # Check heart types
        for ar_term, en_term in self.HEART_TYPES.items():
            if ar_term in text:
                idx = text.find(ar_term)
                context = text[max(0, idx-50):min(len(text), idx+len(ar_term)+50)]
                
                ann = TafsirAnnotation(
                    surah=surah,
                    ayah=ayah,
                    source=source,
                    annotation_type="heart_type",
                    value=en_term,
                    context=context,
                    confidence=0.9,
                    metadata={"arabic": ar_term},
                )
                found.append(ann)
                self.annotations.append(ann)

        return found

    def _extract_speech_acts(
        self, text: str, surah: int, ayah: int, source: str
    ) -> List[TafsirAnnotation]:
        """Extract speech acts (commands, prohibitions, etc.)."""
        found = []

        for act_type, markers in self.SPEECH_ACT_MARKERS.items():
            for marker in markers:
                if marker in text:
                    idx = text.find(marker)
                    context = text[max(0, idx-50):min(len(text), idx+len(marker)+50)]
                    
                    ann = TafsirAnnotation(
                        surah=surah,
                        ayah=ayah,
                        source=source,
                        annotation_type="speech_act",
                        value=act_type,
                        context=context,
                        confidence=0.7,
                        metadata={"marker": marker},
                    )
                    found.append(ann)
                    self.annotations.append(ann)
                    break  # Only one annotation per act type per ayah

        return found

    def get_behavior_summary(self) -> Dict[str, Any]:
        """Get summary of behaviors found across all tafsirs."""
        summary = {
            "total_behaviors": len(self.behavior_index),
            "by_behavior": {},
            "by_source": defaultdict(int),
            "consensus": [],  # Behaviors found in 3+ sources
        }

        for beh_id, occurrences in self.behavior_index.items():
            sources = set(occ[0] for occ in occurrences)
            summary["by_behavior"][beh_id] = {
                "total_mentions": len(occurrences),
                "sources": list(sources),
                "source_count": len(sources),
            }
            
            for source, _, _ in occurrences:
                summary["by_source"][source] += 1

            if len(sources) >= 3:
                summary["consensus"].append({
                    "behavior_id": beh_id,
                    "name_ar": self._behaviors.get(beh_id, {}).get("name_ar", ""),
                    "sources": list(sources),
                })

        summary["by_source"] = dict(summary["by_source"])
        return summary

    def save_annotations(self, output_path: str):
        """Save all annotations to JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in self.annotations:
                record = {
                    "surah": ann.surah,
                    "ayah": ann.ayah,
                    "source": ann.source,
                    "type": ann.annotation_type,
                    "value": ann.value,
                    "context": ann.context,
                    "confidence": ann.confidence,
                    "metadata": ann.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_to_graph(self, graph: QBMKnowledgeGraph) -> int:
        """Add extracted relationships to knowledge graph."""
        edges_added = 0
        
        for rel in self.relationship_index:
            # Try to match target to a behavior
            target = rel.get("target", "")
            rel_type = rel.get("rel_type", "RELATED")
            
            # Find behavior that matches the target
            for beh_id, beh_data in self._behaviors.items():
                name_ar = beh_data.get("name_ar", "")
                if name_ar and target in name_ar:
                    # Add edge from ayah to behavior
                    ayah_key = f"{rel['surah']}:{rel['ayah']}"
                    try:
                        graph.add_relationship(ayah_key, beh_id, rel_type)
                        edges_added += 1
                    except Exception:
                        pass
                    break

        return edges_added
