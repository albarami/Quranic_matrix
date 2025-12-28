"""
Tafsir Integration Module

This module handles:
1. Loading and annotating tafsir from all 5 sources
2. Extracting behavioral mentions from tafsir text
3. Connecting tafsir to the relationship graph
4. Creating semantic embeddings for RAG
5. Building cross-references between tafsir and Quranic annotations

All 5 sources: Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
import json
import re
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Behavioral keywords for extraction (Arabic)
BEHAVIOR_KEYWORDS = {
    # Positive behaviors
    "إيمان": "faith", "صبر": "patience", "شكر": "gratitude", "توبة": "repentance",
    "تقوى": "piety", "إحسان": "excellence", "صدق": "truthfulness", "أمانة": "trustworthiness",
    "عدل": "justice", "رحمة": "mercy", "تواضع": "humility", "زهد": "asceticism",
    "خشوع": "humility_prayer", "ذكر": "remembrance", "دعاء": "supplication",
    "توكل": "reliance", "رضا": "contentment", "حياء": "modesty",
    
    # Negative behaviors
    "كفر": "disbelief", "نفاق": "hypocrisy", "كبر": "arrogance", "حسد": "envy",
    "غيبة": "backbiting", "كذب": "lying", "ظلم": "oppression", "فسق": "transgression",
    "رياء": "showing_off", "غضب": "anger", "بخل": "stinginess", "غفلة": "heedlessness",
    "شرك": "polytheism", "فجور": "immorality", "خيانة": "betrayal",
    
    # Heart states
    "قلب": "heart", "سليم": "sound", "مريض": "sick", "قاسي": "hard",
    "مختوم": "sealed", "ميت": "dead", "منيب": "repentant",
    
    # Agents
    "مؤمن": "believer", "كافر": "disbeliever", "منافق": "hypocrite",
    "نبي": "prophet", "رسول": "messenger", "ملائكة": "angels",
}

# Relationship patterns in tafsir
RELATIONSHIP_PATTERNS = {
    "causes": [
        r"سبب\s+(\w+)",
        r"يؤدي\s+إلى\s+(\w+)",
        r"من\s+أسباب\s+(\w+)",
        r"لأن\s+(\w+)",
    ],
    "effects": [
        r"نتيجة\s+(\w+)",
        r"يترتب\s+عليه\s+(\w+)",
        r"فيكون\s+(\w+)",
        r"عاقبته\s+(\w+)",
    ],
    "opposites": [
        r"ضد\s+(\w+)",
        r"نقيض\s+(\w+)",
        r"عكس\s+(\w+)",
    ],
}


# =============================================================================
# TAFSIR ANNOTATOR
# =============================================================================

class TafsirAnnotator:
    """
    Annotates tafsir text with behavioral labels and extracts relationships.
    """
    
    def __init__(self):
        self.behavior_index = defaultdict(list)  # behavior -> [(source, surah, ayah, context)]
        self.relationship_graph = {
            "causes": defaultdict(set),
            "effects": defaultdict(set),
            "opposites": defaultdict(set),
            "co_mentions": defaultdict(set),
        }
        self.embeddings = {}  # verse_key -> embedding vector
        self.tafsir_data = {source: {} for source in TAFSIR_SOURCES}
    
    def load_all_tafsir(self) -> Dict[str, int]:
        """Load all 5 tafsir sources."""
        stats = {}
        
        for source in TAFSIR_SOURCES:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if filepath.exists():
                count = 0
                with open(filepath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ref = entry.get("reference", {})
                            surah = ref.get("surah")
                            ayah = ref.get("ayah")
                            if surah and ayah:
                                key = f"{surah}:{ayah}"
                                self.tafsir_data[source][key] = {
                                    "text": entry.get("text_ar", ""),
                                    "surah": surah,
                                    "ayah": ayah,
                                    "source": source,
                                }
                                count += 1
                stats[source] = count
                print(f"[TafsirAnnotator] Loaded {source}: {count} entries")
            else:
                stats[source] = 0
                print(f"[TafsirAnnotator] Missing: {source}")
        
        return stats
    
    def annotate_behaviors(self) -> Dict[str, Any]:
        """
        Extract behavioral mentions from all tafsir text.
        Returns statistics about extracted behaviors.
        """
        stats = {
            "total_mentions": 0,
            "by_behavior": defaultdict(int),
            "by_source": defaultdict(int),
        }
        
        for source in TAFSIR_SOURCES:
            for key, data in self.tafsir_data[source].items():
                text = data["text"]
                surah = data["surah"]
                ayah = data["ayah"]
                
                # Extract behaviors from text
                for ar_keyword, en_label in BEHAVIOR_KEYWORDS.items():
                    if ar_keyword in text:
                        # Find context around the keyword
                        idx = text.find(ar_keyword)
                        context_start = max(0, idx - 50)
                        context_end = min(len(text), idx + len(ar_keyword) + 50)
                        context = text[context_start:context_end]
                        
                        # Store in index
                        self.behavior_index[ar_keyword].append({
                            "source": source,
                            "surah": surah,
                            "ayah": ayah,
                            "verse_key": key,
                            "context": context,
                            "en_label": en_label,
                        })
                        
                        stats["total_mentions"] += 1
                        stats["by_behavior"][ar_keyword] += 1
                        stats["by_source"][source] += 1
        
        print(f"[TafsirAnnotator] Extracted {stats['total_mentions']} behavioral mentions")
        return dict(stats)
    
    def build_relationships(self) -> Dict[str, Any]:
        """
        Build relationship graph from tafsir text.
        Extracts causes, effects, opposites, and co-mentions.
        """
        stats = {
            "causes": 0,
            "effects": 0,
            "opposites": 0,
            "co_mentions": 0,
        }
        
        for source in TAFSIR_SOURCES:
            for key, data in self.tafsir_data[source].items():
                text = data["text"]
                
                # Find behaviors mentioned in this tafsir entry
                behaviors_in_text = []
                for ar_keyword in BEHAVIOR_KEYWORDS.keys():
                    if ar_keyword in text:
                        behaviors_in_text.append(ar_keyword)
                
                # Build co-mention relationships
                for i, b1 in enumerate(behaviors_in_text):
                    for b2 in behaviors_in_text[i+1:]:
                        self.relationship_graph["co_mentions"][b1].add(b2)
                        self.relationship_graph["co_mentions"][b2].add(b1)
                        stats["co_mentions"] += 1
                
                # Extract causal relationships using patterns
                for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            # Find which behavior this relates to
                            for behavior in behaviors_in_text:
                                if behavior != match:
                                    self.relationship_graph[rel_type][behavior].add(match)
                                    stats[rel_type] += 1
        
        print(f"[TafsirAnnotator] Built relationships: {stats}")
        return stats
    
    def create_embeddings(self) -> Dict[str, Any]:
        """
        Create semantic embeddings for tafsir text.
        Uses simple TF-IDF-like approach (can be upgraded to transformer embeddings).
        """
        # Build vocabulary
        vocab = defaultdict(int)
        doc_freq = defaultdict(int)
        
        all_docs = []
        for source in TAFSIR_SOURCES:
            for key, data in self.tafsir_data[source].items():
                text = data["text"]
                terms = set(text.split())
                all_docs.append((key, source, terms))
                for term in terms:
                    vocab[term] += 1
                    doc_freq[term] += 1
        
        # Create embeddings (simplified - term frequency vectors)
        num_docs = len(all_docs)
        
        for key, source, terms in all_docs:
            embedding = {}
            for term in terms:
                # TF-IDF-like score
                tf = 1  # Binary TF
                idf = num_docs / (doc_freq[term] + 1)
                embedding[term] = tf * idf
            
            embed_key = f"{source}:{key}"
            self.embeddings[embed_key] = embedding
        
        print(f"[TafsirAnnotator] Created {len(self.embeddings)} embeddings")
        return {
            "total_embeddings": len(self.embeddings),
            "vocab_size": len(vocab),
        }
    
    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search across all tafsir using embeddings.
        """
        query_terms = set(query.split())
        
        scores = []
        for embed_key, embedding in self.embeddings.items():
            score = sum(embedding.get(term, 0) for term in query_terms)
            if score > 0:
                source, verse_key = embed_key.split(":", 1)
                scores.append({
                    "source": source,
                    "verse_key": verse_key,
                    "score": score,
                    "text": self.tafsir_data[source].get(verse_key, {}).get("text", "")[:200],
                })
        
        # Sort by score
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]
    
    def get_behavior_tafsir(self, behavior: str) -> Dict[str, Any]:
        """
        Get all tafsir mentions of a specific behavior across all 5 sources.
        """
        mentions = self.behavior_index.get(behavior, [])
        
        # Group by source
        by_source = defaultdict(list)
        for mention in mentions:
            by_source[mention["source"]].append(mention)
        
        # Get relationships
        relationships = {
            "causes": list(self.relationship_graph["causes"].get(behavior, set())),
            "effects": list(self.relationship_graph["effects"].get(behavior, set())),
            "opposites": list(self.relationship_graph["opposites"].get(behavior, set())),
            "co_mentions": list(self.relationship_graph["co_mentions"].get(behavior, set()))[:20],
        }
        
        return {
            "behavior": behavior,
            "total_mentions": len(mentions),
            "by_source": {source: len(items) for source, items in by_source.items()},
            "sample_mentions": mentions[:10],
            "relationships": relationships,
        }
    
    def cross_reference_with_spans(self, spans: List[Dict]) -> Dict[str, Any]:
        """
        Cross-reference tafsir annotations with Quranic span annotations.
        Finds where tafsir and behavioral annotations align.
        """
        alignments = []
        
        for span in spans:
            ref = span.get("reference", {})
            surah = ref.get("surah")
            ayah = ref.get("ayah")
            if not surah or not ayah:
                continue
            
            verse_key = f"{surah}:{ayah}"
            span_text = span.get("text_ar", "")
            
            # Check each tafsir source for this verse
            tafsir_matches = {}
            for source in TAFSIR_SOURCES:
                if verse_key in self.tafsir_data[source]:
                    tafsir_text = self.tafsir_data[source][verse_key]["text"]
                    
                    # Check if span behavior is mentioned in tafsir
                    for keyword in BEHAVIOR_KEYWORDS.keys():
                        if keyword in span_text and keyword in tafsir_text:
                            if source not in tafsir_matches:
                                tafsir_matches[source] = []
                            tafsir_matches[source].append(keyword)
            
            if tafsir_matches:
                alignments.append({
                    "verse_key": verse_key,
                    "span_text": span_text[:100],
                    "tafsir_matches": tafsir_matches,
                })
        
        return {
            "total_alignments": len(alignments),
            "sample_alignments": alignments[:20],
        }
    
    def save_annotations(self, output_path: Path = None) -> str:
        """
        Save all annotations to a JSONL file.
        """
        if output_path is None:
            output_path = ANNOTATIONS_DIR / "tafsir_behavioral_annotations.jsonl"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        entries = []
        for behavior, mentions in self.behavior_index.items():
            for mention in mentions:
                entries.append({
                    "behavior_ar": behavior,
                    "behavior_en": mention["en_label"],
                    "source": mention["source"],
                    "surah": mention["surah"],
                    "ayah": mention["ayah"],
                    "context": mention["context"],
                })
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"[TafsirAnnotator] Saved {len(entries)} annotations to {output_path}")
        return str(output_path)
    
    def save_relationships(self, output_path: Path = None) -> str:
        """
        Save relationship graph to JSON file.
        """
        if output_path is None:
            output_path = ANNOTATIONS_DIR / "tafsir_relationships.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        graph_data = {}
        for rel_type, relations in self.relationship_graph.items():
            graph_data[rel_type] = {k: list(v) for k, v in relations.items()}
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"[TafsirAnnotator] Saved relationships to {output_path}")
        return str(output_path)


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

_annotator_instance = None

def get_tafsir_annotator() -> TafsirAnnotator:
    """Get or create the tafsir annotator instance."""
    global _annotator_instance
    if _annotator_instance is None:
        _annotator_instance = TafsirAnnotator()
        _annotator_instance.load_all_tafsir()
        _annotator_instance.annotate_behaviors()
        _annotator_instance.build_relationships()
        _annotator_instance.create_embeddings()
    return _annotator_instance


def integrate_tafsir_with_brain(brain, annotator: TafsirAnnotator = None):
    """
    Integrate tafsir annotations into the unified brain.
    """
    if annotator is None:
        annotator = get_tafsir_annotator()
    
    # Add behavior index to brain
    brain.tafsir_behaviors = annotator.behavior_index
    
    # Merge relationship graphs
    for rel_type in ["causes", "effects", "opposites"]:
        for behavior, related in annotator.relationship_graph[rel_type].items():
            if rel_type not in brain.graph:
                brain.graph[rel_type] = defaultdict(set)
            brain.graph[rel_type][behavior].update(related)
    
    # Add co-mentions to co_occurs
    for behavior, co_mentioned in annotator.relationship_graph["co_mentions"].items():
        brain.graph["co_occurs"][behavior].update(co_mentioned)
    
    # Add semantic search capability
    brain.tafsir_embeddings = annotator.embeddings
    brain.tafsir_search = annotator.search_semantic
    
    print("[Integration] Tafsir integrated with unified brain")
    return brain


def run_full_integration():
    """
    Run the full tafsir integration pipeline.
    """
    print("=" * 60)
    print("TAFSIR INTEGRATION PIPELINE")
    print("=" * 60)
    
    annotator = TafsirAnnotator()
    
    # Step 1: Load all tafsir
    print("\n[Step 1] Loading all 5 tafsir sources...")
    load_stats = annotator.load_all_tafsir()
    
    # Step 2: Annotate behaviors
    print("\n[Step 2] Extracting behavioral annotations...")
    behavior_stats = annotator.annotate_behaviors()
    
    # Step 3: Build relationships
    print("\n[Step 3] Building relationship graph...")
    rel_stats = annotator.build_relationships()
    
    # Step 4: Create embeddings
    print("\n[Step 4] Creating semantic embeddings...")
    embed_stats = annotator.create_embeddings()
    
    # Step 5: Save annotations
    print("\n[Step 5] Saving annotations...")
    annotations_path = annotator.save_annotations()
    relationships_path = annotator.save_relationships()
    
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"Tafsir sources loaded: {load_stats}")
    print(f"Behavioral mentions: {behavior_stats['total_mentions']}")
    print(f"Relationships built: {rel_stats}")
    print(f"Embeddings created: {embed_stats['total_embeddings']}")
    print(f"Annotations saved: {annotations_path}")
    print(f"Relationships saved: {relationships_path}")
    
    return annotator


if __name__ == "__main__":
    run_full_integration()
