"""
Layer 6: Cross-Tafsir Semantic Alignment

Semantic alignment across 5 tafsir sources.
Finds agreement, disagreement, and unique insights.

Current (BAD):  if keyword in tafsir1 and keyword in tafsir2 → "agrees"
New (GOOD):     semantic_similarity(tafsir1, tafsir2) → agreement score + analysis
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"
MODELS_DIR = DATA_DIR / "models"

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

TAFSIR_NAMES_AR = {
    "ibn_kathir": "ابن كثير",
    "tabari": "الطبري",
    "qurtubi": "القرطبي",
    "saadi": "السعدي",
    "jalalayn": "الجلالين",
}


# =============================================================================
# TAFSIR ALIGNER
# =============================================================================

class TafsirAligner:
    """
    Semantic alignment across 5 tafsir sources.
    
    Capabilities:
    1. Find where all 5 scholars agree
    2. Find where they disagree
    3. Extract unique insights per scholar
    4. Build behavioral consensus
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = None
        self.tafsir_data = {source: {} for source in TAFSIR_SOURCES}
        self.embeddings_cache = {}
        
        if ST_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
        
        self._load_tafsir()
    
    def _load_tafsir(self):
        """Load all 5 tafsir sources."""
        for source in TAFSIR_SOURCES:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if filepath.exists():
                with open(filepath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ref = entry.get("reference", {})
                            surah = ref.get("surah")
                            ayah = ref.get("ayah")
                            if surah and ayah:
                                key = f"{surah}:{ayah}"
                                self.tafsir_data[source][key] = entry.get("text_ar", "")
                logger.info(f"Loaded {source}: {len(self.tafsir_data[source])} entries")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, with caching."""
        if not self.model or not text:
            return None
        
        # Check cache
        text_hash = hash(text[:500])
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # Compute embedding
        embedding = self.model.encode(text[:512])  # Truncate long texts
        self.embeddings_cache[text_hash] = embedding
        
        return embedding
    
    def align_verse(self, surah: int, ayah: int) -> Dict[str, Any]:
        """
        Align interpretations for a specific verse across all 5 sources.
        
        Returns:
            {
                "verse": "2:255",
                "sources": {...},
                "agreement_score": 0.85,
                "agreement_analysis": {...},
                "disagreement_points": [...],
                "unique_insights": {...}
            }
        """
        verse_key = f"{surah}:{ayah}"
        
        result = {
            "verse": verse_key,
            "sources": {},
            "agreement_score": 0.0,
            "agreement_analysis": {},
            "disagreement_points": [],
            "unique_insights": {},
        }
        
        # Collect tafsir texts and embeddings
        texts = {}
        embeddings = {}
        
        for source in TAFSIR_SOURCES:
            if verse_key in self.tafsir_data[source]:
                text = self.tafsir_data[source][verse_key]
                texts[source] = text
                result["sources"][source] = {
                    "text": text[:500],
                    "length": len(text),
                    "available": True,
                }
                
                emb = self._get_embedding(text)
                if emb is not None:
                    embeddings[source] = emb
            else:
                result["sources"][source] = {"available": False}
        
        if len(embeddings) < 2:
            return result
        
        # Compute pairwise similarities
        sources_list = list(embeddings.keys())
        n = len(sources_list)
        
        if NUMPY_AVAILABLE and SKLEARN_AVAILABLE:
            emb_matrix = np.array([embeddings[s] for s in sources_list])
            sim_matrix = cosine_similarity(emb_matrix)
            
            # Overall agreement score (average pairwise similarity)
            total_sim = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_sim += sim_matrix[i][j]
                    count += 1
            
            result["agreement_score"] = float(total_sim / count) if count > 0 else 0
            
            # Find disagreement points (low similarity pairs)
            for i in range(n):
                for j in range(i+1, n):
                    if sim_matrix[i][j] < 0.7:
                        result["disagreement_points"].append({
                            "source1": sources_list[i],
                            "source2": sources_list[j],
                            "similarity": float(sim_matrix[i][j]),
                        })
            
            # Find unique insights (content far from others)
            for i, source in enumerate(sources_list):
                avg_sim_to_others = np.mean([sim_matrix[i][j] for j in range(n) if j != i])
                if avg_sim_to_others < 0.75:
                    result["unique_insights"][source] = {
                        "uniqueness_score": float(1 - avg_sim_to_others),
                        "text_preview": texts[source][:200],
                    }
        
        # Agreement analysis
        result["agreement_analysis"] = {
            "sources_available": len(texts),
            "high_agreement": result["agreement_score"] > 0.8,
            "consensus": "strong" if result["agreement_score"] > 0.85 else 
                        "moderate" if result["agreement_score"] > 0.7 else "weak",
        }
        
        return result
    
    def find_behavioral_consensus(self, behavior: str) -> Dict[str, Any]:
        """
        Find consensus across 5 tafsir sources for a specific behavior.
        
        Example: find_behavioral_consensus("الكبر")
        Returns where all scholars agree/disagree on الكبر
        """
        result = {
            "behavior": behavior,
            "mentions_by_source": {s: [] for s in TAFSIR_SOURCES},
            "total_mentions": 0,
            "consensus_verses": [],
            "disagreement_verses": [],
        }
        
        # Find all mentions
        for source in TAFSIR_SOURCES:
            for verse_key, text in self.tafsir_data[source].items():
                if behavior in text:
                    result["mentions_by_source"][source].append(verse_key)
                    result["total_mentions"] += 1
        
        # Find verses where multiple sources mention the behavior
        all_verses = set()
        for verses in result["mentions_by_source"].values():
            all_verses.update(verses)
        
        for verse_key in list(all_verses)[:50]:  # Limit for speed
            sources_mentioning = [s for s in TAFSIR_SOURCES 
                                 if verse_key in result["mentions_by_source"][s]]
            
            if len(sources_mentioning) >= 3:
                # Check alignment
                surah, ayah = map(int, verse_key.split(":"))
                alignment = self.align_verse(surah, ayah)
                
                if alignment["agreement_score"] > 0.8:
                    result["consensus_verses"].append({
                        "verse": verse_key,
                        "sources": sources_mentioning,
                        "agreement_score": alignment["agreement_score"],
                    })
                elif alignment["agreement_score"] < 0.6:
                    result["disagreement_verses"].append({
                        "verse": verse_key,
                        "sources": sources_mentioning,
                        "agreement_score": alignment["agreement_score"],
                    })
        
        return result
    
    def compare_scholars_on_topic(self, topic: str) -> Dict[str, Any]:
        """
        Compare how different scholars discuss a topic.
        
        Returns unique perspectives from each scholar.
        """
        result = {
            "topic": topic,
            "scholar_perspectives": {},
            "common_themes": [],
            "unique_contributions": {},
        }
        
        # Collect all mentions per scholar
        scholar_texts = {s: [] for s in TAFSIR_SOURCES}
        
        for source in TAFSIR_SOURCES:
            for verse_key, text in self.tafsir_data[source].items():
                if topic in text:
                    scholar_texts[source].append(text[:300])
        
        # Analyze each scholar's perspective
        for source in TAFSIR_SOURCES:
            texts = scholar_texts[source]
            if texts:
                result["scholar_perspectives"][TAFSIR_NAMES_AR[source]] = {
                    "mention_count": len(texts),
                    "sample": texts[0] if texts else "",
                }
        
        # Find common themes (words appearing in all scholars)
        if all(scholar_texts[s] for s in TAFSIR_SOURCES):
            word_sets = []
            for source in TAFSIR_SOURCES:
                combined = " ".join(scholar_texts[source])
                words = set(combined.split())
                word_sets.append(words)
            
            common_words = word_sets[0]
            for ws in word_sets[1:]:
                common_words &= ws
            
            # Filter to meaningful words (length > 3)
            result["common_themes"] = [w for w in common_words if len(w) > 3][:20]
        
        return result
    
    def get_verse_synthesis(self, surah: int, ayah: int) -> Dict[str, Any]:
        """
        Synthesize all 5 tafsir into a unified understanding.
        
        Not just listing - actual synthesis of perspectives.
        """
        verse_key = f"{surah}:{ayah}"
        alignment = self.align_verse(surah, ayah)
        
        synthesis = {
            "verse": verse_key,
            "alignment": alignment,
            "synthesis": {
                "agreed_points": [],
                "debated_points": [],
                "unique_insights": [],
            },
        }
        
        # Extract key points from each tafsir
        for source in TAFSIR_SOURCES:
            if verse_key in self.tafsir_data[source]:
                text = self.tafsir_data[source][verse_key]
                
                # Simple extraction of key phrases
                sentences = text.split(".")[:3]
                for sent in sentences:
                    if len(sent.strip()) > 20:
                        # Check if this point appears in other tafsir
                        appears_in_others = sum(
                            1 for s in TAFSIR_SOURCES 
                            if s != source and verse_key in self.tafsir_data[s]
                            and any(word in self.tafsir_data[s][verse_key] 
                                   for word in sent.split() if len(word) > 4)
                        )
                        
                        if appears_in_others >= 3:
                            if sent.strip() not in [p["text"] for p in synthesis["synthesis"]["agreed_points"]]:
                                synthesis["synthesis"]["agreed_points"].append({
                                    "text": sent.strip()[:150],
                                    "sources_agreeing": appears_in_others + 1,
                                })
                        elif appears_in_others == 0:
                            synthesis["synthesis"]["unique_insights"].append({
                                "source": TAFSIR_NAMES_AR[source],
                                "text": sent.strip()[:150],
                            })
        
        return synthesis


# =============================================================================
# TESTS
# =============================================================================

def test_tafsir_alignment(aligner: TafsirAligner) -> Dict[str, Any]:
    """Test tafsir alignment capabilities."""
    results = {"passed": 0, "failed": 0, "tests": []}
    
    # Test 1: Align Ayat al-Kursi (2:255)
    alignment = aligner.align_verse(2, 255)
    test1_passed = alignment["agreement_score"] > 0
    results["tests"].append({
        "name": "Align 2:255 (Ayat al-Kursi)",
        "passed": test1_passed,
        "agreement_score": alignment["agreement_score"],
    })
    results["passed" if test1_passed else "failed"] += 1
    
    # Test 2: Find consensus on الكبر
    consensus = aligner.find_behavioral_consensus("الكبر")
    test2_passed = consensus["total_mentions"] > 0
    results["tests"].append({
        "name": "Consensus on الكبر",
        "passed": test2_passed,
        "total_mentions": consensus["total_mentions"],
    })
    results["passed" if test2_passed else "failed"] += 1
    
    # Test 3: Compare scholars
    comparison = aligner.compare_scholars_on_topic("التوبة")
    test3_passed = len(comparison["scholar_perspectives"]) > 0
    results["tests"].append({
        "name": "Compare scholars on التوبة",
        "passed": test3_passed,
        "scholars_found": len(comparison["scholar_perspectives"]),
    })
    results["passed" if test3_passed else "failed"] += 1
    
    logger.info(f"Tests: {results['passed']} passed, {results['failed']} failed")
    return results


# =============================================================================
# MAIN
# =============================================================================

def build_tafsir_alignments() -> Dict[str, Any]:
    """Build tafsir alignments for key verses."""
    logger.info("=" * 60)
    logger.info("BUILDING TAFSIR ALIGNMENTS")
    logger.info("=" * 60)
    
    aligner = TafsirAligner()
    
    # Test
    test_results = test_tafsir_alignment(aligner)
    
    # Build alignments for sample verses
    sample_verses = [(2, 255), (2, 7), (3, 14), (7, 179), (2, 10)]
    alignments = []
    
    for surah, ayah in sample_verses:
        alignment = aligner.align_verse(surah, ayah)
        alignments.append(alignment)
    
    return {
        "status": "complete",
        "tafsir_sources": len(TAFSIR_SOURCES),
        "sample_alignments": len(alignments),
        "test_results": test_results,
    }


_aligner_instance = None

def get_tafsir_aligner() -> TafsirAligner:
    """Get the tafsir aligner."""
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = TafsirAligner()
    return _aligner_instance


if __name__ == "__main__":
    results = build_tafsir_alignments()
    print(json.dumps(results, indent=2, ensure_ascii=False))
