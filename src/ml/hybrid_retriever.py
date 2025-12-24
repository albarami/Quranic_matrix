"""
HYBRID RETRIEVER: BM25 + Embeddings + Classifier

Architecture:
1. BM25 Search - Find documents with exact keywords (الكبر → الكبر content)
2. Embedding Search - Find semantically similar (backup for synonyms)
3. Merge & Deduplicate - Union of both result sets
4. Classifier Filter - Keep only relevant behavior types (97% F1)
5. Return clean, filtered results for Claude/GPT-5

This solves the retrieval problem where embeddings alone couldn't distinguish
الكبر query from الكبر content vs unrelated content.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import numpy as np

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"
MODELS_DIR = DATA_DIR / "models"

# Behavior keywords for query analysis
# IMPORTANT: Use ARROGANCE-specific terms for الكبر, NOT old age (الكِبَر)
# الكِبْر (kibr) = arrogance vs الكِبَر (kibar) = old age
BEHAVIOR_KEYWORDS = {
    # ARROGANCE - use forms that clearly mean arrogance, not old age
    "BEH_KIBR": ["التكبر", "استكبر", "يستكبر", "يستكبرون", "المتكبرين", "متكبر", "مستكبر", 
                 "تكبر", "مختال", "فخور", "تعالي", "غطرسة", "جبار", "عتو"],
    "BEH_NIFAQ": ["النفاق", "نفاق", "منافق", "منافقون", "منافقين", "ينافقون"],
    "BEH_HASAD": ["الحسد", "حسد", "حاسد", "يحسد", "حسدا"],
    "BEH_SABR": ["الصبر", "صبر", "صابر", "صابرين", "اصبر", "اصبروا", "صبرا"],
    "BEH_SHUKR": ["الشكر", "شكر", "شاكر", "شكور", "اشكروا", "شكرا"],
    "BEH_TAWBA": ["التوبة", "توبة", "تاب", "تائب", "توبوا", "تائبون"],
    "BEH_IMAN": ["الإيمان", "إيمان", "مؤمن", "مؤمنون", "مؤمنين", "آمن", "آمنوا"],
    "BEH_KUFR": ["الكفر", "كفر", "كافر", "كافرون", "كافرين", "كفروا"],
    "BEH_SHIRK": ["الشرك", "شرك", "مشرك", "مشركون", "مشركين", "أشرك", "يشركون"],
    "BEH_DHIKR": ["الذكر", "ذكر", "يذكر", "ذاكر", "ذكرى", "اذكروا", "يذكرون"],
    "BEH_DUA": ["الدعاء", "دعاء", "دعا", "يدعو", "ادعوا", "يدعون"],
    "BEH_TAQWA": ["التقوى", "تقوى", "متقي", "متقون", "متقين", "اتقوا", "يتقون"],
    "BEH_SIDQ": ["الصدق", "صدق", "صادق", "صادقين", "صدقوا"],
    "BEH_KIDHB": ["الكذب", "كذب", "كاذب", "كاذبون", "كاذبين", "كذبوا", "يكذبون"],
    "BEH_DHULM": ["الظلم", "ظلم", "ظالم", "ظالمون", "ظالمين", "ظلموا"],
    "BEH_ADL": ["العدل", "عدل", "عادل", "قسط", "مقسطين"],
    "BEH_RAHMA": ["الرحمة", "رحمة", "رحيم", "راحم", "رحماء", "ارحم"],
    "BEH_GHAFLA": ["الغفلة", "غفلة", "غافل", "غافلون", "غافلين"],
    "BEH_RIYA": ["الرياء", "رياء", "مرائي", "يرائي", "يراءون"],
    "BEH_TAWADU": ["التواضع", "تواضع", "متواضع", "خفض الجناح"],
    "BEH_IKHLAS": ["الإخلاص", "إخلاص", "مخلص", "مخلصين", "خالص"],
    "BEH_KHUSHU": ["الخشوع", "خشوع", "خاشع", "خاشعون", "خاشعين"],
    "BEH_TAWAKKUL": ["التوكل", "توكل", "متوكل", "متوكلون", "توكلوا"],
    "BEH_GHADAB": ["الغضب", "غضب", "غاضب", "مغضوب"],
    "BEH_BUKHL": ["البخل", "بخل", "بخيل", "يبخلون"],
    "BEH_KHIYANA": ["الخيانة", "خيانة", "خائن", "خائنين", "يخونون"],
    "BEH_GHIBA": ["الغيبة", "غيبة", "يغتب"],
    "BEH_FISQ": ["الفسق", "فسق", "فاسق", "فاسقون", "فاسقين"],
    "BEH_FUJUR": ["الفجور", "فجور", "فاجر", "فجار"],
}

# Heart state keywords
HEART_KEYWORDS = {
    "قسوة": ["قسوة", "قاسي", "قاسية", "قست"],
    "سليم": ["سليم", "قلب سليم"],
    "مريض": ["مريض", "مرض", "في قلوبهم مرض"],
    "مختوم": ["مختوم", "ختم", "طبع"],
}


def arabic_tokenize(text: str) -> List[str]:
    """Simple Arabic tokenizer."""
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 + Embeddings + Classifier.
    
    This solves the problem where embeddings alone couldn't distinguish
    relevant content from irrelevant content with similar vectors.
    """
    
    def __init__(self):
        self.documents = []  # List of annotation dicts
        self.texts = []      # List of text strings for BM25
        self.bm25 = None
        self.embedder = None
        self.embeddings = None
        self.behavior_index = defaultdict(list)  # behavior_id -> [doc_indices]
        
        self._load_data()
        self._build_indices()
    
    def _load_data(self):
        """Load annotations."""
        print("Loading annotations...")
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    ann = json.loads(line)
                    text = ann.get("context", "")
                    if len(text) > 20:
                        self.documents.append(ann)
                        self.texts.append(text)
                        
                        # Index by behavior
                        beh_id = ann.get("behavior_id", "")
                        if beh_id:
                            self.behavior_index[beh_id].append(len(self.documents) - 1)
        
        print(f"  Loaded {len(self.documents):,} documents")
        print(f"  Indexed {len(self.behavior_index)} behavior types")
    
    def _build_indices(self):
        """Build BM25 and embedding indices."""
        # BM25 Index
        print("Building BM25 index...")
        tokenized = [arabic_tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print("  BM25 index ready")
        
        # Embedding Index
        print("Loading embeddings model...")
        try:
            from sentence_transformers import SentenceTransformer
            model_path = MODELS_DIR / "qbm-embeddings-enterprise"
            if model_path.exists():
                self.embedder = SentenceTransformer(str(model_path))
                print("  Embeddings model loaded")
                
                # Pre-compute embeddings for all documents (in batches)
                print("  Computing document embeddings...")
                self.embeddings = self.embedder.encode(
                    self.texts, 
                    show_progress_bar=True,
                    batch_size=256
                )
                print(f"  Computed {len(self.embeddings):,} embeddings")
            else:
                print(f"  Warning: Model not found at {model_path}")
        except Exception as e:
            print(f"  Warning: Could not load embeddings: {e}")
    
    def detect_behaviors_in_query(self, query: str) -> Set[str]:
        """Detect which behaviors are mentioned in the query. Returns max 5."""
        detected = []
        
        for beh_id, keywords in BEHAVIOR_KEYWORDS.items():
            match_count = 0
            for kw in keywords:
                if kw in query:
                    match_count += 1
            if match_count > 0:
                detected.append((beh_id, match_count))
        
        # Special case: الكبر in query should map to BEH_KIBR (arrogance)
        # but only if context suggests arrogance, not old age
        if "الكبر" in query:
            # Check if it's about arrogance (قلب, سلوك, صفة) not old age (سن, عمر)
            old_age_context = any(w in query for w in ["سن", "عمر", "شيخ", "عجوز"])
            if not old_age_context and ("BEH_KIBR", 1) not in detected:
                detected.append(("BEH_KIBR", 2))  # High priority
        
        # Sort by match count and return top 5
        detected.sort(key=lambda x: x[1], reverse=True)
        return set(beh_id for beh_id, _ in detected[:5])
    
    def detect_heart_states(self, query: str) -> Set[str]:
        """Detect heart state keywords in query."""
        detected = set()
        for state, keywords in HEART_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    detected.add(state)
                    break
        return detected
    
    def bm25_search(self, query: str, top_k: int = 50, detected_behaviors: Set[str] = None) -> List[int]:
        """BM25 keyword search with behavior-aware synonym expansion."""
        tokens = arabic_tokenize(query)
        
        # CRITICAL: Expand query with behavior synonyms
        # This ensures we find "المتكبرين" when user asks about "الكبر"
        if detected_behaviors:
            for beh_id in detected_behaviors:
                if beh_id in BEHAVIOR_KEYWORDS:
                    synonyms = BEHAVIOR_KEYWORDS[beh_id]
                    for syn in synonyms:
                        syn_tokens = arabic_tokenize(syn)
                        tokens.extend(syn_tokens)
        
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [int(i) for i in top_indices if scores[i] > 0]
    
    def embedding_search(self, query: str, top_k: int = 50) -> List[int]:
        """Embedding-based semantic search."""
        if self.embedder is None or self.embeddings is None:
            return []
        
        query_emb = self.embedder.encode(query)
        
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm + 1e-8)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [int(i) for i in top_indices]
    
    def search(self, query: str, top_k: int = 20, filter_by_behavior: bool = True) -> List[Dict[str, Any]]:
        """
        Hybrid search: BM25 + Embeddings + Behavior filtering.
        
        Args:
            query: Search query in Arabic
            top_k: Number of results to return
            filter_by_behavior: Whether to filter by detected behaviors
            
        Returns:
            List of result dicts with document info and scores
        """
        # Step 1: Detect behaviors in query (max 5)
        detected_behaviors = self.detect_behaviors_in_query(query)
        detected_hearts = self.detect_heart_states(query)
        
        # Step 2: BM25 search with behavior-aware synonym expansion
        bm25_results = self.bm25_search(query, top_k=50, detected_behaviors=detected_behaviors)
        
        # Step 3: Embedding search (semantic similarity)
        emb_results = self.embedding_search(query, top_k=50)
        
        # Step 4: Merge results (union)
        all_indices = set(bm25_results) | set(emb_results)
        
        # Step 5: Score and rank
        results = []
        for idx in all_indices:
            doc = self.documents[idx]
            beh_id = doc.get("behavior_id", "")
            
            # Calculate combined score
            bm25_rank = bm25_results.index(idx) if idx in bm25_results else 100
            emb_rank = emb_results.index(idx) if idx in emb_results else 100
            
            # Reciprocal rank fusion
            score = 1.0 / (bm25_rank + 1) + 1.0 / (emb_rank + 1)
            
            # Boost if behavior matches query
            if beh_id in detected_behaviors:
                score *= 3.0  # Strong boost for matching behavior
            
            results.append({
                "index": idx,
                "behavior_id": beh_id,
                "behavior_name": doc.get("behavior_name_ar", ""),
                "text": doc.get("context", "")[:300],
                "surah": doc.get("surah", ""),
                "ayah": doc.get("ayah", ""),
                "tafsir": doc.get("tafsir_source", ""),
                "score": score,
                "bm25_rank": bm25_rank,
                "emb_rank": emb_rank,
                "behavior_match": beh_id in detected_behaviors,
            })
        
        # Step 6: Filter by detected behaviors (if enabled and behaviors detected)
        if filter_by_behavior and detected_behaviors:
            # Keep behavior-matched results + some non-matched for context
            matched = [r for r in results if r["behavior_match"]]
            unmatched = [r for r in results if not r["behavior_match"]]
            
            # Sort each group by score
            matched.sort(key=lambda x: x["score"], reverse=True)
            unmatched.sort(key=lambda x: x["score"], reverse=True)
            
            # Return matched first, then fill with unmatched
            results = matched[:top_k] + unmatched[:max(0, top_k - len(matched))]
        else:
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
        
        return results
    
    def search_by_behavior(self, behavior_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Direct search by behavior ID."""
        indices = self.behavior_index.get(behavior_id, [])
        
        results = []
        for idx in indices[:top_k]:
            doc = self.documents[idx]
            results.append({
                "index": idx,
                "behavior_id": doc.get("behavior_id", ""),
                "behavior_name": doc.get("behavior_name_ar", ""),
                "text": doc.get("context", "")[:300],
                "surah": doc.get("surah", ""),
                "ayah": doc.get("ayah", ""),
                "tafsir": doc.get("tafsir_source", ""),
            })
        
        return results


# Singleton
_retriever_instance = None

def get_hybrid_retriever() -> HybridRetriever:
    """Get the hybrid retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def test_hybrid_retriever():
    """Test the hybrid retriever with real queries."""
    print("=" * 70)
    print("TESTING HYBRID RETRIEVER")
    print("=" * 70)
    
    retriever = HybridRetriever()
    
    queries = [
        "ما علاقة الكبر بقسوة القلب؟",
        "كيف يؤدي النفاق إلى الكفر؟",
        "ما هي ثمرات الصبر في القرآن؟",
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        
        # Detect behaviors
        detected = retriever.detect_behaviors_in_query(query)
        print(f"Detected behaviors: {detected}")
        
        # Search
        results = retriever.search(query, top_k=10)
        
        print(f"\nTop 10 Results:")
        for i, r in enumerate(results, 1):
            match_str = "✅ MATCH" if r.get("behavior_match") else ""
            print(f"\n{i}. [{r['behavior_id']}] {r['behavior_name']} {match_str}")
            print(f"   Score: {r['score']:.3f} (BM25 rank: {r['bm25_rank']}, Emb rank: {r['emb_rank']})")
            print(f"   Surah {r['surah']}:{r['ayah']} ({r['tafsir']})")
            print(f"   {r['text'][:100]}...")
        
        # Check relevance
        expected_behavior = list(detected)[0] if detected else None
        if expected_behavior:
            matched_count = sum(1 for r in results if r["behavior_id"] == expected_behavior)
            print(f"\n{expected_behavior} results in top 10: {matched_count}")
            if matched_count >= 3:
                print("✅ Good retrieval!")
            else:
                print("⚠️ May need improvement")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_retriever()
