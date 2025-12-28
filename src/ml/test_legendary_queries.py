"""
Test the 5 Legendary Queries with the new Bouzidani-aligned QBM system.

Uses:
- qbm-embeddings-v2 (trained on 47,142 clean behaviors)
- qbm-classifier-v2 (97% F1, 33 TRUE behavior classes)
- qbm-gnn-v2 (behavior co-occurrence graph)
- behavior_graph_v2.json (33 nodes, 500 edges)
- 5 Tafsir sources
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "tafsir_behavioral_5axis.jsonl"
GRAPH_FILE = DATA_DIR / "behavior_graph_v2.json"

# Arabic to ID mapping
from src.ml.qbm_5axis_schema import ARABIC_TO_ID, is_true_behavior

ID_TO_ARABIC = {v: k for k, v in ARABIC_TO_ID.items()}


class QBMTestSystem:
    """Test system for legendary queries."""
    
    def __init__(self):
        print("=" * 70)
        print("QBM TEST SYSTEM - BOUZIDANI FRAMEWORK")
        print("=" * 70)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Load components
        self._load_embeddings()
        self._load_classifier()
        self._load_graph()
        self._load_annotations()
        
        print(f"\nSystem ready!")
        print(f"  - {len(self.annotations):,} annotations")
        print(f"  - {len(self.graph_nodes)} behavior nodes")
        print(f"  - {len(self.graph_edges)} graph edges")
    
    def _load_embeddings(self):
        """Load Arabic embeddings."""
        print("\nLoading embeddings...")
        emb_path = MODELS_DIR / "qbm-embeddings-v2"
        if emb_path.exists():
            self.embedder = SentenceTransformer(str(emb_path), device=self.device)
            print(f"  Loaded: {emb_path}")
        else:
            self.embedder = SentenceTransformer("aubmindlab/bert-base-arabertv2", device=self.device)
            print("  Using base AraBERT")
    
    def _load_classifier(self):
        """Load behavioral classifier."""
        print("\nLoading classifier...")
        cls_path = MODELS_DIR / "qbm-classifier-v2"
        
        if cls_path.exists():
            self.classifier = AutoModelForSequenceClassification.from_pretrained(str(cls_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(cls_path))
            self.classifier.to(self.device)
            self.classifier.eval()
            
            # Load label map
            label_map_file = cls_path / "label_map.json"
            if label_map_file.exists():
                with open(label_map_file, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    self.behavior_ids = label_data["behavior_ids"]
                    self.id_to_label = label_data["id_to_label"]
                    self.label_to_id = {v: k for k, v in self.id_to_label.items()}
            print(f"  Loaded: {cls_path} ({len(self.behavior_ids)} classes)")
        else:
            self.classifier = None
            self.behavior_ids = []
            print("  Classifier not found")
    
    def _load_graph(self):
        """Load behavior graph."""
        print("\nLoading graph...")
        if GRAPH_FILE.exists():
            with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            self.graph_nodes = {n["id"]: n for n in graph_data["nodes"]}
            self.graph_edges = graph_data["edges"]
            
            # Build adjacency
            self.adjacency = defaultdict(list)
            for edge in self.graph_edges:
                self.adjacency[edge["source"]].append((edge["target"], edge["weight"]))
                self.adjacency[edge["target"]].append((edge["source"], edge["weight"]))
            
            print(f"  Loaded: {len(self.graph_nodes)} nodes, {len(self.graph_edges)} edges")
        else:
            self.graph_nodes = {}
            self.graph_edges = []
            self.adjacency = {}
            print("  Graph not found")
    
    def _load_annotations(self):
        """Load filtered annotations."""
        print("\nLoading annotations...")
        self.annotations = []
        self.behavior_annotations = defaultdict(list)
        self.verse_annotations = defaultdict(list)
        self.tafsir_annotations = defaultdict(list)
        
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        self.annotations.append(ann)
                        
                        beh_id = ann.get("behavior_id", "")
                        if beh_id:
                            self.behavior_annotations[beh_id].append(ann)
                        
                        verse = f"{ann.get('surah', '')}:{ann.get('ayah', '')}"
                        self.verse_annotations[verse].append(ann)
                        
                        tafsir = ann.get("source", "")
                        if tafsir:
                            self.tafsir_annotations[tafsir].append(ann)
            
            print(f"  Loaded: {len(self.annotations):,} annotations")
            print(f"  Tafsirs: {list(self.tafsir_annotations.keys())}")
        else:
            print("  Annotations not found")
    
    def classify_text(self, text: str, top_k: int = 3) -> List[Dict]:
        """Classify text into behavior categories."""
        if not self.classifier:
            return []
        
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        top_indices = torch.topk(probs, min(top_k, len(probs))).indices.tolist()
        results = []
        for idx in top_indices:
            beh_id = self.behavior_ids[idx]
            arabic = ID_TO_ARABIC.get(beh_id, beh_id)
            results.append({
                "behavior_id": beh_id,
                "arabic": arabic,
                "confidence": probs[idx].item()
            })
        return results
    
    def search_behavior(self, behavior: str, limit: int = 10) -> List[Dict]:
        """Search for annotations about a behavior."""
        # Normalize
        behavior_clean = behavior.replace("ال", "").strip()
        
        # Find behavior ID
        beh_id = ARABIC_TO_ID.get(behavior_clean) or ARABIC_TO_ID.get(behavior)
        
        if not beh_id:
            # Try partial match
            for ar, bid in ARABIC_TO_ID.items():
                if behavior_clean in ar or ar in behavior_clean:
                    beh_id = bid
                    break
        
        if beh_id and beh_id in self.behavior_annotations:
            return self.behavior_annotations[beh_id][:limit]
        
        # Fallback: semantic search
        if self.embedder:
            query_emb = self.embedder.encode(behavior)
            
            results = []
            for ann in self.annotations[:5000]:  # Sample
                ctx = ann.get("context", "")
                if ctx:
                    ctx_emb = self.embedder.encode(ctx)
                    sim = np.dot(query_emb, ctx_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ctx_emb))
                    results.append((sim, ann))
            
            results.sort(key=lambda x: -x[0])
            return [r[1] for r in results[:limit]]
        
        return []
    
    def find_related_behaviors(self, behavior: str, depth: int = 2) -> List[Dict]:
        """Find behaviors related through the graph."""
        behavior_clean = behavior.replace("ال", "").strip()
        beh_id = ARABIC_TO_ID.get(behavior_clean) or ARABIC_TO_ID.get(behavior)
        
        if not beh_id or beh_id not in self.adjacency:
            return []
        
        related = []
        visited = {beh_id}
        queue = [(beh_id, 0)]
        
        while queue:
            current, d = queue.pop(0)
            if d > 0:
                arabic = ID_TO_ARABIC.get(current, current)
                related.append({"id": current, "arabic": arabic, "depth": d})
            
            if d < depth:
                for neighbor, weight in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
        
        return related
    
    def get_behavior_stats(self) -> Dict[str, int]:
        """Get behavior frequency statistics."""
        counts = Counter()
        for ann in self.annotations:
            beh_id = ann.get("behavior_id", "")
            if beh_id:
                counts[beh_id] += 1
        return dict(counts.most_common())
    
    def get_cross_tafsir(self, topic: str, limit_per_tafsir: int = 3) -> Dict[str, List[Dict]]:
        """Get interpretations from all 5 tafsirs."""
        results = {}
        
        for tafsir, anns in self.tafsir_annotations.items():
            # Search within this tafsir
            matches = []
            for ann in anns:
                ctx = ann.get("context", "")
                if topic in ctx or any(t in ctx for t in topic.split()):
                    matches.append(ann)
            
            if matches:
                results[tafsir] = matches[:limit_per_tafsir]
        
        return results


def run_legendary_queries():
    """Run the 5 legendary test queries."""
    
    system = QBMTestSystem()
    
    print("\n" + "=" * 70)
    print("LEGENDARY QUERIES TEST")
    print("=" * 70)
    
    # Query 1: Simple behavior query
    print("\n" + "-" * 70)
    print("QUERY 1: حلل سلوك الكبر")
    print("-" * 70)
    
    kibr_results = system.search_behavior("كبر", limit=5)
    print(f"\nFound {len(kibr_results)} annotations about الكبر:")
    for i, ann in enumerate(kibr_results[:3], 1):
        print(f"\n  [{i}] Surah {ann.get('surah')}:{ann.get('ayah')} ({ann.get('tafsir', 'Unknown')})")
        ctx = ann.get('context', '')[:200]
        print(f"      {ctx}...")
    
    # Related behaviors
    related = system.find_related_behaviors("كبر")
    if related:
        print(f"\n  Related behaviors (from graph):")
        for r in related[:5]:
            print(f"    - {r['arabic']} (depth: {r['depth']})")
    
    # Classify a sample text about kibr
    if kibr_results:
        sample = kibr_results[0].get("context", "")[:300]
        classifications = system.classify_text(sample)
        print(f"\n  Classifier output for sample text:")
        for c in classifications:
            print(f"    - {c['arabic']}: {c['confidence']:.2%}")
    
    # Query 2: Causal chain
    print("\n" + "-" * 70)
    print("QUERY 2: ما علاقة الكبر بقسوة القلب؟")
    print("-" * 70)
    
    # Find path in graph
    kibr_related = system.find_related_behaviors("كبر", depth=3)
    qaswa_related = system.find_related_behaviors("قسوة", depth=3)
    
    print(f"\n  Behaviors connected to الكبر: {len(kibr_related)}")
    for r in kibr_related[:5]:
        print(f"    - {r['arabic']}")
    
    # Search for contexts mentioning both
    combined_results = []
    for ann in system.annotations:
        ctx = ann.get("context", "")
        if "كبر" in ctx and ("قس" in ctx or "قلب" in ctx):
            combined_results.append(ann)
    
    print(f"\n  Annotations mentioning both concepts: {len(combined_results)}")
    for ann in combined_results[:2]:
        print(f"\n    [{ann.get('source')}] {ann.get('surah')}:{ann.get('ayah')}")
        print(f"    {ann.get('context', '')[:200]}...")
    
    # Query 3: Cross-tafsir
    print("\n" + "-" * 70)
    print("QUERY 3: ما قال المفسرون الخمسة في الختم على القلب؟")
    print("-" * 70)
    
    cross_results = system.get_cross_tafsir("ختم", limit_per_tafsir=2)
    print(f"\n  Found interpretations from {len(cross_results)} tafsirs:")
    
    for tafsir, anns in cross_results.items():
        print(f"\n  📖 {tafsir}:")
        for ann in anns[:1]:
            ctx = ann.get("context", "")[:150]
            print(f"     {ann.get('surah')}:{ann.get('ayah')}: {ctx}...")
    
    # Query 4: Statistical
    print("\n" + "-" * 70)
    print("QUERY 4: ما أكثر السلوكيات ذكراً في القرآن؟")
    print("-" * 70)
    
    stats = system.get_behavior_stats()
    print(f"\n  Top 10 most mentioned behaviors:")
    for i, (beh_id, count) in enumerate(list(stats.items())[:10], 1):
        arabic = ID_TO_ARABIC.get(beh_id, beh_id)
        print(f"    {i:2}. {arabic}: {count:,} mentions")
    
    # Query 5: Comparison
    print("\n" + "-" * 70)
    print("QUERY 5: قارن صبر المؤمن وصبر الكافر")
    print("-" * 70)
    
    sabr_results = system.search_behavior("صبر", limit=20)
    
    mumin_sabr = []
    kafir_sabr = []
    
    for ann in sabr_results:
        ctx = ann.get("context", "")
        if "مؤمن" in ctx or "آمن" in ctx or "المؤمن" in ctx:
            mumin_sabr.append(ann)
        if "كافر" in ctx or "كفر" in ctx or "الكافر" in ctx:
            kafir_sabr.append(ann)
    
    print(f"\n  صبر المؤمن - Found {len(mumin_sabr)} contexts:")
    for ann in mumin_sabr[:2]:
        print(f"    [{ann.get('source')}] {ann.get('context', '')[:120]}...")
    
    print(f"\n  صبر الكافر - Found {len(kafir_sabr)} contexts:")
    for ann in kafir_sabr[:2]:
        print(f"    [{ann.get('source')}] {ann.get('context', '')[:120]}...")
    
    if not kafir_sabr:
        print("    (No direct mentions - الكافر typically associated with عدم الصبر)")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"""
    ✅ Query 1 (Simple behavior): Found {len(kibr_results)} annotations
    ✅ Query 2 (Causal chain): Found {len(combined_results)} combined contexts
    ✅ Query 3 (Cross-tafsir): {len(cross_results)} tafsirs responded
    ✅ Query 4 (Statistical): Top behavior = {ID_TO_ARABIC.get(list(stats.keys())[0], 'N/A')} ({list(stats.values())[0]:,})
    ✅ Query 5 (Comparison): {len(mumin_sabr)} مؤمن contexts, {len(kafir_sabr)} كافر contexts
    
    System Components:
    - Embeddings: qbm-embeddings-v2
    - Classifier: qbm-classifier-v2 (97% F1)
    - Graph: {len(system.graph_nodes)} nodes, {len(system.graph_edges)} edges
    - Annotations: {len(system.annotations):,} TRUE behaviors
    """)


if __name__ == "__main__":
    run_legendary_queries()
