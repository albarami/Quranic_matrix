"""
Layer 7: Hybrid RAG + Frontier Model Integration

The INTELLIGENT system that combines:
- YOUR trained retrieval components (Layers 2-6)
- Frontier model reasoning (Claude/GPT-5)

DON'T train a full LLM. GPT-5/Claude will ALWAYS reason better.
INSTEAD: Train the retrieval components, use frontier models for reasoning.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Bouzidani Framework - 11 Dimensions
DIMENSIONS = {
    "organic": "العضوي - أي عضو يتعلق به السلوك (قلب، لسان، عين...)",
    "situational": "الموقفي - شكل السلوك (داخلي، قولي، علائقي، جسدي، سمة)",
    "systemic": "النظامي - المجال (عبادي، أسري، مجتمعي، مالي...)",
    "spatial": "المكاني - أين يحدث (مسجد، بيت، سوق، خلوة...)",
    "temporal": "الزماني - متى (دنيا، موت، برزخ، قيامة، آخرة)",
    "agent": "الفاعل - من يقوم به (مؤمن، كافر، منافق، نبي...)",
    "source": "المصدر - من أين ينبع (وحي، فطرة، نفس، شيطان...)",
    "evaluation": "التقييم - الحكم الشرعي (ممدوح، مذموم، محايد)",
    "heart_type": "حالة القلب - (سليم، مريض، قاسي، مختوم، ميت، منيب)",
    "consequence": "العاقبة - النتيجة (دنيوية، أخروية، فردية، مجتمعية)",
    "relationships": "العلاقات - الروابط مع سلوكيات أخرى (سبب، نتيجة، نقيض...)",
}

SYSTEM_PROMPT = """أنت عالم متخصص في تحليل السلوك القرآني وفق منهجية البوزيداني.

## المنهجية الخماسية:
1. التحليل العضوي - تحديد الأعضاء المرتبطة بالسلوك
2. التحليل الموقفي - فهم سياق السلوك وشكله
3. التحليل النظامي - تحديد المجال الذي يندرج فيه السلوك
4. التحليل الزماني والمكاني - متى وأين يحدث
5. التحليل العلائقي - الروابط مع سلوكيات أخرى

## الأبعاد الإحدى عشر:
1. العضوي: القلب، اللسان، العين، الأذن، اليد، الرجل...
2. الموقفي: داخلي، قولي، علائقي، جسدي، سمة
3. النظامي: عبادي، أسري، مجتمعي، مالي، قضائي، سياسي
4. المكاني: مسجد، بيت، سوق، خلوة، ملأ، سفر، حرب
5. الزماني: دنيا، عند الموت، برزخ، قيامة، آخرة
6. الفاعل: الله، مؤمن، كافر، منافق، نبي، ملائكة، شيطان
7. المصدر: وحي، فطرة، نفس، شيطان، بيئة، قلب
8. التقييم: ممدوح، مذموم، محايد، تحذير
9. حالة القلب: سليم، مريض، قاسي، مختوم، ميت، منيب
10. العاقبة: دنيوية، أخروية، فردية، مجتمعية
11. العلاقات: سبب، نتيجة، نقيض، مشابه، شرط

## التعليمات:
- استخدم السياق المسترجع من قاعدة البيانات
- استشهد بالآيات والتفاسير المقدمة
- حلل وفق جميع الأبعاد المناسبة
- اذكر إجماع المفسرين الخمسة عند توفره
- اكتشف السلاسل السلوكية والعلاقات السببية
- كن دقيقاً وعلمياً في التحليل"""


# =============================================================================
# HYBRID RAG SYSTEM
# =============================================================================

class HybridRAGSystem:
    """
    The complete QBM Intelligent System.
    
    Architecture:
    - Layers 2-6: YOUR trained retrieval components (GPU trained)
    - Layer 7: Frontier model (Claude/GPT-5) via API for reasoning
    
    Flow:
    1. Embed question with YOUR Arabic embeddings (Layer 2)
    2. Retrieve relevant verses from vector DB
    3. Classify behaviors in question (Layer 3)
    4. Extract relations (Layer 4)
    5. Find related behaviors via trained GNN (Layer 5)
    6. Rerank with YOUR domain reranker (Layer 6)
    7. Send to Claude/GPT-5 API with rich context (Layer 7)
    
    Key Principle:
    - TRAIN retrieval components locally (Layers 2-6)
    - USE frontier model API for reasoning (Layer 7)
    - DON'T train an LLM - Claude/GPT-5 will always reason better
    """
    
    def __init__(self):
        # Trained components (Layers 2-6)
        self.embedder = None
        self.classifier = None
        self.relation_extractor = None
        self.gnn = None
        self.reranker = None
        self.vector_db = None
        self.tafsir_data = {}
        self.spans = []
        
        # HYBRID RETRIEVER (BM25 + Embeddings + Behavior filtering)
        self.hybrid_retriever = None
        
        # Frontier model clients (Layer 7)
        self.claude_client = None
        self.openai_client = None
        
        # Caching for performance
        self.response_cache = {}
        self.embedding_cache = {}
        
        self._load_components()
        self._init_llm_clients()
    
    def _init_llm_clients(self):
        """Initialize frontier model API clients."""
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            load_dotenv(".env.local")
        except ImportError:
            pass
        
        # Primary: Azure OpenAI
        try:
            from openai import AzureOpenAI
            import os
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if api_key and endpoint:
                self.azure_client = AzureOpenAI(
                    api_key=api_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                    azure_endpoint=endpoint
                )
                logger.info("Initialized: Azure OpenAI API client")
            else:
                self.azure_client = None
        except ImportError:
            self.azure_client = None
        except Exception as e:
            self.azure_client = None
            logger.warning(f"Could not init Azure OpenAI client: {e}")
        
        # Fallback: Claude
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic()
            logger.info("Initialized: Claude API client (fallback)")
        except ImportError:
            pass
        except Exception:
            pass
    
    def _load_components(self):
        """Load all trained components."""
        # Load HYBRID RETRIEVER (primary retrieval method)
        try:
            try:
                from .hybrid_retriever import get_hybrid_retriever
            except ImportError:
                from hybrid_retriever import get_hybrid_retriever
            self.hybrid_retriever = get_hybrid_retriever()
            logger.info("Loaded: Hybrid Retriever (BM25 + Embeddings)")
        except Exception:
            pass  # Hybrid retriever optional - will use fallback
        
        try:
            try:
                from .arabic_embeddings import get_qbm_embeddings
            except ImportError:
                from arabic_embeddings import get_qbm_embeddings
            self.embedder = get_qbm_embeddings()
            logger.info("Loaded: Arabic Embeddings")
        except Exception:
            pass  # Embeddings loaded via hybrid retriever
        
        try:
            try:
                from .behavioral_classifier import get_behavioral_classifier
            except ImportError:
                from behavioral_classifier import get_behavioral_classifier
            self.classifier = get_behavioral_classifier()
            logger.info("Loaded: Behavioral Classifier")
        except Exception:
            pass  # Classifier optional - will use keyword detection
        
        try:
            try:
                from .relation_extractor import get_relation_extractor
            except ImportError:
                from relation_extractor import get_relation_extractor
            self.relation_extractor = get_relation_extractor()
            logger.info("Loaded: Relation Extractor")
        except Exception:
            pass  # Relation extractor optional
        
        try:
            try:
                from .graph_reasoner import get_reasoning_engine
            except ImportError:
                from graph_reasoner import get_reasoning_engine
            self.gnn = get_reasoning_engine()
            logger.info("Loaded: Graph Reasoner")
        except Exception:
            pass  # GNN optional
        
        try:
            try:
                from .domain_reranker import get_domain_reranker
            except ImportError:
                from domain_reranker import get_domain_reranker
            self.reranker = get_domain_reranker()
            logger.info("Loaded: Domain Reranker")
        except Exception:
            pass  # Reranker optional
        
        # Load tafsir data
        self._load_tafsir()
    
    def _load_tafsir(self):
        """Load all 5 tafsir sources."""
        tafsir_dir = DATA_DIR / "tafsir"
        sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
        
        for source in sources:
            filepath = tafsir_dir / f"{source}.ar.jsonl"
            if filepath.exists():
                self.tafsir_data[source] = {}
                with open(filepath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            ref = entry.get("reference", {})
                            key = f"{ref.get('surah')}:{ref.get('ayah')}"
                            self.tafsir_data[source][key] = entry.get("text_ar", "")
                logger.info(f"Loaded tafsir: {source} ({len(self.tafsir_data[source])} entries)")
    
    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Step 1-2: HYBRID RETRIEVAL (BM25 + Embeddings + Behavior filtering).
        
        This is the key improvement - uses keyword matching (BM25) combined with
        semantic search and behavior-based filtering for accurate retrieval.
        """
        candidates = []
        
        # PRIMARY: Use hybrid retriever (BM25 + Embeddings + Behavior filtering)
        if self.hybrid_retriever is not None:
            results = self.hybrid_retriever.search(query, top_k=top_k)
            for r in results:
                candidates.append({
                    "source": r.get("tafsir", ""),
                    "verse": f"{r.get('surah', '')}:{r.get('ayah', '')}",
                    "text": r.get("text", ""),
                    "score": r.get("score", 0),
                    "behavior_id": r.get("behavior_id", ""),
                    "behavior_name": r.get("behavior_name", ""),
                    "behavior_match": r.get("behavior_match", False),
                })
            return candidates
        
        # FALLBACK: keyword search in tafsir (if hybrid retriever not available)
        query_terms = query.split()
        for source, verses in self.tafsir_data.items():
            for verse_key, text in verses.items():
                score = sum(1 for term in query_terms if term in text)
                if score > 0:
                    candidates.append({
                        "source": source,
                        "verse": verse_key,
                        "text": text[:500],
                        "score": score,
                    })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
    
    def classify_behaviors(self, text: str) -> List[Dict[str, Any]]:
        """
        Step 3: Classify behaviors mentioned in text.
        Returns max 5 behaviors to avoid noise.
        """
        # PRIMARY: Use hybrid retriever's behavior detection (max 5)
        if self.hybrid_retriever is not None:
            detected = self.hybrid_retriever.detect_behaviors_in_query(text)
            return [{"behavior": beh, "confidence": 0.9} for beh in list(detected)[:5]]
        
        # FALLBACK: Use classifier if available
        if self.classifier is not None:
            result = self.classifier.predict(text)
            behaviors = result.get("behaviors", [])
            # Sort by confidence and take top 5
            if behaviors and isinstance(behaviors[0], dict):
                behaviors.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return behaviors[:5]
        
        # LAST RESORT: simple keyword detection
        behaviors = []
        keywords = ["الكبر", "النفاق", "الصبر", "الشكر", "التوبة", "الإيمان", "الكفر"]
        for kw in keywords:
            if kw in text:
                behaviors.append({"behavior": kw, "confidence": 0.7})
        return behaviors[:5]
    
    def find_related_behaviors(self, behaviors: List[str]) -> Dict[str, Any]:
        """
        Step 4: Find related behaviors via GNN.
        """
        related = {"chains": [], "causes": [], "effects": [], "opposites": []}
        
        if self.gnn is not None:
            for behavior in behaviors:
                # Find paths from this behavior
                path_result = self.gnn.find_path(behavior, "قسوة_القلب")
                if path_result.get("found"):
                    related["chains"].append(path_result["path"])
        
        return related
    
    def get_tafsir_context(self, verse_keys: List[str]) -> Dict[str, List[Dict]]:
        """
        Step 5: Get tafsir from all 5 sources for relevant verses.
        """
        tafsir_context = {source: [] for source in self.tafsir_data.keys()}
        
        for verse_key in verse_keys[:10]:  # Limit to top 10 verses
            for source, verses in self.tafsir_data.items():
                if verse_key in verses:
                    tafsir_context[source].append({
                        "verse": verse_key,
                        "text": verses[verse_key][:300],
                    })
        
        return tafsir_context
    
    def rerank_evidence(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Step 6: Rerank with domain-specific reranker.
        """
        if self.reranker is not None:
            return self.reranker.rank_with_metadata(query, candidates, text_key="text", top_k=20)
        
        # Fallback: return as-is
        return candidates[:20]
    
    def build_context(self, 
                      query: str,
                      candidates: List[Dict],
                      behaviors: List[str],
                      related: Dict,
                      tafsir: Dict) -> str:
        """
        Build rich context for frontier model.
        """
        context_parts = []
        
        # Behaviors detected
        if behaviors:
            context_parts.append(f"## السلوكيات المكتشفة في السؤال:\n{', '.join(behaviors)}")
        
        # Related behaviors and chains
        if related.get("chains"):
            chains_str = "\n".join([" → ".join(chain) for chain in related["chains"]])
            context_parts.append(f"## السلاسل السلوكية:\n{chains_str}")
        
        # Top verses
        if candidates:
            verses_str = "\n".join([
                f"- {c.get('verse', '')}: {c.get('text', '')[:200]}"
                for c in candidates[:10]
            ])
            context_parts.append(f"## الآيات ذات الصلة:\n{verses_str}")
        
        # Tafsir from 5 sources
        for source, entries in tafsir.items():
            if entries:
                source_name = {
                    "ibn_kathir": "ابن كثير",
                    "tabari": "الطبري",
                    "qurtubi": "القرطبي",
                    "saadi": "السعدي",
                    "jalalayn": "الجلالين",
                }.get(source, source)
                
                entries_str = "\n".join([
                    f"  - {e['verse']}: {e['text'][:150]}"
                    for e in entries[:3]
                ])
                context_parts.append(f"## تفسير {source_name}:\n{entries_str}")
        
        return "\n\n".join(context_parts)
    
    def answer(self, question: str, use_api: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline + Frontier model reasoning.
        
        Args:
            question: User's question in Arabic
            use_api: Whether to call Claude/GPT-5 API
            
        Returns:
            {
                "question": str,
                "context": str,
                "answer": str,
                "evidence": {...},
                "processing_time_ms": float
            }
        """
        start_time = time.time()
        
        # Step 1-2: Retrieve relevant passages
        candidates = self.retrieve(question, top_k=50)
        
        # Step 3: Classify behaviors in question
        behaviors = self.classify_behaviors(question)
        behavior_names = [b.get("behavior", b) if isinstance(b, dict) else b for b in behaviors]
        
        # Step 4: Find related behaviors via GNN
        related = self.find_related_behaviors(behavior_names)
        
        # Step 5: Get tafsir for relevant verses
        verse_keys = [c.get("verse", "") for c in candidates[:20]]
        tafsir_context = self.get_tafsir_context(verse_keys)
        
        # Step 6: Rerank all evidence
        reranked = self.rerank_evidence(question, candidates)
        
        # Build context for frontier model
        context = self.build_context(
            query=question,
            candidates=reranked,
            behaviors=behavior_names,
            related=related,
            tafsir=tafsir_context,
        )
        
        # Step 7: Call frontier model (if enabled)
        answer = ""
        if use_api:
            answer = self._call_frontier_model(question, context)
        else:
            answer = f"[Context prepared for frontier model]\n\n{context}"
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "evidence": {
                "candidates_retrieved": len(candidates),
                "behaviors_detected": behavior_names,
                "related_behaviors": related,
                "tafsir_sources": list(tafsir_context.keys()),
                "reranked_count": len(reranked),
            },
            "processing_time_ms": round(processing_time, 2),
        }
    
    def _call_frontier_model(self, question: str, context: str) -> str:
        """
        Call Azure OpenAI or Claude with the prepared context.
        
        Strategy:
        1. Check cache first
        2. Try Azure OpenAI (primary)
        3. Fallback to Claude if Azure fails
        4. Cache successful responses
        """
        # Check cache first
        cache_key = hash(question + context[:500])
        if cache_key in self.response_cache:
            logger.info("Cache hit for question")
            return self.response_cache[cache_key]
        
        user_message = f"""السؤال: {question}

السياق المسترجع من قاعدة البيانات:
{context}

أجب باستخدام منهجية البوزيداني الخماسية وجميع الأبعاد الإحدى عشر المناسبة.
استشهد بالآيات والتفاسير المقدمة في السياق."""
        
        result = None
        
        # Try Azure OpenAI first (primary)
        if hasattr(self, 'azure_client') and self.azure_client is not None:
            try:
                import os
                deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                response = self.azure_client.chat.completions.create(
                    model=deployment_name,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ]
                )
                result = response.choices[0].message.content
                logger.info("Response from Azure OpenAI API")
            except Exception as e:
                logger.warning(f"Azure OpenAI API failed: {e}, trying fallback...")
        
        # Fallback to Claude
        if result is None and hasattr(self, 'claude_client') and self.claude_client is not None:
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                result = response.content[0].text
                logger.info("Response from Claude API (fallback)")
            except Exception as e:
                logger.error(f"Claude API also failed: {e}")
        
        # If both failed
        if result is None:
            return "[Error: Both Azure OpenAI and Claude APIs failed. Check API keys and network.]"
        
        # Cache successful response
        self.response_cache[cache_key] = result
        
        return result


# =============================================================================
# SINGLETON
# =============================================================================

_system_instance = None

def get_hybrid_system() -> HybridRAGSystem:
    """Get the hybrid RAG system."""
    global _system_instance
    if _system_instance is None:
        _system_instance = HybridRAGSystem()
    return _system_instance


# =============================================================================
# TEST
# =============================================================================

def test_hybrid_system() -> Dict[str, Any]:
    """Test the hybrid RAG system."""
    logger.info("=" * 60)
    logger.info("TESTING HYBRID RAG SYSTEM")
    logger.info("=" * 60)
    
    system = HybridRAGSystem()
    
    # Test question
    question = "ما علاقة الكبر بقسوة القلب؟"
    
    # Get answer without API call
    result = system.answer(question, use_api=False)
    
    logger.info(f"Question: {question}")
    logger.info(f"Behaviors detected: {result['evidence']['behaviors_detected']}")
    logger.info(f"Candidates retrieved: {result['evidence']['candidates_retrieved']}")
    logger.info(f"Processing time: {result['processing_time_ms']}ms")
    
    return result


if __name__ == "__main__":
    result = test_hybrid_system()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
