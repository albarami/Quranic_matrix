"""
QBM Machine Learning Module - TRUE Intelligence (7 Layers)

Layer 1: Foundation Data (done)
Layer 2: Arabic-First Embeddings - Fine-tuned on QBM spans
Layer 3: Behavioral Classifier - Replaces keyword matching with ML
Layer 4: Relation Extraction - Learns causal relationships
Layer 5: Graph Reasoning - GNN for multi-hop discovery
Layer 6: Tafsir Alignment - Semantic cross-reference
Layer 7: Fine-tuned LLM - JAIS/Qwen trained on QBM framework
"""

# Layer 2: Arabic Embeddings
from .arabic_embeddings import (
    ArabicEmbeddingTrainer,
    train_arabic_embeddings,
    get_qbm_embeddings,
    test_embedding_quality,
)

# Layer 3: Behavioral Classifier
from .behavioral_classifier import (
    BehavioralClassifier,
    train_behavioral_classifier,
    get_behavioral_classifier,
    BEHAVIOR_CLASSES,
)

# Layer 4: Relation Extraction
from .relation_extractor import (
    RelationExtractor,
    train_relation_extractor,
    get_relation_extractor,
    RELATION_TYPES,
)

# Layer 5: Graph Reasoning - LAZY IMPORT (torch_geometric is optional)
# Import only when explicitly requested to avoid Windows DLL crashes
def get_reasoning_engine(*args, **kwargs):
    """Lazy import of graph reasoning engine."""
    from .graph_reasoner import get_reasoning_engine as _get_reasoning_engine
    return _get_reasoning_engine(*args, **kwargs)

def build_and_train_reasoner(*args, **kwargs):
    """Lazy import of graph reasoner builder."""
    from .graph_reasoner import build_and_train_reasoner as _build_and_train_reasoner
    return _build_and_train_reasoner(*args, **kwargs)

# These will be imported lazily when accessed
QBMGraphReasoner = None
ReasoningEngine = None

def _lazy_load_graph_reasoner():
    """Load graph reasoner classes on demand."""
    global QBMGraphReasoner, ReasoningEngine
    if QBMGraphReasoner is None:
        from .graph_reasoner import QBMGraphReasoner as _QBMGraphReasoner
        from .graph_reasoner import ReasoningEngine as _ReasoningEngine
        QBMGraphReasoner = _QBMGraphReasoner
        ReasoningEngine = _ReasoningEngine
    return QBMGraphReasoner, ReasoningEngine

# Layer 6: Tafsir Alignment
from .tafsir_aligner import (
    TafsirAligner,
    build_tafsir_alignments,
    get_tafsir_aligner,
)

# Layer 6: Domain Reranker
from .domain_reranker import (
    DomainReranker,
    train_domain_reranker,
    get_domain_reranker,
)

# Layer 7: Hybrid RAG + Frontier Model (REVISED - don't train LLM, use Claude/GPT-5)
from .hybrid_rag_system import (
    HybridRAGSystem,
    get_hybrid_system,
    test_hybrid_system,
    SYSTEM_PROMPT,
    DIMENSIONS,
)

# Legacy: LLM fine-tuning (kept for reference, but HYBRID approach is better)
from .qbm_llm import (
    QBMLLMFineTuner,
    TrainingDataGenerator,
    prepare_llm_training_data,
    finetune_qbm_llm,
    get_qbm_llm,
)

# Original pipeline (mechanical - to be replaced)
from .embedding_pipeline import (
    UnifiedPipeline,
    EmbeddingEngine,
    VectorDatabase,
    Reranker,
    LabelSchema,
    get_pipeline,
    build_and_save_index,
    DEVICE,
    TORCH_AVAILABLE,
    FAISS_AVAILABLE,
)

__all__ = [
    # Layer 2
    "ArabicEmbeddingTrainer",
    "train_arabic_embeddings",
    "get_qbm_embeddings",
    
    # Layer 3
    "BehavioralClassifier",
    "train_behavioral_classifier",
    "get_behavioral_classifier",
    "BEHAVIOR_CLASSES",
    
    # Layer 4
    "RelationExtractor",
    "train_relation_extractor",
    "get_relation_extractor",
    "RELATION_TYPES",
    
    # Layer 5
    "QBMGraphReasoner",
    "ReasoningEngine",
    "build_and_train_reasoner",
    "get_reasoning_engine",
    
    # Layer 6
    "TafsirAligner",
    "build_tafsir_alignments",
    "get_tafsir_aligner",
    
    # Layer 6: Domain Reranker
    "DomainReranker",
    "train_domain_reranker",
    "get_domain_reranker",
    
    # Layer 7: Hybrid RAG System
    "HybridRAGSystem",
    "get_hybrid_system",
    "test_hybrid_system",
    "SYSTEM_PROMPT",
    "DIMENSIONS",
    
    # Legacy LLM (kept for reference)
    "QBMLLMFineTuner",
    "TrainingDataGenerator",
    "prepare_llm_training_data",
    "finetune_qbm_llm",
    "get_qbm_llm",
    
    # Original pipeline
    "UnifiedPipeline",
    "EmbeddingEngine",
    "VectorDatabase",
    "Reranker",
    "LabelSchema",
    "get_pipeline",
    "build_and_save_index",
    "DEVICE",
    "TORCH_AVAILABLE",
    "FAISS_AVAILABLE",
]
