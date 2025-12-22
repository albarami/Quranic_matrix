"""
Initialize the complete QBM AI System.

This script initializes the knowledge graph, vector store, and RAG pipeline
with all available data from the vocabulary files.
"""

import os
from pathlib import Path

from .graph.qbm_graph import QBMKnowledgeGraph
from .vectors.qbm_vectors import QBMVectorStore
from .rag.qbm_rag import QBMRAGPipeline


def init_qbm_system(
    graph_db_path: str = "data/qbm_graph.db",
    vector_db_path: str = "data/chromadb",
    load_from_existing: bool = True,
) -> QBMRAGPipeline:
    """
    Initialize the complete QBM AI system.

    Args:
        graph_db_path: Path to SQLite database for graph.
        vector_db_path: Path to ChromaDB directory.
        load_from_existing: Whether to load existing data.

    Returns:
        Initialized QBMRAGPipeline instance.
    """
    print("Initializing QBM AI System...")

    # Initialize graph
    graph = QBMKnowledgeGraph(db_path=graph_db_path)

    if load_from_existing and Path(graph_db_path).exists():
        print("Loading existing graph...")
        graph.load()
    else:
        print("Building new graph from vocabulary...")
        _build_graph(graph)

    stats = graph.get_behavior_statistics()
    print(f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    # Initialize vector store
    vector_store = QBMVectorStore(persist_dir=vector_db_path)

    # Check if behaviors are already loaded
    current_stats = vector_store.get_collection_stats()
    if current_stats["behaviors"] == 0:
        print("Loading behaviors into vector store...")
        count = vector_store.add_behaviors_from_graph(graph)
        print(f"Loaded {count} behaviors into vector store")
    else:
        print(f"Vector store already has {current_stats['behaviors']} behaviors")

    # Initialize RAG pipeline
    rag = QBMRAGPipeline(vector_store=vector_store, graph=graph)

    print("\nQBM AI System initialized successfully!")
    print(f"  Graph: {stats['total_behaviors']} behaviors")
    print(f"  Vector Store: {vector_store.get_collection_stats()}")
    print(f"  LLM: {'Available' if rag.client else 'Not configured'}")

    return rag


def _build_graph(graph: QBMKnowledgeGraph) -> None:
    """Build graph from vocabulary files."""
    vocab_dir = Path("vocab")

    # Load behaviors
    behavior_path = vocab_dir / "behavior_concepts.json"
    if behavior_path.exists():
        count = graph.load_behaviors_from_vocab(str(behavior_path))
        print(f"  Loaded {count} behaviors")

    # Add behavioral relationships
    _add_behavioral_relationships(graph)

    # Save graph
    graph.save()


def _add_behavioral_relationships(graph: QBMKnowledgeGraph) -> None:
    """Add known behavioral relationships from the QBM framework."""

    # Causal relationships
    causal_pairs = [
        ("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE"),
        ("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION"),
        ("BEH_COG_FOLLOWING_DESIRE", "BEH_COG_HEEDLESSNESS"),
        ("BEH_SPI_DISBELIEF", "BEH_COG_ARROGANCE"),
        ("BEH_SPI_HYPOCRISY", "BEH_SPEECH_LYING"),
        ("BEH_SPI_FAITH", "BEH_SPI_TAQWA"),
        ("BEH_SPI_TAQWA", "BEH_EMO_PATIENCE"),
        ("BEH_EMO_GRATITUDE", "BEH_SPI_FAITH"),
        ("BEH_EMO_ENVY", "BEH_EMO_HATRED"),
        ("BEH_EMO_HATRED", "BEH_SOC_OPPRESSION"),
    ]

    for cause, effect in causal_pairs:
        if cause in graph.G.nodes and effect in graph.G.nodes:
            graph.add_causal_relationship(cause, effect, confidence=0.9)

    # Opposite relationships
    opposite_pairs = [
        ("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY"),
        ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF"),
        ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING"),
        ("BEH_EMO_GRATITUDE", "BEH_EMO_INGRATITUDE"),
        ("BEH_EMO_PATIENCE", "BEH_EMO_IMPATIENCE"),
        ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION"),
        ("BEH_COG_KNOWLEDGE", "BEH_COG_IGNORANCE"),
        ("BEH_SPI_SINCERITY", "BEH_SPI_SHOWING_OFF"),
    ]

    for beh1, beh2 in opposite_pairs:
        if beh1 in graph.G.nodes and beh2 in graph.G.nodes:
            graph.add_opposite_relationship(beh1, beh2)

    print("  Added behavioral relationships")


def test_rag_query(rag: QBMRAGPipeline, question: str = "ما هو الكبر؟") -> None:
    """Test the RAG pipeline with a sample query."""
    print(f"\n{'='*60}")
    print(f"Testing RAG Query: {question}")
    print("="*60)

    result = rag.query(question)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    print(f"  Ayat: {result['sources']['ayat']}")
    print(f"  Behaviors: {result['sources']['behaviors']}")
    print(f"\nGraph Expansion:")
    for key, values in result['graph_expansion'].items():
        if values:
            print(f"  {key}: {values}")


if __name__ == "__main__":
    # Initialize the system
    rag = init_qbm_system()

    # Test with a sample query
    test_rag_query(rag, "ما هو الكبر وما أسبابه؟")
