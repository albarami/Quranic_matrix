"""
Demo script for the QBM AI System.

This script demonstrates the RAG pipeline with Azure OpenAI.
Run from the project root: python -m src.ai.demo
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .graph.qbm_graph import QBMKnowledgeGraph
from .vectors.qbm_vectors import QBMVectorStore
from .rag.qbm_rag import QBMRAGPipeline


def main():
    """Run the QBM AI demo."""
    print("=" * 60)
    print("QBM AI System Demo")
    print("=" * 60)

    # Check Azure OpenAI configuration
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")

    print(f"\nAzure OpenAI Configuration:")
    print(f"  Endpoint: {endpoint[:30]}..." if endpoint else "  Endpoint: NOT SET")
    print(f"  API Key: {'*' * 10}..." if api_key else "  API Key: NOT SET")
    print(f"  Deployment: {deployment}")

    # Initialize components
    print("\nInitializing components...")

    graph = QBMKnowledgeGraph(db_path="data/qbm_graph.db")
    graph.load()
    print(f"  Graph: {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges")

    vector_store = QBMVectorStore(persist_dir="data/chromadb")
    stats = vector_store.get_collection_stats()
    print(f"  Vector Store: {stats}")

    # Ensure behaviors are loaded
    if stats["behaviors"] == 0:
        print("  Loading behaviors into vector store...")
        count = vector_store.add_behaviors_from_graph(graph)
        print(f"  Loaded {count} behaviors")

    # Initialize RAG pipeline
    rag = QBMRAGPipeline(vector_store=vector_store, graph=graph)
    print(f"  RAG Pipeline: LLM {'available' if rag.client else 'NOT available'}")

    if not rag.client:
        print("\n⚠️  Azure OpenAI not configured. Set environment variables:")
        print("    AZURE_OPENAI_API_KEY")
        print("    AZURE_OPENAI_ENDPOINT")
        print("    AZURE_OPENAI_DEPLOYMENT_NAME")
        return

    # Demo queries
    queries = [
        "ما هو الكبر وما أسبابه؟",
        "ما العلاقة بين الغفلة والكبر؟",
        "ما هي أضداد الصبر؟",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = rag.query(query)

        print(f"\n📖 Answer:\n{result['answer']}")

        print(f"\n📚 Sources:")
        if result["sources"]["behaviors"]:
            print(f"  Behaviors: {', '.join(result['sources']['behaviors'][:3])}")

        if result["graph_expansion"]:
            print(f"\n🔗 Graph Expansion:")
            for key, values in result["graph_expansion"].items():
                if values:
                    print(f"  {key}: {', '.join(values[:3])}")


if __name__ == "__main__":
    main()
