"""
Phase 6: GPU-Accelerated Processing (8x A100-80GB)

Run this script to:
1. Generate embeddings for all 322,939 annotations using 8 GPUs
2. Build PyTorch GPU vector index
3. Test search and reranking
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from src.ai.gpu import GPUEmbeddingPipeline, GPUVectorSearch, CrossEncoderReranker


def _safe_text(text: str) -> str:
    """Return a console-safe string without raising encoding errors."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except Exception:
        return text.encode("unicode_escape").decode("ascii")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 60)
    print("Phase 6: GPU-Accelerated Processing (8x A100-80GB)")
    print("=" * 60)

    # Step 1: Initialize GPU Embedding Pipeline
    print("\n[Step 1] Initializing GPU Embedding Pipeline...")
    embedding_pipeline = GPUEmbeddingPipeline(
        model_name="aubmindlab/bert-base-arabertv2",
        batch_size=128,  # Higher batch size for 8x A100
        use_multi_gpu=True,
    )

    gpu_info = embedding_pipeline.get_gpu_info()
    print(f"\nGPU Count: {gpu_info['count']}")
    print(f"Total GPU Memory: {sum(d['total_memory_gb'] for d in gpu_info['devices']):.1f} GB")

    # Step 2: Generate embeddings for annotations
    print("\n[Step 2] Generating embeddings for 322,939 annotations...")
    os.makedirs("data/embeddings", exist_ok=True)
    
    stats = embedding_pipeline.embed_annotations(
        annotations_path="data/annotations/tafsir_annotations.jsonl",
        output_path="data/embeddings/annotations_embeddings.npy",
        text_field="context",
        show_progress=True,
    )
    print(f"\nAnnotation embedding stats: {stats}")

    # Step 3: Build GPU vector index
    print("\n[Step 3] Building PyTorch GPU vector index...")
    
    embeddings = np.load("data/embeddings/annotations_embeddings.npy")
    
    vector_search = GPUVectorSearch(
        embedding_dim=768,
        use_gpu=True,
    )
    
    with open("data/embeddings/annotations_embeddings_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    vector_search.build_index(embeddings, metadata, show_progress=True)
    
    # Save index
    vector_search.save_index("data/embeddings/gpu_index.npy")
    print(f"\nIndex stats: {vector_search.get_stats()}")

    # Step 4: Initialize reranker
    print("\n[Step 4] Initializing Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(
        model_name="amberoad/bert-multilingual-passage-reranking-msmarco",
        batch_size=32,
    )

    # Step 5: Test search
    print("\n[Step 5] Testing search...")
    test_query = "الكبر والتكبر"  # Arrogance
    query_embedding = embedding_pipeline.embed_texts([test_query], show_progress=False)
    
    distances, indices, results = vector_search.search(query_embedding, k=20)
    
    print(f"\nSearch results for '{_safe_text(test_query)}':")
    for i, (dist, meta) in enumerate(zip(distances[0][:5], results[0][:5])):
        print(f"  {i+1}. Score: {dist:.4f} - {meta.get('source', 'N/A')} {meta.get('surah', '')}:{meta.get('ayah', '')} - {meta.get('type', '')}")

    print("\n" + "=" * 60)
    print("Phase 6 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
