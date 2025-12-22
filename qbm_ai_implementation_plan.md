# QBM AI System: Technical Implementation Plan
# مشروع الذكاء الاصطناعي لمصفوفة السلوك القرآني

> **Version**: 1.1 (Enhanced with Testing & Git Workflow)  
> **Last Updated**: 2025-12-22  
> **Status**: Planning Phase

---

## Tool Stack Decision

| Component | Option | **Selected** | Rationale |
|-----------|--------|--------------|----------|
| Graph DB | Neo4j Enterprise | **NetworkX + SQLite** | Zero cost, Python-native, sufficient for ~50K nodes |
| Vector DB | Pinecone | **ChromaDB** (local) | Free, local, good Arabic support |
| LLM | OpenAI / Local | **Azure OpenAI (GPT-5)** | Enterprise-grade, Arabic support, already configured |
| Embeddings | Various | **Azure OpenAI / AraBERT** | Arabic-optimized |
| Ontology | Protégé + commercial | **RDFLib + OWL** | Python-native, SPARQL support |

### Graph Database: NetworkX + SQLite Persistence

```python
# Why NetworkX over Neo4j:
# 1. FREE - no licensing costs
# 2. Python-native - no separate server
# 3. Rich algorithms - PageRank, community detection, path finding
# 4. Easy persistence - pickle, JSON, GraphML, or SQLite
# 5. Sufficient scale - handles 50K+ nodes easily

# Persistence strategy:
# - Development: pickle/JSON for quick iteration
# - Production: SQLite for durability + GraphML for portability
```

### LLM: Azure OpenAI Configuration

```python
# Using Azure OpenAI with GPT-5 deployment
# Environment variables (from .env):
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_VERSION
# - AZURE_OPENAI_DEPLOYMENT_NAME (gpt-5-chat)

from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Available deployments: gpt-5-chat, gpt-5.1, gpt-5.2
```

---

## Executive Summary

Transform QBM from a database query system into an **AI-powered behavioral discovery engine** that can:
- Find relationships no scholar has seen
- Reason across the entire Quran simultaneously  
- Cross-reference 5 tafsir sources automatically
- Discover emergent patterns in behavioral data

---

## Phase 1: Knowledge Graph (Weeks 1-3)

### 1.1 Graph Architecture

```
                    ┌─────────────┐
                    │   السورة    │
                    │   (Surah)   │
                    └──────┬──────┘
                           │ contains
                           ▼
                    ┌─────────────┐
                    │   الآية     │
                    │   (Ayah)    │
                    └──────┬──────┘
                           │ has_behavior
                           ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   الفاعل    │◄───│  السلوك    │───►│   العضو     │
│   (Agent)   │    │ (Behavior)  │    │  (Organ)    │
└─────────────┘    └──────┬──────┘    └─────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  المصدر  │    │  التقييم │    │ العاقبة  │
   │ (Source) │    │  (Eval)  │    │(Conseq.) │
   └──────────┘    └──────────┘    └──────────┘
```

### 1.2 Node Types

```python
NODE_TYPES = {
    # Core Entities
    "Surah": {"name_ar", "name_en", "number", "revelation_type"},  # مكي/مدني
    "Ayah": {"surah", "number", "text_uthmani", "text_simple"},
    "Span": {"start", "end", "text", "confidence"},
    
    # Behavioral Taxonomy
    "Behavior": {"code", "name_ar", "name_en", "category", "definition"},
    "BehaviorCategory": {"name_ar", "name_en", "parent"},  # Hierarchical
    
    # Contextual Dimensions
    "Agent": {"type", "name_ar"},  # مؤمن، كافر، منافق، شيطان...
    "Organ": {"name_ar", "name_en"},  # قلب، لسان، عين...
    "Source": {"name_ar", "type"},  # وحي، فطرة، نفس، شيطان...
    "Context": {"spatial", "temporal", "systemic", "situational"},
    
    # Heart/Personality
    "HeartType": {"name_ar", "personality", "characteristics"},
    
    # Scholarly Sources
    "Tafsir": {"source", "text", "scholar"},
    "Scholar": {"name", "era", "madhab"},
}
```

### 1.3 Edge Types (Relationships)

```python
EDGE_TYPES = {
    # Structural
    "CONTAINS": ("Surah", "Ayah"),
    "HAS_SPAN": ("Ayah", "Span"),
    "ANNOTATED_AS": ("Span", "Behavior"),
    
    # Behavioral Relationships
    "CAUSES": ("Behavior", "Behavior"),        # الغفلة → الكبر
    "RESULTS_IN": ("Behavior", "Behavior"),    # الكبر → الظلم
    "OPPOSITE_OF": ("Behavior", "Behavior"),   # الكبر ↔ التواضع
    "SIMILAR_TO": ("Behavior", "Behavior"),    # الكبر ~ العُجب
    "LEADS_TO": ("Behavior", "Consequence"),   # الكبر → جهنم
    
    # Contextual
    "PERFORMED_BY": ("Span", "Agent"),
    "INVOLVES_ORGAN": ("Span", "Organ"),
    "SOURCED_FROM": ("Behavior", "Source"),
    "EVALUATED_AS": ("Span", "Evaluation"),
    
    # Heart-Personality-Behavior
    "CHARACTERISTIC_OF": ("Behavior", "HeartType"),
    "MANIFESTS_AS": ("HeartType", "Behavior"),
    
    # Tafsir
    "EXPLAINED_BY": ("Ayah", "Tafsir"),
    "AUTHORED_BY": ("Tafsir", "Scholar"),
    
    # Co-occurrence
    "CO_OCCURS_WITH": ("Behavior", "Behavior"),  # Same ayah
    "CONTRASTED_WITH": ("Behavior", "Behavior"),  # Same context, opposite eval
}
```

### 1.4 Implementation (NetworkX + SQLite)

```python
# src/ai/graph/qbm_graph.py
import networkx as nx
import sqlite3
import json
from pathlib import Path

class QBMKnowledgeGraph:
    """Knowledge graph using NetworkX with SQLite persistence."""
    
    def __init__(self, db_path: str = "data/qbm_graph.db"):
        self.G = nx.MultiDiGraph()  # Directed graph with multiple edge types
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite tables for persistence."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    properties JSON NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    properties JSON,
                    PRIMARY KEY (source, target, edge_type)
                );
                CREATE INDEX IF NOT EXISTS idx_node_type ON nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type);
            """)
    
    def add_behavior(self, code: str, name_ar: str, name_en: str, category: str):
        """Add a behavior node."""
        self.G.add_node(code, node_type="Behavior", name_ar=name_ar, 
                        name_en=name_en, category=category)
    
    def add_relationship(self, source: str, target: str, rel_type: str, **props):
        """Add a typed edge between nodes."""
        self.G.add_edge(source, target, edge_type=rel_type, **props)
    
    def find_causal_chain(self, start: str, end: str, max_depth: int = 5):
        """Find all causal paths between two behaviors."""
        paths = []
        for path in nx.all_simple_paths(self.G, start, end, cutoff=max_depth):
            # Filter to only CAUSES/RESULTS_IN edges
            if self._is_causal_path(path):
                paths.append(path)
        return paths
    
    def get_hub_behaviors(self, top_n: int = 10):
        """Find most connected behaviors using betweenness centrality."""
        centrality = nx.betweenness_centrality(self.G)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def find_communities(self):
        """Detect behavioral clusters using Louvain algorithm."""
        # Convert to undirected for community detection
        undirected = self.G.to_undirected()
        return list(nx.community.louvain_communities(undirected))
    
    def save(self):
        """Persist graph to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Save nodes
            for node, attrs in self.G.nodes(data=True):
                conn.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?)",
                    (node, attrs.get('node_type', 'Unknown'), json.dumps(attrs))
                )
            # Save edges
            for u, v, attrs in self.G.edges(data=True):
                conn.execute(
                    "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?)",
                    (u, v, attrs.get('edge_type', 'RELATED'), json.dumps(attrs))
                )
    
    def load(self):
        """Load graph from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Load nodes
            for row in conn.execute("SELECT id, properties FROM nodes"):
                self.G.add_node(row[0], **json.loads(row[1]))
            # Load edges
            for row in conn.execute("SELECT source, target, properties FROM edges"):
                props = json.loads(row[2]) if row[2] else {}
                self.G.add_edge(row[0], row[1], **props)

# Example usage:
# graph = QBMKnowledgeGraph()
# graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
# graph.add_relationship("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES")
# chains = graph.find_causal_chain("BEH_COG_HEEDLESSNESS", "CONSEQUENCE_HELLFIRE")
```

### 1.5 Graph Analytics

```python
# Centrality Analysis - Which behaviors are most connected?
from networkx import pagerank, betweenness_centrality

# Find "hub" behaviors that connect many others
centrality = betweenness_centrality(G)
# Result: الإيمان, الكفر, النفاق are hubs

# Community Detection - Behavioral clusters
from networkx.algorithms import community
communities = community.louvain_communities(G)
# Result: Discovers natural groupings of related behaviors

# Path Analysis - Behavioral chains
all_paths = nx.all_simple_paths(G, 'الغفلة', 'جهنم', cutoff=5)
# Result: الغفلة → الكبر → الظلم → الختم → جهنم
```

---

## Phase 2: Vector Embeddings & RAG (Weeks 4-6)

### 2.1 Arabic Embedding Strategy

```python
EMBEDDING_MODELS = {
    # Option 1: Arabic-specific (RECOMMENDED)
    "arabert": "aubmindlab/bert-base-arabertv2",  # 768 dim
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    
    # Option 2: Multilingual
    "multilingual-e5": "intfloat/multilingual-e5-large",  # 1024 dim
    
    # Option 3: Fine-tuned for Quran (BEST - requires training)
    "qbm-embed": "your-org/qbm-arabic-embeddings",  # Custom
}
```

### 2.2 What to Embed

```python
EMBEDDING_TARGETS = {
    # Text Embeddings
    "ayah_text": "Full ayah text with tashkeel",
    "span_text": "Annotated span text",
    "tafsir_text": "Tafsir explanation for each ayah",
    
    # Concept Embeddings
    "behavior_definition": "Definition + examples of each behavior",
    "behavior_context": "Behavior + all its contextual dimensions",
    
    # Composite Embeddings
    "ayah_behavioral": "Ayah text + all behaviors + agents + organs",
    "cross_tafsir": "Combined tafsir from all 5 sources for each ayah",
}
```

### 2.3 Vector Database (ChromaDB - Free Alternative)

```python
# Using ChromaDB (FREE, local, persistent) instead of Qdrant/Pinecone
# src/ai/vectors/qbm_vectors.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class QBMVectorStore:
    """Vector store using ChromaDB with Arabic embeddings."""
    
    def __init__(self, persist_dir: str = "data/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = SentenceTransformer("aubmindlab/bert-base-arabertv2")
        
        # Create collections
        self.ayat = self.client.get_or_create_collection(
            name="qbm_ayat",
            metadata={"hnsw:space": "cosine"}
        )
        self.behaviors = self.client.get_or_create_collection(
            name="qbm_behaviors",
            metadata={"hnsw:space": "cosine"}
        )
        self.tafsir = self.client.get_or_create_collection(
            name="qbm_tafsir",
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed(self, text: str) -> list:
        """Generate embedding for Arabic text."""
        return self.embedder.encode(text).tolist()
    
    def add_ayah(self, ayah_id: str, text: str, metadata: dict):
        """Add ayah to vector store."""
        self.ayat.add(
            ids=[ayah_id],
            embeddings=[self.embed(text)],
            metadatas=[metadata],
            documents=[text]
        )
    
    def search_similar(self, query: str, collection: str = "ayat", n: int = 10):
        """Semantic search across collection."""
        coll = getattr(self, collection)
        return coll.query(
            query_embeddings=[self.embed(query)],
            n_results=n
        )

# Example usage:
# store = QBMVectorStore()
# store.add_ayah("2:7", "ختم الله على قلوبهم...", {"surah": 2, "ayah": 7})
# results = store.search_similar("مرض القلب", n=10)
```

### 2.4 RAG Pipeline (Azure OpenAI)

```python
# src/ai/rag/qbm_rag.py
from typing import Dict, List
from openai import AzureOpenAI
import os

class QBMRAGPipeline:
    """RAG pipeline using ChromaDB + NetworkX + Azure OpenAI."""
    
    def __init__(self):
        from .vectors.qbm_vectors import QBMVectorStore
        from .graph.qbm_graph import QBMKnowledgeGraph
        
        self.vector_store = QBMVectorStore()
        self.graph = QBMKnowledgeGraph()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
    
    def query(self, question: str) -> Dict:
        """Full RAG query with graph expansion."""
        
        # Step 1: Retrieve relevant ayat
        ayat = self.vector_store.search_similar(question, "ayat", n=20)
        
        # Step 2: Retrieve relevant behaviors
        behaviors = self.vector_store.search_similar(question, "behaviors", n=10)
        
        # Step 3: Expand via graph (get related behaviors, causes, effects)
        behavior_ids = [b["id"] for b in behaviors["metadatas"][0]]
        expanded = self._expand_graph_context(behavior_ids)
        
        # Step 4: Retrieve tafsir for relevant ayat
        tafsir = self.vector_store.search_similar(question, "tafsir", n=15)
        
        # Step 5: Build context and generate response
        context = self._build_context(ayat, behaviors, expanded, tafsir)
        response = self._generate(question, context)
        
        return {
            "answer": response,
            "sources": ayat["ids"][0],
            "behaviors": behavior_ids,
            "graph_expansion": expanded,
        }
    
    def _expand_graph_context(self, behavior_ids: List[str]) -> Dict:
        """Get related behaviors from graph."""
        expanded = {"causes": [], "effects": [], "opposites": []}
        for bid in behavior_ids:
            # Get neighbors by edge type
            for neighbor in self.graph.G.predecessors(bid):
                edge_data = self.graph.G.get_edge_data(neighbor, bid)
                if edge_data and edge_data.get("edge_type") == "CAUSES":
                    expanded["causes"].append(neighbor)
            for neighbor in self.graph.G.successors(bid):
                edge_data = self.graph.G.get_edge_data(bid, neighbor)
                if edge_data:
                    if edge_data.get("edge_type") == "RESULTS_IN":
                        expanded["effects"].append(neighbor)
                    elif edge_data.get("edge_type") == "OPPOSITE_OF":
                        expanded["opposites"].append(neighbor)
        return expanded
    
    def _build_context(self, ayat, behaviors, expanded, tafsir) -> str:
        """Build context string for LLM."""
        context_parts = [
            "## الآيات ذات الصلة:",
            "\n".join(ayat["documents"][0][:10]),
            "\n## السلوكيات:",
            "\n".join(behaviors["documents"][0][:5]),
            "\n## التفسير:",
            "\n".join(tafsir["documents"][0][:5]),
        ]
        if expanded["causes"]:
            context_parts.append(f"\n## الأسباب: {', '.join(expanded['causes'])}")
        if expanded["effects"]:
            context_parts.append(f"\n## النتائج: {', '.join(expanded['effects'])}")
        return "\n".join(context_parts)
    
    def _generate(self, question: str, context: str) -> str:
        """Generate response using Azure OpenAI."""
        system_prompt = """أنت عالم متخصص في تحليل السلوك القرآني وفق منهجية مصفوفة السلوك القرآني (QBM).
أجب بناءً على السياق المقدم فقط. استخدم الآيات والتفسير لدعم إجابتك."""
        
        user_prompt = f"""السياق:
{context}

السؤال: {question}"""
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
```

### 2.5 Semantic Search Examples

```python
# Example 1: Find behaviors semantically similar to الكبر
similar = vector_db.search(
    collection="qbm_behaviors",
    query_vector=embed("الكبر"),
    limit=10
)
# Returns: العُجب, الغرور, التكبر, الفخر, الخيلاء...

# Example 2: Find ayat about heart disease (even without exact words)
ayat = vector_db.search(
    collection="qbm_ayat", 
    query_vector=embed("مرض القلب والنفاق"),
    limit=20
)
# Returns ayat about قلوبهم مرض even if query doesn't match exactly

# Example 3: Cross-tafsir semantic search
tafsir = vector_db.search(
    collection="qbm_tafsir",
    query_vector=embed("علاقة الكبر بالقلب"),
    filter={"sources": ["ibn_kathir", "qurtubi", "tabari"]},
    limit=30
)
```

---

## Phase 3: Taxonomy & Ontology (Weeks 7-8)

### 3.1 Behavioral Taxonomy Structure

```
السلوكيات القرآنية
├── السلوكيات القلبية (Heart Behaviors)
│   ├── الإيمانية (Faith-related)
│   │   ├── الإيمان بالله
│   │   ├── الإيمان بالملائكة
│   │   ├── الإيمان بالكتب
│   │   ├── الإيمان بالرسل
│   │   ├── الإيمان باليوم الآخر
│   │   └── الإيمان بالقدر
│   ├── العاطفية (Emotional)
│   │   ├── الخوف
│   │   │   ├── خوف الله (ممدوح)
│   │   │   └── خوف الناس (مذموم)
│   │   ├── الرجاء
│   │   ├── المحبة
│   │   ├── الحزن
│   │   └── الفرح
│   └── النفسية (Psychological)
│       ├── الكبر
│       ├── الحسد
│       ├── الغيرة
│       └── العُجب
├── السلوكيات اللسانية (Tongue Behaviors)
│   ├── الإيجابية
│   │   ├── الصدق
│   │   ├── الذكر
│   │   └── الدعاء
│   └── السلبية
│       ├── الكذب
│       ├── الغيبة
│       └── النميمة
├── السلوكيات الجسدية (Physical Behaviors)
│   ├── العبادات
│   │   ├── الصلاة
│   │   ├── الصيام
│   │   └── الحج
│   └── المعاملات
│       ├── الصدقة
│       ├── السرقة
│       └── الزنا
└── السلوكيات العلائقية (Relational Behaviors)
    ├── الأسرية
    │   ├── بر الوالدين
    │   └── صلة الرحم
    └── المجتمعية
        ├── الأمانة
        └── العدل
```

### 3.2 Ontology Definition (OWL/RDF)

```turtle
@prefix qbm: <http://qbm.research/ontology#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Classes
qbm:Behavior a rdfs:Class .
qbm:HeartBehavior rdfs:subClassOf qbm:Behavior .
qbm:TongueBehavior rdfs:subClassOf qbm:Behavior .
qbm:PhysicalBehavior rdfs:subClassOf qbm:Behavior .

# Properties
qbm:hasCause a rdf:Property ;
    rdfs:domain qbm:Behavior ;
    rdfs:range qbm:Behavior .

qbm:hasConsequence a rdf:Property ;
    rdfs:domain qbm:Behavior ;
    rdfs:range qbm:Consequence .

qbm:associatedWithHeart a rdf:Property ;
    rdfs:domain qbm:Behavior ;
    rdfs:range qbm:HeartType .

# Instances
qbm:Pride a qbm:HeartBehavior ;
    qbm:nameAr "الكبر" ;
    qbm:hasCause qbm:Heedlessness ;
    qbm:hasConsequence qbm:Hellfire ;
    qbm:oppositeOf qbm:Humility ;
    qbm:associatedWithHeart qbm:DeadHeart .
```

### 3.3 Reasoning Rules

```python
REASONING_RULES = {
    # Transitivity
    "causal_chain": """
        IF behavior_a CAUSES behavior_b
        AND behavior_b CAUSES behavior_c
        THEN behavior_a INDIRECTLY_CAUSES behavior_c
    """,
    
    # Heart-Behavior inference
    "heart_behavior": """
        IF behavior CHARACTERISTIC_OF heart_type
        AND person HAS heart_type
        THEN person LIKELY_EXHIBITS behavior
    """,
    
    # Evaluation propagation
    "eval_chain": """
        IF behavior_a CAUSES behavior_b
        AND behavior_b EVALUATED_AS blame
        THEN behavior_a CONTRIBUTES_TO blame
    """,
    
    # Opposite inference
    "opposite_eval": """
        IF behavior_a OPPOSITE_OF behavior_b
        AND behavior_a EVALUATED_AS praise
        THEN behavior_b EVALUATED_AS blame
    """,
}
```

---

## Phase 4: Multi-Tafsir Integration (Weeks 9-11)

### 4.1 Tafsir Sources

```python
TAFSIR_SOURCES = {
    "ibn_kathir": {
        "name_ar": "تفسير ابن كثير",
        "scholar": "إسماعيل بن عمر بن كثير",
        "era": "774 هـ",
        "methodology": "بالمأثور",
        "focus": "الحديث والأثر",
        "status": "COMPLETE",  # You have this
    },
    "tabari": {
        "name_ar": "جامع البيان",
        "scholar": "محمد بن جرير الطبري",
        "era": "310 هـ",
        "methodology": "بالمأثور",
        "focus": "الأقوال والروايات",
        "status": "TO_ADD",
    },
    "qurtubi": {
        "name_ar": "الجامع لأحكام القرآن",
        "scholar": "محمد بن أحمد القرطبي",
        "era": "671 هـ",
        "methodology": "بالرأي",
        "focus": "الأحكام الفقهية",
        "status": "TO_ADD",
    },
    "saadi": {
        "name_ar": "تيسير الكريم الرحمن",
        "scholar": "عبد الرحمن السعدي",
        "era": "1376 هـ",
        "methodology": "بالرأي",
        "focus": "المعاني والهدايات",
        "status": "TO_ADD",
    },
    "jalalayn": {
        "name_ar": "تفسير الجلالين",
        "scholar": "جلال الدين المحلي والسيوطي",
        "era": "911 هـ",
        "methodology": "مختصر",
        "focus": "الإيجاز",
        "status": "TO_ADD",
    },
}
```

### 4.2 Cross-Tafsir Analysis

```python
class CrossTafsirAnalyzer:
    """Analyze behavioral insights across multiple tafsir sources"""
    
    def find_consensus(self, surah: int, ayah: int, behavior: str) -> Dict:
        """Find where all scholars agree about a behavior"""
        tafsirs = self.get_all_tafsir(surah, ayah)
        
        consensus = {
            "agree_on": [],
            "disagree_on": [],
            "unique_insights": {},
        }
        
        for aspect in ["evaluation", "cause", "consequence", "related"]:
            mentions = [t.get(aspect) for t in tafsirs.values()]
            if all_same(mentions):
                consensus["agree_on"].append(aspect)
            else:
                consensus["disagree_on"].append(aspect)
        
        # Find unique insights per scholar
        for scholar, tafsir in tafsirs.items():
            unique = find_unique_mentions(tafsir, tafsirs)
            if unique:
                consensus["unique_insights"][scholar] = unique
        
        return consensus
    
    def behavioral_emphasis(self, behavior: str) -> Dict:
        """Which scholars emphasize which aspects of a behavior"""
        results = {}
        for scholar in TAFSIR_SOURCES:
            mentions = self.search_behavior_in_tafsir(behavior, scholar)
            results[scholar] = {
                "frequency": len(mentions),
                "contexts": extract_contexts(mentions),
                "emphasis": extract_emphasis(mentions),
            }
        return results
```

### 4.3 Tafsir Data Schema

```sql
CREATE TABLE tafsir_entries (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    source VARCHAR(50) NOT NULL,  -- ibn_kathir, tabari, etc.
    text_ar TEXT NOT NULL,
    
    -- Extracted behavioral metadata
    behaviors_mentioned TEXT[],
    agents_mentioned TEXT[],
    evaluations TEXT[],
    
    -- Embeddings
    embedding VECTOR(1024),
    
    FOREIGN KEY (surah, ayah) REFERENCES ayat(surah, ayah)
);

CREATE INDEX idx_tafsir_behavior ON tafsir_entries USING GIN(behaviors_mentioned);
CREATE INDEX idx_tafsir_embedding ON tafsir_entries USING ivfflat(embedding);
```

---

## Phase 5: Complete Annotation (Weeks 12-16)

### 5.1 Current State Assessment

```python
ANNOTATION_STATUS = {
    "total_ayat": 6236,
    "annotated_ayat": 6236,  # All ayat have at least one annotation
    "total_spans": 15847,
    "behaviors_defined": 87,
    
    # What needs completion
    "needs_review": {
        "low_confidence_spans": "Spans with confidence < 0.8",
        "missing_dimensions": "Spans without full 11-dimension annotation",
        "single_annotator": "Spans with only 1 annotator agreement",
    },
    
    # What needs addition
    "needs_addition": {
        "related_behaviors": "Cause/effect/opposite relationships",
        "cross_references": "Intra-Quran references",
        "temporal_markers": "Dunya/Akhira/specific time context",
    },
}
```

### 5.2 Annotation Completion Plan

```python
ANNOTATION_TASKS = [
    {
        "task": "Add missing dimensions",
        "description": "For each span, ensure all 11 dimensions are annotated",
        "priority": "HIGH",
        "method": "Semi-automated with human review",
    },
    {
        "task": "Behavioral relationships",
        "description": "Add CAUSES, RESULTS_IN, OPPOSITE_OF edges",
        "priority": "HIGH", 
        "method": "Graph analysis + scholarly validation",
    },
    {
        "task": "Heart type mapping",
        "description": "Link each behavior to associated heart types",
        "priority": "HIGH",
        "method": "Rule-based + manual review",
    },
    {
        "task": "Consequence mapping",
        "description": "Link behaviors to worldly/hereafter consequences",
        "priority": "MEDIUM",
        "method": "Tafsir extraction + manual",
    },
    {
        "task": "Cross-references",
        "description": "Link related ayat discussing same behavior",
        "priority": "MEDIUM",
        "method": "Semantic similarity + manual validation",
    },
    {
        "task": "Confidence scoring",
        "description": "Add confidence scores based on tafsir agreement",
        "priority": "LOW",
        "method": "Cross-tafsir analysis",
    },
]
```

### 5.3 Semi-Automated Annotation

```python
class AnnotationAssistant:
    """AI-assisted annotation for missing dimensions"""
    
    def suggest_dimensions(self, span: Span) -> Dict:
        """Suggest missing dimensional annotations"""
        
        suggestions = {}
        
        # Use embeddings to find similar annotated spans
        similar = self.find_similar_spans(span, limit=10)
        
        # Use graph to infer from context
        context = self.get_ayah_context(span.surah, span.ayah)
        
        # Use tafsir to validate
        tafsir = self.get_tafsir_mentions(span)
        
        for dimension in DIMENSIONS:
            if not span.has_dimension(dimension):
                suggestions[dimension] = {
                    "from_similar": most_common([s.get(dimension) for s in similar]),
                    "from_context": infer_from_context(context, dimension),
                    "from_tafsir": extract_from_tafsir(tafsir, dimension),
                    "confidence": calculate_confidence(...),
                }
        
        return suggestions
```

---

## Phase 6: Model Fine-Tuning (Weeks 17-20)

### 6.1 Hardware Setup

```python
GPU_CLUSTER = {
    "gpus": 8,
    "model": "NVIDIA A100",
    "memory_per_gpu": "80GB",  # or 40GB
    "total_memory": "640GB",
    "interconnect": "NVLink",
    
    # Training capacity
    "max_model_size": "70B parameters",
    "recommended_model": "13B-30B for best quality/speed",
}
```

### 6.2 Model Selection

```python
BASE_MODELS = {
    # Arabic-First Models (RECOMMENDED)
    "jais-30b": {
        "params": "30B",
        "arabic_quality": "Excellent",
        "license": "Research",
        "fine_tune_memory": "~500GB with LoRA",
    },
    "acegpt-13b": {
        "params": "13B", 
        "arabic_quality": "Very Good",
        "license": "Apache 2.0",
        "fine_tune_memory": "~200GB with LoRA",
    },
    
    # Multilingual Options
    "llama3-70b": {
        "params": "70B",
        "arabic_quality": "Good",
        "license": "Meta License",
        "fine_tune_memory": "~600GB with QLoRA",
    },
    "qwen2-72b": {
        "params": "72B",
        "arabic_quality": "Very Good",
        "license": "Qwen License",
        "fine_tune_memory": "~600GB with QLoRA",
    },
}

RECOMMENDATION = "jais-30b or acegpt-13b for Arabic-first quality"
```

### 6.3 Training Data Preparation

```python
TRAINING_DATA = {
    # Instruction-Following Data
    "instruction_data": {
        "behavior_analysis": [
            {
                "instruction": "حلل سلوك الكبر في القرآن الكريم",
                "output": "تحليل شامل مع الآيات والأبعاد...",
            },
            # 10,000+ examples covering all question types
        ],
        "comparison": [...],
        "verse_analysis": [...],
        "personality_analysis": [...],
    },
    
    # Knowledge Grounding Data
    "knowledge_data": {
        "ayat_with_annotations": "6,236 ayat with full behavioral annotations",
        "tafsir_text": "5 tafsir sources, ~50M tokens",
        "behavioral_definitions": "87 behaviors with definitions and examples",
        "relationships": "Graph edges as text descriptions",
    },
    
    # Reasoning Data
    "reasoning_data": {
        "chain_of_thought": "Step-by-step behavioral analysis examples",
        "multi_hop": "Complex queries requiring graph traversal",
        "cross_reference": "Queries requiring multiple ayat",
    },
}
```

### 6.4 Training Strategy

```python
TRAINING_CONFIG = {
    # Stage 1: Continued Pre-training
    "stage1_pretraining": {
        "data": "All Quran + 5 Tafsir + Islamic texts",
        "objective": "Next token prediction",
        "epochs": 3,
        "purpose": "Deep Arabic Quranic understanding",
    },
    
    # Stage 2: Knowledge Injection
    "stage2_knowledge": {
        "data": "QBM database as structured text",
        "objective": "Knowledge grounding",
        "epochs": 5,
        "purpose": "Learn behavioral taxonomy and relationships",
    },
    
    # Stage 3: Instruction Fine-tuning
    "stage3_instruction": {
        "data": "10,000+ instruction-output pairs",
        "objective": "Instruction following",
        "epochs": 3,
        "method": "LoRA/QLoRA",
        "purpose": "Learn to answer systematically",
    },
    
    # Stage 4: RLHF (Optional)
    "stage4_rlhf": {
        "data": "Human preferences on responses",
        "objective": "Align with scholarly expectations",
        "purpose": "Improve quality and accuracy",
    },
}
```

### 6.5 Training Script

```python
# train_qbm.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "inception-mbzuai/jais-30b-v3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA config for efficient fine-tuning
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qbm-jais-30b",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    deepspeed="ds_config.json",  # For multi-GPU
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 6.6 Distributed Training Config

```json
// ds_config.json - DeepSpeed config for 8x A100
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
```

---

## Phase 7: Inference & Discovery System (Weeks 21-24)

### 7.1 Discovery Queries

```python
class QBMDiscoveryEngine:
    """AI system for discovering hidden patterns in Quranic behavior"""
    
    def discover_behavioral_chains(self, max_depth: int = 5) -> List[Chain]:
        """Find all causal chains in the behavioral graph"""
        chains = []
        for behavior in self.graph.get_all_behaviors():
            paths = self.graph.find_all_paths(
                start=behavior,
                edge_type="CAUSES",
                max_depth=max_depth
            )
            chains.extend(paths)
        return self.rank_by_significance(chains)
    
    def discover_hidden_clusters(self) -> List[Cluster]:
        """Find behaviors that co-occur but aren't formally related"""
        # Use graph community detection
        communities = self.graph.detect_communities()
        
        # Find unexpected groupings
        unexpected = []
        for community in communities:
            if not self.formally_related(community):
                unexpected.append({
                    "behaviors": community,
                    "co_occurrence_strength": self.calculate_strength(community),
                    "potential_relationship": self.infer_relationship(community),
                })
        return unexpected
    
    def cross_tafsir_discovery(self, behavior: str) -> Dict:
        """Find insights mentioned by only one scholar"""
        all_mentions = {}
        for scholar in TAFSIR_SOURCES:
            mentions = self.search_tafsir(behavior, scholar)
            all_mentions[scholar] = extract_insights(mentions)
        
        unique_insights = find_unique_across_scholars(all_mentions)
        return unique_insights
    
    def find_behavioral_anomalies(self) -> List[Anomaly]:
        """Find behaviors that break expected patterns"""
        anomalies = []
        
        # Behaviors praised for one agent, blamed for another
        eval_anomalies = self.find_evaluation_anomalies()
        
        # Behaviors with unexpected organ associations
        organ_anomalies = self.find_organ_anomalies()
        
        # Behaviors that appear in unexpected contexts
        context_anomalies = self.find_context_anomalies()
        
        return eval_anomalies + organ_anomalies + context_anomalies
```

### 7.2 Example Discoveries

```python
# What the system might discover:

EXAMPLE_DISCOVERIES = [
    {
        "type": "hidden_chain",
        "discovery": "الغفلة → حب الدنيا → الكبر → الظلم → الختم → جهنم",
        "evidence": ["البقرة: 7", "الإسراء: 37", "غافر: 35", "الزمر: 72"],
        "insight": "6-step chain from heedlessness to hellfire, through pride",
    },
    {
        "type": "unexpected_cluster",
        "discovery": "الغيبة and الحسد co-occur in 73% of ayat",
        "evidence": ["الحجرات: 12", "النساء: 54"],
        "insight": "Backbiting may be driven by envy more than other causes",
    },
    {
        "type": "cross_tafsir",
        "discovery": "Only Qurtubi links الكبر to political tyranny (الاستبداد)",
        "evidence": "Qurtubi on غافر: 35",
        "insight": "Legal/political dimension of pride unique to Qurtubi",
    },
    {
        "type": "pattern",
        "discovery": "Heart-organ correlation: 89% of قلب behaviors are داخلي",
        "evidence": "Statistical analysis",
        "insight": "Heart behaviors rarely manifest externally without intermediate",
    },
    {
        "type": "temporal_pattern",
        "discovery": "الصبر appears 3x more in مكي surahs than مدني",
        "evidence": "Distribution analysis",
        "insight": "Patience emphasized during persecution period",
    },
]
```

---

## Phase 8: Integration & Deployment (Weeks 25-28)

### 8.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QBM AI System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Frontend  │  │   API       │  │   AI Engine             │  │
│  │   Next.js   │──│   FastAPI   │──│   ┌─────────────────┐   │  │
│  │   C1/Thesys │  │             │  │   │ Fine-tuned LLM  │   │  │
│  └─────────────┘  └─────────────┘  │   │ (JAIS-30B)      │   │  │
│                                     │   └────────┬────────┘   │  │
│                                     │            │            │  │
│                                     │   ┌────────▼────────┐   │  │
│                                     │   │  RAG Pipeline   │   │  │
│                                     │   └────────┬────────┘   │  │
│                                     │            │            │  │
│                                     └────────────┼────────────┘  │
│                                                  │               │
│  ┌───────────────────────────────────────────────┼─────────────┐ │
│  │                    Data Layer                 │             │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐│             │ │
│  │  │ Neo4j    │  │ Qdrant   │  │ PostgreSQL   ││             │ │
│  │  │ (Graph)  │  │ (Vector) │  │ (Relational) ││             │ │
│  │  └──────────┘  └──────────┘  └──────────────┘│             │ │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 API Endpoints

```python
# api/main.py
from fastapi import FastAPI

app = FastAPI(title="QBM AI API")

@app.post("/analyze/behavior")
async def analyze_behavior(behavior: str, depth: str = "full"):
    """Full behavioral analysis using 5-step methodology"""
    pass

@app.post("/discover/chains")
async def discover_chains(start_behavior: str, max_depth: int = 5):
    """Discover causal chains from a behavior"""
    pass

@app.post("/compare/personalities")
async def compare_personalities(behavior: str):
    """Compare behavior across Believer/Munafiq/Kafir"""
    pass

@app.post("/search/semantic")
async def semantic_search(query: str, limit: int = 20):
    """Semantic search across Quran and Tafsir"""
    pass

@app.post("/cross-tafsir")
async def cross_tafsir_analysis(surah: int, ayah: int):
    """Analyze ayah across all 5 tafsir sources"""
    pass

@app.post("/discover/patterns")
async def discover_patterns(dimension: str):
    """AI-driven pattern discovery"""
    pass
```

---

## Timeline Summary (Updated with Free Tools)

| Phase | Weeks | Deliverable | Git Tag |
|-------|-------|-------------|----------|
| 1. Knowledge Graph | 1-3 | NetworkX + SQLite graph | `v0.1.0-graph` |
| 2. Vectors & RAG | 4-6 | ChromaDB + RAG pipeline | `v0.2.0-rag` |
| 3. Taxonomy | 7-8 | RDFLib ontology + rules | `v0.3.0-ontology` |
| 4. Multi-Tafsir | 9-11 | 5 tafsir sources integrated | `v0.4.0-tafsir` |
| 5. Complete Annotation | 12-16 | Full 11-dimension annotations | `v0.5.0-annotations` |
| 6. Model Training | 17-20 | Fine-tuned via Azure OpenAI | `v0.6.0-model` |
| 7. Discovery System | 21-24 | Pattern discovery engine | `v0.7.0-discovery` |
| 8. Integration | 25-28 | Production deployment | `v1.0.0` |

**Total: 28 weeks (~7 months)**

---

## Budget Estimate

| Item | Cost | Notes |
|------|------|-------|
| Graph DB | $0 | NetworkX + SQLite |
| Vector DB | $0 | ChromaDB local |
| LLM Inference | Azure OpenAI | Already provisioned (gpt-5-chat, gpt-5.1, gpt-5.2) |
| Embeddings | $0 | AraBERT / Azure OpenAI |
| Tafsir Data | $0 | quran.com API |
| **Infrastructure** | **Existing Azure** | **Enterprise subscription** |

---

## What This System Will Discover

Things no single scholar could find:

1. **Complete causal chains** across 6,236 ayat
2. **Statistical patterns** in behavioral distribution
3. **Cross-tafsir consensus and disagreement** instantly
4. **Hidden clusters** of co-occurring behaviors
5. **Temporal patterns** in Makki vs Madani
6. **Heart-behavior correlations** with statistical confidence
7. **Multi-hop relationships** through graph traversal
8. **Semantic similarities** across different terminology

---

## Final Note

The key insight: **Scholars have depth, AI has breadth.**

A scholar might spend a lifetime studying الكبر in deep detail.
Your AI system can analyze الكبر across:
- All 6,236 ayat simultaneously
- All 5 tafsir sources
- All relationships to other behaviors
- All personality types
- All temporal/spatial contexts

And then do the same for all 87 behaviors in seconds.

**The combination of scholarly depth (Bouzidani framework) + AI breadth (your system) = discoveries neither could make alone.**

---

## Phase Testing Requirements

### Phase 1: Knowledge Graph - Test Suite

```python
# tests/ai/test_knowledge_graph.py
import pytest
from src.ai.graph.qbm_graph import QBMKnowledgeGraph

class TestKnowledgeGraph:
    """Test suite for Phase 1: Knowledge Graph."""
    
    @pytest.fixture
    def graph(self, tmp_path):
        """Create test graph with temp database."""
        return QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))
    
    # --- Node Tests ---
    def test_add_behavior_node(self, graph):
        """Test adding a behavior node."""
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        assert "BEH_COG_ARROGANCE" in graph.G.nodes
        assert graph.G.nodes["BEH_COG_ARROGANCE"]["name_ar"] == "الكبر"
    
    def test_add_all_vocab_behaviors(self, graph):
        """Test loading all 87 behaviors from vocab."""
        import json
        with open("vocab/behavior_concepts.json") as f:
            vocab = json.load(f)
        for category, behaviors in vocab["categories"].items():
            for b in behaviors:
                graph.add_behavior(b["id"], b["ar"], b["en"], category)
        assert len(graph.G.nodes) >= 87
    
    # --- Edge Tests ---
    def test_add_causal_relationship(self, graph):
        """Test adding CAUSES edge."""
        graph.add_behavior("BEH_COG_HEEDLESSNESS", "الغفلة", "Heedlessness", "cognitive")
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_relationship("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES")
        
        edges = list(graph.G.edges(data=True))
        assert len(edges) == 1
        assert edges[0][2]["edge_type"] == "CAUSES"
    
    def test_opposite_relationship_bidirectional(self, graph):
        """Test OPPOSITE_OF creates bidirectional edge."""
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_COG_HUMILITY", "التواضع", "Humility", "cognitive")
        graph.add_relationship("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY", "OPPOSITE_OF")
        graph.add_relationship("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE", "OPPOSITE_OF")
        
        assert graph.G.has_edge("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY")
        assert graph.G.has_edge("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE")
    
    # --- Path Finding Tests ---
    def test_find_causal_chain(self, graph):
        """Test finding causal chain between behaviors."""
        # Build chain: الغفلة → الكبر → الظلم
        graph.add_behavior("BEH_COG_HEEDLESSNESS", "الغفلة", "Heedlessness", "cognitive")
        graph.add_behavior("BEH_COG_ARROGANCE", "الكبر", "Arrogance", "cognitive")
        graph.add_behavior("BEH_SOC_OPPRESSION", "الظلم", "Oppression", "social")
        graph.add_relationship("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE", "CAUSES")
        graph.add_relationship("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION", "RESULTS_IN")
        
        chains = graph.find_causal_chain("BEH_COG_HEEDLESSNESS", "BEH_SOC_OPPRESSION")
        assert len(chains) >= 1
        assert "BEH_COG_ARROGANCE" in chains[0]
    
    # --- Analytics Tests ---
    def test_hub_behaviors_centrality(self, graph):
        """Test finding hub behaviors."""
        # Create hub structure
        graph.add_behavior("HUB", "مركز", "Hub", "test")
        for i in range(5):
            graph.add_behavior(f"SPOKE_{i}", f"فرع_{i}", f"Spoke_{i}", "test")
            graph.add_relationship("HUB", f"SPOKE_{i}", "RELATED")
        
        hubs = graph.get_hub_behaviors(top_n=1)
        assert hubs[0][0] == "HUB"
    
    def test_community_detection(self, graph):
        """Test behavioral clustering."""
        # Create two distinct clusters
        for i in range(3):
            graph.add_behavior(f"CLUSTER_A_{i}", f"أ_{i}", f"A_{i}", "test")
        for i in range(3):
            graph.add_behavior(f"CLUSTER_B_{i}", f"ب_{i}", f"B_{i}", "test")
        
        # Connect within clusters
        graph.add_relationship("CLUSTER_A_0", "CLUSTER_A_1", "RELATED")
        graph.add_relationship("CLUSTER_A_1", "CLUSTER_A_2", "RELATED")
        graph.add_relationship("CLUSTER_B_0", "CLUSTER_B_1", "RELATED")
        graph.add_relationship("CLUSTER_B_1", "CLUSTER_B_2", "RELATED")
        
        communities = graph.find_communities()
        assert len(communities) >= 2
    
    # --- Persistence Tests ---
    def test_save_and_load(self, graph, tmp_path):
        """Test graph persistence to SQLite."""
        graph.add_behavior("BEH_TEST", "اختبار", "Test", "test")
        graph.add_relationship("BEH_TEST", "BEH_TEST", "SELF_REF")
        graph.save()
        
        # Create new graph and load
        graph2 = QBMKnowledgeGraph(db_path=str(tmp_path / "test_graph.db"))
        graph2.load()
        
        assert "BEH_TEST" in graph2.G.nodes
        assert graph2.G.has_edge("BEH_TEST", "BEH_TEST")


# Run: pytest tests/ai/test_knowledge_graph.py -v
```

### Phase 2: Vector Embeddings & RAG - Test Suite

```python
# tests/ai/test_vectors.py
import pytest
from src.ai.vectors.qbm_vectors import QBMVectorStore

class TestVectorStore:
    """Test suite for Phase 2: Vector Embeddings."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create test vector store."""
        return QBMVectorStore(persist_dir=str(tmp_path / "chromadb"))
    
    # --- Embedding Tests ---
    def test_embed_arabic_text(self, store):
        """Test Arabic text embedding."""
        embedding = store.embed("الكبر من أمراض القلب")
        assert len(embedding) == 768  # AraBERT dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_consistency(self, store):
        """Test same text produces same embedding."""
        text = "ختم الله على قلوبهم"
        emb1 = store.embed(text)
        emb2 = store.embed(text)
        assert emb1 == emb2
    
    # --- Collection Tests ---
    def test_add_ayah(self, store):
        """Test adding ayah to collection."""
        store.add_ayah(
            ayah_id="2:7",
            text="ختم الله على قلوبهم وعلى سمعهم",
            metadata={"surah": 2, "ayah": 7, "behaviors": ["BEH_SEAL"]}
        )
        assert store.ayat.count() == 1
    
    def test_add_multiple_ayat(self, store):
        """Test batch adding ayat."""
        ayat = [
            ("2:6", "إن الذين كفروا سواء عليهم", {"surah": 2, "ayah": 6}),
            ("2:7", "ختم الله على قلوبهم", {"surah": 2, "ayah": 7}),
            ("2:8", "ومن الناس من يقول آمنا", {"surah": 2, "ayah": 8}),
        ]
        for aid, text, meta in ayat:
            store.add_ayah(aid, text, meta)
        assert store.ayat.count() == 3
    
    # --- Search Tests ---
    def test_semantic_search_arabic(self, store):
        """Test semantic search returns relevant results."""
        # Add test data
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2, "ayah": 7})
        store.add_ayah("2:10", "في قلوبهم مرض", {"surah": 2, "ayah": 10})
        store.add_ayah("3:14", "زين للناس حب الشهوات", {"surah": 3, "ayah": 14})
        
        # Search for heart-related
        results = store.search_similar("أمراض القلب", "ayat", n=2)
        
        # Should return heart-related ayat first
        assert "2:7" in results["ids"][0] or "2:10" in results["ids"][0]
    
    def test_search_with_filter(self, store):
        """Test filtered search by metadata."""
        store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2, "ayah": 7})
        store.add_ayah("3:7", "هو الذي أنزل عليك الكتاب", {"surah": 3, "ayah": 7})
        
        results = store.ayat.query(
            query_embeddings=[store.embed("القلب")],
            n_results=10,
            where={"surah": 2}
        )
        assert all(m["surah"] == 2 for m in results["metadatas"][0])


# tests/ai/test_rag.py
import pytest
from unittest.mock import patch, MagicMock

class TestRAGPipeline:
    """Test suite for Phase 2: RAG Pipeline."""
    
    @pytest.fixture
    def rag(self, tmp_path):
        """Create test RAG pipeline with mocked LLM."""
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "Test response"}}
            from src.ai.rag.qbm_rag import QBMRAGPipeline
            return QBMRAGPipeline()
    
    def test_query_returns_structured_response(self, rag):
        """Test RAG query returns expected structure."""
        # Add test data first
        rag.vector_store.add_ayah("2:7", "ختم الله على قلوبهم", {"surah": 2})
        
        result = rag.query("ما هو الكبر؟")
        
        assert "answer" in result
        assert "sources" in result
        assert "behaviors" in result
        assert "graph_expansion" in result
    
    def test_graph_expansion(self, rag):
        """Test graph context expansion."""
        # Setup graph with relationships
        rag.graph.add_behavior("BEH_A", "أ", "A", "test")
        rag.graph.add_behavior("BEH_B", "ب", "B", "test")
        rag.graph.add_relationship("BEH_A", "BEH_B", "CAUSES")
        
        expanded = rag._expand_graph_context(["BEH_B"])
        assert "BEH_A" in expanded["causes"]


# Run: pytest tests/ai/test_vectors.py tests/ai/test_rag.py -v
```

### Phase 3: Taxonomy & Ontology - Test Suite

```python
# tests/ai/test_ontology.py
import pytest
from rdflib import Graph, Namespace, RDF, RDFS

class TestOntology:
    """Test suite for Phase 3: Taxonomy & Ontology."""
    
    @pytest.fixture
    def ontology(self):
        """Load QBM ontology."""
        g = Graph()
        g.parse("data/ontology/qbm_ontology.ttl", format="turtle")
        return g
    
    # --- Structure Tests ---
    def test_behavior_class_exists(self, ontology):
        """Test Behavior class is defined."""
        QBM = Namespace("http://qbm.research/ontology#")
        assert (QBM.Behavior, RDF.type, RDFS.Class) in ontology
    
    def test_behavior_subclasses(self, ontology):
        """Test behavior subclasses exist."""
        QBM = Namespace("http://qbm.research/ontology#")
        subclasses = ["HeartBehavior", "TongueBehavior", "PhysicalBehavior"]
        for sc in subclasses:
            assert (QBM[sc], RDFS.subClassOf, QBM.Behavior) in ontology
    
    def test_all_vocab_behaviors_in_ontology(self, ontology):
        """Test all 87 behaviors have ontology entries."""
        import json
        with open("vocab/behavior_concepts.json") as f:
            vocab = json.load(f)
        
        QBM = Namespace("http://qbm.research/ontology#")
        for category, behaviors in vocab["categories"].items():
            for b in behaviors:
                # Check behavior instance exists
                query = f"""
                    ASK {{ ?b qbm:behaviorId "{b['id']}" }}
                """
                assert ontology.query(query)
    
    # --- Relationship Tests ---
    def test_causes_property(self, ontology):
        """Test hasCause property is defined."""
        QBM = Namespace("http://qbm.research/ontology#")
        assert (QBM.hasCause, RDF.type, RDF.Property) in ontology
    
    def test_opposite_relationship(self, ontology):
        """Test oppositeOf relationships."""
        query = """
            SELECT ?a ?b WHERE {
                ?a qbm:oppositeOf ?b .
            }
        """
        results = list(ontology.query(query))
        assert len(results) > 0  # Should have opposite pairs
    
    # --- Reasoning Tests ---
    def test_transitive_causes(self, ontology):
        """Test transitive causal inference."""
        # If A causes B and B causes C, then A indirectly causes C
        query = """
            SELECT ?a ?c WHERE {
                ?a qbm:hasCause ?b .
                ?b qbm:hasCause ?c .
            }
        """
        results = list(ontology.query(query))
        # Verify transitive chains exist
        assert len(results) >= 0  # Will have results after data population


# Run: pytest tests/ai/test_ontology.py -v
```

### Phase 4: Multi-Tafsir - Test Suite

```python
# tests/ai/test_tafsir.py
import pytest

class TestTafsirIntegration:
    """Test suite for Phase 4: Multi-Tafsir Integration."""
    
    @pytest.fixture
    def analyzer(self):
        """Create tafsir analyzer."""
        from src.ai.tafsir.cross_tafsir import CrossTafsirAnalyzer
        return CrossTafsirAnalyzer()
    
    # --- Data Availability Tests ---
    def test_all_five_sources_available(self, analyzer):
        """Test all 5 tafsir sources are loaded."""
        sources = analyzer.get_available_sources()
        expected = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
        for src in expected:
            assert src in sources
    
    def test_tafsir_coverage(self, analyzer):
        """Test tafsir exists for all 6236 ayat."""
        for surah in range(1, 115):
            ayat_count = analyzer.get_ayat_count(surah)
            for ayah in range(1, ayat_count + 1):
                tafsirs = analyzer.get_all_tafsir(surah, ayah)
                assert len(tafsirs) >= 1, f"Missing tafsir for {surah}:{ayah}"
    
    # --- Consensus Tests ---
    def test_find_consensus(self, analyzer):
        """Test consensus finding across scholars."""
        consensus = analyzer.find_consensus(2, 7, "الختم")
        
        assert "agree_on" in consensus
        assert "disagree_on" in consensus
        assert "unique_insights" in consensus
    
    def test_behavioral_emphasis(self, analyzer):
        """Test scholar emphasis analysis."""
        emphasis = analyzer.behavioral_emphasis("الكبر")
        
        for scholar in ["ibn_kathir", "qurtubi"]:
            assert scholar in emphasis
            assert "frequency" in emphasis[scholar]
            assert "contexts" in emphasis[scholar]
    
    # --- Search Tests ---
    def test_search_behavior_in_tafsir(self, analyzer):
        """Test searching for behavior mentions."""
        results = analyzer.search_behavior_in_tafsir("الكبر", "ibn_kathir")
        
        assert len(results) > 0
        # الكبر should appear in multiple tafsir entries
        assert any("كبر" in r["text"] for r in results)


# Run: pytest tests/ai/test_tafsir.py -v
```

### Phase 5-8: Integration Tests

```python
# tests/ai/test_integration.py
import pytest

class TestEndToEndIntegration:
    """Integration tests for complete AI system."""
    
    @pytest.fixture
    def system(self):
        """Initialize complete QBM AI system."""
        from src.ai.qbm_ai import QBMAISystem
        return QBMAISystem()
    
    # --- End-to-End Query Tests ---
    def test_behavioral_analysis_query(self, system):
        """Test full behavioral analysis."""
        result = system.analyze_behavior("الكبر")
        
        assert "definition" in result
        assert "ayat" in result
        assert "causes" in result
        assert "effects" in result
        assert "tafsir_insights" in result
    
    def test_discovery_query(self, system):
        """Test pattern discovery."""
        chains = system.discover_chains("الغفلة", max_depth=5)
        
        assert len(chains) > 0
        # Should find path to negative consequences
    
    def test_cross_tafsir_query(self, system):
        """Test cross-tafsir analysis."""
        result = system.cross_tafsir_analysis(2, 7)
        
        assert len(result["sources"]) == 5
        assert "consensus" in result
    
    # --- Performance Tests ---
    def test_query_latency(self, system):
        """Test query completes within acceptable time."""
        import time
        
        start = time.time()
        system.query("ما هي أسباب الكبر؟")
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # Should complete within 10 seconds
    
    def test_graph_traversal_performance(self, system):
        """Test graph operations are fast."""
        import time
        
        start = time.time()
        system.graph.find_causal_chain("BEH_COG_HEEDLESSNESS", "CONSEQUENCE_HELLFIRE")
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Graph ops should be sub-second


# Run: pytest tests/ai/test_integration.py -v
```

---

## Git Workflow & Commit Strategy

### Branch Strategy

```
main (production)
  └── develop (integration)
        ├── feature/phase1-knowledge-graph
        ├── feature/phase2-vectors-rag
        ├── feature/phase3-ontology
        ├── feature/phase4-tafsir
        ├── feature/phase5-annotations
        ├── feature/phase6-model-training
        ├── feature/phase7-discovery
        └── feature/phase8-integration
```

### Phase 1 Git Commands

```bash
# Start Phase 1
git checkout develop
git checkout -b feature/phase1-knowledge-graph

# After implementing graph module
git add src/ai/graph/
git commit -m "feat(graph): implement QBMKnowledgeGraph with NetworkX + SQLite

- Add node types: Behavior, Ayah, Agent, Organ, etc.
- Add edge types: CAUSES, RESULTS_IN, OPPOSITE_OF, etc.
- Implement SQLite persistence for graph data
- Add centrality analysis and community detection"

# After adding tests
git add tests/ai/test_knowledge_graph.py
git commit -m "test(graph): add comprehensive test suite for knowledge graph

- Node creation and attribute tests
- Edge relationship tests
- Path finding and causal chain tests
- Persistence save/load tests
- Analytics tests (centrality, communities)"

# After tests pass
git push origin feature/phase1-knowledge-graph

# Create PR and merge to develop
git checkout develop
git merge feature/phase1-knowledge-graph
git tag -a v0.1.0-graph -m "Phase 1: Knowledge Graph complete"
git push origin develop --tags
```

### Phase 2 Git Commands

```bash
# Start Phase 2
git checkout develop
git checkout -b feature/phase2-vectors-rag

# After implementing vector store
git add src/ai/vectors/
git commit -m "feat(vectors): implement QBMVectorStore with ChromaDB

- Arabic embeddings using AraBERT
- Collections for ayat, behaviors, tafsir
- Semantic search with cosine similarity
- Persistent storage in data/chromadb/"

# After implementing RAG
git add src/ai/rag/
git commit -m "feat(rag): implement QBMRAGPipeline with Ollama

- Graph-expanded context retrieval
- Multi-source context building (ayat, behaviors, tafsir)
- Arabic prompt templates
- Ollama integration for local LLM inference"

# After tests
git add tests/ai/test_vectors.py tests/ai/test_rag.py
git commit -m "test(rag): add vector store and RAG pipeline tests"

git push origin feature/phase2-vectors-rag
git checkout develop
git merge feature/phase2-vectors-rag
git tag -a v0.2.0-rag -m "Phase 2: Vectors & RAG complete"
git push origin develop --tags
```

### Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding tests
- `docs`: Documentation
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

**Scopes:**
- `graph`: Knowledge graph
- `vectors`: Vector embeddings
- `rag`: RAG pipeline
- `ontology`: Taxonomy/ontology
- `tafsir`: Tafsir integration
- `model`: Model training
- `discovery`: Discovery engine
- `api`: API endpoints

### GitHub Actions CI

```yaml
# .github/workflows/ai-tests.yml
name: QBM AI Tests

on:
  push:
    branches: [develop, main]
    paths:
      - 'src/ai/**'
      - 'tests/ai/**'
  pull_request:
    branches: [develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ai/ -v --cov=src/ai --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Release Checklist

```markdown
## Phase X Release Checklist

- [ ] All tests passing (`pytest tests/ai/test_phase_x.py -v`)
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Tagged with `vX.X.X-phase`
- [ ] Pushed to GitHub
- [ ] GitHub release created with notes
```
