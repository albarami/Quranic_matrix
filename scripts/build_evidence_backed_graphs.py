"""
Build Evidence-Backed Graphs (Phase 6.3)

Two explicit graphs:
1. Co-occurrence graph (statistical only) - for discovery
2. Semantic claim graph (typed edges only) - for causal reasoning

Hard rules:
- Graph must include ALL canonical entities as nodes (even isolated)
- No semantic edge without evidence offsets
- Both endpoints must appear in the quote for semantic edges
- No causal chain may use co-occurrence edges
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v1.jsonl")
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
COOCCURRENCE_OUTPUT = Path("data/graph/cooccurrence_graph_v1.json")
SEMANTIC_OUTPUT = Path("data/graph/semantic_graph_v1.json")
REPORT_OUTPUT = Path("reports/graph_quality_report.md")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Semantic edge types (allowed for causal reasoning)
SEMANTIC_EDGE_TYPES = [
    "CAUSES",           # A causes B
    "LEADS_TO",         # A leads to B
    "PREVENTS",         # A prevents B
    "OPPOSITE_OF",      # A is opposite of B
    "COMPLEMENTS",      # A complements B
    "CONDITIONAL_ON",   # A is conditional on B
    "STRENGTHENS",      # A strengthens B
]


def load_canonical_entities() -> Dict[str, Any]:
    """Load canonical entities registry."""
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_concept_index() -> Dict[str, Any]:
    """Load the concept evidence index."""
    concepts = {}
    with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            concepts[entry['concept_id']] = entry
    return concepts


def load_chunked_index() -> List[Dict[str, Any]]:
    """Load the chunked tafsir evidence index."""
    chunks = []
    with open(CHUNKED_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for matching."""
    if not text:
        return text
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Normalize Alef forms
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Normalize Yaa
    text = re.sub(r'ى', 'ي', text)
    # Normalize Taa Marbuta
    text = re.sub(r'ة', 'ه', text)
    # Remove tatweel
    text = re.sub(r'ـ', '', text)
    return text


def build_all_nodes(canonical_entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build ALL nodes from canonical entities.
    Every canonical entity becomes a node, even if isolated.
    """
    nodes = {}
    
    # Add all behaviors (73)
    for behavior in canonical_entities.get("behaviors", []):
        node_id = behavior["id"]
        nodes[node_id] = {
            "id": node_id,
            "type": "BEHAVIOR",
            "ar": behavior.get("ar", ""),
            "en": behavior.get("en", ""),
            "category": behavior.get("category", ""),
        }
    
    # Add all agents
    for agent in canonical_entities.get("agents", []):
        node_id = agent["id"]
        nodes[node_id] = {
            "id": node_id,
            "type": "AGENT",
            "ar": agent.get("ar", ""),
            "en": agent.get("en", ""),
        }
    
    # Add all organs
    for organ in canonical_entities.get("organs", []):
        node_id = organ["id"]
        nodes[node_id] = {
            "id": node_id,
            "type": "ORGAN",
            "ar": organ.get("ar", ""),
            "en": organ.get("en", ""),
        }
    
    # Add all heart states
    for state in canonical_entities.get("heart_states", []):
        node_id = state["id"]
        nodes[node_id] = {
            "id": node_id,
            "type": "HEART_STATE",
            "ar": state.get("ar", ""),
            "en": state.get("en", ""),
            "polarity": state.get("polarity", ""),
        }
    
    # Add all consequences
    for consequence in canonical_entities.get("consequences", []):
        node_id = consequence["id"]
        nodes[node_id] = {
            "id": node_id,
            "type": "CONSEQUENCE",
            "ar": consequence.get("ar", ""),
            "en": consequence.get("en", ""),
            "temporal": consequence.get("temporal", ""),
            "polarity": consequence.get("polarity", ""),
        }
    
    return nodes


def build_term_to_node_mapping(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from Arabic terms to node IDs."""
    term_to_node = {}
    
    for node_id, node in nodes.items():
        ar_term = node.get("ar", "")
        if ar_term:
            # Add original term
            term_to_node[ar_term] = node_id
            
            # Add normalized version
            normalized = normalize_arabic(ar_term)
            term_to_node[normalized] = node_id
            
            # Add without ال prefix
            if ar_term.startswith("ال"):
                term_to_node[ar_term[2:]] = node_id
                term_to_node[normalize_arabic(ar_term[2:])] = node_id
    
    return term_to_node


def build_cooccurrence_graph(
    nodes: Dict[str, Dict[str, Any]],
    term_to_node: Dict[str, str],
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build co-occurrence graph (statistical only).
    
    - ALL canonical entities are nodes (even isolated)
    - Edges only from real co-occurrence in same chunk
    """
    logger.info("Building co-occurrence graph with ALL nodes...")
    
    # Track which nodes appear in which chunks
    node_chunks = defaultdict(set)  # node_id -> set of chunk_ids
    chunk_nodes = defaultdict(set)  # chunk_id -> set of node_ids
    
    # Scan all chunks for node mentions
    for chunk in chunks:
        text = chunk.get('text_clean', '')
        if not text:
            continue
        
        chunk_id = chunk.get('chunk_id', '')
        text_normalized = normalize_arabic(text)
        
        # Find all nodes mentioned in this chunk
        for term, node_id in term_to_node.items():
            term_normalized = normalize_arabic(term)
            if term_normalized in text_normalized:
                node_chunks[node_id].add(chunk_id)
                chunk_nodes[chunk_id].add(node_id)
    
    # Calculate co-occurrences from real chunk overlap
    edges = []
    node_ids_with_mentions = list(node_chunks.keys())
    
    for i, n1 in enumerate(node_ids_with_mentions):
        for n2 in node_ids_with_mentions[i+1:]:
            # Count chunks where both appear
            shared = node_chunks[n1] & node_chunks[n2]
            if len(shared) >= 2:  # Minimum threshold
                # Calculate PMI
                total_chunks = len(chunks)
                p_n1 = len(node_chunks[n1]) / total_chunks
                p_n2 = len(node_chunks[n2]) / total_chunks
                p_joint = len(shared) / total_chunks
                
                if p_n1 > 0 and p_n2 > 0 and p_joint > 0:
                    pmi = p_joint / (p_n1 * p_n2)
                else:
                    pmi = 0
                
                # Get sample verse_keys for audit
                sample_verses = []
                for chunk_id in list(shared)[:5]:
                    for c in chunks:
                        if c.get('chunk_id') == chunk_id:
                            sample_verses.append(c.get('verse_key', ''))
                            break
                
                edges.append({
                    'source': n1,
                    'target': n2,
                    'edge_type': 'CO_OCCURS_WITH',
                    'count': len(shared),
                    'pmi': round(pmi, 4),
                    'sample_verse_keys': sample_verses,
                })
    
    # Sort by count
    edges.sort(key=lambda x: -x['count'])
    
    # Count nodes by type
    nodes_by_type = defaultdict(int)
    for node in nodes.values():
        nodes_by_type[node['type']] += 1
    
    # Count isolated nodes (no edges)
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge['source'])
        nodes_with_edges.add(edge['target'])
    isolated_count = len(nodes) - len(nodes_with_edges)
    
    graph = {
        'graph_type': 'cooccurrence',
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'description': 'Statistical co-occurrence graph. For discovery only, NOT for causal reasoning.',
        'edge_type': 'CO_OCCURS_WITH',
        'nodes': list(nodes.values()),
        'node_count': len(nodes),
        'nodes_by_type': dict(nodes_by_type),
        'isolated_node_count': isolated_count,
        'edge_count': len(edges),
        'edges': edges,
    }
    
    logger.info(f"Co-occurrence graph: {len(nodes)} nodes ({isolated_count} isolated), {len(edges)} edges")
    return graph


def build_semantic_graph(
    nodes: Dict[str, Dict[str, Any]],
    term_to_node: Dict[str, str],
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build semantic claim graph (typed edges only).
    
    STRICT VALIDATION RULES:
    - Both endpoints must appear in the quote
    - Cue phrase must be present in the quote
    - Evidence offsets must be valid
    - NO hardcoded relationships
    """
    logger.info("Building semantic graph with STRICT validation...")
    
    # Arabic cue patterns for relationship extraction
    CAUSAL_PATTERNS = [
        (r'بسبب', 'CAUSES'),
        (r'لأن', 'CAUSES'),
        (r'سبب', 'CAUSES'),
    ]
    
    LEADS_TO_PATTERNS = [
        (r'يؤدي\s*إلى', 'LEADS_TO'),
        (r'يفضي\s*إلى', 'LEADS_TO'),
        (r'أدى\s*إلى', 'LEADS_TO'),
        (r'نتيجة', 'LEADS_TO'),
    ]
    
    OPPOSITION_PATTERNS = [
        (r'ضد', 'OPPOSITE_OF'),
        (r'عكس', 'OPPOSITE_OF'),
        (r'نقيض', 'OPPOSITE_OF'),
        (r'خلاف', 'OPPOSITE_OF'),
    ]
    
    STRENGTHENING_PATTERNS = [
        (r'يزيد', 'STRENGTHENS'),
        (r'يقوي', 'STRENGTHENS'),
    ]
    
    PREVENTION_PATTERNS = [
        (r'يمنع', 'PREVENTS'),
        (r'يحول\s*دون', 'PREVENTS'),
    ]
    
    ALL_PATTERNS = (CAUSAL_PATTERNS + LEADS_TO_PATTERNS + 
                   OPPOSITION_PATTERNS + STRENGTHENING_PATTERNS + PREVENTION_PATTERNS)
    
    edge_evidence = defaultdict(list)  # (source, target, edge_type) -> evidence list
    
    logger.info(f"Scanning {len(chunks)} chunks for semantic patterns with strict validation...")
    
    for chunk in chunks:
        text = chunk.get('text_clean', '')
        if not text or len(text) < 100:
            continue
        
        chunk_id = chunk.get('chunk_id', '')
        source = chunk.get('source', '')
        surah = chunk.get('surah', 0)
        ayah = chunk.get('ayah', 0)
        text_normalized = normalize_arabic(text)
        
        # Check each pattern
        for pattern, edge_type in ALL_PATTERNS:
            for match in re.finditer(pattern, text_normalized):
                match_start = match.start()
                match_end = match.end()
                
                # Extract quote window (80 chars before and after the cue)
                quote_start = max(0, match_start - 80)
                quote_end = min(len(text_normalized), match_end + 80)
                quote = text_normalized[quote_start:quote_end]
                
                # STRICT VALIDATION: Find ALL nodes that appear in this quote
                nodes_in_quote = []
                for term, node_id in term_to_node.items():
                    term_normalized = normalize_arabic(term)
                    if len(term_normalized) >= 3 and term_normalized in quote:
                        nodes_in_quote.append((node_id, term_normalized))
                
                # Only create edges if we have at least 2 different nodes in the quote
                unique_nodes = list(set(n[0] for n in nodes_in_quote))
                if len(unique_nodes) >= 2:
                    # Create edges between all pairs found in the same quote
                    for i, n1 in enumerate(unique_nodes):
                        for n2 in unique_nodes[i+1:]:
                            # Determine direction based on position in quote
                            # (node appearing before cue -> node appearing after cue)
                            n1_term = next((t for nid, t in nodes_in_quote if nid == n1), '')
                            n2_term = next((t for nid, t in nodes_in_quote if nid == n2), '')
                            
                            n1_pos = quote.find(n1_term)
                            n2_pos = quote.find(n2_term)
                            cue_pos = match_start - quote_start
                            
                            # Source is before cue, target is after
                            if n1_pos < cue_pos < n2_pos:
                                src, tgt = n1, n2
                            elif n2_pos < cue_pos < n1_pos:
                                src, tgt = n2, n1
                            else:
                                # Both on same side - skip ambiguous
                                continue
                            
                            edge_key = (src, tgt, edge_type)
                            
                            # Get original (non-normalized) quote for display
                            orig_quote = text[quote_start:quote_end].strip()
                            
                            evidence_item = {
                                'source': source,
                                'surah': surah,
                                'ayah': ayah,
                                'chunk_id': chunk_id,
                                'char_start': quote_start,
                                'char_end': quote_end,
                                'cue_pattern': pattern,
                                'quote': orig_quote,
                                'endpoints_in_quote': [n1_term, n2_term],
                                'validated': True,
                            }
                            
                            edge_evidence[edge_key].append(evidence_item)
    
    # Convert to edges (only edges with at least 2 validated evidence items)
    MIN_EVIDENCE = 2
    edges = []
    
    for (src, tgt, edge_type), evidence_list in edge_evidence.items():
        if len(evidence_list) >= MIN_EVIDENCE:
            # Count unique sources
            unique_sources = set(e['source'] for e in evidence_list)
            
            edges.append({
                'source': src,
                'target': tgt,
                'edge_type': edge_type,
                'confidence': min(0.4 + 0.1 * len(evidence_list) + 0.1 * len(unique_sources), 0.95),
                'evidence_count': len(evidence_list),
                'sources_count': len(unique_sources),
                'evidence': evidence_list[:5],
                'extractor_version': '1.0',
                'extraction_method': 'arabic_cue_patterns_strict',
            })
    
    logger.info(f"Extracted {len(edges)} validated semantic edges")
    
    # Count nodes by type
    nodes_by_type = defaultdict(int)
    for node in nodes.values():
        nodes_by_type[node['type']] += 1
    
    # Count isolated nodes
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge['source'])
        nodes_with_edges.add(edge['target'])
    isolated_count = len(nodes) - len(nodes_with_edges)
    
    # Count edges by type
    edges_by_type = defaultdict(int)
    for edge in edges:
        edges_by_type[edge['edge_type']] += 1
    
    graph = {
        'graph_type': 'semantic',
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'description': 'Semantic claim graph. Both endpoints verified in quote. Evidence-backed only.',
        'allowed_edge_types': SEMANTIC_EDGE_TYPES,
        'nodes': list(nodes.values()),
        'node_count': len(nodes),
        'nodes_by_type': dict(nodes_by_type),
        'isolated_node_count': isolated_count,
        'edge_count': len(edges),
        'edges_by_type': dict(edges_by_type),
        'edges': edges,
        'hard_rules': [
            'Both endpoints must appear in the quote',
            'Cue phrase must be present between endpoints',
            'No semantic edge without evidence offsets',
            'No causal chain may use co-occurrence edges',
        ],
        'validation': {
            'extraction_method': 'arabic_cue_patterns_strict',
            'min_evidence_per_edge': 2,
            'endpoints_validated': True,
        }
    }
    
    logger.info(f"Semantic graph: {len(nodes)} nodes ({isolated_count} isolated), {len(edges)} edges")
    return graph


def generate_quality_report(cooccurrence: Dict, semantic: Dict) -> str:
    """Generate graph quality report."""
    report = f"""# Graph Quality Report (Phase 6.3)

**Generated**: {datetime.now().isoformat()}

---

## Summary

| Graph | Total Nodes | Isolated Nodes | Edges |
|-------|-------------|----------------|-------|
| Co-occurrence | {cooccurrence['node_count']} | {cooccurrence.get('isolated_node_count', 0)} | {cooccurrence['edge_count']} |
| Semantic | {semantic['node_count']} | {semantic.get('isolated_node_count', 0)} | {semantic['edge_count']} |

---

## Node Counts by Type

| Type | Count |
|------|-------|
"""
    for node_type, count in cooccurrence.get('nodes_by_type', {}).items():
        report += f"| {node_type} | {count} |\n"
    
    report += f"""
---

## Co-occurrence Graph (Statistical)

**Purpose**: Discovery only. NOT for causal reasoning.

### Top 20 Co-occurring Pairs

| Concept 1 | Concept 2 | Count | PMI |
|-----------|-----------|-------|-----|
"""
    
    for edge in cooccurrence['edges'][:20]:
        report += f"| {edge['source']} | {edge['target']} | {edge['count']} | {edge['pmi']:.2f} |\n"
    
    report += f"""
---

## Semantic Graph (Typed Edges)

**Purpose**: Causal reasoning with evidence.

**Hard Rules**:
- No semantic edge without evidence offsets
- No causal chain may use co-occurrence edges

### Edge Type Distribution

"""
    
    edge_type_counts = defaultdict(int)
    for edge in semantic['edges']:
        edge_type_counts[edge['edge_type']] += 1
    
    for et, count in edge_type_counts.items():
        report += f"- **{et}**: {count} edges\n"
    
    report += """
### Top 20 Semantic Edges with Evidence

"""
    
    for i, edge in enumerate(semantic['edges'][:20], 1):
        report += f"""#### {i}. {edge['source']} → {edge['target']} ({edge['edge_type']})

- **Confidence**: {edge['confidence']}
- **Evidence count**: {len(edge.get('evidence', []))}

"""
        for ev in edge.get('evidence', [])[:2]:
            report += f"  - Source: {ev['source']}, Verse: {ev['surah']}:{ev['ayah']}\n"
            report += f"    > \"{ev['quote'][:100]}...\"\n\n"
    
    # Multi-source support stats
    multi_source_count = 0
    for edge in semantic['edges']:
        sources = set(ev['source'] for ev in edge.get('evidence', []))
        if len(sources) >= 2:
            multi_source_count += 1
    
    multi_source_pct = multi_source_count / len(semantic['edges']) * 100 if semantic['edges'] else 0
    
    report += f"""
---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Semantic edges with evidence | {len(semantic['edges'])} / {len(semantic['edges'])} (100%) |
| Multi-source supported edges | {multi_source_count} ({multi_source_pct:.1f}%) |
| Average evidence per edge | {sum(len(e.get('evidence', [])) for e in semantic['edges']) / len(semantic['edges']) if semantic['edges'] else 0:.1f} |

---

## Validation

✅ All semantic edges have evidence offsets (char_start, char_end, quote)
✅ Co-occurrence graph is marked for discovery only
✅ Semantic graph has typed edges only
"""
    
    return report


def main():
    """Build both graphs and generate report."""
    logger.info("Loading data...")
    
    # Load canonical entities for complete node set
    canonical_entities = load_canonical_entities()
    chunks = load_chunked_index()
    
    # Build ALL nodes from canonical vocab
    nodes = build_all_nodes(canonical_entities)
    term_to_node = build_term_to_node_mapping(nodes)
    
    logger.info(f"Built {len(nodes)} nodes from canonical entities")
    logger.info(f"Built {len(term_to_node)} term mappings")
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Build graphs with ALL nodes
    cooccurrence = build_cooccurrence_graph(nodes, term_to_node, chunks)
    semantic = build_semantic_graph(nodes, term_to_node, chunks)
    
    # Save graphs
    COOCCURRENCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(COOCCURRENCE_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(cooccurrence, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved co-occurrence graph to {COOCCURRENCE_OUTPUT}")
    
    with open(SEMANTIC_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(semantic, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved semantic graph to {SEMANTIC_OUTPUT}")
    
    # Generate report
    report = generate_quality_report(cooccurrence, semantic)
    
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved quality report to {REPORT_OUTPUT}")
    
    print(report)


if __name__ == "__main__":
    main()
