"""
Build Semantic Graph v2 (Phase 8.3)

Enhanced semantic graph with:
1. Cue phrase REQUIREMENT for causal types (CAUSES, LEADS_TO, PREVENTS, STRENGTHENS)
2. Confidence calibration based on:
   - Multi-source support
   - Cue phrase strength
   - Evidence count
3. Audit report with top 50 highest-confidence edges

Hard rules:
- Both endpoints must appear in the quote
- Cue phrase must be present for causal edge types
- No semantic edge without evidence offsets
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
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
SEMANTIC_OUTPUT = Path("data/graph/semantic_graph_v2.json")
AUDIT_REPORT = Path("reports/graph_audit_v1.md")

CORE_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Semantic edge types
CAUSAL_EDGE_TYPES = ["CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"]
NON_CAUSAL_EDGE_TYPES = ["OPPOSITE_OF", "COMPLEMENTS", "CONDITIONAL_ON"]
ALL_SEMANTIC_EDGE_TYPES = CAUSAL_EDGE_TYPES + NON_CAUSAL_EDGE_TYPES

# Cue phrase patterns with strength ratings
# Strong cues = higher confidence boost
# Weak cues = lower confidence boost, capped confidence
CUE_PATTERNS = {
    # CAUSES patterns
    'CAUSES': [
        {'pattern': r'بسبب', 'strength': 'strong', 'ar': 'بسبب'},
        {'pattern': r'لأن', 'strength': 'strong', 'ar': 'لأن'},
        {'pattern': r'سبب\s+في', 'strength': 'strong', 'ar': 'سبب في'},
        {'pattern': r'سبب', 'strength': 'medium', 'ar': 'سبب'},
        {'pattern': r'علة', 'strength': 'strong', 'ar': 'علة'},
        {'pattern': r'من\s+أجل', 'strength': 'medium', 'ar': 'من أجل'},
    ],
    # LEADS_TO patterns
    'LEADS_TO': [
        {'pattern': r'يؤدي\s*إلى', 'strength': 'strong', 'ar': 'يؤدي إلى'},
        {'pattern': r'يفضي\s*إلى', 'strength': 'strong', 'ar': 'يفضي إلى'},
        {'pattern': r'أدى\s*إلى', 'strength': 'strong', 'ar': 'أدى إلى'},
        {'pattern': r'نتيجة', 'strength': 'medium', 'ar': 'نتيجة'},
        {'pattern': r'ينتج\s*عنه', 'strength': 'strong', 'ar': 'ينتج عنه'},
        {'pattern': r'عاقبة', 'strength': 'strong', 'ar': 'عاقبة'},
        {'pattern': r'يورث', 'strength': 'medium', 'ar': 'يورث'},
    ],
    # PREVENTS patterns
    'PREVENTS': [
        {'pattern': r'يمنع', 'strength': 'strong', 'ar': 'يمنع'},
        {'pattern': r'يحول\s*دون', 'strength': 'strong', 'ar': 'يحول دون'},
        {'pattern': r'يحجب', 'strength': 'medium', 'ar': 'يحجب'},
        {'pattern': r'يصد\s*عن', 'strength': 'strong', 'ar': 'يصد عن'},
    ],
    # STRENGTHENS patterns
    'STRENGTHENS': [
        {'pattern': r'يزيد', 'strength': 'medium', 'ar': 'يزيد'},
        {'pattern': r'يقوي', 'strength': 'strong', 'ar': 'يقوي'},
        {'pattern': r'يعزز', 'strength': 'strong', 'ar': 'يعزز'},
        {'pattern': r'يضاعف', 'strength': 'strong', 'ar': 'يضاعف'},
    ],
    # OPPOSITE_OF patterns
    'OPPOSITE_OF': [
        {'pattern': r'ضد', 'strength': 'strong', 'ar': 'ضد'},
        {'pattern': r'عكس', 'strength': 'strong', 'ar': 'عكس'},
        {'pattern': r'نقيض', 'strength': 'strong', 'ar': 'نقيض'},
        {'pattern': r'خلاف', 'strength': 'medium', 'ar': 'خلاف'},
    ],
    # COMPLEMENTS patterns
    'COMPLEMENTS': [
        {'pattern': r'مع', 'strength': 'weak', 'ar': 'مع'},
        {'pattern': r'يكمل', 'strength': 'strong', 'ar': 'يكمل'},
        {'pattern': r'مقترن\s*ب', 'strength': 'strong', 'ar': 'مقترن ب'},
    ],
    # CONDITIONAL_ON patterns
    'CONDITIONAL_ON': [
        {'pattern': r'إذا', 'strength': 'medium', 'ar': 'إذا'},
        {'pattern': r'شرط', 'strength': 'strong', 'ar': 'شرط'},
        {'pattern': r'بشرط', 'strength': 'strong', 'ar': 'بشرط'},
        {'pattern': r'لا\s*يكون\s*إلا', 'strength': 'strong', 'ar': 'لا يكون إلا'},
    ],
}

# Confidence calibration constants
BASE_CONFIDENCE = 0.3
EVIDENCE_BOOST = 0.05  # Per evidence item (capped)
SOURCE_BOOST = 0.08    # Per unique source (capped)
STRONG_CUE_BOOST = 0.15
MEDIUM_CUE_BOOST = 0.08
WEAK_CUE_CAP = 0.6     # Max confidence for weak cues
MAX_CONFIDENCE = 0.95


def load_canonical_entities() -> Dict[str, Any]:
    """Load canonical entities registry."""
    with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)
    return text


def build_all_nodes(canonical_entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build ALL nodes from canonical entities."""
    nodes = {}
    
    for behavior in canonical_entities.get("behaviors", []):
        nodes[behavior["id"]] = {
            "id": behavior["id"],
            "type": "BEHAVIOR",
            "ar": behavior.get("ar", ""),
            "en": behavior.get("en", ""),
        }
    
    for agent in canonical_entities.get("agents", []):
        nodes[agent["id"]] = {
            "id": agent["id"],
            "type": "AGENT",
            "ar": agent.get("ar", ""),
            "en": agent.get("en", ""),
        }
    
    for organ in canonical_entities.get("organs", []):
        nodes[organ["id"]] = {
            "id": organ["id"],
            "type": "ORGAN",
            "ar": organ.get("ar", ""),
            "en": organ.get("en", ""),
        }
    
    for state in canonical_entities.get("heart_states", []):
        nodes[state["id"]] = {
            "id": state["id"],
            "type": "HEART_STATE",
            "ar": state.get("ar", ""),
            "en": state.get("en", ""),
        }
    
    for consequence in canonical_entities.get("consequences", []):
        nodes[consequence["id"]] = {
            "id": consequence["id"],
            "type": "CONSEQUENCE",
            "ar": consequence.get("ar", ""),
            "en": consequence.get("en", ""),
        }
    
    return nodes


def build_term_to_node_mapping(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from Arabic terms to node IDs."""
    term_to_node = {}
    
    for node_id, node in nodes.items():
        ar_term = node.get("ar", "")
        if ar_term:
            term_to_node[ar_term] = node_id
            normalized = normalize_arabic(ar_term)
            term_to_node[normalized] = node_id
            if ar_term.startswith("ال"):
                term_to_node[ar_term[2:]] = node_id
                term_to_node[normalize_arabic(ar_term[2:])] = node_id
    
    return term_to_node


def calculate_confidence(
    evidence_count: int,
    sources_count: int,
    cue_strength: str,
    edge_type: str
) -> float:
    """
    Calculate calibrated confidence score.
    
    Factors:
    - Base confidence
    - Evidence count boost (capped at 5)
    - Source count boost (capped at 5)
    - Cue phrase strength
    """
    confidence = BASE_CONFIDENCE
    
    # Evidence boost (capped)
    confidence += min(evidence_count, 5) * EVIDENCE_BOOST
    
    # Source boost (capped)
    confidence += min(sources_count, 5) * SOURCE_BOOST
    
    # Cue strength boost
    if cue_strength == 'strong':
        confidence += STRONG_CUE_BOOST
    elif cue_strength == 'medium':
        confidence += MEDIUM_CUE_BOOST
    elif cue_strength == 'weak':
        confidence = min(confidence, WEAK_CUE_CAP)
    
    # Cap at max
    return min(confidence, MAX_CONFIDENCE)


def build_semantic_graph_v2(
    nodes: Dict[str, Dict[str, Any]],
    term_to_node: Dict[str, str],
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build semantic graph v2 with cue phrase requirement and confidence calibration.
    """
    logger.info("Building semantic graph v2 with cue phrase validation...")
    
    edge_evidence = defaultdict(list)  # (source, target, edge_type) -> evidence list
    edge_cue_strengths = defaultdict(set)  # (source, target, edge_type) -> set of cue strengths
    
    logger.info(f"Scanning {len(chunks)} chunks...")
    
    for chunk in chunks:
        text = chunk.get('text_clean', '')
        if not text or len(text) < 100:
            continue
        
        chunk_id = chunk.get('chunk_id', '')
        source = chunk.get('source', '')
        surah = chunk.get('surah', 0)
        ayah = chunk.get('ayah', 0)
        text_normalized = normalize_arabic(text)
        
        # Check each edge type and its patterns
        for edge_type, patterns in CUE_PATTERNS.items():
            for cue_info in patterns:
                pattern = cue_info['pattern']
                strength = cue_info['strength']
                cue_ar = cue_info['ar']
                
                for match in re.finditer(pattern, text_normalized):
                    match_start = match.start()
                    match_end = match.end()
                    
                    # Extract quote window
                    quote_start = max(0, match_start - 80)
                    quote_end = min(len(text_normalized), match_end + 80)
                    quote = text_normalized[quote_start:quote_end]
                    
                    # Find nodes in quote
                    nodes_in_quote = []
                    for term, node_id in term_to_node.items():
                        term_normalized = normalize_arabic(term)
                        if len(term_normalized) >= 3 and term_normalized in quote:
                            nodes_in_quote.append((node_id, term_normalized))
                    
                    # Need at least 2 different nodes
                    unique_nodes = list(set(n[0] for n in nodes_in_quote))
                    if len(unique_nodes) < 2:
                        continue
                    
                    # Create edges between pairs
                    for i, n1 in enumerate(unique_nodes):
                        for n2 in unique_nodes[i+1:]:
                            n1_term = next((t for nid, t in nodes_in_quote if nid == n1), '')
                            n2_term = next((t for nid, t in nodes_in_quote if nid == n2), '')
                            
                            n1_pos = quote.find(n1_term)
                            n2_pos = quote.find(n2_term)
                            cue_pos = match_start - quote_start
                            
                            # Determine direction
                            if n1_pos < cue_pos < n2_pos:
                                src, tgt = n1, n2
                            elif n2_pos < cue_pos < n1_pos:
                                src, tgt = n2, n1
                            else:
                                continue
                            
                            edge_key = (src, tgt, edge_type)
                            
                            # Get original quote
                            orig_quote = text[quote_start:quote_end].strip()
                            
                            evidence_item = {
                                'source': source,
                                'surah': surah,
                                'ayah': ayah,
                                'verse_key': f"{surah}:{ayah}",
                                'chunk_id': chunk_id,
                                'char_start': quote_start,
                                'char_end': quote_end,
                                'cue_phrase': cue_ar,
                                'cue_strength': strength,
                                'quote': orig_quote,
                                'endpoints_validated': [n1_term, n2_term],
                            }
                            
                            edge_evidence[edge_key].append(evidence_item)
                            edge_cue_strengths[edge_key].add(strength)
    
    # Convert to edges with confidence calibration
    MIN_EVIDENCE = 2
    edges = []
    
    for (src, tgt, edge_type), evidence_list in edge_evidence.items():
        if len(evidence_list) < MIN_EVIDENCE:
            continue
        
        # Get unique sources
        unique_sources = set(e['source'] for e in evidence_list)
        
        # Get strongest cue
        cue_strengths = edge_cue_strengths[(src, tgt, edge_type)]
        if 'strong' in cue_strengths:
            best_strength = 'strong'
        elif 'medium' in cue_strengths:
            best_strength = 'medium'
        else:
            best_strength = 'weak'
        
        # Calculate calibrated confidence
        confidence = calculate_confidence(
            evidence_count=len(evidence_list),
            sources_count=len(unique_sources),
            cue_strength=best_strength,
            edge_type=edge_type
        )
        
        # Get sample cue phrases used
        cue_phrases_used = list(set(e['cue_phrase'] for e in evidence_list))
        
        edges.append({
            'source': src,
            'target': tgt,
            'edge_type': edge_type,
            'confidence': round(confidence, 3),
            'evidence_count': len(evidence_list),
            'sources_count': len(unique_sources),
            'sources': list(unique_sources),
            'cue_strength': best_strength,
            'cue_phrases': cue_phrases_used,
            'evidence': evidence_list[:5],  # Top 5 for storage
            'validation': {
                'endpoints_in_quote': True,
                'cue_phrase_present': True,
                'min_evidence_met': True,
            }
        })
    
    # Sort by confidence
    edges.sort(key=lambda x: -x['confidence'])
    
    logger.info(f"Extracted {len(edges)} validated semantic edges")
    
    # Stats
    nodes_by_type = defaultdict(int)
    for node in nodes.values():
        nodes_by_type[node['type']] += 1
    
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge['source'])
        nodes_with_edges.add(edge['target'])
    isolated_count = len(nodes) - len(nodes_with_edges)
    
    edges_by_type = defaultdict(int)
    for edge in edges:
        edges_by_type[edge['edge_type']] += 1
    
    # Confidence distribution
    high_conf = len([e for e in edges if e['confidence'] >= 0.7])
    medium_conf = len([e for e in edges if 0.5 <= e['confidence'] < 0.7])
    low_conf = len([e for e in edges if e['confidence'] < 0.5])
    
    graph = {
        'graph_type': 'semantic',
        'version': '2.0',
        'created_at': datetime.now().isoformat(),
        'description': 'Semantic claim graph v2. Cue phrase required for causal types. Confidence calibrated.',
        'allowed_edge_types': ALL_SEMANTIC_EDGE_TYPES,
        'causal_edge_types': CAUSAL_EDGE_TYPES,
        'nodes': list(nodes.values()),
        'node_count': len(nodes),
        'nodes_by_type': dict(nodes_by_type),
        'isolated_node_count': isolated_count,
        'edge_count': len(edges),
        'edges_by_type': dict(edges_by_type),
        'edges': edges,
        'confidence_distribution': {
            'high_confidence_0.7+': high_conf,
            'medium_confidence_0.5-0.7': medium_conf,
            'low_confidence_below_0.5': low_conf,
        },
        'hard_rules': [
            'Both endpoints must appear in the quote',
            'Cue phrase must be present for causal edge types',
            'No semantic edge without evidence offsets',
            'Confidence calibrated by evidence count, source count, cue strength',
        ],
        'calibration': {
            'base_confidence': BASE_CONFIDENCE,
            'evidence_boost': EVIDENCE_BOOST,
            'source_boost': SOURCE_BOOST,
            'strong_cue_boost': STRONG_CUE_BOOST,
            'medium_cue_boost': MEDIUM_CUE_BOOST,
            'weak_cue_cap': WEAK_CUE_CAP,
            'max_confidence': MAX_CONFIDENCE,
        }
    }
    
    return graph


def generate_audit_report(graph: Dict[str, Any]) -> str:
    """Generate audit report with top 50 highest-confidence edges."""
    
    edges = graph.get('edges', [])
    top_50 = edges[:50]
    
    report = f"""# Semantic Graph Audit Report v1

**Generated**: {datetime.now().isoformat()}
**Graph Version**: {graph.get('version', '2.0')}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Nodes | {graph['node_count']} |
| Isolated Nodes | {graph.get('isolated_node_count', 0)} |
| Total Edges | {graph['edge_count']} |
| High Confidence (≥0.7) | {graph['confidence_distribution']['high_confidence_0.7+']} |
| Medium Confidence (0.5-0.7) | {graph['confidence_distribution']['medium_confidence_0.5-0.7']} |
| Low Confidence (<0.5) | {graph['confidence_distribution']['low_confidence_below_0.5']} |

---

## Edges by Type

| Edge Type | Count |
|-----------|-------|
"""
    
    for edge_type, count in graph.get('edges_by_type', {}).items():
        report += f"| {edge_type} | {count} |\n"
    
    report += """
---

## Top 50 Highest-Confidence Edges

Each edge shows:
- Source → Target (Edge Type)
- Confidence score
- Evidence count and sources
- Cue phrases used
- Sample evidence quote with **cue phrase highlighted**

"""
    
    for i, edge in enumerate(top_50, 1):
        src = edge['source']
        tgt = edge['target']
        edge_type = edge['edge_type']
        confidence = edge['confidence']
        evidence_count = edge['evidence_count']
        sources = edge.get('sources', [])
        cue_phrases = edge.get('cue_phrases', [])
        
        report += f"""### {i}. {src} → {tgt}

- **Edge Type**: {edge_type}
- **Confidence**: {confidence}
- **Evidence Count**: {evidence_count}
- **Sources**: {', '.join(sources)}
- **Cue Phrases**: {', '.join(cue_phrases)}

"""
        
        # Show first evidence with highlighted cue
        if edge.get('evidence'):
            ev = edge['evidence'][0]
            quote = ev.get('quote', '')
            cue = ev.get('cue_phrase', '')
            
            # Highlight cue phrase in quote
            if cue and cue in quote:
                highlighted = quote.replace(cue, f"**{cue}**")
            else:
                highlighted = quote
            
            report += f"""**Sample Evidence** ({ev.get('source', 'unknown')}, {ev.get('verse_key', '')}):

> {highlighted}

"""
        
        report += "---\n\n"
    
    report += """
## Validation Rules Applied

1. **Endpoint Validation**: Both source and target terms must appear in the evidence quote.
2. **Cue Phrase Requirement**: For causal edge types (CAUSES, LEADS_TO, PREVENTS, STRENGTHENS), a cue phrase must be present.
3. **Minimum Evidence**: At least 2 evidence items required per edge.
4. **Confidence Calibration**:
   - Base: 0.3
   - +0.05 per evidence (max 5)
   - +0.08 per unique source (max 5)
   - +0.15 for strong cue, +0.08 for medium cue
   - Weak cues capped at 0.6 confidence

## No Fabrication Guarantee

Every edge in this graph has:
- Real evidence from tafsir corpus
- Validated endpoints in quote
- Cue phrase present in quote
- Character offsets for audit
"""
    
    return report


def main():
    """Build semantic graph v2 and generate audit report."""
    logger.info("Loading data...")
    
    canonical_entities = load_canonical_entities()
    chunks = load_chunked_index()
    
    nodes = build_all_nodes(canonical_entities)
    term_to_node = build_term_to_node_mapping(nodes)
    
    logger.info(f"Built {len(nodes)} nodes, {len(term_to_node)} term mappings")
    
    # Build semantic graph
    semantic_graph = build_semantic_graph_v2(nodes, term_to_node, chunks)
    
    # Save graph
    SEMANTIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(semantic_graph, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote semantic graph to {SEMANTIC_OUTPUT}")
    
    # Generate and save audit report
    audit_report = generate_audit_report(semantic_graph)
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_REPORT, 'w', encoding='utf-8') as f:
        f.write(audit_report)
    logger.info(f"Wrote audit report to {AUDIT_REPORT}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Semantic Graph v2 Summary")
    print("=" * 60)
    print(f"Total nodes: {semantic_graph['node_count']}")
    print(f"Isolated nodes: {semantic_graph.get('isolated_node_count', 0)}")
    print(f"Total edges: {semantic_graph['edge_count']}")
    print(f"\nEdges by type:")
    for et, count in semantic_graph.get('edges_by_type', {}).items():
        print(f"  {et}: {count}")
    print(f"\nConfidence distribution:")
    for level, count in semantic_graph['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    return semantic_graph


if __name__ == "__main__":
    main()
