"""
QBM Unified Graph - TRUE Integration (No Silos)

This module creates a SINGLE unified graph where:
- Every verse connects to ALL 5 tafsir sources
- Every behavioral annotation links to tafsir mentions
- Relationships flow bidirectionally: Tafsir ↔ Spans ↔ Verses ↔ Graph
- All data is interconnected as ONE BRAIN

NO SILOS - Everything is connected.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict
import json
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TAFSIR_DIR = DATA_DIR / "tafsir"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]

# Behavioral keywords for cross-referencing
BEHAVIOR_KEYWORDS = {
    "إيمان": "faith", "صبر": "patience", "شكر": "gratitude", "توبة": "repentance",
    "تقوى": "piety", "إحسان": "excellence", "صدق": "truthfulness", "أمانة": "trustworthiness",
    "عدل": "justice", "رحمة": "mercy", "تواضع": "humility", "خشوع": "humility_prayer",
    "ذكر": "remembrance", "دعاء": "supplication", "توكل": "reliance", "رضا": "contentment",
    "كفر": "disbelief", "نفاق": "hypocrisy", "كبر": "arrogance", "حسد": "envy",
    "غيبة": "backbiting", "كذب": "lying", "ظلم": "oppression", "فسق": "transgression",
    "رياء": "showing_off", "غضب": "anger", "بخل": "stinginess", "غفلة": "heedlessness",
    "شرك": "polytheism", "قلب": "heart", "سليم": "sound", "مريض": "sick",
    "قاسي": "hard", "مختوم": "sealed", "ميت": "dead", "منيب": "repentant",
    "مؤمن": "believer", "كافر": "disbeliever", "منافق": "hypocrite",
}


# =============================================================================
# UNIFIED GRAPH NODE TYPES
# =============================================================================

class NodeType:
    VERSE = "verse"           # Quran verse (surah:ayah)
    SPAN = "span"             # Behavioral annotation
    TAFSIR = "tafsir"         # Tafsir entry (source:surah:ayah)
    BEHAVIOR = "behavior"     # Behavioral concept
    AGENT = "agent"           # Agent type (believer, kafir, etc.)
    ORGAN = "organ"           # Body organ (heart, tongue, etc.)
    DIMENSION = "dimension"   # 11 dimensions


class EdgeType:
    VERSE_HAS_SPAN = "verse_has_span"           # Verse → Span
    VERSE_HAS_TAFSIR = "verse_has_tafsir"       # Verse → Tafsir
    SPAN_MENTIONS_BEHAVIOR = "span_mentions"    # Span → Behavior
    TAFSIR_MENTIONS_BEHAVIOR = "tafsir_mentions"  # Tafsir → Behavior
    BEHAVIOR_CAUSES = "causes"                  # Behavior → Behavior
    BEHAVIOR_EFFECTS = "effects"                # Behavior → Behavior
    BEHAVIOR_OPPOSITE = "opposite"              # Behavior ↔ Behavior
    BEHAVIOR_SIMILAR = "similar"                # Behavior ↔ Behavior
    SPAN_HAS_AGENT = "has_agent"                # Span → Agent
    SPAN_HAS_ORGAN = "has_organ"                # Span → Organ
    TAFSIR_AGREES = "agrees_with"               # Tafsir ↔ Tafsir
    TAFSIR_ELABORATES = "elaborates"            # Tafsir → Tafsir


# =============================================================================
# UNIFIED GRAPH
# =============================================================================

class UnifiedGraph:
    """
    A single unified graph connecting ALL QBM data:
    - 6,236 verses
    - 322,939 behavioral annotations (spans)
    - 31,180 tafsir entries (5 sources × 6,236 verses)
    - 76,597 tafsir behavioral mentions
    - Relationships between all entities
    """
    
    def __init__(self):
        # Nodes: {node_id: {type, data}}
        self.nodes = {}
        
        # Edges: {edge_type: {from_id: {to_id: edge_data}}}
        self.edges = defaultdict(lambda: defaultdict(dict))
        
        # Reverse edges for bidirectional traversal
        self.reverse_edges = defaultdict(lambda: defaultdict(dict))
        
        # Indices for fast lookup
        self.by_type = defaultdict(set)      # {node_type: set of node_ids}
        self.by_verse = defaultdict(set)     # {verse_key: set of node_ids}
        self.by_behavior = defaultdict(set)  # {behavior: set of node_ids}
        
        # Statistics
        self.stats = {
            "nodes": 0,
            "edges": 0,
            "verses": 0,
            "spans": 0,
            "tafsir_entries": 0,
            "behaviors": 0,
        }
    
    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================
    
    def add_node(self, node_id: str, node_type: str, data: Dict[str, Any]) -> str:
        """Add a node to the graph."""
        self.nodes[node_id] = {
            "type": node_type,
            "data": data,
        }
        self.by_type[node_type].add(node_id)
        self.stats["nodes"] += 1
        return node_id
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    # =========================================================================
    # EDGE OPERATIONS (Bidirectional)
    # =========================================================================
    
    def add_edge(self, from_id: str, to_id: str, edge_type: str, data: Dict[str, Any] = None):
        """Add an edge (and reverse edge for bidirectional traversal)."""
        if data is None:
            data = {}
        
        self.edges[edge_type][from_id][to_id] = data
        self.reverse_edges[edge_type][to_id][from_id] = data
        self.stats["edges"] += 1
    
    def get_neighbors(self, node_id: str, edge_type: str = None, direction: str = "out") -> List[str]:
        """Get neighbors of a node."""
        neighbors = []
        
        if direction in ["out", "both"]:
            if edge_type:
                neighbors.extend(self.edges[edge_type].get(node_id, {}).keys())
            else:
                for et in self.edges:
                    neighbors.extend(self.edges[et].get(node_id, {}).keys())
        
        if direction in ["in", "both"]:
            if edge_type:
                neighbors.extend(self.reverse_edges[edge_type].get(node_id, {}).keys())
            else:
                for et in self.reverse_edges:
                    neighbors.extend(self.reverse_edges[et].get(node_id, {}).keys())
        
        return list(set(neighbors))
    
    # =========================================================================
    # LOAD ALL DATA INTO UNIFIED GRAPH
    # =========================================================================
    
    def load_verses(self, spans: List[Dict[str, Any]]):
        """Load verses from spans and create verse nodes."""
        verses_seen = set()
        
        for span in spans:
            ref = span.get("reference", {})
            surah = ref.get("surah")
            ayah = ref.get("ayah")
            
            if surah and ayah:
                verse_key = f"{surah}:{ayah}"
                
                if verse_key not in verses_seen:
                    # Create verse node
                    self.add_node(
                        f"verse:{verse_key}",
                        NodeType.VERSE,
                        {"surah": surah, "ayah": ayah, "key": verse_key}
                    )
                    verses_seen.add(verse_key)
                    self.stats["verses"] += 1
        
        print(f"[UnifiedGraph] Loaded {len(verses_seen)} verses")
    
    def load_spans(self, spans: List[Dict[str, Any]]):
        """Load behavioral annotations and connect to verses."""
        for i, span in enumerate(spans):
            span_id = f"span:{i}"
            ref = span.get("reference", {})
            surah = ref.get("surah")
            ayah = ref.get("ayah")
            
            # Create span node
            self.add_node(span_id, NodeType.SPAN, span)
            self.stats["spans"] += 1
            
            # Connect to verse
            if surah and ayah:
                verse_key = f"{surah}:{ayah}"
                verse_id = f"verse:{verse_key}"
                self.add_edge(verse_id, span_id, EdgeType.VERSE_HAS_SPAN)
                self.by_verse[verse_key].add(span_id)
            
            # Extract and connect behaviors
            text = span.get("text_ar", "")
            for ar_keyword, en_label in BEHAVIOR_KEYWORDS.items():
                if ar_keyword in text:
                    behavior_id = f"behavior:{ar_keyword}"
                    
                    # Create behavior node if not exists
                    if behavior_id not in self.nodes:
                        self.add_node(behavior_id, NodeType.BEHAVIOR, {
                            "ar": ar_keyword, "en": en_label
                        })
                        self.stats["behaviors"] += 1
                    
                    # Connect span to behavior
                    self.add_edge(span_id, behavior_id, EdgeType.SPAN_MENTIONS_BEHAVIOR)
                    self.by_behavior[ar_keyword].add(span_id)
            
            # Connect to agent
            agent = span.get("agent", {}).get("type")
            if agent:
                agent_id = f"agent:{agent}"
                if agent_id not in self.nodes:
                    self.add_node(agent_id, NodeType.AGENT, {"type": agent})
                self.add_edge(span_id, agent_id, EdgeType.SPAN_HAS_AGENT)
            
            # Connect to organ
            organ = span.get("organ")
            if organ:
                organ_id = f"organ:{organ}"
                if organ_id not in self.nodes:
                    self.add_node(organ_id, NodeType.ORGAN, {"name": organ})
                self.add_edge(span_id, organ_id, EdgeType.SPAN_HAS_ORGAN)
        
        print(f"[UnifiedGraph] Loaded {self.stats['spans']} spans")
    
    def load_tafsir(self):
        """Load ALL 5 tafsir sources and connect to verses."""
        for source in TAFSIR_SOURCES:
            filepath = TAFSIR_DIR / f"{source}.ar.jsonl"
            if not filepath.exists():
                print(f"[UnifiedGraph] Missing tafsir: {source}")
                continue
            
            count = 0
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        ref = entry.get("reference", {})
                        surah = ref.get("surah")
                        ayah = ref.get("ayah")
                        text = entry.get("text_ar", "")
                        
                        if surah and ayah:
                            verse_key = f"{surah}:{ayah}"
                            tafsir_id = f"tafsir:{source}:{verse_key}"
                            
                            # Create tafsir node
                            self.add_node(tafsir_id, NodeType.TAFSIR, {
                                "source": source,
                                "surah": surah,
                                "ayah": ayah,
                                "text": text,
                            })
                            self.stats["tafsir_entries"] += 1
                            count += 1
                            
                            # Connect to verse (BIDIRECTIONAL via add_edge)
                            verse_id = f"verse:{verse_key}"
                            if verse_id in self.nodes:
                                self.add_edge(verse_id, tafsir_id, EdgeType.VERSE_HAS_TAFSIR)
                            
                            # Extract and connect behaviors from tafsir
                            for ar_keyword, en_label in BEHAVIOR_KEYWORDS.items():
                                if ar_keyword in text:
                                    behavior_id = f"behavior:{ar_keyword}"
                                    
                                    if behavior_id not in self.nodes:
                                        self.add_node(behavior_id, NodeType.BEHAVIOR, {
                                            "ar": ar_keyword, "en": en_label
                                        })
                                        self.stats["behaviors"] += 1
                                    
                                    # Connect tafsir to behavior
                                    self.add_edge(tafsir_id, behavior_id, EdgeType.TAFSIR_MENTIONS_BEHAVIOR)
                                    self.by_behavior[ar_keyword].add(tafsir_id)
            
            print(f"[UnifiedGraph] Loaded {source}: {count} entries")
    
    def build_behavior_relationships(self):
        """Build relationships between behaviors based on co-occurrence."""
        # Find behaviors that co-occur in same verse
        for verse_key, node_ids in self.by_verse.items():
            behaviors_in_verse = set()
            
            for node_id in node_ids:
                # Get behaviors connected to this node
                behavior_neighbors = self.get_neighbors(node_id, EdgeType.SPAN_MENTIONS_BEHAVIOR, "out")
                behaviors_in_verse.update(behavior_neighbors)
            
            # Create co-occurrence edges between behaviors
            behaviors_list = list(behaviors_in_verse)
            for i, b1 in enumerate(behaviors_list):
                for b2 in behaviors_list[i+1:]:
                    self.add_edge(b1, b2, EdgeType.BEHAVIOR_SIMILAR, {"source": "co_occurrence"})
        
        print(f"[UnifiedGraph] Built behavior relationships")
    
    def connect_tafsir_cross_references(self):
        """Connect tafsir entries that discuss the same verse."""
        # Group tafsir by verse
        tafsir_by_verse = defaultdict(list)
        
        for node_id in self.by_type[NodeType.TAFSIR]:
            node = self.nodes[node_id]
            data = node["data"]
            verse_key = f"{data['surah']}:{data['ayah']}"
            tafsir_by_verse[verse_key].append(node_id)
        
        # Connect tafsir entries for same verse
        for verse_key, tafsir_ids in tafsir_by_verse.items():
            if len(tafsir_ids) > 1:
                for i, t1 in enumerate(tafsir_ids):
                    for t2 in tafsir_ids[i+1:]:
                        # Check if they mention same behaviors
                        t1_behaviors = set(self.get_neighbors(t1, EdgeType.TAFSIR_MENTIONS_BEHAVIOR, "out"))
                        t2_behaviors = set(self.get_neighbors(t2, EdgeType.TAFSIR_MENTIONS_BEHAVIOR, "out"))
                        
                        common = t1_behaviors & t2_behaviors
                        if common:
                            self.add_edge(t1, t2, EdgeType.TAFSIR_AGREES, {
                                "common_behaviors": list(common)
                            })
        
        print(f"[UnifiedGraph] Connected tafsir cross-references")
    
    # =========================================================================
    # UNIFIED QUERIES
    # =========================================================================
    
    def query_verse(self, surah: int, ayah: int) -> Dict[str, Any]:
        """
        Get EVERYTHING connected to a verse:
        - All behavioral annotations (spans)
        - All 5 tafsir sources
        - All behaviors mentioned
        - All relationships
        """
        verse_key = f"{surah}:{ayah}"
        verse_id = f"verse:{verse_key}"
        
        if verse_id not in self.nodes:
            return {"error": f"Verse {verse_key} not found"}
        
        result = {
            "verse": verse_key,
            "spans": [],
            "tafsir": {},
            "behaviors": [],
            "agents": [],
            "organs": [],
        }
        
        # Get spans
        span_ids = self.get_neighbors(verse_id, EdgeType.VERSE_HAS_SPAN, "out")
        for span_id in span_ids:
            span_node = self.get_node(span_id)
            if span_node:
                result["spans"].append(span_node["data"])
                
                # Get behaviors from span
                behavior_ids = self.get_neighbors(span_id, EdgeType.SPAN_MENTIONS_BEHAVIOR, "out")
                for b_id in behavior_ids:
                    b_node = self.get_node(b_id)
                    if b_node and b_node["data"]["ar"] not in [b["ar"] for b in result["behaviors"]]:
                        result["behaviors"].append(b_node["data"])
                
                # Get agents
                agent_ids = self.get_neighbors(span_id, EdgeType.SPAN_HAS_AGENT, "out")
                for a_id in agent_ids:
                    a_node = self.get_node(a_id)
                    if a_node and a_node["data"]["type"] not in result["agents"]:
                        result["agents"].append(a_node["data"]["type"])
                
                # Get organs
                organ_ids = self.get_neighbors(span_id, EdgeType.SPAN_HAS_ORGAN, "out")
                for o_id in organ_ids:
                    o_node = self.get_node(o_id)
                    if o_node and o_node["data"]["name"] not in result["organs"]:
                        result["organs"].append(o_node["data"]["name"])
        
        # Get tafsir from ALL 5 sources
        tafsir_ids = self.get_neighbors(verse_id, EdgeType.VERSE_HAS_TAFSIR, "out")
        for tafsir_id in tafsir_ids:
            tafsir_node = self.get_node(tafsir_id)
            if tafsir_node:
                source = tafsir_node["data"]["source"]
                result["tafsir"][source] = {
                    "text": tafsir_node["data"]["text"][:500],
                    "behaviors_mentioned": [],
                }
                
                # Get behaviors mentioned in this tafsir
                t_behavior_ids = self.get_neighbors(tafsir_id, EdgeType.TAFSIR_MENTIONS_BEHAVIOR, "out")
                for b_id in t_behavior_ids:
                    b_node = self.get_node(b_id)
                    if b_node:
                        result["tafsir"][source]["behaviors_mentioned"].append(b_node["data"]["ar"])
        
        return result
    
    def query_behavior(self, behavior: str) -> Dict[str, Any]:
        """
        Get EVERYTHING connected to a behavior:
        - All verses where it appears
        - All spans mentioning it
        - All tafsir discussing it
        - Related behaviors
        """
        behavior_id = f"behavior:{behavior}"
        
        if behavior_id not in self.nodes:
            return {"error": f"Behavior {behavior} not found"}
        
        result = {
            "behavior": behavior,
            "behavior_data": self.get_node(behavior_id)["data"],
            "verses": [],
            "spans": [],
            "tafsir_mentions": {source: [] for source in TAFSIR_SOURCES},
            "related_behaviors": [],
            "statistics": {
                "total_span_mentions": 0,
                "total_tafsir_mentions": 0,
                "by_source": {},
            }
        }
        
        # Get all nodes mentioning this behavior (spans and tafsir)
        mentioning_nodes = self.by_behavior.get(behavior, set())
        
        for node_id in mentioning_nodes:
            node = self.get_node(node_id)
            if not node:
                continue
            
            if node["type"] == NodeType.SPAN:
                result["spans"].append(node["data"])
                result["statistics"]["total_span_mentions"] += 1
                
                # Get verse
                ref = node["data"].get("reference", {})
                verse_key = f"{ref.get('surah')}:{ref.get('ayah')}"
                if verse_key not in result["verses"]:
                    result["verses"].append(verse_key)
            
            elif node["type"] == NodeType.TAFSIR:
                source = node["data"]["source"]
                result["tafsir_mentions"][source].append({
                    "verse": f"{node['data']['surah']}:{node['data']['ayah']}",
                    "text": node["data"]["text"][:200],
                })
                result["statistics"]["total_tafsir_mentions"] += 1
                result["statistics"]["by_source"][source] = result["statistics"]["by_source"].get(source, 0) + 1
        
        # Get related behaviors
        related_ids = self.get_neighbors(behavior_id, EdgeType.BEHAVIOR_SIMILAR, "both")
        for r_id in related_ids[:20]:
            r_node = self.get_node(r_id)
            if r_node:
                result["related_behaviors"].append(r_node["data"]["ar"])
        
        return result
    
    def traverse(self, start_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Traverse the graph from any starting node.
        Returns all connected nodes up to max_depth.
        """
        visited = set()
        levels = {0: [start_id]}
        
        for depth in range(max_depth):
            if depth not in levels:
                break
            
            levels[depth + 1] = []
            for node_id in levels[depth]:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                neighbors = self.get_neighbors(node_id, direction="both")
                for n in neighbors:
                    if n not in visited:
                        levels[depth + 1].append(n)
        
        # Build result
        result = {
            "start": start_id,
            "max_depth": max_depth,
            "nodes_visited": len(visited),
            "by_level": {},
        }
        
        for level, node_ids in levels.items():
            result["by_level"][level] = []
            for node_id in node_ids[:10]:  # Limit per level
                node = self.get_node(node_id)
                if node:
                    result["by_level"][level].append({
                        "id": node_id,
                        "type": node["type"],
                    })
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            **self.stats,
            "node_types": {t: len(ids) for t, ids in self.by_type.items()},
            "edge_types": {t: sum(len(targets) for targets in sources.values()) 
                         for t, sources in self.edges.items()},
        }


# =============================================================================
# SINGLETON AND INITIALIZATION
# =============================================================================

_graph_instance = None

def get_unified_graph(spans: List[Dict[str, Any]] = None) -> UnifiedGraph:
    """Get or create the unified graph instance."""
    global _graph_instance
    
    if _graph_instance is None and spans is not None:
        print("[UnifiedGraph] Building unified graph...")
        start_time = time.time()
        
        _graph_instance = UnifiedGraph()
        _graph_instance.load_verses(spans)
        _graph_instance.load_spans(spans)
        _graph_instance.load_tafsir()
        _graph_instance.build_behavior_relationships()
        _graph_instance.connect_tafsir_cross_references()
        
        elapsed = time.time() - start_time
        print(f"[UnifiedGraph] Built in {elapsed:.2f}s")
        print(f"[UnifiedGraph] Stats: {_graph_instance.get_stats()}")
    
    return _graph_instance


def reset_unified_graph():
    """Reset the graph instance."""
    global _graph_instance
    _graph_instance = None
