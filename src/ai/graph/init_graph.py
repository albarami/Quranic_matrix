"""
Initialize the QBM Knowledge Graph with vocabulary data.

This script loads all behaviors, agents, and organs from the vocab files
and creates the initial knowledge graph with behavioral relationships.
"""

import json
from pathlib import Path

from .qbm_graph import QBMKnowledgeGraph


def init_qbm_graph(db_path: str = "data/qbm_graph.db") -> QBMKnowledgeGraph:
    """
    Initialize and populate the QBM knowledge graph.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Populated QBMKnowledgeGraph instance.
    """
    graph = QBMKnowledgeGraph(db_path=db_path)
    vocab_dir = Path("vocab")

    # Load behaviors
    behavior_path = vocab_dir / "behavior_concepts.json"
    if behavior_path.exists():
        count = graph.load_behaviors_from_vocab(str(behavior_path))
        print(f"Loaded {count} behaviors")

    # Load agents
    agents_path = vocab_dir / "agents.json"
    if agents_path.exists():
        with open(agents_path, "r", encoding="utf-8") as f:
            agents_data = json.load(f)
        for agent in agents_data.get("agents", []):
            graph.add_agent(
                agent_id=agent["id"],
                name_ar=agent.get("ar", ""),
                agent_type=agent.get("type", "unknown"),
            )
        print(f"Loaded {len(agents_data.get('agents', []))} agents")

    # Load organs
    organs_path = vocab_dir / "organs.json"
    if organs_path.exists():
        with open(organs_path, "r", encoding="utf-8") as f:
            organs_data = json.load(f)
        for organ in organs_data.get("organs", []):
            graph.add_organ(
                organ_id=organ["id"],
                name_ar=organ.get("ar", ""),
                name_en=organ.get("en", ""),
            )
        print(f"Loaded {len(organs_data.get('organs', []))} organs")

    # Add known behavioral relationships (from Bouzidani framework)
    _add_behavioral_relationships(graph)

    # Save the graph
    graph.save()
    print(f"Graph saved to {db_path}")

    return graph


def _add_behavioral_relationships(graph: QBMKnowledgeGraph) -> None:
    """Add known behavioral relationships from the QBM framework."""

    # Causal relationships (based on Quranic analysis)
    causal_pairs = [
        # Heedlessness chain
        ("BEH_COG_HEEDLESSNESS", "BEH_COG_ARROGANCE"),  # الغفلة → الكبر
        ("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION"),  # الكبر → الظلم
        ("BEH_COG_FOLLOWING_DESIRE", "BEH_COG_HEEDLESSNESS"),  # اتباع الهوى → الغفلة

        # Disbelief chain
        ("BEH_SPI_DISBELIEF", "BEH_COG_ARROGANCE"),  # الكفر → الكبر
        ("BEH_SPI_HYPOCRISY", "BEH_SPEECH_LYING"),  # النفاق → الكذب

        # Positive chains
        ("BEH_SPI_FAITH", "BEH_SPI_TAQWA"),  # الإيمان → التقوى
        ("BEH_SPI_TAQWA", "BEH_EMO_PATIENCE"),  # التقوى → الصبر
        ("BEH_EMO_GRATITUDE", "BEH_SPI_FAITH"),  # الشكر → الإيمان

        # Envy chain
        ("BEH_EMO_ENVY", "BEH_EMO_HATRED"),  # الحسد → البغض
        ("BEH_EMO_HATRED", "BEH_SOC_OPPRESSION"),  # البغض → الظلم
    ]

    for cause, effect in causal_pairs:
        if cause in graph.G.nodes and effect in graph.G.nodes:
            graph.add_causal_relationship(cause, effect, confidence=0.9)

    # Opposite relationships
    opposite_pairs = [
        ("BEH_COG_ARROGANCE", "BEH_COG_HUMILITY"),  # الكبر ↔ التواضع
        ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF"),  # الإيمان ↔ الكفر
        ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING"),  # الصدق ↔ الكذب
        ("BEH_EMO_GRATITUDE", "BEH_EMO_INGRATITUDE"),  # الشكر ↔ الكفران
        ("BEH_EMO_PATIENCE", "BEH_EMO_IMPATIENCE"),  # الصبر ↔ الجزع
        ("BEH_EMO_HOPE", "BEH_EMO_FEAR_ALLAH"),  # الرجاء ↔ الخوف (balanced pair)
        ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION"),  # العدل ↔ الظلم
        ("BEH_COG_KNOWLEDGE", "BEH_COG_IGNORANCE"),  # العلم ↔ الجهل
        ("BEH_COG_CERTAINTY", "BEH_COG_DOUBT"),  # اليقين ↔ الشك
        ("BEH_SPI_SINCERITY", "BEH_SPI_SHOWING_OFF"),  # الإخلاص ↔ الرياء
    ]

    for beh1, beh2 in opposite_pairs:
        if beh1 in graph.G.nodes and beh2 in graph.G.nodes:
            graph.add_opposite_relationship(beh1, beh2)

    # Similar behaviors
    similar_pairs = [
        ("BEH_COG_ARROGANCE", "BEH_SPI_SHOWING_OFF"),  # الكبر ~ الرياء
        ("BEH_EMO_FEAR_ALLAH", "BEH_SPI_TAQWA"),  # الخوف من الله ~ التقوى
        ("BEH_SPI_WORSHIP", "BEH_SPI_PRAYER"),  # العبادة ~ الصلاة
        ("BEH_FIN_CHARITY", "BEH_FIN_ZAKAT"),  # الصدقة ~ الزكاة
    ]

    for beh1, beh2 in similar_pairs:
        if beh1 in graph.G.nodes and beh2 in graph.G.nodes:
            graph.add_relationship(beh1, beh2, "SIMILAR_TO")
            graph.add_relationship(beh2, beh1, "SIMILAR_TO")

    print(f"Added behavioral relationships")


if __name__ == "__main__":
    graph = init_qbm_graph()
    stats = graph.get_behavior_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Behaviors: {stats['total_behaviors']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Edge types: {stats['edge_types']}")
