"""
QBM Ontology using RDFLib for behavioral taxonomy and reasoning.

This module provides OWL ontology support for the Quranic Behavior Matrix,
enabling semantic reasoning over behavioral relationships.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from rdflib import Graph, Namespace, Literal, URIRef, BNode
    from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False


# QBM Namespace
QBM_NS = "http://qbm.research/ontology#"


class QBMOntology:
    """OWL Ontology for QBM behavioral taxonomy using RDFLib."""

    def __init__(self, ontology_path: Optional[str] = None):
        """
        Initialize the QBM ontology.

        Args:
            ontology_path: Path to existing ontology file (TTL/RDF/OWL).
        """
        if not RDFLIB_AVAILABLE:
            raise ImportError("rdflib is required. Install with: pip install rdflib")

        self.g = Graph()
        self.QBM = Namespace(QBM_NS)

        # Bind namespaces
        self.g.bind("qbm", self.QBM)
        self.g.bind("owl", OWL)
        self.g.bind("skos", SKOS)

        if ontology_path and Path(ontology_path).exists():
            self.load(ontology_path)
        else:
            self._init_base_ontology()

    def _init_base_ontology(self) -> None:
        """Initialize the base QBM ontology structure."""
        QBM = self.QBM

        # Ontology metadata
        self.g.add((URIRef(QBM_NS), RDF.type, OWL.Ontology))
        self.g.add((URIRef(QBM_NS), RDFS.label, Literal("Quranic Behavior Matrix Ontology", lang="en")))
        self.g.add((URIRef(QBM_NS), RDFS.label, Literal("أنطولوجيا مصفوفة السلوك القرآني", lang="ar")))

        # =====================================================================
        # Core Classes
        # =====================================================================

        # Behavior (root class)
        self.g.add((QBM.Behavior, RDF.type, OWL.Class))
        self.g.add((QBM.Behavior, RDFS.label, Literal("Behavior", lang="en")))
        self.g.add((QBM.Behavior, RDFS.label, Literal("سلوك", lang="ar")))

        # Behavior subclasses by organ
        behavior_subclasses = [
            ("HeartBehavior", "Heart Behavior", "سلوك القلب"),
            ("TongueBehavior", "Tongue Behavior", "سلوك اللسان"),
            ("LimbBehavior", "Limb Behavior", "سلوك الجوارح"),
            ("EyeBehavior", "Eye Behavior", "سلوك العين"),
            ("EarBehavior", "Ear Behavior", "سلوك الأذن"),
        ]

        for class_id, label_en, label_ar in behavior_subclasses:
            self.g.add((QBM[class_id], RDF.type, OWL.Class))
            self.g.add((QBM[class_id], RDFS.subClassOf, QBM.Behavior))
            self.g.add((QBM[class_id], RDFS.label, Literal(label_en, lang="en")))
            self.g.add((QBM[class_id], RDFS.label, Literal(label_ar, lang="ar")))

        # Behavior subclasses by category
        category_classes = [
            ("CognitiveBehavior", "Cognitive Behavior", "سلوك معرفي"),
            ("EmotionalBehavior", "Emotional Behavior", "سلوك عاطفي"),
            ("SpiritualBehavior", "Spiritual Behavior", "سلوك روحي"),
            ("SocialBehavior", "Social Behavior", "سلوك اجتماعي"),
            ("SpeechBehavior", "Speech Behavior", "سلوك لساني"),
            ("FinancialBehavior", "Financial Behavior", "سلوك مالي"),
            ("PhysicalBehavior", "Physical Behavior", "سلوك جسدي"),
        ]

        for class_id, label_en, label_ar in category_classes:
            self.g.add((QBM[class_id], RDF.type, OWL.Class))
            self.g.add((QBM[class_id], RDFS.subClassOf, QBM.Behavior))
            self.g.add((QBM[class_id], RDFS.label, Literal(label_en, lang="en")))
            self.g.add((QBM[class_id], RDFS.label, Literal(label_ar, lang="ar")))

        # Agent class
        self.g.add((QBM.Agent, RDF.type, OWL.Class))
        self.g.add((QBM.Agent, RDFS.label, Literal("Agent", lang="en")))
        self.g.add((QBM.Agent, RDFS.label, Literal("فاعل", lang="ar")))

        # Organ class
        self.g.add((QBM.Organ, RDF.type, OWL.Class))
        self.g.add((QBM.Organ, RDFS.label, Literal("Organ", lang="en")))
        self.g.add((QBM.Organ, RDFS.label, Literal("عضو", lang="ar")))

        # Ayah class
        self.g.add((QBM.Ayah, RDF.type, OWL.Class))
        self.g.add((QBM.Ayah, RDFS.label, Literal("Ayah", lang="en")))
        self.g.add((QBM.Ayah, RDFS.label, Literal("آية", lang="ar")))

        # Consequence class
        self.g.add((QBM.Consequence, RDF.type, OWL.Class))
        self.g.add((QBM.Consequence, RDFS.label, Literal("Consequence", lang="en")))
        self.g.add((QBM.Consequence, RDFS.label, Literal("نتيجة", lang="ar")))

        # =====================================================================
        # Object Properties (Relationships)
        # =====================================================================

        # hasCause - behavior causes another behavior
        self.g.add((QBM.hasCause, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.hasCause, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.hasCause, RDFS.range, QBM.Behavior))
        self.g.add((QBM.hasCause, RDFS.label, Literal("has cause", lang="en")))
        self.g.add((QBM.hasCause, RDFS.label, Literal("له سبب", lang="ar")))

        # resultsIn - inverse of hasCause
        self.g.add((QBM.resultsIn, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.resultsIn, OWL.inverseOf, QBM.hasCause))
        self.g.add((QBM.resultsIn, RDFS.label, Literal("results in", lang="en")))
        self.g.add((QBM.resultsIn, RDFS.label, Literal("يؤدي إلى", lang="ar")))

        # oppositeOf - symmetric property
        self.g.add((QBM.oppositeOf, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.oppositeOf, RDF.type, OWL.SymmetricProperty))
        self.g.add((QBM.oppositeOf, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.oppositeOf, RDFS.range, QBM.Behavior))
        self.g.add((QBM.oppositeOf, RDFS.label, Literal("opposite of", lang="en")))
        self.g.add((QBM.oppositeOf, RDFS.label, Literal("ضد", lang="ar")))

        # similarTo - symmetric property
        self.g.add((QBM.similarTo, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.similarTo, RDF.type, OWL.SymmetricProperty))
        self.g.add((QBM.similarTo, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.similarTo, RDFS.range, QBM.Behavior))
        self.g.add((QBM.similarTo, RDFS.label, Literal("similar to", lang="en")))
        self.g.add((QBM.similarTo, RDFS.label, Literal("مشابه لـ", lang="ar")))

        # performedBy - behavior performed by agent
        self.g.add((QBM.performedBy, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.performedBy, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.performedBy, RDFS.range, QBM.Agent))

        # involvesOrgan - behavior involves organ
        self.g.add((QBM.involvesOrgan, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.involvesOrgan, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.involvesOrgan, RDFS.range, QBM.Organ))

        # mentionedIn - behavior mentioned in ayah
        self.g.add((QBM.mentionedIn, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.mentionedIn, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.mentionedIn, RDFS.range, QBM.Ayah))

        # leadsTo - behavior leads to consequence
        self.g.add((QBM.leadsTo, RDF.type, OWL.ObjectProperty))
        self.g.add((QBM.leadsTo, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.leadsTo, RDFS.range, QBM.Consequence))

        # =====================================================================
        # Data Properties
        # =====================================================================

        # behaviorId
        self.g.add((QBM.behaviorId, RDF.type, OWL.DatatypeProperty))
        self.g.add((QBM.behaviorId, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.behaviorId, RDFS.range, XSD.string))

        # nameArabic
        self.g.add((QBM.nameArabic, RDF.type, OWL.DatatypeProperty))
        self.g.add((QBM.nameArabic, RDFS.range, XSD.string))

        # nameEnglish
        self.g.add((QBM.nameEnglish, RDF.type, OWL.DatatypeProperty))
        self.g.add((QBM.nameEnglish, RDFS.range, XSD.string))

        # polarity (positive/negative)
        self.g.add((QBM.polarity, RDF.type, OWL.DatatypeProperty))
        self.g.add((QBM.polarity, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.polarity, RDFS.range, XSD.string))

        # evaluation (praise/blame)
        self.g.add((QBM.evaluation, RDF.type, OWL.DatatypeProperty))
        self.g.add((QBM.evaluation, RDFS.domain, QBM.Behavior))
        self.g.add((QBM.evaluation, RDFS.range, XSD.string))

    # =========================================================================
    # Instance Creation
    # =========================================================================

    def add_behavior(
        self,
        behavior_id: str,
        name_ar: str,
        name_en: str,
        category: str,
        polarity: str = "neutral",
        evaluation: str = "neutral",
    ) -> URIRef:
        """
        Add a behavior instance to the ontology.

        Args:
            behavior_id: Unique behavior ID (e.g., BEH_COG_ARROGANCE).
            name_ar: Arabic name.
            name_en: English name.
            category: Behavior category.
            polarity: positive/negative/neutral.
            evaluation: praise/blame/neutral.

        Returns:
            URIRef of the created behavior.
        """
        QBM = self.QBM
        behavior_uri = QBM[behavior_id]

        # Determine class based on category
        category_map = {
            "cognitive": QBM.CognitiveBehavior,
            "emotional": QBM.EmotionalBehavior,
            "spiritual": QBM.SpiritualBehavior,
            "social": QBM.SocialBehavior,
            "speech": QBM.SpeechBehavior,
            "financial": QBM.FinancialBehavior,
            "physical": QBM.PhysicalBehavior,
        }
        behavior_class = category_map.get(category.lower(), QBM.Behavior)

        # Add instance
        self.g.add((behavior_uri, RDF.type, behavior_class))
        self.g.add((behavior_uri, QBM.behaviorId, Literal(behavior_id)))
        self.g.add((behavior_uri, QBM.nameArabic, Literal(name_ar, lang="ar")))
        self.g.add((behavior_uri, QBM.nameEnglish, Literal(name_en, lang="en")))
        self.g.add((behavior_uri, RDFS.label, Literal(name_ar, lang="ar")))
        self.g.add((behavior_uri, RDFS.label, Literal(name_en, lang="en")))
        self.g.add((behavior_uri, QBM.polarity, Literal(polarity)))
        self.g.add((behavior_uri, QBM.evaluation, Literal(evaluation)))

        return behavior_uri

    def add_causal_relationship(self, cause_id: str, effect_id: str) -> None:
        """Add a causal relationship between behaviors."""
        QBM = self.QBM
        self.g.add((QBM[effect_id], QBM.hasCause, QBM[cause_id]))

    def add_opposite_relationship(self, behavior1_id: str, behavior2_id: str) -> None:
        """Add an opposite relationship between behaviors."""
        QBM = self.QBM
        self.g.add((QBM[behavior1_id], QBM.oppositeOf, QBM[behavior2_id]))

    def add_similar_relationship(self, behavior1_id: str, behavior2_id: str) -> None:
        """Add a similarity relationship between behaviors."""
        QBM = self.QBM
        self.g.add((QBM[behavior1_id], QBM.similarTo, QBM[behavior2_id]))

    # =========================================================================
    # SPARQL Queries
    # =========================================================================

    def query(self, sparql: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query.

        Args:
            sparql: SPARQL query string.

        Returns:
            List of result dictionaries.
        """
        results = []
        for row in self.g.query(sparql):
            result = {}
            for var in row.labels:
                val = row[var]
                if val is not None:
                    result[str(var)] = str(val)
            results.append(result)
        return results

    def get_all_behaviors(self) -> List[Dict[str, str]]:
        """Get all behaviors in the ontology."""
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?behavior ?id ?nameAr ?nameEn
            WHERE {
                ?behavior a/rdfs:subClassOf* qbm:Behavior .
                ?behavior qbm:behaviorId ?id .
                OPTIONAL { ?behavior qbm:nameArabic ?nameAr }
                OPTIONAL { ?behavior qbm:nameEnglish ?nameEn }
            }
        """
        return self.query(sparql)

    def get_causes(self, behavior_id: str) -> List[str]:
        """Get all causes of a behavior."""
        sparql = f"""
            PREFIX qbm: <http://qbm.research/ontology#>
            
            SELECT ?cause
            WHERE {{
                qbm:{behavior_id} qbm:hasCause ?cause .
            }}
        """
        results = self.query(sparql)
        return [r["cause"].split("#")[-1] for r in results]

    def get_effects(self, behavior_id: str) -> List[str]:
        """Get all effects of a behavior."""
        sparql = f"""
            PREFIX qbm: <http://qbm.research/ontology#>
            
            SELECT ?effect
            WHERE {{
                ?effect qbm:hasCause qbm:{behavior_id} .
            }}
        """
        results = self.query(sparql)
        return [r["effect"].split("#")[-1] for r in results]

    def get_opposites(self, behavior_id: str) -> List[str]:
        """Get all opposites of a behavior."""
        sparql = f"""
            PREFIX qbm: <http://qbm.research/ontology#>
            
            SELECT ?opposite
            WHERE {{
                qbm:{behavior_id} qbm:oppositeOf ?opposite .
            }}
        """
        results = self.query(sparql)
        return [r["opposite"].split("#")[-1] for r in results]

    def find_causal_chain(self, start_id: str, end_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Find causal chains between two behaviors using SPARQL property paths.

        Note: This is a simplified version. For complex paths, use the graph module.
        """
        # Check direct path first
        sparql = f"""
            PREFIX qbm: <http://qbm.research/ontology#>
            
            ASK {{
                qbm:{end_id} qbm:hasCause+ qbm:{start_id} .
            }}
        """
        result = self.g.query(sparql)
        if result.askAnswer:
            # Path exists, but SPARQL property paths don't return the path itself
            # Return a placeholder indicating path exists
            return [[start_id, "...", end_id]]
        return []

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str, format: str = "turtle") -> None:
        """
        Save ontology to file.

        Args:
            path: Output file path.
            format: RDF format (turtle, xml, n3, nt).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=path, format=format)

    def load(self, path: str) -> None:
        """
        Load ontology from file.

        Args:
            path: Input file path.
        """
        # Detect format from extension
        ext = Path(path).suffix.lower()
        format_map = {
            ".ttl": "turtle",
            ".rdf": "xml",
            ".owl": "xml",
            ".n3": "n3",
            ".nt": "nt",
        }
        fmt = format_map.get(ext, "turtle")
        self.g.parse(path, format=fmt)

    # =========================================================================
    # Bulk Loading
    # =========================================================================

    def load_from_graph(self, graph) -> int:
        """
        Load behaviors and relationships from QBMKnowledgeGraph.

        Args:
            graph: QBMKnowledgeGraph instance.

        Returns:
            Number of behaviors loaded.
        """
        count = 0

        # Load behaviors
        behaviors = graph.get_nodes_by_type("Behavior")
        for behavior_id, attrs in behaviors:
            self.add_behavior(
                behavior_id=behavior_id,
                name_ar=attrs.get("name_ar", ""),
                name_en=attrs.get("name_en", ""),
                category=attrs.get("category", ""),
            )
            count += 1

        # Load relationships
        for source, target, data in graph.G.edges(data=True):
            edge_type = data.get("edge_type", "")
            if edge_type == "CAUSES":
                self.add_causal_relationship(source, target)
            elif edge_type == "OPPOSITE_OF":
                self.add_opposite_relationship(source, target)
            elif edge_type == "SIMILAR_TO":
                self.add_similar_relationship(source, target)

        return count

    def get_statistics(self) -> Dict[str, int]:
        """Get ontology statistics."""
        stats = {
            "total_triples": len(self.g),
            "behaviors": 0,
            "classes": 0,
            "properties": 0,
        }

        # Count behaviors
        sparql = """
            PREFIX qbm: <http://qbm.research/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT (COUNT(?b) as ?count)
            WHERE {
                ?b a/rdfs:subClassOf* qbm:Behavior .
                ?b qbm:behaviorId ?id .
            }
        """
        for row in self.g.query(sparql):
            stats["behaviors"] = int(row[0])

        # Count classes
        for _ in self.g.subjects(RDF.type, OWL.Class):
            stats["classes"] += 1

        # Count properties
        for _ in self.g.subjects(RDF.type, OWL.ObjectProperty):
            stats["properties"] += 1
        for _ in self.g.subjects(RDF.type, OWL.DatatypeProperty):
            stats["properties"] += 1

        return stats
