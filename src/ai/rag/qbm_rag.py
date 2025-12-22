"""
QBM RAG Pipeline using ChromaDB + NetworkX + Azure OpenAI.

This module provides the Retrieval-Augmented Generation pipeline for
answering questions about Quranic behaviors using the QBM framework.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False


class QBMRAGPipeline:
    """RAG pipeline using ChromaDB + NetworkX + Azure OpenAI."""

    # System prompt for QBM analysis
    SYSTEM_PROMPT_AR = """أنت عالم متخصص في تحليل السلوك القرآني وفق منهجية مصفوفة السلوك القرآني (QBM).

المنهجية تتضمن 11 بُعداً للتحليل:
1. الفاعل (Agent): من يقوم بالسلوك
2. العضو (Organ): العضو المرتبط بالسلوك
3. المصدر (Source): مصدر السلوك
4. التقييم (Evaluation): مدح أو ذم
5. السياق المكاني (Spatial): دنيا أو آخرة
6. السياق الزماني (Temporal): ماضي، حاضر، مستقبل
7. السياق النظامي (Systemic): فردي أو جماعي
8. الإشارة الشرعية (Deontic): واجب، مندوب، مباح، مكروه، حرام
9. شكل السلوك (Form): فعل، ترك، صفة
10. القطبية (Polarity): إيجابي أو سلبي
11. الثقة (Confidence): درجة اليقين

أجب بناءً على السياق المقدم فقط. استخدم الآيات والتفسير لدعم إجابتك.
اذكر مراجع الآيات بصيغة (السورة: الآية)."""

    SYSTEM_PROMPT_EN = """You are a scholar specialized in Quranic behavioral analysis using the QBM (Quranic Behavior Matrix) methodology.

The methodology includes 11 dimensions of analysis:
1. Agent: Who performs the behavior
2. Organ: The organ associated with the behavior
3. Source: Origin of the behavior
4. Evaluation: Praise or blame
5. Spatial context: Dunya or Akhira
6. Temporal context: Past, present, future
7. Systemic context: Individual or collective
8. Deontic signal: Obligatory, recommended, permissible, disliked, forbidden
9. Behavior form: Action, omission, attribute
10. Polarity: Positive or negative
11. Confidence: Degree of certainty

Answer based only on the provided context. Use verses and tafsir to support your answer.
Cite verse references in the format (Surah: Ayah)."""

    def __init__(
        self,
        vector_store=None,
        graph=None,
        language: str = "ar",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: QBMVectorStore instance (created if None).
            graph: QBMKnowledgeGraph instance (created if None).
            language: Response language ("ar" or "en").
        """
        # Initialize vector store
        if vector_store is None:
            from ..vectors.qbm_vectors import QBMVectorStore
            self.vector_store = QBMVectorStore()
        else:
            self.vector_store = vector_store

        # Initialize graph
        if graph is None:
            from ..graph.qbm_graph import QBMKnowledgeGraph
            self.graph = QBMKnowledgeGraph()
            self.graph.load()
        else:
            self.graph = graph

        # Initialize Azure OpenAI client
        self.client = None
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
        self._init_azure_client()

        # Set language
        self.language = language
        self.system_prompt = (
            self.SYSTEM_PROMPT_AR if language == "ar" else self.SYSTEM_PROMPT_EN
        )

    def _init_azure_client(self) -> None:
        """Initialize Azure OpenAI client from environment variables."""
        if not AZURE_OPENAI_AVAILABLE:
            print("Warning: openai package not available. LLM generation disabled.")
            return

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        if not api_key or not endpoint:
            print("Warning: Azure OpenAI credentials not found in environment.")
            return

        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            print(f"Azure OpenAI client initialized (deployment: {self.deployment})")
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI client: {e}")

    # -------------------------------------------------------------------------
    # Main Query Interface
    # -------------------------------------------------------------------------

    def query(
        self,
        question: str,
        n_ayat: int = 10,
        n_behaviors: int = 5,
        n_tafsir: int = 5,
        expand_graph: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG query with graph expansion.

        Args:
            question: User question in Arabic or English.
            n_ayat: Number of ayat to retrieve.
            n_behaviors: Number of behaviors to retrieve.
            n_tafsir: Number of tafsir entries to retrieve.
            expand_graph: Whether to expand context via graph.

        Returns:
            Dict with answer, sources, behaviors, and graph expansion.
        """
        # Step 1: Retrieve relevant ayat
        ayat_results = self.vector_store.search_ayat(question, n=n_ayat)

        # Step 2: Retrieve relevant behaviors
        behavior_results = self.vector_store.search_behaviors(question, n=n_behaviors)

        # Step 3: Expand via graph (get related behaviors, causes, effects)
        expanded = {}
        if expand_graph and behavior_results.get("ids"):
            behavior_ids = behavior_results["ids"][0] if behavior_results["ids"] else []
            expanded = self._expand_graph_context(behavior_ids)

        # Step 4: Retrieve tafsir for relevant ayat
        tafsir_results = self.vector_store.search_tafsir(question, n=n_tafsir)

        # Step 5: Build context and generate response
        context = self._build_context(
            ayat_results, behavior_results, expanded, tafsir_results
        )
        answer = self._generate(question, context)

        return {
            "answer": answer,
            "sources": {
                "ayat": ayat_results.get("ids", [[]])[0],
                "behaviors": behavior_results.get("ids", [[]])[0],
                "tafsir": tafsir_results.get("ids", [[]])[0],
            },
            "graph_expansion": expanded,
            "context_used": context[:2000] + "..." if len(context) > 2000 else context,
        }

    def analyze_behavior(
        self,
        behavior: str,
        depth: str = "full",
    ) -> Dict[str, Any]:
        """
        Analyze a specific behavior using the QBM framework.

        Args:
            behavior: Behavior name in Arabic or English.
            depth: Analysis depth ("brief", "standard", "full").

        Returns:
            Comprehensive behavioral analysis.
        """
        # Find the behavior in the graph
        behavior_results = self.vector_store.search_behaviors(behavior, n=1)

        if not behavior_results.get("ids") or not behavior_results["ids"][0]:
            return {"error": f"Behavior not found: {behavior}"}

        behavior_id = behavior_results["ids"][0][0]
        behavior_data = self.graph.get_node(behavior_id)

        # Get relationships from graph
        causes = self._get_related_behaviors(behavior_id, "CAUSES", "in")
        effects = self._get_related_behaviors(behavior_id, "CAUSES", "out")
        opposites = self._get_related_behaviors(behavior_id, "OPPOSITE_OF", "both")
        similar = self._get_related_behaviors(behavior_id, "SIMILAR_TO", "both")

        # Get relevant ayat
        ayat = self.vector_store.search_ayat(behavior, n=10)

        # Get tafsir insights
        tafsir = self.vector_store.search_tafsir(behavior, n=5)

        # Generate analysis
        prompt = self._build_analysis_prompt(
            behavior, behavior_data, causes, effects, opposites, ayat, tafsir, depth
        )
        analysis = self._generate(prompt, "")

        return {
            "behavior": behavior,
            "behavior_id": behavior_id,
            "data": behavior_data,
            "causes": causes,
            "effects": effects,
            "opposites": opposites,
            "similar": similar,
            "ayat": ayat.get("ids", [[]])[0],
            "analysis": analysis,
        }

    def discover_chains(
        self,
        start_behavior: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """
        Discover causal chains starting from a behavior.

        Args:
            start_behavior: Starting behavior name or ID.
            max_depth: Maximum chain length.

        Returns:
            List of causal chains.
        """
        # Find behavior ID
        if start_behavior.startswith("BEH_"):
            behavior_id = start_behavior
        else:
            results = self.vector_store.search_behaviors(start_behavior, n=1)
            if not results.get("ids") or not results["ids"][0]:
                return []
            behavior_id = results["ids"][0][0]

        # Find all paths to consequences
        chains = []
        consequences = self.graph.get_nodes_by_type("Consequence")

        for conseq_id, _ in consequences:
            paths = self.graph.find_causal_chain(behavior_id, conseq_id, max_depth)
            chains.extend(paths)

        # Also find chains to other behaviors
        behaviors = self.graph.get_nodes_by_type("Behavior")
        for target_id, _ in behaviors:
            if target_id != behavior_id:
                paths = self.graph.find_causal_chain(behavior_id, target_id, max_depth)
                chains.extend(paths)

        return chains

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    def _expand_graph_context(self, behavior_ids: List[str]) -> Dict[str, List[str]]:
        """Get related behaviors from graph."""
        expanded = {"causes": [], "effects": [], "opposites": [], "similar": []}

        for bid in behavior_ids:
            if bid not in self.graph.G:
                continue

            # Get incoming edges (causes)
            for source, _, data in self.graph.G.in_edges(bid, data=True):
                edge_type = data.get("edge_type")
                if edge_type == "CAUSES":
                    node_data = self.graph.get_node(source)
                    if node_data:
                        expanded["causes"].append(node_data.get("name_ar", source))

            # Get outgoing edges (effects, opposites, similar)
            for _, target, data in self.graph.G.out_edges(bid, data=True):
                edge_type = data.get("edge_type")
                node_data = self.graph.get_node(target)
                name = node_data.get("name_ar", target) if node_data else target

                if edge_type == "CAUSES" or edge_type == "RESULTS_IN":
                    expanded["effects"].append(name)
                elif edge_type == "OPPOSITE_OF":
                    expanded["opposites"].append(name)
                elif edge_type == "SIMILAR_TO":
                    expanded["similar"].append(name)

        # Remove duplicates
        for key in expanded:
            expanded[key] = list(set(expanded[key]))

        return expanded

    def _get_related_behaviors(
        self, behavior_id: str, rel_type: str, direction: str
    ) -> List[Dict[str, str]]:
        """Get related behaviors by relationship type."""
        related = []
        rels = self.graph.get_relationships(behavior_id, rel_type, direction)

        for source, target, data in rels:
            other_id = target if source == behavior_id else source
            node_data = self.graph.get_node(other_id)
            if node_data:
                related.append({
                    "id": other_id,
                    "name_ar": node_data.get("name_ar", ""),
                    "name_en": node_data.get("name_en", ""),
                })

        return related

    def _build_context(
        self,
        ayat: Dict,
        behaviors: Dict,
        expanded: Dict,
        tafsir: Dict,
    ) -> str:
        """Build context string for LLM."""
        parts = []

        # Add ayat
        if ayat.get("documents") and ayat["documents"][0]:
            parts.append("## الآيات ذات الصلة:")
            for i, (doc, meta) in enumerate(
                zip(ayat["documents"][0][:5], ayat.get("metadatas", [[]])[0][:5])
            ):
                ref = ayat["ids"][0][i] if ayat.get("ids") else ""
                parts.append(f"- ({ref}): {doc[:200]}")

        # Add behaviors
        if behaviors.get("documents") and behaviors["documents"][0]:
            parts.append("\n## السلوكيات:")
            for doc in behaviors["documents"][0][:3]:
                parts.append(f"- {doc}")

        # Add graph expansion
        if expanded:
            if expanded.get("causes"):
                parts.append(f"\n## الأسباب: {', '.join(expanded['causes'][:5])}")
            if expanded.get("effects"):
                parts.append(f"## النتائج: {', '.join(expanded['effects'][:5])}")
            if expanded.get("opposites"):
                parts.append(f"## الأضداد: {', '.join(expanded['opposites'][:5])}")

        # Add tafsir
        if tafsir.get("documents") and tafsir["documents"][0]:
            parts.append("\n## التفسير:")
            for doc in tafsir["documents"][0][:3]:
                parts.append(f"- {doc[:300]}...")

        return "\n".join(parts)

    def _build_analysis_prompt(
        self,
        behavior: str,
        behavior_data: Dict,
        causes: List,
        effects: List,
        opposites: List,
        ayat: Dict,
        tafsir: Dict,
        depth: str,
    ) -> str:
        """Build prompt for behavioral analysis."""
        name_ar = behavior_data.get("name_ar", behavior) if behavior_data else behavior
        name_en = behavior_data.get("name_en", "") if behavior_data else ""

        prompt = f"حلل سلوك {name_ar}"
        if name_en:
            prompt += f" ({name_en})"
        prompt += " في القرآن الكريم"

        if depth == "full":
            prompt += " بشكل شامل يتضمن: التعريف، الأسباب، النتائج، الأضداد، الآيات المتعلقة، والتفسير."
        elif depth == "standard":
            prompt += " مع ذكر الأسباب والنتائج والآيات الرئيسية."
        else:
            prompt += " بإيجاز."

        return prompt

    # -------------------------------------------------------------------------
    # LLM Generation
    # -------------------------------------------------------------------------

    def _generate(self, question: str, context: str) -> str:
        """Generate response using Azure OpenAI."""
        if self.client is None:
            return "[LLM not available - Azure OpenAI client not initialized]"

        user_prompt = f"""السياق:
{context}

السؤال: {question}""" if context else question

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error generating response: {e}]"

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline components."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "graph": self.graph.get_behavior_statistics(),
            "llm_available": self.client is not None,
            "deployment": self.deployment,
        }
