"""
Phase 2: Deterministic Answer Generator

Generates answers from computed analysis payloads.
Numbers ONLY from computed payload - LLM cannot invent counts/percentages.

RULES:
- All numbers must come from payload.computed_numbers or payload tables
- Optional LLM rewriter may only rephrase text without adding facts
- Mandatory validator gate before returning LLM-rephrased text
- Deterministic, professional Arabic output with structured headings
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.benchmarks.analysis_payload import AnalysisPayload

logger = logging.getLogger(__name__)


class ValidatorGateError(Exception):
    """Raised when LLM output fails validation."""

    def __init__(self, violations: List[str]):
        self.violations = violations
        super().__init__(f"Validator gate failed: {violations}")


def validate_no_new_claims(
    payload: AnalysisPayload,
    llm_output: str,
    allow_common: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Phase 2: Validator gate for LLM output.

    Ensures LLM did not invent new numbers/percentages not in the payload.

    Args:
        payload: The computed analysis payload
        llm_output: The LLM-generated answer text
        allow_common: Allow common numbers (0-14, 100) without validation

    Returns:
        (is_valid, list_of_violations)
    """
    if not llm_output:
        return True, []

    # Get all valid numbers from payload
    payload_numbers = payload.get_all_numbers()

    # Add common numbers that are always allowed
    if allow_common:
        common_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 100}
        payload_numbers.update(float(n) for n in common_numbers)

    # Extract all numbers from LLM output
    llm_numbers: Set[float] = set()
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%?', llm_output):
        num_str = match.group(1)
        try:
            num = float(num_str)
            llm_numbers.add(num)
            # Also add as int if it's a whole number
            if num == int(num):
                llm_numbers.add(float(int(num)))
        except ValueError:
            pass

    # Check for violations (numbers in LLM output not in payload)
    violations = []
    for num in llm_numbers:
        if num not in payload_numbers:
            # Allow verse references (surah:ayah numbers)
            if _is_likely_verse_reference(num, llm_output):
                continue
            violations.append(f"LLM invented number: {num}")

    is_valid = len(violations) == 0
    if not is_valid:
        logger.warning(f"[VALIDATOR] LLM output failed validation: {violations}")

    return is_valid, violations


def _is_likely_verse_reference(num: float, text: str) -> bool:
    """Check if a number is likely part of a verse reference (surah:ayah)."""
    # Allow surah numbers (1-114) and ayah numbers (1-286)
    if num <= 0:
        return False
    if num <= 286 and num == int(num):
        # Check if it appears in a verse reference context
        int_num = int(num)
        patterns = [
            rf'{int_num}:\d+',  # surah:ayah
            rf'\d+:{int_num}',  # surah:ayah
            rf'سورة.*{int_num}',  # surah mention
            rf'الآية\s*{int_num}',  # ayah mention
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
    return False


def generate_answer(
    payload: AnalysisPayload,
    include_evidence: bool = True,
    max_evidence_items: int = 10,
) -> str:
    """
    Generate deterministic answer from computed payload.

    This is a template-based generator that produces structured Arabic output.
    Numbers ONLY come from payload.computed_numbers.

    Args:
        payload: The computed analysis payload
        include_evidence: Whether to include evidence citations
        max_evidence_items: Maximum evidence items to include

    Returns:
        Structured Arabic answer text
    """
    sections = []

    # 1. Header based on question class
    header = _generate_header(payload)
    sections.append(header)

    # 2. Entity summary
    if payload.entities:
        entity_section = _generate_entity_section(payload)
        sections.append(entity_section)

    # 3. Graph analysis (if applicable)
    if payload.graph_output.paths or payload.graph_output.cycles or payload.graph_output.centrality:
        graph_section = _generate_graph_section(payload)
        sections.append(graph_section)

    # 4. Tables
    for table in payload.tables:
        table_section = _generate_table_section(table)
        sections.append(table_section)

    # 5. Evidence citations
    if include_evidence:
        evidence_section = _generate_evidence_section(payload, max_evidence_items)
        sections.append(evidence_section)

    # 6. Statistics summary
    stats_section = _generate_statistics_section(payload)
    sections.append(stats_section)

    # 7. Gaps acknowledgment (if any)
    if payload.gaps:
        gaps_section = _generate_gaps_section(payload)
        sections.append(gaps_section)

    return "\n\n".join(filter(None, sections))


def _generate_header(payload: AnalysisPayload) -> str:
    """Generate header based on question class."""
    class_headers = {
        "causal_chain": "## تحليل السلاسل السببية",
        "GRAPH_CAUSAL": "## تحليل السلاسل السببية",
        "cross_tafsir_comparative": "## مقارنة التفاسير",
        "CROSS_TAFSIR_ANALYSIS": "## مقارنة التفاسير",
        "behavior_profile_11axis": "## الملف السلوكي الشامل",
        "PROFILE_11D": "## الملف السلوكي الشامل",
        "network_centrality": "## تحليل مركزية الشبكة",
        "GRAPH_METRICS": "## تحليل مركزية الشبكة",
        "state_transition": "## تحولات حالة القلب",
        "HEART_STATE": "## تحولات حالة القلب",
        "agent_attribution": "## تحليل الفاعلين",
        "AGENT_ANALYSIS": "## تحليل الفاعلين",
        "complete_analysis": "## التحليل الشامل",
        "COMPLETE_ANALYSIS": "## التحليل الشامل",
        "free_text": "## الإجابة",
        "FREE_TEXT": "## الإجابة",
    }

    header = class_headers.get(payload.question_class, "## التحليل")
    return f"{header}\n\n**السؤال:** {payload.question[:200]}..."


def _generate_entity_section(payload: AnalysisPayload) -> str:
    """Generate entity summary section."""
    if not payload.entities:
        return ""

    lines = ["### الكيانات المستخلصة", ""]
    lines.append("| المعرف | الاسم العربي | النوع | عدد الورودات |")
    lines.append("|--------|-------------|-------|--------------|")

    for entity in payload.entities[:10]:
        lines.append(
            f"| {entity.entity_id} | {entity.label_ar} | {entity.entity_type} | {entity.total_mentions} |"
        )

    lines.append(f"\n**إجمالي الكيانات:** {payload.computed_numbers.get('entities_resolved', len(payload.entities))}")

    return "\n".join(lines)


def _generate_graph_section(payload: AnalysisPayload) -> str:
    """Generate graph analysis section."""
    lines = ["### تحليل الشبكة السلوكية", ""]

    # Paths
    if payload.graph_output.paths:
        lines.append(f"**المسارات السببية المكتشفة:** {payload.computed_numbers.get('causal_paths_found', len(payload.graph_output.paths))}")
        for i, path in enumerate(payload.graph_output.paths[:5], 1):
            if isinstance(path, list):
                nodes = " → ".join(str(p.get("source", p.get("target", p))) if isinstance(p, dict) else str(p) for p in path)
                lines.append(f"  {i}. {nodes}")
            elif isinstance(path, dict):
                nodes = path.get("nodes", [])
                if nodes:
                    lines.append(f"  {i}. {' → '.join(str(n) for n in nodes)}")
        lines.append("")

    # Cycles
    if payload.graph_output.cycles:
        lines.append(f"**الدورات التعزيزية:** {payload.computed_numbers.get('cycles_found', len(payload.graph_output.cycles))}")
        for i, cycle in enumerate(payload.graph_output.cycles[:3], 1):
            if isinstance(cycle, dict):
                nodes = cycle.get("nodes", [])
                if nodes:
                    lines.append(f"  {i}. {' → '.join(str(n) for n in nodes)}")
        lines.append("")

    # Centrality
    if payload.graph_output.centrality:
        centrality = payload.graph_output.centrality
        lines.append(f"**إجمالي العقد:** {centrality.get('total_nodes', 'N/A')}")
        lines.append(f"**إجمالي الحواف:** {centrality.get('total_edges', 'N/A')}")

        top_nodes = centrality.get("top_by_degree", [])
        if top_nodes:
            lines.append("\n**أعلى العقد مركزية:**")
            for node in top_nodes[:5]:
                lines.append(f"  - {node.get('id', 'N/A')}: درجة {node.get('degree', 0)}")

    # Causal density
    if payload.graph_output.causal_density:
        density = payload.graph_output.causal_density
        lines.append(f"\n**الكثافة السببية الإجمالية:** {density.get('total_causal_edges', 'N/A')} حافة")

        top_outgoing = density.get("outgoing_top10", [])
        if top_outgoing:
            lines.append("\n**أعلى السلوكيات تأثيراً (صادر):**")
            for item in top_outgoing[:5]:
                label = item.get("label", {}).get("ar", item.get("id", ""))
                lines.append(f"  - {label}: {item.get('count', 0)} حواف صادرة")

    return "\n".join(lines)


def _generate_table_section(table) -> str:
    """Generate a table section."""
    lines = [f"### {table.name}", ""]

    if table.columns and table.rows:
        # Header
        lines.append("| " + " | ".join(table.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(table.columns)) + "|")

        # Rows
        for row in table.rows[:10]:
            values = [str(row.get(col, "")) for col in table.columns]
            lines.append("| " + " | ".join(values) + " |")

    # Totals
    if table.totals:
        lines.append("\n**الإجماليات:**")
        for key, value in table.totals.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.2%}" if value <= 1 else f"  - {key}: {value:.2f}")
            else:
                lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def _generate_evidence_section(payload: AnalysisPayload, max_items: int) -> str:
    """Generate evidence citations section."""
    lines = ["### الأدلة والمصادر", ""]

    # Quran verses
    if payload.quran_evidence:
        lines.append(f"**آيات القرآن الكريم:** {payload.computed_numbers.get('quran_verse_count', len(payload.quran_evidence))}")
        for ev in payload.quran_evidence[:max_items]:
            text_preview = ev.text[:100] + "..." if len(ev.text) > 100 else ev.text
            lines.append(f"  - [{ev.verse_key}]: {text_preview}")
        lines.append("")

    # Tafsir sources
    if payload.tafsir_evidence:
        sources_count = payload.computed_numbers.get('tafsir_sources_with_evidence', 0)
        total_chunks = payload.computed_numbers.get('total_tafsir_chunks', 0)
        lines.append(f"**مصادر التفسير:** {sources_count}/7 مصادر، {total_chunks} اقتباس")

        for source, chunks in payload.tafsir_evidence.items():
            if chunks:
                lines.append(f"\n**{source}:** {len(chunks)} اقتباس")
                for chunk in chunks[:2]:
                    text_preview = chunk.text[:80] + "..." if len(chunk.text) > 80 else chunk.text
                    lines.append(f"  > [{chunk.verse_key}]: \"{text_preview}\"")

    return "\n".join(lines)


def _generate_statistics_section(payload: AnalysisPayload) -> str:
    """Generate statistics summary section."""
    lines = ["### الإحصائيات المحسوبة", ""]

    numbers = payload.computed_numbers
    if not numbers:
        return ""

    # Key statistics
    key_stats = [
        ("عدد الكيانات المستخلصة", "entities_resolved"),
        ("عدد آيات القرآن", "quran_verse_count"),
        ("عدد اقتباسات التفسير", "total_tafsir_chunks"),
        ("مصادر التفسير المتاحة", "tafsir_sources_with_evidence"),
        ("المسارات السببية", "causal_paths_found"),
        ("الدورات المكتشفة", "cycles_found"),
        ("الحواف السببية الإجمالية", "total_causal_edges"),
        ("نسبة الاتفاق", "consensus_percentage"),
    ]

    for label, key in key_stats:
        if key in numbers and numbers[key]:
            value = numbers[key]
            if key == "consensus_percentage":
                lines.append(f"- **{label}:** {value:.1f}%")
            elif isinstance(value, float) and value != int(value):
                lines.append(f"- **{label}:** {value:.2f}")
            else:
                lines.append(f"- **{label}:** {int(value)}")

    return "\n".join(lines)


def _generate_gaps_section(payload: AnalysisPayload) -> str:
    """Generate gaps acknowledgment section."""
    if not payload.gaps:
        return ""

    lines = ["### الفجوات والقيود", ""]
    lines.append("**ملاحظة:** البيانات التالية غير متوفرة:")

    gap_descriptions = {
        "no_entities_resolved": "لم يتم استخلاص كيانات من السؤال",
        "no_quran_evidence": "لم يتم العثور على آيات قرآنية متعلقة",
        "no_causal_paths": "لم يتم العثور على مسارات سببية",
        "no_cycles_found": "لم يتم العثور على دورات تعزيزية",
        "no_centrality_computed": "لم يتم حساب المركزية",
    }

    for gap in payload.gaps:
        if gap.startswith("missing_tafsir_"):
            source = gap.replace("missing_tafsir_", "")
            lines.append(f"  - مصدر التفسير '{source}' غير متاح")
        elif gap in gap_descriptions:
            lines.append(f"  - {gap_descriptions[gap]}")
        else:
            lines.append(f"  - {gap}")

    return "\n".join(lines)


def generate_answer_with_llm_rewrite(
    payload: AnalysisPayload,
    llm_rewriter: Optional[callable] = None,
    strict_validation: bool = True,
) -> Tuple[str, bool, List[str]]:
    """
    Generate answer with optional LLM rewriting.

    The LLM may only rephrase the deterministic answer, not add new facts.
    Validator gate ensures no new claims are introduced.

    Args:
        payload: The computed analysis payload
        llm_rewriter: Optional function(text: str) -> str for LLM rewriting
        strict_validation: If True, reject answers that fail validation

    Returns:
        (answer_text, validation_passed, violations)
    """
    # Generate deterministic base answer
    base_answer = generate_answer(payload)

    if llm_rewriter is None:
        return base_answer, True, []

    # LLM rewrite
    try:
        rewritten = llm_rewriter(base_answer)
    except Exception as e:
        logger.warning(f"LLM rewriter failed: {e}")
        return base_answer, True, []

    # Validator gate
    is_valid, violations = validate_no_new_claims(payload, rewritten)

    if not is_valid and strict_validation:
        logger.warning(f"LLM rewrite rejected due to validation failures: {violations}")
        return base_answer, False, violations

    return rewritten, is_valid, violations
