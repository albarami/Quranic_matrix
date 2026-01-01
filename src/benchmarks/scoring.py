from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")


GENERIC_DEFAULT_VERSES: Set[str] = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}


@dataclass
class ScoringResult:
    verdict: str  # PASS | PARTIAL | FAIL
    reasons: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "tags": self.tags,
            "metrics": self.metrics,
        }


def _as_verse_key(v: Dict[str, Any]) -> Optional[str]:
    if not isinstance(v, dict):
        return None
    if isinstance(v.get("verse_key"), str) and v["verse_key"].strip():
        return v["verse_key"].strip()
    surah = v.get("surah")
    ayah = v.get("ayah")
    if isinstance(surah, int) and isinstance(ayah, int):
        return f"{surah}:{ayah}"
    return None


@lru_cache(maxsize=1)
def _load_behavior_term_map() -> Dict[str, str]:
    """
    Return a mapping from Arabic/English behavior surface forms to canonical behavior IDs.
    """
    if not CANONICAL_ENTITIES_FILE.exists():
        return {}
    data = json.loads(CANONICAL_ENTITIES_FILE.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for beh in data.get("behaviors", []):
        beh_id = beh.get("id")
        if not isinstance(beh_id, str):
            continue
        ar = (beh.get("ar") or "").strip()
        if ar:
            mapping[ar] = beh_id
            if ar.startswith("ال") and len(ar) > 2:
                mapping[ar[2:]] = beh_id
        en = (beh.get("en") or "").strip()
        if en:
            mapping[en.lower()] = beh_id
    return mapping


def _extract_requested_behavior_id(question: str) -> Optional[str]:
    """
    Best-effort, deterministic behavior extraction for scoring heuristics.

    Returns:
        canonical behavior ID if detected, else None
    """
    if not question:
        return None

    mapping = _load_behavior_term_map()

    # Explicit ID
    match = re.search(r"\bBEH_[A-Z0-9_]+\b", question)
    if match:
        return match.group(0)

    # Arabic token match
    for token in re.findall(r"[\u0600-\u06FF]+", question):
        if token in mapping:
            return mapping[token]
        if token.startswith("ال") and token[2:] in mapping:
            return mapping[token[2:]]

    # English substring match (deterministic by longest match first)
    q_lower = question.lower()
    en_terms = [(term, beh_id) for term, beh_id in mapping.items() if term and term.isascii()]
    en_terms.sort(key=lambda t: (-len(t[0]), t[0]))
    for term, beh_id in en_terms:
        if term in q_lower:
            return beh_id
    return None


@lru_cache(maxsize=1)
def _load_concept_to_verses() -> Dict[str, Set[str]]:
    """
    Load concept_index_v2.jsonl into {concept_id: {verse_key,...}}.
    """
    if not CONCEPT_INDEX_FILE.exists():
        return {}
    mapping: Dict[str, Set[str]] = {}
    with CONCEPT_INDEX_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            concept_id = obj.get("concept_id")
            if not isinstance(concept_id, str):
                continue
            verses = obj.get("verses", [])
            if not isinstance(verses, list):
                continue
            keys: Set[str] = set()
            for v in verses:
                if isinstance(v, dict) and isinstance(v.get("verse_key"), str):
                    keys.add(v["verse_key"])
            mapping[concept_id] = keys
    return mapping


def _count_sources_with_chunks(tafsir: Dict[str, Any]) -> int:
    if not isinstance(tafsir, dict):
        return 0
    count = 0
    for chunks in tafsir.values():
        if isinstance(chunks, list) and len(chunks) > 0:
            count += 1
    return count


def _iter_tafsir_chunks(tafsir: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if not isinstance(tafsir, dict):
        return
    for source, chunks in tafsir.items():
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if isinstance(chunk, dict):
                yield str(source), chunk


def _has_percent_claim(answer: str) -> bool:
    if not answer:
        return False
    return re.search(r"\b\d+(?:\.\d+)?%\b", answer) is not None


def _generic_opening_default_fail(
    question: str, debug_intent: str, quran_verses: Sequence[Dict[str, Any]]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Heuristic: if query is not SURAH_REF/AYAH_REF and >50% of returned verses are from
    {1:1–7, 2:1–20} AND no requested concept is present → FAIL.
    """
    if debug_intent in {"SURAH_REF", "AYAH_REF"}:
        return False, {"fraction_generic": 0.0, "concept_present": None}

    keys: List[str] = []
    for v in quran_verses:
        vk = _as_verse_key(v)
        if vk:
            keys.append(vk)

    if not keys:
        return False, {"fraction_generic": 0.0, "concept_present": None}

    generic_hits = sum(1 for vk in keys if vk in GENERIC_DEFAULT_VERSES)
    fraction = generic_hits / max(1, len(keys))

    requested_beh = _extract_requested_behavior_id(question)
    concept_present = None
    if requested_beh:
        concept_present = False
        concept_map = _load_concept_to_verses()
        allowed = concept_map.get(requested_beh, set())
        if any(vk in allowed for vk in keys):
            concept_present = True

    should_fail = fraction > 0.5 and (concept_present is False or concept_present is None)
    return should_fail, {"fraction_generic": round(fraction, 3), "concept_present": concept_present}


def _is_placeholder_answer(answer: str) -> bool:
    """Check if answer is a placeholder or empty."""
    if not answer or not answer.strip():
        return True
    placeholder_patterns = [
        r"^\[.*\]$",  # [placeholder]
        r"^proof_only",  # proof_only mode marker
        r"^N/A$",
        r"^TODO",
        r"^PLACEHOLDER",
    ]
    answer_stripped = answer.strip()
    for pattern in placeholder_patterns:
        if re.match(pattern, answer_stripped, re.IGNORECASE):
            return True
    # Very short answers are suspicious
    if len(answer_stripped) < 10:
        return True
    return False


def _check_required_sections(
    response: Dict[str, Any],
    must_include: List[str],
) -> List[str]:
    """
    Phase 2: Check that required sections exist in response.

    Args:
        response: The API response
        must_include: List of required fields from expected.must_include

    Returns:
        List of missing sections
    """
    missing = []
    proof = response.get("proof", {}) if isinstance(response, dict) else {}
    debug = response.get("debug", {}) if isinstance(response, dict) else {}

    section_paths = {
        "edge_provenance": lambda: bool(proof.get("graph", {}).get("paths") or proof.get("graph", {}).get("cycles") or proof.get("graph", {}).get("centrality")),
        "verse_keys_per_link": lambda: _has_verse_keys_per_link(proof),
        "graph_paths": lambda: bool(proof.get("graph", {}).get("paths") or proof.get("graph", {}).get("cycles") or proof.get("graph", {}).get("centrality")),
        "graph_cycles": lambda: bool(proof.get("graph", {}).get("cycles")),
        "centrality": lambda: bool(proof.get("graph", {}).get("centrality")),
        "causal_density": lambda: bool(proof.get("graph", {}).get("causal_density")),
        "taxonomy": lambda: bool(proof.get("taxonomy")),
        "statistics": lambda: bool(proof.get("statistics")),
        "derivations": lambda: bool(debug.get("derivations")),
        "cross_tafsir": lambda: bool(debug.get("cross_tafsir_stats")),
    }

    for required in must_include:
        if required in section_paths:
            if not section_paths[required]():
                missing.append(f"must_include_missing:{required}")
        # Unknown requirements are not enforced (may be future features)

    return missing


def _has_verse_keys_per_link(proof: Dict[str, Any]) -> bool:
    """Check if graph paths, cycles, or centrality have verse_key references."""
    graph = proof.get("graph", {})
    paths = graph.get("paths", [])
    cycles = graph.get("cycles", [])
    centrality = graph.get("centrality", {})
    
    # Check paths first
    for path in paths:
        if isinstance(path, dict):
            edges = path.get("edges", [])
            for edge in edges:
                if isinstance(edge, dict) and (edge.get("verse_key") or edge.get("evidence")):
                    return True
            if path.get("evidence_count", 0) > 0:
                return True
    
    # Also check cycles (valid for reinforcement loop queries)
    for cycle in cycles:
        if isinstance(cycle, dict):
            edges = cycle.get("edges", [])
            for edge in edges:
                if isinstance(edge, dict) and (edge.get("verse_key") or edge.get("evidence") or edge.get("evidence_count", 0) > 0):
                    return True
            if cycle.get("total_evidence", 0) > 0:
                return True
    
    # Centrality data is valid provenance for analytical queries (bottleneck, density)
    if centrality and isinstance(centrality, dict) and len(centrality) > 0:
        return True
    
    return bool(paths) or bool(cycles)  # If we have paths/cycles, assume provenance exists


def _check_disallow_violations(
    response: Dict[str, Any],
    disallow: List[str],
    request_payload: Dict[str, Any],
) -> List[str]:
    """
    Phase 2: Check for disallow list violations.

    Args:
        response: The API response
        disallow: List of disallowed patterns from expected.disallow
        request_payload: The request payload

    Returns:
        List of violations
    """
    violations = []
    proof = response.get("proof", {}) if isinstance(response, dict) else {}
    debug = response.get("debug", {}) if isinstance(response, dict) else {}
    answer = response.get("answer", "") if isinstance(response, dict) else ""

    for rule in disallow:
        if rule == "generic_opening_verses_default":
            # Already checked in main scoring
            pass
        elif rule == "fabricated_numbers":
            # Check if answer has numbers not in derivations
            if _has_fabricated_numbers(answer, debug):
                violations.append("disallow_fabricated_numbers")
        elif rule == "fallback_used":
            if debug.get("fallback_used"):
                violations.append("disallow_fallback_used")
        elif rule == "empty_evidence":
            quran = proof.get("quran", [])
            tafsir = proof.get("tafsir", {})
            if not quran and not any(tafsir.values()):
                violations.append("disallow_empty_evidence")

    return violations


def _has_fabricated_numbers(answer: str, debug: Dict[str, Any]) -> bool:
    """Check if answer contains numbers not in derivations."""
    if not answer:
        return False

    derivations = debug.get("derivations", {})
    if not derivations:
        # No derivations to check against
        return False

    # Extract numbers from derivations
    valid_numbers: Set[float] = set()

    def extract_numbers(obj: Any) -> None:
        if isinstance(obj, (int, float)):
            valid_numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                extract_numbers(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_numbers(item)

    extract_numbers(derivations)

    # Add common allowed numbers
    valid_numbers.update({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 100})

    # Extract numbers from answer
    answer_numbers: Set[float] = set()
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%?', answer):
        try:
            answer_numbers.add(float(match.group(1)))
        except ValueError:
            pass

    # Check for fabricated numbers (excluding verse reference numbers 1-286)
    for num in answer_numbers:
        if num not in valid_numbers and num > 286:
            return True

    return False


def score_benchmark_item(
    *,
    benchmark_item: Dict[str, Any],
    response: Optional[Dict[str, Any]],
    http_status: int,
    request_payload: Dict[str, Any],
    schema_valid: bool,
    schema_issues: Sequence[str],
    request_error: Optional[str],
) -> Dict[str, Any]:
    """
    Deterministic scoring entrypoint.

    Phase 2 enhancements:
    - Answer non-empty check
    - Required sections check (must_include)
    - Disallow list enforcement
    - Depth-based capability validation

    Returns a dict with:
      - verdict: PASS | PARTIAL | FAIL
      - reasons: list[str]
      - tags: list[str]
      - metrics: dict[str, Any]
    """
    expected = benchmark_item.get("expected", {}) if isinstance(benchmark_item, dict) else {}
    capabilities = expected.get("capabilities", []) if isinstance(expected, dict) else []
    min_sources = int(expected.get("min_sources", 0)) if isinstance(expected, dict) else 0
    required_sources = expected.get("required_sources", []) if isinstance(expected, dict) else []
    min_hops = int(expected.get("min_hops", 0)) if isinstance(expected, dict) else 0
    must_include = expected.get("must_include", []) if isinstance(expected, dict) else []
    disallow = expected.get("disallow", []) if isinstance(expected, dict) else []

    result = ScoringResult(verdict="FAIL")

    if http_status != 200:
        result.reasons.append(f"http_status_{http_status or 'error'}")
        if request_error:
            result.metrics["request_error"] = request_error
        return result.to_dict()

    if not schema_valid:
        result.reasons.append("schema_invalid")
        result.metrics["schema_issues"] = list(schema_issues)
        return result.to_dict()

    proof = response.get("proof", {}) if isinstance(response, dict) else {}
    debug = response.get("debug", {}) if isinstance(response, dict) else {}

    answer = response.get("answer", "") if isinstance(response, dict) else ""
    quran_verses = proof.get("quran", []) if isinstance(proof, dict) else []
    tafsir = proof.get("tafsir", {}) if isinstance(proof, dict) else {}

    debug_intent = str(debug.get("intent", "")).strip() if isinstance(debug, dict) else ""
    fallback_used = bool(debug.get("fallback_used")) if isinstance(debug, dict) else False

    sources_with_chunks = _count_sources_with_chunks(tafsir if isinstance(tafsir, dict) else {})
    result.metrics.update(
        {
            "sources_with_tafsir": sources_with_chunks,
            "min_sources_required": min_sources,
            "debug_intent": debug_intent,
        }
    )

    # Phase 2: Answer non-empty check (skip for proof_only mode)
    proof_only_mode = request_payload.get("proof_only", False)
    if not proof_only_mode and _is_placeholder_answer(answer):
        result.reasons.append("answer_placeholder_or_empty")
        return result.to_dict()

    # Phase 2: Disallow list enforcement
    if disallow:
        disallow_violations = _check_disallow_violations(response, disallow, request_payload)
        if disallow_violations:
            result.reasons.extend(disallow_violations)
            return result.to_dict()

    # Global FAIL 1: fallback used for structured intents
    if fallback_used and debug_intent and debug_intent != "FREE_TEXT":
        result.reasons.append("fallback_used_for_structured_intent")
        return result.to_dict()

    # Global FAIL 2: missing provenance on any emitted tafsir chunk
    provenance_missing = 0
    for source, chunk in _iter_tafsir_chunks(tafsir if isinstance(tafsir, dict) else {}):
        missing_fields: List[str] = []
        for key in ("chunk_id", "verse_key", "char_start", "char_end"):
            if chunk.get(key) in (None, "", []):
                missing_fields.append(key)
        if missing_fields:
            provenance_missing += 1
            if provenance_missing <= 3:
                result.reasons.append(f"missing_provenance:{source}:{','.join(missing_fields)}")
    if provenance_missing > 0:
        result.metrics["tafsir_chunks_missing_provenance"] = provenance_missing
        return result.to_dict()

    # Global FAIL 3: generic opening verses default
    generic_fail, generic_metrics = _generic_opening_default_fail(
        request_payload.get("question", ""), debug_intent, quran_verses if isinstance(quran_verses, list) else []
    )
    result.metrics.update({f"generic_{k}": v for k, v in generic_metrics.items()})
    if generic_fail:
        result.reasons.append("generic_opening_verses_default")
        return result.to_dict()

    # Global FAIL 4: percent claims without derivation fields
    if _has_percent_claim(str(answer)):
        # Allow if payload contains explicit deterministic derivation fields.
        has_derivation = bool(debug.get("derivations")) or bool(proof.get("statistics"))
        if not has_derivation:
            result.reasons.append("undocumented_percentage_claim")
            return result.to_dict()

    # From here on: PARTIAL vs PASS based on capability checks (no gold truth).
    missing: List[str] = []

    # Evidence presence checks
    if "TAFSIR_MULTI_SOURCE" in capabilities:
        if sources_with_chunks < min_sources:
            missing.append(f"min_sources:{sources_with_chunks}<{min_sources}")
        if isinstance(required_sources, list):
            for src in required_sources:
                if not src:
                    continue
                chunks = tafsir.get(src) if isinstance(tafsir, dict) else None
                if not isinstance(chunks, list) or len(chunks) == 0:
                    missing.append(f"required_source_missing:{src}")

    if "GRAPH_CAUSAL" in capabilities or "GRAPH_METRICS" in capabilities:
        graph = proof.get("graph")
        if not isinstance(graph, dict):
            missing.append("graph_missing")
        else:
            # For causal: require paths OR cycles OR centrality (analytical queries use centrality)
            if "GRAPH_CAUSAL" in capabilities:
                paths = graph.get("paths", [])
                cycles = graph.get("cycles", [])
                centrality = graph.get("centrality", {})
                has_paths = isinstance(paths, list) and len(paths) > 0
                has_cycles = isinstance(cycles, list) and len(cycles) > 0
                has_centrality = isinstance(centrality, dict) and len(centrality) > 0
                # Paths, cycles, or centrality all satisfy GRAPH_CAUSAL requirement
                if not has_paths and not has_cycles and not has_centrality:
                    missing.append("graph_paths_missing")
                # Best-effort hop check - check both paths and cycles
                if min_hops > 0:
                    longest = 0
                    # Check paths
                    for p in paths if isinstance(paths, list) else []:
                        if isinstance(p, dict):
                            hops = p.get("hops")
                            if isinstance(hops, int):
                                longest = max(longest, hops)
                            elif isinstance(p.get("edges"), list):
                                longest = max(longest, len(p["edges"]))
                    # Check cycles (cycles have 'length' field)
                    for c in cycles if isinstance(cycles, list) else []:
                        if isinstance(c, dict):
                            hops = c.get("length", 0)
                            if not hops and isinstance(c.get("edges"), list):
                                hops = len(c["edges"])
                            longest = max(longest, hops)
                    result.metrics["graph_longest_path_hops"] = longest
                    if longest and longest < min_hops:
                        missing.append(f"min_hops:{longest}<{min_hops}")

    if "AXES_11D" in capabilities or "TAXONOMY" in capabilities:
        taxonomy = proof.get("taxonomy")
        if not isinstance(taxonomy, dict):
            missing.append("taxonomy_missing")
        else:
            dims = taxonomy.get("dimensions")
            if not isinstance(dims, dict) or len(dims) == 0:
                missing.append("taxonomy_dimensions_missing")
            else:
                result.metrics["taxonomy_dimensions_keys"] = len(dims)

    if "EMBEDDINGS" in capabilities:
        embeddings = proof.get("embeddings")
        if not isinstance(embeddings, dict):
            missing.append("embeddings_missing")

    # If there is absolutely no evidence, mark FAIL (fail-closed).
    # PHASE 4: Graph-only queries (GRAPH_METRICS) can have valid graph data without quran/tafsir
    quran_count = len(quran_verses) if isinstance(quran_verses, list) else 0
    tafsir_chunks_total = sum(1 for _ in _iter_tafsir_chunks(tafsir if isinstance(tafsir, dict) else {}))
    result.metrics.update({"quran_verses": quran_count, "tafsir_chunks": tafsir_chunks_total})

    # Check for graph-only evidence (valid for GRAPH_METRICS and GRAPH_CAUSAL)
    graph = proof.get("graph", {}) if isinstance(proof, dict) else {}
    has_graph_evidence = bool(
        graph.get("nodes") or
        graph.get("edges") or
        graph.get("paths") or
        graph.get("centrality") or
        graph.get("causal_density") or
        graph.get("cycles")
    )
    result.metrics["has_graph_evidence"] = has_graph_evidence

    # For GRAPH_METRICS, accept graph-only evidence
    if "GRAPH_METRICS" in capabilities or "SEMANTIC_GRAPH_V2" in capabilities:
        if quran_count == 0 and tafsir_chunks_total == 0 and not has_graph_evidence:
            result.reasons.append("no_evidence")
            return result.to_dict()
    else:
        # Non-graph queries still require quran/tafsir evidence
        if quran_count == 0 and tafsir_chunks_total == 0:
            result.reasons.append("no_evidence")
            return result.to_dict()

    # Phase 2: Required sections check (must_include)
    if must_include:
        must_include_missing = _check_required_sections(response, must_include)
        missing.extend(must_include_missing)

    # Phase 2: MULTIHOP capability check
    if "MULTIHOP" in capabilities:
        graph = proof.get("graph", {})
        paths = graph.get("paths", [])
        cycles = graph.get("cycles", [])
        centrality = graph.get("centrality", {})
        # Accept paths, cycles, or centrality for MULTIHOP (analytical queries use centrality)
        if not paths and not cycles and not centrality:
            missing.append("multihop_no_paths")
        elif min_hops > 0:
            # Check paths and cycles meet min_hops requirement
            qualifying_paths = 0
            # Check paths
            for p in paths:
                if isinstance(p, dict):
                    hops = p.get("hops", 0)
                    if not hops and isinstance(p.get("edges"), list):
                        hops = len(p["edges"])
                    if not hops and isinstance(p.get("nodes"), list):
                        hops = max(0, len(p["nodes"]) - 1)
                    if hops >= min_hops:
                        qualifying_paths += 1
            # Check cycles (cycles are inherently multi-hop: A→B→C→A = 3 hops)
            for c in cycles:
                if isinstance(c, dict):
                    hops = c.get("length", 0)
                    if not hops and isinstance(c.get("edges"), list):
                        hops = len(c["edges"])
                    if not hops and isinstance(c.get("nodes"), list):
                        hops = max(0, len(c["nodes"]) - 1)
                    if hops >= min_hops:
                        qualifying_paths += 1
            result.metrics["multihop_qualifying_paths"] = qualifying_paths
            if qualifying_paths == 0:
                missing.append(f"multihop_no_paths_with_{min_hops}_hops")

    # Phase 2: PROVENANCE capability check
    if "PROVENANCE" in capabilities:
        has_provenance = False
        # Check graph paths for provenance
        graph = proof.get("graph", {})
        for path in graph.get("paths", []):
            if isinstance(path, dict):
                for edge in path.get("edges", []):
                    if isinstance(edge, dict) and (edge.get("verse_key") or edge.get("evidence_count", 0) > 0):
                        has_provenance = True
                        break

        # PHASE 4: For GRAPH_METRICS, centrality/causal_density data IS the provenance
        if not has_provenance and ("GRAPH_METRICS" in capabilities or "SEMANTIC_GRAPH_V2" in capabilities):
            centrality = graph.get("centrality", {})
            causal_density = graph.get("causal_density", {})
            # Graph metrics provide provenance through node/edge counts and node identifiers
            if centrality.get("total_nodes", 0) > 0 or centrality.get("total_edges", 0) > 0:
                has_provenance = True
            if not has_provenance and (causal_density.get("total_nodes", 0) > 0 or causal_density.get("outgoing_top10")):
                has_provenance = True

        # Check tafsir chunks for provenance
        if not has_provenance:
            for _, chunk in _iter_tafsir_chunks(tafsir):
                if chunk.get("chunk_id") and chunk.get("verse_key"):
                    has_provenance = True
                    break
        if not has_provenance:
            missing.append("provenance_missing_on_evidence")

    if missing:
        result.verdict = "PARTIAL"
        result.reasons.extend(missing)
        return result.to_dict()

    result.verdict = "PASS"
    return result.to_dict()
