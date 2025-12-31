"""
Evaluation Harness for QBM Brain

Evaluates capability engines against the 200-question benchmark.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.capabilities import get_engine, list_engines
from src.capabilities.base import CapabilityResult
from src.capabilities.registry import SECTION_CAPABILITIES


# Generic opening verses to disallow
GENERIC_OPENING_VERSES = {f"1:{i}" for i in range(1, 8)} | {f"2:{i}" for i in range(1, 21)}


@dataclass
class EvaluationResult:
    """Result of evaluating a single benchmark question."""
    
    question_id: str
    section: str
    title: str
    verdict: str  # PASS, PARTIAL, FAIL
    reasons: List[str] = field(default_factory=list)
    capabilities_tested: List[str] = field(default_factory=list)
    capabilities_passed: List[str] = field(default_factory=list)
    capabilities_failed: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    provenance_count: int = 0
    verse_count: int = 0
    has_generic_verses: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "section": self.section,
            "title": self.title,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "capabilities_tested": self.capabilities_tested,
            "capabilities_passed": self.capabilities_passed,
            "capabilities_failed": self.capabilities_failed,
            "execution_time_ms": self.execution_time_ms,
            "provenance_count": self.provenance_count,
            "verse_count": self.verse_count,
            "has_generic_verses": self.has_generic_verses,
        }


class EvaluationHarness:
    """
    Harness for evaluating capability engines against benchmark questions.
    
    For each question:
    1. Determine required capabilities from expected.capabilities
    2. Execute each capability engine
    3. Validate results against expected requirements
    4. Aggregate into PASS/PARTIAL/FAIL verdict
    """
    
    def __init__(self):
        self.benchmark_path = Path("data/benchmarks/qbm_legendary_200.v1.jsonl")
        self.results: List[EvaluationResult] = []
        self.summary: Dict[str, Any] = {}
    
    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load benchmark questions from JSONL."""
        questions = []
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions
    
    def evaluate_question(self, question: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single benchmark question."""
        start_time = time.time()
        
        q_id = question.get("id", "?")
        section = question.get("section", "?")
        title = question.get("title", "?")
        expected = question.get("expected", {})
        
        required_caps = expected.get("capabilities", [])
        min_sources = expected.get("min_sources", 1)
        required_sources = expected.get("required_sources", [])
        must_include = expected.get("must_include", [])
        disallow = expected.get("disallow", [])
        
        result = EvaluationResult(
            question_id=q_id,
            section=section,
            title=title,
            verdict="PASS",
            capabilities_tested=required_caps.copy(),
        )
        
        all_verses: Set[str] = set()
        all_provenance: List[Dict] = []
        sources_found: Set[str] = set()
        
        # Execute each required capability
        for cap_id in required_caps:
            engine = get_engine(cap_id)
            if not engine:
                result.capabilities_failed.append(cap_id)
                result.reasons.append(f"No engine for capability: {cap_id}")
                continue
            
            try:
                cap_result = engine.execute(
                    question.get("question_ar", ""),
                    params={"question": question}
                )
                
                if cap_result.success:
                    result.capabilities_passed.append(cap_id)
                    
                    # Collect verses
                    for v in cap_result.verses:
                        vk = v.get("verse_key", "")
                        if vk:
                            all_verses.add(vk)
                        source = v.get("source", "")
                        if source:
                            sources_found.add(source)
                    
                    # Collect provenance
                    all_provenance.extend(cap_result.provenance)
                else:
                    result.capabilities_failed.append(cap_id)
                    result.reasons.extend(cap_result.errors)
                    
            except Exception as e:
                result.capabilities_failed.append(cap_id)
                result.reasons.append(f"Engine {cap_id} error: {str(e)}")
        
        # Check for generic opening verses (disallow check)
        if "generic_opening_verses_default" in disallow:
            generic_found = all_verses & GENERIC_OPENING_VERSES
            if len(generic_found) > len(all_verses) * 0.5 and len(all_verses) > 0:
                result.has_generic_verses = True
                result.reasons.append(f"Too many generic opening verses: {generic_found}")
        
        # Check required sources
        missing_sources = set(required_sources) - sources_found
        if missing_sources and required_sources:
            result.reasons.append(f"Missing required sources: {missing_sources}")
        
        # Check provenance requirement
        if "edge_provenance" in must_include or "PROVENANCE" in required_caps:
            if len(all_provenance) == 0:
                result.reasons.append("Missing provenance")
        
        # Determine verdict
        # PASS: All capabilities passed, no critical failures
        # PARTIAL: Some capabilities passed, minor issues
        # FAIL: No capabilities passed or critical violations
        
        if result.capabilities_failed and len(result.capabilities_passed) == 0:
            result.verdict = "FAIL"
        elif result.has_generic_verses:
            result.verdict = "FAIL"
        elif len(result.capabilities_passed) == len(required_caps) and len(required_caps) > 0:
            # All capabilities passed - check if there are blocking issues
            blocking_reasons = [r for r in result.reasons if "Missing provenance" in r]
            if blocking_reasons:
                result.verdict = "PARTIAL"
            else:
                result.verdict = "PASS"
        elif result.capabilities_failed:
            result.verdict = "PARTIAL"
        elif result.reasons:
            result.verdict = "PARTIAL"
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        result.provenance_count = len(all_provenance)
        result.verse_count = len(all_verses)
        
        return result
    
    def run(self, limit: Optional[int] = None, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run evaluation on benchmark questions.
        
        Args:
            limit: Maximum number of questions to evaluate
            sections: Only evaluate these sections (e.g., ["A", "B"])
            
        Returns:
            Summary report with pass/fail metrics
        """
        questions = self.load_benchmark()
        
        # Filter by sections if specified
        if sections:
            sections_upper = [s.upper() for s in sections]
            questions = [q for q in questions if q.get("section", "").upper() in sections_upper]
        
        # Apply limit
        if limit:
            questions = questions[:limit]
        
        self.results = []
        
        for question in questions:
            result = self.evaluate_question(question)
            self.results.append(result)
        
        # Build summary
        self.summary = self._build_summary()
        
        return self.summary
    
    def _build_summary(self) -> Dict[str, Any]:
        """Build summary statistics from results."""
        totals = {
            "total": len(self.results),
            "PASS": 0,
            "PARTIAL": 0,
            "FAIL": 0,
        }
        
        per_section: Dict[str, Dict[str, int]] = {}
        per_capability: Dict[str, Dict[str, int]] = {}
        fail_reasons: Dict[str, int] = {}
        
        for r in self.results:
            totals[r.verdict] += 1
            
            # Per section
            sec = r.section
            if sec not in per_section:
                per_section[sec] = {"total": 0, "PASS": 0, "PARTIAL": 0, "FAIL": 0}
            per_section[sec]["total"] += 1
            per_section[sec][r.verdict] += 1
            
            # Per capability
            for cap in r.capabilities_passed:
                if cap not in per_capability:
                    per_capability[cap] = {"tested": 0, "passed": 0, "failed": 0}
                per_capability[cap]["tested"] += 1
                per_capability[cap]["passed"] += 1
            
            for cap in r.capabilities_failed:
                if cap not in per_capability:
                    per_capability[cap] = {"tested": 0, "passed": 0, "failed": 0}
                per_capability[cap]["tested"] += 1
                per_capability[cap]["failed"] += 1
            
            # Fail reasons
            if r.verdict == "FAIL":
                for reason in r.reasons:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        
        # Calculate pass rate
        pass_rate = totals["PASS"] / totals["total"] * 100 if totals["total"] > 0 else 0
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "totals": totals,
            "pass_rate": round(pass_rate, 1),
            "per_section": per_section,
            "per_capability": per_capability,
            "top_fail_reasons": sorted(fail_reasons.items(), key=lambda x: -x[1])[:10],
        }
    
    def save_report(self, output_path: Path) -> None:
        """Save evaluation report to JSON."""
        report = {
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def generate_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# QBM Evaluation Report",
            "",
            f"**Timestamp**: {self.summary.get('timestamp', 'N/A')}",
            f"**Pass Rate**: {self.summary.get('pass_rate', 0)}%",
            "",
            "## Summary",
            "",
        ]
        
        totals = self.summary.get("totals", {})
        lines.append(f"- **Total**: {totals.get('total', 0)}")
        lines.append(f"- **PASS**: {totals.get('PASS', 0)}")
        lines.append(f"- **PARTIAL**: {totals.get('PARTIAL', 0)}")
        lines.append(f"- **FAIL**: {totals.get('FAIL', 0)}")
        lines.append("")
        
        lines.append("## Per Section")
        lines.append("")
        lines.append("| Section | Total | PASS | PARTIAL | FAIL |")
        lines.append("|---------|-------|------|---------|------|")
        
        for sec, stats in sorted(self.summary.get("per_section", {}).items()):
            lines.append(
                f"| {sec} | {stats['total']} | {stats['PASS']} | {stats['PARTIAL']} | {stats['FAIL']} |"
            )
        
        lines.append("")
        lines.append("## Per Capability")
        lines.append("")
        lines.append("| Capability | Tested | Passed | Failed |")
        lines.append("|------------|--------|--------|--------|")
        
        for cap, stats in sorted(self.summary.get("per_capability", {}).items()):
            lines.append(
                f"| {cap} | {stats['tested']} | {stats['passed']} | {stats['failed']} |"
            )
        
        lines.append("")
        lines.append("## Top Fail Reasons")
        lines.append("")
        
        for reason, count in self.summary.get("top_fail_reasons", []):
            lines.append(f"- **{count}x**: {reason}")
        
        return "\n".join(lines)
