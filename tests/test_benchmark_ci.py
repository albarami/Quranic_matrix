"""
CI Benchmark Harness Test

Tests that the 200-question benchmark can be run and produces expected results.
This test is designed to be run in CI to verify benchmark integrity.

Usage:
    pytest tests/test_benchmark_ci.py -v
    
    # Run actual benchmark (slow, requires full data):
    pytest tests/test_benchmark_ci.py -v --run-benchmark
"""

import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmarks" / "qbm_legendary_200.v1.jsonl"
EVAL_REPORTS_DIR = PROJECT_ROOT / "reports" / "eval"


# Note: pytest_addoption should be in conftest.py, not here
# For now, we'll use environment variable instead

@pytest.fixture
def run_benchmark():
    """Check if RUN_BENCHMARK env var is set."""
    import os
    return os.environ.get("RUN_BENCHMARK", "").lower() in ("1", "true", "yes")


class TestBenchmarkDataset:
    """Test benchmark dataset integrity."""
    
    def test_benchmark_file_exists(self):
        """Test benchmark file exists."""
        assert BENCHMARK_PATH.exists(), f"Benchmark file not found: {BENCHMARK_PATH}"
    
    def test_benchmark_has_200_questions(self):
        """Test benchmark has exactly 200 questions."""
        if not BENCHMARK_PATH.exists():
            pytest.skip("Benchmark file not found")
        
        count = 0
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        
        assert count == 200, f"Expected 200 questions, found {count}"
    
    def test_benchmark_questions_valid(self):
        """Test all benchmark questions have required fields."""
        if not BENCHMARK_PATH.exists():
            pytest.skip("Benchmark file not found")
        
        required_fields = {"id", "section"}
        # Question can be in various fields
        question_fields = {"question", "query", "question_ar", "question_en"}
        
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                q = json.loads(line)
                missing = required_fields - set(q.keys())
                assert not missing, f"Question {i} missing fields: {missing}"
                
                # Check for any question field
                has_question = bool(question_fields & set(q.keys()))
                assert has_question, f"Question {i} missing question field"
    
    def test_benchmark_sections_complete(self):
        """Test all 10 sections (A-J) are present."""
        if not BENCHMARK_PATH.exists():
            pytest.skip("Benchmark file not found")
        
        sections = set()
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    sections.add(q.get("section", "").upper())
        
        expected = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
        missing = expected - sections
        assert not missing, f"Missing sections: {missing}"


class TestEvaluationReports:
    """Test evaluation report integrity."""
    
    def test_eval_reports_dir_exists(self):
        """Test eval reports directory exists."""
        assert EVAL_REPORTS_DIR.exists(), f"Eval reports dir not found: {EVAL_REPORTS_DIR}"
    
    def test_latest_eval_report_exists(self):
        """Test at least one eval report exists."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        assert len(reports) > 0, "No evaluation reports found"
    
    def test_latest_eval_report_valid(self):
        """Test latest eval report is valid JSON with expected structure."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        if not reports:
            pytest.skip("No evaluation reports found")
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Check structure - may have meta at top or in summary
        assert "summary" in report
        assert "results" in report
        
        # Check summary
        summary = report["summary"]
        assert "pass_rate" in summary
    
    def test_latest_eval_report_200_questions(self):
        """Test latest eval report covers 200 questions."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        if not reports:
            pytest.skip("No evaluation reports found")
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Total may be in totals dict or calculated from results
        summary = report["summary"]
        if "totals" in summary:
            total = summary["totals"].get("total", 0)
        else:
            total = len(report.get("results", []))
        assert total == 200, f"Expected 200 questions, found {total}"
    
    def test_latest_eval_report_100_percent_pass(self):
        """Test latest eval report has 100% pass rate."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        if not reports:
            pytest.skip("No evaluation reports found")
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        pass_rate = report["summary"].get("pass_rate", 0)
        assert pass_rate == 100.0, f"Expected 100% pass rate, got {pass_rate}%"


class TestBenchmarkHarness:
    """Test benchmark harness can run (optional, slow)."""
    
    @pytest.mark.skipif(
        not BENCHMARK_PATH.exists(),
        reason="Benchmark file not found"
    )
    def test_harness_smoke_test(self, run_benchmark):
        """Run benchmark harness smoke test (1 question per section)."""
        if not run_benchmark:
            pytest.skip("Use --run-benchmark to run actual benchmark")
        
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        
        from src.eval.harness import EvaluationHarness
        
        harness = EvaluationHarness()
        
        # Run smoke test (1 question per section)
        results = harness.run_smoke_test()
        
        assert results is not None
        assert "summary" in results
        assert results["summary"]["totals"]["total"] == 10  # 1 per section


class TestBenchmarkReproducibility:
    """Test benchmark results are reproducible."""
    
    def test_eval_reports_have_timestamps(self):
        """Test eval reports have timestamps for reproducibility."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        if not reports:
            pytest.skip("No evaluation reports found")
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Timestamp may be in meta or summary
        has_timestamp = (
            ("meta" in report and "timestamp_utc" in report["meta"]) or
            ("summary" in report and "timestamp" in report["summary"])
        )
        assert has_timestamp, "Report should have a timestamp"
    
    def test_eval_reports_have_pass_rate(self):
        """Test eval reports have pass rate info."""
        if not EVAL_REPORTS_DIR.exists():
            pytest.skip("Eval reports dir not found")
        
        reports = list(EVAL_REPORTS_DIR.glob("eval_report_*.json"))
        if not reports:
            pytest.skip("No evaluation reports found")
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Should have pass_rate in summary
        assert "summary" in report
        assert "pass_rate" in report["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
