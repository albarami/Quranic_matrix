"""
Evaluation Runner for QBM Brain

Command-line interface for running the evaluation harness.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .harness import EvaluationHarness


def run_benchmark_suite(
    limit: int = None,
    sections: list = None,
    output_dir: str = "artifacts",
) -> dict:
    """
    Run the benchmark suite and generate reports.
    
    Args:
        limit: Maximum number of questions to evaluate
        sections: Only evaluate these sections
        output_dir: Directory for output files
        
    Returns:
        Summary dictionary
    """
    harness = EvaluationHarness()
    
    print("=" * 60)
    print("QBM EVALUATION HARNESS")
    print("=" * 60)
    print(f"Benchmark: {harness.benchmark_path}")
    print(f"Limit: {limit or 'all'}")
    print(f"Sections: {sections or 'all'}")
    print()
    
    # Run evaluation
    summary = harness.run(limit=limit, sections=sections)
    
    # Print summary
    print("RESULTS:")
    print(f"  Total: {summary['totals']['total']}")
    print(f"  PASS: {summary['totals']['PASS']}")
    print(f"  PARTIAL: {summary['totals']['PARTIAL']}")
    print(f"  FAIL: {summary['totals']['FAIL']}")
    print(f"  Pass Rate: {summary['pass_rate']}%")
    print()
    
    # Save reports
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    
    # JSON report
    json_path = output_path / f"eval_report_{timestamp}.json"
    harness.save_report(json_path)
    print(f"JSON report: {json_path}")
    
    # Markdown report
    md_path = output_path / f"eval_report_{timestamp}.md"
    md_content = harness.generate_markdown_report()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")
    
    # Also save latest report
    latest_json = output_path / "eval_report.json"
    harness.save_report(latest_json)
    
    print()
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run QBM evaluation harness")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--section", action="append", dest="sections", help="Only run specific sections")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (first 2 per section)")
    
    args = parser.parse_args()
    
    limit = args.limit
    sections = args.sections
    
    # Smoke test: first 2 per section = 20 questions
    if args.smoke:
        limit = 20
    
    summary = run_benchmark_suite(
        limit=limit,
        sections=sections,
        output_dir=args.output,
    )
    
    # Exit with error if pass rate < 50%
    if summary["pass_rate"] < 50:
        print("WARNING: Pass rate below 50%")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
