"""
Generate Phase 0 Baseline Report
Documents current fallback usage rate before remediation.

Usage:
    python scripts/generate_baseline_report.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import generate_baseline_report, STANDARD_QUERIES


def main():
    """Generate baseline report and save to reports/"""
    print("=" * 60)
    print("QBM Phase 0: Baseline Report Generation")
    print("=" * 60)
    
    # Initialize system
    print("\n[1/3] Initializing QBM Full Power System...")
    try:
        from src.ml.full_power_system import FullPowerQBMSystem
        from src.ml.mandatory_proof_system import integrate_with_system
        
        system = FullPowerQBMSystem()
        
        # Build index if needed
        status = system.get_status()
        if status["vector_search"].get("status") == "not_built":
            print("  Building vector index...")
            system.build_index()
            system.build_graph()
        
        # Add proof system
        system = integrate_with_system(system)
        print("  System initialized successfully.")
        
    except Exception as e:
        print(f"  ERROR: Failed to initialize system: {e}")
        # Generate a placeholder report
        report = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "SYSTEM_INIT_FAILED",
        }
        save_report(report)
        return
    
    # Generate baseline
    print("\n[2/3] Running baseline queries...")
    
    # Extended query set for comprehensive baseline
    extended_queries = STANDARD_QUERIES + [
        "ارسم السلسلة من الغفلة إلى جهنم",
        "شبكة علاقات الإيمان",
        "التحليل الإحصائي الكامل للسلوكيات",
        "النفاق عبر الأبعاد الإحدى عشر",
        "رحلة القلب من السلامة إلى الموت",
    ]
    
    baseline = generate_baseline_report(system, extended_queries)
    
    # Add metadata
    baseline["timestamp"] = datetime.now().isoformat()
    baseline["phase"] = 0
    baseline["description"] = "Pre-remediation baseline - documents fallback usage before fixes"
    
    # Print summary
    print("\n[3/3] Baseline Summary:")
    print(f"  Total queries: {baseline['total_queries']}")
    print(f"  Fallbacks used: {baseline['total_fallbacks']}")
    print(f"  Fallback rate: {baseline['fallback_rate']:.1%}")
    print("\n  Component fallback counts:")
    for component, count in baseline["component_fallback_counts"].items():
        if isinstance(count, dict):
            for sub, sub_count in count.items():
                print(f"    {component}.{sub}: {sub_count}")
        else:
            print(f"    {component}: {count}")
    
    # Save report
    save_report(baseline)
    
    print("\n" + "=" * 60)
    print("Baseline report generated successfully!")
    print("=" * 60)


def save_report(baseline: dict):
    """Save baseline report to reports directory"""
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Save JSON
    json_path = reports_dir / f"baseline_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")
    
    # Save Markdown
    md_path = reports_dir / f"baseline_{date_str}.md"
    md_content = generate_markdown_report(baseline)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Saved: {md_path}")


def generate_markdown_report(baseline: dict) -> str:
    """Generate markdown report from baseline data"""
    
    if "error" in baseline:
        return f"""# QBM Phase 0 Baseline Report

**Date:** {baseline.get('timestamp', 'unknown')}  
**Status:** ❌ SYSTEM INITIALIZATION FAILED

## Error

```
{baseline.get('error', 'Unknown error')}
```

The baseline report could not be generated because the system failed to initialize.
This needs to be fixed before proceeding with remediation.
"""
    
    fallback_rate = baseline.get('fallback_rate', 0)
    status_emoji = "✅" if fallback_rate == 0 else "⚠️" if fallback_rate < 0.5 else "❌"
    
    md = f"""# QBM Phase 0 Baseline Report

**Date:** {baseline.get('timestamp', 'unknown')}  
**Phase:** 0 - Baseline and Instrumentation  
**Status:** {status_emoji} Fallback Rate: {fallback_rate:.1%}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Queries | {baseline.get('total_queries', 0)} |
| Fallbacks Used | {baseline.get('total_fallbacks', 0)} |
| Fallback Rate | {fallback_rate:.1%} |

---

## Component Fallback Counts

| Component | Fallback Count |
|-----------|----------------|
"""
    
    component_counts = baseline.get('component_fallback_counts', {})
    for component, count in component_counts.items():
        if isinstance(count, dict):
            for sub, sub_count in count.items():
                md += f"| {component}.{sub} | {sub_count} |\n"
        else:
            md += f"| {component} | {count} |\n"
    
    md += """
---

## Query Results

"""
    
    for i, result in enumerate(baseline.get('results', []), 1):
        query = result.get('query', 'Unknown')
        fallback = result.get('fallback_used', False)
        status = "❌ FALLBACK" if fallback else "✅ PRIMARY"
        score = result.get('validation_score', 0)
        
        md += f"### Query {i}: {query[:40]}...\n\n"
        md += f"- **Status:** {status}\n"
        md += f"- **Validation Score:** {score:.1f}%\n"
        
        if fallback:
            reasons = result.get('fallback_reasons', [])
            if reasons:
                md += f"- **Fallback Reasons:**\n"
                for reason in reasons:
                    md += f"  - {reason}\n"
        
        dist = result.get('retrieval_distribution', {})
        if dist:
            md += f"- **Retrieval Distribution:** {dist}\n"
        
        if result.get('error'):
            md += f"- **Error:** {result['error']}\n"
        
        md += "\n"
    
    md += """---

## Interpretation

"""
    
    if fallback_rate == 0:
        md += """✅ **Excellent!** No fallbacks were used. The primary retrieval path is working correctly.

This is the target state after remediation.
"""
    elif fallback_rate < 0.3:
        md += f"""⚠️ **Partial Success.** {fallback_rate:.1%} of queries required fallbacks.

Some queries are working via primary path, but others are being patched by fallbacks.
Review the specific queries that triggered fallbacks.
"""
    else:
        md += f"""❌ **Critical Issue.** {fallback_rate:.1%} of queries required fallbacks.

The primary retrieval path is not working for most queries. The system is relying
heavily on fallback mechanisms to produce results.

**This is the problem Phase 0 is designed to expose.**

The remediation plan will address these issues in subsequent phases.
"""
    
    md += """
---

*Report generated by Phase 0 instrumentation*
"""
    
    return md


if __name__ == "__main__":
    main()
