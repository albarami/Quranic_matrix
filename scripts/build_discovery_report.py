#!/usr/bin/env python3
"""
Build Discovery Report Artifacts.

Phase 3 of Behavior Mastery Plan:
- Multi-hop path enumeration
- Bridge behavior detection
- Community detection
- Motif discovery
- Link predictions (sandboxed)

Usage:
    python scripts/build_discovery_report.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.discovery import (
    DiscoveryEngine,
    DISCOVERY_REPORT_PATH,
    BRIDGES_PATH,
    COMMUNITIES_PATH,
    MOTIFS_PATH,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Building Discovery Report Artifacts")
    logger.info("=" * 60)
    
    # Build discovery report
    engine = DiscoveryEngine()
    engine.load_graph()
    report_hash = engine.save_discovery_artifacts()
    report = engine.generate_discovery_report()
    
    # Report results
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Report hash: {report_hash}")
    logger.info(f"Discovery report: {DISCOVERY_REPORT_PATH}")
    logger.info(f"Bridges file: {BRIDGES_PATH}")
    logger.info(f"Communities file: {COMMUNITIES_PATH}")
    logger.info(f"Motifs file: {MOTIFS_PATH}")
    
    # Report statistics
    stats = report["statistics"]
    logger.info("-" * 60)
    logger.info("Discovery Statistics:")
    logger.info(f"  Behaviors analyzed: {stats['behaviors_analyzed']}")
    logger.info(f"  Multi-hop paths found: {stats['total_paths']}")
    logger.info(f"  Bridge behaviors: {stats['total_bridges']}")
    logger.info(f"  Communities detected: {stats['total_communities']}")
    logger.info(f"  Motifs found: {stats['total_motifs']}")
    logger.info(f"  Link predictions (sandboxed): {stats['total_link_predictions']}")
    logger.info("-" * 60)
    
    # Validate link predictions are sandboxed
    for lp in report.get("link_predictions", []):
        if lp.get("is_confirmed", True) or not lp.get("promotion_blocked", False):
            logger.error(f"VALIDATION FAILED: Link prediction {lp['hypothesis_id']} not properly sandboxed")
            sys.exit(1)
    
    logger.info("SUCCESS: Discovery report generated with all hypotheses properly sandboxed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
