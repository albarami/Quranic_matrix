#!/usr/bin/env python3
"""
Build Behavior Mastery Artifacts.

Phase 2 of Behavior Mastery Plan:
- Generates 87 behavior dossiers
- Creates mastery_summary.json with coverage stats
- Creates mastery_manifest.json with hashes

Usage:
    python scripts/build_behavior_mastery.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mastery.assembler import build_all_dossiers, DOSSIER_OUTPUT_DIR, SUMMARY_OUTPUT_PATH, MANIFEST_OUTPUT_PATH


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Building Behavior Mastery Artifacts")
    logger.info("=" * 60)
    
    # Build all dossiers
    dossiers = build_all_dossiers()
    
    # Report results
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total dossiers: {len(dossiers)}")
    logger.info(f"Dossier directory: {DOSSIER_OUTPUT_DIR}")
    logger.info(f"Summary file: {SUMMARY_OUTPUT_PATH}")
    logger.info(f"Manifest file: {MANIFEST_OUTPUT_PATH}")
    
    # Validate 87 behaviors
    if len(dossiers) != 87:
        logger.error(f"VALIDATION FAILED: Expected 87 dossiers, got {len(dossiers)}")
        sys.exit(1)
    
    # Report coverage
    with_quran = sum(1 for d in dossiers if d.quran_evidence)
    with_tafsir = sum(1 for d in dossiers if d.tafsir_evidence)
    with_relations = sum(1 for d in dossiers if d.outgoing_edges or d.incoming_edges)
    avg_completeness = sum(d.completeness_score for d in dossiers) / len(dossiers)
    
    logger.info("-" * 60)
    logger.info("Coverage Statistics:")
    logger.info(f"  Behaviors with Quran evidence: {with_quran}/87")
    logger.info(f"  Behaviors with tafsir evidence: {with_tafsir}/87")
    logger.info(f"  Behaviors with relationships: {with_relations}/87")
    logger.info(f"  Average completeness score: {avg_completeness:.3f}")
    logger.info("-" * 60)
    
    logger.info("SUCCESS: All 87 behavior dossiers generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
