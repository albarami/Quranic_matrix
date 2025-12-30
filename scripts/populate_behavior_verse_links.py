#!/usr/bin/env python3
"""
Phase 2: Populate behavior_verse_links table

Loads behavior→verse evidence from concept_index_v2.jsonl into the 
behavior_verse_links table defined in 001_ssot_extensions.sql.

This script:
1. Reads concept_index_v2.jsonl for behavior-verse mappings
2. Filters out surah_intro entries (per plan requirement)
3. Populates behavior_verse_links with proper evidence types
4. Updates semantic_edges with from/to entity_type and provenance

Usage:
    python scripts/populate_behavior_verse_links.py --dry-run
    python scripts/populate_behavior_verse_links.py --db-url postgresql://...
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data paths
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import execute_batch, Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed. Database operations will be simulated.")


class BehaviorVerseLinkLoader:
    """Loads behavior→verse links into PostgreSQL."""
    
    def __init__(self, db_url: Optional[str] = None, dry_run: bool = False):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.dry_run = dry_run
        self.conn = None
        self.stats = {
            "behaviors_processed": 0,
            "verse_links_created": 0,
            "surah_intro_filtered": 0,
            "edges_updated": 0,
        }
        self.valid_behavior_ids: Set[str] = set()
    
    def connect(self):
        """Connect to PostgreSQL."""
        if self.dry_run or not PSYCOPG2_AVAILABLE:
            logger.info("Dry run mode - no database connection")
            return
        
        if not self.db_url:
            raise ValueError("DATABASE_URL not set")
        
        self.conn = psycopg2.connect(self.db_url)
        logger.info("Connected to PostgreSQL")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed PostgreSQL connection")
    
    def load_valid_behavior_ids(self) -> Set[str]:
        """Load valid behavior IDs from canonical_entities.json."""
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        behavior_ids = {b["id"] for b in data.get("behaviors", [])}
        logger.info(f"Loaded {len(behavior_ids)} valid behavior IDs")
        return behavior_ids
    
    def determine_evidence_type(self, chunk: dict) -> str:
        """
        Determine evidence type based on chunk characteristics.
        
        Types:
        - EXPLICIT: Direct mention in verse text
        - IMPLICIT: Implied by context
        - CONTEXTUAL: Related through surrounding verses
        - TAFSIR_DERIVED: Derived from tafsir interpretation
        """
        source = chunk.get("source", "")
        
        # If from tafsir, it's TAFSIR_DERIVED
        if source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", 
                      "muyassar", "baghawi", "waseet"]:
            return "TAFSIR_DERIVED"
        
        # Default to EXPLICIT for verse-based evidence
        return "EXPLICIT"
    
    def determine_directness(self, chunk: dict) -> str:
        """
        Determine directness of evidence.
        
        - direct: Behavior explicitly mentioned
        - indirect: Behavior implied through related concept
        - inferred: Behavior inferred from context
        """
        # For now, tafsir-derived is indirect, verse-based is direct
        source = chunk.get("source", "")
        if source in ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn",
                      "muyassar", "baghawi", "waseet"]:
            return "indirect"
        return "direct"
    
    def is_surah_intro(self, chunk: dict) -> bool:
        """
        Check if chunk is a surah introduction (should be filtered).
        
        Surah intros are identified by:
        - ayah = 0
        - entry_type = 'surah_intro' (if present)
        - chunk_id containing 'intro'
        """
        if chunk.get("ayah", 1) == 0:
            return True
        
        if chunk.get("entry_type") == "surah_intro":
            return True
        
        chunk_id = chunk.get("chunk_id", "")
        if "intro" in chunk_id.lower():
            return True
        
        return False
    
    def load_behavior_verse_links(self) -> int:
        """
        Load behavior→verse links from concept_index_v2.jsonl.
        
        Filters out surah_intro entries per plan requirement.
        """
        logger.info("Loading behavior→verse links...")
        
        self.valid_behavior_ids = self.load_valid_behavior_ids()
        
        links = []
        seen = set()  # Track unique (behavior_id, surah, ayah, evidence_type)
        
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                behavior_id = concept.get("concept_id", "")
                
                # Skip non-behavior entities
                if not behavior_id.startswith("BEH_"):
                    continue
                
                # Skip if not in valid behaviors (for now, log warning)
                if behavior_id not in self.valid_behavior_ids:
                    logger.debug(f"Skipping unknown behavior: {behavior_id}")
                    continue
                
                self.stats["behaviors_processed"] += 1
                
                # Process verse links
                for verse in concept.get("verses", []):
                    surah = verse.get("surah", 0)
                    ayah = verse.get("ayah", 0)
                    
                    if surah == 0 or ayah == 0:
                        continue
                    
                    # Create link with EXPLICIT evidence type for verse-based
                    key = (behavior_id, surah, ayah, "EXPLICIT")
                    if key not in seen:
                        seen.add(key)
                        links.append({
                            "behavior_id": behavior_id,
                            "surah": surah,
                            "ayah": ayah,
                            "evidence_type": "EXPLICIT",
                            "directness": "direct",
                            "provenance": "concept_index_v2",
                            "confidence": 0.8,  # Default confidence for verse matches
                        })
                
                # Process tafsir chunk links
                for chunk in concept.get("tafsir_chunks", []):
                    # Filter surah intros
                    if self.is_surah_intro(chunk):
                        self.stats["surah_intro_filtered"] += 1
                        continue
                    
                    surah = chunk.get("surah", 0)
                    ayah = chunk.get("ayah", 0)
                    
                    if surah == 0 or ayah == 0:
                        continue
                    
                    evidence_type = self.determine_evidence_type(chunk)
                    directness = self.determine_directness(chunk)
                    
                    key = (behavior_id, surah, ayah, evidence_type)
                    if key not in seen:
                        seen.add(key)
                        links.append({
                            "behavior_id": behavior_id,
                            "surah": surah,
                            "ayah": ayah,
                            "evidence_type": evidence_type,
                            "directness": directness,
                            "provenance": chunk.get("source", "unknown"),
                            "confidence": 0.7,  # Lower confidence for tafsir-derived
                        })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(links)} behavior→verse links")
            logger.info(f"  Behaviors processed: {self.stats['behaviors_processed']}")
            logger.info(f"  Surah intros filtered: {self.stats['surah_intro_filtered']}")
            self.stats["verse_links_created"] = len(links)
            return len(links)
        
        # Insert into database
        insert_sql = """
            INSERT INTO behavior_verse_links 
                (behavior_id, surah, ayah, evidence_type, directness, provenance, confidence)
            VALUES 
                (%(behavior_id)s, %(surah)s, %(ayah)s, %(evidence_type)s, 
                 %(directness)s, %(provenance)s, %(confidence)s)
            ON CONFLICT (behavior_id, surah, ayah, evidence_type) DO UPDATE SET
                confidence = GREATEST(behavior_verse_links.confidence, EXCLUDED.confidence)
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, links, page_size=1000)
        self.conn.commit()
        
        self.stats["verse_links_created"] = len(links)
        logger.info(f"Loaded {len(links)} behavior→verse links")
        return len(links)
    
    def update_semantic_edges_provenance(self) -> int:
        """
        Update semantic_edges with from/to entity_type and provenance JSONB.
        """
        logger.info("Updating semantic_edges with entity types and provenance...")
        
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        # Build entity type lookup
        entity_types = {}
        for node in graph.get("nodes", []):
            entity_types[node["id"]] = node.get("type", "UNKNOWN")
        
        updates = []
        for edge in graph.get("edges", []):
            from_id = edge.get("source")
            to_id = edge.get("target")
            edge_type = edge.get("edge_type")
            
            # Build provenance summary
            evidence = edge.get("evidence", [])
            sources = list(set(e.get("source", "unknown") for e in evidence))
            
            provenance = {
                "sources_list": sources,
                "evidence_count": len(evidence),
                "confidence": edge.get("confidence", 0.5),
                "cue_phrases": edge.get("cue_phrases", []),
                "validation": edge.get("validation", {}),
            }
            
            updates.append({
                "from_entity_id": from_id,
                "to_entity_id": to_id,
                "edge_type": edge_type,
                "from_entity_type": entity_types.get(from_id, "UNKNOWN"),
                "to_entity_type": entity_types.get(to_id, "UNKNOWN"),
                "provenance": json.dumps(provenance),
            })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would update {len(updates)} semantic edges with provenance")
            self.stats["edges_updated"] = len(updates)
            return len(updates)
        
        # Update edges
        update_sql = """
            UPDATE semantic_edges SET
                from_entity_type = %(from_entity_type)s,
                to_entity_type = %(to_entity_type)s,
                provenance = %(provenance)s::jsonb
            WHERE from_entity_id = %(from_entity_id)s 
              AND to_entity_id = %(to_entity_id)s
              AND edge_type = %(edge_type)s
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, update_sql, updates, page_size=500)
        self.conn.commit()
        
        self.stats["edges_updated"] = len(updates)
        logger.info(f"Updated {len(updates)} semantic edges")
        return len(updates)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return loading statistics."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dry_run": self.dry_run,
            "stats": self.stats,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the full loading process."""
        try:
            self.connect()
            self.load_behavior_verse_links()
            self.update_semantic_edges_provenance()
            return self.get_stats()
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Populate behavior_verse_links table")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without database")
    args = parser.parse_args()
    
    loader = BehaviorVerseLinkLoader(
        db_url=args.db_url,
        dry_run=args.dry_run or not args.db_url
    )
    
    stats = loader.run()
    
    print("\n" + "=" * 60)
    print("BEHAVIOR→VERSE LINKS LOADING COMPLETE")
    print("=" * 60)
    print(json.dumps(stats, indent=2))
    
    # Save stats to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    stats_file = artifacts_dir / "behavior_verse_links_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats saved to: {stats_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
