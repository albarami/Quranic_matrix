"""
Load Truth Layer to PostgreSQL (Phase 8.4)

Loads all truth layer data from JSONL artifacts into PostgreSQL.
Ensures data integrity and provides roundtrip verification.

Usage:
    python scripts/load_truth_layer_to_postgres.py --db-url postgresql://user:pass@host/db
    
    Or set DATABASE_URL environment variable.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data paths
CANONICAL_ENTITIES_FILE = Path("vocab/canonical_entities.json")
CONCEPT_INDEX_FILE = Path("data/evidence/concept_index_v2.jsonl")
CHUNKED_INDEX_FILE = Path("data/evidence/evidence_index_v2_chunked.jsonl")
SEMANTIC_GRAPH_FILE = Path("data/graph/semantic_graph_v2.json")
SCHEMA_FILE = Path("schemas/postgres_truth_layer.sql")
QURAN_FILE = Path("data/quran/uthmani_hafs_v1.tok_v1.json")
TAFSIR_DIR = Path("data/tafsir")

# Try to import psycopg2, but make it optional
try:
    import psycopg2
    from psycopg2.extras import execute_batch, Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed. Database operations will be simulated.")


class TruthLayerLoader:
    """Loads truth layer data into PostgreSQL."""
    
    def __init__(self, db_url: Optional[str] = None, dry_run: bool = False):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.dry_run = dry_run
        self.conn = None
        self.stats = {
            "verses_loaded": 0,
            "entities_loaded": 0,
            "chunks_loaded": 0,
            "tafsir_entries_loaded": 0,
            "mentions_loaded": 0,
            "edges_loaded": 0,
            "edge_evidence_loaded": 0,
        }
    
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
    
    def execute_schema(self):
        """Execute the schema SQL."""
        if self.dry_run or not self.conn:
            logger.info(f"Would execute schema from {SCHEMA_FILE}")
            return
        
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        
        with self.conn.cursor() as cur:
            cur.execute(schema_sql)
        self.conn.commit()
        logger.info("Schema executed successfully")

    def load_verses(self) -> int:
        """Load Quran verses into the verses table."""
        logger.info("Loading Quran verses...")

        if not QURAN_FILE.exists():
            logger.warning(f"Quran file not found: {QURAN_FILE}")
            return 0

        with open(QURAN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        verses = []

        if "surahs" in data:
            for surah_data in data.get("surahs", []):
                surah_num = surah_data.get("surah", 0)
                for ayah_data in surah_data.get("ayat", []):
                    ayah_num = ayah_data.get("ayah", 0)
                    text_uthmani = ayah_data.get("text", "")
                    tokens = ayah_data.get("tokens", [])
                    word_count = len(tokens) if tokens else len(text_uthmani.split())

                    verses.append({
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "text_uthmani": text_uthmani,
                        "text_simple": ayah_data.get("text_simple", ""),
                        "word_count": word_count,
                    })
        else:
            verse_list = data if isinstance(data, list) else data.get("verses", [])
            for v in verse_list:
                surah = v.get("surah", v.get("sura_num", 0))
                ayah = v.get("ayah", v.get("aya_num", 0))
                text_uthmani = v.get("text_uthmani", v.get("text", ""))
                word_count = len(v.get("tokens", [])) if v.get("tokens") else len(text_uthmani.split())

                verses.append({
                    "surah": surah,
                    "ayah": ayah,
                    "text_uthmani": text_uthmani,
                    "text_simple": v.get("text_simple", ""),
                    "word_count": word_count,
                })

        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(verses)} verses")
            self.stats["verses_loaded"] = len(verses)
            return len(verses)

        insert_sql = """
            INSERT INTO verses (surah, ayah, text_uthmani, text_simple, word_count)
            VALUES (%(surah)s, %(ayah)s, %(text_uthmani)s, %(text_simple)s, %(word_count)s)
            ON CONFLICT (surah, ayah) DO UPDATE SET
                text_uthmani = EXCLUDED.text_uthmani,
                text_simple = EXCLUDED.text_simple,
                word_count = EXCLUDED.word_count
        """

        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, verses, page_size=1000)
        self.conn.commit()

        self.stats["verses_loaded"] = len(verses)
        logger.info(f"Loaded {len(verses)} verses")
        return len(verses)
    
    def load_entities(self) -> int:
        """Load canonical entities."""
        logger.info("Loading canonical entities...")
        
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entities = []
        
        # Behaviors
        for b in data.get("behaviors", []):
            roots = b.get("roots", [])
            synonyms = b.get("synonyms", [])
            evidence_mode = "lexical" if (roots or synonyms) else "annotation"
            entities.append({
                "entity_id": b["id"],
                "entity_type": "BEHAVIOR",
                "label_ar": b.get("ar", ""),
                "label_en": b.get("en", ""),
                "category": b.get("category", ""),
                "polarity": None,
                "temporal": None,
                "roots": roots,
                "synonyms": synonyms,
                "evidence_mode": evidence_mode,
                "lexical_required": bool(roots or synonyms),
                "metadata": json.dumps({"roots": roots, "synonyms": synonyms}),
            })
        
        # Agents
        for a in data.get("agents", []):
            entities.append({
                "entity_id": a["id"],
                "entity_type": "AGENT",
                "label_ar": a.get("ar", ""),
                "label_en": a.get("en", ""),
                "category": None,
                "polarity": None,
                "temporal": None,
                "roots": None,
                "synonyms": None,
                "evidence_mode": None,
                "lexical_required": None,
                "metadata": None,
            })
        
        # Organs
        for o in data.get("organs", []):
            entities.append({
                "entity_id": o["id"],
                "entity_type": "ORGAN",
                "label_ar": o.get("ar", ""),
                "label_en": o.get("en", ""),
                "category": o.get("domain", ""),
                "polarity": None,
                "temporal": None,
                "roots": None,
                "synonyms": None,
                "evidence_mode": None,
                "lexical_required": None,
                "metadata": None,
            })
        
        # Heart states
        for h in data.get("heart_states", []):
            entities.append({
                "entity_id": h["id"],
                "entity_type": "HEART_STATE",
                "label_ar": h.get("ar", ""),
                "label_en": h.get("en", ""),
                "category": None,
                "polarity": h.get("polarity", ""),
                "temporal": None,
                "roots": None,
                "synonyms": None,
                "evidence_mode": None,
                "lexical_required": None,
                "metadata": None,
            })
        
        # Consequences
        for c in data.get("consequences", []):
            entities.append({
                "entity_id": c["id"],
                "entity_type": "CONSEQUENCE",
                "label_ar": c.get("ar", ""),
                "label_en": c.get("en", ""),
                "category": None,
                "polarity": c.get("polarity", ""),
                "temporal": c.get("temporal", ""),
                "roots": None,
                "synonyms": None,
                "evidence_mode": None,
                "lexical_required": None,
                "metadata": None,
            })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(entities)} entities")
            self.stats["entities_loaded"] = len(entities)
            return len(entities)
        
        # Insert into database
        insert_sql = """
            INSERT INTO entities (
                entity_id, entity_type, label_ar, label_en, category,
                polarity, temporal, roots, synonyms, evidence_mode,
                lexical_required, metadata
            )
            VALUES (
                %(entity_id)s, %(entity_type)s, %(label_ar)s, %(label_en)s, %(category)s,
                %(polarity)s, %(temporal)s, %(roots)s, %(synonyms)s, %(evidence_mode)s,
                %(lexical_required)s, %(metadata)s
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                label_ar = EXCLUDED.label_ar,
                label_en = EXCLUDED.label_en,
                category = EXCLUDED.category,
                roots = EXCLUDED.roots,
                synonyms = EXCLUDED.synonyms,
                evidence_mode = EXCLUDED.evidence_mode,
                lexical_required = EXCLUDED.lexical_required
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, entities)
        self.conn.commit()
        
        self.stats["entities_loaded"] = len(entities)
        logger.info(f"Loaded {len(entities)} entities")
        return len(entities)
    
    def load_chunks(self, limit: Optional[int] = None) -> int:
        """Load tafsir chunks."""
        logger.info("Loading tafsir chunks...")
        
        chunks = []
        with open(CHUNKED_INDEX_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                chunk = json.loads(line)
                chunks.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "source": chunk.get("source", "unknown"),
                    "surah": chunk.get("surah", 1),
                    "ayah": chunk.get("ayah", 1),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text_clean": chunk.get("text_clean", ""),
                    "text_original": chunk.get("text", ""),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", len(chunk.get("text_clean", ""))),
                    "word_count": len(chunk.get("text_clean", "").split()),
                })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(chunks)} chunks")
            self.stats["chunks_loaded"] = len(chunks)
            return len(chunks)
        
        # Note: This requires verses table to be populated first
        # For now, we'll skip the foreign key constraint
        insert_sql = """
            INSERT INTO tafsir_chunks (chunk_id, source, surah, ayah, chunk_index, text_clean, text_original, char_start, char_end, word_count)
            VALUES (%(chunk_id)s, %(source)s, %(surah)s, %(ayah)s, %(chunk_index)s, %(text_clean)s, %(text_original)s, %(char_start)s, %(char_end)s, %(word_count)s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text_clean = EXCLUDED.text_clean
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, chunks, page_size=1000)
        self.conn.commit()
        
        self.stats["chunks_loaded"] = len(chunks)
        logger.info(f"Loaded {len(chunks)} chunks")
        return len(chunks)

    def load_tafsir_entries(self, sources: Optional[List[str]] = None, limit: Optional[int] = None) -> int:
        """Load raw tafsir entries into tafsir_chunks (chunk_index=0)."""
        logger.info("Loading raw tafsir entries...")

        sources = sources or ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]
        entries = []

        for source in sources:
            jsonl_path = TAFSIR_DIR / f"{source}.ar.jsonl"
            if not jsonl_path.exists():
                logger.warning(f"Tafsir file not found: {jsonl_path}")
                continue

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ref = data.get("reference", {})
                    surah = ref.get("surah", 0)
                    ayah = ref.get("ayah", 0)
                    text_ar = data.get("text_ar", "")
                    entry_type = "surah_intro" if ayah == 0 else "verse"
                    chunk_id = f"{source}:{surah}:{ayah}:0"

                    entries.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "surah": surah,
                        "ayah": ayah,
                        "chunk_index": 0,
                        "text_clean": text_ar,
                        "text_original": text_ar,
                        "char_start": 0,
                        "char_end": len(text_ar),
                        "word_count": len(text_ar.split()),
                        "entry_type": entry_type,
                    })

        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(entries)} tafsir entries")
            self.stats["tafsir_entries_loaded"] = len(entries)
            return len(entries)

        insert_sql = """
            INSERT INTO tafsir_chunks (
                chunk_id, source, surah, ayah, chunk_index,
                text_clean, text_original, char_start, char_end,
                word_count, entry_type
            )
            VALUES (
                %(chunk_id)s, %(source)s, %(surah)s, %(ayah)s, %(chunk_index)s,
                %(text_clean)s, %(text_original)s, %(char_start)s, %(char_end)s,
                %(word_count)s, %(entry_type)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                text_clean = EXCLUDED.text_clean,
                text_original = EXCLUDED.text_original,
                word_count = EXCLUDED.word_count,
                entry_type = EXCLUDED.entry_type
        """

        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, entries, page_size=1000)
        self.conn.commit()

        self.stats["tafsir_entries_loaded"] = len(entries)
        logger.info(f"Loaded {len(entries)} tafsir entries")
        return len(entries)
    
    def load_mentions(self) -> int:
        """Load entity mentions from concept index."""
        logger.info("Loading entity mentions...")
        
        mentions = []
        with open(CONCEPT_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                concept = json.loads(line)
                entity_id = concept.get("concept_id")
                
                for chunk in concept.get("tafsir_chunks", []):
                    mentions.append({
                        "entity_id": entity_id,
                        "source": chunk.get("source", "unknown"),
                        "surah": chunk.get("surah", 1),
                        "ayah": chunk.get("ayah", 1),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "char_start": chunk.get("char_start", 0),
                        "char_end": chunk.get("char_end", 0),
                        "quote": chunk.get("quote", "")[:500],  # Truncate long quotes
                        "confidence": 1.0,
                    })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(mentions)} mentions")
            self.stats["mentions_loaded"] = len(mentions)
            return len(mentions)
        
        insert_sql = """
            INSERT INTO mentions (entity_id, source, surah, ayah, chunk_id, char_start, char_end, quote, confidence)
            VALUES (%(entity_id)s, %(source)s, %(surah)s, %(ayah)s, %(chunk_id)s, %(char_start)s, %(char_end)s, %(quote)s, %(confidence)s)
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, insert_sql, mentions, page_size=1000)
        self.conn.commit()
        
        self.stats["mentions_loaded"] = len(mentions)
        logger.info(f"Loaded {len(mentions)} mentions")
        return len(mentions)
    
    def load_semantic_edges(self) -> int:
        """Load semantic edges and their evidence."""
        logger.info("Loading semantic edges...")
        
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
        
        edges = []
        edge_evidence = []
        
        for edge in graph.get("edges", []):
            import uuid
            edge_uuid = str(uuid.uuid4())
            
            edges.append({
                "edge_id": edge_uuid,
                "from_entity_id": edge.get("source"),
                "to_entity_id": edge.get("target"),
                "edge_type": edge.get("edge_type"),
                "confidence": edge.get("confidence", 0.5),
                "evidence_count": edge.get("evidence_count", 0),
                "sources_count": edge.get("sources_count", 0),
                "cue_strength": edge.get("cue_strength"),
                "cue_phrases": edge.get("cue_phrases", []),
                "metadata": json.dumps({"validation": edge.get("validation", {})}),
            })
            
            # Add evidence for this edge
            for ev in edge.get("evidence", []):
                edge_evidence.append({
                    "edge_id": edge_uuid,
                    "source": ev.get("source", "unknown"),
                    "surah": ev.get("surah", 1),
                    "ayah": ev.get("ayah", 1),
                    "chunk_id": ev.get("chunk_id", ""),
                    "char_start": ev.get("char_start", 0),
                    "char_end": ev.get("char_end", 0),
                    "quote": ev.get("quote", "")[:500],
                    "cue_phrase": ev.get("cue_phrase", ""),
                    "endpoints_validated": True,
                })
        
        if self.dry_run or not self.conn:
            logger.info(f"Would load {len(edges)} edges and {len(edge_evidence)} evidence items")
            self.stats["edges_loaded"] = len(edges)
            self.stats["edge_evidence_loaded"] = len(edge_evidence)
            return len(edges)
        
        # Insert edges
        edge_sql = """
            INSERT INTO semantic_edges (edge_id, from_entity_id, to_entity_id, edge_type, confidence, evidence_count, sources_count, cue_strength, cue_phrases, metadata)
            VALUES (%(edge_id)s::uuid, %(from_entity_id)s, %(to_entity_id)s, %(edge_type)s, %(confidence)s, %(evidence_count)s, %(sources_count)s, %(cue_strength)s, %(cue_phrases)s, %(metadata)s)
            ON CONFLICT (from_entity_id, to_entity_id, edge_type) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                evidence_count = EXCLUDED.evidence_count
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, edge_sql, edges, page_size=500)
        self.conn.commit()
        
        # Insert evidence
        evidence_sql = """
            INSERT INTO edge_evidence (edge_id, source, surah, ayah, chunk_id, char_start, char_end, quote, cue_phrase, endpoints_validated)
            VALUES (%(edge_id)s::uuid, %(source)s, %(surah)s, %(ayah)s, %(chunk_id)s, %(char_start)s, %(char_end)s, %(quote)s, %(cue_phrase)s, %(endpoints_validated)s)
        """
        
        with self.conn.cursor() as cur:
            execute_batch(cur, evidence_sql, edge_evidence, page_size=1000)
        self.conn.commit()
        
        self.stats["edges_loaded"] = len(edges)
        self.stats["edge_evidence_loaded"] = len(edge_evidence)
        logger.info(f"Loaded {len(edges)} edges and {len(edge_evidence)} evidence items")
        return len(edges)
    
    def verify_counts(self) -> Dict[str, Any]:
        """Verify loaded counts match JSONL exports."""
        logger.info("Verifying counts...")
        
        verification = {
            "entities": {"jsonl": 0, "db": 0, "match": False},
            "edges": {"jsonl": 0, "db": 0, "match": False},
        }
        
        # Count from JSONL
        with open(CANONICAL_ENTITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            verification["entities"]["jsonl"] = (
                len(data.get("behaviors", [])) +
                len(data.get("agents", [])) +
                len(data.get("organs", [])) +
                len(data.get("heart_states", [])) +
                len(data.get("consequences", []))
            )
        
        with open(SEMANTIC_GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
            verification["edges"]["jsonl"] = len(graph.get("edges", []))
        
        if self.dry_run or not self.conn:
            verification["entities"]["db"] = self.stats["entities_loaded"]
            verification["edges"]["db"] = self.stats["edges_loaded"]
        else:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM entities")
                verification["entities"]["db"] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM semantic_edges")
                verification["edges"]["db"] = cur.fetchone()[0]
        
        verification["entities"]["match"] = (
            verification["entities"]["jsonl"] == verification["entities"]["db"]
        )
        verification["edges"]["match"] = (
            verification["edges"]["jsonl"] == verification["edges"]["db"]
        )
        
        return verification
    
    def run(self, skip_chunks: bool = False, skip_verses: bool = False, tafsir_mode: str = "chunked"):
        """Run the full load process."""
        logger.info("Starting truth layer load...")
        
        self.connect()
        
        try:
            self.execute_schema()
            if not skip_verses:
                self.load_verses()
            self.load_entities()
            
            if tafsir_mode == "raw":
                self.load_tafsir_entries()
            elif not skip_chunks:
                self.load_chunks(limit=10000)  # Limit for initial load
            
            self.load_mentions()
            self.load_semantic_edges()
            
            verification = self.verify_counts()
            
            logger.info("\n" + "=" * 60)
            logger.info("Load Summary")
            logger.info("=" * 60)
            logger.info(f"Verses loaded: {self.stats['verses_loaded']}")
            logger.info(f"Entities loaded: {self.stats['entities_loaded']}")
            logger.info(f"Chunks loaded: {self.stats['chunks_loaded']}")
            logger.info(f"Tafsir entries loaded: {self.stats['tafsir_entries_loaded']}")
            logger.info(f"Mentions loaded: {self.stats['mentions_loaded']}")
            logger.info(f"Edges loaded: {self.stats['edges_loaded']}")
            logger.info(f"Edge evidence loaded: {self.stats['edge_evidence_loaded']}")
            logger.info("\nVerification:")
            for key, val in verification.items():
                status = "✓" if val["match"] else "✗"
                logger.info(f"  {key}: JSONL={val['jsonl']}, DB={val['db']} {status}")
            
            return self.stats, verification
            
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Load truth layer to PostgreSQL")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without database")
    parser.add_argument("--skip-chunks", action="store_true", help="Skip loading chunks (large)")
    parser.add_argument("--skip-verses", action="store_true", help="Skip loading verses")
    parser.add_argument("--tafsir-mode", choices=["chunked", "raw"], default="chunked",
                        help="Load tafsir from chunked index or raw JSONL")
    
    args = parser.parse_args()
    
    loader = TruthLayerLoader(db_url=args.db_url, dry_run=args.dry_run)
    stats, verification = loader.run(
        skip_chunks=args.skip_chunks,
        skip_verses=args.skip_verses,
        tafsir_mode=args.tafsir_mode,
    )
    
    print("\n" + "=" * 60)
    print("Truth Layer Load Complete")
    print("=" * 60)
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print(f"Verification: {json.dumps(verification, indent=2)}")


if __name__ == "__main__":
    main()
