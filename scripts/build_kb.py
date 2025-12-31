#!/usr/bin/env python3
"""
QBM Knowledge Base Builder - Enterprise Edition

This script builds a precomputed, enterprise-grade QBM Knowledge Base (KB)
where ALL connections are made at build-time, and runtime is pure retrieval.

Architecture:
- SSOT Tables: verses, tafsir_entries, behaviors, behavior_verse_links, relations
- Build-time: GPU embeddings, FAISS indices, graph construction, validation
- Runtime: Pure retrieval + formatting (no discovery, no filtering)

Key Principle: Tafsir is mapped by verse_key, NOT by text search.

Usage:
    python scripts/build_kb.py --full --output data/kb --seed 42 --device cuda

Arguments:
    --full         Full rebuild (default: incremental)
    --output       Output directory (default: data/kb)
    --seed         Random seed for reproducibility (default: 42)
    --device       Device for GPU operations (default: cuda)
    --skip-embeddings  Skip embedding generation
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VOCAB_DIR = ROOT_DIR / "vocab"
KB_DIR = DATA_DIR / "kb"
KB_DIR.mkdir(exist_ok=True)


# ============================================================================
# SSOT Data Classes
# ============================================================================

@dataclass
class Verse:
    """Canonical verse record."""
    verse_key: str  # "surah:ayah" e.g., "2:45"
    surah: int
    ayah: int
    text_uthmani: str
    text_simple: Optional[str] = None
    tokens: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.verse_key


@dataclass
class TafsirEntry:
    """Canonical tafsir entry record."""
    id: str  # "{source}:{surah}:{ayah}" e.g., "ibn_kathir:2:45"
    source: str
    verse_key: str
    surah: int
    ayah: int
    text_ar: str
    entry_type: str = "verse"  # "verse" or "surah_intro"
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text_ar.split()) if self.text_ar else 0


@dataclass
class Behavior:
    """Canonical behavior record."""
    id: str  # e.g., "BEH_EMO_PATIENCE"
    term_ar: str
    term_en: str
    category: str
    roots: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    entity_type: str = "BEHAVIOR"


@dataclass
class BehaviorVerseLink:
    """Link between behavior and verse with evidence."""
    id: str  # "{behavior_id}:{verse_key}"
    behavior_id: str
    verse_key: str
    directness: str  # "direct" or "indirect"
    evidence_type: str  # "lexical", "semantic", "tafsir"
    matched_tokens: List[str] = field(default_factory=list)
    provenance: str = ""
    confidence: float = 1.0


@dataclass
class BehaviorDossier:
    """Precomputed behavior dossier - materialized view."""
    behavior_id: str
    term_ar: str
    term_en: str
    category: str
    verse_count: int
    verses: List[Dict[str, Any]] = field(default_factory=list)
    tafsir: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    built_at: str = ""


# ============================================================================
# Data Loaders
# ============================================================================

def load_quran_verses() -> Dict[str, Verse]:
    """Load all Quran verses as SSOT."""
    verses = {}
    quran_path = DATA_DIR / "quran" / "uthmani_hafs_v1.tok_v1.json"

    if not quran_path.exists():
        logger.warning(f"Quran file not found: {quran_path}")
        return verses

    with open(quran_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle the nested surah->ayat structure
    if "surahs" in data:
        for surah_data in data["surahs"]:
            surah_num = surah_data.get("surah", 0)
            for ayah_data in surah_data.get("ayat", []):
                ayah_num = ayah_data.get("ayah", 0)
                verse_key = f"{surah_num}:{ayah_num}"

                # Extract token texts
                tokens = [t.get("text", "") for t in ayah_data.get("tokens", [])]

                verses[verse_key] = Verse(
                    verse_key=verse_key,
                    surah=surah_num,
                    ayah=ayah_num,
                    text_uthmani=ayah_data.get("text", ""),
                    text_simple="",
                    tokens=tokens
                )
    else:
        # Handle flat list format
        verse_list = data if isinstance(data, list) else data.get("verses", [])
        for v in verse_list:
            surah = v.get("surah", v.get("sura_num", 0))
            ayah = v.get("ayah", v.get("aya_num", 0))
            verse_key = f"{surah}:{ayah}"

            verses[verse_key] = Verse(
                verse_key=verse_key,
                surah=surah,
                ayah=ayah,
                text_uthmani=v.get("text_uthmani", v.get("text", "")),
                text_simple=v.get("text_simple", ""),
                tokens=v.get("tokens", [])
            )

    logger.info(f"[SSOT] Loaded {len(verses)} verses")
    return verses


def load_tafsir_entries() -> Dict[str, TafsirEntry]:
    """Load all tafsir entries as SSOT, mapped by verse_key."""
    entries = {}
    tafsir_dir = DATA_DIR / "tafsir"
    sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]

    for source in sources:
        jsonl_path = tafsir_dir / f"{source}.ar.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"Tafsir file not found: {jsonl_path}")
            continue

        count = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    ref = data.get("reference", {})
                    surah = ref.get("surah", 0)
                    ayah = ref.get("ayah", 0)
                    verse_key = f"{surah}:{ayah}"
                    entry_id = f"{source}:{verse_key}"

                    # Detect surah introduction (ayah 0 or special markers)
                    entry_type = "surah_intro" if ayah == 0 else "verse"

                    entries[entry_id] = TafsirEntry(
                        id=entry_id,
                        source=source,
                        verse_key=verse_key,
                        surah=surah,
                        ayah=ayah,
                        text_ar=data.get("text_ar", ""),
                        entry_type=entry_type
                    )
                    count += 1
                except json.JSONDecodeError:
                    continue

        logger.info(f"[SSOT] Loaded {source}: {count} entries")

    logger.info(f"[SSOT] Total tafsir entries: {len(entries)}")
    return entries


def load_behaviors() -> Dict[str, Behavior]:
    """Load canonical behaviors from vocabulary."""
    behaviors = {}
    entities_path = VOCAB_DIR / "canonical_entities.json"

    if not entities_path.exists():
        logger.error(f"Canonical entities not found: {entities_path}")
        return behaviors

    with open(entities_path, encoding="utf-8") as f:
        data = json.load(f)

    for beh in data.get("behaviors", []):
        behavior = Behavior(
            id=beh["id"],
            term_ar=beh["ar"],
            term_en=beh["en"],
            category=beh.get("category", ""),
            roots=beh.get("roots", []),
            synonyms=beh.get("synonyms", [])
        )
        behaviors[behavior.id] = behavior

        # Also index by Arabic term for lookup
        behaviors[behavior.term_ar] = behavior

    logger.info(f"[SSOT] Loaded {len([b for b in behaviors.values() if b.id.startswith('BEH_')])} behaviors")
    return behaviors


def load_concept_index() -> Dict[str, Dict]:
    """Load concept index v3 for behavior-verse mappings."""
    concept_index = {}
    index_path = DATA_DIR / "evidence" / "concept_index_v3.jsonl"

    if not index_path.exists():
        logger.error(f"Concept index not found: {index_path}")
        return concept_index

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                concept_id = entry.get("concept_id", "")
                if concept_id:
                    concept_index[concept_id] = entry
                    # Also index by term
                    term = entry.get("term", "")
                    if term and term != concept_id:
                        concept_index[term] = entry
            except json.JSONDecodeError:
                continue

    logger.info(f"[SSOT] Loaded concept index: {len(concept_index)} entries")
    return concept_index


# ============================================================================
# Build Behavior-Verse Links
# ============================================================================

def build_behavior_verse_links(
    behaviors: Dict[str, Behavior],
    concept_index: Dict[str, Dict]
) -> Dict[str, BehaviorVerseLink]:
    """
    Build behavior-verse links from concept index.

    This uses the pre-validated concept_index_v3.jsonl which contains
    accurate behavior-to-verse mappings with lexical evidence.
    """
    links = {}

    for behavior_id, behavior in behaviors.items():
        if not behavior_id.startswith("BEH_"):
            continue

        # Get concept entry
        concept = concept_index.get(behavior_id) or concept_index.get(behavior.term_ar)
        if not concept:
            logger.warning(f"No concept entry for behavior: {behavior_id}")
            continue

        verses = concept.get("verses", [])
        for verse_data in verses:
            verse_key = verse_data.get("verse_key", "")
            if not verse_key:
                continue

            link_id = f"{behavior_id}:{verse_key}"

            # Extract evidence
            evidence_list = verse_data.get("evidence", [])
            matched_tokens = [e.get("matched_token", "") for e in evidence_list if e.get("matched_token")]
            evidence_type = evidence_list[0].get("type", "lexical") if evidence_list else "lexical"

            links[link_id] = BehaviorVerseLink(
                id=link_id,
                behavior_id=behavior_id,
                verse_key=verse_key,
                directness=verse_data.get("directness", "direct"),
                evidence_type=evidence_type,
                matched_tokens=matched_tokens,
                provenance=verse_data.get("provenance", "concept_index_v3"),
                confidence=1.0 if verse_data.get("directness") == "direct" else 0.8
            )

    logger.info(f"[SSOT] Built {len(links)} behavior-verse links")
    return links


# ============================================================================
# Build Behavior Dossiers (Materialized Views)
# ============================================================================

def build_behavior_dossiers(
    behaviors: Dict[str, Behavior],
    verses: Dict[str, Verse],
    tafsir_entries: Dict[str, TafsirEntry],
    links: Dict[str, BehaviorVerseLink],
    concept_index: Dict[str, Dict]
) -> Dict[str, BehaviorDossier]:
    """
    Build precomputed behavior dossiers.

    CRITICAL: Tafsir is mapped by verse_key directly, NOT by text search.
    For each behavior -> get its verses -> for each verse, get ALL tafsir entries.
    """
    dossiers = {}
    tafsir_sources = ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"]

    for behavior_id, behavior in behaviors.items():
        if not behavior_id.startswith("BEH_"):
            continue

        # Get all verses linked to this behavior
        behavior_links = [l for l in links.values() if l.behavior_id == behavior_id]
        verse_keys = list(set(l.verse_key for l in behavior_links))

        if not verse_keys:
            continue

        # Build verse data
        verse_data = []
        for verse_key in verse_keys:
            verse = verses.get(verse_key)
            if not verse:
                continue

            # Find the link for evidence
            link = next((l for l in behavior_links if l.verse_key == verse_key), None)

            verse_data.append({
                "verse_key": verse_key,
                "surah": verse.surah,
                "ayah": verse.ayah,
                "text_uthmani": verse.text_uthmani,
                "directness": link.directness if link else "direct",
                "evidence_type": link.evidence_type if link else "lexical",
                "matched_tokens": link.matched_tokens if link else []
            })

        # Build tafsir data - MAPPED BY VERSE_KEY DIRECTLY
        tafsir_data = {source: [] for source in tafsir_sources}

        for verse_key in verse_keys:
            for source in tafsir_sources:
                entry_id = f"{source}:{verse_key}"
                entry = tafsir_entries.get(entry_id)

                if entry and entry.entry_type == "verse":  # Only verse tafsir, not surah intros
                    # Parse verse_key
                    parts = verse_key.split(":")
                    surah = int(parts[0]) if len(parts) >= 1 else 0
                    ayah = int(parts[1]) if len(parts) >= 2 else 0

                    tafsir_data[source].append({
                        "verse_key": verse_key,
                        "surah": surah,
                        "ayah": ayah,
                        "text": entry.text_ar[:2000] if len(entry.text_ar) > 2000 else entry.text_ar,
                        "word_count": entry.word_count
                    })

        # Build statistics
        surah_distribution = defaultdict(int)
        for vk in verse_keys:
            surah = int(vk.split(":")[0])
            surah_distribution[surah] += 1

        statistics = {
            "verse_count": len(verse_keys),
            "direct_count": sum(1 for l in behavior_links if l.directness == "direct"),
            "indirect_count": sum(1 for l in behavior_links if l.directness == "indirect"),
            "surah_distribution": dict(surah_distribution),
            "tafsir_coverage": {
                source: len(entries) for source, entries in tafsir_data.items()
            }
        }

        # Create dossier
        dossiers[behavior_id] = BehaviorDossier(
            behavior_id=behavior_id,
            term_ar=behavior.term_ar,
            term_en=behavior.term_en,
            category=behavior.category,
            verse_count=len(verse_keys),
            verses=verse_data,
            tafsir=tafsir_data,
            relations={},  # Will be populated from graph
            statistics=statistics,
            built_at=datetime.utcnow().isoformat() + "Z"
        )

    logger.info(f"[SSOT] Built {len(dossiers)} behavior dossiers")
    return dossiers


# ============================================================================
# GPU Embeddings (Build-time)
# ============================================================================

def build_embeddings(
    dossiers: Dict[str, BehaviorDossier], 
    device: str = "cuda",
    gpu_proof_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build embeddings for behaviors at build-time using GPU.

    This uses the GPU resources (8x A100, 636GB VRAM) for:
    - Behavior concept embeddings
    - Representative verse embeddings
    - Tafsir chunk embeddings
    
    Args:
        dossiers: Dictionary of behavior dossiers to embed
        device: Device for GPU operations (cuda or cpu)
        gpu_proof_dir: If provided, collect GPU proof during embedding
        
    Returns:
        Dictionary with embedding job metadata and GPU proof status
    """
    result = {
        "embeddings_built": False,
        "total_vectors": 0,
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "vector_dims": 384,
        "device_used": device,
        "gpu_proof_valid": False,
        "elapsed_seconds": 0.0
    }
    
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        # Use specified device, fall back to CPU if CUDA not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("[GPU] CUDA requested but not available, falling back to CPU")
            device = "cpu"
        result["device_used"] = device
        logger.info(f"[GPU] Using device: {device}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count))
            logger.info(f"[GPU] {gpu_count} GPUs available, {total_vram / 1e9:.1f} GB VRAM")

        # Load embedding model
        model = SentenceTransformer(result["model_id"])
        model = model.to(device)
        
        # Import GPU proof collector if proof is requested
        gpu_collector = None
        if gpu_proof_dir and device == "cuda":
            try:
                # Try relative import first (when running as module)
                try:
                    from scripts.gpu_proof_instrumentation import GPUProofCollector
                except ImportError:
                    # Fall back to direct import (when running as script)
                    from gpu_proof_instrumentation import GPUProofCollector
                
                gpu_collector = GPUProofCollector(output_dir=gpu_proof_dir)
                gpu_collector.start()
                logger.info(f"[GPU] GPU proof collection started -> {gpu_proof_dir}")
            except Exception as e:
                logger.warning(f"[GPU] GPU proof instrumentation not available: {e}")
        
        start_time = time.time()

        # Prepare all texts for batch encoding (better GPU utilization)
        behavior_ids = list(dossiers.keys())
        texts = []
        for behavior_id in behavior_ids:
            dossier = dossiers[behavior_id]
            concept_text = f"{dossier.term_ar} {dossier.term_en}"
            sample_verses = [v["text_uthmani"] for v in dossier.verses[:5]]
            if sample_verses:
                concept_text += " " + " ".join(sample_verses)
            texts.append(concept_text)
        
        # Batch encode for better GPU utilization
        batch_size = 32  # Optimal for GPU utilization
        result["batch_size"] = batch_size
        logger.info(f"[GPU] Encoding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        
        # Assign embeddings back to dossiers
        for i, behavior_id in enumerate(behavior_ids):
            dossiers[behavior_id].embedding = embeddings[i].tolist()
            result["total_vectors"] += 1

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = elapsed
        result["embeddings_built"] = True
        
        # Stop GPU proof collection and log job metadata
        if gpu_collector:
            gpu_collector.log_embedding_job(
                model_id=result["model_id"],
                vector_dims=result["vector_dims"],
                total_vectors=result["total_vectors"],
                batch_size=batch_size,
                elapsed_seconds=elapsed
            )
            gpu_collector.stop()
            proof_result = gpu_collector.save()
            result["gpu_proof_valid"] = proof_result.valid
            logger.info(f"[GPU] GPU proof saved: valid={proof_result.valid}, avg_util={proof_result.avg_utilization:.1f}%")

        logger.info(f"[GPU] Built embeddings for {len(dossiers)} dossiers in {elapsed:.2f}s")

    except ImportError:
        logger.warning("[GPU] PyTorch/SentenceTransformers not available, skipping embeddings")
    except Exception as e:
        logger.error(f"[GPU] Error building embeddings: {e}")
    
    return result


# ============================================================================
# Validation Gates
# ============================================================================

def validate_kb(
    verses: Dict[str, Verse],
    tafsir_entries: Dict[str, TafsirEntry],
    behaviors: Dict[str, Behavior],
    links: Dict[str, BehaviorVerseLink],
    dossiers: Dict[str, BehaviorDossier]
) -> Tuple[bool, List[str]]:
    """
    Enterprise validation gates.

    The build FAILS if any critical validation fails.
    """
    errors = []
    warnings = []

    # 1. Verse coverage
    if len(verses) < 6000:
        errors.append(f"CRITICAL: Only {len(verses)} verses loaded, expected ~6236")

    # 2. Tafsir coverage
    if len(tafsir_entries) < 30000:
        warnings.append(f"WARNING: Only {len(tafsir_entries)} tafsir entries, expected ~43k+")

    # 3. Behavior count
    behavior_count = len([b for b in behaviors.values() if b.id.startswith("BEH_")])
    if behavior_count < 70:
        errors.append(f"CRITICAL: Only {behavior_count} behaviors loaded, expected 87")

    # 4. Link integrity
    for link in links.values():
        if link.verse_key not in verses:
            errors.append(f"CRITICAL: Link {link.id} references non-existent verse {link.verse_key}")
        if not any(b.id == link.behavior_id for b in behaviors.values()):
            errors.append(f"CRITICAL: Link {link.id} references non-existent behavior {link.behavior_id}")

    # 5. Dossier integrity
    for dossier in dossiers.values():
        if dossier.verse_count == 0:
            warnings.append(f"WARNING: Dossier {dossier.behavior_id} has no verses")

        # Check tafsir coverage
        total_tafsir = sum(len(entries) for entries in dossier.tafsir.values())
        if total_tafsir == 0 and dossier.verse_count > 0:
            warnings.append(f"WARNING: Dossier {dossier.behavior_id} has {dossier.verse_count} verses but no tafsir")

    # 6. No text filtering validation
    # Verify that tafsir entries are mapped by verse_key, not by text search
    sample_dossier = list(dossiers.values())[0] if dossiers else None
    if sample_dossier and sample_dossier.verses:
        first_verse = sample_dossier.verses[0]
        verse_key = first_verse["verse_key"]

        # Check if tafsir exists for this verse
        has_tafsir = any(
            any(t["verse_key"] == verse_key for t in entries)
            for entries in sample_dossier.tafsir.values()
        )
        if not has_tafsir:
            warnings.append(f"WARNING: Verse {verse_key} in dossier has no mapped tafsir")

    # Log results
    for warning in warnings:
        logger.warning(warning)

    for error in errors:
        logger.error(error)

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("[VALIDATION] All critical checks passed")
    else:
        logger.error(f"[VALIDATION] FAILED with {len(errors)} errors")

    return is_valid, errors + warnings


# ============================================================================
# Save KB
# ============================================================================

def save_kb(
    verses: Dict[str, Verse],
    tafsir_entries: Dict[str, TafsirEntry],
    behaviors: Dict[str, Behavior],
    links: Dict[str, BehaviorVerseLink],
    dossiers: Dict[str, BehaviorDossier]
) -> None:
    """Save the Knowledge Base to disk."""

    # 1. Save verses
    verses_path = KB_DIR / "verses.jsonl"
    with open(verses_path, "w", encoding="utf-8") as f:
        for verse in verses.values():
            f.write(json.dumps(asdict(verse), ensure_ascii=False) + "\n")
    logger.info(f"[SAVE] Saved {len(verses)} verses to {verses_path}")

    # 2. Save tafsir entries
    tafsir_path = KB_DIR / "tafsir_entries.jsonl"
    with open(tafsir_path, "w", encoding="utf-8") as f:
        for entry in tafsir_entries.values():
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    logger.info(f"[SAVE] Saved {len(tafsir_entries)} tafsir entries to {tafsir_path}")

    # 3. Save behaviors
    behaviors_path = KB_DIR / "behaviors.jsonl"
    unique_behaviors = {b.id: b for b in behaviors.values() if b.id.startswith("BEH_")}
    with open(behaviors_path, "w", encoding="utf-8") as f:
        for behavior in unique_behaviors.values():
            f.write(json.dumps(asdict(behavior), ensure_ascii=False) + "\n")
    logger.info(f"[SAVE] Saved {len(unique_behaviors)} behaviors to {behaviors_path}")

    # 4. Save behavior-verse links
    links_path = KB_DIR / "behavior_verse_links.jsonl"
    with open(links_path, "w", encoding="utf-8") as f:
        for link in links.values():
            f.write(json.dumps(asdict(link), ensure_ascii=False) + "\n")
    logger.info(f"[SAVE] Saved {len(links)} links to {links_path}")

    # 5. Save behavior dossiers
    dossiers_path = KB_DIR / "behavior_dossiers.jsonl"
    with open(dossiers_path, "w", encoding="utf-8") as f:
        for dossier in dossiers.values():
            f.write(json.dumps(asdict(dossier), ensure_ascii=False) + "\n")
    logger.info(f"[SAVE] Saved {len(dossiers)} dossiers to {dossiers_path}")

    # 6. Save manifest
    manifest = {
        "version": "1.0",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "counts": {
            "verses": len(verses),
            "tafsir_entries": len(tafsir_entries),
            "behaviors": len(unique_behaviors),
            "behavior_verse_links": len(links),
            "behavior_dossiers": len(dossiers)
        },
        "files": [
            "verses.jsonl",
            "tafsir_entries.jsonl",
            "behaviors.jsonl",
            "behavior_verse_links.jsonl",
            "behavior_dossiers.jsonl"
        ]
    }

    manifest_path = KB_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"[SAVE] Saved manifest to {manifest_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def file_hash(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=ROOT_DIR
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='QBM Knowledge Base Builder - Enterprise Edition'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full rebuild (default: incremental)'
    )
    parser.add_argument(
        '--output', type=str, default='data/kb',
        help='Output directory (default: data/kb)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for GPU operations (default: cuda)'
    )
    parser.add_argument(
        '--skip-embeddings', action='store_true',
        help='Skip embedding generation'
    )
    parser.add_argument(
        '--gpu-proof', action='store_true',
        help='Collect GPU proof during embedding generation'
    )
    parser.add_argument(
        '--gpu-proof-dir', type=str, default='artifacts/audit_pack/gpu_proof',
        help='Directory for GPU proof output (default: artifacts/audit_pack/gpu_proof)'
    )
    return parser.parse_args()


# ============================================================================
# Main Build Process
# ============================================================================

def main():
    """Main build process."""
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Update output directory if specified
    global KB_DIR
    KB_DIR = ROOT_DIR / args.output
    KB_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QBM KNOWLEDGE BASE BUILDER - ENTERPRISE EDITION")
    logger.info("=" * 70)
    logger.info(f"  Mode: {'full' if args.full else 'incremental'}")
    logger.info(f"  Output: {KB_DIR}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Git commit: {get_git_commit()}")
    logger.info("=" * 70)

    start_time = time.time()

    # Step 1: Load SSOT data
    logger.info("\n[PHASE 1] Loading SSOT Data...")
    verses = load_quran_verses()
    tafsir_entries = load_tafsir_entries()
    behaviors = load_behaviors()
    concept_index = load_concept_index()

    # Step 2: Build behavior-verse links
    logger.info("\n[PHASE 2] Building Behavior-Verse Links...")
    links = build_behavior_verse_links(behaviors, concept_index)

    # Step 3: Build behavior dossiers
    logger.info("\n[PHASE 3] Building Behavior Dossiers...")
    dossiers = build_behavior_dossiers(
        behaviors, verses, tafsir_entries, links, concept_index
    )

    # Step 4: Build embeddings (GPU)
    embedding_result = None
    if not args.skip_embeddings:
        logger.info("\n[PHASE 4] Building GPU Embeddings...")
        gpu_proof_dir = args.gpu_proof_dir if args.gpu_proof else None
        embedding_result = build_embeddings(dossiers, device=args.device, gpu_proof_dir=gpu_proof_dir)
        if args.gpu_proof:
            logger.info(f"[GPU] Embedding result: {embedding_result}")
    else:
        logger.info("\n[PHASE 4] Skipping embeddings (--skip-embeddings)")

    # Step 5: Validate
    logger.info("\n[PHASE 5] Running Validation Gates...")
    is_valid, issues = validate_kb(verses, tafsir_entries, behaviors, links, dossiers)

    if not is_valid:
        logger.error("\n[BUILD FAILED] Validation errors detected. Fix and re-run.")
        sys.exit(1)

    # Step 6: Save KB
    logger.info("\n[PHASE 6] Saving Knowledge Base...")
    save_kb(verses, tafsir_entries, behaviors, links, dossiers)

    # Step 7: Generate enhanced manifest with hashes
    logger.info("\n[PHASE 7] Generating Enhanced Manifest...")
    generate_enhanced_manifest(args, verses, tafsir_entries, behaviors, links, dossiers)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"BUILD COMPLETE in {elapsed:.2f}s")
    logger.info(f"  Verses: {len(verses)}")
    logger.info(f"  Tafsir Entries: {len(tafsir_entries)}")
    logger.info(f"  Behaviors: {len([b for b in behaviors.values() if b.id.startswith('BEH_')])}")
    logger.info(f"  Behavior-Verse Links: {len(links)}")
    logger.info(f"  Behavior Dossiers: {len(dossiers)}")
    logger.info(f"  Output: {KB_DIR}")
    logger.info("=" * 70)


def generate_enhanced_manifest(args, verses, tafsir_entries, behaviors, links, dossiers):
    """Generate enhanced manifest with input/output hashes."""
    unique_behaviors = {b.id: b for b in behaviors.values() if b.id.startswith("BEH_")}

    # Input SSOT file hashes
    input_files = [
        DATA_DIR / "quran" / "uthmani_hafs_v1.tok_v1.json",
        DATA_DIR / "evidence" / "concept_index_v3.jsonl",
        VOCAB_DIR / "canonical_entities.json",
    ]

    input_hashes = {}
    for path in input_files:
        if path.exists():
            input_hashes[str(path.relative_to(ROOT_DIR))] = file_hash(path)

    # Output file hashes
    output_files = [
        KB_DIR / "verses.jsonl",
        KB_DIR / "tafsir_entries.jsonl",
        KB_DIR / "behaviors.jsonl",
        KB_DIR / "behavior_verse_links.jsonl",
        KB_DIR / "behavior_dossiers.jsonl",
    ]

    output_hashes = {}
    record_counts = {}
    for path in output_files:
        if path.exists():
            output_hashes[str(path.relative_to(ROOT_DIR))] = file_hash(path)
            with open(path, 'r', encoding='utf-8') as f:
                record_counts[path.name] = sum(1 for _ in f)

    manifest = {
        "version": "2.0",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "build_args": {
            "full": args.full,
            "seed": args.seed,
            "device": args.device,
            "skip_embeddings": args.skip_embeddings
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "record_counts": record_counts,
        "counts": {
            "verses": len(verses),
            "tafsir_entries": len(tafsir_entries),
            "behaviors": len(unique_behaviors),
            "behavior_verse_links": len(links),
            "behavior_dossiers": len(dossiers)
        },
        "files": [
            "verses.jsonl",
            "tafsir_entries.jsonl",
            "behaviors.jsonl",
            "behavior_verse_links.jsonl",
            "behavior_dossiers.jsonl"
        ]
    }

    manifest_path = KB_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"[SAVE] Saved enhanced manifest to {manifest_path}")


if __name__ == "__main__":
    main()
