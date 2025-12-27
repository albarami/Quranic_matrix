"""
Reviews Router - /api/reviews/*

Phase 7.4: Scholar review workflow
Phase 7.4 Fix C: Backend abstraction + edge/chunk references

Provides API endpoints for scholar review of QBM annotations:
- Submit reviews for spans/annotations/edges/chunks
- Track review status (pending, approved, rejected)
- Query review history
- Export reviewed annotations

Storage backend abstraction:
- SQLite for local development
- Postgres for production (set REVIEWS_DB_URL env var)

Scholars can review:
- span_id: Behavioral annotation spans
- edge_id: Semantic graph edges
- chunk_id: Tafsir/evidence chunks
"""

import os
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ..dependencies import DATA_DIR

router = APIRouter(prefix="/api/reviews", tags=["Reviews"])


# =============================================================================
# Phase 7.4 Fix C: Storage Backend Abstraction
# =============================================================================

class ReviewStorageBackend(ABC):
    """Abstract base class for review storage backends."""
    
    @abstractmethod
    def init_schema(self) -> None:
        """Initialize database schema."""
        pass
    
    @abstractmethod
    def create_review(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new review."""
        pass
    
    @abstractmethod
    def get_review(self, review_id: int) -> Optional[Dict[str, Any]]:
        """Get a review by ID."""
        pass
    
    @abstractmethod
    def update_review(self, review_id: int, updates: Dict[str, Any], actor_id: str) -> Optional[Dict[str, Any]]:
        """Update a review."""
        pass
    
    @abstractmethod
    def delete_review(self, review_id: int) -> bool:
        """Delete a review."""
        pass
    
    @abstractmethod
    def list_reviews(self, filters: Dict[str, Any], limit: int, offset: int) -> Dict[str, Any]:
        """List reviews with filters."""
        pass
    
    @abstractmethod
    def get_review_history(self, review_id: int) -> List[Dict[str, Any]]:
        """Get history for a review."""
        pass


class SQLiteReviewBackend(ReviewStorageBackend):
    """SQLite implementation of review storage."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_schema()
    
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_schema(self) -> None:
        """Initialize SQLite schema with edge/chunk references."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Reviews table with edge_id and chunk_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- Reference fields (at least one should be set)
                span_id TEXT,
                edge_id TEXT,
                chunk_id TEXT,
                -- Verse reference
                surah INTEGER,
                ayah INTEGER,
                verse_key TEXT,
                -- Reviewer info
                reviewer_id TEXT NOT NULL,
                reviewer_name TEXT,
                -- Review content
                status TEXT DEFAULT 'pending',
                rating INTEGER,
                comment TEXT,
                corrections JSON,
                -- Review type
                review_type TEXT DEFAULT 'span',
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Review history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT,
                actor_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_id) REFERENCES reviews(id)
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_span ON reviews(span_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_edge ON reviews(edge_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_chunk ON reviews(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_type ON reviews(review_type)")
        
        conn.commit()
        conn.close()
    
    def create_review(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Determine review type
        review_type = "span"
        if review_data.get("edge_id"):
            review_type = "edge"
        elif review_data.get("chunk_id"):
            review_type = "chunk"
        
        # Build verse_key
        verse_key = None
        if review_data.get("surah") and review_data.get("ayah"):
            verse_key = f"{review_data['surah']}:{review_data['ayah']}"
        
        corrections_json = json.dumps(review_data.get("corrections")) if review_data.get("corrections") else None
        
        cursor.execute("""
            INSERT INTO reviews (span_id, edge_id, chunk_id, surah, ayah, verse_key, 
                                 reviewer_id, reviewer_name, rating, comment, corrections, review_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            review_data.get("span_id"),
            review_data.get("edge_id"),
            review_data.get("chunk_id"),
            review_data.get("surah"),
            review_data.get("ayah"),
            verse_key,
            review_data["reviewer_id"],
            review_data.get("reviewer_name"),
            review_data.get("rating"),
            review_data.get("comment"),
            corrections_json,
            review_type
        ))
        
        review_id = cursor.lastrowid
        
        # Log history
        cursor.execute("""
            INSERT INTO review_history (review_id, action, new_status, actor_id)
            VALUES (?, 'created', 'pending', ?)
        """, (review_id, review_data["reviewer_id"]))
        
        conn.commit()
        
        # Fetch created review
        cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        conn.close()
        
        return self._row_to_dict(row)
    
    def get_review(self, review_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        return self._row_to_dict(row)
    
    def update_review(self, review_id: int, updates: Dict[str, Any], actor_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Check exists
        cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        existing = cursor.fetchone()
        if not existing:
            conn.close()
            return None
        
        old_status = existing["status"]
        
        # Build update
        update_parts = []
        params = []
        
        if "status" in updates:
            update_parts.append("status = ?")
            params.append(updates["status"])
        
        if "rating" in updates:
            update_parts.append("rating = ?")
            params.append(updates["rating"])
        
        if "comment" in updates:
            update_parts.append("comment = ?")
            params.append(updates["comment"])
        
        if "corrections" in updates:
            update_parts.append("corrections = ?")
            params.append(json.dumps(updates["corrections"]))
        
        if not update_parts:
            conn.close()
            return self._row_to_dict(existing)
        
        update_parts.append("updated_at = CURRENT_TIMESTAMP")
        
        query = f"UPDATE reviews SET {', '.join(update_parts)} WHERE id = ?"
        params.append(review_id)
        
        cursor.execute(query, params)
        
        # Log history
        new_status = updates.get("status", old_status)
        cursor.execute("""
            INSERT INTO review_history (review_id, action, old_status, new_status, actor_id)
            VALUES (?, 'updated', ?, ?, ?)
        """, (review_id, old_status, new_status, actor_id))
        
        conn.commit()
        
        # Fetch updated
        cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
        row = cursor.fetchone()
        conn.close()
        
        return self._row_to_dict(row)
    
    def delete_review(self, review_id: int) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM reviews WHERE id = ?", (review_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        cursor.execute("DELETE FROM review_history WHERE review_id = ?", (review_id,))
        cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        
        conn.commit()
        conn.close()
        return True
    
    def list_reviews(self, filters: Dict[str, Any], limit: int, offset: int) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM reviews WHERE 1=1"
        count_query = "SELECT COUNT(*) as total FROM reviews WHERE 1=1"
        params = []
        count_params = []
        
        if filters.get("status"):
            query += " AND status = ?"
            count_query += " AND status = ?"
            params.append(filters["status"])
            count_params.append(filters["status"])
        
        if filters.get("reviewer_id"):
            query += " AND reviewer_id = ?"
            count_query += " AND reviewer_id = ?"
            params.append(filters["reviewer_id"])
            count_params.append(filters["reviewer_id"])
        
        if filters.get("surah"):
            query += " AND surah = ?"
            count_query += " AND surah = ?"
            params.append(filters["surah"])
            count_params.append(filters["surah"])
        
        if filters.get("review_type"):
            query += " AND review_type = ?"
            count_query += " AND review_type = ?"
            params.append(filters["review_type"])
            count_params.append(filters["review_type"])
        
        if filters.get("edge_id"):
            query += " AND edge_id = ?"
            count_query += " AND edge_id = ?"
            params.append(filters["edge_id"])
            count_params.append(filters["edge_id"])
        
        if filters.get("chunk_id"):
            query += " AND chunk_id = ?"
            count_query += " AND chunk_id = ?"
            params.append(filters["chunk_id"])
            count_params.append(filters["chunk_id"])
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()["total"]
        
        conn.close()
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "reviews": [self._row_to_dict(row) for row in rows]
        }
    
    def get_review_history(self, review_id: int) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM review_history 
            WHERE review_id = ? 
            ORDER BY timestamp DESC
        """, (review_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite row to dict with JSON parsing."""
        result = dict(row)
        if result.get("corrections"):
            try:
                result["corrections"] = json.loads(result["corrections"])
            except:
                pass
        return result


# =============================================================================
# Backend Factory
# =============================================================================

def get_storage_backend() -> ReviewStorageBackend:
    """
    Get the appropriate storage backend based on environment.
    
    Set REVIEWS_DB_URL for Postgres (future), otherwise uses SQLite.
    """
    db_url = os.getenv("REVIEWS_DB_URL")
    
    if db_url and db_url.startswith("postgresql://"):
        # Future: Postgres backend
        raise NotImplementedError("Postgres backend not yet implemented. Use SQLite for now.")
    
    # Default: SQLite
    return SQLiteReviewBackend(DATA_DIR / "reviews.db")


# Initialize storage backend
_storage_backend: Optional[ReviewStorageBackend] = None


def get_backend() -> ReviewStorageBackend:
    """Get or create storage backend singleton."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = get_storage_backend()
    return _storage_backend


# =============================================================================
# Request/Response Models
# =============================================================================

class ReviewCreate(BaseModel):
    """
    Request model for creating a review.
    
    Phase 7.4 Fix C: Scholars can review spans, edges, or chunks.
    At least one of span_id, edge_id, or chunk_id should be provided.
    """
    # Reference fields - at least one should be set
    span_id: Optional[str] = Field(None, description="ID of the span being reviewed")
    edge_id: Optional[str] = Field(None, description="ID of the semantic edge being reviewed")
    chunk_id: Optional[str] = Field(None, description="ID of the evidence chunk being reviewed")
    # Verse reference
    surah: Optional[int] = Field(None, ge=1, le=114)
    ayah: Optional[int] = Field(None, ge=1)
    # Reviewer info
    reviewer_id: str = Field(..., min_length=1, description="Reviewer identifier")
    reviewer_name: Optional[str] = Field(None, description="Reviewer display name")
    # Review content
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comment: Optional[str] = Field(None, max_length=2000)
    corrections: Optional[dict] = Field(None, description="Suggested corrections")


class ReviewUpdate(BaseModel):
    """Request model for updating a review."""
    status: Optional[str] = Field(None, description="pending, approved, rejected")
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=2000)
    corrections: Optional[dict] = None


class ReviewResponse(BaseModel):
    """Response model for a review."""
    id: int
    span_id: Optional[str]
    edge_id: Optional[str]
    chunk_id: Optional[str]
    surah: Optional[int]
    ayah: Optional[int]
    verse_key: Optional[str]
    reviewer_id: str
    reviewer_name: Optional[str]
    status: str
    rating: Optional[int]
    comment: Optional[str]
    corrections: Optional[dict]
    review_type: str
    created_at: str
    updated_at: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status")
async def reviews_status():
    """Get reviews system status and statistics."""
    backend = get_backend()
    result = backend.list_reviews({}, limit=1, offset=0)
    
    # Get counts by status
    pending = backend.list_reviews({"status": "pending"}, limit=1, offset=0)["total"]
    approved = backend.list_reviews({"status": "approved"}, limit=1, offset=0)["total"]
    rejected = backend.list_reviews({"status": "rejected"}, limit=1, offset=0)["total"]
    
    # Get counts by type
    span_count = backend.list_reviews({"review_type": "span"}, limit=1, offset=0)["total"]
    edge_count = backend.list_reviews({"review_type": "edge"}, limit=1, offset=0)["total"]
    chunk_count = backend.list_reviews({"review_type": "chunk"}, limit=1, offset=0)["total"]
    
    return {
        "status": "ready",
        "backend": "sqlite",
        "statistics": {
            "total_reviews": result["total"],
            "by_status": {
                "pending": pending,
                "approved": approved,
                "rejected": rejected
            },
            "by_type": {
                "span": span_count,
                "edge": edge_count,
                "chunk": chunk_count
            }
        },
        "endpoints": {
            "list": "GET /api/reviews",
            "create": "POST /api/reviews",
            "get": "GET /api/reviews/{id}",
            "update": "PUT /api/reviews/{id}",
            "delete": "DELETE /api/reviews/{id}",
            "by_edge": "GET /api/reviews/edge/{edge_id}",
            "by_chunk": "GET /api/reviews/chunk/{chunk_id}",
            "export": "GET /api/reviews/export"
        }
    }


@router.get("")
async def list_reviews(
    status: Optional[str] = Query(None, description="Filter by status"),
    reviewer_id: Optional[str] = Query(None, description="Filter by reviewer"),
    review_type: Optional[str] = Query(None, description="Filter by type: span, edge, chunk"),
    surah: Optional[int] = Query(None, ge=1, le=114),
    edge_id: Optional[str] = Query(None, description="Filter by edge ID"),
    chunk_id: Optional[str] = Query(None, description="Filter by chunk ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List reviews with optional filters."""
    backend = get_backend()
    
    filters = {}
    if status:
        filters["status"] = status
    if reviewer_id:
        filters["reviewer_id"] = reviewer_id
    if review_type:
        filters["review_type"] = review_type
    if surah:
        filters["surah"] = surah
    if edge_id:
        filters["edge_id"] = edge_id
    if chunk_id:
        filters["chunk_id"] = chunk_id
    
    return backend.list_reviews(filters, limit, offset)


@router.post("")
async def create_review(review: ReviewCreate):
    """
    Create a new review.
    
    Phase 7.4 Fix C: Supports reviewing spans, edges, or chunks.
    """
    backend = get_backend()
    
    review_data = {
        "span_id": review.span_id,
        "edge_id": review.edge_id,
        "chunk_id": review.chunk_id,
        "surah": review.surah,
        "ayah": review.ayah,
        "reviewer_id": review.reviewer_id,
        "reviewer_name": review.reviewer_name,
        "rating": review.rating,
        "comment": review.comment,
        "corrections": review.corrections
    }
    
    return backend.create_review(review_data)


@router.get("/{review_id}")
async def get_review(review_id: int):
    """Get a specific review by ID with history."""
    backend = get_backend()
    
    review = backend.get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    review["history"] = backend.get_review_history(review_id)
    
    return review


@router.put("/{review_id}")
async def update_review(review_id: int, update: ReviewUpdate, actor_id: str = Query(...)):
    """Update a review."""
    backend = get_backend()
    
    if update.status is not None and update.status not in ["pending", "approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be: pending, approved, rejected")
    
    updates = {}
    if update.status is not None:
        updates["status"] = update.status
    if update.rating is not None:
        updates["rating"] = update.rating
    if update.comment is not None:
        updates["comment"] = update.comment
    if update.corrections is not None:
        updates["corrections"] = update.corrections
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    result = backend.update_review(review_id, updates, actor_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    return result


@router.delete("/{review_id}")
async def delete_review(review_id: int, actor_id: str = Query(...)):
    """Delete a review."""
    backend = get_backend()
    
    if not backend.delete_review(review_id):
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    return {"deleted": True, "id": review_id}


@router.get("/edge/{edge_id}")
async def get_reviews_by_edge(edge_id: str):
    """Get all reviews for a specific semantic edge."""
    backend = get_backend()
    result = backend.list_reviews({"edge_id": edge_id}, limit=100, offset=0)
    
    return {
        "edge_id": edge_id,
        "total": result["total"],
        "reviews": result["reviews"]
    }


@router.get("/chunk/{chunk_id}")
async def get_reviews_by_chunk(chunk_id: str):
    """Get all reviews for a specific evidence chunk."""
    backend = get_backend()
    result = backend.list_reviews({"chunk_id": chunk_id}, limit=100, offset=0)
    
    return {
        "chunk_id": chunk_id,
        "total": result["total"],
        "reviews": result["reviews"]
    }


@router.get("/export")
async def export_reviews(
    status: Optional[str] = Query(None, description="Filter by status"),
    review_type: Optional[str] = Query(None, description="Filter by type: span, edge, chunk"),
    format: str = Query("json", description="Export format: json or csv")
):
    """Export reviews for external processing."""
    backend = get_backend()
    
    filters = {}
    if status:
        filters["status"] = status
    if review_type:
        filters["review_type"] = review_type
    
    result = backend.list_reviews(filters, limit=10000, offset=0)
    
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "total": result["total"],
        "format": format,
        "filters": filters,
        "reviews": result["reviews"]
    }
