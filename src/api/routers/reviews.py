"""
Reviews Router - /api/reviews/*

Phase 7.4: Scholar review workflow

Provides API endpoints for scholar review of QBM annotations:
- Submit reviews for spans/annotations
- Track review status (pending, approved, rejected)
- Query review history
- Export reviewed annotations

Note: Uses SQLite for local development. Production should use Postgres.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ..dependencies import DATA_DIR

router = APIRouter(prefix="/api/reviews", tags=["Reviews"])

# Database path
REVIEWS_DB = DATA_DIR / "reviews.db"


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(REVIEWS_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize reviews database schema."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            span_id TEXT NOT NULL,
            surah INTEGER,
            ayah INTEGER,
            reviewer_id TEXT NOT NULL,
            reviewer_name TEXT,
            status TEXT DEFAULT 'pending',
            rating INTEGER,
            comment TEXT,
            corrections JSON,
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id)")
    
    conn.commit()
    conn.close()


# Initialize database on module load
init_db()


# =============================================================================
# Request/Response Models
# =============================================================================

class ReviewCreate(BaseModel):
    """Request model for creating a review."""
    span_id: str = Field(..., description="ID of the span being reviewed")
    surah: Optional[int] = Field(None, ge=1, le=114)
    ayah: Optional[int] = Field(None, ge=1)
    reviewer_id: str = Field(..., min_length=1, description="Reviewer identifier")
    reviewer_name: Optional[str] = Field(None, description="Reviewer display name")
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
    span_id: str
    surah: Optional[int]
    ayah: Optional[int]
    reviewer_id: str
    reviewer_name: Optional[str]
    status: str
    rating: Optional[int]
    comment: Optional[str]
    corrections: Optional[dict]
    created_at: str
    updated_at: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status")
async def reviews_status():
    """Get reviews system status and statistics."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get counts by status
    cursor.execute("""
        SELECT status, COUNT(*) as count 
        FROM reviews 
        GROUP BY status
    """)
    status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
    
    # Get total reviews
    cursor.execute("SELECT COUNT(*) as total FROM reviews")
    total = cursor.fetchone()["total"]
    
    # Get recent activity
    cursor.execute("""
        SELECT COUNT(*) as recent 
        FROM reviews 
        WHERE created_at > datetime('now', '-7 days')
    """)
    recent = cursor.fetchone()["recent"]
    
    conn.close()
    
    return {
        "status": "ready",
        "database": str(REVIEWS_DB),
        "statistics": {
            "total_reviews": total,
            "by_status": status_counts,
            "last_7_days": recent
        },
        "endpoints": {
            "list": "GET /api/reviews",
            "create": "POST /api/reviews",
            "get": "GET /api/reviews/{id}",
            "update": "PUT /api/reviews/{id}",
            "delete": "DELETE /api/reviews/{id}",
            "by_span": "GET /api/reviews/span/{span_id}",
            "export": "GET /api/reviews/export"
        }
    }


@router.get("")
async def list_reviews(
    status: Optional[str] = Query(None, description="Filter by status"),
    reviewer_id: Optional[str] = Query(None, description="Filter by reviewer"),
    surah: Optional[int] = Query(None, ge=1, le=114),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List reviews with optional filters."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Build query
    query = "SELECT * FROM reviews WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if reviewer_id:
        query += " AND reviewer_id = ?"
        params.append(reviewer_id)
    
    if surah:
        query += " AND surah = ?"
        params.append(surah)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Get total count
    count_query = "SELECT COUNT(*) as total FROM reviews WHERE 1=1"
    count_params = []
    if status:
        count_query += " AND status = ?"
        count_params.append(status)
    if reviewer_id:
        count_query += " AND reviewer_id = ?"
        count_params.append(reviewer_id)
    if surah:
        count_query += " AND surah = ?"
        count_params.append(surah)
    
    cursor.execute(count_query, count_params)
    total = cursor.fetchone()["total"]
    
    conn.close()
    
    reviews = []
    for row in rows:
        review = dict(row)
        if review.get("corrections"):
            try:
                review["corrections"] = json.loads(review["corrections"])
            except:
                pass
        reviews.append(review)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "reviews": reviews
    }


@router.post("")
async def create_review(review: ReviewCreate):
    """Create a new review."""
    conn = get_db()
    cursor = conn.cursor()
    
    corrections_json = json.dumps(review.corrections) if review.corrections else None
    
    cursor.execute("""
        INSERT INTO reviews (span_id, surah, ayah, reviewer_id, reviewer_name, rating, comment, corrections)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        review.span_id,
        review.surah,
        review.ayah,
        review.reviewer_id,
        review.reviewer_name,
        review.rating,
        review.comment,
        corrections_json
    ))
    
    review_id = cursor.lastrowid
    
    # Log history
    cursor.execute("""
        INSERT INTO review_history (review_id, action, new_status, actor_id)
        VALUES (?, 'created', 'pending', ?)
    """, (review_id, review.reviewer_id))
    
    conn.commit()
    
    # Fetch created review
    cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    row = cursor.fetchone()
    conn.close()
    
    result = dict(row)
    if result.get("corrections"):
        try:
            result["corrections"] = json.loads(result["corrections"])
        except:
            pass
    
    return result


@router.get("/{review_id}")
async def get_review(review_id: int):
    """Get a specific review by ID."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    # Get history
    cursor.execute("""
        SELECT * FROM review_history 
        WHERE review_id = ? 
        ORDER BY timestamp DESC
    """, (review_id,))
    history = [dict(h) for h in cursor.fetchall()]
    
    conn.close()
    
    result = dict(row)
    if result.get("corrections"):
        try:
            result["corrections"] = json.loads(result["corrections"])
        except:
            pass
    result["history"] = history
    
    return result


@router.put("/{review_id}")
async def update_review(review_id: int, update: ReviewUpdate, actor_id: str = Query(...)):
    """Update a review."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check exists
    cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    old_status = existing["status"]
    
    # Build update
    updates = []
    params = []
    
    if update.status is not None:
        if update.status not in ["pending", "approved", "rejected"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        updates.append("status = ?")
        params.append(update.status)
    
    if update.rating is not None:
        updates.append("rating = ?")
        params.append(update.rating)
    
    if update.comment is not None:
        updates.append("comment = ?")
        params.append(update.comment)
    
    if update.corrections is not None:
        updates.append("corrections = ?")
        params.append(json.dumps(update.corrections))
    
    if not updates:
        conn.close()
        raise HTTPException(status_code=400, detail="No updates provided")
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    
    query = f"UPDATE reviews SET {', '.join(updates)} WHERE id = ?"
    params.append(review_id)
    
    cursor.execute(query, params)
    
    # Log history
    new_status = update.status if update.status else old_status
    cursor.execute("""
        INSERT INTO review_history (review_id, action, old_status, new_status, actor_id)
        VALUES (?, 'updated', ?, ?, ?)
    """, (review_id, old_status, new_status, actor_id))
    
    conn.commit()
    
    # Fetch updated
    cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    row = cursor.fetchone()
    conn.close()
    
    result = dict(row)
    if result.get("corrections"):
        try:
            result["corrections"] = json.loads(result["corrections"])
        except:
            pass
    
    return result


@router.delete("/{review_id}")
async def delete_review(review_id: int, actor_id: str = Query(...)):
    """Delete a review."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM reviews WHERE id = ?", (review_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    
    cursor.execute("DELETE FROM review_history WHERE review_id = ?", (review_id,))
    cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    
    conn.commit()
    conn.close()
    
    return {"deleted": True, "id": review_id}


@router.get("/span/{span_id}")
async def get_reviews_by_span(span_id: str):
    """Get all reviews for a specific span."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM reviews 
        WHERE span_id = ? 
        ORDER BY created_at DESC
    """, (span_id,))
    rows = cursor.fetchall()
    conn.close()
    
    reviews = []
    for row in rows:
        review = dict(row)
        if review.get("corrections"):
            try:
                review["corrections"] = json.loads(review["corrections"])
            except:
                pass
        reviews.append(review)
    
    return {
        "span_id": span_id,
        "total": len(reviews),
        "reviews": reviews
    }


@router.get("/export")
async def export_reviews(
    status: Optional[str] = Query(None, description="Filter by status"),
    format: str = Query("json", description="Export format: json or csv")
):
    """Export reviews for external processing."""
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM reviews"
    params = []
    
    if status:
        query += " WHERE status = ?"
        params.append(status)
    
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    reviews = []
    for row in rows:
        review = dict(row)
        if review.get("corrections"):
            try:
                review["corrections"] = json.loads(review["corrections"])
            except:
                pass
        reviews.append(review)
    
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "total": len(reviews),
        "format": format,
        "reviews": reviews
    }
