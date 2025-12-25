"""
QBM Provenance Schema - Phase 3
Tracks the origin and extraction details of all annotations.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnnotationWithProvenance(BaseModel):
    """
    Annotation with full provenance tracking.
    Every extracted behavior annotation must include this metadata.
    """
    text: str = Field(..., description="The annotated text span")
    behavior: str = Field(..., description="The identified behavior")
    surah: int = Field(..., ge=1, le=114, description="Surah number (1-114)")
    ayah: int = Field(..., ge=1, description="Ayah number")
    tafsir_source: Literal[
        "ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"
    ] = Field(..., description="Source tafsir")
    char_start: int = Field(..., ge=0, description="Start character offset in source")
    char_end: int = Field(..., ge=0, description="End character offset in source")
    source_file: str = Field(..., description="Original source file path")
    extractor_version: str = Field(default="2.0.0", description="Extractor version")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of extraction"
    )
    
    # Optional fields for enhanced provenance
    extraction_method: Optional[Literal[
        "morphology", "keyword", "ml_classifier", "manual"
    ]] = Field(default=None, description="Method used for extraction")
    reviewer: Optional[str] = Field(default=None, description="Human reviewer if any")
    review_status: Literal[
        "pending", "approved", "rejected", "needs_review"
    ] = Field(default="pending", description="Review status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CleanedTafsirRecord(BaseModel):
    """
    A cleaned tafsir record with provenance.
    Used for storing cleaned tafsir content.
    """
    source_id: str = Field(..., description="Tafsir source identifier")
    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    text_original: str = Field(..., description="Original text with HTML")
    text_cleaned: str = Field(..., description="Cleaned text without HTML")
    html_contamination_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Percentage of original text that was HTML"
    )
    cleaned_at: datetime = Field(default_factory=datetime.utcnow)
    cleaner_version: str = Field(default="1.0.0")


class ProvenanceReport(BaseModel):
    """
    Summary report of provenance for a batch of annotations.
    """
    total_annotations: int
    annotations_with_provenance: int
    provenance_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description="Percentage of annotations with complete provenance"
    )
    sources_breakdown: dict = Field(
        default_factory=dict,
        description="Count of annotations per tafsir source"
    )
    extraction_methods_breakdown: dict = Field(
        default_factory=dict,
        description="Count of annotations per extraction method"
    )
    avg_confidence: float = Field(default=0.0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
