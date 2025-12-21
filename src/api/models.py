"""Pydantic models for QBM API."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Reference(BaseModel):
    """Quran reference (surah:ayah)."""
    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)


class Agent(BaseModel):
    """Agent performing the behavior."""
    type: str = Field(..., description="Agent type (e.g., AGT_ALLAH, AGT_BELIEVER)")
    referent: Optional[str] = None


class Normative(BaseModel):
    """Normative layer annotations."""
    evaluation: Optional[str] = None
    deontic_signal: Optional[str] = None
    speech_mode: Optional[str] = None


class Axes(BaseModel):
    """Behavioral axes."""
    systemic: Optional[str] = None


class Action(BaseModel):
    """Action annotations."""
    textual_eval: Optional[str] = None


class Span(BaseModel):
    """QBM annotation span."""
    id: Optional[str] = Field(None, alias="span_id")
    reference: Reference
    text_ar: Optional[str] = None
    agent: Optional[Agent] = None
    behavior_form: Optional[str] = None
    normative: Optional[Normative] = None
    axes: Optional[Axes] = None
    action: Optional[Action] = None
    
    class Config:
        populate_by_name = True


class SpanResponse(BaseModel):
    """Response containing a single span."""
    span: Span


class SpansResponse(BaseModel):
    """Response containing multiple spans."""
    total: int
    spans: List[Span]


class DatasetMetadata(BaseModel):
    """Dataset metadata."""
    tier: str
    version: str
    exported_at: str
    total_spans: int


class DatasetResponse(BaseModel):
    """Full dataset response."""
    metadata: DatasetMetadata
    spans: List[Span]


class StatsResponse(BaseModel):
    """Dataset statistics response."""
    total_spans: int
    unique_surahs: int
    unique_ayat: int
    agent_types: Dict[str, int]
    behavior_forms: Dict[str, int]
    evaluations: Dict[str, int]
    deontic_signals: Dict[str, int]


class VocabularyItem(BaseModel):
    """Vocabulary item."""
    value: str
    label: Optional[str] = None
    description: Optional[str] = None


class VocabulariesResponse(BaseModel):
    """Controlled vocabularies response."""
    agent_types: List[str]
    behavior_forms: List[str]
    evaluations: List[str]
    deontic_signals: List[str]
    speech_modes: List[str]
    systemic: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    dataset_loaded: bool
