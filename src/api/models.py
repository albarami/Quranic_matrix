#!/usr/bin/env python3
"""Pydantic models for QBM API responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Reference(BaseModel):
    """Quran reference (surah:ayah or surah:ayah:span)."""
    surah: int = Field(..., ge=1, le=114, description="Surah number (1-114)")
    ayah: int = Field(..., ge=1, description="Ayah number")
    span: Optional[int] = Field(None, ge=1, description="Span number within ayah")


class Agent(BaseModel):
    """Agent performing/experiencing the behavior."""
    type: str = Field(..., description="Agent type code (e.g., AGT_BELIEVER)")
    label_en: Optional[str] = Field(None, description="English label")
    label_ar: Optional[str] = Field(None, description="Arabic label")


class Behavior(BaseModel):
    """Behavioral classification."""
    form: str = Field(..., description="Behavior form (inner_state, speech_act, etc.)")
    action_class: Optional[str] = Field(None, description="Action classification")


class Normative(BaseModel):
    """Normative/textual analysis."""
    speech_mode: str = Field(..., description="Speech mode (command, prohibition, informative, narrative)")
    evaluation: str = Field(..., description="Evaluation (praise, blame, promise, warning, neutral)")
    deontic_signal: str = Field(..., description="Deontic signal (amr, nahy, targhib, tarhib, khabar)")
    textual_eval: Optional[str] = Field(None, description="Textual evaluation")


class Axes(BaseModel):
    """Situational axes."""
    situational: str = Field(..., description="Situational axis value")


class Evidence(BaseModel):
    """Evidence and support."""
    support_type: str = Field(..., description="Type of textual support")
    tafsir_consulted: Optional[bool] = Field(None, description="Whether tafsir was consulted")


class SpanAnnotation(BaseModel):
    """Complete span annotation."""
    span_id: str = Field(..., description="Unique span identifier (e.g., QBM_00001)")
    reference: Reference
    raw_text_ar: str = Field(..., description="Arabic text of the span")
    translation_en: Optional[str] = Field(None, description="English translation")
    agent: Agent
    behavior: Behavior
    normative: Normative
    axes: Optional[Axes] = None
    evidence: Optional[Evidence] = None
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DatasetMeta(BaseModel):
    """Dataset metadata."""
    tier: str = Field(..., description="Dataset tier (gold, silver, research)")
    version: str = Field(..., description="Dataset version")
    exported_at: datetime
    total_spans: int
    total_ayat: int
    total_surahs: int
    avg_iaa_kappa: Optional[float] = Field(None, description="Average Cohen's Kappa")


class DatasetResponse(BaseModel):
    """Full dataset response."""
    meta: DatasetMeta
    spans: List[SpanAnnotation]


class SearchQuery(BaseModel):
    """Search query parameters."""
    surah: Optional[int] = Field(None, ge=1, le=114, description="Filter by surah")
    agent_type: Optional[str] = Field(None, description="Filter by agent type")
    behavior_form: Optional[str] = Field(None, description="Filter by behavior form")
    evaluation: Optional[str] = Field(None, description="Filter by evaluation")
    deontic_signal: Optional[str] = Field(None, description="Filter by deontic signal")
    text_search: Optional[str] = Field(None, description="Search in Arabic text")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")


class StatsResponse(BaseModel):
    """Dataset statistics response."""
    total_spans: int
    total_ayat: int
    surahs_covered: int
    agent_distribution: Dict[str, int]
    behavior_distribution: Dict[str, int]
    evaluation_distribution: Dict[str, int]
    deontic_distribution: Dict[str, int]
    avg_iaa_kappa: Optional[float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    dataset_loaded: bool
    spans_available: int
