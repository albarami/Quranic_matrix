#!/usr/bin/env python3
"""
Set up PostgreSQL database for QBM production.

Usage:
    # With PostgreSQL
    DATABASE_URL=postgresql://user:pass@localhost/qbm python src/scripts/setup_database.py
    
    # With SQLite (for development)
    DATABASE_URL=sqlite:///data/qbm.db python src/scripts/setup_database.py
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    JSON, ForeignKey, DateTime, Text, Index
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

Base = declarative_base()


class Span(Base):
    """QBM span record - core annotation unit."""
    __tablename__ = 'spans'
    
    id = Column(String(50), primary_key=True)
    surah = Column(Integer, nullable=False, index=True)
    ayah = Column(Integer, nullable=False, index=True)
    token_start = Column(Integer)
    token_end = Column(Integer)
    raw_text_ar = Column(Text, nullable=False)
    
    # Behavior layer
    behavior_form = Column(String(50))
    behavior_concepts = Column(JSON)
    
    # Agent layer
    agent_type = Column(String(50))
    agent_subtype = Column(String(50))
    agent_referent = Column(String(100))
    
    # Thematic constructs
    thematic_constructs = Column(JSON)
    
    # Normative layer
    normative_textual = Column(JSON)
    
    # Periodicity
    periodicity = Column(JSON)
    
    # Workflow
    status = Column(String(20), default='draft', index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assertions = relationship("Assertion", back_populates="span", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="span", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_span_ref', 'surah', 'ayah'),
    )


class Assertion(Base):
    """Individual assertion within a span."""
    __tablename__ = 'assertions'
    
    id = Column(String(60), primary_key=True)
    span_id = Column(String(50), ForeignKey('spans.id'), nullable=False, index=True)
    
    axis = Column(String(30), nullable=False, index=True)
    value = Column(String(50), nullable=False)
    support_type = Column(String(30))
    confidence = Column(Float)
    negated = Column(Boolean, default=False)
    
    # Justification
    justification_code = Column(String(30))
    justification = Column(Text)
    
    # Organ-specific fields
    organ_semantic_domains = Column(JSON)
    
    span = relationship("Span", back_populates="assertions")


class Review(Base):
    """Review/adjudication record for a span."""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    span_id = Column(String(50), ForeignKey('spans.id'), nullable=False, index=True)
    
    annotator_id = Column(String(50), index=True)
    reviewer_id = Column(String(50))
    status = Column(String(20))  # approved, disputed, pending
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    span = relationship("Span", back_populates="reviews")


class Annotator(Base):
    """Annotator profile and statistics."""
    __tablename__ = 'annotators'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    role = Column(String(30))  # annotator, reviewer, lead
    
    # Statistics
    spans_annotated = Column(Integer, default=0)
    spans_reviewed = Column(Integer, default=0)
    iaa_score = Column(Float)
    
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TafsirConsultation(Base):
    """Log of tafsir consultations during annotation."""
    __tablename__ = 'tafsir_consultations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    span_id = Column(String(50), ForeignKey('spans.id'), index=True)
    annotator_id = Column(String(50))
    
    tafsir_source = Column(String(50))
    influenced_decision = Column(Boolean, default=False)
    note = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class CoverageTracker(Base):
    """Track annotation coverage by surah."""
    __tablename__ = 'coverage_tracker'
    
    surah = Column(Integer, primary_key=True)
    total_ayat = Column(Integer, nullable=False)
    annotated_ayat = Column(Integer, default=0)
    reviewed_ayat = Column(Integer, default=0)
    approved_ayat = Column(Integer, default=0)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def setup_database(db_url: str = None):
    """Create all database tables."""
    db_url = db_url or os.environ.get('DATABASE_URL', 'sqlite:///data/qbm.db')
    
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    
    print(f"Database tables created at {db_url}")
    print("Tables created:")
    for table in Base.metadata.tables:
        print(f"  - {table}")
    
    return engine


def init_coverage_tracker(engine):
    """Initialize coverage tracker with surah ayat counts."""
    SURAH_AYAT = [
        7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
        123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
        112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
        34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
        54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
        60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
        14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
        28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
        29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
        15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
        11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
        5, 4, 5, 6
    ]
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    for surah_num, ayat_count in enumerate(SURAH_AYAT, 1):
        existing = session.query(CoverageTracker).filter_by(surah=surah_num).first()
        if not existing:
            tracker = CoverageTracker(surah=surah_num, total_ayat=ayat_count)
            session.add(tracker)
    
    session.commit()
    session.close()
    print(f"Coverage tracker initialized for 114 surahs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Set up QBM production database")
    parser.add_argument("--url", help="Database URL (or set DATABASE_URL env var)")
    parser.add_argument("--init-coverage", action="store_true", help="Initialize coverage tracker")
    args = parser.parse_args()
    
    engine = setup_database(args.url)
    
    if args.init_coverage:
        init_coverage_tracker(engine)
