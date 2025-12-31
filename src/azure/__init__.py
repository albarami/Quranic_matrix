"""
Azure OpenAI Integration Module

Phase 6: Tool-first orchestration with function-calling.
- Function-calling tools for QBM queries
- Verifier gate for citation validity
- Fail-closed on violations
"""

from .tools import QBMTools, TOOL_DEFINITIONS
from .orchestrator import QBMOrchestrator
from .verifier import CitationVerifier, VerificationResult

__all__ = [
    "QBMTools",
    "TOOL_DEFINITIONS", 
    "QBMOrchestrator",
    "CitationVerifier",
    "VerificationResult",
]
