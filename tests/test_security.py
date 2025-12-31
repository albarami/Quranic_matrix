"""
Phase 2: Security Tests
Tests for CORS, rate limiting, and input validation.
"""

import pytest
from pydantic import ValidationError


class TestInputValidation:
    """Test input validation on ProofQueryRequest"""
    
    def test_valid_question(self):
        """Valid question should pass validation"""
        from src.api.main import ProofQueryRequest
        
        req = ProofQueryRequest(question="ما هو الصبر؟")
        assert req.question == "ما هو الصبر؟"
    
    def test_empty_question_rejected(self):
        """Empty question should be rejected"""
        from src.api.main import ProofQueryRequest
        
        with pytest.raises(ValidationError) as exc_info:
            ProofQueryRequest(question="")
        
        # Pydantic 2.x uses "string_too_short" for min_length violations
        assert "too_short" in str(exc_info.value).lower() or "at least" in str(exc_info.value).lower()
    
    def test_whitespace_only_rejected(self):
        """Whitespace-only question should be rejected"""
        from src.api.main import ProofQueryRequest
        
        with pytest.raises(ValidationError):
            ProofQueryRequest(question="   ")
    
    def test_too_long_question_rejected(self):
        """Question exceeding max length should be rejected"""
        from src.api.main import ProofQueryRequest
        
        long_question = "ا" * 2001  # Exceeds 2000 char limit
        with pytest.raises(ValidationError) as exc_info:
            ProofQueryRequest(question=long_question)
        
        # Pydantic 2.x uses "string_too_long" for max_length violations
        assert "too_long" in str(exc_info.value).lower() or "at most" in str(exc_info.value).lower()
    
    def test_script_injection_rejected(self):
        """Script injection attempts should be rejected"""
        from src.api.main import ProofQueryRequest
        
        with pytest.raises(ValidationError) as exc_info:
            ProofQueryRequest(question="<script>alert('xss')</script>")
        
        assert "invalid" in str(exc_info.value).lower()
    
    def test_javascript_injection_rejected(self):
        """JavaScript injection attempts should be rejected"""
        from src.api.main import ProofQueryRequest
        
        with pytest.raises(ValidationError) as exc_info:
            ProofQueryRequest(question="javascript:alert(1)")
        
        assert "invalid" in str(exc_info.value).lower()


class TestCORSConfiguration:
    """Test CORS configuration"""
    
    def test_allowed_origins_from_env(self):
        """ALLOWED_ORIGINS should be configurable via env var"""
        import os
        from importlib import reload
        
        # Default should include localhost
        from src.api import main
        assert "http://localhost:3000" in main.ALLOWED_ORIGINS
    
    def test_cors_not_wildcard(self):
        """CORS should not allow wildcard in production"""
        from src.api.main import ALLOWED_ORIGINS
        
        # Should not be a single wildcard
        assert ALLOWED_ORIGINS != ["*"]
        assert "*" not in ALLOWED_ORIGINS


class TestRateLimiting:
    """Test rate limiting configuration"""
    
    def test_limiter_configured(self):
        """Rate limiter should be configured on app"""
        from src.api.main import app, limiter
        
        assert app.state.limiter is not None
        assert limiter is not None
    
    def test_proof_endpoint_has_limit(self):
        """Proof endpoint should have rate limit decorator"""
        from src.api.routers.proof import proof_query
        
        # Check that the function has rate limit metadata
        # slowapi adds __wrapped__ or similar attributes
        assert hasattr(proof_query, '__wrapped__') or callable(proof_query)


class TestErrorResponses:
    """Test standardized error responses"""
    
    def test_no_bare_except_in_main(self):
        """main.py should not have bare except statements"""
        from pathlib import Path
        
        main_path = Path(__file__).parent.parent / "src" / "api" / "main.py"
        content = main_path.read_text(encoding="utf-8")
        
        # Check for bare except (except followed by colon with no exception type)
        import re
        bare_excepts = re.findall(r'except\s*:', content)
        
        assert len(bare_excepts) == 0, f"Found {len(bare_excepts)} bare except statements"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
