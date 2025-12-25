"""
Phase 5: Embedding Evaluator Tests
Tests for the embedding model evaluation infrastructure.
"""

import pytest
import json
from pathlib import Path


class TestGoldBenchmark:
    """Test the gold standard benchmark file"""
    
    def test_gold_file_exists(self):
        """Gold benchmark file should exist"""
        gold_file = Path("data/evaluation/semantic_similarity_gold.jsonl")
        assert gold_file.exists(), "Gold benchmark file missing"
    
    def test_gold_file_format(self):
        """Gold file should have correct format"""
        gold_file = Path("data/evaluation/semantic_similarity_gold.jsonl")
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    entry = json.loads(line)
                    assert "id" in entry, f"Line {i}: missing id"
                    assert "text_a" in entry, f"Line {i}: missing text_a"
                    assert "text_b" in entry, f"Line {i}: missing text_b"
                    assert "expected_similarity" in entry, f"Line {i}: missing expected_similarity"
                    assert entry["expected_similarity"] in ["high", "low"], f"Line {i}: invalid expected_similarity"
    
    def test_gold_has_both_categories(self):
        """Gold file should have both high and low similarity pairs"""
        gold_file = Path("data/evaluation/semantic_similarity_gold.jsonl")
        
        high_count = 0
        low_count = 0
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry["expected_similarity"] == "high":
                        high_count += 1
                    else:
                        low_count += 1
        
        assert high_count >= 10, f"Need at least 10 high similarity pairs, got {high_count}"
        assert low_count >= 10, f"Need at least 10 low similarity pairs, got {low_count}"


class TestModelRegistry:
    """Test the model registry"""
    
    def test_registry_file_exists(self):
        """Registry file should exist after evaluation"""
        registry_file = Path("data/models/registry.json")
        assert registry_file.exists(), "Registry file missing"
    
    def test_registry_format(self):
        """Registry should have correct format"""
        registry_file = Path("data/models/registry.json")
        
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        assert "active_model" in registry
        assert "models" in registry
        assert "evaluation_history" in registry
    
    def test_registry_has_evaluated_models(self):
        """Registry should have at least one evaluated model"""
        registry_file = Path("data/models/registry.json")
        
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        assert len(registry["models"]) >= 1, "No models evaluated"
    
    def test_model_entries_have_required_fields(self):
        """Each model entry should have required fields"""
        registry_file = Path("data/models/registry.json")
        
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        required_fields = ["name", "accuracy", "passes_threshold"]
        
        for model_name, model_data in registry["models"].items():
            for field in required_fields:
                assert field in model_data, f"Model {model_name} missing field: {field}"


class TestEvaluationResults:
    """Test the evaluation results"""
    
    def test_results_file_exists(self):
        """Evaluation results file should exist"""
        results_file = Path("data/evaluation/evaluation_results.json")
        assert results_file.exists(), "Evaluation results file missing"
    
    def test_results_have_metrics(self):
        """Results should include key metrics"""
        results_file = Path("data/evaluation/evaluation_results.json")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        required_metrics = ["accuracy", "high_sim_accuracy", "low_sim_accuracy", "opposite_separation"]
        
        for model_name, model_results in results.items():
            for metric in required_metrics:
                assert metric in model_results, f"Model {model_name} missing metric: {metric}"


class TestEmbeddingEvaluator:
    """Test the EmbeddingEvaluator class"""
    
    def test_evaluator_loads_gold_pairs(self):
        """Evaluator should load gold pairs on init"""
        from src.ml.embedding_evaluator import EmbeddingEvaluator
        
        evaluator = EmbeddingEvaluator()
        assert len(evaluator.gold_pairs) >= 20, "Should have at least 20 gold pairs"
    
    def test_evaluation_result_dataclass(self):
        """EvaluationResult should have correct fields"""
        from src.ml.embedding_evaluator import EvaluationResult
        
        result = EvaluationResult(
            model_name="test",
            accuracy=0.5,
            high_sim_accuracy=0.6,
            low_sim_accuracy=0.4,
            avg_high_similarity=0.7,
            avg_low_similarity=0.3,
            opposite_separation=0.4,
            details=[],
        )
        
        result_dict = result.to_dict()
        assert result_dict["accuracy"] == 0.5
        assert result_dict["passes_threshold"] == False  # 0.5 < 0.75


@pytest.mark.slow
class TestModelEvaluation:
    """Test actual model evaluation (slow - requires model loading)"""
    
    def test_evaluate_current_model(self):
        """Evaluate the current model"""
        from src.ml.embedding_evaluator import EmbeddingEvaluator
        
        evaluator = EmbeddingEvaluator()
        evaluator.load_model("aubmindlab/bert-base-arabertv2")
        result = evaluator.evaluate()
        
        # Should return valid metrics
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.high_sim_accuracy <= 1
        assert 0 <= result.low_sim_accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
