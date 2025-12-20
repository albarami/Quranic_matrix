#!/usr/bin/env python3
"""
Validate QBM span records against JSON schema.

Usage:
    python validate_schema.py <filepath>
"""

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("Please install jsonschema: pip install jsonschema")
    sys.exit(1)


def load_schema():
    """Load the QBM span schema."""
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "qbm_span_record_v1.schema.json"
    if not schema_path.exists():
        # Fallback to inline minimal schema
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "QBM Span Record",
            "type": "object",
            "required": ["span_id", "reference"],
            "properties": {
                "span_id": {"type": "string", "pattern": "^QBM_[0-9]{5}$"},
                "reference": {
                    "type": "object",
                    "required": ["surah", "ayah"],
                    "properties": {
                        "surah": {"type": "integer", "minimum": 1, "maximum": 114},
                        "ayah": {"type": "integer", "minimum": 1}
                    }
                }
            }
        }
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_span(span: dict, schema: dict) -> list:
    """Validate a single span, return list of errors."""
    errors = []
    validator = Draft7Validator(schema)
    for error in validator.iter_errors(span):
        errors.append({
            "path": list(error.path),
            "message": error.message,
            "value": str(error.instance)[:100]
        })
    return errors


def validate_file(filepath: str) -> dict:
    """Validate all spans in a file."""
    schema = load_schema()
    
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        spans = data
    elif "annotations" in data:
        spans = data["annotations"]
    elif "selections" in data:
        spans = data["selections"]
    else:
        spans = [data]
    
    results = {
        "total": len(spans),
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    for i, span in enumerate(spans):
        errors = validate_span(span, schema)
        if errors:
            results["invalid"] += 1
            results["errors"].append({
                "index": i,
                "id": span.get("span_id", span.get("id", f"span_{i}")),
                "errors": errors
            })
        else:
            results["valid"] += 1
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <filepath>")
        sys.exit(1)
    
    results = validate_file(sys.argv[1])
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print(f"\n--- Summary ---")
    print(f"Total: {results['total']}")
    print(f"Valid: {results['valid']}")
    print(f"Invalid: {results['invalid']}")
    
    if results["invalid"] > 0:
        sys.exit(1)
