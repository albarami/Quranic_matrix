# Quranic Human-Behavior Classification Matrix (QBM)
## Complete Project Plan: Pilot to Production

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Phase 0: Project Setup (Week 1-2)](#2-phase-0-project-setup-week-1-2)
3. [Phase 1: Self-Calibration (Week 3-4)](#3-phase-1-self-calibration-week-3-4)
4. [Phase 2: Micro-Pilot (Weeks 5-8)](#4-phase-2-micro-pilot-weeks-5-8)
5. [Phase 3: Full Pilot (Weeks 9-16)](#5-phase-3-full-pilot-weeks-9-16)
6. [Phase 4: Tafsir Integration (Weeks 17-24)](#6-phase-4-tafsir-integration-weeks-17-24)
7. [Phase 5: Scale-Up (Weeks 25-40)](#7-phase-5-scale-up-weeks-25-40)
8. [Phase 6: Full Quran Coverage (Weeks 41-60)](#8-phase-6-full-quran-coverage-weeks-41-60)
9. [Phase 7: Production Release (Weeks 61-70)](#9-phase-7-production-release-weeks-61-70)
10. [Phase 8: Publication & Launch (Weeks 71-78)](#10-phase-8-publication--launch-weeks-71-78)
11. [Phase 9: Maintenance & Growth (Ongoing)](#11-phase-9-maintenance--growth-ongoing)
12. [Git Workflow](#12-git-workflow)
13. [Quality Gates](#13-quality-gates)
14. [Risk Management](#14-risk-management)
15. [Resource Requirements](#15-resource-requirements)
16. [Success Metrics](#16-success-metrics)

---

# 1. PROJECT OVERVIEW

## 1.1 Vision
Create the world's first academically rigorous, computationally structured dataset of Quranic behavioral classifications, grounded in Islamic scholarship and suitable for research, education, and ethical AI applications.

## 1.2 Key Deliverables
| Deliverable | Description | Pilot Target | Production Target |
|-------------|-------------|--------------|-------------------|
| Gold Dataset | Fully validated, reviewer-approved annotations | 500 spans | **4,000+ spans** |
| Silver Dataset | High-confidence annotations meeting ESS threshold | 1,000 spans | **10,000+ spans** |
| Research Dataset | All annotations including disputed | 2,500 spans | **20,000+ spans** |
| **Quran Coverage** | Ayat annotated | 500 ayat | **6,236 ayat (100%)** |
| **Tafsir Integration** | Structured tafsir database + lookup tools | Manual | **5+ sources, DB + API** |
| Coding Manual | Comprehensive annotator training guide | v1.0 | **v2.0** |
| API/Tools | Export tools, validation scripts, graph builder | Basic | **Production-ready** |
| Publications | Academic papers documenting methodology | Draft | **2-3 papers** |

## 1.3 Timeline Summary
| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 0: Setup | Weeks 1-2 | Repository ready, tools configured |
| Phase 1: Self-Calibration | Weeks 3-4 | Personal baseline established |
| Phase 2: Micro-Pilot | Weeks 5-8 | 100 spans, IAA measured |
| Phase 3: Full Pilot | Weeks 9-16 | 500 spans, Gold v0.1 released |
| **Phase 4: Tafsir Integration** | **Weeks 17-24** | **Tafsir DB + lookup tools** |
| **Phase 5: Scale-Up** | **Weeks 25-40** | **3,000 ayat, 10,000 spans** |
| **Phase 6: Full Coverage** | **Weeks 41-60** | **6,236 ayat (100%), 15,000+ spans** |
| **Phase 7: Production** | **Weeks 61-70** | **Full release, API live** |
| **Phase 8: Publication** | **Weeks 71-78** | **2-3 papers, v1.0.0 release** |
| Phase 9: Maintenance | Ongoing | Continuous improvement |

**Total Duration: 78 weeks (~18 months)**

## 1.4 Budget Summary
| Category | Pilot (Weeks 1-16) | Full Project (78 weeks) |
|----------|-------------------|-------------------------|
| Personnel | $15,000 | $217,800 |
| Infrastructure | $1,000 | $4,500 |
| Software/Tools | $500 | $3,000 |
| Contingency | $2,000 | $33,795 |
| **TOTAL** | **$18,500** | **$259,095** |

> **Note:** Budget can be significantly reduced with volunteer annotators and academic partnerships. Minimum viable: $80,000 with reduced team and extended timeline.

---

# 2. PHASE 0: PROJECT SETUP (Weeks 1-2)

## 2.1 Day 1: GitHub Repository Setup

### Task 0.1.1: Create Repository
```bash
# Create new repository on GitHub
# Name: quranic-behavior-matrix
# Visibility: Private (initially)
# Initialize with README

# Clone locally
git clone https://github.com/YOUR_USERNAME/quranic-behavior-matrix.git
cd quranic-behavior-matrix
```

### Task 0.1.2: Create Folder Structure
```bash
mkdir -p {data/{raw,processed,annotations,exports},docs,src/{scripts,validation,api},config,tests,models}

# Create .gitignore
cat > .gitignore << 'EOF'
# Data files (large)
data/raw/*.json
data/processed/*.json
*.xml

# Keep structure
!data/raw/.gitkeep
!data/processed/.gitkeep

# Python
__pycache__/
*.py[cod]
.env
venv/

# Node
node_modules/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Secrets
*.secret
*.key
credentials.json
EOF

# Create placeholder files
touch data/raw/.gitkeep data/processed/.gitkeep data/annotations/.gitkeep data/exports/.gitkeep
```

### Task 0.1.3: Initial Commit
```bash
git add .
git commit -m "chore: initial project structure"
git push origin main
```

## 2.2 Day 2: Add Project Documentation

### Task 0.2.1: Copy Specification Documents
```bash
# Copy all documents to docs/
cp QURANIC_BEHAVIOR_CLASSIFICATION_MATRIX.md docs/
cp coding_manual_v1.docx docs/
cp quranic_matrix_english_v2.docx docs/
cp quranic_matrix_arabic_v2.docx docs/

git add docs/
git commit -m "docs: add specification and coding manual v1"
git push origin main
```

### Task 0.2.2: Create README.md
```bash
cat > README.md << 'EOF'
# Quranic Human-Behavior Classification Matrix (QBM)

A structured dataset of Quranic behavioral classifications grounded in Islamic scholarship.

## Project Status: Phase 0 - Setup

## Quick Links
- [Specification](docs/QURANIC_BEHAVIOR_CLASSIFICATION_MATRIX.md)
- [Coding Manual](docs/coding_manual_v1.docx)
- [Controlled Vocabularies](config/controlled_vocabularies_v1.json)

## License
Research use only. Contact for permissions.

## Citation
[To be added after publication]
EOF

git add README.md
git commit -m "docs: add project README"
git push origin main
```

## 2.3 Day 3: Add Configuration Files

### Task 0.3.1: Add Vocabularies and Config
```bash
cp controlled_vocabularies_v1.json config/
cp label_studio_config.json config/
cp gold_standard_examples.json config/

git add config/
git commit -m "config: add controlled vocabularies and Label Studio config"
git push origin main
```

### Task 0.3.2: Add Data Files (use Git LFS for large files)
```bash
# Install Git LFS
git lfs install

# Track large JSON files
git lfs track "*.json"
git add .gitattributes

# Add data files
cp quran_tokenized_full.json data/raw/
cp quran_pilot_50.json data/processed/
cp label_studio_import.json data/processed/
cp quran_index.json data/raw/

git add data/
git commit -m "data: add tokenized Quran text and pilot selection"
git push origin main
```

## 2.4 Day 4: Set Up Development Environment

### Task 0.4.1: Create Python Environment
```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Validation
jsonschema>=4.17.0
pydantic>=2.0.0

# NLP/Arabic
camel-tools>=1.5.0
pyarabic>=0.6.15

# Statistics (IAA)
scikit-learn>=1.3.0
krippendorff>=0.6.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Graph
neo4j>=5.0.0
networkx>=3.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Export
python-docx>=0.8.11
openpyxl>=3.1.0
EOF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt

git add requirements.txt
git commit -m "chore: add Python dependencies"
git push origin main
```

### Task 0.4.2: Create JSON Schema for Validation
```bash
cat > config/span_schema.json << 'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "QBM Span Record",
  "type": "object",
  "required": ["id", "reference", "span", "agent", "assertions", "normative_textual", "review"],
  "properties": {
    "id": {"type": "string", "pattern": "^QBM_[0-9]{5}$"},
    "quran_text_version": {"type": "string"},
    "tokenization_id": {"type": "string"},
    "reference": {
      "type": "object",
      "required": ["surah", "ayah"],
      "properties": {
        "surah": {"type": "integer", "minimum": 1, "maximum": 114},
        "ayah": {"type": "integer", "minimum": 1}
      }
    },
    "span": {
      "type": "object",
      "required": ["token_start", "token_end", "raw_text_ar"],
      "properties": {
        "token_start": {"type": "integer", "minimum": 0},
        "token_end": {"type": "integer", "minimum": 1},
        "raw_text_ar": {"type": "string"},
        "boundary_confidence": {"enum": ["certain", "probable", "uncertain"]}
      }
    },
    "behavior": {
      "type": "object",
      "properties": {
        "concepts": {"type": "array", "items": {"type": "string", "pattern": "^BEH_"}},
        "form": {"enum": ["speech_act", "physical_act", "inner_state", "trait_disposition", "relational_act", "omission", "mixed", "unknown"]}
      }
    },
    "agent": {
      "type": "object",
      "required": ["type"],
      "properties": {
        "type": {"type": "string", "pattern": "^AGT_"},
        "explicit": {"type": "boolean"},
        "support_type": {"enum": ["direct", "indirect"]}
      }
    },
    "assertions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["assertion_id", "axis", "value"],
        "properties": {
          "assertion_id": {"type": "string"},
          "axis": {"type": "string", "pattern": "^AX_"},
          "value": {"type": "string"},
          "support_type": {"enum": ["direct", "indirect"]},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "negated": {"type": "boolean"},
          "negation_type": {"enum": ["absolute", "conditional", "exceptionless_affirmation", "unknown"]}
        }
      }
    },
    "normative_textual": {
      "type": "object",
      "properties": {
        "speech_mode": {"enum": ["command", "prohibition", "informative", "narrative", "parable", "unknown"]},
        "evaluation": {"enum": ["praise", "blame", "warning", "promise", "neutral", "mixed", "unknown"]},
        "quran_deontic_signal": {"enum": ["amr", "nahy", "targhib", "tarhib", "khabar"]}
      }
    },
    "review": {
      "type": "object",
      "required": ["status"],
      "properties": {
        "status": {"enum": ["draft", "disputed", "approved"]},
        "annotator_id": {"type": "string"},
        "reviewer_id": {"type": "string"}
      }
    }
  }
}
EOF

git add config/span_schema.json
git commit -m "config: add JSON schema for span validation"
git push origin main
```

## 2.5 Day 5: Set Up Label Studio

### Task 0.5.1: Install and Configure Label Studio
```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start

# In browser (http://localhost:8080):
# 1. Create account
# 2. Create project: "QBM Pilot"
# 3. Settings > Labeling Interface > paste XML from config/label_studio_config.json
# 4. Import > Upload label_studio_import.json
```

### Task 0.5.2: Document Setup
```bash
cat > docs/SETUP.md << 'EOF'
# Development Setup Guide

## Prerequisites
- Python 3.10+
- Git LFS
- Node.js 18+ (optional, for tooling)

## Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/quranic-behavior-matrix.git
cd quranic-behavior-matrix
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Label Studio Setup
1. `label-studio start`
2. Create project "QBM Pilot"
3. Import config from `config/label_studio_config.json`
4. Import data from `data/processed/label_studio_import.json`

## Running Tests
```bash
pytest tests/ -v
```
EOF

git add docs/SETUP.md
git commit -m "docs: add development setup guide"
git push origin main
```

## 2.6 Day 6-7: Create Validation Scripts

### Task 0.6.1: Schema Validation Script
```bash
cat > src/validation/validate_schema.py << 'EOF'
#!/usr/bin/env python3
"""Validate QBM span records against JSON schema."""

import json
import sys
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator

def load_schema():
    schema_path = Path(__file__).parent.parent.parent / "config" / "span_schema.json"
    with open(schema_path) as f:
        return json.load(f)

def validate_span(span: dict, schema: dict) -> list:
    """Validate a single span, return list of errors."""
    errors = []
    validator = Draft7Validator(schema)
    for error in validator.iter_errors(span):
        errors.append({
            "path": list(error.path),
            "message": error.message,
            "value": error.instance
        })
    return errors

def validate_file(filepath: str) -> dict:
    """Validate all spans in a file."""
    schema = load_schema()
    
    with open(filepath) as f:
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
                "id": span.get("id", f"span_{i}"),
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
    print(json.dumps(results, indent=2))
    
    if results["invalid"] > 0:
        sys.exit(1)
EOF

chmod +x src/validation/validate_schema.py
git add src/validation/
git commit -m "feat: add schema validation script"
git push origin main
```

### Task 0.6.2: Vocabulary Validation Script
```bash
cat > src/validation/validate_vocabularies.py << 'EOF'
#!/usr/bin/env python3
"""Validate that all IDs in annotations match controlled vocabularies."""

import json
import sys
from pathlib import Path

def load_vocabularies():
    vocab_path = Path(__file__).parent.parent.parent / "config" / "controlled_vocabularies_v1.json"
    with open(vocab_path) as f:
        return json.load(f)

def extract_valid_ids(vocabs: dict) -> dict:
    """Extract all valid IDs from vocabularies."""
    valid_ids = {
        "axes": set(),
        "agents": set(),
        "organs": set(),
        "systemic": set(),
        "spatial": set(),
        "temporal": set(),
        "behavior_concepts": set(),
        "thematic_constructs": set(),
        "action_class": set(),
        "action_textual_eval": set(),
        "periodicity": set(),
        "justification_codes": set()
    }
    
    # Extract IDs from each vocabulary
    for item in vocabs.get("axes", {}).get("items", []):
        valid_ids["axes"].add(item["id"])
    
    for item in vocabs.get("agents", {}).get("items", []):
        valid_ids["agents"].add(item["id"])
    
    for item in vocabs.get("organs", {}).get("items", []):
        valid_ids["organs"].add(item["id"])
    
    for item in vocabs.get("systemic", {}).get("items", []):
        valid_ids["systemic"].add(item["id"])
    
    for item in vocabs.get("spatial", {}).get("items", []):
        valid_ids["spatial"].add(item["id"])
    
    for item in vocabs.get("temporal", {}).get("items", []):
        valid_ids["temporal"].add(item["id"])
    
    # Behavior concepts are nested
    for category, items in vocabs.get("behavior_concepts", {}).get("categories", {}).items():
        for item in items:
            valid_ids["behavior_concepts"].add(item["id"])
    
    for item in vocabs.get("thematic_constructs", {}).get("items", []):
        valid_ids["thematic_constructs"].add(item["id"])
    
    for item in vocabs.get("action_class", {}).get("items", []):
        valid_ids["action_class"].add(item["id"])
    
    for item in vocabs.get("action_textual_eval", {}).get("items", []):
        valid_ids["action_textual_eval"].add(item["id"])
    
    for item in vocabs.get("periodicity", {}).get("items", []):
        valid_ids["periodicity"].add(item["id"])
    
    for item in vocabs.get("justification_codes", {}).get("items", []):
        valid_ids["justification_codes"].add(item["id"])
    
    return valid_ids

def validate_span_vocabularies(span: dict, valid_ids: dict) -> list:
    """Check all IDs in a span against valid vocabularies."""
    errors = []
    
    # Check agent type
    agent_type = span.get("agent", {}).get("type")
    if agent_type and agent_type not in valid_ids["agents"]:
        errors.append(f"Invalid agent type: {agent_type}")
    
    # Check behavior concepts
    for concept in span.get("behavior", {}).get("concepts", []):
        if concept not in valid_ids["behavior_concepts"]:
            errors.append(f"Invalid behavior concept: {concept}")
    
    # Check thematic constructs
    for construct in span.get("thematic_constructs", []):
        if construct not in valid_ids["thematic_constructs"]:
            errors.append(f"Invalid thematic construct: {construct}")
    
    # Check assertions
    for assertion in span.get("assertions", []):
        axis = assertion.get("axis")
        value = assertion.get("value")
        
        if axis and axis not in valid_ids["axes"]:
            errors.append(f"Invalid axis: {axis}")
        
        # Check value based on axis type
        if axis == "AX_ORGAN" and value not in valid_ids["organs"]:
            errors.append(f"Invalid organ: {value}")
        elif axis == "AX_SYSTEMIC" and value not in valid_ids["systemic"]:
            errors.append(f"Invalid systemic: {value}")
        elif axis == "AX_SPATIAL" and value not in valid_ids["spatial"]:
            errors.append(f"Invalid spatial: {value}")
        elif axis == "AX_TEMPORAL" and value not in valid_ids["temporal"]:
            errors.append(f"Invalid temporal: {value}")
        elif axis == "AX_ACTION_CLASS" and value not in valid_ids["action_class"]:
            errors.append(f"Invalid action class: {value}")
        elif axis == "AX_ACTION_TEXTUAL_EVAL" and value not in valid_ids["action_textual_eval"]:
            errors.append(f"Invalid action eval: {value}")
    
    # Check periodicity
    periodicity = span.get("periodicity", {}).get("category")
    if periodicity and periodicity not in valid_ids["periodicity"]:
        errors.append(f"Invalid periodicity: {periodicity}")
    
    return errors

def validate_file(filepath: str) -> dict:
    """Validate all spans in a file against vocabularies."""
    vocabs = load_vocabularies()
    valid_ids = extract_valid_ids(vocabs)
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        spans = data
    elif "annotations" in data:
        spans = data["annotations"]
    else:
        spans = [data]
    
    results = {
        "total": len(spans),
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    for i, span in enumerate(spans):
        errors = validate_span_vocabularies(span, valid_ids)
        if errors:
            results["invalid"] += 1
            results["errors"].append({
                "index": i,
                "id": span.get("id", f"span_{i}"),
                "errors": errors
            })
        else:
            results["valid"] += 1
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_vocabularies.py <filepath>")
        sys.exit(1)
    
    results = validate_file(sys.argv[1])
    print(json.dumps(results, indent=2))
    
    if results["invalid"] > 0:
        sys.exit(1)
EOF

chmod +x src/validation/validate_vocabularies.py
git add src/validation/
git commit -m "feat: add vocabulary validation script"
git push origin main
```

## 2.7 Phase 0 Completion Checklist

```markdown
## Phase 0 Checklist
- [ ] GitHub repository created
- [ ] Folder structure established
- [ ] All specification documents added
- [ ] Controlled vocabularies configured
- [ ] JSON schema defined
- [ ] Python environment set up
- [ ] Label Studio installed and configured
- [ ] Pilot data imported to Label Studio
- [ ] Validation scripts created
- [ ] All changes committed and pushed
```

### Phase 0 Final Commit
```bash
git add .
git commit -m "milestone: complete Phase 0 - project setup"
git tag -a v0.1.0 -m "Phase 0 Complete: Project Setup"
git push origin main --tags
```

---

# 3. PHASE 1: SELF-CALIBRATION (Weeks 3-4)

## 3.1 Day 1-2: Study Materials

### Task 1.1.1: Deep Read of Coding Manual
- Read coding_manual_v1.docx cover to cover
- Highlight unclear sections
- Note questions for clarification

### Task 1.1.2: Study Gold Examples
```bash
# Open gold_standard_examples.json
# Study examples 1-10 in detail
# Note annotation patterns
```

## 3.2 Day 3-4: Blind Annotation Practice

### Task 1.2.1: Annotate Without Answers
1. Open Label Studio
2. Select 10 spans from pilot
3. Annotate completely WITHOUT looking at gold examples
4. Export your annotations

### Task 1.2.2: Create Self-Test Script
```bash
cat > src/scripts/self_test.py << 'EOF'
#!/usr/bin/env python3
"""Compare your annotations against gold standard."""

import json
import sys
from pathlib import Path

def load_gold_standards():
    path = Path(__file__).parent.parent.parent / "config" / "gold_standard_examples.json"
    with open(path) as f:
        data = json.load(f)
    return {ex["reference"]["surah"] * 1000 + ex["reference"]["ayah"]: ex 
            for ex in data["examples"]}

def compare_annotation(yours: dict, gold: dict) -> dict:
    """Compare your annotation to gold standard."""
    results = {
        "matches": [],
        "mismatches": [],
        "missing": [],
        "extra": []
    }
    
    # Compare agent
    your_agent = yours.get("agent", {}).get("type")
    gold_agent = gold.get("agent", {}).get("type")
    if your_agent == gold_agent:
        results["matches"].append(f"agent: {your_agent}")
    else:
        results["mismatches"].append(f"agent: yours={your_agent}, gold={gold_agent}")
    
    # Compare behavior concepts
    your_concepts = set(yours.get("behavior", {}).get("concepts", []))
    gold_concepts = set(gold.get("behavior", {}).get("concepts", []))
    
    for c in your_concepts & gold_concepts:
        results["matches"].append(f"concept: {c}")
    for c in gold_concepts - your_concepts:
        results["missing"].append(f"concept: {c}")
    for c in your_concepts - gold_concepts:
        results["extra"].append(f"concept: {c}")
    
    # Compare normative
    for field in ["speech_mode", "evaluation", "quran_deontic_signal"]:
        your_val = yours.get("normative_textual", {}).get(field)
        gold_val = gold.get("normative_textual", {}).get(field)
        if your_val == gold_val:
            results["matches"].append(f"{field}: {your_val}")
        else:
            results["mismatches"].append(f"{field}: yours={your_val}, gold={gold_val}")
    
    return results

def calculate_score(results: dict) -> float:
    """Calculate agreement score."""
    total = len(results["matches"]) + len(results["mismatches"]) + len(results["missing"])
    if total == 0:
        return 0.0
    return len(results["matches"]) / total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python self_test.py <your_annotations.json>")
        sys.exit(1)
    
    gold = load_gold_standards()
    
    with open(sys.argv[1]) as f:
        yours = json.load(f)
    
    total_score = 0
    count = 0
    
    for annotation in yours:
        ref = annotation.get("reference", {})
        key = ref.get("surah", 0) * 1000 + ref.get("ayah", 0)
        
        if key in gold:
            results = compare_annotation(annotation, gold[key])
            score = calculate_score(results)
            total_score += score
            count += 1
            
            print(f"\n=== {ref.get('surah')}:{ref.get('ayah')} ===")
            print(f"Score: {score:.2%}")
            print(f"Matches: {results['matches']}")
            print(f"Mismatches: {results['mismatches']}")
            print(f"Missing: {results['missing']}")
            print(f"Extra: {results['extra']}")
    
    if count > 0:
        print(f"\n=== OVERALL SCORE: {total_score/count:.2%} ===")
EOF

git add src/scripts/
git commit -m "feat: add self-test comparison script"
git push origin main
```

## 3.3 Day 5-6: Analyze Mistakes

### Task 1.3.1: Document Difficulties
```bash
cat > docs/calibration_notes.md << 'EOF'
# Self-Calibration Notes

## Date: [DATE]

## Difficult Decisions

### 1. Agent Identification
- [Document specific cases where agent was unclear]

### 2. Heart Semantic Domains
- [Document cases where domain selection was difficult]

### 3. Action Class
- [Document volitional vs instinctive edge cases]

### 4. Negation Patterns
- [Document complex negation structures]

## Questions for Clarification
1. [Question 1]
2. [Question 2]

## Proposed Manual Additions
1. [Suggestion 1]
2. [Suggestion 2]

## Personal Accuracy by Category
| Category | Accuracy |
|----------|----------|
| Agent | X% |
| Behavior concepts | X% |
| Normative | X% |
| Organs | X% |
EOF

git add docs/calibration_notes.md
git commit -m "docs: add calibration notes template"
git push origin main
```

## 3.4 Day 7: Update Manual

### Task 1.4.1: Revise Coding Manual
Based on self-calibration findings:
1. Add clarifications for difficult cases
2. Add more examples
3. Update decision flowcharts

```bash
git add docs/
git commit -m "docs: update coding manual based on self-calibration"
git push origin main
```

## 3.5 Phase 1 Completion

```bash
git add .
git commit -m "milestone: complete Phase 1 - self-calibration"
git tag -a v0.2.0 -m "Phase 1 Complete: Self-Calibration"
git push origin main --tags
```

---

# 4. PHASE 2: MICRO-PILOT (Weeks 5-8)

## 4.1 Week 3: Recruit and Train Annotators

### Task 2.1.1: Annotator Requirements
```markdown
## Annotator Job Description

### Requirements
- Native Arabic speaker
- Bachelor's degree minimum (Islamic Studies preferred)
- Familiarity with Quranic Arabic
- Basic tafsir knowledge
- Available 10+ hours/week for 8 weeks

### Compensation
- [Specify hourly rate or stipend]

### Training
- 8 hours initial training
- Ongoing calibration sessions
```

### Task 2.1.2: Training Session Plan
```bash
cat > docs/training_plan.md << 'EOF'
# Annotator Training Plan (8 Hours)

## Session 1: Introduction (2 hours)
- Project overview and goals
- Ethical considerations
- Tool walkthrough (Label Studio)

## Session 2: Core Concepts (2 hours)
- Span segmentation rules
- Agent identification
- Evidence types

## Session 3: Axes Deep Dive (2 hours)
- Organic axis + Heart domains
- Situational axis
- Systemic, Spatial, Temporal

## Session 4: Practice (2 hours)
- Annotate 10 examples together
- Discuss disagreements
- Q&A
EOF

git add docs/training_plan.md
git commit -m "docs: add annotator training plan"
git push origin main
```

## 4.2 Week 4: Double Annotation

### Task 2.2.1: Annotation Assignment
```
Micro-Pilot: 100 spans
- Annotator 1: Spans 1-100
- Annotator 2: Spans 1-100
- You (lead): Spans 1-100

All three annotate the same 100 spans independently.
```

### Task 2.2.2: Create IAA Calculation Script
```bash
cat > src/validation/calculate_iaa.py << 'EOF'
#!/usr/bin/env python3
"""Calculate Inter-Annotator Agreement (IAA) metrics."""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False

def load_annotations(filepath: str) -> dict:
    """Load annotations and index by span ID."""
    with open(filepath) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        annotations = data
    elif "annotations" in data:
        annotations = data["annotations"]
    else:
        annotations = [data]
    
    return {a.get("id", f"span_{i}"): a for i, a in enumerate(annotations)}

def extract_labels(annotations: dict, field_path: str) -> dict:
    """Extract specific labels from annotations."""
    labels = {}
    for span_id, annotation in annotations.items():
        value = annotation
        for key in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        labels[span_id] = value
    return labels

def calculate_cohen_kappa(labels1: dict, labels2: dict) -> float:
    """Calculate Cohen's Kappa between two annotators."""
    common_ids = set(labels1.keys()) & set(labels2.keys())
    
    if not common_ids:
        return None
    
    y1 = [labels1[id] for id in sorted(common_ids)]
    y2 = [labels2[id] for id in sorted(common_ids)]
    
    # Filter out None values
    valid = [(a, b) for a, b in zip(y1, y2) if a is not None and b is not None]
    if not valid:
        return None
    
    y1_valid, y2_valid = zip(*valid)
    
    try:
        return cohen_kappa_score(y1_valid, y2_valid)
    except:
        return None

def calculate_percent_agreement(labels1: dict, labels2: dict) -> float:
    """Calculate simple percent agreement."""
    common_ids = set(labels1.keys()) & set(labels2.keys())
    
    if not common_ids:
        return None
    
    agreements = 0
    total = 0
    
    for id in common_ids:
        v1, v2 = labels1[id], labels2[id]
        if v1 is not None and v2 is not None:
            total += 1
            if v1 == v2:
                agreements += 1
    
    return agreements / total if total > 0 else None

def calculate_jaccard(labels1: dict, labels2: dict) -> float:
    """Calculate Jaccard similarity for multi-label fields."""
    common_ids = set(labels1.keys()) & set(labels2.keys())
    
    if not common_ids:
        return None
    
    scores = []
    for id in common_ids:
        set1 = set(labels1[id] or [])
        set2 = set(labels2[id] or [])
        
        if not set1 and not set2:
            continue
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        scores.append(intersection / union if union > 0 else 0)
    
    return np.mean(scores) if scores else None

def main(file1: str, file2: str, file3: str = None):
    """Calculate IAA between annotators."""
    ann1 = load_annotations(file1)
    ann2 = load_annotations(file2)
    ann3 = load_annotations(file3) if file3 else None
    
    fields = [
        ("agent.type", "kappa"),
        ("behavior.form", "kappa"),
        ("behavior.concepts", "jaccard"),
        ("normative_textual.speech_mode", "kappa"),
        ("normative_textual.evaluation", "kappa"),
        ("normative_textual.quran_deontic_signal", "kappa"),
    ]
    
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 60)
    
    for field, method in fields:
        labels1 = extract_labels(ann1, field)
        labels2 = extract_labels(ann2, field)
        
        if method == "kappa":
            score_12 = calculate_cohen_kappa(labels1, labels2)
            pct_12 = calculate_percent_agreement(labels1, labels2)
        else:
            score_12 = calculate_jaccard(labels1, labels2)
            pct_12 = score_12
        
        print(f"\n{field}:")
        print(f"  Ann1 vs Ann2: {method}={score_12:.3f if score_12 else 'N/A'}, agreement={pct_12:.1%if pct_12 else 'N/A'}")
        
        if ann3:
            labels3 = extract_labels(ann3, field)
            if method == "kappa":
                score_13 = calculate_cohen_kappa(labels1, labels3)
                score_23 = calculate_cohen_kappa(labels2, labels3)
            else:
                score_13 = calculate_jaccard(labels1, labels3)
                score_23 = calculate_jaccard(labels2, labels3)
            
            print(f"  Ann1 vs Ann3: {method}={score_13:.3f if score_13 else 'N/A'}")
            print(f"  Ann2 vs Ann3: {method}={score_23:.3f if score_23 else 'N/A'}")
            
            if score_12 and score_13 and score_23:
                avg = (score_12 + score_13 + score_23) / 3
                print(f"  Average: {avg:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calculate_iaa.py <ann1.json> <ann2.json> [ann3.json]")
        sys.exit(1)
    
    main(*sys.argv[1:])
EOF

git add src/validation/
git commit -m "feat: add IAA calculation script"
git push origin main
```

### Task 2.2.3: Run IAA Analysis
```bash
# After all annotators complete their work
python src/validation/calculate_iaa.py \
    data/annotations/annotator1.json \
    data/annotations/annotator2.json \
    data/annotations/lead.json > reports/iaa_micropilot.txt

git add reports/
git commit -m "data: add micro-pilot IAA results"
git push origin main
```

## 4.3 IAA Targets for Micro-Pilot

| Metric | Target | Action if Not Met |
|--------|--------|-------------------|
| Agent type κ | ≥ 0.70 | Revise agent rules |
| Behavior concepts Jaccard | ≥ 0.60 | Review taxonomy |
| Speech mode κ | ≥ 0.75 | Add more examples |
| Deontic signal κ | ≥ 0.70 | Clarify flowchart |

## 4.4 Phase 2 Completion

### Task 2.4.1: Adjudicate Disagreements
```bash
cat > src/scripts/find_disagreements.py << 'EOF'
#!/usr/bin/env python3
"""Find spans where annotators disagreed."""

import json
import sys

def find_disagreements(files: list) -> list:
    """Find all spans where any two annotators disagree."""
    annotations = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                annotations.append({a["id"]: a for a in data})
            else:
                annotations.append({a["id"]: a for a in data.get("annotations", [data])})
    
    disagreements = []
    all_ids = set()
    for ann in annotations:
        all_ids.update(ann.keys())
    
    for span_id in sorted(all_ids):
        values = []
        for ann in annotations:
            if span_id in ann:
                values.append(ann[span_id].get("agent", {}).get("type"))
        
        if len(set(v for v in values if v)) > 1:
            disagreements.append({
                "id": span_id,
                "field": "agent.type",
                "values": values
            })
    
    return disagreements

if __name__ == "__main__":
    disagreements = find_disagreements(sys.argv[1:])
    for d in disagreements:
        print(f"{d['id']}: {d['field']} = {d['values']}")
EOF

git add src/scripts/
git commit -m "feat: add disagreement finder script"
git push origin main
```

### Task 2.4.2: Create Adjudication Meeting Notes
```bash
mkdir -p docs/adjudication
cat > docs/adjudication/micropilot_session1.md << 'EOF'
# Adjudication Session 1 - Micro-Pilot

## Date: [DATE]
## Attendees: [Names]

## Disagreements Reviewed

### Span QBM_00001 (2:3)
- **Field:** agent.type
- **Ann1:** AGT_BELIEVER
- **Ann2:** AGT_HUMAN_GENERAL
- **Decision:** AGT_BELIEVER
- **Rationale:** الذين يؤمنون explicitly identifies believers

### Span QBM_00005 (2:8)
- **Field:** behavior.concepts
- **Ann1:** [BEH_SPI_HYPOCRISY]
- **Ann2:** [BEH_SPEECH_LYING, BEH_SPI_HYPOCRISY]
- **Decision:** [BEH_SPEECH_LYING, BEH_SPI_HYPOCRISY]
- **Rationale:** Both lying (claim of faith) and hypocrisy present

## Manual Updates Needed
1. Add clarification about يقول + negative claim pattern
2. [Additional updates]
EOF

git add docs/adjudication/
git commit -m "docs: add adjudication session template"
git push origin main
```

### Task 2.4.3: Final Micro-Pilot Commit
```bash
git add .
git commit -m "milestone: complete Phase 2 - micro-pilot (100 spans, IAA measured)"
git tag -a v0.3.0 -m "Phase 2 Complete: Micro-Pilot"
git push origin main --tags
```

---

# 5. PHASE 3: FULL PILOT (Weeks 9-16)

## 5.1 Week 5: Scale to 500 Spans

### Task 3.1.1: Expand Pilot Selection
```bash
cat > src/scripts/expand_pilot.py << 'EOF'
#!/usr/bin/env python3
"""Expand pilot selection to 500 spans."""

import json
import random
from pathlib import Path

def expand_pilot():
    # Load full Quran
    with open("data/raw/quran_tokenized_full.json") as f:
        quran = json.load(f)
    
    # Priority surahs for behavioral content
    priority_surahs = [
        2,   # Al-Baqarah - comprehensive
        3,   # Aal-Imran - faith/battle
        4,   # An-Nisa - social/family
        5,   # Al-Ma'idah - legal
        6,   # Al-An'am - tawheed
        7,   # Al-A'raf - stories
        9,   # At-Tawbah - hypocrisy
        16,  # An-Nahl - gratitude
        17,  # Al-Isra - ethics
        24,  # An-Nur - social
        49,  # Al-Hujurat - conduct
        63,  # Al-Munafiqun - hypocrisy
        107, # Al-Ma'un - small acts
    ]
    
    selections = []
    
    for surah in quran["surahs"]:
        if surah["surah_number"] in priority_surahs:
            # Select more from priority surahs
            n_select = min(50, len(surah["ayat"]))
        else:
            # Sample from others
            n_select = min(5, len(surah["ayat"]))
        
        sampled = random.sample(surah["ayat"], n_select)
        for ayah in sampled:
            selections.append({
                "reference": f"{surah['surah_number']}:{ayah['ayah_number']}",
                "surah_number": surah["surah_number"],
                "surah_name": surah["name_ar"],
                "ayah_number": ayah["ayah_number"],
                "text_ar": ayah["text_ar"],
                "tokens": ayah["tokens"]
            })
    
    # Trim to 500
    if len(selections) > 500:
        selections = random.sample(selections, 500)
    
    output = {
        "metadata": {
            "subset": "pilot_500",
            "total_spans": len(selections)
        },
        "selections": sorted(selections, key=lambda x: (x["surah_number"], x["ayah_number"]))
    }
    
    with open("data/processed/pilot_500.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Created pilot with {len(selections)} spans")

if __name__ == "__main__":
    expand_pilot()
EOF

python src/scripts/expand_pilot.py

git add data/processed/pilot_500.json src/scripts/
git commit -m "data: expand pilot to 500 spans"
git push origin main
```

### Task 3.1.2: Divide Work
```markdown
## Pilot Work Distribution

| Annotator | Spans | Overlap |
|-----------|-------|---------|
| Annotator 1 | 1-200 | 1-50 (shared) |
| Annotator 2 | 101-300 | 101-150 (shared with A1), 251-300 (shared with A3) |
| Annotator 3 | 251-450 | 251-300 (shared) |
| Lead | 401-500 + adjudication | 401-450 (shared with A3) |

Total: 500 spans
Double-annotated: ~200 spans (40%)
```

## 5.2 Weeks 6-7: Annotation Sprint

### Task 3.2.1: Weekly Progress Tracking
```bash
cat > src/scripts/progress_report.py << 'EOF'
#!/usr/bin/env python3
"""Generate progress report from Label Studio export."""

import json
import sys
from datetime import datetime
from collections import defaultdict

def generate_report(filepath: str):
    with open(filepath) as f:
        data = json.load(f)
    
    stats = {
        "total": len(data),
        "by_status": defaultdict(int),
        "by_annotator": defaultdict(int),
        "by_surah": defaultdict(int)
    }
    
    for item in data:
        status = item.get("review", {}).get("status", "draft")
        stats["by_status"][status] += 1
        
        annotator = item.get("review", {}).get("annotator_id", "unknown")
        stats["by_annotator"][annotator] += 1
        
        surah = item.get("reference", {}).get("surah", 0)
        stats["by_surah"][surah] += 1
    
    print(f"Progress Report - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 40)
    print(f"Total spans: {stats['total']}")
    print(f"\nBy Status:")
    for status, count in stats["by_status"].items():
        print(f"  {status}: {count} ({count/stats['total']*100:.1f}%)")
    print(f"\nBy Annotator:")
    for ann, count in stats["by_annotator"].items():
        print(f"  {ann}: {count}")

if __name__ == "__main__":
    generate_report(sys.argv[1])
EOF

git add src/scripts/
git commit -m "feat: add progress report script"
git push origin main
```

### Task 3.2.2: Weekly Check-ins
```markdown
## Weekly Check-in Template

### Week X Progress
- Spans completed: X/500
- IAA current: X%
- Blockers: [List]
- Questions: [List]

### Action Items
1. [Item]
2. [Item]
```

## 5.3 Week 8: Review and Export

### Task 3.3.1: Create Export Script
```bash
cat > src/scripts/export_gold.py << 'EOF'
#!/usr/bin/env python3
"""Export Gold/Silver/Research datasets."""

import json
import sys
from datetime import datetime
from pathlib import Path

def export_datasets(input_file: str, output_dir: str):
    with open(input_file) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        annotations = data
    else:
        annotations = data.get("annotations", [])
    
    gold = []
    silver = []
    research = []
    
    for ann in annotations:
        status = ann.get("review", {}).get("status", "draft")
        
        # Calculate average confidence
        confidences = [a.get("confidence", 0.5) for a in ann.get("assertions", [])]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        if status == "approved":
            gold.append(ann)
            silver.append(ann)
            research.append(ann)
        elif status == "draft" and avg_confidence >= 0.75:
            silver.append(ann)
            research.append(ann)
        else:
            research.append(ann)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Export Gold
    gold_file = output_path / f"qbm_gold_{timestamp}.json"
    with open(gold_file, "w") as f:
        json.dump({
            "metadata": {
                "tier": "gold",
                "exported": timestamp,
                "total_spans": len(gold),
                "criteria": "reviewer_approved"
            },
            "spans": gold
        }, f, ensure_ascii=False, indent=2)
    
    # Export Silver
    silver_file = output_path / f"qbm_silver_{timestamp}.json"
    with open(silver_file, "w") as f:
        json.dump({
            "metadata": {
                "tier": "silver",
                "exported": timestamp,
                "total_spans": len(silver),
                "criteria": "approved OR (draft AND confidence >= 0.75)"
            },
            "spans": silver
        }, f, ensure_ascii=False, indent=2)
    
    # Export Research
    research_file = output_path / f"qbm_research_{timestamp}.json"
    with open(research_file, "w") as f:
        json.dump({
            "metadata": {
                "tier": "research",
                "exported": timestamp,
                "total_spans": len(research),
                "criteria": "all_annotations"
            },
            "spans": research
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Exported:")
    print(f"  Gold: {len(gold)} spans → {gold_file}")
    print(f"  Silver: {len(silver)} spans → {silver_file}")
    print(f"  Research: {len(research)} spans → {research_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_gold.py <input.json> <output_dir>")
        sys.exit(1)
    
    export_datasets(sys.argv[1], sys.argv[2])
EOF

git add src/scripts/
git commit -m "feat: add Gold/Silver/Research export script"
git push origin main
```

### Task 3.3.2: Run Full Validation
```bash
# Run all validations
python src/validation/validate_schema.py data/exports/qbm_gold_*.json
python src/validation/validate_vocabularies.py data/exports/qbm_gold_*.json
python src/validation/calculate_iaa.py data/annotations/*.json

git add data/exports/ reports/
git commit -m "data: pilot v0.1 export (500 spans)"
git push origin main
```

### Task 3.3.3: Phase 3 Completion
```bash
git add .
git commit -m "milestone: complete Phase 3 - full pilot (500 spans)"
git tag -a v0.4.0 -m "Phase 3 Complete: Full Pilot - Gold v0.1"
git push origin main --tags
```

---

# 6. PHASE 4: TAFSIR INTEGRATION (Weeks 17-24)

> **Reference:** See `QBM_END_TO_END_UPDATE.md` for complete tafsir scripts and database schema.

## 6.1 Weeks 17-18: Tafsir Data Acquisition

### Task 4.1.1: Download Tafsir Sources
```bash
# Create tafsir directory
mkdir -p data/tafsir

# Download from Quran.com API
python src/scripts/download_tafsir.py \
    --sources quran_api \
    --output data/tafsir/

# Download priority tafsir individually
python src/scripts/download_tafsir.py --tafsir ibn_kathir_ar --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir tabari --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir qurtubi --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir saadi --output data/tafsir/
python src/scripts/download_tafsir.py --tafsir jalalayn_ar --output data/tafsir/

git add data/tafsir/
git commit -m "data: add tafsir sources (ibn kathir, tabari, qurtubi, saadi, jalalayn)"
git push origin main
```

### Task 4.1.2: Verify Tafsir Coverage
```bash
# Check download completeness
python -c "
import json
from pathlib import Path

tafsir_dir = Path('data/tafsir')
for f in tafsir_dir.glob('*.json'):
    if f.name != 'tafsir_index.json':
        data = json.load(open(f))
        ayat_count = len(data.get('ayat', {}))
        print(f'{f.name}: {ayat_count} ayat')
"
```

## 6.2 Weeks 19-20: Database Setup

### Task 4.2.1: Initialize Tafsir Database
```bash
# Set up PostgreSQL (if not already done)
createdb qbm

# Run tafsir schema setup
python src/scripts/setup_tafsir_db.py \
    --host localhost \
    --db qbm \
    --user postgres \
    --password YOUR_PASSWORD

# Load tafsir data into database
for tafsir in ibn_kathir_ar tabari qurtubi saadi jalalayn_ar; do
    python src/scripts/setup_tafsir_db.py \
        --load data/tafsir/${tafsir}.json \
        --source-id TAFSIR_${tafsir^^}
done
```

### Task 4.2.2: Verify Database
```sql
-- Check tafsir coverage
SELECT * FROM tafsir_coverage_summary;

-- Verify all 6,236 ayat have tafsir
SELECT source_id, COUNT(*) as ayat_count 
FROM tafsir_content 
GROUP BY source_id;
```

## 6.3 Weeks 21-22: Tafsir Lookup Tool

### Task 4.3.1: Test Lookup Tool
```bash
# Test single ayah lookup
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255

# Test range lookup
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --end-ayah 257

# Test comparison mode
python src/scripts/tafsir_lookup.py --surah 2 --ayah 255 --compare --json

# Test search
python src/scripts/tafsir_lookup.py --search "الكرسي"
```

### Task 4.3.2: Integrate with Label Studio
```json
// Add to Label Studio annotation interface
// config/label_studio_config.json - add tafsir panel
{
    "panels": {
        "tafsir": {
            "enabled": true,
            "position": "right",
            "sources": ["ibn_kathir_ar", "tabari", "qurtubi"],
            "auto_load": true
        }
    }
}
```

## 6.4 Weeks 23-24: Annotator Training on Tafsir Usage

### Task 4.4.1: Create Tafsir Consultation Protocol
```markdown
## Tafsir Consultation Protocol

### When to Consult Tafsir
1. **Agent identification** - unclear who is addressed
2. **Behavior classification** - ambiguous action type
3. **Context determination** - need سبب نزول
4. **Disputed interpretations** - multiple valid readings

### Consultation Hierarchy
1. Ibn Kathir (primary - most comprehensive)
2. Al-Tabari (linguistic depth)
3. Al-Qurtubi (fiqh implications)
4. Al-Sa'di (modern clarity)

### Documentation Requirements
- Record which tafsir was consulted
- Note if tafsir influenced decision
- Flag disagreements between sources
```

### Task 4.4.2: Phase 4 Completion
```bash
# Verify Phase 4 completion checklist
# - [ ] 5+ tafsir sources downloaded
# - [ ] Database populated with all 6,236 ayat
# - [ ] Lookup tool functional
# - [ ] Annotators trained on tafsir protocol
# - [ ] Label Studio integrated

git add .
git commit -m "milestone: complete Phase 4 - tafsir integration"
git tag -a v0.4.0 -m "Phase 4 Complete - Tafsir Integration"
git push origin main --tags
```

---

# 7. PHASE 5: SCALE-UP (Weeks 25-40)

## 7.1 Weeks 25-28: Expand Team

### Task 5.1.1: Recruit Additional Annotators
- Target: 6-9 total annotators
- Train new annotators using refined manual + tafsir protocol
- Pair new annotators with experienced ones
- Assign tafsir consultant for edge cases

### Task 5.1.2: Set Up Production Database
```bash
cat > src/scripts/setup_database.py << 'EOF'
#!/usr/bin/env python3
"""Set up PostgreSQL database for production."""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

Base = declarative_base()

class Span(Base):
    __tablename__ = 'spans'
    
    id = Column(String, primary_key=True)
    surah = Column(Integer, nullable=False)
    ayah = Column(Integer, nullable=False)
    token_start = Column(Integer)
    token_end = Column(Integer)
    raw_text_ar = Column(String, nullable=False)
    behavior_form = Column(String)
    behavior_concepts = Column(JSON)
    agent_type = Column(String)
    thematic_constructs = Column(JSON)
    normative_textual = Column(JSON)
    periodicity = Column(JSON)
    status = Column(String, default='draft')
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    assertions = relationship("Assertion", back_populates="span")
    reviews = relationship("Review", back_populates="span")

class Assertion(Base):
    __tablename__ = 'assertions'
    
    id = Column(String, primary_key=True)
    span_id = Column(String, ForeignKey('spans.id'))
    axis = Column(String, nullable=False)
    value = Column(String, nullable=False)
    support_type = Column(String)
    confidence = Column(Float)
    negated = Column(Boolean, default=False)
    justification_code = Column(String)
    justification = Column(String)
    
    span = relationship("Span", back_populates="assertions")

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    span_id = Column(String, ForeignKey('spans.id'))
    annotator_id = Column(String)
    reviewer_id = Column(String)
    status = Column(String)
    notes = Column(String)
    created_at = Column(DateTime)
    
    span = relationship("Span", back_populates="reviews")

def setup_database():
    db_url = os.environ.get('DATABASE_URL', 'postgresql://localhost/qbm')
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print(f"Database tables created at {db_url}")

if __name__ == "__main__":
    setup_database()
EOF

git add src/scripts/
git commit -m "feat: add database setup script"
git push origin main
```

## 7.2 Weeks 29-36: Bulk Annotation

### Task 5.2.1: Weekly Targets
```markdown
## Weekly Annotation Targets (Scale-Up Phase)

| Week Block | Surahs | Ayat Target | Cumulative Ayat | Spans Target |
|------------|--------|-------------|-----------------|-------------|
| 29-32 | 1-14 | 1,500 | 1,500 | 4,500 |
| 33-36 | 15-28 | 1,500 | 3,000 | 9,000 |
| 37-40 | Review + gaps | - | 3,000 | 10,000 |
```

### Task 5.2.2: Automated Quality Checks
```bash
cat > src/scripts/quality_check.py << 'EOF'
#!/usr/bin/env python3
"""Run automated quality checks on annotations."""

import json
import sys

def check_quality(filepath: str) -> dict:
    with open(filepath) as f:
        data = json.load(f)
    
    issues = []
    
    for span in data if isinstance(data, list) else data.get("spans", []):
        span_id = span.get("id", "unknown")
        
        # Check: Agent must be present
        if not span.get("agent", {}).get("type"):
            issues.append(f"{span_id}: Missing agent type")
        
        # Check: At least one assertion
        if not span.get("assertions"):
            issues.append(f"{span_id}: No assertions")
        
        # Check: Confidence scores in range
        for i, a in enumerate(span.get("assertions", [])):
            conf = a.get("confidence")
            if conf is not None and (conf < 0 or conf > 1):
                issues.append(f"{span_id}: Assertion {i} has invalid confidence {conf}")
        
        # Check: Normative layer present
        if not span.get("normative_textual", {}).get("speech_mode"):
            issues.append(f"{span_id}: Missing speech_mode")
        
        # Check: Heart domains if ORG_HEART
        for a in span.get("assertions", []):
            if a.get("axis") == "AX_ORGAN" and a.get("value") == "ORG_HEART":
                if not a.get("organ_semantic_domains"):
                    issues.append(f"{span_id}: ORG_HEART without semantic domains")
    
    return {
        "total_spans": len(data) if isinstance(data, list) else len(data.get("spans", [])),
        "issues_found": len(issues),
        "issues": issues
    }

if __name__ == "__main__":
    results = check_quality(sys.argv[1])
    print(json.dumps(results, indent=2))
    if results["issues_found"] > 0:
        sys.exit(1)
EOF

git add src/scripts/
git commit -m "feat: add automated quality check script"
git push origin main
```

## 7.3 Weeks 37-40: Review Sprint + Coverage Audit

### Task 5.3.1: Senior Review Process
```markdown
## Review Protocol

1. Each span reviewed by senior reviewer
2. Status updated: draft → approved OR disputed
3. Disputed spans go to adjudication committee
4. Track review metrics:
   - Spans/hour
   - Approval rate
   - Common issues
```

### Task 5.3.2: Coverage Audit
```bash
# Run coverage audit
python src/scripts/coverage_audit.py \
    --annotations data/annotations/ \
    --output reports/coverage/ \
    --create-batches

# Review coverage report
cat reports/coverage/coverage_report_*.json | jq '.summary'
```

### Task 5.3.3: Phase 5 Completion
```bash
git add .
git commit -m "milestone: complete Phase 5 - scale-up (3000 ayat, 10000 spans)"
git tag -a v0.5.0 -m "Phase 5 Complete: Scale-Up"
git push origin main --tags
```

---

# 8. PHASE 6: FULL QURAN COVERAGE (Weeks 41-60)

## 8.1 Coverage Strategy

### Target: 6,236 ayat (100%)

| Week Block | Surahs | Ayat Target | Cumulative |
|------------|--------|-------------|------------|
| 41-45 | 29-50 | 1,200 | 4,200 |
| 46-50 | 51-80 | 1,100 | 5,300 |
| 51-55 | 81-100 | 500 | 5,800 |
| 56-60 | 101-114 + gaps | 436 | 6,236 |

## 8.2 Weeks 41-45: Surahs 29-50

### Task 6.1.1: Prepare Batches
```bash
# Generate batches for surahs 29-50
python src/scripts/select_full_quran.py \
    --surahs 29-50 \
    --output data/processed/batches_week41-45/

# Distribute to annotators
python src/scripts/distribute_batches.py \
    --input data/processed/batches_week41-45/ \
    --annotators 6 \
    --overlap 0.2
```

### Task 6.1.2: Weekly Quality Gates
```markdown
## Weekly Quality Check
- [ ] IAA ≥ 0.72 maintained
- [ ] Tafsir consultation logged
- [ ] All spans validated against schema
- [ ] Coverage tracker updated
```

## 8.3 Weeks 46-50: Surahs 51-80

### Task 6.2.1: Continue Annotation
```bash
# Similar process for surahs 51-80
python src/scripts/select_full_quran.py \
    --surahs 51-80 \
    --output data/processed/batches_week46-50/
```

## 8.4 Weeks 51-55: Surahs 81-100

### Task 6.3.1: Short Surahs Strategy
```markdown
## Short Surah Annotation Notes
- Many short surahs have dense behavioral content
- May require multiple spans per ayah
- Focus on eschatological themes (akhira behaviors)
```

## 8.5 Weeks 56-60: Final Coverage + Gap Filling

### Task 6.4.1: Coverage Audit
```bash
# Check for gaps
python src/scripts/coverage_audit.py \
    --annotations data/annotations/ \
    --output reports/coverage/ \
    --create-batches

# Fill gaps
python src/scripts/distribute_batches.py \
    --input reports/coverage/gap_batches/ \
    --annotators 6
```

### Task 6.4.2: Final Coverage Verification
```bash
# Verify 100% coverage
python -c "
from src.scripts.select_full_quran import TOTAL_AYAT
import json

with open('data/processed/full_quran_batches/coverage_tracker.json') as f:
    tracker = json.load(f)

annotated = sum(s['annotated'] for s in tracker['surahs'].values())
print(f'Annotated: {annotated}/{TOTAL_AYAT} ({100*annotated/TOTAL_AYAT:.1f}%)')
"
```

### Task 6.4.3: Phase 6 Completion
```bash
git add data/annotations/
git commit -m "milestone: 100% Quran coverage achieved (6,236 ayat)"
git tag -a v0.6.0 -m "Phase 6 Complete - Full Quran Coverage"
git push origin main --tags
```

---

# 9. PHASE 7: PRODUCTION RELEASE (Weeks 61-70)

## 9.1 Weeks 61-64: API Development

### Task 7.1.1: Create FastAPI Backend
```bash
cat > src/api/main.py << 'EOF'
#!/usr/bin/env python3
"""QBM REST API."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path

app = FastAPI(
    title="Quranic Behavior Matrix API",
    description="API for accessing QBM dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load data (in production, use database)
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "exports"

def load_dataset(tier: str = "gold"):
    files = list(DATA_PATH.glob(f"qbm_{tier}_*.json"))
    if not files:
        return {"spans": []}
    latest = sorted(files)[-1]
    with open(latest) as f:
        return json.load(f)

class SpanResponse(BaseModel):
    id: str
    reference: dict
    raw_text_ar: str
    behavior: Optional[dict]
    agent: dict
    assertions: List[dict]
    normative_textual: dict

@app.get("/")
def root():
    return {"message": "QBM API v1.0", "docs": "/docs"}

@app.get("/spans", response_model=List[SpanResponse])
def get_spans(
    tier: str = Query("gold", enum=["gold", "silver", "research"]),
    surah: Optional[int] = None,
    agent: Optional[str] = None,
    behavior: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """Get spans with optional filters."""
    data = load_dataset(tier)
    spans = data.get("spans", [])
    
    if surah:
        spans = [s for s in spans if s.get("reference", {}).get("surah") == surah]
    
    if agent:
        spans = [s for s in spans if s.get("agent", {}).get("type") == agent]
    
    if behavior:
        spans = [s for s in spans if behavior in s.get("behavior", {}).get("concepts", [])]
    
    return spans[:limit]

@app.get("/spans/{span_id}")
def get_span(span_id: str, tier: str = "gold"):
    """Get a specific span by ID."""
    data = load_dataset(tier)
    for span in data.get("spans", []):
        if span.get("id") == span_id:
            return span
    raise HTTPException(status_code=404, detail="Span not found")

@app.get("/stats")
def get_stats(tier: str = "gold"):
    """Get dataset statistics."""
    data = load_dataset(tier)
    spans = data.get("spans", [])
    
    return {
        "tier": tier,
        "total_spans": len(spans),
        "by_agent": {},  # TODO: compute
        "by_surah": {},  # TODO: compute
        "metadata": data.get("metadata", {})
    }

@app.get("/vocabularies")
def get_vocabularies():
    """Get controlled vocabularies."""
    vocab_path = Path(__file__).parent.parent.parent / "config" / "controlled_vocabularies_v1.json"
    with open(vocab_path) as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

git add src/api/
git commit -m "feat: add REST API"
git push origin main
```

## 9.2 Weeks 65-66: Documentation

### Task 7.2.1: API Documentation
```bash
cat > docs/API.md << 'EOF'
# QBM API Documentation

## Base URL
```
https://api.qbm.example.com/v1
```

## Endpoints

### GET /spans
Get spans with optional filters.

**Parameters:**
- `tier` (string): gold, silver, research (default: gold)
- `surah` (int): Filter by surah number
- `agent` (string): Filter by agent type (e.g., AGT_BELIEVER)
- `behavior` (string): Filter by behavior concept (e.g., BEH_SPI_FAITH)
- `limit` (int): Max results (default: 100, max: 1000)

**Example:**
```bash
curl "https://api.qbm.example.com/v1/spans?surah=2&agent=AGT_BELIEVER&limit=10"
```

### GET /spans/{id}
Get a specific span by ID.

### GET /stats
Get dataset statistics.

### GET /vocabularies
Get controlled vocabularies.
EOF

git add docs/
git commit -m "docs: add API documentation"
git push origin main
```

## 9.3 Weeks 67-68: Testing

### Task 7.3.1: Create Test Suite
```bash
cat > tests/test_api.py << 'EOF'
"""API tests."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "QBM API" in response.json()["message"]

def test_get_spans():
    response = client.get("/spans?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_vocabularies():
    response = client.get("/vocabularies")
    assert response.status_code == 200
    assert "axes" in response.json()

def test_invalid_tier():
    response = client.get("/spans?tier=invalid")
    assert response.status_code == 422
EOF

cat > tests/test_validation.py << 'EOF'
"""Validation tests."""

import pytest
import json
from pathlib import Path
from src.validation.validate_schema import validate_span, load_schema

def test_valid_span():
    schema = load_schema()
    span = {
        "id": "QBM_00001",
        "reference": {"surah": 2, "ayah": 3},
        "span": {
            "token_start": 0,
            "token_end": 5,
            "raw_text_ar": "test"
        },
        "agent": {"type": "AGT_BELIEVER"},
        "assertions": [],
        "normative_textual": {},
        "review": {"status": "draft"}
    }
    errors = validate_span(span, schema)
    assert len(errors) == 0

def test_invalid_span_missing_agent():
    schema = load_schema()
    span = {
        "id": "QBM_00001",
        "reference": {"surah": 2, "ayah": 3},
        "span": {"token_start": 0, "token_end": 5, "raw_text_ar": "test"},
        "assertions": [],
        "normative_textual": {},
        "review": {"status": "draft"}
    }
    errors = validate_span(span, schema)
    assert len(errors) > 0
EOF

# Run tests
pytest tests/ -v

git add tests/
git commit -m "test: add API and validation tests"
git push origin main
```

## 9.4 Weeks 69-70: Launch

### Task 7.4.1: Final Release
```bash
# Run full test suite
pytest tests/ -v --cov=src --cov-report=html

# Validate all exports
python src/validation/validate_schema.py data/exports/qbm_gold_*.json
python src/validation/validate_vocabularies.py data/exports/qbm_gold_*.json

# Generate final statistics
python src/scripts/progress_report.py data/exports/qbm_gold_*.json > reports/final_stats.txt

# Tag release
git add .
git commit -m "milestone: production release v1.0.0"
git tag -a v1.0.0 -m "Production Release v1.0.0 - QBM Gold Dataset"
git push origin main --tags
```

### Task 7.4.2: Create Release Notes
```bash
cat > CHANGELOG.md << 'EOF'
# Changelog

## [1.0.0] - 2025-XX-XX

### Added
- Gold dataset: X,XXX spans
- Silver dataset: X,XXX spans
- Research dataset: XX,XXX spans
- REST API
- Validation tools
- Export scripts

### Dataset Coverage
- 114 surahs represented
- XX behavior concepts used
- Average IAA: 0.XX (Cohen's κ)

### Contributors
- Dr. Ibrahim Bouzidani (Foundational Research)
- Salim Al-Barami (Project Lead)
- [Annotators]
- [Reviewers]
EOF

git add CHANGELOG.md
git commit -m "docs: add changelog for v1.0.0"
git push origin main
```

---

# 10. PHASE 8: PUBLICATION & LAUNCH (Weeks 71-78)

## 10.1 Weeks 71-74: Academic Papers

### Task 8.1.1: Methodology Paper
```markdown
## Paper 1: Methodology
**Title:** "QBM: A Multi-Dimensional Framework for Quranic Behavioral Classification"

### Outline
1. Introduction - gap in computational Islamic studies
2. Related Work - existing Quranic NLP, behavioral taxonomies
3. Framework Design - axes, vocabularies, annotation protocol
4. Pilot Study - IAA results, annotator feedback
5. Discussion - challenges, limitations
6. Conclusion - contributions, future work

### Target Venues
- ACL/EMNLP (NLP track)
- Digital Humanities conferences
- Islamic Studies journals
```

### Task 8.1.2: Dataset Paper
```markdown
## Paper 2: Dataset
**Title:** "The QBM Dataset: 20,000+ Annotated Behavioral Spans from the Complete Quran"

### Outline
1. Dataset Overview - statistics, coverage
2. Annotation Process - team, training, quality control
3. Data Format - schema, vocabularies
4. Analysis - distribution of behaviors, agents, axes
5. Use Cases - NLP, education, research
6. Access - API, download, license

### Target Venues
- LREC (Language Resources)
- NeurIPS Datasets Track
- ACL Resource Papers
```

## 10.2 Weeks 75-78: Public Launch

### Task 8.2.1: Public Repository
```bash
# Prepare public release
git checkout -b release/v1.0.0

# Update README for public
cat > README.md << 'EOF'
# Quranic Human-Behavior Classification Matrix (QBM)

The world's first comprehensive, academically rigorous dataset of Quranic behavioral classifications.

## Dataset Statistics
- **6,236 ayat** (100% Quran coverage)
- **20,000+ spans** annotated
- **80+ behavior concepts**
- **5 tafsir sources** integrated
- **IAA ≥ 0.75** (Cohen's κ)

## Quick Start
```bash
pip install qbm-api
from qbm import QBMClient
client = QBMClient()
spans = client.get_spans(surah=2, agent="AGT_BELIEVER")
```

## Documentation
- [API Reference](docs/API.md)
- [Dataset Schema](docs/SCHEMA.md)
- [Coding Manual](docs/CODING_MANUAL.md)

## Citation
```bibtex
@dataset{qbm2025,
  title={QBM: Quranic Human-Behavior Classification Matrix},
  author={Al-Barami, Salim and Bouzidani, Ibrahim},
  year={2025},
  publisher={GitHub},
  url={https://github.com/qbm-project/quranic-behavior-matrix}
}
```

## License
CC BY-NC-SA 4.0 (Research use)
EOF

git add .
git commit -m "docs: prepare public release v1.0.0"
git tag -a v1.0.0 -m "QBM v1.0.0 - Full Quran Coverage"
git push origin main --tags
```

### Task 8.2.2: Phase 8 Completion
```bash
# Final checklist
# - [ ] Methodology paper submitted
# - [ ] Dataset paper submitted
# - [ ] Public repository ready
# - [ ] API documentation complete
# - [ ] Press release drafted
# - [ ] Academic partners notified

git commit -m "milestone: complete Phase 8 - publication and launch"
git tag -a v1.0.0-final -m "Phase 8 Complete - QBM v1.0.0 Public Release"
git push origin main --tags
```

---

# 11. PHASE 9: MAINTENANCE & GROWTH (Ongoing)

## 11.1 Monthly Tasks

### Task 9.1.1: Quality Monitoring
```bash
# Monthly quality check
python src/validation/validate_schema.py data/exports/qbm_gold_*.json
python src/scripts/quality_check.py data/exports/qbm_gold_*.json
```

### Task 9.1.2: Vocabulary Updates
```markdown
## Vocabulary Update Process
1. Propose new term with justification
2. Review by committee
3. Add to controlled_vocabularies_v1.json
4. Update version number
5. Commit with clear message
```

## 11.2 Quarterly Tasks

### Task 9.2.1: Dataset Expansion
- Add 500-1000 new spans per quarter
- Focus on specialized domains (e.g., economic behaviors, family relations)
- Incorporate user feedback and corrections
- Update tafsir sources as new translations become available

### Task 9.2.2: Community Engagement
- Respond to GitHub issues and PRs
- Host quarterly community calls
- Maintain academic partnerships
- Present at conferences

---

# 12. GIT WORKFLOW

## 9.1 Branch Strategy

```
main                 # Production-ready code
├── develop          # Integration branch
├── feature/XXX      # New features
├── fix/XXX          # Bug fixes
└── release/vX.X.X   # Release preparation
```

## 9.2 Commit Message Format

```
type(scope): subject

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- data: Data changes
- test: Tests
- chore: Maintenance

Examples:
- feat(api): add span filtering by behavior concept
- data(annotations): complete surah 49 annotation
- docs(manual): add heart domain examples
- fix(validation): handle empty assertions array
```

## 9.3 Release Process

```bash
# 1. Create release branch
git checkout -b release/v1.0.0 develop

# 2. Update version numbers
# 3. Run full test suite
pytest tests/ -v

# 4. Merge to main
git checkout main
git merge release/v1.0.0

# 5. Tag release
git tag -a v1.0.0 -m "Release v1.0.0"

# 6. Push
git push origin main --tags

# 7. Merge back to develop
git checkout develop
git merge main
```

---

# 13. QUALITY GATES

## 13.1 Phase Gates

| Phase | Gate | Criteria |
|-------|------|----------|
| 0-1 | Setup Complete | All tools configured, repo ready |
| 2 | Calibration Pass | Personal accuracy ≥ 80% |
| 3 | Micro-Pilot Pass | IAA ≥ 0.65 (average), 100 spans |
| 4 | Pilot Pass | IAA ≥ 0.70, Gold ≥ 500 spans |
| **5** | **Tafsir Ready** | **5+ sources, DB live, lookup functional** |
| **6** | **Scale Pass** | **IAA ≥ 0.72, 3,000 ayat, 10,000 spans** |
| **7** | **Full Coverage** | **6,236 ayat (100%), 15,000+ spans** |
| **8** | **Production Pass** | **All tests pass, API live, docs complete** |
| **9** | **Publication Pass** | **2+ papers submitted, v1.0.0 released** |

## 13.2 Continuous Quality

| Check | Frequency | Tool |
|-------|-----------|------|
| Schema validation | Every commit | validate_schema.py |
| Vocabulary validation | Every commit | validate_vocabularies.py |
| IAA calculation | Weekly | calculate_iaa.py |
| Quality check | Weekly | quality_check.py |
| Full test suite | Before release | pytest |

---

# 14. RISK MANAGEMENT

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Annotator dropout | High | Medium | Cross-train, document everything, competitive compensation |
| Low IAA | High | Medium | More examples, frequent calibration, weekly check-ins |
| Scope creep | Medium | High | Strict phase gates, change control process |
| Data loss | Critical | Low | Daily backups, Git LFS, cloud redundancy |
| Interpretation disputes | Medium | High | Clear adjudication process, tafsir consultation |
| **Annotator burnout** | **High** | **Medium** | **Rotate batches, limit daily quota, breaks** |
| **IAA degradation at scale** | **High** | **Medium** | **Weekly calibration sessions, gold standard checks** |
| **Tafsir API rate limits** | **Medium** | **High** | **Implement caching, async downloads, local DB** |
| **Budget overrun** | **High** | **Medium** | **Phased funding, volunteer recruitment, partnerships** |
| **Database performance** | **Medium** | **Low** | **Index optimization, partitioning, caching** |

---

# 15. RESOURCE REQUIREMENTS

## 15.1 Human Resources

| Role | Count | Hours/Week | Duration | Notes |
|------|-------|------------|----------|-------|
| Project Lead | 1 | 20 | Full project | You (Salim) |
| Senior Annotator | 3 | 15 | Weeks 5-70 | Islamic studies background |
| Junior Annotator | 6 | 10 | Weeks 25-70 | Trained on coding manual |
| Tafsir Consultant | 1 | 5 | Weeks 17-70 | Scholar for edge cases |
| Reviewer | 2 | 10 | Weeks 9-70 | Final quality gate |
| Developer | 1 | 15 | Weeks 1-70 | API, pipeline, tools |

## 15.2 Infrastructure

| Item | Cost/Month | Duration | Total |
|------|------------|----------|-------|
| Cloud hosting (API) | $100 | 18 months | $1,800 |
| Database (PostgreSQL) | $50 | 18 months | $900 |
| Label Studio hosting | $50 | 18 months | $900 |
| Git LFS storage | $20 | 18 months | $360 |
| Backup storage | $30 | 18 months | $540 |
| **Infrastructure Total** | | | **$4,500** |

## 15.3 Personnel Costs

| Role | Rate/Hour | Hours | Total |
|------|-----------|-------|-------|
| Senior Annotators (3) | $25 | 3 × 15 × 66 weeks | $74,250 |
| Junior Annotators (6) | $15 | 6 × 10 × 46 weeks | $41,400 |
| Tafsir Consultant | $50 | 5 × 54 weeks | $13,500 |
| Reviewers (2) | $30 | 2 × 10 × 62 weeks | $37,200 |
| Developer | $40 | 15 × 70 weeks | $42,000 |
| **Personnel Total** | | | **$208,350** |

## 15.4 Total Budget Estimate (Full Project)

| Category | Amount (USD) |
|----------|--------------|
| Personnel | $208,350 |
| Infrastructure | $4,500 |
| Software/Tools | $3,000 |
| Contingency (15%) | $32,378 |
| **TOTAL** | **$248,228** |

### Budget Notes
- Can be reduced significantly with volunteer annotators
- Academic partnerships can offset costs
- Phased funding approach recommended
- **Minimum viable: $80,000** (reduced team, longer timeline, volunteer annotators)

---

# 16. SUCCESS METRICS

## 16.1 Quantitative

| Metric | Pilot Target | Scale Target | Production Target |
|--------|--------------|--------------|-------------------|
| Gold spans | 500 | 2,000 | **4,000+** |
| Silver spans | 1,000 | 5,000 | **10,000+** |
| Research spans | 2,500 | 10,000 | **20,000+** |
| **Ayat coverage** | 500 | 3,000 | **6,236 (100%)** |
| IAA (Cohen's κ) | ≥ 0.70 | ≥ 0.72 | **≥ 0.75** |
| Surahs | 20+ | 60+ | **114 (100%)** |
| Behavior concepts | 30+ | 50+ | **80+** |
| **Tafsir sources** | Manual | 3 | **5+** |
| **Tafsir consultations** | N/A | 50% | **80%** |

## 16.2 Qualitative

- [ ] Coding manual stable (no major revisions in 8 weeks)
- [ ] Annotator satisfaction (survey score ≥ 4/5)
- [ ] Academic endorsement (3+ scholars review)
- [ ] **Tafsir integration validated by Islamic studies faculty**
- [ ] Publication acceptance (2+ papers submitted)
- [ ] User adoption (API users, downloads)
- [ ] **Complete Quran behavioral map published**

## 16.3 Milestone Checkpoints

| Phase | Gate | Criteria |
|-------|------|----------|
| 0-1 | Setup Complete | Tools configured |
| 2-3 | Pilot Pass | IAA ≥ 0.70, 500 spans |
| **4** | **Tafsir Ready** | **5 sources, DB live** |
| **5** | **Scale Pass** | **3,000 ayat, 10,000 spans** |
| **6** | **Full Coverage** | **6,236 ayat (100%)** |
| **7** | **Production** | **API live, docs complete** |
| **8** | **Publication** | **2+ papers submitted** |

---

# APPENDIX: QUICK REFERENCE

## Git Commands Cheat Sheet

```bash
# Daily workflow
git pull origin main
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "feat: description"
git push origin feature/my-feature
# Create PR on GitHub

# Release
git tag -a v0.X.0 -m "Phase X Complete"
git push origin main --tags

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View history
git log --oneline -10
```

## Validation Commands

```bash
# Schema validation
python src/validation/validate_schema.py <file.json>

# Vocabulary validation
python src/validation/validate_vocabularies.py <file.json>

# IAA calculation
python src/validation/calculate_iaa.py <ann1.json> <ann2.json>

# Quality check
python src/scripts/quality_check.py <file.json>

# Export datasets
python src/scripts/export_gold.py <input.json> data/exports/
```

## Contact & Support

- Project Lead: [Your Email]
- Repository: https://github.com/YOUR_USERNAME/quranic-behavior-matrix
- Documentation: [URL]

---

*Document Version: 2.0*
*Last Updated: December 2025*
*Aligned with: QBM_END_TO_END_UPDATE.md v2.0*
