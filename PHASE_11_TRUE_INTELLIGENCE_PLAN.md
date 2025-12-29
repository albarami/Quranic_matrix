# Phase 11: HYBRID Intelligence Implementation Plan
# RAG + Frontier Models (REVISED Architecture)

## Key Insight
DON'T train a full LLM. GPT-5/Claude will ALWAYS reason better.
INSTEAD: Train the retrieval components (Layers 2-6), use frontier models for reasoning (Layer 7).

## Current State Assessment

### What Was Built (Mechanical - Level 1):
| Component | Status | Reality |
|-----------|--------|---------|
| 5 Tafsir Sources | ✅ | 31,180 entries (good data) |
| Unified Graph | ✅ | 43,698 nodes, 127,757 edges (storage only) |
| GPU Available | ✅ | CUDA available (but idle) |
| Behavioral Extraction | ❌ | Keyword matching (`if "كبر" in text`) |
| Embeddings | ❌ | Generic multilingual, not trained on QBM |
| Relationships | ❌ | Co-occurrence, not causal understanding |

### The Problem:
```python
# Current "intelligence":
if "كبر" in tafsir_text:
    mentions.append("الكبر")  # Catches أكبر, الكبير, كبرت - FALSE POSITIVES

# This is 1990s string matching, not 2025 AI
```

---

## The 7 Layers Implementation Plan

### Layer 1: Foundation Data ✅ COMPLETE
- 6,236 verses
- 15,847+ behavioral spans  
- 5 tafsir sources (31,180 entries)
- 87 behavior taxonomy
- 11 dimensional framework

---

## Layer 2: Arabic-First Embeddings (Week 1-2)

### Task 2.1: Setup Arabic Embedding Base
**File:** `src/ml/arabic_embeddings.py`

```python
# Use Arabic-specific models, not generic multilingual
BASE_MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv2",      # Best for MSA
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",  # Classical + MSA
    "marbert": "UBC-NLP/MARBERT",                     # Social media Arabic
}
```

### Task 2.2: Create Training Pairs from QBM Data
**Input:** Annotated behavioral spans
**Output:** Contrastive learning pairs

```python
# Positive pair: span text ↔ correct behavior definition
# Negative pair: span text ↔ wrong behavior definition
train_examples = [
    InputExample(texts=[span.text, behavior_definition], label=1.0),
    InputExample(texts=[span.text, wrong_behavior], label=0.0),
]
```

### Task 2.3: Fine-tune on GPU
**Hardware:** 8x A100 (use them!)
**Output:** `models/qbm-arabic-embeddings/`

### Tests for Layer 2:
```python
def test_embedding_quality():
    """الكبر should be closer to التكبر than to أكبر"""
    model = load_qbm_embeddings()
    
    kibr = model.encode("الكبر")  # Arrogance
    takabbur = model.encode("التكبر")  # Being arrogant
    akbar = model.encode("أكبر")  # Greater
    
    assert cosine_sim(kibr, takabbur) > cosine_sim(kibr, akbar)

def test_quranic_context():
    """Same word, different meaning based on context"""
    # الله أكبر (Allah is greater) ≠ استكبر (was arrogant)
    assert model.encode("الله أكبر") != model.encode("استكبر فرعون")
```

### Deliverables:
- [ ] `src/ml/arabic_embeddings.py` - Fine-tuning script
- [ ] `models/qbm-arabic-embeddings/` - Trained model
- [ ] `tests/test_embeddings.py` - Quality tests
- [ ] Embedding dimension: 768 (AraBERT)

---

## Layer 3: Behavioral Classifier (Week 3-4)

### Task 3.1: Replace Keyword Matching with ML
**Current (BAD):**
```python
if "كبر" in text:
    behavior = "الكبر"  # FALSE POSITIVES
```

**New (GOOD):**
```python
prediction = classifier.predict(text, context)
# Returns: {"behavior": "الكبر", "confidence": 0.94}
```

### Task 3.2: Multi-label Classification
**Input:** Verse text + surrounding context
**Output:** 87 behavior labels with confidence scores

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv2",
    num_labels=87,  # One for each behavior
    problem_type="multi_label_classification"
)
```

### Task 3.3: Training Data Preparation
```python
train_data = []
for span in annotated_spans:
    train_data.append({
        "text": f"{span.verse_text} [SEP] {span.context}",
        "labels": one_hot_encode(span.behaviors)  # Multi-label
    })
```

### Tests for Layer 3:
```python
def test_no_false_positives():
    """أكبر should NOT be classified as الكبر"""
    result = classifier.predict("الله أكبر")
    assert "الكبر" not in result.behaviors

def test_context_awareness():
    """Same root, different behavior based on context"""
    # استكبر فرعون = arrogance
    result1 = classifier.predict("استكبر فرعون على موسى")
    assert "الكبر" in result1.behaviors
    
    # كبر سنه = grew old (not arrogance)
    result2 = classifier.predict("كبر سنه وضعف بصره")
    assert "الكبر" not in result2.behaviors

def test_multi_label():
    """Verse can have multiple behaviors"""
    result = classifier.predict("وإذا قيل لهم آمنوا كما آمن الناس قالوا أنؤمن كما آمن السفهاء")
    assert "النفاق" in result.behaviors
    assert "الكبر" in result.behaviors
```

### Deliverables:
- [ ] `src/ml/behavioral_classifier.py` - Classifier model
- [ ] `models/qbm-behavior-classifier/` - Trained model
- [ ] `tests/test_classifier.py` - Accuracy tests
- [ ] Target accuracy: >90% on held-out test set

---

## Layer 4: Relation Extraction (Week 5-6)

### Task 4.1: Define Relationship Types
```python
RELATION_TYPES = {
    "CAUSES": "يسبب",           # الغفلة → الكبر
    "RESULTS_IN": "ينتج عنه",    # الكبر → الظلم
    "PREVENTS": "يمنع",          # التقوى → المعصية
    "OPPOSITE_OF": "نقيض",       # الصدق ↔ الكذب
    "PREREQUISITE": "شرط لـ",    # الإيمان → قبول العمل
    "INTENSIFIES": "يزيد",       # الإصرار → القسوة
}
```

### Task 4.2: Train Relation Extraction Model
**Input:** Text with two behaviors marked
**Output:** Relationship type + direction + confidence

```python
# Training example:
{
    "text": "ثُمَّ قَسَتْ قُلُوبُكُم مِّن بَعْدِ ذَٰلِكَ",
    "entity1": "الذنب",
    "entity2": "القسوة",
    "relation": "CAUSES",
    "direction": "entity1 → entity2"
}
```

### Tests for Layer 4:
```python
def test_causal_extraction():
    """Should identify الكبر CAUSES قسوة القلب"""
    result = relation_extractor.predict(
        "الكبر يؤدي إلى قسوة القلب"
    )
    assert result.relation == "CAUSES"
    assert result.source == "الكبر"
    assert result.target == "قسوة القلب"

def test_bidirectional():
    """Opposites should be bidirectional"""
    result = relation_extractor.predict("الصدق نقيض الكذب")
    assert result.relation == "OPPOSITE_OF"
    assert result.bidirectional == True
```

### Deliverables:
- [ ] `src/ml/relation_extractor.py` - Extraction model
- [ ] `models/qbm-relation-extractor/` - Trained model
- [ ] `data/relations/extracted_relations.jsonl` - Discovered relations
- [ ] Target: Discover 1000+ new causal relationships

---

## Layer 5: Graph Neural Network Reasoning (Week 7-8)

### Task 5.1: Convert Graph to PyTorch Geometric Format
```python
from torch_geometric.data import Data

# Nodes: behaviors, verses, tafsir entries
# Edges: relationships with types
data = Data(
    x=node_embeddings,      # From Layer 2
    edge_index=edge_index,  # Connectivity
    edge_attr=edge_types,   # Relationship types
)
```

### Task 5.2: Train GNN for Multi-hop Reasoning
```python
class QBMGraphReasoner(torch.nn.Module):
    def __init__(self):
        self.conv1 = GATConv(768, 256, heads=8)
        self.conv2 = GATConv(256*8, 256, heads=4)
        self.conv3 = GATConv(256*4, 128, heads=2)
    
    def find_path(self, start, end, max_hops=5):
        """Find behavioral chain: الغفلة → ? → ? → جهنم"""
        pass
    
    def discover_patterns(self):
        """Find patterns not explicitly annotated"""
        pass
```

### Tests for Layer 5:
```python
def test_multi_hop_reasoning():
    """Should find path from الغفلة to قسوة القلب"""
    path = gnn.find_path("الغفلة", "قسوة القلب")
    assert len(path) >= 2
    assert path[0] == "الغفلة"
    assert path[-1] == "قسوة القلب"

def test_pattern_discovery():
    """Should discover patterns not in training data"""
    patterns = gnn.discover_patterns()
    # Should find things like:
    # "Behaviors with agent=منافق always have evaluation=مذموم"
    assert len(patterns) > 0
```

### Deliverables:
- [ ] `src/ml/graph_reasoner.py` - GNN model
- [ ] `models/qbm-graph-reasoner/` - Trained model
- [ ] `data/discoveries/` - Discovered patterns
- [ ] Target: 100+ discovered patterns

---

## Layer 6: Cross-Tafsir Semantic Alignment (Week 9-10)

### Task 6.1: Semantic Similarity Between Tafsir
```python
class TafsirAligner:
    def align_interpretations(self, verse_id):
        """Compare 5 tafsir semantically, not by keywords"""
        tafsirs = load_all_tafsir(verse_id)
        embeddings = {k: self.model.encode(v) for k, v in tafsirs.items()}
        
        return {
            "agreement": find_semantic_agreement(embeddings),
            "disagreement": find_semantic_disagreement(embeddings),
            "unique_insights": find_unique_per_source(embeddings),
        }
```

### Task 6.2: Behavioral Consensus Detection
```python
def find_behavioral_consensus(behavior: str):
    """For الكبر, what do all 5 scholars agree on?"""
    mentions = find_all_mentions(behavior)
    
    for verse_id, mentions in grouped_mentions.items():
        alignment = align_interpretations(verse_id)
        # Identify consensus vs disagreement
```

### Tests for Layer 6:
```python
def test_agreement_detection():
    """Should find where all 5 scholars agree"""
    alignment = aligner.align("2:7")  # الختم على القلوب
    assert alignment.agreement_score > 0.8
    assert "consequence" in alignment.agreed_aspects

def test_unique_insight_detection():
    """Should find insights unique to one scholar"""
    alignment = aligner.align("2:10")
    assert len(alignment.unique_insights["qurtubi"]) > 0
```

### Deliverables:
- [ ] `src/ml/tafsir_aligner.py` - Alignment model
- [ ] `data/alignments/` - Verse-by-verse alignments
- [ ] `data/consensus/` - Behavioral consensus data
- [ ] Target: Alignment for all 6,236 verses

---

## Layer 7: Fine-tuned Arabic LLM (Week 11-14)

### Task 7.1: Prepare Training Data
```python
training_data = []

# Behavioral analysis
for span in spans:
    training_data.append({
        "instruction": f"حلل السلوك في: {span.verse_text}",
        "output": generate_11_dimension_analysis(span)
    })

# Comparisons
for behavior in behaviors:
    training_data.append({
        "instruction": f"قارن {behavior} بين المؤمن والكافر",
        "output": generate_comparison(behavior)
    })

# Chain analysis
for chain in chains:
    training_data.append({
        "instruction": f"ما السلسلة من {chain.start} إلى {chain.end}؟",
        "output": generate_chain_analysis(chain)
    })

# Tafsir cross-reference
for verse in verses:
    training_data.append({
        "instruction": f"ما أقوال المفسرين في {verse.ref}؟",
        "output": generate_tafsir_comparison(verse)
    })
```

### Task 7.2: Fine-tune JAIS-30B with LoRA
```python
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("inception-mbzuai/jais-30b-v3")

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

# Train on 8x A100
trainer = Trainer(
    model=model,
    train_dataset=training_data,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=True,
        deepspeed="ds_config.json",
    )
)
```

### Tests for Layer 7:
```python
def test_framework_understanding():
    """Model should think in 11 dimensions"""
    response = model.generate("حلل سلوك الكبر")
    
    for dim in ["organic", "situational", "systemic", "spatial", 
                "temporal", "agent", "source", "evaluation", 
                "heart_type", "consequence", "relationships"]:
        assert dim in response or arabic_dim[dim] in response

def test_causal_reasoning():
    """Model should explain WHY, not just WHAT"""
    response = model.generate("ما علاقة الكبر بقسوة القلب؟")
    
    assert "يسبب" in response or "يؤدي" in response
    assert "سلسلة" in response or "مراحل" in response

def test_tafsir_synthesis():
    """Model should synthesize 5 tafsir, not just list"""
    response = model.generate("ما إجماع المفسرين على الختم؟")
    
    assert "اتفق" in response or "أجمع" in response
    assert "ابن كثير" in response
    assert "الطبري" in response
```

### Deliverables:
- [ ] `src/ml/qbm_llm.py` - Fine-tuning script
- [ ] `models/qbm-jais-30b-finetuned/` - Trained model
- [ ] `data/training/llm_training_data.jsonl` - Training data
- [ ] Target: Model that thinks in QBM framework

---

## Integration & Testing (Week 15-16)

### Task: Connect All 7 Layers
```python
class QBMIntelligentSystem:
    def __init__(self):
        self.embeddings = load_qbm_embeddings()      # Layer 2
        self.classifier = load_behavior_classifier()  # Layer 3
        self.relation_extractor = load_relation_model()  # Layer 4
        self.graph_reasoner = load_gnn()              # Layer 5
        self.tafsir_aligner = load_aligner()          # Layer 6
        self.llm = load_qbm_llm()                     # Layer 7
    
    def answer(self, question: str) -> str:
        # 1. Classify question type
        # 2. Extract relevant behaviors
        # 3. Find relationships
        # 4. Reason over graph
        # 5. Align tafsir
        # 6. Generate response with LLM
        pass
```

### End-to-End Tests:
```python
def test_intelligent_response():
    """The ultimate test: intelligent answer"""
    question = "ما علاقة الكبر بقسوة القلب؟"
    
    response = system.answer(question)
    
    # Should NOT be: "They co-occur 3 times"
    # Should BE: Causal chain with evidence
    
    assert "يسبب" in response
    assert "سلسلة" in response
    assert "ابن كثير" in response
    assert "confidence" in response or "%" in response
```

---

## Git Commits Plan

### Commit 1: Layer 2 - Arabic Embeddings
```bash
git add src/ml/arabic_embeddings.py tests/test_embeddings.py
git commit -m "feat(ml): Add Arabic-first embeddings fine-tuning

- Fine-tune AraBERT on QBM behavioral spans
- Create contrastive learning pairs
- Replace generic multilingual embeddings
- Tests for embedding quality"
```

### Commit 2: Layer 3 - Behavioral Classifier
```bash
git add src/ml/behavioral_classifier.py tests/test_classifier.py
git commit -m "feat(ml): Add behavioral classifier replacing keyword matching

- Multi-label classification for 87 behaviors
- Context-aware prediction
- Eliminates false positives (أكبر ≠ الكبر)
- >90% accuracy on test set"
```

### Commit 3: Layer 4 - Relation Extraction
```bash
git add src/ml/relation_extractor.py data/relations/
git commit -m "feat(ml): Add relation extraction model

- 6 relationship types (CAUSES, PREVENTS, etc.)
- Learns causal patterns from text
- Discovers relationships not manually annotated"
```

### Commit 4: Layer 5 - Graph Reasoning
```bash
git add src/ml/graph_reasoner.py
git commit -m "feat(ml): Add GNN for multi-hop graph reasoning

- GAT-based architecture
- Multi-hop path finding
- Pattern discovery
- Learns from graph structure"
```

### Commit 5: Layer 6 - Tafsir Alignment
```bash
git add src/ml/tafsir_aligner.py data/alignments/
git commit -m "feat(ml): Add cross-tafsir semantic alignment

- Semantic similarity (not keyword matching)
- Agreement/disagreement detection
- Unique insight extraction per scholar"
```

### Commit 6: Layer 7 - Fine-tuned LLM
```bash
git add src/ml/qbm_llm.py data/training/
git commit -m "feat(ml): Add JAIS-30B fine-tuned on QBM framework

- LoRA fine-tuning on 8x A100
- Training data from all QBM components
- Model thinks in 11-dimensional framework"
```

### Commit 7: Integration
```bash
git add src/ml/intelligent_system.py tests/test_integration.py
git commit -m "feat(ml): Integrate all 7 layers into unified intelligent system

- All layers connected
- End-to-end intelligent responses
- From mechanical to learning system"
```

---

## Success Metrics

| Metric | Current (Mechanical) | Target (Intelligent) |
|--------|---------------------|---------------------|
| False positive rate | ~40% | <5% |
| Relationship accuracy | N/A (rule-based) | >85% |
| Path discovery | Manual only | Automatic |
| Tafsir alignment | Keyword overlap | Semantic similarity |
| Response quality | "Co-occur 3 times" | Causal chains + evidence |
| GPU utilization | ~0% (idle) | >80% (training) |

---

## Timeline Summary

| Week | Layer | Deliverable |
|------|-------|-------------|
| 1-2 | Layer 2 | Arabic embeddings fine-tuned |
| 3-4 | Layer 3 | Behavioral classifier trained |
| 5-6 | Layer 4 | Relation extraction model |
| 7-8 | Layer 5 | GNN graph reasoner |
| 9-10 | Layer 6 | Tafsir semantic alignment |
| 11-14 | Layer 7 | JAIS-30B fine-tuned |
| 15-16 | Integration | All layers connected |

---

## The Difference This Makes

### Before (Mechanical):
```
Q: "ما علاقة الكبر بقسوة القلب؟"
A: "They appear together 3 times" ❌ USELESS
```

### After (Intelligent):
```
Q: "ما علاقة الكبر بقسوة القلب؟"
A: "الكبر يسبب القسوة عبر سلسلة:
    الكبر → الإعراض → عدم التأثر → القسوة
    
    الدليل: 12 آية بثقة 89%
    
    إجماع المفسرين:
    - ابن كثير: الكبر سبب مباشر
    - القرطبي: من آثار الكبر على القلب
    - الطبري: يتدرج من الكبر إلى القسوة
    
    اكتشاف: هذا النمط يتكرر مع المنافقين أكثر من الكفار" ✅ INTELLIGENT
```

---

## Next Steps

1. **Start with Layer 2** - Fine-tune Arabic embeddings
2. **Use the GPUs** - They should be TRAINING, not idle
3. **Test each layer** - Don't proceed without passing tests
4. **Commit incrementally** - One layer per commit
5. **Validate with scholars** - Ensure Islamic accuracy
