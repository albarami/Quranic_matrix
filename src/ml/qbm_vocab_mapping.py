"""
QBM Vocabulary Mapping

Maps existing Arabic behavior annotations to proper controlled BEH_* IDs
from the official vocab/behavior_concepts.json schema.

Based on:
- vocab/behavior_concepts.json (7 categories, ~70 concepts)
- Bouzidani's 5-context framework
- behavioral_map_research.md findings
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
VOCAB_DIR = PROJECT_ROOT / "vocab"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# LOAD OFFICIAL VOCABULARY
# =============================================================================

def load_behavior_concepts() -> Dict:
    """Load official behavior concepts from vocab."""
    with open(VOCAB_DIR / "behavior_concepts.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def load_agents() -> Dict:
    """Load agent types from vocab."""
    with open(VOCAB_DIR / "agents.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def load_organs() -> Dict:
    """Load organ vocabulary."""
    with open(VOCAB_DIR / "organs.json", 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# ARABIC TO BEH_* MAPPING
# =============================================================================

# Map from Arabic annotation labels to official BEH_* IDs
# Based on vocab/behavior_concepts.json categories

ARABIC_TO_BEH_ID: Dict[str, str] = {
    # =========================================================================
    # SPEECH BEHAVIORS (BEH_SPEECH_*)
    # =========================================================================
    "صدق": "BEH_SPEECH_TRUTHFULNESS",
    "كذب": "BEH_SPEECH_LYING",
    "غيبة": "BEH_SPEECH_BACKBITING",
    "دعاء": "BEH_SPEECH_SUPPLICATION",
    
    # =========================================================================
    # EMOTIONAL BEHAVIORS (BEH_EMO_*)
    # =========================================================================
    "شكر": "BEH_EMO_GRATITUDE",
    "صبر": "BEH_EMO_PATIENCE",
    "حسد": "BEH_EMO_ENVY",
    "غضب": "BEH_EMO_ANGER",
    "رضا": "BEH_EMO_CONTENTMENT",  # Not in vocab, add as custom
    
    # =========================================================================
    # SPIRITUAL BEHAVIORS (BEH_SPI_*)
    # =========================================================================
    "إيمان": "BEH_SPI_FAITH",
    "كفر": "BEH_SPI_DISBELIEF",
    "نفاق": "BEH_SPI_HYPOCRISY",
    "رياء": "BEH_SPI_SHOWING_OFF",
    "تقوى": "BEH_SPI_TAQWA",
    "توكل": "BEH_SPI_TAWAKKUL",
    "توبة": "BEH_SPI_REPENTANCE",
    "ذكر": "BEH_SPI_REMEMBRANCE",
    "خشوع": "BEH_SPI_KHUSHU",  # Humility in worship
    "إخلاص": "BEH_SPI_SINCERITY",
    
    # =========================================================================
    # SOCIAL BEHAVIORS (BEH_SOC_*)
    # =========================================================================
    "ظلم": "BEH_SOC_OPPRESSION",
    "عدل": "BEH_SOC_JUSTICE",
    "إحسان": "BEH_SOC_EXCELLENCE",  # Ihsan
    "رحمة": "BEH_SOC_MERCY",
    "خيانة": "BEH_SOC_BETRAYAL",
    "أمانة": "BEH_SOC_TRUSTWORTHINESS",
    "فجور": "BEH_SOC_CORRUPTION",
    "فسق": "BEH_SOC_TRANSGRESSION",
    
    # =========================================================================
    # COGNITIVE BEHAVIORS (BEH_COG_*)
    # =========================================================================
    "كبر": "BEH_COG_ARROGANCE",
    "تواضع": "BEH_COG_HUMILITY",
    "غفلة": "BEH_COG_HEEDLESSNESS",
    
    # =========================================================================
    # FINANCIAL BEHAVIORS (BEH_FIN_*)
    # =========================================================================
    "بخل": "BEH_FIN_HOARDING",
    "زهد": "BEH_FIN_ASCETICISM",  # Custom - renunciation of worldly
    
    # =========================================================================
    # PHYSICAL BEHAVIORS (BEH_PHY_*)
    # =========================================================================
    "حياء": "BEH_PHY_MODESTY",  # Physical modesty
    
    # =========================================================================
    # POLYTHEISM/SHIRK - Special category
    # =========================================================================
    "شرك": "BEH_SPI_SHIRK",  # Polytheism - major spiritual deviation
}


# Behavior form mapping based on Bouzidani's situational context
BEH_ID_TO_FORM: Dict[str, str] = {
    # Speech acts
    "BEH_SPEECH_TRUTHFULNESS": "speech_act",
    "BEH_SPEECH_LYING": "speech_act",
    "BEH_SPEECH_BACKBITING": "speech_act",
    "BEH_SPEECH_SUPPLICATION": "speech_act",
    
    # Inner states (emotional)
    "BEH_EMO_GRATITUDE": "inner_state",
    "BEH_EMO_PATIENCE": "trait_disposition",
    "BEH_EMO_ENVY": "inner_state",
    "BEH_EMO_ANGER": "inner_state",
    "BEH_EMO_CONTENTMENT": "inner_state",
    
    # Inner states (spiritual)
    "BEH_SPI_FAITH": "inner_state",
    "BEH_SPI_DISBELIEF": "inner_state",
    "BEH_SPI_HYPOCRISY": "inner_state",
    "BEH_SPI_SHOWING_OFF": "inner_state",
    "BEH_SPI_TAQWA": "trait_disposition",
    "BEH_SPI_TAWAKKUL": "inner_state",
    "BEH_SPI_REPENTANCE": "mixed",
    "BEH_SPI_REMEMBRANCE": "mixed",
    "BEH_SPI_KHUSHU": "inner_state",
    "BEH_SPI_SINCERITY": "inner_state",
    "BEH_SPI_SHIRK": "inner_state",
    
    # Relational acts (social)
    "BEH_SOC_OPPRESSION": "relational_act",
    "BEH_SOC_JUSTICE": "relational_act",
    "BEH_SOC_EXCELLENCE": "trait_disposition",
    "BEH_SOC_MERCY": "trait_disposition",
    "BEH_SOC_BETRAYAL": "relational_act",
    "BEH_SOC_TRUSTWORTHINESS": "trait_disposition",
    "BEH_SOC_CORRUPTION": "relational_act",
    "BEH_SOC_TRANSGRESSION": "trait_disposition",
    
    # Cognitive
    "BEH_COG_ARROGANCE": "inner_state",
    "BEH_COG_HUMILITY": "trait_disposition",
    "BEH_COG_HEEDLESSNESS": "inner_state",
    
    # Financial
    "BEH_FIN_HOARDING": "trait_disposition",
    "BEH_FIN_ASCETICISM": "trait_disposition",
    
    # Physical
    "BEH_PHY_MODESTY": "trait_disposition",
}


# Textual evaluation based on Quranic praise/blame
BEH_ID_TO_EVAL: Dict[str, str] = {
    # Praised (صالح)
    "BEH_SPEECH_TRUTHFULNESS": "EVAL_SALIH",
    "BEH_SPEECH_SUPPLICATION": "EVAL_SALIH",
    "BEH_EMO_GRATITUDE": "EVAL_SALIH",
    "BEH_EMO_PATIENCE": "EVAL_SALIH",
    "BEH_EMO_CONTENTMENT": "EVAL_SALIH",
    "BEH_SPI_FAITH": "EVAL_SALIH",
    "BEH_SPI_TAQWA": "EVAL_SALIH",
    "BEH_SPI_TAWAKKUL": "EVAL_SALIH",
    "BEH_SPI_REPENTANCE": "EVAL_SALIH",
    "BEH_SPI_REMEMBRANCE": "EVAL_SALIH",
    "BEH_SPI_KHUSHU": "EVAL_SALIH",
    "BEH_SPI_SINCERITY": "EVAL_SALIH",
    "BEH_SOC_JUSTICE": "EVAL_SALIH",
    "BEH_SOC_EXCELLENCE": "EVAL_SALIH",
    "BEH_SOC_MERCY": "EVAL_SALIH",
    "BEH_SOC_TRUSTWORTHINESS": "EVAL_SALIH",
    "BEH_COG_HUMILITY": "EVAL_SALIH",
    "BEH_FIN_ASCETICISM": "EVAL_SALIH",
    "BEH_PHY_MODESTY": "EVAL_SALIH",
    
    # Blamed (سيء)
    "BEH_SPEECH_LYING": "EVAL_SAYYI",
    "BEH_SPEECH_BACKBITING": "EVAL_SAYYI",
    "BEH_EMO_ENVY": "EVAL_SAYYI",
    "BEH_SPI_DISBELIEF": "EVAL_SAYYI",
    "BEH_SPI_HYPOCRISY": "EVAL_SAYYI",
    "BEH_SPI_SHOWING_OFF": "EVAL_SAYYI",
    "BEH_SPI_SHIRK": "EVAL_SAYYI",
    "BEH_SOC_OPPRESSION": "EVAL_SAYYI",
    "BEH_SOC_BETRAYAL": "EVAL_SAYYI",
    "BEH_SOC_CORRUPTION": "EVAL_SAYYI",
    "BEH_SOC_TRANSGRESSION": "EVAL_SAYYI",
    "BEH_COG_ARROGANCE": "EVAL_SAYYI",
    "BEH_COG_HEEDLESSNESS": "EVAL_SAYYI",
    "BEH_FIN_HOARDING": "EVAL_SAYYI",
    
    # Neutral (context-dependent)
    "BEH_EMO_ANGER": "EVAL_NEUTRAL",
}


# Systemic context mapping (which system the behavior relates to)
BEH_ID_TO_SYSTEMIC: Dict[str, List[str]] = {
    # Self + God
    "BEH_SPI_FAITH": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_DISBELIEF": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_TAQWA": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_TAWAKKUL": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_REPENTANCE": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_REMEMBRANCE": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_KHUSHU": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_SINCERITY": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_HYPOCRISY": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPI_SHOWING_OFF": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SPI_SHIRK": ["SYS_SELF", "SYS_GOD"],
    "BEH_SPEECH_SUPPLICATION": ["SYS_SELF", "SYS_GOD"],
    
    # Self + Creation (social)
    "BEH_SOC_OPPRESSION": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SOC_JUSTICE": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SOC_MERCY": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SOC_BETRAYAL": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SOC_TRUSTWORTHINESS": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SPEECH_TRUTHFULNESS": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SPEECH_LYING": ["SYS_SELF", "SYS_CREATION"],
    "BEH_SPEECH_BACKBITING": ["SYS_SELF", "SYS_CREATION"],
    "BEH_EMO_ENVY": ["SYS_SELF", "SYS_CREATION"],
    
    # Self only
    "BEH_EMO_PATIENCE": ["SYS_SELF"],
    "BEH_EMO_ANGER": ["SYS_SELF"],
    "BEH_EMO_CONTENTMENT": ["SYS_SELF"],
    "BEH_EMO_GRATITUDE": ["SYS_SELF", "SYS_GOD"],
    "BEH_COG_ARROGANCE": ["SYS_SELF"],
    "BEH_COG_HUMILITY": ["SYS_SELF"],
    "BEH_COG_HEEDLESSNESS": ["SYS_SELF"],
    "BEH_FIN_HOARDING": ["SYS_SELF"],
    "BEH_FIN_ASCETICISM": ["SYS_SELF", "SYS_LIFE"],
    "BEH_PHY_MODESTY": ["SYS_SELF"],
    "BEH_SOC_EXCELLENCE": ["SYS_SELF", "SYS_GOD", "SYS_CREATION"],
    "BEH_SOC_CORRUPTION": ["SYS_SELF", "SYS_SOCIETY"],
    "BEH_SOC_TRANSGRESSION": ["SYS_SELF", "SYS_GOD"],
}


# =============================================================================
# NON-BEHAVIOR LABELS (to exclude)
# =============================================================================

NON_BEHAVIOR_ARABIC = {
    # Entities (not behaviors)
    "نبي": "ENT_PROPHET",
    "رسول": "ENT_MESSENGER", 
    "ملائكة": "ENT_ANGEL",
    
    # Agent states (not behaviors)
    "مؤمن": "AGT_BELIEVER",
    "كافر": "AGT_DISBELIEVER",
    "منافق": "AGT_HYPOCRITE",
    
    # Organ/state labels (not behaviors)
    "قلب": "ORG_HEART",
    "سليم": "STATE_SOUND",
    "مريض": "STATE_SICK",
    "قاسي": "STATE_HARD",
    "مختوم": "STATE_SEALED",
    "ميت": "STATE_DEAD",
    "منيب": "STATE_REPENTANT",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_beh_id(arabic: str) -> Optional[str]:
    """Get BEH_* ID for Arabic behavior term."""
    return ARABIC_TO_BEH_ID.get(arabic)


def is_true_behavior(arabic: str) -> bool:
    """Check if Arabic term is a true behavior (not entity/state)."""
    return arabic in ARABIC_TO_BEH_ID


def is_non_behavior(arabic: str) -> bool:
    """Check if Arabic term is a non-behavior (entity/state)."""
    return arabic in NON_BEHAVIOR_ARABIC


def get_behavior_form(beh_id: str) -> str:
    """Get behavior form for BEH_* ID."""
    return BEH_ID_TO_FORM.get(beh_id, "unknown")


def get_textual_eval(beh_id: str) -> str:
    """Get textual evaluation for BEH_* ID."""
    return BEH_ID_TO_EVAL.get(beh_id, "EVAL_UNKNOWN")


def get_systemic_context(beh_id: str) -> List[str]:
    """Get systemic context for BEH_* ID."""
    return BEH_ID_TO_SYSTEMIC.get(beh_id, ["SYS_UNKNOWN"])


# =============================================================================
# ANNOTATION CONVERSION
# =============================================================================

def convert_annotation(ann: Dict) -> Optional[Dict]:
    """
    Convert annotation from Arabic label to proper BEH_* schema.
    Returns None if not a true behavior.
    """
    arabic = ann.get("behavior_ar", "")
    
    if not arabic:
        return None
    
    if is_non_behavior(arabic):
        return None
    
    beh_id = get_beh_id(arabic)
    if not beh_id:
        return None
    
    return {
        # Original fields
        "surah": ann.get("surah"),
        "ayah": ann.get("ayah"),
        "context": ann.get("context", ""),
        "source": ann.get("source", ""),
        
        # Converted fields
        "behavior_ar": arabic,
        "behavior_en": ann.get("behavior_en", ""),
        "behavior_id": beh_id,
        "behavior_form": get_behavior_form(beh_id),
        "textual_eval": get_textual_eval(beh_id),
        "systemic_context": get_systemic_context(beh_id),
        
        # Action class (all volitional for true behaviors)
        "action_class": "ACT_VOLITIONAL",
    }


def convert_all_annotations():
    """Convert all annotations to proper schema."""
    input_file = DATA_DIR / "annotations" / "tafsir_behavioral_annotations.jsonl"
    output_file = DATA_DIR / "annotations" / "tafsir_behavioral_annotations_schema.jsonl"
    
    converted = []
    skipped_non_behavior = Counter()
    skipped_unknown = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ann = json.loads(line)
                arabic = ann.get("behavior_ar", "")
                
                if is_non_behavior(arabic):
                    skipped_non_behavior[arabic] += 1
                    continue
                
                result = convert_annotation(ann)
                if result:
                    converted.append(result)
                else:
                    skipped_unknown[arabic] += 1
    
    # Save converted
    with open(output_file, 'w', encoding='utf-8') as f:
        for ann in converted:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    
    # Print stats
    print("=" * 70)
    print("ANNOTATION CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Converted: {len(converted):,} TRUE behavior annotations")
    print(f"\nSkipped non-behaviors: {sum(skipped_non_behavior.values()):,}")
    for arabic, count in skipped_non_behavior.most_common():
        print(f"  {arabic}: {count:,}")
    
    if skipped_unknown:
        print(f"\nSkipped unknown: {sum(skipped_unknown.values()):,}")
        for arabic, count in skipped_unknown.most_common():
            print(f"  {arabic}: {count:,}")
    
    # Print behavior distribution
    print(f"\nBehavior distribution in converted data:")
    beh_counts = Counter(ann["behavior_id"] for ann in converted)
    for beh_id, count in beh_counts.most_common():
        print(f"  {beh_id}: {count:,}")
    
    return converted


def print_mapping_stats():
    """Print mapping statistics."""
    print("=" * 70)
    print("QBM VOCABULARY MAPPING")
    print("=" * 70)
    
    print(f"\nTRUE BEHAVIORS ({len(ARABIC_TO_BEH_ID)} mappings):")
    for arabic, beh_id in sorted(ARABIC_TO_BEH_ID.items(), key=lambda x: x[1]):
        form = get_behavior_form(beh_id)
        eval_ = get_textual_eval(beh_id)
        print(f"  {arabic:10} -> {beh_id:30} [{form:15}] [{eval_}]")
    
    print(f"\nNON-BEHAVIORS ({len(NON_BEHAVIOR_ARABIC)} labels to exclude):")
    for arabic, label in NON_BEHAVIOR_ARABIC.items():
        print(f"  {arabic:10} -> {label} (EXCLUDED)")


if __name__ == "__main__":
    print_mapping_stats()
    print("\n")
    convert_all_annotations()
