"""
Quranic Human-Behavior Classification Matrix (QBM)
Usul al-Fiqh Aligned Implementation

Based on:
- Bouzidani's paper on Quranic behavior matrix in Islamic psychology context
- فطرة البصيرة (fitrah-based moral insight) + موضوعية الأخلاق (objective ethics)
- Three-world model: sensory / psychic-cognitive / spiritual
- عمل/نية/غريزة framing

Key Principles:
- No evidence → no label (fail-closed)
- Direct vs indirect tracked per label
- Qur'an-only signals separated from juristic derivations
- Controlled IDs prevent drift

Label Families:
- BEH_* : Behavior concepts (actions, traits, dispositions, inner states)
- THM_* : Thematic constructs (accountability, testimony, reward/punishment)
- META_* : Meta-Quranic discourse constructs (revelation, rhetoric)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json


# =============================================================================
# ENUMS - Controlled Vocabularies (Anti-drift)
# =============================================================================

class BehaviorCategory(Enum):
    """Three label families - do not conflate."""
    BEHAVIOR_CONCEPT = "BEH"      # Actions, traits, dispositions, inner states
    THEMATIC_CONSTRUCT = "THM"    # Frames/themes
    META_DISCOURSE = "META"       # Revelation process, rhetoric
    ENTITY = "ENT"                # NOT a behavior - entity mention
    STATE_LABEL = "STL"           # NOT a behavior - person state/label


class BehaviorForm(Enum):
    """Structural form of behavior (behavior.form)."""
    SPEECH_ACT = "speech_act"
    PHYSICAL_ACT = "physical_act"
    INNER_STATE = "inner_state"
    TRAIT_DISPOSITION = "trait_disposition"
    RELATIONAL_ACT = "relational_act"
    OMISSION = "omission"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ActionClass(Enum):
    """AX_ACTION_CLASS - Paper-aligned."""
    ACT_INSTINCTIVE_OR_AUTOMATIC = "غريزي/لا إرادي"
    ACT_VOLITIONAL = "إرادي/مكتسب"
    ACT_UNKNOWN = "unknown"


class ActionTextualEval(Enum):
    """AX_ACTION_TEXTUAL_EVAL - How Quran evaluates the action."""
    EVAL_SALIH = "صالح"
    EVAL_GHAYR_SALIH = "غير صالح"
    EVAL_NEUTRAL = "neutral"
    EVAL_NOT_APPLICABLE = "not_applicable"
    EVAL_UNKNOWN = "unknown"


class SupportType(Enum):
    """Evidence support type."""
    DIRECT = "direct"      # Explicit wording/structure
    INDIRECT = "indirect"  # Implication, inference, metaphor


class IndicationTag(Enum):
    """Usul indication tags."""
    DALALAH_MANTUQ = "dalalah_mantuq"        # Explicit meaning
    DALALAH_MAFHUM = "dalalah_mafhum"        # Implied meaning
    NARRATIVE_INFERENCE = "narrative_inference"
    METAPHOR_METONYMY = "metaphor_metonymy"
    SABAB_NUZUL_USED = "sabab_nuzul_used"


class SystemicAxis(Enum):
    """AX_SYSTEMIC - Multi-label frame."""
    SYS_SELF = "النفس"
    SYS_CREATION = "الخلق"
    SYS_GOD = "الخالق"
    SYS_COSMOS = "الكون"
    SYS_LIFE = "الحياة"


class AgentType(Enum):
    """Agent types - controlled."""
    AGT_BELIEVER = "مؤمن"
    AGT_HYPOCRITE = "منافق"
    AGT_DISBELIEVER = "كافر"
    AGT_HUMAN_GENERAL = "إنسان"
    AGT_PROPHET_MESSENGER = "نبي/رسول"
    AGT_ANGEL = "ملك"
    AGT_JINN = "جن"
    AGT_ANIMAL = "حيوان"
    AGT_COLLECTIVE = "جماعة"
    AGT_UNKNOWN = "unknown"


class OrganLabel(Enum):
    """Organ labels for organic axis."""
    ORG_HEART = "قلب"
    ORG_TONGUE = "لسان"
    ORG_EYE = "عين"
    ORG_EAR = "أذن"
    ORG_HAND = "يد"
    ORG_FOOT = "رجل"
    ORG_SKIN = "جلد"
    ORG_CHEST = "صدر"
    ORG_FACE = "وجه"


class OrganRole(Enum):
    """Role of organ in the span."""
    TOOL = "tool"
    PERCEPTION = "perception"
    ACCOUNTABILITY_WITNESS = "accountability_witness"
    METAPHOR = "metaphor"
    UNKNOWN = "unknown"


class OrganSemanticDomain(Enum):
    """Semantic domains for ORG_HEART."""
    PHYSIOLOGICAL = "physiological"
    COGNITIVE = "cognitive"
    SPIRITUAL = "spiritual"
    EMOTIONAL = "emotional"


class SpeechMode(Enum):
    """Normative textual speech mode."""
    COMMAND = "command"
    PROHIBITION = "prohibition"
    INFORMATIVE = "informative"
    NARRATIVE = "narrative"
    PARABLE = "parable"
    UNKNOWN = "unknown"


class QuranDeonticSignal(Enum):
    """Quran-only deontic signals (not juristic rulings)."""
    AMR = "أمر"           # Command
    NAHY = "نهي"          # Prohibition
    TARGHIB = "ترغيب"     # Encouragement with reward
    TARHIB = "ترهيب"      # Warning with threat
    KHABAR = "خبر"        # Informative/report


# =============================================================================
# BEHAVIOR CONCEPT TAXONOMY (BEH_*)
# =============================================================================

@dataclass
class BehaviorConcept:
    """Definition of a Quranic behavior concept."""
    id: str                          # Controlled ID (BEH_*)
    arabic: str                      # Arabic term
    english: str                     # English translation
    category: BehaviorCategory       # Must be BEHAVIOR_CONCEPT
    form: BehaviorForm               # Structural form
    definition_ar: str               # Arabic definition
    definition_en: str               # English definition
    is_true_behavior: bool           # True = actual behavior, False = entity/state
    action_class: ActionClass        # Volitional vs automatic
    default_eval: ActionTextualEval  # Default textual evaluation
    opposite_id: str = ""            # Opposite behavior ID
    related_ids: List[str] = field(default_factory=list)
    systemic_frame: List[SystemicAxis] = field(default_factory=list)


# =============================================================================
# COMPLETE BEHAVIOR TAXONOMY
# =============================================================================

BEHAVIOR_TAXONOMY: Dict[str, BehaviorConcept] = {
    
    # =========================================================================
    # أمراض القلوب - HEART DISEASES (Negative Inner States)
    # =========================================================================
    
    "BEH_HEART_ARROGANCE": BehaviorConcept(
        id="BEH_HEART_ARROGANCE",
        arabic="كبر",
        english="arrogance",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="رؤية النفس فوق الآخرين وازدراء الحق - بطر الحق وغمط الناس",
        definition_en="Seeing oneself above others and rejecting truth",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_CHAR_HUMILITY",
        related_ids=["BEH_CHAR_OPPRESSION", "BEH_HEART_HEEDLESSNESS"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_ENVY": BehaviorConcept(
        id="BEH_HEART_ENVY",
        arabic="حسد",
        english="envy",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="تمني زوال النعمة عن الغير",
        definition_en="Wishing for the removal of blessings from others",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_STINGINESS"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_HEART_HEEDLESSNESS": BehaviorConcept(
        id="BEH_HEART_HEEDLESSNESS",
        arabic="غفلة",
        english="heedlessness",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="الإعراض عن ذكر الله والآخرة",
        definition_en="Turning away from remembrance of Allah and the Hereafter",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_WORSHIP_REMEMBRANCE",
        related_ids=["BEH_HEART_ARROGANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_HYPOCRISY": BehaviorConcept(
        id="BEH_HEART_HYPOCRISY",
        arabic="نفاق",
        english="hypocrisy",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="إظهار الإيمان وإبطان الكفر",
        definition_en="Displaying faith while concealing disbelief",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_HEART_SINCERITY",
        related_ids=["BEH_SPEECH_LYING", "BEH_HEART_SHOWING_OFF"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_SHOWING_OFF": BehaviorConcept(
        id="BEH_HEART_SHOWING_OFF",
        arabic="رياء",
        english="showing_off",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="العمل لأجل الناس لا لله",
        definition_en="Performing deeds for people's sake, not for Allah",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_HEART_SINCERITY",
        related_ids=["BEH_HEART_HYPOCRISY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_HEART_ANGER": BehaviorConcept(
        id="BEH_HEART_ANGER",
        arabic="غضب",
        english="anger",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="ثوران النفس لدفع المكروه",
        definition_en="Agitation of the soul to repel harm",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_NEUTRAL,  # Context-dependent
        opposite_id="BEH_CHAR_FORBEARANCE",
        related_ids=["BEH_CHAR_OPPRESSION"],
        systemic_frame=[SystemicAxis.SYS_SELF]
    ),
    
    # =========================================================================
    # فضائل القلوب - HEART VIRTUES (Positive Inner States)
    # =========================================================================
    
    "BEH_HEART_FAITH": BehaviorConcept(
        id="BEH_HEART_FAITH",
        arabic="إيمان",
        english="faith",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="التصديق بالله ورسله واليوم الآخر مع العمل",
        definition_en="Belief in Allah, His messengers, and the Last Day with action",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_CHAR_DISBELIEF",
        related_ids=["BEH_HEART_PIETY", "BEH_HEART_RELIANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_PIETY": BehaviorConcept(
        id="BEH_HEART_PIETY",
        arabic="تقوى",
        english="piety",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="حفظ النفس مما يؤثم بفعل الواجبات وترك المحرمات",
        definition_en="Protecting oneself from sin by fulfilling obligations and avoiding prohibitions",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_CHAR_TRANSGRESSION",
        related_ids=["BEH_HEART_FAITH", "BEH_WORSHIP_HUMILITY_PRAYER"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_RELIANCE": BehaviorConcept(
        id="BEH_HEART_RELIANCE",
        arabic="توكل",
        english="reliance_on_Allah",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="الاعتماد على الله مع الأخذ بالأسباب",
        definition_en="Depending on Allah while taking means",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_HEART_FAITH", "BEH_CHAR_PATIENCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_CONTENTMENT": BehaviorConcept(
        id="BEH_HEART_CONTENTMENT",
        arabic="رضا",
        english="contentment",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="طمأنينة القلب بقضاء الله",
        definition_en="Tranquility of heart with Allah's decree",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_PATIENCE", "BEH_WORSHIP_GRATITUDE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_HEART_SINCERITY": BehaviorConcept(
        id="BEH_HEART_SINCERITY",
        arabic="إخلاص",
        english="sincerity",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="تصفية العمل من كل شائبة لغير الله",
        definition_en="Purifying deeds from any impurity for other than Allah",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_HEART_SHOWING_OFF",
        related_ids=["BEH_HEART_FAITH"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_WORSHIP_HUMILITY_PRAYER": BehaviorConcept(
        id="BEH_WORSHIP_HUMILITY_PRAYER",
        arabic="خشوع",
        english="humility_in_prayer",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="سكون القلب وخضوعه لله في الصلاة",
        definition_en="Stillness and submission of heart to Allah in prayer",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_HEART_HEEDLESSNESS",
        related_ids=["BEH_HEART_PIETY", "BEH_WORSHIP_REMEMBRANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    # =========================================================================
    # أعمال العبادة - ACTS OF WORSHIP
    # =========================================================================
    
    "BEH_WORSHIP_REMEMBRANCE": BehaviorConcept(
        id="BEH_WORSHIP_REMEMBRANCE",
        arabic="ذكر",
        english="remembrance",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.MIXED,  # Heart + tongue
        definition_ar="استحضار عظمة الله في القلب واللسان",
        definition_en="Bringing to mind Allah's greatness in heart and tongue",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_HEART_HEEDLESSNESS",
        related_ids=["BEH_WORSHIP_SUPPLICATION", "BEH_WORSHIP_GRATITUDE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_WORSHIP_SUPPLICATION": BehaviorConcept(
        id="BEH_WORSHIP_SUPPLICATION",
        arabic="دعاء",
        english="supplication",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.SPEECH_ACT,
        definition_ar="طلب العبد من ربه بخضوع وتضرع",
        definition_en="Servant's request from Lord with humility and submission",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_WORSHIP_REMEMBRANCE", "BEH_HEART_RELIANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_WORSHIP_REPENTANCE": BehaviorConcept(
        id="BEH_WORSHIP_REPENTANCE",
        arabic="توبة",
        english="repentance",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.MIXED,
        definition_ar="الرجوع إلى الله بترك الذنب والندم عليه والعزم على عدم العودة",
        definition_en="Returning to Allah by abandoning sin, regretting it, and resolving not to return",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_WORSHIP_REMEMBRANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_WORSHIP_GRATITUDE": BehaviorConcept(
        id="BEH_WORSHIP_GRATITUDE",
        arabic="شكر",
        english="gratitude",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.MIXED,
        definition_ar="الاعتراف بنعمة المنعم مع الخضوع له واستعمال النعمة في طاعته",
        definition_en="Acknowledging the blessing of the Bestower with submission and using blessings in obedience",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_WORSHIP_REMEMBRANCE", "BEH_HEART_CONTENTMENT"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    # =========================================================================
    # أخلاق حسنة - GOOD CHARACTER TRAITS
    # =========================================================================
    
    "BEH_SPEECH_TRUTHFULNESS": BehaviorConcept(
        id="BEH_SPEECH_TRUTHFULNESS",
        arabic="صدق",
        english="truthfulness",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.SPEECH_ACT,
        definition_ar="مطابقة القول للواقع والعمل للقول",
        definition_en="Conformity of speech to reality and action to speech",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_SPEECH_LYING",
        related_ids=["BEH_CHAR_TRUSTWORTHINESS", "BEH_HEART_FAITH"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_PATIENCE": BehaviorConcept(
        id="BEH_CHAR_PATIENCE",
        arabic="صبر",
        english="patience",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="حبس النفس على ما تكره ابتغاء مرضاة الله",
        definition_en="Restraining the soul from what it dislikes seeking Allah's pleasure",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_HEART_CONTENTMENT", "BEH_HEART_RELIANCE"],
        systemic_frame=[SystemicAxis.SYS_SELF]
    ),
    
    "BEH_CHAR_JUSTICE": BehaviorConcept(
        id="BEH_CHAR_JUSTICE",
        arabic="عدل",
        english="justice",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.RELATIONAL_ACT,
        definition_ar="إعطاء كل ذي حق حقه",
        definition_en="Giving everyone their due right",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_CHAR_OPPRESSION",
        related_ids=["BEH_CHAR_EXCELLENCE", "BEH_CHAR_TRUSTWORTHINESS"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_MERCY": BehaviorConcept(
        id="BEH_CHAR_MERCY",
        arabic="رحمة",
        english="mercy",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="رقة في القلب تقتضي الإحسان إلى الغير",
        definition_en="Tenderness in the heart that necessitates kindness to others",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_EXCELLENCE", "BEH_CHAR_JUSTICE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_HUMILITY": BehaviorConcept(
        id="BEH_CHAR_HUMILITY",
        arabic="تواضع",
        english="humility",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="خفض الجناح للناس وعدم التعالي عليهم",
        definition_en="Lowering the wing to people and not being arrogant over them",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_HEART_ARROGANCE",
        related_ids=["BEH_CHAR_MODESTY", "BEH_CHAR_MERCY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_MODESTY": BehaviorConcept(
        id="BEH_CHAR_MODESTY",
        arabic="حياء",
        english="modesty",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="انقباض النفس عن القبيح خوفاً من الذم",
        definition_en="Restraint of the soul from ugliness fearing blame",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_HUMILITY"],
        systemic_frame=[SystemicAxis.SYS_SELF]
    ),
    
    "BEH_CHAR_EXCELLENCE": BehaviorConcept(
        id="BEH_CHAR_EXCELLENCE",
        arabic="إحسان",
        english="excellence",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="أن تعبد الله كأنك تراه فإن لم تكن تراه فإنه يراك",
        definition_en="To worship Allah as if you see Him, for if you don't see Him, He sees you",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_JUSTICE", "BEH_CHAR_MERCY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_TRUSTWORTHINESS": BehaviorConcept(
        id="BEH_CHAR_TRUSTWORTHINESS",
        arabic="أمانة",
        english="trustworthiness",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="أداء الحقوق والوفاء بالعهود",
        definition_en="Fulfilling rights and keeping covenants",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_CHAR_BETRAYAL",
        related_ids=["BEH_SPEECH_TRUTHFULNESS", "BEH_CHAR_JUSTICE"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_ASCETICISM": BehaviorConcept(
        id="BEH_CHAR_ASCETICISM",
        arabic="زهد",
        english="asceticism",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="ترك ما لا ينفع في الآخرة",
        definition_en="Abandoning what does not benefit in the Hereafter",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="",
        related_ids=["BEH_HEART_PIETY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_LIFE]
    ),
    
    "BEH_CHAR_FORBEARANCE": BehaviorConcept(
        id="BEH_CHAR_FORBEARANCE",
        arabic="حلم",
        english="forbearance",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="ضبط النفس عند الغضب",
        definition_en="Self-control when angry",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_SALIH,
        opposite_id="BEH_HEART_ANGER",
        related_ids=["BEH_CHAR_PATIENCE"],
        systemic_frame=[SystemicAxis.SYS_SELF]
    ),
    
    # =========================================================================
    # أخلاق سيئة - BAD CHARACTER TRAITS
    # =========================================================================
    
    "BEH_CHAR_DISBELIEF": BehaviorConcept(
        id="BEH_CHAR_DISBELIEF",
        arabic="كفر",
        english="disbelief",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="جحود الحق وستره بعد معرفته",
        definition_en="Denying and concealing truth after knowing it",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_HEART_FAITH",
        related_ids=["BEH_CHAR_POLYTHEISM", "BEH_CHAR_OPPRESSION"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_CHAR_POLYTHEISM": BehaviorConcept(
        id="BEH_CHAR_POLYTHEISM",
        arabic="شرك",
        english="polytheism",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.INNER_STATE,
        definition_ar="جعل شريك لله في ربوبيته أو ألوهيته أو أسمائه وصفاته",
        definition_en="Associating partners with Allah in His lordship, divinity, or names and attributes",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="",
        related_ids=["BEH_CHAR_DISBELIEF", "BEH_CHAR_OPPRESSION"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_CHAR_OPPRESSION": BehaviorConcept(
        id="BEH_CHAR_OPPRESSION",
        arabic="ظلم",
        english="oppression",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.RELATIONAL_ACT,
        definition_ar="وضع الشيء في غير موضعه - تجاوز الحد",
        definition_en="Placing something in other than its place - transgressing limits",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_CHAR_JUSTICE",
        related_ids=["BEH_HEART_ARROGANCE", "BEH_CHAR_DISBELIEF"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_SPEECH_LYING": BehaviorConcept(
        id="BEH_SPEECH_LYING",
        arabic="كذب",
        english="lying",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.SPEECH_ACT,
        definition_ar="الإخبار بخلاف الواقع",
        definition_en="Reporting contrary to reality",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_SPEECH_TRUTHFULNESS",
        related_ids=["BEH_HEART_HYPOCRISY", "BEH_CHAR_BETRAYAL"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_STINGINESS": BehaviorConcept(
        id="BEH_CHAR_STINGINESS",
        arabic="بخل",
        english="stinginess",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="منع الواجب من المال",
        definition_en="Withholding obligatory wealth",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="",
        related_ids=["BEH_HEART_ENVY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_TRANSGRESSION": BehaviorConcept(
        id="BEH_CHAR_TRANSGRESSION",
        arabic="فسق",
        english="transgression",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="الخروج عن طاعة الله",
        definition_en="Departing from obedience to Allah",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_HEART_PIETY",
        related_ids=["BEH_CHAR_DISBELIEF", "BEH_CHAR_IMMORALITY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_GOD]
    ),
    
    "BEH_CHAR_BETRAYAL": BehaviorConcept(
        id="BEH_CHAR_BETRAYAL",
        arabic="خيانة",
        english="betrayal",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.RELATIONAL_ACT,
        definition_ar="نقض العهد والغدر",
        definition_en="Breaking covenant and treachery",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_CHAR_TRUSTWORTHINESS",
        related_ids=["BEH_SPEECH_LYING", "BEH_HEART_HYPOCRISY"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_SPEECH_BACKBITING": BehaviorConcept(
        id="BEH_SPEECH_BACKBITING",
        arabic="غيبة",
        english="backbiting",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.SPEECH_ACT,
        definition_ar="ذكر أخيك بما يكره في غيبته",
        definition_en="Mentioning your brother with what he dislikes in his absence",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="",
        related_ids=["BEH_SPEECH_LYING"],
        systemic_frame=[SystemicAxis.SYS_SELF, SystemicAxis.SYS_CREATION]
    ),
    
    "BEH_CHAR_IMMORALITY": BehaviorConcept(
        id="BEH_CHAR_IMMORALITY",
        arabic="فجور",
        english="immorality",
        category=BehaviorCategory.BEHAVIOR_CONCEPT,
        form=BehaviorForm.TRAIT_DISPOSITION,
        definition_ar="الميل إلى الفساد والانحراف عن الحق",
        definition_en="Inclination toward corruption and deviation from truth",
        is_true_behavior=True,
        action_class=ActionClass.ACT_VOLITIONAL,
        default_eval=ActionTextualEval.EVAL_GHAYR_SALIH,
        opposite_id="BEH_HEART_PIETY",
        related_ids=["BEH_CHAR_TRANSGRESSION", "BEH_CHAR_OPPRESSION"],
        systemic_frame=[SystemicAxis.SYS_SELF]
    ),
    
    # =========================================================================
    # NOT BEHAVIORS - ENTITIES (كيانات)
    # =========================================================================
    
    "ENT_PROPHET": BehaviorConcept(
        id="ENT_PROPHET",
        arabic="نبي",
        english="prophet",
        category=BehaviorCategory.ENTITY,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من أوحي إليه بشرع ولم يؤمر بتبليغه",
        definition_en="One who received revelation but was not commanded to convey it",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "ENT_MESSENGER": BehaviorConcept(
        id="ENT_MESSENGER",
        arabic="رسول",
        english="messenger",
        category=BehaviorCategory.ENTITY,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من أوحي إليه بشرع وأمر بتبليغه",
        definition_en="One who received revelation and was commanded to convey it",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "ENT_ANGEL": BehaviorConcept(
        id="ENT_ANGEL",
        arabic="ملائكة",
        english="angels",
        category=BehaviorCategory.ENTITY,
        form=BehaviorForm.UNKNOWN,
        definition_ar="مخلوقات نورانية لا يعصون الله ما أمرهم",
        definition_en="Creatures of light who do not disobey Allah's commands",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    # =========================================================================
    # NOT BEHAVIORS - STATE LABELS (حالات)
    # =========================================================================
    
    "STL_BELIEVER": BehaviorConcept(
        id="STL_BELIEVER",
        arabic="مؤمن",
        english="believer",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من آمن بالله ورسله - وصف للشخص",
        definition_en="One who believed in Allah and His messengers - person descriptor",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_DISBELIEVER": BehaviorConcept(
        id="STL_DISBELIEVER",
        arabic="كافر",
        english="disbeliever",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من جحد الحق - وصف للشخص",
        definition_en="One who denied truth - person descriptor",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_HYPOCRITE": BehaviorConcept(
        id="STL_HYPOCRITE",
        arabic="منافق",
        english="hypocrite",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من أظهر الإيمان وأبطن الكفر - وصف للشخص",
        definition_en="One who displayed faith while concealing disbelief - person descriptor",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_HEART": BehaviorConcept(
        id="STL_HEART",
        arabic="قلب",
        english="heart",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="محل الإيمان والعقل - عضو",
        definition_en="Seat of faith and intellect - organ",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_SOUND": BehaviorConcept(
        id="STL_SOUND",
        arabic="سليم",
        english="sound",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="صفة للقلب السالم من الشرك والشبهات",
        definition_en="Attribute of heart free from polytheism and doubts",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_DEAD": BehaviorConcept(
        id="STL_DEAD",
        arabic="ميت",
        english="dead",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="من فارقت روحه جسده",
        definition_en="One whose soul departed from body",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_SICK": BehaviorConcept(
        id="STL_SICK",
        arabic="مريض",
        english="sick",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="صفة للقلب المصاب بالشبهات أو الشهوات",
        definition_en="Attribute of heart afflicted with doubts or desires",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_HARD": BehaviorConcept(
        id="STL_HARD",
        arabic="قاسي",
        english="hard",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="صفة للقلب الذي لا يلين للحق",
        definition_en="Attribute of heart that does not soften to truth",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_SEALED": BehaviorConcept(
        id="STL_SEALED",
        arabic="مختوم",
        english="sealed",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="صفة للقلب المطبوع عليه بسبب الكفر",
        definition_en="Attribute of heart sealed due to disbelief",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
    
    "STL_REPENTANT": BehaviorConcept(
        id="STL_REPENTANT",
        arabic="منيب",
        english="repentant",
        category=BehaviorCategory.STATE_LABEL,
        form=BehaviorForm.UNKNOWN,
        definition_ar="الراجع إلى الله - وصف للشخص",
        definition_en="One returning to Allah - person descriptor",
        is_true_behavior=False,
        action_class=ActionClass.ACT_UNKNOWN,
        default_eval=ActionTextualEval.EVAL_NOT_APPLICABLE,
    ),
}


# =============================================================================
# THEMATIC CONSTRUCTS (THM_*)
# =============================================================================

THEMATIC_CONSTRUCTS = {
    "THM_ACCOUNTABILITY": "المحاسبة والمسؤولية",
    "THM_TESTIMONY": "الشهادة",
    "THM_REWARD": "الثواب والجزاء",
    "THM_PUNISHMENT": "العقاب والعذاب",
    "THM_GUIDANCE": "الهداية",
    "THM_MISGUIDANCE": "الضلال",
    "THM_CREATION": "الخلق",
    "THM_RESURRECTION": "البعث والنشور",
    "THM_COVENANT": "العهد والميثاق",
    "THM_TRIAL": "الابتلاء والفتنة",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_true_behaviors() -> List[str]:
    """Get list of TRUE behavior IDs (not entities/states)."""
    return [k for k, v in BEHAVIOR_TAXONOMY.items() if v.is_true_behavior]


def get_non_behaviors() -> List[str]:
    """Get list of non-behavior IDs (entities/states)."""
    return [k for k, v in BEHAVIOR_TAXONOMY.items() if not v.is_true_behavior]


def get_behavior_by_arabic(arabic: str) -> Optional[BehaviorConcept]:
    """Get behavior concept by Arabic term."""
    for concept in BEHAVIOR_TAXONOMY.values():
        if concept.arabic == arabic:
            return concept
    return None


def get_behavior_id_by_arabic(arabic: str) -> Optional[str]:
    """Get behavior ID by Arabic term."""
    for id, concept in BEHAVIOR_TAXONOMY.items():
        if concept.arabic == arabic:
            return id
    return None


def is_true_behavior(arabic_or_id: str) -> bool:
    """Check if label is a true behavior."""
    # Check by ID
    if arabic_or_id in BEHAVIOR_TAXONOMY:
        return BEHAVIOR_TAXONOMY[arabic_or_id].is_true_behavior
    # Check by Arabic
    concept = get_behavior_by_arabic(arabic_or_id)
    return concept.is_true_behavior if concept else False


def get_arabic_to_id_mapping() -> Dict[str, str]:
    """Get mapping from Arabic terms to controlled IDs."""
    return {v.arabic: k for k, v in BEHAVIOR_TAXONOMY.items()}


def print_taxonomy_stats():
    """Print taxonomy statistics."""
    true_behaviors = get_true_behaviors()
    non_behaviors = get_non_behaviors()
    
    print("=" * 70)
    print("QURANIC BEHAVIOR MATRIX - USUL AL-FIQH ALIGNED TAXONOMY")
    print("=" * 70)
    print(f"Total concepts: {len(BEHAVIOR_TAXONOMY)}")
    print(f"TRUE behaviors (BEH_*): {len(true_behaviors)}")
    print(f"Non-behaviors (ENT_*, STL_*): {len(non_behaviors)}")
    print()
    
    # Group by category
    categories = {}
    for id, concept in BEHAVIOR_TAXONOMY.items():
        cat = concept.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((id, concept))
    
    for cat, items in sorted(categories.items()):
        print(f"\n{cat} ({len(items)} items):")
        for id, concept in items:
            eval_str = concept.default_eval.value if concept.default_eval else "N/A"
            print(f"  {id}: {concept.arabic} ({concept.english}) - {eval_str}")


if __name__ == "__main__":
    print_taxonomy_stats()
