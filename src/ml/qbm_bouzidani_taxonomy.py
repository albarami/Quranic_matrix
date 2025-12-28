"""
QBM Taxonomy Based on Bouzidani's Framework
"السلوك البشري في سياقه القرآني"

Key Principles from the Paper:
1. موضوعية الأخلاق (Objective Ethics) - Divine guidance as standard
2. فطرة البصيرة (Fitrah-based Moral Insight) - Innate moral capacity
3. العبودية (Servitude) - Behavior oriented toward worship

Three Worlds Model (Ibn Khaldun):
- العالم الحسي (Sensory World) - Physical behaviors
- العالم النفسي (Psychic World) - Thoughts, emotions
- العالم الروحي (Spiritual World) - Intentions, worship

Three Types of Behavior (عمل):
1. غريزي/لا إرادي (Instinctive/Automatic) - Not morally evaluated
2. عمل صالح (Righteous Action) - Praised, rewarded
3. عمل سيء/غير صالح (Unrighteous Action) - Blamed, punished

Five Contexts (السياقات الخمسة):
1. العضوي (Organic) - Body organs
2. الموضعي (Situational) - Internal/External
3. النسقي (Systemic) - Social systems
4. الزماني (Temporal) - Time dimension
5. المكاني (Spatial) - Location

The Role of النية (Intention):
- Transforms routine acts into worship
- "إنما الأعمال بالنية"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
VOCAB_DIR = PROJECT_ROOT / "vocab"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# ENUMS - Based on Bouzidani's Framework
# =============================================================================

class ActionClass(Enum):
    """Three types of behavior from Bouzidani's paper."""
    GHARIZI = "غريزي/لا إرادي"      # Instinctive/Automatic - not morally evaluated
    IRADI = "إرادي/مكتسب"           # Volitional/Acquired - subject to moral evaluation
    UNKNOWN = "غير معروف"


class ActionEvaluation(Enum):
    """Moral evaluation of volitional actions."""
    AMAL_SALIH = "عمل صالح"          # Righteous action - praised, rewarded
    AMAL_SAYYI = "عمل سيء"           # Unrighteous action - blamed, punished
    NEUTRAL = "محايد"                # Neutral - depends on intention
    NOT_APPLICABLE = "غير منطبق"     # For instinctive behaviors
    UNKNOWN = "غير معروف"


class BehaviorForm(Enum):
    """Form of behavior (from situational context)."""
    QAWLI = "قولي"                   # Speech act (فعل كلامي)
    FI3LI = "فعلي"                   # Physical act (فعل جسدي)
    BATINI = "باطني"                 # Inner state (حالة داخلية)
    SIFA = "صفة"                     # Trait/disposition (سمة)
    ILAQI = "علائقي"                 # Relational act (فعل علائقي)
    MURAKKAB = "مركب"                # Mixed/compound
    UNKNOWN = "غير معروف"


class OrganicContext(Enum):
    """Organic context - body organs involved."""
    QALB = "قلب"                     # Heart - central organ
    LISAN = "لسان"                   # Tongue - speech
    AYN = "عين"                      # Eye - vision
    UDHN = "أذن"                     # Ear - hearing
    YAD = "يد"                       # Hand - action
    RIJL = "رجل"                     # Foot - movement
    JILD = "جلد"                     # Skin - witness
    SADR = "صدر"                     # Chest - emotions
    NAFS = "نفس"                     # Soul/self
    UNKNOWN = "غير معروف"


class SituationalContext(Enum):
    """Situational context - internal vs external."""
    ZAHIR = "ظاهر"                   # External/manifest
    BATIN = "باطن"                   # Internal/hidden
    MURAKKAB = "مركب"                # Both internal and external
    UNKNOWN = "غير معروف"


class SystemicContext(Enum):
    """Systemic context - social systems."""
    NAFS = "النفس"                   # Self
    KHALQ = "الخلق"                  # Creation/others
    KHALIQ = "الخالق"                # Creator (Allah)
    USRA = "الأسرة"                  # Family
    MUJTAMA = "المجتمع"              # Society
    UNKNOWN = "غير معروف"


class TemporalContext(Enum):
    """Temporal context - time dimension."""
    ANI = "آني"                      # Immediate/instant
    MUTAKARRIR = "متكرر"             # Repeated/habitual
    DAIM = "دائم"                    # Permanent/constant
    MUSTAQBALI = "مستقبلي"           # Future-oriented (intention)
    UNKNOWN = "غير معروف"


class SpatialContext(Enum):
    """Spatial context - location."""
    MASJID = "مسجد"                  # Mosque
    BAYT = "بيت"                     # Home
    SUQ = "سوق"                      # Market/public
    UNKNOWN = "غير معروف"


# =============================================================================
# BEHAVIOR CATEGORIES (Based on Bouzidani's Classification)
# =============================================================================

class BehaviorCategory(Enum):
    """Categories of behavior based on Bouzidani's framework."""
    # Worship behaviors (عبادات)
    IBADA_SHU3URIYA = "عبادة شعورية"      # Emotional worship (خوف، رجاء، محبة)
    IBADA_FIKRIYA = "عبادة فكرية"         # Cognitive worship (تدبر، تفكر، تأمل)
    IBADA_SULUKIYA = "عبادة سلوكية"       # Behavioral worship (صلاة، صيام، زكاة)
    
    # Heart diseases (أمراض القلوب)
    MARAD_QALB = "مرض قلب"
    
    # Heart virtues (فضائل القلوب)
    FADILA_QALB = "فضيلة قلب"
    
    # Good character (أخلاق حسنة)
    KHULUQ_HASAN = "خلق حسن"
    
    # Bad character (أخلاق سيئة)
    KHULUQ_SAYYI = "خلق سيء"
    
    # Entity mention (not a behavior)
    ENTITY = "كيان"
    
    # State label (not a behavior)
    STATE = "حالة"


# =============================================================================
# BEHAVIOR DEFINITION
# =============================================================================

@dataclass
class BehaviorDefinition:
    """Complete behavior definition based on Bouzidani's framework."""
    id: str                              # Controlled ID (BEH_*)
    arabic: str                          # Arabic term
    english: str                         # English translation
    
    # Classification
    category: BehaviorCategory           # Category
    is_true_behavior: bool               # True = actual behavior
    
    # Action classification (from paper)
    action_class: ActionClass            # غريزي vs إرادي
    action_eval: ActionEvaluation        # صالح vs سيء
    
    # Form (situational)
    form: BehaviorForm                   # قولي، فعلي، باطني
    situational: SituationalContext      # ظاهر vs باطن
    
    # Organic context
    primary_organ: OrganicContext        # Primary organ involved
    
    # Systemic context
    systemic: List[SystemicContext] = field(default_factory=list)
    
    # Definitions
    definition_ar: str = ""
    definition_en: str = ""
    
    # Relations
    opposite_id: str = ""
    related_ids: List[str] = field(default_factory=list)
    
    # Quranic evidence
    quranic_roots: List[str] = field(default_factory=list)


# =============================================================================
# COMPLETE TAXONOMY
# =============================================================================

BOUZIDANI_TAXONOMY: Dict[str, BehaviorDefinition] = {
    
    # =========================================================================
    # عبادات شعورية - EMOTIONAL WORSHIP (Inner states toward Allah)
    # =========================================================================
    
    "BEH_IBADA_KHAWF": BehaviorDefinition(
        id="BEH_IBADA_KHAWF",
        arabic="خوف من الله",
        english="fear_of_Allah",
        category=BehaviorCategory.IBADA_SHU3URIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="خشية الله تعالى والخوف من عقابه",
        quranic_roots=["خ-و-ف", "خ-ش-ي"]
    ),
    
    "BEH_IBADA_RAJA": BehaviorDefinition(
        id="BEH_IBADA_RAJA",
        arabic="رجاء",
        english="hope_in_Allah",
        category=BehaviorCategory.IBADA_SHU3URIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="رجاء رحمة الله وثوابه",
        quranic_roots=["ر-ج-و"]
    ),
    
    "BEH_IBADA_MAHABBA": BehaviorDefinition(
        id="BEH_IBADA_MAHABBA",
        arabic="محبة الله",
        english="love_of_Allah",
        category=BehaviorCategory.IBADA_SHU3URIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="حب الله تعالى فوق كل شيء",
        quranic_roots=["ح-ب-ب"]
    ),
    
    # =========================================================================
    # عبادات فكرية - COGNITIVE WORSHIP
    # =========================================================================
    
    "BEH_IBADA_TADABBUR": BehaviorDefinition(
        id="BEH_IBADA_TADABBUR",
        arabic="تدبر",
        english="contemplation",
        category=BehaviorCategory.IBADA_FIKRIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="التأمل في آيات الله وكتابه",
        quranic_roots=["د-ب-ر"]
    ),
    
    "BEH_IBADA_TAFAKKUR": BehaviorDefinition(
        id="BEH_IBADA_TAFAKKUR",
        arabic="تفكر",
        english="reflection",
        category=BehaviorCategory.IBADA_FIKRIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="التفكر في خلق السماوات والأرض",
        quranic_roots=["ف-ك-ر"]
    ),
    
    # =========================================================================
    # عبادات سلوكية - BEHAVIORAL WORSHIP
    # =========================================================================
    
    "BEH_IBADA_DHIKR": BehaviorDefinition(
        id="BEH_IBADA_DHIKR",
        arabic="ذكر",
        english="remembrance",
        category=BehaviorCategory.IBADA_SULUKIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.MURAKKAB,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="ذكر الله باللسان والقلب",
        opposite_id="BEH_MARAD_GHAFLA",
        quranic_roots=["ذ-ك-ر"]
    ),
    
    "BEH_IBADA_DUA": BehaviorDefinition(
        id="BEH_IBADA_DUA",
        arabic="دعاء",
        english="supplication",
        category=BehaviorCategory.IBADA_SULUKIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.QAWLI,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.LISAN,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="طلب العبد من ربه بخضوع",
        quranic_roots=["د-ع-و"]
    ),
    
    "BEH_IBADA_TAWBA": BehaviorDefinition(
        id="BEH_IBADA_TAWBA",
        arabic="توبة",
        english="repentance",
        category=BehaviorCategory.IBADA_SULUKIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.MURAKKAB,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="الرجوع إلى الله بترك الذنب والندم",
        quranic_roots=["ت-و-ب"]
    ),
    
    "BEH_IBADA_SHUKR": BehaviorDefinition(
        id="BEH_IBADA_SHUKR",
        arabic="شكر",
        english="gratitude",
        category=BehaviorCategory.IBADA_SULUKIYA,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.MURAKKAB,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="الاعتراف بنعمة المنعم مع الخضوع",
        quranic_roots=["ش-ك-ر"]
    ),
    
    # =========================================================================
    # فضائل القلوب - HEART VIRTUES
    # =========================================================================
    
    "BEH_FADILA_IMAN": BehaviorDefinition(
        id="BEH_FADILA_IMAN",
        arabic="إيمان",
        english="faith",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="التصديق بالله ورسله واليوم الآخر",
        opposite_id="BEH_MARAD_KUFR",
        quranic_roots=["أ-م-ن"]
    ),
    
    "BEH_FADILA_TAQWA": BehaviorDefinition(
        id="BEH_FADILA_TAQWA",
        arabic="تقوى",
        english="piety",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="حفظ النفس مما يؤثم",
        opposite_id="BEH_MARAD_FISQ",
        quranic_roots=["و-ق-ي"]
    ),
    
    "BEH_FADILA_TAWAKKUL": BehaviorDefinition(
        id="BEH_FADILA_TAWAKKUL",
        arabic="توكل",
        english="reliance",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="الاعتماد على الله مع الأخذ بالأسباب",
        quranic_roots=["و-ك-ل"]
    ),
    
    "BEH_FADILA_RIDA": BehaviorDefinition(
        id="BEH_FADILA_RIDA",
        arabic="رضا",
        english="contentment",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="طمأنينة القلب بقضاء الله",
        quranic_roots=["ر-ض-ي"]
    ),
    
    "BEH_FADILA_IKHLAS": BehaviorDefinition(
        id="BEH_FADILA_IKHLAS",
        arabic="إخلاص",
        english="sincerity",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="تصفية العمل من كل شائبة لغير الله",
        opposite_id="BEH_MARAD_RIYA",
        quranic_roots=["خ-ل-ص"]
    ),
    
    "BEH_FADILA_KHUSHU": BehaviorDefinition(
        id="BEH_FADILA_KHUSHU",
        arabic="خشوع",
        english="humility_prayer",
        category=BehaviorCategory.FADILA_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="سكون القلب وخضوعه لله",
        quranic_roots=["خ-ش-ع"]
    ),
    
    # =========================================================================
    # أمراض القلوب - HEART DISEASES
    # =========================================================================
    
    "BEH_MARAD_KIBR": BehaviorDefinition(
        id="BEH_MARAD_KIBR",
        arabic="كبر",
        english="arrogance",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="بطر الحق وغمط الناس",
        opposite_id="BEH_KHULUQ_TAWADU",
        quranic_roots=["ك-ب-ر"]
    ),
    
    "BEH_MARAD_HASAD": BehaviorDefinition(
        id="BEH_MARAD_HASAD",
        arabic="حسد",
        english="envy",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="تمني زوال النعمة عن الغير",
        quranic_roots=["ح-س-د"]
    ),
    
    "BEH_MARAD_GHAFLA": BehaviorDefinition(
        id="BEH_MARAD_GHAFLA",
        arabic="غفلة",
        english="heedlessness",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="الإعراض عن ذكر الله والآخرة",
        opposite_id="BEH_IBADA_DHIKR",
        quranic_roots=["غ-ف-ل"]
    ),
    
    "BEH_MARAD_NIFAQ": BehaviorDefinition(
        id="BEH_MARAD_NIFAQ",
        arabic="نفاق",
        english="hypocrisy",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="إظهار الإيمان وإبطان الكفر",
        quranic_roots=["ن-ف-ق"]
    ),
    
    "BEH_MARAD_RIYA": BehaviorDefinition(
        id="BEH_MARAD_RIYA",
        arabic="رياء",
        english="showing_off",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="العمل لأجل الناس لا لله",
        opposite_id="BEH_FADILA_IKHLAS",
        quranic_roots=["ر-أ-ي"]
    ),
    
    "BEH_MARAD_GHADAB": BehaviorDefinition(
        id="BEH_MARAD_GHADAB",
        arabic="غضب",
        english="anger",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.NEUTRAL,  # Context-dependent
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="ثوران النفس لدفع المكروه",
        quranic_roots=["غ-ض-ب"]
    ),
    
    "BEH_MARAD_KUFR": BehaviorDefinition(
        id="BEH_MARAD_KUFR",
        arabic="كفر",
        english="disbelief",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="جحود الحق وستره بعد معرفته",
        opposite_id="BEH_FADILA_IMAN",
        quranic_roots=["ك-ف-ر"]
    ),
    
    "BEH_MARAD_SHIRK": BehaviorDefinition(
        id="BEH_MARAD_SHIRK",
        arabic="شرك",
        english="polytheism",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.BATINI,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="جعل شريك لله في ربوبيته أو ألوهيته",
        quranic_roots=["ش-ر-ك"]
    ),
    
    "BEH_MARAD_FISQ": BehaviorDefinition(
        id="BEH_MARAD_FISQ",
        arabic="فسق",
        english="transgression",
        category=BehaviorCategory.MARAD_QALB,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ],
        definition_ar="الخروج عن طاعة الله",
        opposite_id="BEH_FADILA_TAQWA",
        quranic_roots=["ف-س-ق"]
    ),
    
    # =========================================================================
    # أخلاق حسنة - GOOD CHARACTER
    # =========================================================================
    
    "BEH_KHULUQ_SIDQ": BehaviorDefinition(
        id="BEH_KHULUQ_SIDQ",
        arabic="صدق",
        english="truthfulness",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.QAWLI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.LISAN,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="مطابقة القول للواقع",
        opposite_id="BEH_KHULUQ_KIDHB",
        quranic_roots=["ص-د-ق"]
    ),
    
    "BEH_KHULUQ_SABR": BehaviorDefinition(
        id="BEH_KHULUQ_SABR",
        arabic="صبر",
        english="patience",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="حبس النفس على ما تكره",
        quranic_roots=["ص-ب-ر"]
    ),
    
    "BEH_KHULUQ_ADL": BehaviorDefinition(
        id="BEH_KHULUQ_ADL",
        arabic="عدل",
        english="justice",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.ILAQI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="إعطاء كل ذي حق حقه",
        opposite_id="BEH_KHULUQ_DHULM",
        quranic_roots=["ع-د-ل"]
    ),
    
    "BEH_KHULUQ_RAHMA": BehaviorDefinition(
        id="BEH_KHULUQ_RAHMA",
        arabic="رحمة",
        english="mercy",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="رقة في القلب تقتضي الإحسان",
        quranic_roots=["ر-ح-م"]
    ),
    
    "BEH_KHULUQ_TAWADU": BehaviorDefinition(
        id="BEH_KHULUQ_TAWADU",
        arabic="تواضع",
        english="humility",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="خفض الجناح للناس",
        opposite_id="BEH_MARAD_KIBR",
        quranic_roots=["و-ض-ع"]
    ),
    
    "BEH_KHULUQ_HAYA": BehaviorDefinition(
        id="BEH_KHULUQ_HAYA",
        arabic="حياء",
        english="modesty",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="انقباض النفس عن القبيح",
        quranic_roots=["ح-ي-ي"]
    ),
    
    "BEH_KHULUQ_IHSAN": BehaviorDefinition(
        id="BEH_KHULUQ_IHSAN",
        arabic="إحسان",
        english="excellence",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALIQ, SystemicContext.KHALQ],
        definition_ar="أن تعبد الله كأنك تراه",
        quranic_roots=["ح-س-ن"]
    ),
    
    "BEH_KHULUQ_AMANA": BehaviorDefinition(
        id="BEH_KHULUQ_AMANA",
        arabic="أمانة",
        english="trustworthiness",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="أداء الحقوق والوفاء بالعهود",
        opposite_id="BEH_KHULUQ_KHIYANA",
        quranic_roots=["أ-م-ن"]
    ),
    
    "BEH_KHULUQ_ZUHD": BehaviorDefinition(
        id="BEH_KHULUQ_ZUHD",
        arabic="زهد",
        english="asceticism",
        category=BehaviorCategory.KHULUQ_HASAN,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SALIH,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.BATIN,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="ترك ما لا ينفع في الآخرة",
        quranic_roots=["ز-ه-د"]
    ),
    
    # =========================================================================
    # أخلاق سيئة - BAD CHARACTER
    # =========================================================================
    
    "BEH_KHULUQ_KIDHB": BehaviorDefinition(
        id="BEH_KHULUQ_KIDHB",
        arabic="كذب",
        english="lying",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.QAWLI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.LISAN,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="الإخبار بخلاف الواقع",
        opposite_id="BEH_KHULUQ_SIDQ",
        quranic_roots=["ك-ذ-ب"]
    ),
    
    "BEH_KHULUQ_DHULM": BehaviorDefinition(
        id="BEH_KHULUQ_DHULM",
        arabic="ظلم",
        english="oppression",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.ILAQI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="وضع الشيء في غير موضعه",
        opposite_id="BEH_KHULUQ_ADL",
        quranic_roots=["ظ-ل-م"]
    ),
    
    "BEH_KHULUQ_BUKHL": BehaviorDefinition(
        id="BEH_KHULUQ_BUKHL",
        arabic="بخل",
        english="stinginess",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="منع الواجب من المال",
        quranic_roots=["ب-خ-ل"]
    ),
    
    "BEH_KHULUQ_KHIYANA": BehaviorDefinition(
        id="BEH_KHULUQ_KHIYANA",
        arabic="خيانة",
        english="betrayal",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.ILAQI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="نقض العهد والغدر",
        opposite_id="BEH_KHULUQ_AMANA",
        quranic_roots=["خ-و-ن"]
    ),
    
    "BEH_KHULUQ_GHIBA": BehaviorDefinition(
        id="BEH_KHULUQ_GHIBA",
        arabic="غيبة",
        english="backbiting",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.QAWLI,
        situational=SituationalContext.ZAHIR,
        primary_organ=OrganicContext.LISAN,
        systemic=[SystemicContext.NAFS, SystemicContext.KHALQ],
        definition_ar="ذكر أخيك بما يكره في غيبته",
        quranic_roots=["غ-ي-ب"]
    ),
    
    "BEH_KHULUQ_FUJUR": BehaviorDefinition(
        id="BEH_KHULUQ_FUJUR",
        arabic="فجور",
        english="immorality",
        category=BehaviorCategory.KHULUQ_SAYYI,
        is_true_behavior=True,
        action_class=ActionClass.IRADI,
        action_eval=ActionEvaluation.AMAL_SAYYI,
        form=BehaviorForm.SIFA,
        situational=SituationalContext.MURAKKAB,
        primary_organ=OrganicContext.QALB,
        systemic=[SystemicContext.NAFS],
        definition_ar="الميل إلى الفساد والانحراف",
        quranic_roots=["ف-ج-ر"]
    ),
    
    # =========================================================================
    # NON-BEHAVIORS - ENTITIES
    # =========================================================================
    
    "ENT_NABI": BehaviorDefinition(
        id="ENT_NABI",
        arabic="نبي",
        english="prophet",
        category=BehaviorCategory.ENTITY,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.UNKNOWN,
        definition_ar="من أوحي إليه بشرع"
    ),
    
    "ENT_RASUL": BehaviorDefinition(
        id="ENT_RASUL",
        arabic="رسول",
        english="messenger",
        category=BehaviorCategory.ENTITY,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.UNKNOWN,
        definition_ar="من أوحي إليه وأمر بالتبليغ"
    ),
    
    "ENT_MALAK": BehaviorDefinition(
        id="ENT_MALAK",
        arabic="ملائكة",
        english="angels",
        category=BehaviorCategory.ENTITY,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.UNKNOWN,
        definition_ar="مخلوقات نورانية"
    ),
    
    # =========================================================================
    # NON-BEHAVIORS - STATES
    # =========================================================================
    
    "STATE_MUMIN": BehaviorDefinition(
        id="STATE_MUMIN",
        arabic="مؤمن",
        english="believer",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="من آمن بالله - وصف للشخص"
    ),
    
    "STATE_KAFIR": BehaviorDefinition(
        id="STATE_KAFIR",
        arabic="كافر",
        english="disbeliever",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="من جحد الحق - وصف للشخص"
    ),
    
    "STATE_MUNAFIQ": BehaviorDefinition(
        id="STATE_MUNAFIQ",
        arabic="منافق",
        english="hypocrite",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="من أظهر الإيمان وأبطن الكفر - وصف"
    ),
    
    "STATE_QALB": BehaviorDefinition(
        id="STATE_QALB",
        arabic="قلب",
        english="heart",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="محل الإيمان والعقل - عضو"
    ),
    
    "STATE_SALIM": BehaviorDefinition(
        id="STATE_SALIM",
        arabic="سليم",
        english="sound",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="صفة للقلب السالم"
    ),
    
    "STATE_MAYYIT": BehaviorDefinition(
        id="STATE_MAYYIT",
        arabic="ميت",
        english="dead",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.UNKNOWN,
        definition_ar="من فارقت روحه جسده"
    ),
    
    "STATE_MARID": BehaviorDefinition(
        id="STATE_MARID",
        arabic="مريض",
        english="sick",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="صفة للقلب المصاب"
    ),
    
    "STATE_QASI": BehaviorDefinition(
        id="STATE_QASI",
        arabic="قاسي",
        english="hard",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="صفة للقلب الذي لا يلين"
    ),
    
    "STATE_MAKHTUM": BehaviorDefinition(
        id="STATE_MAKHTUM",
        arabic="مختوم",
        english="sealed",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="صفة للقلب المطبوع عليه"
    ),
    
    "STATE_MUNIB": BehaviorDefinition(
        id="STATE_MUNIB",
        arabic="منيب",
        english="repentant",
        category=BehaviorCategory.STATE,
        is_true_behavior=False,
        action_class=ActionClass.UNKNOWN,
        action_eval=ActionEvaluation.NOT_APPLICABLE,
        form=BehaviorForm.UNKNOWN,
        situational=SituationalContext.UNKNOWN,
        primary_organ=OrganicContext.QALB,
        definition_ar="الراجع إلى الله - وصف"
    ),
}


# =============================================================================
# ARABIC TO ID MAPPING
# =============================================================================

ARABIC_TO_BOUZIDANI_ID: Dict[str, str] = {
    defn.arabic: id for id, defn in BOUZIDANI_TAXONOMY.items()
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_true_behaviors() -> List[str]:
    """Get list of TRUE behavior IDs."""
    return [k for k, v in BOUZIDANI_TAXONOMY.items() if v.is_true_behavior]


def get_non_behaviors() -> List[str]:
    """Get list of non-behavior IDs."""
    return [k for k, v in BOUZIDANI_TAXONOMY.items() if not v.is_true_behavior]


def get_behavior_by_arabic(arabic: str) -> Optional[BehaviorDefinition]:
    """Get behavior definition by Arabic term."""
    id = ARABIC_TO_BOUZIDANI_ID.get(arabic)
    if id:
        return BOUZIDANI_TAXONOMY.get(id)
    return None


def is_true_behavior(arabic: str) -> bool:
    """Check if Arabic term is a true behavior."""
    defn = get_behavior_by_arabic(arabic)
    return defn.is_true_behavior if defn else False


def print_taxonomy_stats():
    """Print taxonomy statistics."""
    true_behaviors = get_true_behaviors()
    non_behaviors = get_non_behaviors()
    
    print("=" * 70)
    print("BOUZIDANI TAXONOMY - السلوك البشري في سياقه القرآني")
    print("=" * 70)
    print(f"Total concepts: {len(BOUZIDANI_TAXONOMY)}")
    print(f"TRUE behaviors: {len(true_behaviors)}")
    print(f"Non-behaviors: {len(non_behaviors)}")
    print()
    
    # Group by category
    categories = {}
    for id, defn in BOUZIDANI_TAXONOMY.items():
        cat = defn.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((id, defn))
    
    for cat, items in sorted(categories.items()):
        print(f"\n{cat} ({len(items)} items):")
        for id, defn in items:
            eval_str = defn.action_eval.value
            form_str = defn.form.value
            print(f"  {defn.arabic:15} -> {id:25} [{form_str:10}] [{eval_str}]")


if __name__ == "__main__":
    print_taxonomy_stats()
