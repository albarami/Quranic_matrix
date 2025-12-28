"""
Quranic Behavioral Matrix (QBM) - Proper Behavior Taxonomy

Based on Islamic psychology and Quranic sciences (علم النفس القرآني):
- Behaviors are actions, states of the heart, or moral qualities
- NOT entity mentions (prophet, messenger, angels)
- NOT person labels (believer, disbeliever)
- NOT body parts or adjectives

Categories:
1. أمراض القلوب (Heart Diseases) - Negative spiritual states
2. فضائل القلوب (Heart Virtues) - Positive spiritual states  
3. أعمال العبادة (Acts of Worship) - Spiritual behaviors
4. أخلاق حسنة (Good Character) - Virtuous behaviors
5. أخلاق سيئة (Bad Character) - Sinful behaviors
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class BehaviorCategory(Enum):
    """Categories of Quranic behaviors."""
    HEART_DISEASE = "أمراض القلوب"      # Negative heart states
    HEART_VIRTUE = "فضائل القلوب"       # Positive heart states
    WORSHIP = "أعمال العبادة"           # Acts of worship
    GOOD_CHARACTER = "أخلاق حسنة"       # Good moral traits
    BAD_CHARACTER = "أخلاق سيئة"        # Bad moral traits
    ENTITY = "كيانات"                   # NOT a behavior - entity mention
    STATE = "حالات"                     # NOT a behavior - person state


@dataclass
class BehaviorDefinition:
    """Definition of a Quranic behavior."""
    arabic: str
    english: str
    category: BehaviorCategory
    is_behavior: bool
    definition_ar: str
    opposite: str = ""
    related: List[str] = None


# =============================================================================
# COMPLETE TAXONOMY
# =============================================================================

BEHAVIOR_TAXONOMY: Dict[str, BehaviorDefinition] = {
    # =========================================================================
    # TRUE BEHAVIORS - أمراض القلوب (Heart Diseases)
    # =========================================================================
    "كبر": BehaviorDefinition(
        arabic="كبر",
        english="arrogance",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="رؤية النفس فوق الآخرين وازدراء الحق",
        opposite="تواضع",
        related=["ظلم", "كفر", "غفلة"]
    ),
    "حسد": BehaviorDefinition(
        arabic="حسد",
        english="envy",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="تمني زوال النعمة عن الغير",
        opposite="",
        related=["بخل", "غضب"]
    ),
    "غفلة": BehaviorDefinition(
        arabic="غفلة",
        english="heedlessness",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="الإعراض عن ذكر الله والآخرة",
        opposite="ذكر",
        related=["كبر", "فسق"]
    ),
    "نفاق": BehaviorDefinition(
        arabic="نفاق",
        english="hypocrisy",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="إظهار الإيمان وإبطان الكفر",
        opposite="إخلاص",
        related=["كذب", "رياء"]
    ),
    "رياء": BehaviorDefinition(
        arabic="رياء",
        english="showing_off",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="العمل لأجل الناس لا لله",
        opposite="إخلاص",
        related=["نفاق", "كبر"]
    ),
    "غضب": BehaviorDefinition(
        arabic="غضب",
        english="anger",
        category=BehaviorCategory.HEART_DISEASE,
        is_behavior=True,
        definition_ar="ثوران النفس لدفع المكروه",
        opposite="حلم",
        related=["ظلم", "فجور"]
    ),
    
    # =========================================================================
    # TRUE BEHAVIORS - فضائل القلوب (Heart Virtues)
    # =========================================================================
    "إيمان": BehaviorDefinition(
        arabic="إيمان",
        english="faith",
        category=BehaviorCategory.HEART_VIRTUE,
        is_behavior=True,
        definition_ar="التصديق بالله ورسله واليوم الآخر مع العمل",
        opposite="كفر",
        related=["تقوى", "صدق", "توكل"]
    ),
    "تقوى": BehaviorDefinition(
        arabic="تقوى",
        english="piety",
        category=BehaviorCategory.HEART_VIRTUE,
        is_behavior=True,
        definition_ar="حفظ النفس مما يؤثم بفعل الواجبات وترك المحرمات",
        opposite="فسق",
        related=["إيمان", "خشوع"]
    ),
    "توكل": BehaviorDefinition(
        arabic="توكل",
        english="reliance",
        category=BehaviorCategory.HEART_VIRTUE,
        is_behavior=True,
        definition_ar="الاعتماد على الله مع الأخذ بالأسباب",
        opposite="",
        related=["إيمان", "صبر"]
    ),
    "رضا": BehaviorDefinition(
        arabic="رضا",
        english="contentment",
        category=BehaviorCategory.HEART_VIRTUE,
        is_behavior=True,
        definition_ar="طمأنينة القلب بقضاء الله",
        opposite="سخط",
        related=["صبر", "شكر"]
    ),
    "خشوع": BehaviorDefinition(
        arabic="خشوع",
        english="humility_prayer",
        category=BehaviorCategory.HEART_VIRTUE,
        is_behavior=True,
        definition_ar="سكون القلب وخضوعه لله",
        opposite="غفلة",
        related=["تقوى", "ذكر"]
    ),
    
    # =========================================================================
    # TRUE BEHAVIORS - أعمال العبادة (Acts of Worship)
    # =========================================================================
    "ذكر": BehaviorDefinition(
        arabic="ذكر",
        english="remembrance",
        category=BehaviorCategory.WORSHIP,
        is_behavior=True,
        definition_ar="استحضار عظمة الله في القلب واللسان",
        opposite="غفلة",
        related=["دعاء", "شكر", "توبة"]
    ),
    "دعاء": BehaviorDefinition(
        arabic="دعاء",
        english="supplication",
        category=BehaviorCategory.WORSHIP,
        is_behavior=True,
        definition_ar="طلب العبد من ربه بخضوع وتضرع",
        opposite="",
        related=["ذكر", "توكل"]
    ),
    "توبة": BehaviorDefinition(
        arabic="توبة",
        english="repentance",
        category=BehaviorCategory.WORSHIP,
        is_behavior=True,
        definition_ar="الرجوع إلى الله بترك الذنب والندم عليه",
        opposite="إصرار",
        related=["ذكر", "استغفار"]
    ),
    "شكر": BehaviorDefinition(
        arabic="شكر",
        english="gratitude",
        category=BehaviorCategory.WORSHIP,
        is_behavior=True,
        definition_ar="الاعتراف بنعمة المنعم مع الخضوع له",
        opposite="كفران",
        related=["ذكر", "رضا"]
    ),
    
    # =========================================================================
    # TRUE BEHAVIORS - أخلاق حسنة (Good Character)
    # =========================================================================
    "صدق": BehaviorDefinition(
        arabic="صدق",
        english="truthfulness",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="مطابقة القول للواقع والعمل للقول",
        opposite="كذب",
        related=["أمانة", "إيمان"]
    ),
    "صبر": BehaviorDefinition(
        arabic="صبر",
        english="patience",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="حبس النفس على ما تكره",
        opposite="جزع",
        related=["رضا", "توكل"]
    ),
    "عدل": BehaviorDefinition(
        arabic="عدل",
        english="justice",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="إعطاء كل ذي حق حقه",
        opposite="ظلم",
        related=["إحسان", "أمانة"]
    ),
    "رحمة": BehaviorDefinition(
        arabic="رحمة",
        english="mercy",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="رقة في القلب تقتضي الإحسان",
        opposite="قسوة",
        related=["إحسان", "عدل"]
    ),
    "تواضع": BehaviorDefinition(
        arabic="تواضع",
        english="humility",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="خفض الجناح للناس وعدم التعالي",
        opposite="كبر",
        related=["حياء", "رحمة"]
    ),
    "حياء": BehaviorDefinition(
        arabic="حياء",
        english="modesty",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="انقباض النفس عن القبيح",
        opposite="وقاحة",
        related=["تواضع", "عفة"]
    ),
    "إحسان": BehaviorDefinition(
        arabic="إحسان",
        english="excellence",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="أن تعبد الله كأنك تراه",
        opposite="إساءة",
        related=["عدل", "رحمة"]
    ),
    "أمانة": BehaviorDefinition(
        arabic="أمانة",
        english="trustworthiness",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="أداء الحقوق والوفاء بالعهود",
        opposite="خيانة",
        related=["صدق", "عدل"]
    ),
    "زهد": BehaviorDefinition(
        arabic="زهد",
        english="asceticism",
        category=BehaviorCategory.GOOD_CHARACTER,
        is_behavior=True,
        definition_ar="ترك ما لا ينفع في الآخرة",
        opposite="طمع",
        related=["تقوى", "قناعة"]
    ),
    
    # =========================================================================
    # TRUE BEHAVIORS - أخلاق سيئة (Bad Character)
    # =========================================================================
    "كفر": BehaviorDefinition(
        arabic="كفر",
        english="disbelief",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="جحود الحق وستره بعد معرفته",
        opposite="إيمان",
        related=["شرك", "ظلم", "كبر"]
    ),
    "شرك": BehaviorDefinition(
        arabic="شرك",
        english="polytheism",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="جعل شريك لله في ربوبيته أو ألوهيته",
        opposite="توحيد",
        related=["كفر", "ظلم"]
    ),
    "ظلم": BehaviorDefinition(
        arabic="ظلم",
        english="oppression",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="وضع الشيء في غير موضعه",
        opposite="عدل",
        related=["كبر", "كفر"]
    ),
    "كذب": BehaviorDefinition(
        arabic="كذب",
        english="lying",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="الإخبار بخلاف الواقع",
        opposite="صدق",
        related=["نفاق", "خيانة"]
    ),
    "بخل": BehaviorDefinition(
        arabic="بخل",
        english="stinginess",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="منع الواجب من المال",
        opposite="كرم",
        related=["حسد", "طمع"]
    ),
    "فسق": BehaviorDefinition(
        arabic="فسق",
        english="transgression",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="الخروج عن طاعة الله",
        opposite="تقوى",
        related=["كفر", "فجور"]
    ),
    "خيانة": BehaviorDefinition(
        arabic="خيانة",
        english="betrayal",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="نقض العهد والغدر",
        opposite="أمانة",
        related=["كذب", "نفاق"]
    ),
    "غيبة": BehaviorDefinition(
        arabic="غيبة",
        english="backbiting",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="ذكر أخيك بما يكره في غيبته",
        opposite="",
        related=["نميمة", "كذب"]
    ),
    "فجور": BehaviorDefinition(
        arabic="فجور",
        english="immorality",
        category=BehaviorCategory.BAD_CHARACTER,
        is_behavior=True,
        definition_ar="الميل إلى الفساد والانحراف",
        opposite="تقوى",
        related=["فسق", "ظلم"]
    ),
    
    # =========================================================================
    # NOT BEHAVIORS - كيانات (Entities)
    # =========================================================================
    "نبي": BehaviorDefinition(
        arabic="نبي",
        english="prophet",
        category=BehaviorCategory.ENTITY,
        is_behavior=False,
        definition_ar="من أوحي إليه بشرع ولم يؤمر بتبليغه",
    ),
    "رسول": BehaviorDefinition(
        arabic="رسول",
        english="messenger",
        category=BehaviorCategory.ENTITY,
        is_behavior=False,
        definition_ar="من أوحي إليه بشرع وأمر بتبليغه",
    ),
    "ملائكة": BehaviorDefinition(
        arabic="ملائكة",
        english="angels",
        category=BehaviorCategory.ENTITY,
        is_behavior=False,
        definition_ar="مخلوقات نورانية لا يعصون الله",
    ),
    
    # =========================================================================
    # NOT BEHAVIORS - حالات (States/Labels)
    # =========================================================================
    "مؤمن": BehaviorDefinition(
        arabic="مؤمن",
        english="believer",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="من آمن بالله ورسله",
    ),
    "كافر": BehaviorDefinition(
        arabic="كافر",
        english="disbeliever",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="من جحد الحق",
    ),
    "منافق": BehaviorDefinition(
        arabic="منافق",
        english="hypocrite",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="من أظهر الإيمان وأبطن الكفر",
    ),
    "قلب": BehaviorDefinition(
        arabic="قلب",
        english="heart",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="محل الإيمان والعقل",
    ),
    "سليم": BehaviorDefinition(
        arabic="سليم",
        english="sound",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="صفة للقلب السالم من الشرك",
    ),
    "ميت": BehaviorDefinition(
        arabic="ميت",
        english="dead",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="من فارقت روحه جسده",
    ),
    "مريض": BehaviorDefinition(
        arabic="مريض",
        english="sick",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="صفة للقلب المصاب بالشبهات أو الشهوات",
    ),
    "قاسي": BehaviorDefinition(
        arabic="قاسي",
        english="hard",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="صفة للقلب الذي لا يلين",
    ),
    "مختوم": BehaviorDefinition(
        arabic="مختوم",
        english="sealed",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="صفة للقلب المطبوع عليه",
    ),
    "منيب": BehaviorDefinition(
        arabic="منيب",
        english="repentant",
        category=BehaviorCategory.STATE,
        is_behavior=False,
        definition_ar="الراجع إلى الله",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_true_behaviors() -> List[str]:
    """Get list of true behavioral labels (not entities/states)."""
    return [k for k, v in BEHAVIOR_TAXONOMY.items() if v.is_behavior]


def get_non_behaviors() -> List[str]:
    """Get list of non-behavioral labels (entities/states)."""
    return [k for k, v in BEHAVIOR_TAXONOMY.items() if not v.is_behavior]


def get_behaviors_by_category(category: BehaviorCategory) -> List[str]:
    """Get behaviors by category."""
    return [k for k, v in BEHAVIOR_TAXONOMY.items() if v.category == category]


def get_behavior_definition(behavior: str) -> BehaviorDefinition:
    """Get definition for a behavior."""
    return BEHAVIOR_TAXONOMY.get(behavior)


def get_opposite(behavior: str) -> str:
    """Get opposite behavior."""
    defn = BEHAVIOR_TAXONOMY.get(behavior)
    return defn.opposite if defn else ""


def get_related(behavior: str) -> List[str]:
    """Get related behaviors."""
    defn = BEHAVIOR_TAXONOMY.get(behavior)
    return defn.related if defn and defn.related else []


def is_true_behavior(label: str) -> bool:
    """Check if label is a true behavior (not entity/state)."""
    defn = BEHAVIOR_TAXONOMY.get(label)
    return defn.is_behavior if defn else False


# =============================================================================
# STATISTICS
# =============================================================================

def print_taxonomy_stats():
    """Print taxonomy statistics."""
    true_behaviors = get_true_behaviors()
    non_behaviors = get_non_behaviors()
    
    print("=" * 60)
    print("QURANIC BEHAVIOR TAXONOMY STATISTICS")
    print("=" * 60)
    print(f"Total labels: {len(BEHAVIOR_TAXONOMY)}")
    print(f"True behaviors: {len(true_behaviors)}")
    print(f"Non-behaviors (entities/states): {len(non_behaviors)}")
    print()
    
    print("TRUE BEHAVIORS BY CATEGORY:")
    for cat in BehaviorCategory:
        if cat not in [BehaviorCategory.ENTITY, BehaviorCategory.STATE]:
            behaviors = get_behaviors_by_category(cat)
            print(f"  {cat.value}: {len(behaviors)}")
            for b in behaviors:
                print(f"    - {b}")
    
    print()
    print("NON-BEHAVIORS (to exclude from training):")
    for b in non_behaviors:
        defn = BEHAVIOR_TAXONOMY[b]
        print(f"  - {b} ({defn.english}) - {defn.category.value}")


if __name__ == "__main__":
    print_taxonomy_stats()
