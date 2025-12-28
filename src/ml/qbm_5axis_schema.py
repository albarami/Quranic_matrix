"""
QBM 5-Axis Schema Based on Bouzidani's Matrix (جدول 02)

5 Classification Axes:
1. التصنيف العضوي البيولوجي (Organic)
2. التصنيف الموضعي (Situational) 
3. التصنيف النسقي (Systemic)
4. التصنيف المكاني (Spatial)
5. التصنيف الزماني (Temporal)

Behavior Types with النية (intention):
- غريزي → لا ينطبق (no moral eval)
- بنية → عمل صالح
- بدون نية → عمل سيء
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# Axis 1: Organic
class OrganType(Enum):
    YAD = "اليد"
    RIJL = "الرجل"
    AYN = "العين"
    UDHN = "الأذن"
    LISAN = "اللسان"
    QALB = "القلب"
    NAFS = "النفس"
    UNKNOWN = "غير معروف"

class OrganicDomain(Enum):
    ZAHIRI = "ظاهري"
    BATINI = "باطني"
    MURAKKAB = "مركب"
    UNKNOWN = "غير معروف"

class BatiniSubtype(Enum):
    ATIFA = "عاطفة"
    RAY = "رأي"
    FIKR = "فكر"
    NAFS = "نفس"
    UNKNOWN = "غير معروف"

# Axis 2: Situational
class SituationalContext(Enum):
    NAFS = "النفس"
    AFAQ = "الآفاق"
    KHALIQ = "الخالق"
    KAWN = "الكون"
    HAYAT = "الحياة"
    UNKNOWN = "غير معروف"

# Axis 3: Systemic
class SystemicContext(Enum):
    BAYT = "البيت"
    AMAL = "العمل"
    MAKAN_AAM = "مكان عام"
    UNKNOWN = "غير معروف"

# Axis 4: Spatial
class SpatialContext(Enum):
    BAYT = "البيت"
    MASJID = "المسجد"
    SUQ = "السوق"
    UNKNOWN = "غير معروف"

# Axis 5: Temporal
class TemporalContext(Enum):
    SABAHAN = "صباحاً"
    DHUHRAN = "ظهراً"
    ASRAN = "عصراً"
    LAYLAN = "ليلاً"
    DAIM = "دائم"
    UNKNOWN = "غير معروف"

# Behavior Type & Intention
class BehaviorType(Enum):
    GHARIZI = "غريزي"
    IRADI = "إرادي"
    UNKNOWN = "غير معروف"

class IntentionStatus(Enum):
    BI_NIYA = "بنية"
    BIDUN_NIYA = "بدون نية"
    LA_YANTABIQ = "لا ينطبق"
    UNKNOWN = "غير معروف"

class MoralEvaluation(Enum):
    AMAL_SALIH = "عمل صالح"
    AMAL_SAYYI = "عمل سيء"
    LA_YANTABIQ = "لا ينطبق"
    MUHAYID = "محايد"
    UNKNOWN = "غير معروف"

# Aunger & Curtis
class ControlLevel(Enum):
    TAFA3ULI = "تفاعلي"
    MUHAFFIZ = "محفز"
    TANFIDHI = "تنفيذي"
    UNKNOWN = "غير معروف"

class BehaviorCategory(Enum):
    IBADA_SHU3URIYA = "عبادة شعورية"
    IBADA_FIKRIYA = "عبادة فكرية"
    IBADA_SULUKIYA = "عبادة سلوكية"
    FADILA_QALB = "فضيلة قلب"
    MARAD_QALB = "مرض قلب"
    KHULUQ_HASAN = "خلق حسن"
    KHULUQ_SAYYI = "خلق سيء"
    KAYAN = "كيان"
    HALA = "حالة"


@dataclass
class BehaviorRecord:
    """Behavior record with 5-axis classification."""
    id: str
    arabic: str
    english: str
    category: BehaviorCategory
    is_true_behavior: bool
    behavior_type: BehaviorType
    intention_status: IntentionStatus
    moral_evaluation: MoralEvaluation
    primary_organ: OrganType
    organic_domain: OrganicDomain
    batini_subtype: Optional[BatiniSubtype] = None
    situational: List[SituationalContext] = field(default_factory=list)
    systemic: List[SystemicContext] = field(default_factory=list)
    spatial: List[SpatialContext] = field(default_factory=list)
    temporal: List[TemporalContext] = field(default_factory=list)
    control_level: ControlLevel = ControlLevel.UNKNOWN
    definition_ar: str = ""
    opposite_id: str = ""
    quranic_roots: List[str] = field(default_factory=list)


# Arabic to ID mapping for existing annotations
ARABIC_TO_ID = {
    "ذكر": "BEH_DHIKR", "دعاء": "BEH_DUA", "توبة": "BEH_TAWBA", "شكر": "BEH_SHUKR",
    "إيمان": "BEH_IMAN", "تقوى": "BEH_TAQWA", "توكل": "BEH_TAWAKKUL", "رضا": "BEH_RIDA",
    "إخلاص": "BEH_IKHLAS", "خشوع": "BEH_KHUSHU",
    "كبر": "BEH_KIBR", "حسد": "BEH_HASAD", "غفلة": "BEH_GHAFLA", "نفاق": "BEH_NIFAQ",
    "رياء": "BEH_RIYA", "غضب": "BEH_GHADAB", "كفر": "BEH_KUFR", "شرك": "BEH_SHIRK", "فسق": "BEH_FISQ",
    "صدق": "BEH_SIDQ", "صبر": "BEH_SABR", "عدل": "BEH_ADL", "رحمة": "BEH_RAHMA",
    "تواضع": "BEH_TAWADU", "حياء": "BEH_HAYA", "إحسان": "BEH_IHSAN", "أمانة": "BEH_AMANA", "زهد": "BEH_ZUHD",
    "كذب": "BEH_KIDHB", "ظلم": "BEH_DHULM", "بخل": "BEH_BUKHL", "خيانة": "BEH_KHIYANA",
    "غيبة": "BEH_GHIBA", "فجور": "BEH_FUJUR",
    # Non-behaviors
    "نبي": "ENT_NABI", "رسول": "ENT_RASUL", "ملائكة": "ENT_MALAK",
    "مؤمن": "STATE_MUMIN", "كافر": "STATE_KAFIR", "منافق": "STATE_MUNAFIQ",
    "قلب": "STATE_QALB", "سليم": "STATE_SALIM", "ميت": "STATE_MAYYIT",
    "مريض": "STATE_MARID", "قاسي": "STATE_QASI", "مختوم": "STATE_MAKHTUM", "منيب": "STATE_MUNIB",
}

# Non-behavior IDs
NON_BEHAVIOR_IDS = {"ENT_NABI", "ENT_RASUL", "ENT_MALAK", "STATE_MUMIN", "STATE_KAFIR",
                    "STATE_MUNAFIQ", "STATE_QALB", "STATE_SALIM", "STATE_MAYYIT",
                    "STATE_MARID", "STATE_QASI", "STATE_MAKHTUM", "STATE_MUNIB"}


def is_true_behavior(arabic: str) -> bool:
    """Check if Arabic term is a true behavior."""
    beh_id = ARABIC_TO_ID.get(arabic)
    return beh_id is not None and beh_id not in NON_BEHAVIOR_IDS


def get_behavior_id(arabic: str) -> Optional[str]:
    """Get behavior ID for Arabic term."""
    return ARABIC_TO_ID.get(arabic)


if __name__ == "__main__":
    print("QBM 5-Axis Schema")
    print(f"Total mappings: {len(ARABIC_TO_ID)}")
    print(f"True behaviors: {sum(1 for a in ARABIC_TO_ID if is_true_behavior(a))}")
    print(f"Non-behaviors: {len(NON_BEHAVIOR_IDS)}")
