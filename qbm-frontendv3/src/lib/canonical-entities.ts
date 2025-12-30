// Canonical Entity Types and Data for QBM Synonym Explorer
// Based on vocab/canonical_entities.json

export type EntityType = "behavior" | "agent" | "organ" | "heart_state" | "consequence";

export interface CanonicalEntity {
  id: string;
  type: EntityType;
  ar: string;
  en: string;
  roots: string[];
  synonyms: string[];
  occurrences?: number;
  category?: string;
}

export const ENTITY_COUNTS = {
  behavior: 73,
  agent: 14,
  organ: 40,
  heart_state: 12,
  consequence: 16,
  total: 155,
  totalSynonyms: 720
};

export const ENTITY_TYPE_LABELS: Record<EntityType | "all", { en: string; ar: string; color: string }> = {
  all: { en: "All", ar: "الكل", color: "slate" },
  behavior: { en: "Behaviors", ar: "السلوكيات", color: "emerald" },
  agent: { en: "Agents", ar: "الفاعلون", color: "blue" },
  organ: { en: "Organs", ar: "الأعضاء", color: "purple" },
  heart_state: { en: "Heart States", ar: "أحوال القلب", color: "pink" },
  consequence: { en: "Consequences", ar: "العواقب", color: "amber" },
};

// Sample Behaviors (from canonical_entities.json)
export const SAMPLE_BEHAVIORS: CanonicalEntity[] = [
  {
    id: "BEH_EMO_PATIENCE",
    type: "behavior",
    ar: "الصبر",
    en: "Patience",
    roots: ["ص-ب-ر"],
    synonyms: ["صبر", "يصبر", "صابر", "صابرين", "الصابرون", "اصبر", "اصبروا", "صبرا"],
    occurrences: 103,
    category: "emotional"
  },
  {
    id: "BEH_EMO_GRATITUDE",
    type: "behavior",
    ar: "الشكر",
    en: "Gratitude",
    roots: ["ش-ك-ر"],
    synonyms: ["شكر", "يشكر", "شاكر", "شاكرين", "الشاكرون", "اشكر", "اشكروا"],
    occurrences: 75,
    category: "emotional"
  },
  {
    id: "BEH_SPI_PRAYER",
    type: "behavior",
    ar: "الصلاة",
    en: "Prayer",
    roots: ["ص-ل-و"],
    synonyms: ["صلاة", "يصلي", "صلوا", "المصلين", "مصلى", "صلوات", "أقيموا الصلاة"],
    occurrences: 99,
    category: "spiritual"
  },
  {
    id: "BEH_SPEECH_TRUTHFULNESS",
    type: "behavior",
    ar: "الصدق",
    en: "Truthfulness",
    roots: ["ص-د-ق"],
    synonyms: ["صدق", "يصدق", "صادق", "صادقين", "الصادقون", "صدقوا", "صدقت"],
    occurrences: 87,
    category: "speech"
  },
  {
    id: "BEH_SPI_PROSTRATION",
    type: "behavior",
    ar: "السجود",
    en: "Prostration",
    roots: ["س-ج-د"],
    synonyms: ["سجد", "يسجد", "ساجد", "ساجدين", "السجود", "اسجدوا", "مسجد", "ساجدون"],
    occurrences: 92,
    category: "spiritual"
  },
  {
    id: "BEH_SPI_FAITH",
    type: "behavior",
    ar: "الإيمان",
    en: "Faith",
    roots: ["أ-م-ن"],
    synonyms: ["آمنوا", "يؤمنون", "مؤمنين", "المؤمنين", "المؤمنون", "آمن", "إيمان"],
    occurrences: 812,
    category: "spiritual"
  },
  {
    id: "BEH_SPI_TAQWA",
    type: "behavior",
    ar: "التقوى",
    en: "God-Consciousness",
    roots: ["و-ق-ي"],
    synonyms: ["اتقوا", "يتقون", "المتقين", "المتقون", "تقوى", "اتقى", "تتقون"],
    occurrences: 258,
    category: "spiritual"
  },
  {
    id: "BEH_SPI_REPENTANCE",
    type: "behavior",
    ar: "التوبة",
    en: "Repentance",
    roots: ["ت-و-ب"],
    synonyms: ["تابوا", "يتوبون", "التائبين", "التائبون", "تاب", "توبوا", "توبة"],
    occurrences: 87,
    category: "spiritual"
  },
  {
    id: "BEH_SOC_JUSTICE",
    type: "behavior",
    ar: "العدل",
    en: "Justice",
    roots: ["ع-د-ل"],
    synonyms: ["عدلوا", "يعدلون", "العادلين", "اعدلوا", "بالعدل", "قسط", "المقسطين"],
    occurrences: 28,
    category: "social"
  },
  {
    id: "BEH_SOC_OPPRESSION",
    type: "behavior",
    ar: "الظلم",
    en: "Oppression",
    roots: ["ظ-ل-م"],
    synonyms: ["ظلموا", "يظلمون", "الظالمين", "الظالمون", "ظالمين", "ظالم", "ظلم"],
    occurrences: 289,
    category: "social"
  }
];

// Sample Agents
export const SAMPLE_AGENTS: CanonicalEntity[] = [
  {
    id: "AGT_BELIEVER",
    type: "agent",
    ar: "المؤمن",
    en: "Believer",
    roots: ["أ-م-ن"],
    synonyms: ["مؤمن", "مؤمنين", "المؤمنون", "الذين آمنوا", "أهل الإيمان", "آمنوا"],
    occurrences: 812
  },
  {
    id: "AGT_DISBELIEVER",
    type: "agent",
    ar: "الكافر",
    en: "Disbeliever",
    roots: ["ك-ف-ر"],
    synonyms: ["كافر", "كافرين", "الكافرون", "الذين كفروا", "كفروا", "الكفار"],
    occurrences: 525
  },
  {
    id: "AGT_HYPOCRITE",
    type: "agent",
    ar: "المنافق",
    en: "Hypocrite",
    roots: ["ن-ف-ق"],
    synonyms: ["منافق", "منافقين", "المنافقون", "الذين نافقوا", "المنافقات"],
    occurrences: 37
  },
  {
    id: "AGT_PROPHET",
    type: "agent",
    ar: "النبي",
    en: "Prophet",
    roots: ["ن-ب-أ"],
    synonyms: ["نبي", "أنبياء", "النبيين", "رسول", "رسل", "المرسلين", "الرسول"],
    occurrences: 124
  },
  {
    id: "AGT_OPPRESSOR",
    type: "agent",
    ar: "الظالم",
    en: "Oppressor",
    roots: ["ظ-ل-م"],
    synonyms: ["ظالم", "ظالمين", "الظالمون", "الذين ظلموا", "ظلموا"],
    occurrences: 289
  },
  {
    id: "AGT_RIGHTEOUS",
    type: "agent",
    ar: "الصالح",
    en: "Righteous",
    roots: ["ص-ل-ح"],
    synonyms: ["صالح", "صالحين", "الصالحون", "الصالحات", "أصلح"],
    occurrences: 62
  },
  {
    id: "AGT_PATIENT",
    type: "agent",
    ar: "الصابر",
    en: "Patient One",
    roots: ["ص-ب-ر"],
    synonyms: ["صابر", "صابرين", "الصابرون", "الصابرين"],
    occurrences: 103
  }
];

// Sample Organs
export const SAMPLE_ORGANS: CanonicalEntity[] = [
  {
    id: "ORG_HEART",
    type: "organ",
    ar: "القلب",
    en: "Heart",
    roots: ["ق-ل-ب"],
    synonyms: ["قلب", "قلوب", "قلوبهم", "قلبه", "أفئدة", "فؤاد", "صدر", "صدور"],
    occurrences: 132
  },
  {
    id: "ORG_TONGUE",
    type: "organ",
    ar: "اللسان",
    en: "Tongue",
    roots: ["ل-س-ن"],
    synonyms: ["لسان", "ألسنة", "ألسنتهم", "لسانه", "ألسنتكم"],
    occurrences: 25
  },
  {
    id: "ORG_EYE",
    type: "organ",
    ar: "العين",
    en: "Eye",
    roots: ["ع-ي-ن"],
    synonyms: ["عين", "أعين", "عيون", "أعينهم", "بصر", "أبصار", "أبصارهم"],
    occurrences: 65
  },
  {
    id: "ORG_EAR",
    type: "organ",
    ar: "الأذن",
    en: "Ear",
    roots: ["أ-ذ-ن"],
    synonyms: ["أذن", "آذان", "آذانهم", "سمع", "أسماع", "سمعهم"],
    occurrences: 47
  },
  {
    id: "ORG_HAND",
    type: "organ",
    ar: "اليد",
    en: "Hand",
    roots: ["ي-د-ي"],
    synonyms: ["يد", "أيدي", "أيديهم", "يده", "يمين", "أيمان", "بأيديكم"],
    occurrences: 120
  },
  {
    id: "ORG_FOOT",
    type: "organ",
    ar: "القدم",
    en: "Foot",
    roots: ["ق-د-م"],
    synonyms: ["قدم", "أقدام", "أقدامهم", "رجل", "أرجل", "أرجلهم"],
    occurrences: 16
  }
];

// Sample Heart States
export const SAMPLE_HEART_STATES: CanonicalEntity[] = [
  {
    id: "HRT_HARDNESS",
    type: "heart_state",
    ar: "قسوة القلب",
    en: "Hardness of Heart",
    roots: ["ق-س-و"],
    synonyms: ["قسوة", "قست", "قاسية", "قلوبهم قاسية", "أشد قسوة"],
    occurrences: 12
  },
  {
    id: "HRT_SOFTNESS",
    type: "heart_state",
    ar: "لين القلب",
    en: "Softness of Heart",
    roots: ["ل-ي-ن"],
    synonyms: ["لين", "تلين", "لينة", "جلودهم وقلوبهم"],
    occurrences: 8
  },
  {
    id: "HRT_SEALING",
    type: "heart_state",
    ar: "ختم القلب",
    en: "Sealing of Heart",
    roots: ["خ-ت-م"],
    synonyms: ["ختم", "طبع", "أغلف", "غشاوة", "ختم الله", "طبع على قلوبهم"],
    occurrences: 17
  },
  {
    id: "HRT_TRANQUILITY",
    type: "heart_state",
    ar: "سكينة القلب",
    en: "Tranquility",
    roots: ["س-ك-ن"],
    synonyms: ["سكينة", "اطمئنان", "تطمئن", "مطمئنة", "اطمأن"],
    occurrences: 13
  },
  {
    id: "HRT_FEAR",
    type: "heart_state",
    ar: "خشية القلب",
    en: "Fear/Awe",
    roots: ["خ-ش-ي"],
    synonyms: ["خشية", "يخشى", "خاشعة", "وجلت", "يخشون", "الخاشعين"],
    occurrences: 48
  },
  {
    id: "HRT_DISEASE",
    type: "heart_state",
    ar: "مرض القلب",
    en: "Disease of Heart",
    roots: ["م-ر-ض"],
    synonyms: ["مرض", "في قلوبهم مرض", "مرضا", "مريض"],
    occurrences: 12
  }
];

// Sample Consequences
export const SAMPLE_CONSEQUENCES: CanonicalEntity[] = [
  {
    id: "CSQ_JANNAH",
    type: "consequence",
    ar: "الجنة",
    en: "Paradise",
    roots: ["ج-ن-ن"],
    synonyms: ["جنة", "جنات", "الفردوس", "دار السلام", "جنات النعيم", "جنات عدن"],
    occurrences: 147
  },
  {
    id: "CSQ_JAHANNAM",
    type: "consequence",
    ar: "جهنم",
    en: "Hellfire",
    roots: ["ج-ه-ن-م"],
    synonyms: ["جهنم", "النار", "سعير", "جحيم", "لظى", "الحطمة", "سقر", "هاوية"],
    occurrences: 126
  },
  {
    id: "CSQ_FORGIVENESS",
    type: "consequence",
    ar: "المغفرة",
    en: "Forgiveness",
    roots: ["غ-ف-ر"],
    synonyms: ["مغفرة", "غفور", "يغفر", "غفران", "استغفر", "غافر", "اغفر"],
    occurrences: 234
  },
  {
    id: "CSQ_PUNISHMENT",
    type: "consequence",
    ar: "العذاب",
    en: "Punishment",
    roots: ["ع-ذ-ب"],
    synonyms: ["عذاب", "يعذب", "معذبين", "عذاب أليم", "عذاب شديد", "عذاب عظيم"],
    occurrences: 322
  },
  {
    id: "CSQ_MERCY",
    type: "consequence",
    ar: "الرحمة",
    en: "Mercy",
    roots: ["ر-ح-م"],
    synonyms: ["رحمة", "رحيم", "الرحمن", "يرحم", "ارحم", "رحماء", "أرحم الراحمين"],
    occurrences: 339
  },
  {
    id: "CSQ_GUIDANCE",
    type: "consequence",
    ar: "الهداية",
    en: "Guidance",
    roots: ["ه-د-ي"],
    synonyms: ["هداية", "هدى", "يهدي", "اهدنا", "المهتدين", "هاد"],
    occurrences: 316
  }
];

// Combined all entities
export const ALL_ENTITIES: CanonicalEntity[] = [
  ...SAMPLE_BEHAVIORS,
  ...SAMPLE_AGENTS,
  ...SAMPLE_ORGANS,
  ...SAMPLE_HEART_STATES,
  ...SAMPLE_CONSEQUENCES,
];

// Get entities by type
export function getEntitiesByType(type: EntityType | "all"): CanonicalEntity[] {
  if (type === "all") return ALL_ENTITIES;
  return ALL_ENTITIES.filter(e => e.type === type);
}

// Search entities
export function searchEntities(query: string, type: EntityType | "all" = "all"): CanonicalEntity[] {
  const entities = getEntitiesByType(type);
  if (!query.trim()) return entities;

  const lowerQuery = query.toLowerCase();
  return entities.filter(entity =>
    entity.ar.includes(query) ||
    entity.en.toLowerCase().includes(lowerQuery) ||
    entity.id.toLowerCase().includes(lowerQuery) ||
    entity.synonyms.some(s => s.includes(query)) ||
    entity.roots.some(r => r.includes(query))
  );
}
