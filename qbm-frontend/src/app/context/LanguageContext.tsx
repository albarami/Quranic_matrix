"use client";

import { createContext, useContext, useState, ReactNode } from "react";

type Language = "en" | "ar";

interface Translations {
  nav: {
    home: string;
    research: string;
    explorer: string;
    dashboard: string;
    insights: string;
    subtitle: string;
  };
  research: {
    title: string;
    subtitle: string;
    examples: string;
    prompts: {
      statistics: { title: string; prompt: string };
      baqarah: { title: string; prompt: string };
      agent: { title: string; prompt: string };
      tafsir: { title: string; prompt: string };
    };
  };
  explorer: {
    title: string;
    subtitle: string;
    search: string;
    filterBy: string;
    coverage: string;
    totalSurahs: string;
    annotatedAyat: string;
    behavioralSpans: string;
    avgCoverage: string;
  };
  dashboard: {
    title: string;
    subtitle: string;
    refreshAll: string;
    export: string;
    updated: string;
    quranCoverage: string;
    totalSpans: string;
    goldTier: string;
    avgIAA: string;
  };
  insights: {
    title: string;
    subtitle: string;
    patternsDiscovered: string;
    crossReferences: string;
    behavioralClusters: string;
    tafsirCorrelations: string;
  };
  common: {
    loading: string;
    error: string;
    back: string;
    viewAll: string;
  };
}

const translations: Record<Language, Translations> = {
  en: {
    nav: {
      home: "Home",
      research: "Research",
      explorer: "Explorer",
      dashboard: "Dashboard",
      insights: "Insights",
      subtitle: "Behavioral Matrix",
    },
    research: {
      title: "QBM Research Assistant",
      subtitle: "AI-powered Quranic behavioral analysis",
      examples: "Examples:",
      prompts: {
        statistics: { title: "Project Statistics", prompt: "Show me the QBM project statistics with charts" },
        baqarah: { title: "Surah Al-Baqarah", prompt: "What behavioral annotations are in Surah Al-Baqarah?" },
        agent: { title: "Search by Agent", prompt: "Show me all behaviors attributed to Allah (AGT_ALLAH)" },
        tafsir: { title: "Compare Tafsir", prompt: "Compare tafsir sources for Ayat al-Kursi (2:255)" },
      },
    },
    explorer: {
      title: "Quran Explorer",
      subtitle: "Browse all 114 surahs with behavioral annotations",
      search: "Search surahs...",
      filterBy: "Filter by behavior:",
      coverage: "Coverage",
      totalSurahs: "Total Surahs",
      annotatedAyat: "Annotated Ayat",
      behavioralSpans: "Behavioral Spans",
      avgCoverage: "Avg Coverage",
    },
    dashboard: {
      title: "Project Dashboard",
      subtitle: "Real-time statistics and AI-generated visualizations",
      refreshAll: "Refresh All",
      export: "Export",
      updated: "Updated",
      quranCoverage: "Quran Coverage",
      totalSpans: "Total Spans",
      goldTier: "Gold Tier",
      avgIAA: "Avg IAA",
    },
    insights: {
      title: "Research Insights",
      subtitle: "Patterns and discoveries uncovered through computational analysis",
      patternsDiscovered: "Patterns Discovered",
      crossReferences: "Cross-References",
      behavioralClusters: "Behavioral Clusters",
      tafsirCorrelations: "Tafsir Correlations",
    },
    common: {
      loading: "Loading...",
      error: "Error",
      back: "Back",
      viewAll: "View all",
    },
  },
  ar: {
    nav: {
      home: "الرئيسية",
      research: "البحث",
      explorer: "المستكشف",
      dashboard: "لوحة التحكم",
      insights: "الرؤى",
      subtitle: "المصفوفة السلوكية",
    },
    research: {
      title: "مساعد بحث QBM",
      subtitle: "تحليل سلوكي قرآني مدعوم بالذكاء الاصطناعي",
      examples: "أمثلة:",
      prompts: {
        statistics: { title: "إحصائيات المشروع", prompt: "أظهر لي إحصائيات مشروع QBM مع الرسوم البيانية" },
        baqarah: { title: "سورة البقرة", prompt: "ما هي التصنيفات السلوكية في سورة البقرة؟" },
        agent: { title: "البحث حسب الفاعل", prompt: "أظهر لي جميع السلوكيات المنسوبة إلى الله (AGT_ALLAH)" },
        tafsir: { title: "مقارنة التفسير", prompt: "قارن مصادر التفسير لآية الكرسي (2:255)" },
      },
    },
    explorer: {
      title: "مستكشف القرآن",
      subtitle: "تصفح جميع السور الـ 114 مع التصنيفات السلوكية",
      search: "ابحث في السور...",
      filterBy: "تصفية حسب السلوك:",
      coverage: "التغطية",
      totalSurahs: "إجمالي السور",
      annotatedAyat: "الآيات المصنفة",
      behavioralSpans: "النطاقات السلوكية",
      avgCoverage: "متوسط التغطية",
    },
    dashboard: {
      title: "لوحة التحكم",
      subtitle: "إحصائيات في الوقت الفعلي وتصورات مولدة بالذكاء الاصطناعي",
      refreshAll: "تحديث الكل",
      export: "تصدير",
      updated: "آخر تحديث",
      quranCoverage: "تغطية القرآن",
      totalSpans: "إجمالي النطاقات",
      goldTier: "المستوى الذهبي",
      avgIAA: "متوسط IAA",
    },
    insights: {
      title: "رؤى البحث",
      subtitle: "أنماط واكتشافات من خلال التحليل الحسابي",
      patternsDiscovered: "الأنماط المكتشفة",
      crossReferences: "المراجع المتقاطعة",
      behavioralClusters: "المجموعات السلوكية",
      tafsirCorrelations: "ارتباطات التفسير",
    },
    common: {
      loading: "جاري التحميل...",
      error: "خطأ",
      back: "رجوع",
      viewAll: "عرض الكل",
    },
  },
};

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: Translations;
  isRTL: boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>("en");

  const value: LanguageContextType = {
    language,
    setLanguage,
    t: translations[language],
    isRTL: language === "ar",
  };

  return (
    <LanguageContext.Provider value={value}>
      <div dir={language === "ar" ? "rtl" : "ltr"} className={language === "ar" ? "font-arabic" : ""}>
        {children}
      </div>
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
