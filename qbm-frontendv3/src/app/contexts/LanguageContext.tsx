"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Language = "en" | "ar";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  isRTL: boolean;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    "nav.home": "Home",
    "nav.research": "Research",
    "nav.explorer": "Explorer",
    "nav.discovery": "Discovery",
    "nav.dashboard": "Dashboard",
    "nav.insights": "Insights",
    "nav.proof": "Proof",
    "nav.synonyms": "Synonyms",
    "nav.annotate": "Annotate",
    "nav.graph": "Graph",
    "nav.genome": "Genome",
    "nav.reviews": "Reviews",
    
    // Home page
    "home.title": "Quranic Behavioral Matrix",
    "home.tagline": "World's First AI-Powered Quranic Behavior Research Platform",
    "home.subtitle": "Behavioral mapping across the Quranic corpus",
    "home.description": "Discover behavioral patterns across the Quranic dataset using AI-powered natural language queries, interactive visualizations, and integrated classical tafsir analysis.",
    "home.explore_dataset": "Explore Dataset",
    "home.ayat_analyzed": "Ayat Analyzed",
    "home.behavioral_spans": "Behavioral Spans",
    "home.tafsir_sources": "Tafsir Sources",
    "home.behavior_concepts": "Behavior Concepts",
    "home.dataset_coverage": "Dataset coverage",
    "home.classified_annotations": "Classified annotations",
    "home.integrated_sources": "Integrated sources",
    "home.unique_behaviors": "Unique behaviors",
    "home.ai_insights": "AI-Discovered Insights",
    "home.what_uncovered": "What We've Uncovered",
    "home.insights_description": "Using advanced NLP and classical tafsir integration, our AI has identified fascinating patterns in Quranic behavioral guidance.",
    "home.start_research": "Start Research",
    "home.view_dashboard": "View Dashboard",
    "home.try_research": "Try Research Assistant",
    "home.inner_states": "Inner States",
    "home.inner_states_desc": "Internal psychological and spiritual states referenced in the Quran",
    "home.speech_acts": "Speech Acts",
    "home.speech_acts_desc": "Verbal behaviors including commands, prohibitions, and guidance",
    "home.relational_acts": "Relational Acts",
    "home.relational_acts_desc": "Interpersonal behaviors governing relationships and community",
    "home.physical_acts": "Physical Acts",
    "home.physical_acts_desc": "Bodily actions and physical behaviors mentioned in the Quran",
    "home.powered_by": "Powered by Generative UI",
    "home.ask_anything": "Ask Anything. Get Visual Answers.",
    "home.ask_description": "Our AI does not just answer questions - it generates interactive visualizations, comparison tables, and drill-down charts in real time.",
    "home.recent_annotations": "Recent Annotations",
    "home.reference": "Reference",
    "home.behavior": "Behavior",
    "home.agent": "Agent",
    "home.annotator": "Annotator",
    "home.annotated": "Annotated",
    "home.tafsir_comparison": "Tafsir Comparison",
    "home.no_tafsir": "No tafsir available for this ayah.",
    "home.chart": "Chart",
    "home.table": "Table",
    "home.comparison": "Comparison",
    "home.selected_ayah": "Selected Ayah",
    "home.loading_verse": "Loading verse...",
    "home.verse_not_available": "Verse not available",
    
    // Explorer
    "explorer.title": "Quran Explorer",
    "explorer.subtitle": "Browse surahs with behavioral annotations",
    "explorer.search_placeholder": "Search surahs...",
    "explorer.total_surahs": "Total Surahs",
    "explorer.annotated_ayat": "Annotated Ayat",
    "explorer.behavioral_spans": "Behavioral Spans",
    "explorer.avg_coverage": "Avg Coverage",
    "explorer.filter_by_behavior": "Filter by Behavior",
    "explorer.all_behaviors": "All Behaviors",
    "explorer.ayat": "ayat",
    "explorer.spans": "spans",
    "explorer.coverage": "coverage",
    "explorer.view_details": "View Details",
    "explorer.close": "Close",
    "explorer.generating": "Generating detailed view...",
    
    // Dashboard
    "dashboard.title": "Research Dashboard",
    "dashboard.subtitle": "Real-time analytics and AI-generated insights",
    "dashboard.total_spans": "Total Spans",
    "dashboard.unique_ayat": "Unique Ayat",
    "dashboard.behavior_forms": "Behavior Forms",
    "dashboard.agent_types": "Agent Types",
    "dashboard.behavior_distribution": "Behavior Distribution",
    "dashboard.agent_distribution": "Agent Distribution",
    "dashboard.top_surahs": "Top Surahs by Annotations",
    "dashboard.refresh": "Refresh All",
    "dashboard.export": "Export",
    "dashboard.generating": "Generating...",
    "dashboard.quran_coverage": "Quran Coverage",
    "dashboard.unique_surahs": "Unique Surahs",
    "dashboard.with_annotations": "With annotations",
    "dashboard.annotated_verses": "Annotated verses",
    "dashboard.behavioral_annotations": "Behavioral annotations",
    
    // Insights
    "insights.title": "AI Insights",
    "insights.subtitle": "Discover patterns in Quranic behavioral guidance",
    "insights.generating": "Generating insights...",
    
    // Research
    "research.title": "QBM Research Assistant",
    "research.subtitle": "Ask questions about Quranic behaviors in natural language",
    "research.description": "I'll generate interactive charts, comparison tables, and tafsir panels to answer your research questions.",
    "research.behavioral_analysis": "Behavioral Analysis",
    "research.tafsir_exploration": "Tafsir Exploration",
    "research.statistical_insights": "Statistical Insights",
    "research.cross_reference": "Cross-Reference",
    "research.try_examples": "Try these examples",
    "research.click_to_try": "Click to try",
    "research.interactive_charts": "Interactive Charts",
    "research.pie_bar_line": "Pie, bar, line visualizations",
    "research.tafsir_integration": "Tafsir Integration",
    "research.classical_sources": "5 classical sources",
    "research.smart_search": "Smart Search",
    "research.arabic_english": "Arabic & English queries",
    "research.pattern_discovery": "Pattern Discovery",
    "research.ai_insights": "AI-powered insights",
    "research.select_example": "Select an example from the sidebar or type your own question below",
    "research.type_question": "Type your research question...",
    "research.powered_by_c1": "Powered by C1 Generative UI",
    "research.ask_natural": "Ask questions in natural language. Get interactive visualizations.",
    
    // Common
    "common.loading": "Loading...",
    "common.error": "Error",
    "common.no_data": "No data available",
  },
  ar: {
    // Navigation
    "nav.home": "الرئيسية",
    "nav.research": "البحث",
    "nav.explorer": "المستكشف",
    "nav.discovery": "الاكتشاف",
    "nav.dashboard": "لوحة التحكم",
    "nav.insights": "الرؤى",
    "nav.proof": "الإثبات",
    "nav.synonyms": "المرادفات",
    "nav.annotate": "التعليق",
    "nav.graph": "الشبكة",
    "nav.genome": "الجينوم",
    "nav.reviews": "المراجعات",
    
    // Home page
    "home.title": "مصفوفة السلوك القرآني",
    "home.tagline": "أول منصة بحثية في العالم مدعومة بالذكاء الاصطناعي للسلوك القرآني",
    "home.subtitle": "رسم خريطة السلوك عبر المصحف القرآني",
    "home.description": "اكتشف الأنماط السلوكية في مجموعة البيانات القرآنية باستخدام استعلامات اللغة الطبيعية المدعومة بالذكاء الاصطناعي والتصورات التفاعلية وتحليل التفسير الكلاسيكي المتكامل.",
    "home.explore_dataset": "استكشف البيانات",
    "home.ayat_analyzed": "الآيات المحللة",
    "home.behavioral_spans": "النطاقات السلوكية",
    "home.tafsir_sources": "مصادر التفسير",
    "home.behavior_concepts": "المفاهيم السلوكية",
    "home.dataset_coverage": "تغطية البيانات",
    "home.classified_annotations": "التعليقات المصنفة",
    "home.integrated_sources": "المصادر المتكاملة",
    "home.unique_behaviors": "السلوكيات الفريدة",
    "home.ai_insights": "رؤى الذكاء الاصطناعي",
    "home.what_uncovered": "ما اكتشفناه",
    "home.insights_description": "باستخدام معالجة اللغة الطبيعية المتقدمة وتكامل التفسير الكلاسيكي، حدد الذكاء الاصطناعي أنماطاً رائعة في التوجيه السلوكي القرآني.",
    "home.start_research": "ابدأ البحث",
    "home.view_dashboard": "عرض لوحة التحكم",
    "home.try_research": "جرب مساعد البحث",
    "home.inner_states": "الحالات الداخلية",
    "home.inner_states_desc": "الحالات النفسية والروحية الداخلية المذكورة في القرآن",
    "home.speech_acts": "الأفعال الكلامية",
    "home.speech_acts_desc": "السلوكيات اللفظية بما في ذلك الأوامر والنواهي والتوجيهات",
    "home.relational_acts": "الأفعال العلائقية",
    "home.relational_acts_desc": "السلوكيات الشخصية التي تحكم العلاقات والمجتمع",
    "home.physical_acts": "الأفعال الجسدية",
    "home.physical_acts_desc": "الأفعال الجسدية والسلوكيات المادية المذكورة في القرآن",
    "home.powered_by": "مدعوم بواجهة المستخدم التوليدية",
    "home.ask_anything": "اسأل أي شيء. احصل على إجابات مرئية.",
    "home.ask_description": "الذكاء الاصطناعي لا يجيب على الأسئلة فحسب - بل ينشئ تصورات تفاعلية وجداول مقارنة ورسوم بيانية تفصيلية في الوقت الفعلي.",
    "home.recent_annotations": "التعليقات الأخيرة",
    "home.reference": "المرجع",
    "home.behavior": "السلوك",
    "home.agent": "الفاعل",
    "home.annotator": "المعلق",
    "home.annotated": "تاريخ التعليق",
    "home.tafsir_comparison": "مقارنة التفسير",
    "home.no_tafsir": "لا يوجد تفسير متاح لهذه الآية.",
    "home.chart": "رسم بياني",
    "home.table": "جدول",
    "home.comparison": "مقارنة",
    "home.selected_ayah": "الآية المختارة",
    "home.loading_verse": "جاري تحميل الآية...",
    "home.verse_not_available": "الآية غير متوفرة",
    
    // Explorer
    "explorer.title": "مستكشف القرآن",
    "explorer.subtitle": "تصفح السور مع التعليقات السلوكية",
    "explorer.search_placeholder": "البحث في السور...",
    "explorer.total_surahs": "إجمالي السور",
    "explorer.annotated_ayat": "الآيات المشروحة",
    "explorer.behavioral_spans": "النطاقات السلوكية",
    "explorer.avg_coverage": "متوسط التغطية",
    "explorer.filter_by_behavior": "تصفية حسب السلوك",
    "explorer.all_behaviors": "جميع السلوكيات",
    "explorer.ayat": "آية",
    "explorer.spans": "نطاق",
    "explorer.coverage": "تغطية",
    "explorer.view_details": "عرض التفاصيل",
    "explorer.close": "إغلاق",
    "explorer.generating": "جاري إنشاء العرض التفصيلي...",
    
    // Dashboard
    "dashboard.title": "لوحة البحث",
    "dashboard.subtitle": "تحليلات في الوقت الفعلي ورؤى مولدة بالذكاء الاصطناعي",
    "dashboard.total_spans": "إجمالي النطاقات",
    "dashboard.unique_ayat": "الآيات الفريدة",
    "dashboard.behavior_forms": "أشكال السلوك",
    "dashboard.agent_types": "أنواع الفاعلين",
    "dashboard.behavior_distribution": "توزيع السلوك",
    "dashboard.agent_distribution": "توزيع الفاعلين",
    "dashboard.top_surahs": "أكثر السور تعليقاً",
    "dashboard.refresh": "تحديث الكل",
    "dashboard.export": "تصدير",
    "dashboard.generating": "جاري الإنشاء...",
    "dashboard.quran_coverage": "تغطية القرآن",
    "dashboard.unique_surahs": "السور الفريدة",
    "dashboard.with_annotations": "مع التعليقات",
    "dashboard.annotated_verses": "الآيات المشروحة",
    "dashboard.behavioral_annotations": "التعليقات السلوكية",
    
    // Insights
    "insights.title": "رؤى الذكاء الاصطناعي",
    "insights.subtitle": "اكتشف الأنماط في التوجيه السلوكي القرآني",
    "insights.generating": "جاري إنشاء الرؤى...",
    
    // Research
    "research.title": "مساعد البحث القرآني",
    "research.subtitle": "اطرح أسئلة حول السلوكيات القرآنية باللغة الطبيعية",
    "research.description": "سأنشئ رسوماً بيانية تفاعلية وجداول مقارنة ولوحات تفسير للإجابة على أسئلتك البحثية.",
    "research.behavioral_analysis": "تحليل السلوك",
    "research.tafsir_exploration": "استكشاف التفسير",
    "research.statistical_insights": "رؤى إحصائية",
    "research.cross_reference": "المراجع المتقاطعة",
    "research.try_examples": "جرب هذه الأمثلة",
    "research.click_to_try": "انقر للتجربة",
    "research.interactive_charts": "رسوم بيانية تفاعلية",
    "research.pie_bar_line": "تصورات دائرية وشريطية وخطية",
    "research.tafsir_integration": "تكامل التفسير",
    "research.classical_sources": "5 مصادر كلاسيكية",
    "research.smart_search": "بحث ذكي",
    "research.arabic_english": "استعلامات عربية وإنجليزية",
    "research.pattern_discovery": "اكتشاف الأنماط",
    "research.ai_insights": "رؤى مدعومة بالذكاء الاصطناعي",
    "research.select_example": "اختر مثالاً من الشريط الجانبي أو اكتب سؤالك أدناه",
    "research.type_question": "اكتب سؤالك البحثي...",
    "research.powered_by_c1": "مدعوم بواجهة C1 التوليدية",
    "research.ask_natural": "اطرح أسئلة باللغة الطبيعية. احصل على تصورات تفاعلية.",
    
    // Common
    "common.loading": "جاري التحميل...",
    "common.error": "خطأ",
    "common.no_data": "لا توجد بيانات متاحة",
  },
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>("en");

  useEffect(() => {
    const saved = localStorage.getItem("qbm-language") as Language;
    if (saved && (saved === "en" || saved === "ar")) {
      setLanguageState(saved);
    }
  }, []);

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem("qbm-language", lang);
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.documentElement.lang = lang;
  };

  useEffect(() => {
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
    document.documentElement.lang = language;
  }, [language]);

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t, isRTL: language === "ar" }}>
      {children}
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
