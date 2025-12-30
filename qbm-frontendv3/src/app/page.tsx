"use client";

import { useState, useEffect, useMemo, type ReactNode } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  BookOpen,
  BarChart3,
  Search,
  Sparkles,
  ChevronRight,
  Globe,
  Users,
  Brain,
  Layers,
  TrendingUp,
  MessageSquare,
  FileText,
  ArrowRight,
  Play,
  Zap,
  CheckCircle,
  Shield,
} from "lucide-react";
import { useLanguage } from "./contexts/LanguageContext";
import { useDashboardStats, useRecentSpans } from "@/lib/api/hooks";
import { qbmClient } from "@/lib/api/qbm-client";

// Comprehensive translations for HomePage
const HOME_TEXT = {
  ar: {
    // Stats section
    ayatAnalyzed: "آيات محللة",
    datasetCoverage: "تغطية مجموعة البيانات",
    behavioralSpans: "نطاقات سلوكية",
    classifiedAnnotations: "تعليقات مصنفة",
    tafsirSources: "مصادر التفسير",
    classicalScholars: "علماء كلاسيكيون",
    behaviorForms: "أشكال السلوك",
    uniqueClassifications: "تصنيفات فريدة",
    // Features section
    platformFeatures: "ميزات منصة البحث",
    platformDescription: "مصممة للأكاديميين والعلماء والباحثين الذين يستكشفون تقاطع الدراسات الإسلامية واللغويات الحاسوبية.",
    naturalLanguageResearch: "البحث باللغة الطبيعية",
    naturalLanguageDesc: "اطرح أسئلة بالعربية أو الإنجليزية. الذكاء الاصطناعي ينشئ تصورات تفاعلية من استفساراتك.",
    liveDashboard: "لوحة المعلومات المباشرة",
    liveDashboardDesc: "إحصائيات في الوقت الفعلي ومقاييس التغطية ومؤشرات الجودة لمجموعة البيانات بأكملها.",
    quranExplorer: "مستكشف القرآن",
    quranExplorerDesc: "متصفح مرئي للقرآن مع التعليقات السلوكية وتكامل التفسير.",
    tafsirIntegration: "تكامل التفسير",
    tafsirIntegrationDesc: "مقارنة جنبًا إلى جنب لمصادر تفسير متعددة لأي آية.",
    behaviorTaxonomy: "تصنيف السلوك",
    behaviorTaxonomyDesc: "مفاهيم السلوك منظمة وفق إطار البوزيداني ذي السياقات الخمسة.",
    multiTierQuality: "جودة متعددة المستويات",
    multiTierQualityDesc: "مستويات ذهبية وفضية وبحثية مع مقاييس اتفاق المعلقين.",
    // CTA section
    readyToExplore: "هل أنت مستعد لاستكشاف السلوكيات القرآنية؟",
    ctaDescription: "انضم إلى الباحثين حول العالم الذين يستخدمون الذكاء الاصطناعي لاكتشاف الأنماط في توجيهات القرآن حول السلوك البشري.",
    startResearch: "ابدأ البحث",
    viewDashboard: "عرض لوحة المعلومات",
    // Footer
    platform: "المنصة",
    researchAssistant: "مساعد البحث",
    dashboard: "لوحة المعلومات",
    aiInsights: "رؤى الذكاء الاصطناعي",
    resources: "الموارد",
    apiDocumentation: "وثائق API",
    codingManual: "دليل الترميز",
    researchPapers: "أوراق بحثية",
    about: "حول",
    openSourceProject: "مشروع مفتوح المصدر",
    builtForScholarship: "مصمم للدراسات الإسلامية",
    contributionsWelcome: "المساهمات مرحب بها",
    copyright: "© 2025 مصفوفة السلوك القرآني. مصمم للدراسات الإسلامية.",
    behavioralMatrix: "مصفوفة السلوك",
    qbmDescription: "مجموعة بيانات منظمة لتصنيفات السلوك القرآني مبنية على الدراسات الإسلامية.",
    noDataAvailable: "لا توجد بيانات متاحة.",
    noTafsirText: "لا يوجد نص تفسير متاح.",
  },
  en: {
    // Stats section
    ayatAnalyzed: "Ayat Analyzed",
    datasetCoverage: "Dataset coverage",
    behavioralSpans: "Behavioral Spans",
    classifiedAnnotations: "Classified annotations",
    tafsirSources: "Tafsir Sources",
    classicalScholars: "Classical scholars",
    behaviorForms: "Behavior Forms",
    uniqueClassifications: "Unique classifications",
    // Features section
    platformFeatures: "Research Platform Features",
    platformDescription: "Built for academics, scholars, and researchers exploring the intersection of Islamic scholarship and computational linguistics.",
    naturalLanguageResearch: "Natural Language Research",
    naturalLanguageDesc: "Ask questions in plain English or Arabic. Our AI generates interactive visualizations from your queries.",
    liveDashboard: "Live Dashboard",
    liveDashboardDesc: "Real-time statistics, coverage metrics, and quality indicators for the entire dataset.",
    quranExplorer: "Quran Explorer",
    quranExplorerDesc: "Visual browser for the Quran with behavioral annotations and tafsir integration.",
    tafsirIntegration: "Tafsir Integration",
    tafsirIntegrationDesc: "Side-by-side comparison of multiple tafsir sources for any ayah.",
    behaviorTaxonomy: "Behavior Taxonomy",
    behaviorTaxonomyDesc: "Behavior concepts organized by Bouzidani's five-context framework.",
    multiTierQuality: "Multi-Tier Quality",
    multiTierQualityDesc: "Gold, Silver, and Research tiers with inter-annotator agreement metrics.",
    // CTA section
    readyToExplore: "Ready to Explore Quranic Behaviors?",
    ctaDescription: "Join researchers worldwide using AI to uncover patterns in the Quran's guidance on human behavior.",
    startResearch: "Start Research",
    viewDashboard: "View Dashboard",
    // Footer
    platform: "Platform",
    researchAssistant: "Research Assistant",
    dashboard: "Dashboard",
    aiInsights: "AI Insights",
    resources: "Resources",
    apiDocumentation: "API Documentation",
    codingManual: "Coding Manual",
    researchPapers: "Research Papers",
    about: "About",
    openSourceProject: "Open Source Project",
    builtForScholarship: "Built for Islamic Scholarship",
    contributionsWelcome: "Contributions Welcome",
    copyright: "© 2025 Quranic Behavioral Matrix. Built for Islamic scholarship.",
    behavioralMatrix: "Behavioral Matrix",
    qbmDescription: "A structured dataset of Quranic behavioral classifications grounded in Islamic scholarship.",
    noDataAvailable: "No data available.",
    noTafsirText: "No tafsir text available.",
  },
};

type StatsState = {
  totalSpans: number;
  uniqueAyat: number;
  behaviorForms: Record<string, number>;
  agentTypes: Record<string, number>;
  tafsirSources: string[];
  datasetTier: string;
  topSurahs: { surah: number; surah_name?: string; spans: number }[];
};

type SpanSummary = {
  span_id?: string;
  reference?: { surah?: number; ayah?: number; surah_name?: string };
  text_ar?: string;
  behavior_form?: string;
  agent?: { type?: string };
  normative?: { evaluation?: string; deontic_signal?: string };
  annotator?: string;
  annotated_at?: string;
};

// Animated counter component
function AnimatedCounter({
  value,
  suffix = "",
  prefix = "",
}: {
  value: number;
  suffix?: string;
  prefix?: string;
}) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [value]);

  return (
    <span>
      {prefix}
      {count.toLocaleString()}
      {suffix}
    </span>
  );
}

// Featured insight card
function InsightCard({
  icon: Icon,
  title,
  subtitle,
  description,
  metric,
  color,
}: {
  icon: any;
  title: string;
  subtitle?: string;
  description: string;
  metric: string;
  color: string;
}) {
  return (
    <motion.div
      whileHover={{ y: -8, scale: 1.02 }}
      className="insight-card group"
    >
      <div className={`insight-icon ${color}`}>
        <Icon className="w-6 h-6 text-emerald-700" />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 mb-1">{title}</h3>
      {subtitle ? (
        <p className="text-sm text-emerald-600 mb-3">{subtitle}</p>
      ) : null}
      <p className="text-gray-600 text-sm mb-4">{description}</p>
      <div className="flex items-center justify-between">
        <span className="text-2xl font-bold text-emerald-600">{metric}</span>
        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-emerald-500 group-hover:translate-x-1 transition-all" />
      </div>
    </motion.div>
  );
}

// Research example preview
function ResearchExample({ query, resultType, preview }: { query: string; resultType: string; preview: ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow">
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-600">{query}</span>
        </div>
      </div>
      <div className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-4 h-4 text-amber-500" />
          <span className="text-xs font-medium text-amber-600 uppercase">{resultType}</span>
        </div>
        {preview}
      </div>
    </div>
  );
}

export default function HomePage() {
  const { t, isRTL, language } = useLanguage();
  const txt = HOME_TEXT[language as keyof typeof HOME_TEXT] || HOME_TEXT.en;
  const [activeTab, setActiveTab] = useState<'chart' | 'table' | 'comparison'>('chart');
  const [tafsirComparison, setTafsirComparison] = useState<any | null>(null);
  const [exampleQueries, setExampleQueries] = useState<string[]>([]);

  // Load data via hooks
  const { data: backendStats, isLoading: statsLoading } = useDashboardStats();
  const { data: recentData, isLoading: spansLoading } = useRecentSpans(6);

  const isLoading = statsLoading || spansLoading;
  const recentSpans = recentData?.spans || [];
  const featuredSpan = recentSpans[0] || null;

  // Derive stats from backend data
  const stats: StatsState | null = useMemo(() => {
    if (!backendStats) return null;
    return {
      totalSpans: backendStats.total_spans || 0,
      uniqueAyat: backendStats.unique_ayat || 0,
      behaviorForms: backendStats.behavior_forms || {},
      agentTypes: backendStats.agent_types || {},
      tafsirSources: [],
      datasetTier: backendStats.dataset_tier || "",
      topSurahs: backendStats.top_surahs || [],
    };
  }, [backendStats]);

  // Behavior name translations for example queries
  const behaviorNamesAr: Record<string, string> = {
    'inner_state': 'الحالة الداخلية',
    'speech_act': 'الفعل الكلامي',
    'relational_act': 'الفعل العلائقي',
    'physical_act': 'الفعل الجسدي',
    'trait_disposition': 'السمة',
  };

  // Agent name translations for example queries
  const agentNamesAr: Record<string, string> = {
    'AGT_ALLAH': 'الله',
    'AGT_BELIEVER': 'المؤمن',
    'AGT_DISBELIEVER': 'الكافر',
    'AGT_HUMAN_GENERAL': 'الإنسان',
    'AGT_PROPHET': 'النبي',
    'AGT_HYPOCRITE': 'المنافق',
    'AGT_WRONGDOER': 'الظالم',
  };

  useEffect(() => {
    const buildQueries = () => {
      if (!stats) {
        setExampleQueries([]);
        return;
      }

      const queries: string[] = [];
      const topAgent = Object.entries(stats.agentTypes)
        .sort((a, b) => (b[1] as number) - (a[1] as number))[0];
      const topBehavior = Object.entries(stats.behaviorForms)
        .sort((a, b) => (b[1] as number) - (a[1] as number))[0];
      const topSurah = stats.topSurahs[0];
      const featuredRef =
        featuredSpan?.reference?.surah && featuredSpan?.reference?.ayah
          ? `${featuredSpan.reference.surah}:${featuredSpan.reference.ayah}`
          : null;

      if (language === 'ar') {
        // Arabic queries
        if (topBehavior) {
          const behaviorAr = behaviorNamesAr[topBehavior[0]] || topBehavior[0].replace(/_/g, " ");
          queries.push(`حلل سلوكيات ${behaviorAr} في مجموعة البيانات`);
        }
        if (topAgent) {
          const agentAr = agentNamesAr[topAgent[0]] || topAgent[0];
          queries.push(`أظهر السلوكيات المتعلقة بـ ${agentAr}`);
        }
        if (topSurah) {
          queries.push(`تصور توزيع السلوكيات في سورة ${topSurah.surah}`);
        }
        if (featuredRef) {
          queries.push(`قارن التفاسير للآية ${featuredRef}`);
        }
      } else {
        // English queries
        if (topBehavior) {
          queries.push(`Analyze ${topBehavior[0].replace(/_/g, " ")} behaviors in the dataset`);
        }
        if (topAgent) {
          queries.push(`Show behaviors for ${topAgent[0]}`);
        }
        if (topSurah) {
          queries.push(`Visualize behavior distribution in Surah ${topSurah.surah}`);
        }
        if (featuredRef) {
          queries.push(`Compare tafsir for ${featuredRef}`);
        }
      }

      setExampleQueries(queries.slice(0, 4));
    };

    buildQueries();
  }, [stats, featuredSpan, language]);

  useEffect(() => {
    const loadTafsir = async () => {
      // Load tafsir for the static featured verse An-Nahl 16:97
      try {
        const data = await qbmClient.getTafsirComparison(16, 97);
        setTafsirComparison(data);
      } catch (error) {
        console.error("Failed to load tafsir comparison:", error);
      }
    };

    loadTafsir();
  }, []);

  const behaviorForms = stats?.behaviorForms || {};
  const agentTypes = stats?.agentTypes || {};
  const tafsirSourceCount = stats?.tafsirSources?.length ?? null;
  const tafsirEntries = tafsirComparison?.tafsir ? Object.entries(tafsirComparison.tafsir) : [];
  const statItems = [
    { value: stats?.uniqueAyat ?? null, label: txt.ayatAnalyzed, sublabel: txt.datasetCoverage, icon: BookOpen },
    { value: stats?.totalSpans ?? null, label: txt.behavioralSpans, sublabel: txt.classifiedAnnotations, icon: Layers },
    { value: tafsirSourceCount, label: txt.tafsirSources, sublabel: txt.classicalScholars, icon: FileText },
    { value: stats ? Object.keys(behaviorForms).length : null, label: txt.behaviorForms, sublabel: txt.uniqueClassifications, icon: Brain },
  ];
  const innerStateCount = behaviorForms["inner_state"];
  const speechActCount = behaviorForms["speech_act"];
  const relationalActCount = behaviorForms["relational_act"];
  const physicalActCount = behaviorForms["physical_act"];
  const demoQueries = exampleQueries.length
    ? exampleQueries
    : isLoading
      ? ["Loading queries..."]
      : ["No example queries available."];
  const tafsirTitle = "النحل 16:97";
  const totalSpans = stats?.totalSpans ?? 0;
  const agentRows = Object.entries(agentTypes)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .slice(0, 5)
    .map(([label, value]) => ({
      label,
      value: value as number,
      percent: totalSpans ? Math.round((value as number) / totalSpans * 100) : 0,
    }));

  return (
    <div className={`min-h-screen ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* ============================================
          HERO SECTION
          ============================================ */}
      <section className="hero-gradient text-white relative overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 islamic-pattern opacity-10" />
        
        {/* Floating elements */}
        <div className="absolute top-20 left-10 w-20 h-20 bg-white/10 rounded-full blur-xl animate-float" />
        <div className="absolute bottom-20 right-20 w-32 h-32 bg-amber-500/20 rounded-full blur-2xl animate-float" style={{ animationDelay: '2s' }} />
        
        <div className="max-w-7xl mx-auto px-6 py-20 relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: Text content */}
            <div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="flex flex-wrap items-center gap-3 mb-6"
              >
                <span className="inline-flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full text-sm">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  {t("home.tagline")}
                </span>
                <span
                  className="inline-flex items-center gap-2 bg-emerald-900/30 px-4 py-2 rounded-full border border-emerald-500/50 cursor-help"
                  title={language === 'ar'
                    ? "هذا النظام اجتاز 200 سؤال اختبار صارم عبر 10 فئات بدقة 100٪. لا توجد بيانات ملفقة."
                    : "This system passed 200 rigorous test questions across 10 categories with 100% accuracy. No fabricated data."}
                >
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-300 font-medium text-sm">200/200 {language === 'ar' ? 'معتمد' : 'Validated'}</span>
                </span>
              </motion.div>
              
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="text-5xl lg:text-6xl font-bold mb-4"
              >
                {t("home.title")}
              </motion.h1>
              
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="text-2xl text-emerald-200 mb-6"
              >
                {t("home.subtitle")}
              </motion.p>
              
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="text-lg text-emerald-100 mb-8 max-w-xl"
              >
                {t("home.description")}
              </motion.p>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                className="flex flex-wrap gap-4"
              >
                <Link
                  href="/research"
                  className="inline-flex items-center gap-2 bg-white text-emerald-800 px-6 py-3 rounded-xl font-semibold hover:bg-emerald-50 transition-colors shadow-lg"
                >
                  <MessageSquare className="w-5 h-5" />
                  {t("home.start_research")}
                </Link>
                <Link
                  href="/explorer"
                  className="inline-flex items-center gap-2 bg-emerald-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-emerald-500 transition-colors border border-emerald-400"
                >
                  <Play className="w-5 h-5" />
                  {t("home.explore_dataset")}
                </Link>
              </motion.div>
            </div>
            
            {/* Right: Featured Verse - An-Nahl 16:97 */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="relative"
            >
              <div className="verse-container animate-pulse-glow">
                <p className="quran-text-large text-center text-gray-800 mb-4" dir="rtl">
                  ﴿ مَنْ عَمِلَ صَالِحًا مِّن ذَكَرٍ أَوْ أُنثَىٰ وَهُوَ مُؤْمِنٌ فَلَنُحْيِيَنَّهُ حَيَاةً طَيِّبَةً ۖ وَلَنَجْزِيَنَّهُمْ أَجْرَهُم بِأَحْسَنِ مَا كَانُوا يَعْمَلُونَ ﴾
                </p>
                <p className="text-center text-gray-600">
                  سورة النحل ١٦:٩٧
                </p>
                <div className="flex items-center justify-center gap-4 mt-4">
                  <span className="text-sm font-medium text-amber-700 bg-amber-100 px-3 py-1 rounded-full">
                    16:97
                  </span>
                  <span className="behavior-tag positive">
                    <Brain className="w-3 h-3" /> righteous_action
                  </span>
                  <span className="behavior-tag">
                    <Sparkles className="w-3 h-3" /> praise
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ============================================
          STATISTICS BAR
          ============================================ */}
      <section className="bg-white border-b border-gray-200 py-8">
        <div className="max-w-7xl mx-auto px-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {statItems.map((stat, i) => {
              const showValue = !isLoading && stat.value !== null;
              return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="flex justify-center mb-2">
                  <stat.icon className="w-6 h-6 text-emerald-600" />
                </div>
                <div className="text-3xl lg:text-4xl font-bold text-emerald-600">
                  {showValue ? (
                    <AnimatedCounter
                      value={stat.value as number}
                      suffix={(stat.value as number) > 100 ? "+" : ""}
                    />
                  ) : (
                    <span>--</span>
                  )}
                </div>
                <div className="font-medium text-gray-900">{stat.label}</div>
                <div className="text-sm text-gray-500">{stat.sublabel}</div>
              </motion.div>
            );})}
          </div>
        </div>
      </section>

      {/* ============================================
          AI-DISCOVERED INSIGHTS
          ============================================ */}
      <section className="py-20 bg-gray-50 islamic-pattern">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="inline-flex items-center gap-2 bg-amber-100 text-amber-700 px-4 py-2 rounded-full text-sm font-medium mb-4"
            >
              <Sparkles className="w-4 h-4" />
              {t("home.ai_insights")}
            </motion.div>
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              {t("home.what_uncovered")}
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              {t("home.insights_description")}
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <InsightCard
              icon={Brain}
              title={t("home.inner_states")}
              description={t("home.inner_states_desc")}
              metric={typeof innerStateCount === "number" ? innerStateCount.toLocaleString() : "--"}
              color="bg-gradient-to-br from-rose-100 to-rose-200"
            />
            <InsightCard
              icon={MessageSquare}
              title={t("home.speech_acts")}
              description={t("home.speech_acts_desc")}
              metric={typeof speechActCount === "number" ? speechActCount.toLocaleString() : "--"}
              color="bg-gradient-to-br from-blue-100 to-blue-200"
            />
            <InsightCard
              icon={Users}
              title={t("home.relational_acts")}
              description={t("home.relational_acts_desc")}
              metric={typeof relationalActCount === "number" ? relationalActCount.toLocaleString() : "--"}
              color="bg-gradient-to-br from-purple-100 to-purple-200"
            />
            <InsightCard
              icon={TrendingUp}
              title={t("home.physical_acts")}
              description={t("home.physical_acts_desc")}
              metric={typeof physicalActCount === "number" ? physicalActCount.toLocaleString() : "--"}
              color="bg-gradient-to-br from-amber-100 to-amber-200"
            />
          </div>
        </div>
      </section>

      {/* ============================================
          LIVE RESEARCH DEMO
          ============================================ */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <span className="inline-flex items-center gap-2 bg-emerald-100 text-emerald-700 px-4 py-2 rounded-full text-sm font-medium mb-4">
                  <Zap className="w-4 h-4" />
                  {t("home.powered_by")}
                </span>
                <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
                  {t("home.ask_anything")}
                </h2>
                <p className="text-gray-600 mb-6">
                  {t("home.ask_description")}
                </p>
                
                <div className="space-y-3 mb-8">
                  {demoQueries.map((query, i) => (
                    <div key={i} className="flex items-center gap-3 text-gray-600">
                      <div className="w-6 h-6 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 text-xs font-medium">
                        {i + 1}
                      </div>
                      <span className="text-sm">"{query}"</span>
                    </div>
                  ))}
                </div>
                
                <Link
                  href="/research"
                  className="inline-flex items-center gap-2 bg-emerald-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-emerald-700 transition-colors"
                >
                  {t("home.try_research")}
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </motion.div>
            </div>
            
            {/* Demo visualization */}
            <div className="space-y-4">
              {/* Tab selector */}
              <div className="flex gap-2 bg-gray-100 p-1 rounded-lg w-fit">
                {(['chart', 'table', 'comparison'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      activeTab === tab
                        ? "bg-white text-emerald-700 shadow"
                        : "text-gray-600 hover:text-gray-900"
                    }`}
                  >
                    {t(`home.${tab}`)}
                  </button>
                ))}
              </div>
              
              {/* Demo content */}
              <div className="chart-container h-80">
                {activeTab === 'chart' && (
                  <div className="h-full">
                    <h3 className="font-semibold text-gray-900 mb-4">{t("dashboard.agent_distribution")}</h3>
                    <div className="space-y-3">
                      {agentRows.length ? (
                        agentRows.map((item) => (
                          <div key={item.label} className="flex items-center gap-4">
                            <div className="w-28 text-sm text-gray-600">{item.label}</div>
                            <div className="flex-1 h-8 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                style={{ width: `${item.percent}%` }}
                                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-300"
                              />
                            </div>
                            <div className="w-16 text-right text-sm font-medium text-gray-900">
                              {item.value.toLocaleString()}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-sm text-gray-500">No data available.</div>
                      )}
                    </div>
                  </div>
                )}
                
                {activeTab === 'table' && (
                  <div className="h-full overflow-auto">
                    <h3 className="font-semibold text-gray-900 mb-4">{t("home.recent_annotations")}</h3>
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left">{t("home.reference")}</th>
                          <th className="px-3 py-2 text-left">{t("home.behavior")}</th>
                          <th className="px-3 py-2 text-left">{t("home.agent")}</th>
                          <th className="px-3 py-2 text-left">{t("home.annotator")}</th>
                          <th className="px-3 py-2 text-left">{t("home.annotated")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {isLoading ? (
                          <tr>
                            <td colSpan={5} className="text-center py-4 text-gray-500">
                              {t("common.loading")}
                            </td>
                          </tr>
                        ) : recentSpans.length ? (
                          recentSpans.slice(0, 4).map((span, i) => {
                            const reference =
                              span.reference?.surah && span.reference?.ayah
                                ? `${span.reference.surah}:${span.reference.ayah}`
                                : "--";
                            const annotatedAt = span.annotated_at ? new Date(span.annotated_at) : null;
                            const annotatedLabel =
                              annotatedAt && !Number.isNaN(annotatedAt.getTime())
                                ? annotatedAt.toLocaleDateString()
                                : "--";
                            return (
                              <tr
                                key={span.span_id || `${reference}-${i}`}
                                className="border-t border-gray-100 hover:bg-gray-50"
                              >
                                <td className="px-3 py-2 font-medium text-emerald-600">{reference}</td>
                                <td className="px-3 py-2">{span.behavior_form || "--"}</td>
                                <td className="px-3 py-2">{span.agent?.type || "--"}</td>
                                <td className="px-3 py-2">{span.annotator || "--"}</td>
                                <td className="px-3 py-2 text-gray-500">{annotatedLabel}</td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr>
                            <td colSpan={5} className="text-center py-4 text-gray-500">
                              {t("common.no_data")}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
                {activeTab === 'comparison' && (
                  <div className="h-full">
                    <h3 className="font-semibold text-gray-900 mb-4">
                      {t("home.tafsir_comparison")}: {tafsirTitle}
                    </h3>
                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 h-56 overflow-auto">
                      {tafsirEntries.length ? (
                        tafsirEntries.map(([key, entry], idx) => {
                          const tafsirEntry = entry as { source_name?: string; text?: string };
                          const label = tafsirEntry.source_name || key;
                          const colors = [
                            "bg-emerald-50 text-emerald-800",
                            "bg-amber-50 text-amber-800",
                            "bg-purple-50 text-purple-800",
                            "bg-blue-50 text-blue-800",
                            "bg-rose-50 text-rose-800",
                          ];
                          const colorClass = colors[idx % colors.length];
                          return (
                            <div key={key} className={`${colorClass.split(' ')[0]} rounded-lg p-3`}>
                              <div className={`font-medium ${colorClass.split(' ')[1]} mb-2`}>{label}</div>
                              <p className="text-sm text-gray-600 font-arabic text-xs" dir="rtl">
                                {tafsirEntry.text
                                  ? tafsirEntry.text.slice(0, 200) + "..."
                                  : "No tafsir text available."}
                              </p>
                            </div>
                          );
                        })
                      ) : (
                        <div className="col-span-3 text-sm text-gray-500">
                          {t("home.no_tafsir")}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================
          FEATURES GRID
          ============================================ */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              {txt.platformFeatures}
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              {txt.platformDescription}
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: MessageSquare,
                title: txt.naturalLanguageResearch,
                description: txt.naturalLanguageDesc,
                link: "/research",
                color: "bg-emerald-500",
              },
              {
                icon: BarChart3,
                title: txt.liveDashboard,
                description: txt.liveDashboardDesc,
                link: "/dashboard",
                color: "bg-blue-500",
              },
              {
                icon: Globe,
                title: txt.quranExplorer,
                description: txt.quranExplorerDesc,
                link: "/explorer",
                color: "bg-purple-500",
              },
              {
                icon: FileText,
                title: txt.tafsirIntegration,
                description: txt.tafsirIntegrationDesc,
                link: "/research",
                color: "bg-amber-500",
              },
              {
                icon: Brain,
                title: txt.behaviorTaxonomy,
                description: txt.behaviorTaxonomyDesc,
                link: "/insights",
                color: "bg-rose-500",
              },
              {
                icon: Layers,
                title: txt.multiTierQuality,
                description: txt.multiTierQualityDesc,
                link: "/dashboard",
                color: "bg-cyan-500",
              },
            ].map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
              >
                <Link href={feature.link} className="block h-full">
                  <div className="stat-card h-full group cursor-pointer">
                    <div className={`w-12 h-12 ${feature.color} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
                    <p className="text-gray-600 text-sm">{feature.description}</p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ============================================
          CTA SECTION
          ============================================ */}
      <section className="py-20 hero-gradient text-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-4">
              {txt.readyToExplore}
            </h2>
            <p className="text-emerald-100 text-lg mb-8 max-w-2xl mx-auto">
              {txt.ctaDescription}
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/research"
                className="inline-flex items-center gap-2 bg-white text-emerald-800 px-8 py-4 rounded-xl font-semibold hover:bg-emerald-50 transition-colors shadow-lg"
              >
                <MessageSquare className="w-5 h-5" />
                {txt.startResearch}
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 bg-transparent border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white/10 transition-colors"
              >
                <BarChart3 className="w-5 h-5" />
                {txt.viewDashboard}
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ============================================
          FOOTER
          ============================================ */}
      <footer className="bg-gray-900 text-gray-400 py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold">Q</span>
                </div>
                <div>
                  <div className="font-bold text-white">QBM</div>
                  <div className="text-xs text-gray-500">{txt.behavioralMatrix}</div>
                </div>
              </div>
              <p className="text-sm">
                {txt.qbmDescription}
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">{txt.platform}</h4>
              <ul className="space-y-2 text-sm">
                <li><Link href="/research" className="hover:text-emerald-400">{txt.researchAssistant}</Link></li>
                <li><Link href="/dashboard" className="hover:text-emerald-400">{txt.dashboard}</Link></li>
                <li><Link href="/explorer" className="hover:text-emerald-400">{txt.quranExplorer}</Link></li>
                <li><Link href="/insights" className="hover:text-emerald-400">{txt.aiInsights}</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">{txt.resources}</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#" className="hover:text-emerald-400">{txt.apiDocumentation}</a></li>
                <li><a href="#" className="hover:text-emerald-400">{txt.codingManual}</a></li>
                <li><a href="#" className="hover:text-emerald-400">{txt.researchPapers}</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">{txt.about}</h4>
              <ul className="space-y-2 text-sm">
                <li>{txt.openSourceProject}</li>
                <li>{txt.builtForScholarship}</li>
                <li>{txt.contributionsWelcome}</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm">
            <p>{txt.copyright}</p>
            <p className="mt-2 text-emerald-400">
              {language === 'ar' ? 'تم تطويره بالكامل بواسطة سالم البرعمي' : 'Developed end-to-end by Salim Albarami'}
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
