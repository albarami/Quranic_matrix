"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import {
  Sparkles,
  Brain,
  TrendingUp,
  GitCompare,
  Lightbulb,
  BookOpen,
  ArrowRight,
  ChevronRight,
  Network,
  Target,
  X,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import { useDashboardStats } from "@/lib/api/hooks";

const CHART_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

// Insights will be generated from real backend data
interface InsightData {
  id: string;
  title: string;
  description: string;
  icon: any;
  color: string;
  metric: string;
  category: string;
}

// Translations for Insights page
const INSIGHTS_TEXT = {
  en: {
    aiPowered: "AI-Powered Discovery",
    title: "Research Insights",
    subtitle: "Patterns and discoveries uncovered through computational analysis of",
    annotations: "behavioral annotations across the complete Quran.",
    totalSpans: "Total Spans",
    behaviorForms: "Behavior Forms",
    insightsGenerated: "Insights Generated",
    dataSource: "Data Source",
    featuredDiscoveries: "Featured Discoveries",
    methodology: "Research Methodology",
    methodologyText: "These insights are generated through statistical analysis of the QBM dataset, combining NLP techniques with classical tafsir validation. Each pattern has been verified against at least 3 tafsir sources for scholarly accuracy.",
    keyFinding: "Key Finding",
    dataDistribution: "Data Distribution",
    methodologyNote: "This insight was discovered through statistical analysis of",
    methodologyNote2: "behavioral annotations across the complete Quran. The data has been validated against classical tafsir sources for scholarly accuracy.",
    close: "Close",
    viewFullProof: "View Full Proof",
    // Insight titles
    innerStatesDominate: "Inner States Dominate",
    speechActsAnalysis: "Speech Acts Analysis",
    divineAgency: "Divine Agency in Guidance",
    believerDisbeliever: "Believer vs Disbeliever",
    relationalBehaviors: "Relational Behaviors",
    // Categories
    behaviorFormsCategory: "Behavior Forms",
    agentAnalysis: "Agent Analysis",
    agentComparison: "Agent Comparison",
  },
  ar: {
    aiPowered: "اكتشاف بالذكاء الاصطناعي",
    title: "رؤى البحث",
    subtitle: "أنماط واكتشافات تم الكشف عنها من خلال التحليل الحسابي لـ",
    annotations: "تعليق سلوكي عبر القرآن الكريم كاملاً.",
    totalSpans: "إجمالي النطاقات",
    behaviorForms: "أشكال السلوك",
    insightsGenerated: "الرؤى المُنشأة",
    dataSource: "مصدر البيانات",
    featuredDiscoveries: "الاكتشافات المميزة",
    methodology: "منهجية البحث",
    methodologyText: "يتم إنشاء هذه الرؤى من خلال التحليل الإحصائي لمجموعة بيانات QBM، بدمج تقنيات معالجة اللغة الطبيعية مع التحقق من التفاسير الكلاسيكية. تم التحقق من كل نمط مقابل 3 مصادر تفسير على الأقل للدقة العلمية.",
    keyFinding: "النتيجة الرئيسية",
    dataDistribution: "توزيع البيانات",
    methodologyNote: "تم اكتشاف هذه الرؤية من خلال التحليل الإحصائي لـ",
    methodologyNote2: "تعليق سلوكي عبر القرآن الكريم كاملاً. تم التحقق من البيانات مقابل مصادر التفسير الكلاسيكية للدقة العلمية.",
    close: "إغلاق",
    viewFullProof: "عرض الإثبات الكامل",
    // Insight titles
    innerStatesDominate: "هيمنة الحالات الداخلية",
    speechActsAnalysis: "تحليل الأفعال الكلامية",
    divineAgency: "الفاعلية الإلهية في الهداية",
    believerDisbeliever: "المؤمن مقابل الكافر",
    relationalBehaviors: "السلوكيات العلائقية",
    // Categories
    behaviorFormsCategory: "أشكال السلوك",
    agentAnalysis: "تحليل الفاعل",
    agentComparison: "مقارنة الفاعلين",
  }
};

export default function InsightsPage() {
  const { language, isRTL } = useLanguage();
  const txt = INSIGHTS_TEXT[language as 'en' | 'ar'] || INSIGHTS_TEXT.en;
  const [selectedInsight, setSelectedInsight] = useState<InsightData | null>(null);

  // Load stats via hook
  const { data: backendStats, isLoading } = useDashboardStats();

  // Derive insights from backend stats
  const { insights, stats } = useMemo(() => {
    if (!backendStats) {
      return { insights: [] as InsightData[], stats: { totalSpans: 0, patterns: 0, datasetTier: "" } };
    }

    const totalSpans = backendStats.total_spans || 0;
    const agentTypes = backendStats.agent_types || {};
    const behaviorForms = backendStats.behavior_forms || {};
    const pct = (count: number) =>
      totalSpans ? ((count / totalSpans) * 100).toFixed(1) : "0.0";

    const innerStateCount = behaviorForms["inner_state"] || 0;
    const speechActCount = behaviorForms["speech_act"] || 0;
    const relationalActCount = behaviorForms["relational_act"] || 0;
    const divineCount = agentTypes["AGT_ALLAH"] || 0;
    const believerCount = agentTypes["AGT_BELIEVER"] || 0;
    const disbelieverCount = agentTypes["AGT_DISBELIEVER"] || 0;

    const realInsights: InsightData[] = [
      {
        id: "inner-states",
        title: "inner-states",
        description: `${pct(innerStateCount)}%`,
        icon: Brain,
        color: "from-rose-500 to-pink-600",
        metric: `${innerStateCount.toLocaleString()}`,
        category: "behavior-forms",
      },
      {
        id: "speech-acts",
        title: "speech-acts",
        description: `${pct(speechActCount)}%`,
        icon: TrendingUp,
        color: "from-blue-500 to-cyan-600",
        metric: `${speechActCount.toLocaleString()}`,
        category: "behavior-forms",
      },
      {
        id: "divine-agency",
        title: "divine-agency",
        description: `${pct(divineCount)}%`,
        icon: Target,
        color: "from-amber-500 to-orange-600",
        metric: `${pct(divineCount)}%`,
        category: "agent-analysis",
      },
      {
        id: "believer-disbeliever",
        title: "believer-disbeliever",
        description: `${believerCount.toLocaleString()}|${disbelieverCount.toLocaleString()}`,
        icon: GitCompare,
        color: "from-purple-500 to-indigo-600",
        metric: `${(believerCount + disbelieverCount).toLocaleString()}`,
        category: "agent-comparison",
      },
      {
        id: "relational-acts",
        title: "relational-acts",
        description: `${pct(relationalActCount)}%`,
        icon: Network,
        color: "from-emerald-500 to-teal-600",
        metric: `${relationalActCount.toLocaleString()}`,
        category: "behavior-forms",
      },
    ];

    return {
      insights: realInsights,
      stats: {
        totalSpans,
        patterns: Object.keys(behaviorForms).length,
        datasetTier: backendStats.dataset_tier || "",
      },
    };
  }, [backendStats]);

  // Prepare chart data for selected insight
  const getInsightChartData = (insight: InsightData | null) => {
    if (!insight || !backendStats) return [];
    
    if (insight.id === 'inner-states' || insight.id === 'speech-acts' || insight.id === 'relational-acts') {
      // Behavior forms chart
      const forms = backendStats.behavior_forms || {};
      return Object.entries(forms)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 6)
        .map(([name, value], i) => ({ name: name.replace(/_/g, ' '), value, fill: CHART_COLORS[i % CHART_COLORS.length] }));
    } else if (insight.id === 'divine-agency' || insight.id === 'believer-disbeliever') {
      // Agent types chart
      const agents = backendStats.agent_types || {};
      return Object.entries(agents)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 6)
        .map(([name, value], i) => ({ name, value, fill: CHART_COLORS[i % CHART_COLORS.length] }));
    }
    return [];
  };

  const chartData = getInsightChartData(selectedInsight);

  return (
      <div className={`min-h-[calc(100vh-64px)] bg-gray-50 ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
        {/* Header */}
        <div className="bg-gradient-to-br from-purple-700 via-purple-800 to-indigo-900 text-white">
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-white/10 rounded-lg">
                <Sparkles className="w-6 h-6 text-amber-400" />
              </div>
              <span className="text-purple-200 font-medium">{txt.aiPowered}</span>
            </div>
            <h1 className="text-4xl font-bold mb-4">{txt.title}</h1>
            <p className="text-purple-200 max-w-2xl text-lg">
              {txt.subtitle} {stats.totalSpans.toLocaleString()} {txt.annotations}
            </p>

            {/* Quick stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              {[
                { label: txt.totalSpans, value: stats.totalSpans.toLocaleString() },
                { label: txt.behaviorForms, value: stats.patterns.toString() },
                { label: txt.insightsGenerated, value: insights.length.toString() },
                { label: txt.dataSource, value: stats.datasetTier ? (language === 'ar' ? (stats.datasetTier === 'silver' ? 'فضي' : stats.datasetTier === 'gold' ? 'ذهبي' : stats.datasetTier) : stats.datasetTier.toUpperCase()) : "--" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-white/10 backdrop-blur rounded-lg p-4 text-center"
                >
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <div className="text-sm text-purple-200">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex gap-8">
            {/* Insights List */}
            <div className={`${selectedInsight ? "w-1/3" : "w-full"} transition-all`}>
              <div className="flex items-center gap-2 mb-6">
                <Lightbulb className="w-5 h-5 text-amber-500" />
                <h2 className="text-xl font-semibold text-gray-900">
                  {txt.featuredDiscoveries}
                </h2>
              </div>

              <div
                className={`grid ${
                  selectedInsight ? "grid-cols-1" : "md:grid-cols-2 lg:grid-cols-3"
                } gap-4`}
              >
                {insights.map((insight, i) => (
                  <motion.button
                    key={insight.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: i * 0.1 }}
                    onClick={() => setSelectedInsight(insight)}
                    className={`text-left bg-white rounded-xl border overflow-hidden transition-all hover:shadow-lg group ${
                      selectedInsight?.id === insight.id
                        ? "border-purple-500 shadow-lg"
                        : "border-gray-200 hover:border-purple-300"
                    }`}
                  >
                    {/* Gradient header */}
                    <div className={`h-2 bg-gradient-to-r ${insight.color}`} />

                    <div className="p-5">
                      <div className="flex items-start justify-between mb-3">
                        <div
                          className={`p-2 rounded-lg bg-gradient-to-br ${insight.color} bg-opacity-10`}
                        >
                          <insight.icon className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                          {insight.category === 'behavior-forms' ? txt.behaviorFormsCategory : 
                           insight.category === 'agent-analysis' ? txt.agentAnalysis : 
                           insight.category === 'agent-comparison' ? txt.agentComparison : insight.category}
                        </span>
                      </div>

                      <h3 className="font-semibold text-gray-900 mb-1 group-hover:text-purple-700">
                        {insight.title === 'inner-states' ? txt.innerStatesDominate :
                         insight.title === 'speech-acts' ? txt.speechActsAnalysis :
                         insight.title === 'divine-agency' ? txt.divineAgency :
                         insight.title === 'believer-disbeliever' ? txt.believerDisbeliever :
                         insight.title === 'relational-acts' ? txt.relationalBehaviors : insight.title}
                      </h3>
                      {!selectedInsight && (
                        <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                          {insight.description}
                        </p>
                      )}

                      <div className="flex items-center justify-between">
                        <span className="text-lg font-bold text-gray-900">
                          {insight.metric} {language === 'ar' ? 'نطاق' : 'spans'}
                        </span>
                        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-purple-500 group-hover:translate-x-1 transition-all" />
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>

              {/* Methodology note */}
              {!selectedInsight && (
                <div className="mt-8 p-6 bg-purple-50 rounded-xl border border-purple-100">
                  <h3 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
                    <BookOpen className="w-5 h-5" />
                    {txt.methodology}
                  </h3>
                  <p className="text-sm text-purple-700">
                    {txt.methodologyText}
                  </p>
                </div>
              )}
            </div>

            {/* Detail Panel */}
            {selectedInsight && (
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                className="w-2/3 bg-white rounded-2xl border border-gray-200 overflow-hidden"
              >
                {/* Header */}
                <div
                  className={`bg-gradient-to-r ${selectedInsight.color} text-white p-6`}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <selectedInsight.icon className="w-8 h-8" />
                    <span className="text-white/80 text-sm font-medium">
                      {selectedInsight.category === 'behavior-forms' ? txt.behaviorFormsCategory : 
                       selectedInsight.category === 'agent-analysis' ? txt.agentAnalysis : 
                       selectedInsight.category === 'agent-comparison' ? txt.agentComparison : selectedInsight.category}
                    </span>
                  </div>
                  <h2 className="text-2xl font-bold mb-1">
                    {selectedInsight.title === 'inner-states' ? txt.innerStatesDominate :
                     selectedInsight.title === 'speech-acts' ? txt.speechActsAnalysis :
                     selectedInsight.title === 'divine-agency' ? txt.divineAgency :
                     selectedInsight.title === 'believer-disbeliever' ? txt.believerDisbeliever :
                     selectedInsight.title === 'relational-acts' ? txt.relationalBehaviors : selectedInsight.title}
                  </h2>
                </div>

                {/* Content - Native Charts */}
                <div className="p-6 max-h-[calc(100vh-350px)] overflow-y-auto custom-scrollbar">
                  {/* Key Finding */}
                  <div className="mb-6 p-4 bg-purple-50 rounded-xl border border-purple-100">
                    <h3 className="font-bold text-purple-900 mb-2">{txt.keyFinding}</h3>
                    <p className="text-purple-700">{selectedInsight.description}</p>
                    <p className="text-2xl font-bold text-purple-900 mt-3">{selectedInsight.metric} {language === 'ar' ? 'نطاق' : 'spans'}</p>
                  </div>

                  {/* Chart Visualization */}
                  {chartData.length > 0 && (
                    <div className="mb-6">
                      <h4 className="font-bold text-gray-800 mb-3">{txt.dataDistribution}</h4>
                      <div className="bg-gray-50 rounded-xl p-4">
                        {selectedInsight.id.includes('agent') || selectedInsight.id.includes('believer') ? (
                          <ResponsiveContainer width="100%" height={250}>
                            <PieChart>
                              <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={50}
                                outerRadius={90}
                                paddingAngle={5}
                                dataKey="value"
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                              >
                                {chartData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.fill} />
                                ))}
                              </Pie>
                              <Tooltip />
                            </PieChart>
                          </ResponsiveContainer>
                        ) : (
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={chartData} layout="vertical">
                              <XAxis type="number" />
                              <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                              <Tooltip />
                              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {chartData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.fill} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Methodology Note */}
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <h4 className="font-bold text-gray-800 mb-2">{txt.methodology}</h4>
                    <p className="text-sm text-gray-600">
                      {txt.methodologyNote} {stats.totalSpans.toLocaleString()} {txt.methodologyNote2}
                    </p>
                  </div>
                </div>

                {/* Footer actions */}
                <div className="border-t border-gray-200 p-4 flex items-center justify-between bg-gray-50">
                  <button
                    onClick={() => setSelectedInsight(null)}
                    className="text-gray-600 hover:text-gray-900 flex items-center gap-2"
                  >
                    <X className="w-4 h-4" />
                    {txt.close}
                  </button>
                  <div className="flex items-center gap-2">
                    <a 
                      href={`/proof?q=${encodeURIComponent(selectedInsight.title)}`}
                      className="px-4 py-2 text-sm font-medium bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
                    >
                      {txt.viewFullProof}
                      <ArrowRight className="w-4 h-4" />
                    </a>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
  );
}
