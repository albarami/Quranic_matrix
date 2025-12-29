"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, Legend } from "recharts";
import {
  RefreshCw,
  Download,
  Calendar,
  TrendingUp,
  BarChart3,
  Activity,
  Sparkles,
  CheckCircle2,
  Clock,
  Users,
  Brain,
  BookOpen,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const CHART_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

type SpanSummary = {
  span_id?: string;
  reference?: { surah?: number; ayah?: number; surah_name?: string };
  behavior_form?: string;
  agent?: { type?: string };
  annotator?: string;
  annotated_at?: string;
};

// Stats will be loaded from real backend
interface QuickStat {
  label: string;
  value: string;
  subtext: string;
  icon: any;
  color: string;
  bgColor: string;
  trend?: string;
}

const DEFAULT_STATS: QuickStat[] = [
  {
    label: "Quran Coverage",
    value: "Loading...",
    subtext: "Fetching from database",
    icon: CheckCircle2,
    color: "text-emerald-600",
    bgColor: "bg-emerald-50",
  },
  {
    label: "Total Spans",
    value: "Loading...",
    subtext: "Fetching from database",
    icon: Brain,
    color: "text-blue-600",
    bgColor: "bg-blue-50",
  },
  {
    label: "Unique Surahs",
    value: "Loading...",
    subtext: "With annotations",
    icon: Activity,
    color: "text-amber-600",
    bgColor: "bg-amber-50",
  },
  {
    label: "Unique Ayat",
    value: "Loading...",
    subtext: "Annotated verses",
    icon: TrendingUp,
    color: "text-purple-600",
    bgColor: "bg-purple-50",
  },
];

// Dashboard section definitions - now rendered natively with Recharts
interface DashboardSection {
  id: string;
  title: string;
  icon: any;
}


// Dashboard translations
const DASHBOARD_TEXT = {
  en: {
    coverageProgress: "Coverage Progress",
    behaviorDistribution: "Behavior Distribution",
    agentAnalysis: "Agent Analysis",
    topSurahs: "Top Surahs by Annotations",
    recentActivity: "Recent Activity",
    viewAll: "View all",
    reference: "Reference",
    behavior: "Behavior",
    agent: "Agent",
    annotator: "Annotator",
    tier: "Tier",
    time: "Time",
    noRecentSpans: "No recent spans available.",
    loadingData: "Loading data...",
    noBehaviorData: "No behavior data available",
    noAgentData: "No agent data available",
    noSurahData: "No surah data available",
    ayat: "ayat",
    behavioralSpans: "behavioral spans",
    quranCoverage: "Quran Coverage",
    totalSpans: "Total Spans",
    uniqueSurahs: "Unique Surahs",
    uniqueAyat: "Unique Ayat",
    behavioralAnnotations: "Behavioral annotations",
    withAnnotations: "With annotations",
    annotatedVerses: "Annotated verses",
    of: "of",
    updated: "Updated",
    silver: "SILVER",
    gold: "GOLD",
  },
  ar: {
    coverageProgress: "تقدم التغطية",
    behaviorDistribution: "توزيع السلوكيات",
    agentAnalysis: "تحليل الفاعلين",
    topSurahs: "أكثر السور تعليقاً",
    recentActivity: "النشاط الأخير",
    viewAll: "عرض الكل",
    reference: "المرجع",
    behavior: "السلوك",
    agent: "الفاعل",
    annotator: "المُعلِّق",
    tier: "المستوى",
    time: "الوقت",
    noRecentSpans: "لا توجد نطاقات حديثة.",
    loadingData: "جاري تحميل البيانات...",
    noBehaviorData: "لا توجد بيانات سلوكية",
    noAgentData: "لا توجد بيانات فاعلين",
    noSurahData: "لا توجد بيانات سور",
    ayat: "آية",
    behavioralSpans: "نطاق سلوكي",
    quranCoverage: "تغطية القرآن",
    totalSpans: "إجمالي النطاقات",
    uniqueSurahs: "السور الفريدة",
    uniqueAyat: "الآيات الفريدة",
    behavioralAnnotations: "تعليقات سلوكية",
    withAnnotations: "مع تعليقات",
    annotatedVerses: "آيات مُعلَّقة",
    of: "من",
    updated: "تم التحديث",
    silver: "فضي",
    gold: "ذهبي",
  }
};

export default function DashboardPage() {
  const { t, isRTL, language } = useLanguage();
  const txt = DASHBOARD_TEXT[language as 'en' | 'ar'] || DASHBOARD_TEXT.en;
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [quickStats, setQuickStats] = useState<QuickStat[]>(DEFAULT_STATS);
  const [backendStats, setBackendStats] = useState<any>(null);
  const [recentSpans, setRecentSpans] = useState<SpanSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load real stats from backend on mount
  useEffect(() => {
    loadRealStats();
  }, []);

  const loadRealStats = async () => {
    try {
      const [statsRes, recentRes] = await Promise.all([
        fetch(`${BACKEND_URL}/stats`),
        fetch(`${BACKEND_URL}/spans/recent?limit=5`),
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        const totalAyat = data.total_ayat ?? data.unique_ayat ?? 0;
        const uniqueAyat = data.unique_ayat ?? 0;
        const coveragePct =
          data.coverage_pct ??
          (totalAyat ? Math.round((uniqueAyat / totalAyat) * 1000) / 10 : 0);

        const coverageValue = totalAyat ? `${coveragePct}%` : "--";
        const coverageSubtext = totalAyat
          ? `${uniqueAyat.toLocaleString()} of ${totalAyat.toLocaleString()} ayat`
          : `${uniqueAyat.toLocaleString()} ayat`;

        setQuickStats([
          {
            label: "quranCoverage",
            value: coverageValue,
            subtext: coverageSubtext,
            icon: CheckCircle2,
            color: "text-emerald-600",
            bgColor: "bg-emerald-50",
          },
          {
            label: "totalSpans",
            value: (data.total_spans || 0).toLocaleString(),
            subtext: "behavioralAnnotations",
            icon: Brain,
            color: "text-blue-600",
            bgColor: "bg-blue-50",
          },
          {
            label: "uniqueSurahs",
            value: (data.unique_surahs || 0).toString(),
            subtext: "withAnnotations",
            icon: Activity,
            color: "text-amber-600",
            bgColor: "bg-amber-50",
          },
          {
            label: "uniqueAyat",
            value: (uniqueAyat || 0).toLocaleString(),
            subtext: "annotatedVerses",
            icon: TrendingUp,
            color: "text-purple-600",
            bgColor: "bg-purple-50",
          },
        ]);
        setBackendStats(data);
      }

      if (recentRes.ok) {
        const recentData = await recentRes.json();
        setRecentSpans(recentData.spans || []);
      }
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Failed to load stats:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare chart data from backend stats
  const behaviorChartData = backendStats?.behavior_forms
    ? Object.entries(backendStats.behavior_forms)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 8)
        .map(([name, value], i) => ({ name: name.replace(/_/g, ' '), value, fill: CHART_COLORS[i % CHART_COLORS.length] }))
    : [];

  const agentChartData = backendStats?.agent_types
    ? Object.entries(backendStats.agent_types)
        .sort((a, b) => (b[1] as number) - (a[1] as number))
        .slice(0, 6)
        .map(([name, value], i) => ({ name, value, fill: CHART_COLORS[i % CHART_COLORS.length] }))
    : [];

  const evaluationChartData = backendStats?.evaluations
    ? Object.entries(backendStats.evaluations)
        .map(([name, value], i) => ({ name, value, fill: CHART_COLORS[i % CHART_COLORS.length] }))
    : [];

  const surahChartData = backendStats?.top_surahs
    ? backendStats.top_surahs.slice(0, 10).map((s: any, i: number) => ({
        name: `${s.surah}`,
        spans: s.spans,
        fill: CHART_COLORS[i % CHART_COLORS.length],
      }))
    : [];

  const coveragePct = backendStats
    ? backendStats.coverage_pct ?? 
      (backendStats.total_ayat ? Math.round((backendStats.unique_ayat / backendStats.total_ayat) * 100) : 0)
    : 0;

  const tierLabel = backendStats?.dataset_tier
    ? String(backendStats.dataset_tier).toUpperCase()
    : "--";

  return (
      <div className={`min-h-[calc(100vh-64px)] bg-gray-50 ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-700 to-blue-800 text-white">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold flex items-center gap-3">
                  <BarChart3 className="w-8 h-8" />
                  {t("dashboard.title")}
                </h1>
                <p className="text-blue-200 mt-1">
                  {t("dashboard.subtitle")}
                </p>
              </div>
              <div className="flex items-center gap-4">
                {lastUpdated && (
                  <div className="flex items-center gap-2 text-blue-200 text-sm">
                    <Clock className="w-4 h-4" />
                    {txt.updated} {lastUpdated.toLocaleTimeString()}
                  </div>
                )}
                <button
                  onClick={loadRealStats}
                  disabled={isLoading}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2 rounded-lg transition-colors"
                >
                  <RefreshCw
                    className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`}
                  />
                  {t("dashboard.refresh")}
                </button>
                <button className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors">
                  <Download className="w-4 h-4" />
                  {t("dashboard.export")}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="max-w-7xl mx-auto px-6 -mt-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {quickStats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="bg-white rounded-xl p-5 shadow-lg border border-gray-100"
              >
                <div className="flex items-start justify-between">
                  <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                    <stat.icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                  {stat.trend && (
                    <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full">
                      {stat.trend}
                    </span>
                  )}
                </div>
                <div className="mt-4">
                  <div className={`text-3xl font-bold ${stat.color}`}>{stat.value}</div>
                  <div className="text-sm font-medium text-gray-900 mt-1">{(txt as any)[stat.label] || stat.label}</div>
                  <div className="text-xs text-gray-500">{(txt as any)[stat.subtext] || stat.subtext}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Dashboard Grid - Native Charts */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Coverage Progress */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                  {txt.coverageProgress}
                </h3>
              </div>
              <div className="p-6 min-h-[300px] flex items-center justify-center">
                {isLoading ? (
                  <Sparkles className="w-8 h-8 text-emerald-500 animate-pulse" />
                ) : (
                  <div className="text-center">
                    <div className="relative w-40 h-40 mx-auto mb-4">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="80" cy="80" r="70" fill="none" stroke="#e5e7eb" strokeWidth="12" />
                        <circle
                          cx="80" cy="80" r="70" fill="none" stroke="#10b981" strokeWidth="12"
                          strokeDasharray={`${coveragePct * 4.4} 440`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-4xl font-bold text-emerald-600">{coveragePct}%</span>
                      </div>
                    </div>
                    <p className="text-gray-600">
                      {backendStats?.unique_ayat?.toLocaleString() || 0} / {backendStats?.total_ayat?.toLocaleString() || '6,236'} {txt.ayat}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      {backendStats?.total_spans?.toLocaleString() || 0} {txt.behavioralSpans}
                    </p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Behavior Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <Brain className="w-5 h-5 text-blue-600" />
                  {txt.behaviorDistribution}
                </h3>
              </div>
              <div className="p-6 min-h-[300px]">
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <Sparkles className="w-8 h-8 text-blue-500 animate-pulse" />
                  </div>
                ) : behaviorChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={behaviorChartData} layout="vertical">
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {behaviorChartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-400">
                    {txt.noBehaviorData}
                  </div>
                )}
              </div>
            </motion.div>

            {/* Agent Analysis */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <Users className="w-5 h-5 text-purple-600" />
                  {txt.agentAnalysis}
                </h3>
              </div>
              <div className="p-6 min-h-[300px]">
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <Sparkles className="w-8 h-8 text-purple-500 animate-pulse" />
                  </div>
                ) : agentChartData.length > 0 ? (
                  <div className="flex items-center gap-4">
                    <ResponsiveContainer width="50%" height={250}>
                      <PieChart>
                        <Pie
                          data={agentChartData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          paddingAngle={3}
                          dataKey="value"
                        >
                          {agentChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value: number) => value.toLocaleString()} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="w-1/2 space-y-2">
                      {agentChartData.map((entry, i) => (
                        <div key={i} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.fill }} />
                            <span className="text-gray-700">{entry.name.replace('AGT_', '')}</span>
                          </div>
                          <span className="font-medium text-gray-900">{(entry.value as number).toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-400">
                    {txt.noAgentData}
                  </div>
                )}
              </div>
            </motion.div>

            {/* Top Surahs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-amber-600" />
                  {txt.topSurahs}
                </h3>
              </div>
              <div className="p-6 min-h-[300px]">
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <Sparkles className="w-8 h-8 text-amber-500 animate-pulse" />
                  </div>
                ) : surahChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={surahChartData}>
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="spans" radius={[4, 4, 0, 0]}>
                        {surahChartData.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-400">
                    {txt.noSurahData}
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        </div>

        {/* Full-width Activity Section */}
        <div className="max-w-7xl mx-auto px-6 pb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
            className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
              <h3 className="font-semibold text-gray-900">{txt.recentActivity}</h3>
              <button className="text-sm text-emerald-600 hover:text-emerald-700">
                {txt.viewAll}
              </button>
            </div>
            <div className="p-6">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-gray-500 border-b border-gray-100">
                      <th className="pb-3 font-medium">{txt.reference}</th>
                      <th className="pb-3 font-medium">{txt.behavior}</th>
                      <th className="pb-3 font-medium">{txt.agent}</th>
                      <th className="pb-3 font-medium">{txt.annotator}</th>
                      <th className="pb-3 font-medium">{txt.tier}</th>
                      <th className="pb-3 font-medium">{txt.time}</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {recentSpans.length ? (
                      recentSpans.map((span, i) => {
                        const reference =
                          span.reference?.surah && span.reference?.ayah
                            ? `${span.reference.surah}:${span.reference.ayah}`
                            : "--";
                        const annotatedAt = span.annotated_at ? new Date(span.annotated_at) : null;
                        const timeLabel =
                          annotatedAt && !Number.isNaN(annotatedAt.getTime())
                            ? annotatedAt.toLocaleString()
                            : "--";
                        const annotator = span.annotator || "--";
                        return (
                          <tr
                            key={span.span_id || `${reference}-${i}`}
                            className="border-b border-gray-50 hover:bg-gray-50"
                          >
                            <td className="py-3">
                              <span className="font-medium text-emerald-600">{reference}</span>
                            </td>
                            <td className="py-3">{span.behavior_form || "--"}</td>
                            <td className="py-3">{span.agent?.type || "--"}</td>
                            <td className="py-3">
                              <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs font-medium">
                                  {annotator.charAt(0)}
                                </div>
                                {annotator}
                              </div>
                            </td>
                            <td className="py-3">
                              <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
                                {tierLabel === 'SILVER' ? txt.silver : tierLabel === 'GOLD' ? txt.gold : tierLabel}
                              </span>
                            </td>
                            <td className="py-3 text-gray-500">{timeLabel}</td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={6} className="py-4 text-center text-gray-500">
                          {backendStats ? txt.noRecentSpans : txt.loadingData}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
  );
}
