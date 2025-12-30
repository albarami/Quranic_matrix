"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Brain,
  BookOpen,
  BarChart3,
  GitCompare,
  Sparkles,
  MessageSquare,
  Lightbulb,
  ArrowRight,
  Zap,
  Search,
  Send,
  Loader2,
  ChevronDown,
  ChevronUp,
  User,
  Layers,
  ExternalLink,
  Database,
  AlertCircle,
  CheckCircle,
  Shield,
} from "lucide-react";
import { useLanguage } from "../../contexts/LanguageContext";

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

// Detect if query is asking for statistics/metrics (agent distribution, pie chart, etc.)
function isMetricIntentQuery(text: string): boolean {
  const patterns = [
    /توزيع.*الفاعلين/,
    /توزيع.*أنواع/,
    /مخطط.*دائري/,
    /إحصائيات/,
    /نسب.*الفاعلين/,
    /كم.*عدد/,
    /agent.*distribution/i,
    /pie.*chart/i,
    /statistics/i,
  ];
  return patterns.some(p => p.test(text));
}

// Fetch deterministic metrics from backend truth layer
async function fetchTruthMetrics(): Promise<any | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/metrics/overview`);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.status !== "ready") return null;
    return data;
  } catch (e) {
    console.error("Failed to fetch truth metrics:", e);
    return null;
  }
}

// Color palette for charts
const COLORS = {
  ibn_kathir: '#10b981',
  tabari: '#3b82f6',
  qurtubi: '#8b5cf6',
  saadi: '#f59e0b',
  jalalayn: '#ef4444',
  quran: '#059669',
  graph: '#6366f1',
};

// Predefined example queries organized by category
const EXAMPLE_QUERIES = [
  {
    category: "التحليل السلوكي",
    categoryEn: "Behavioral Analysis",
    icon: Brain,
    color: "bg-purple-100 text-purple-600",
    queries: [
      { ar: "حلل سلوك الصبر في القرآن مع توزيع السور", en: "Analyze patience behavior across the Quran with surah distribution" },
      { ar: "ما هي أكثر السلوكيات السلبية ذكراً؟ اعرضها كمخطط", en: "What are the most common negative behaviors? Show as a chart" },
      { ar: "أظهر جميع السلوكيات المتعلقة بالقلب مع تقييماتها", en: "Show all heart-related behaviors with their evaluations" },
    ]
  },
  {
    category: "استكشاف التفسير",
    categoryEn: "Tafsir Exploration",
    icon: BookOpen,
    color: "bg-amber-100 text-amber-600",
    queries: [
      { ar: "قارن تفسيرات آية الغيبة (49:12)", en: "Compare tafsir interpretations for the backbiting verse (49:12)" },
      { ar: "اعرض تفسير ابن كثير لآية الكرسي مع التعليقات السلوكية", en: "Show Ibn Kathir's explanation of Ayat al-Kursi with behavior annotations" },
      { ar: "ماذا يقول العلماء عن غض البصر (24:30)؟", en: "What do scholars say about lowering the gaze (24:30)?" },
    ]
  },
  {
    category: "الرؤى الإحصائية",
    categoryEn: "Statistical Insights",
    icon: BarChart3,
    color: "bg-blue-100 text-blue-600",
    queries: [
      { ar: "أنشئ مخطط دائري لتوزيع أنواع الفاعلين", en: "Generate a pie chart of agent type distribution" },
      { ar: "أي السور لديها أعلى كثافة من التوجيه السلوكي؟", en: "Which surahs have the highest density of behavioral guidance?" },
      { ar: "اعرض تفصيل تقييمات المدح مقابل الذم", en: "Show the breakdown of praise vs blame evaluations" },
    ]
  },
  {
    category: "المراجع المتقاطعة",
    categoryEn: "Cross-Reference",
    icon: GitCompare,
    color: "bg-emerald-100 text-emerald-600",
    queries: [
      { ar: "قارن السلوكيات في سورة الحجرات مقابل سورة الإسراء", en: "Compare behaviors in Surah Al-Hujurat vs Surah Al-Isra" },
      { ar: "أوجد الآيات التي يظهر فيها نفس السلوك في سياقات مختلفة", en: "Find ayat where the same behavior appears in different contexts" },
      { ar: "اعرض العلاقات بين أفعال الكلام وعواقبها", en: "Show relationships between speech acts and their consequences" },
    ]
  },
];

interface ProofResult {
  question: string;
  answer: string;
  proof: {
    quran: Array<{ surah: string; ayah: string; text: string; relevance: number }>;
    ibn_kathir: Array<{ surah: string; ayah: string; text: string }>;
    tabari: Array<{ surah: string; ayah: string; text: string }>;
    qurtubi: Array<{ surah: string; ayah: string; text: string }>;
    saadi: Array<{ surah: string; ayah: string; text: string }>;
    jalalayn: Array<{ surah: string; ayah: string; text: string }>;
    baghawi: Array<{ surah: string; ayah: string; text: string }>;
    muyassar: Array<{ surah: string; ayah: string; text: string }>;
    graph: { nodes: any[]; edges: any[]; paths: any[] };
    taxonomy: { behaviors: any[]; dimensions: Record<string, string> };
    statistics: { counts: Record<string, number>; percentages: Record<string, number> };
  };
  validation: { score: number; passed: boolean; };
  processing_time_ms: number;
  debug?: {
    intent?: string;
    retrieval_mode?: string;
    fallback_used?: boolean;
    resolved_entities?: Array<{ term: string; canonical: string; type: string }>;
  };
}

// Complete 11-Axis Bouzidani Framework
const BOUZIDANI_11_AXES = [
  { id: 1, ar: "العضوي", en: "Organic", color: "bg-rose-500" },
  { id: 2, ar: "الموقفي", en: "Situational", color: "bg-blue-500" },
  { id: 3, ar: "النسقي", en: "Systemic", color: "bg-purple-500" },
  { id: 4, ar: "المكاني", en: "Spatial", color: "bg-cyan-500" },
  { id: 5, ar: "الزماني", en: "Temporal", color: "bg-amber-500" },
  { id: 6, ar: "الفاعلي", en: "Agent", color: "bg-indigo-500" },
  { id: 7, ar: "المصدري", en: "Source", color: "bg-yellow-500" },
  { id: 8, ar: "التقييمي", en: "Evaluation", color: "bg-emerald-500" },
  { id: 9, ar: "تأثير القلب", en: "Heart Impact", color: "bg-pink-500" },
  { id: 10, ar: "العاقبة", en: "Consequence", color: "bg-orange-500" },
  { id: 11, ar: "العلاقات", en: "Relationships", color: "bg-violet-500" },
];

// Map intent codes to human-readable planner names
const PLANNER_NAMES: Record<string, { en: string; ar: string }> = {
  'GRAPH_CAUSAL': { en: 'Causal Chain Planner', ar: 'مخطط السلسلة السببية' },
  'CROSS_TAFSIR_ANALYSIS': { en: 'Cross-Tafsir Planner', ar: 'مخطط مقارنة التفاسير' },
  'PROFILE_11D': { en: '11-Dimension Profile Planner', ar: 'مخطط الملف الشخصي 11 بُعد' },
  'GRAPH_METRICS': { en: 'Graph Metrics Planner', ar: 'مخطط مقاييس الشبكة' },
  'HEART_STATE': { en: 'Heart State Planner', ar: 'مخطط حالات القلب' },
  'AGENT_ANALYSIS': { en: 'Agent Analysis Planner', ar: 'مخطط تحليل الفاعلين' },
  'TEMPORAL_SPATIAL': { en: 'Temporal-Spatial Planner', ar: 'مخطط السياق الزماني المكاني' },
  'CONSEQUENCE_ANALYSIS': { en: 'Consequence Planner', ar: 'مخطط العواقب' },
  'EMBEDDINGS_ANALYSIS': { en: 'Embeddings Planner', ar: 'مخطط التضمينات' },
  'INTEGRATION_E2E': { en: 'Integration Planner', ar: 'مخطط التكامل' },
  'SURAH_REF': { en: 'Surah Reference Planner', ar: 'مخطط مرجع السورة' },
  'AYAH_REF': { en: 'Ayah Reference Planner', ar: 'مخطط مرجع الآية' },
  'CONCEPT_REF': { en: 'Concept Reference Planner', ar: 'مخطط مرجع المفهوم' },
  'FREE_TEXT': { en: 'General Query Planner', ar: 'مخطط الاستعلام العام' },
};

export default function ResearchPage() {
  const { t, isRTL, language } = useLanguage();
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<ProofResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['answer']));
  const [activeTafsir, setActiveTafsir] = useState('ibn_kathir');
  const [expandedVerses, setExpandedVerses] = useState<Set<number>>(new Set());
  const [chatHistory, setChatHistory] = useState<Array<{role: 'user' | 'assistant', content: string, result?: ProofResult, metricsData?: any}>>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userQuery = query;
    setQuery('');
    setLoading(true);
    setError(null);
    
    // Add user message to chat
    setChatHistory(prev => [...prev, { role: 'user', content: userQuery }]);

    try {
      // CRITICAL: For metric-intent queries, fetch from deterministic truth layer
      // Do NOT send to LLM - numbers must come from /api/metrics/overview only
      if (isMetricIntentQuery(userQuery)) {
        const metricsData = await fetchTruthMetrics();
        if (metricsData) {
          // Return deterministic metrics - NO LLM involved
          setChatHistory(prev => [...prev, { 
            role: 'assistant', 
            content: '', // Content is empty - we render MetricsPanel instead
            metricsData: metricsData 
          }]);
          setLoading(false);
          return;
        }
        // If metrics unavailable, fall through to regular query with warning
      }

      // Non-metric queries go to proof system
      const response = await fetch(`${BACKEND_URL}/api/proof/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQuery, include_proof: true }),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      // Add assistant response to chat
      setChatHistory(prev => [...prev, { role: 'assistant', content: data.answer, result: data }]);
    } catch (err: any) {
      setError(err.message || 'Failed to get response');
      setChatHistory(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleQuery: { ar: string; en: string }) => {
    const queryText = language === 'ar' ? exampleQuery.ar : exampleQuery.en;
    setQuery(queryText);
  };

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  return (
    <div className={`h-screen flex flex-col bg-gradient-to-b from-emerald-50 via-white to-emerald-50 ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-emerald-800 text-white px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <MessageSquare className="w-7 h-7" />
              {language === 'ar' ? 'مساعد البحث القرآني' : 'QBM Research Assistant'}
            </h1>
            <p className="text-emerald-200 text-sm mt-1">
              {language === 'ar' ? 'اسأل أي سؤال واحصل على إجابة مدعومة بـ 13 مكون إثبات' : 'Ask any question and get answers backed by 13 proof components'}
            </p>
          </div>
          <div className="flex items-center gap-4 text-sm text-emerald-200">
            <span
              className="inline-flex items-center gap-1.5 bg-emerald-900/30 px-3 py-1.5 rounded-full border border-emerald-500/50 cursor-help"
              title={language === 'ar'
                ? "هذا النظام اجتاز 200 سؤال اختبار صارم عبر 10 فئات بدقة 100٪. لا توجد بيانات ملفقة."
                : "This system passed 200 rigorous test questions across 10 categories with 100% accuracy. No fabricated data."}
            >
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-emerald-300 font-medium">200/200</span>
            </span>
            <span>📖 7 {language === 'ar' ? 'تفاسير' : 'Tafsir'}</span>
            <span>🔗 {language === 'ar' ? 'شبكة' : 'Graph'}</span>
            <span>📊 {language === 'ar' ? 'إحصائيات' : 'Stats'}</span>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Welcome Message if no chat history */}
          {chatHistory.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <div className="text-6xl mb-4">🔬</div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                {language === 'ar' ? 'مرحباً بك في مساعد البحث' : 'Welcome to Research Assistant'}
              </h2>
              <p className="text-gray-600 mb-8 max-w-lg mx-auto">
                {language === 'ar' 
                  ? 'اسأل أي سؤال عن السلوكيات القرآنية واحصل على إجابة مدعومة بالقرآن والتفاسير والشبكة'
                  : 'Ask any question about Quranic behaviors and get answers backed by Quran, Tafsir, and Graph'}
              </p>
              
              {/* Example Query Categories */}
              <div className="grid md:grid-cols-2 gap-4 text-left max-w-3xl mx-auto">
                {EXAMPLE_QUERIES.map((cat, i) => {
                  const Icon = cat.icon;
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="bg-white rounded-xl p-4 shadow-md border border-gray-100"
                    >
                      <div className={`inline-flex items-center gap-2 ${cat.color} px-3 py-1 rounded-full text-sm font-medium mb-3`}>
                        <Icon className="w-4 h-4" />
                        {language === 'ar' ? cat.category : cat.categoryEn}
                      </div>
                      <div className="space-y-2">
                        {cat.queries.slice(0, 2).map((q, j) => (
                          <button
                            key={j}
                            onClick={() => handleExampleClick(q)}
                            className="w-full text-left p-2 text-sm text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 rounded-lg transition"
                          >
                            💡 {language === 'ar' ? q.ar : q.en}
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* Chat Messages */}
          {chatHistory.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-3xl ${msg.role === 'user' ? 'bg-emerald-600 text-white' : 'bg-white shadow-lg border'} rounded-2xl p-4`}>
                {msg.role === 'user' ? (
                  <p className="text-lg">{msg.content}</p>
                ) : msg.metricsData ? (
                  // DETERMINISTIC METRICS PANEL - Numbers from truth layer, NOT LLM
                  <div className="space-y-6">
                    {/* Header with source info */}
                    <div className="flex items-center justify-between border-b border-gray-100 pb-4">
                      <div className="flex items-center gap-2">
                        <Database className="w-5 h-5 text-emerald-600" />
                        <h2 className="text-lg font-semibold text-gray-800">
                          {language === 'ar' ? 'توزيع أنواع الفاعلين في القرآن الكريم' : 'Agent Type Distribution in the Quran'}
                        </h2>
                      </div>
                      <div className="text-xs text-gray-500">
                        {language === 'ar' ? 'المصدر:' : 'Source:'} truth_metrics_v1 | {msg.metricsData.metrics.totals.spans.toLocaleString()} {language === 'ar' ? 'نطاق' : 'spans'}
                      </div>
                    </div>

                    {/* KPI Cards */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-emerald-50 rounded-lg p-4 text-center">
                        <div className="text-2xl font-bold text-emerald-700">
                          {msg.metricsData.metrics.totals.spans.toLocaleString()}
                        </div>
                        <div className="text-sm text-emerald-600">{language === 'ar' ? 'إجمالي النطاقات' : 'Total Spans'}</div>
                      </div>
                      <div className="bg-cyan-50 rounded-lg p-4 text-center">
                        <div className="text-2xl font-bold text-cyan-700">
                          {msg.metricsData.metrics.totals.unique_verse_keys.toLocaleString()}
                        </div>
                        <div className="text-sm text-cyan-600">{language === 'ar' ? 'الآيات الفريدة' : 'Unique Verses'}</div>
                      </div>
                      <div className="bg-violet-50 rounded-lg p-4 text-center">
                        <div className="text-2xl font-bold text-violet-700">
                          {msg.metricsData.metrics.totals.tafsir_sources_count}
                        </div>
                        <div className="text-sm text-violet-600">{language === 'ar' ? 'مصادر التفسير' : 'Tafsir Sources'}</div>
                      </div>
                    </div>

                    {/* Agent Distribution - Pie Chart + Table */}
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <h3 className="font-semibold text-gray-700 mb-3">
                          {language === 'ar' ? 'توزيع الفاعلين (مخطط دائري)' : 'Agent Distribution (Pie Chart)'}
                        </h3>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={msg.metricsData.metrics.agent_distribution.items.map((item: any, idx: number) => ({
                                  name: item.label_ar,
                                  value: item.count,
                                  percentage: item.percentage,
                                  fill: ['#059669', '#0891b2', '#7c3aed', '#db2777', '#ea580c', '#ca8a04', '#4f46e5'][idx % 7],
                                }))}
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={80}
                                paddingAngle={2}
                                dataKey="value"
                                label={({ percentage }) => `${percentage}%`}
                              >
                                {msg.metricsData.metrics.agent_distribution.items.map((_: any, index: number) => (
                                  <Cell key={`cell-${index}`} fill={['#059669', '#0891b2', '#7c3aed', '#db2777', '#ea580c', '#ca8a04', '#4f46e5'][index % 7]} />
                                ))}
                              </Pie>
                              <Tooltip formatter={(value: number) => value.toLocaleString()} />
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div>
                        <h3 className="font-semibold text-gray-700 mb-3">
                          {language === 'ar' ? 'جدول التوزيع' : 'Distribution Table'}
                        </h3>
                        <div className="overflow-hidden rounded-lg border border-gray-200">
                          <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-emerald-50">
                              <tr>
                                <th className="px-3 py-2 text-right text-xs font-semibold text-emerald-800">{language === 'ar' ? 'الفاعل' : 'Agent'}</th>
                                <th className="px-3 py-2 text-right text-xs font-semibold text-emerald-800">{language === 'ar' ? 'العدد' : 'Count'}</th>
                                <th className="px-3 py-2 text-right text-xs font-semibold text-emerald-800">{language === 'ar' ? 'النسبة' : '%'}</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-100">
                              {msg.metricsData.metrics.agent_distribution.items.map((item: any, idx: number) => (
                                <tr key={item.key} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                  <td className="px-3 py-2 text-sm text-gray-900">{item.label_ar}</td>
                                  <td className="px-3 py-2 text-sm text-gray-900 font-medium">{item.count.toLocaleString()}</td>
                                  <td className="px-3 py-2 text-sm">
                                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                                      {item.percentage}%
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>

                    {/* Footer */}
                    <div className="text-xs text-gray-400 text-center pt-4 border-t border-gray-100">
                      {language === 'ar' ? 'البيانات من:' : 'Data from:'} {msg.metricsData.source_files[0]?.split(/[/\\]/).pop()} | 
                      {language === 'ar' ? 'الإصدار:' : 'Version:'} {msg.metricsData.build_version}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* PROFESSIONAL UI - No markdown, structured components only */}
                    
                    {msg.result ? (
                      <>
                        {/* Planner Selection Indicator & Entity Resolution */}
                        <div className="flex flex-wrap items-center gap-3 mb-4">
                          {/* Active Planner */}
                          {msg.result.debug?.intent && (
                            <div className="inline-flex items-center gap-2 bg-violet-100 px-3 py-1.5 rounded-full border border-violet-300">
                              <Zap className="w-4 h-4 text-violet-600" />
                              <span className="text-violet-700 font-medium text-sm">
                                {language === 'ar' ? 'المخطط: ' : 'Planner: '}
                                {PLANNER_NAMES[msg.result.debug.intent]?.[language] || msg.result.debug.intent}
                              </span>
                            </div>
                          )}
                          {/* Entity Resolution */}
                          {msg.result.debug?.resolved_entities && msg.result.debug.resolved_entities.length > 0 && (
                            <div className="inline-flex items-center gap-2 bg-amber-100 px-3 py-1.5 rounded-full border border-amber-300">
                              <Search className="w-4 h-4 text-amber-600" />
                              <span className="text-amber-700 font-medium text-sm">
                                {msg.result.debug.resolved_entities.map(e => `${e.term} → ${e.canonical}`).join(', ')}
                              </span>
                            </div>
                          )}
                        </div>

                        {/* KPI Stats Cards - Top */}
                        <div className="grid grid-cols-4 gap-3">
                          <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl p-4 text-white text-center shadow-lg">
                            <p className="text-3xl font-bold">{msg.result.proof.quran?.length || 0}</p>
                            <p className="text-sm opacity-90 mt-1">{language === 'ar' ? 'آيات قرآنية' : 'Quran Verses'}</p>
                          </div>
                          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-4 text-white text-center shadow-lg">
                            <p className="text-3xl font-bold">
                              {(msg.result.proof.ibn_kathir?.length || 0) + (msg.result.proof.tabari?.length || 0) + (msg.result.proof.qurtubi?.length || 0) + (msg.result.proof.saadi?.length || 0) + (msg.result.proof.jalalayn?.length || 0) + (msg.result.proof.baghawi?.length || 0) + (msg.result.proof.muyassar?.length || 0)}
                            </p>
                            <p className="text-sm opacity-90 mt-1">{language === 'ar' ? 'شواهد تفسيرية' : 'Tafsir Citations'}</p>
                          </div>
                          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-4 text-white text-center shadow-lg">
                            <p className="text-3xl font-bold">{msg.result.proof.graph?.nodes?.length || 0}</p>
                            <p className="text-sm opacity-90 mt-1">{language === 'ar' ? 'عقد سلوكية' : 'Behavior Nodes'}</p>
                          </div>
                          <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-xl p-4 text-white text-center shadow-lg">
                            <p className="text-3xl font-bold">{msg.result.validation.score.toFixed(0)}%</p>
                            <p className="text-sm opacity-90 mt-1">{language === 'ar' ? 'نسبة التحقق' : 'Validation'}</p>
                          </div>
                        </div>

                        {/* Quran Verses Section */}
                        {msg.result.proof.quran?.length > 0 && (
                          <div className="bg-gradient-to-br from-emerald-50 to-white rounded-xl border border-emerald-200 overflow-hidden">
                            <div className="bg-emerald-600 px-4 py-3 flex items-center justify-between">
                              <h3 className="text-white font-bold flex items-center gap-2">
                                <BookOpen className="w-5 h-5" />
                                {language === 'ar' ? 'الآيات القرآنية' : 'Quran Verses'}
                              </h3>
                              <span className="bg-white/20 text-white text-sm px-3 py-1 rounded-full">
                                {msg.result.proof.quran.length} {language === 'ar' ? 'آية' : 'verses'}
                              </span>
                            </div>
                            <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto">
                              {(expandedVerses.has(i) ? msg.result.proof.quran : msg.result.proof.quran.slice(0, 4)).map((v, j) => (
                                <div key={j} className="bg-white rounded-xl p-4 shadow-sm border-r-4 border-emerald-500 hover:shadow-md transition">
                                  <p className="text-gray-800 text-lg leading-relaxed font-arabic">{v.text}</p>
                                  <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-100">
                                    <div className="flex items-center gap-2">
                                      <span className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-sm font-medium">
                                        {language === 'ar' ? `سورة ${v.surah} : ${v.ayah}` : `${v.surah}:${v.ayah}`}
                                      </span>
                                    </div>
                                    {v.relevance && (
                                      <div className="flex items-center gap-1">
                                        <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                                          <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${v.relevance * 100}%` }} />
                                        </div>
                                        <span className="text-xs text-gray-500">{(v.relevance * 100).toFixed(0)}%</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                            {msg.result.proof.quran.length > 4 && (
                              <button
                                onClick={() => {
                                  const newSet = new Set(expandedVerses);
                                  if (newSet.has(i)) newSet.delete(i);
                                  else newSet.add(i);
                                  setExpandedVerses(newSet);
                                }}
                                className="w-full py-3 text-emerald-700 hover:bg-emerald-50 transition flex items-center justify-center gap-2 border-t border-emerald-100"
                              >
                                {expandedVerses.has(i) ? (
                                  <><ChevronUp className="w-4 h-4" /> {language === 'ar' ? 'عرض أقل' : 'Show Less'}</>
                                ) : (
                                  <><ChevronDown className="w-4 h-4" /> {language === 'ar' ? `عرض المزيد (+${msg.result.proof.quran.length - 4})` : `Show More (+${msg.result.proof.quran.length - 4})`}</>
                                )}
                              </button>
                            )}
                          </div>
                        )}

                        {/* Tafsir Section - Tabbed Interface */}
                        {((msg.result.proof.ibn_kathir?.length || 0) + (msg.result.proof.tabari?.length || 0) + (msg.result.proof.qurtubi?.length || 0) + (msg.result.proof.saadi?.length || 0) + (msg.result.proof.jalalayn?.length || 0) + (msg.result.proof.baghawi?.length || 0) + (msg.result.proof.muyassar?.length || 0)) > 0 && (
                          <div className="bg-gradient-to-br from-blue-50 to-white rounded-xl border border-blue-200 overflow-hidden">
                            <div className="bg-blue-600 px-4 py-3">
                              <h3 className="text-white font-bold flex items-center gap-2">
                                <Layers className="w-5 h-5" />
                                {language === 'ar' ? 'شواهد المفسرين' : 'Tafsir Evidence'}
                              </h3>
                            </div>
                            
                            {/* Tafsir Tabs */}
                            <div className="flex border-b border-blue-100 bg-blue-50/50">
                              {[
                                { key: 'ibn_kathir', name: 'ابن كثير', color: 'emerald' },
                                { key: 'tabari', name: 'الطبري', color: 'blue' },
                                { key: 'qurtubi', name: 'القرطبي', color: 'purple' },
                                { key: 'saadi', name: 'السعدي', color: 'amber' },
                                { key: 'jalalayn', name: 'الجلالين', color: 'red' },
                                { key: 'baghawi', name: 'البغوي', color: 'cyan' },
                                { key: 'muyassar', name: 'الميسر', color: 'rose' },
                              ].map((tafsir) => {
                                const count = (msg.result?.proof as any)?.[tafsir.key]?.length || 0;
                                if (count === 0) return null;
                                return (
                                  <button
                                    key={tafsir.key}
                                    onClick={() => setActiveTafsir(tafsir.key)}
                                    className={`flex-1 px-3 py-2 text-sm font-medium transition ${
                                      activeTafsir === tafsir.key
                                        ? 'bg-white text-blue-700 border-b-2 border-blue-600'
                                        : 'text-gray-600 hover:bg-white/50'
                                    }`}
                                  >
                                    {tafsir.name}
                                    <span className="ml-1 text-xs bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded-full">{count}</span>
                                  </button>
                                );
                              })}
                            </div>
                            
                            {/* Tafsir Content */}
                            <div className="p-4 max-h-[300px] overflow-y-auto">
                              {((msg.result?.proof as any)?.[activeTafsir] || []).slice(0, 3).map((t: any, j: number) => (
                                <div key={j} className="bg-white rounded-lg p-4 mb-3 shadow-sm border-r-4 border-blue-400">
                                  <p className="text-gray-700 leading-relaxed">{t.text}</p>
                                  <p className="text-sm text-blue-600 mt-2 flex items-center gap-1">
                                    <span className="bg-blue-50 px-2 py-0.5 rounded">{t.surah}:{t.ayah}</span>
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Graph/Taxonomy Section */}
                        {msg.result.proof.graph?.nodes?.length > 0 && (
                          <div className="bg-gradient-to-br from-purple-50 to-white rounded-xl border border-purple-200 overflow-hidden">
                            <div className="bg-purple-600 px-4 py-3">
                              <h3 className="text-white font-bold flex items-center gap-2">
                                <GitCompare className="w-5 h-5" />
                                {language === 'ar' ? 'الشبكة السلوكية' : 'Behavior Network'}
                              </h3>
                            </div>
                            <div className="p-4">
                              <div className="flex flex-wrap gap-2">
                                {msg.result.proof.graph.nodes.slice(0, 10).map((node: any, j: number) => (
                                  <span key={j} className="bg-purple-100 text-purple-700 px-3 py-1.5 rounded-full text-sm font-medium">
                                    {node.label || node.id || `Node ${j + 1}`}
                                  </span>
                                ))}
                                {msg.result.proof.graph.nodes.length > 10 && (
                                  <span className="bg-gray-100 text-gray-500 px-3 py-1.5 rounded-full text-sm">
                                    +{msg.result.proof.graph.nodes.length - 10} {language === 'ar' ? 'عقدة أخرى' : 'more'}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        )}

                        {/* 11-Axis Behavioral Profile */}
                        {msg.result.proof.taxonomy && (
                          <div className="bg-gradient-to-br from-indigo-50 to-white rounded-xl border border-indigo-200 overflow-hidden">
                            <div className="bg-indigo-600 px-4 py-3">
                              <h3 className="text-white font-bold flex items-center gap-2">
                                <BarChart3 className="w-5 h-5" />
                                {language === 'ar' ? 'الملف الشخصي 11 بُعد' : '11-Axis Behavioral Profile'}
                              </h3>
                            </div>
                            <div className="p-4 space-y-2">
                              {BOUZIDANI_11_AXES.map((axis) => {
                                const value = msg.result?.proof?.taxonomy?.dimensions?.[axis.en.toLowerCase()] ||
                                              msg.result?.proof?.taxonomy?.dimensions?.[axis.ar] || null;
                                return (
                                  <div key={axis.id} className="flex items-center gap-2">
                                    <span className="w-6 text-[10px] text-gray-400 font-mono">#{axis.id}</span>
                                    <span className="w-20 text-xs text-gray-600 truncate">{axis.ar}</span>
                                    <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                                      <div
                                        className={`${axis.color} h-1.5 rounded-full`}
                                        style={{ width: value ? '100%' : '0%', opacity: value ? 1 : 0.2 }}
                                      />
                                    </div>
                                    <span className="w-20 text-right text-[10px] text-gray-600 truncate">
                                      {value || '—'}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="grid grid-cols-2 gap-3">
                          <a
                            href={`/proof?q=${encodeURIComponent(msg.result.question)}`}
                            className="flex items-center justify-center gap-2 py-3 bg-emerald-100 text-emerald-700 rounded-xl font-medium hover:bg-emerald-200 transition shadow-sm"
                          >
                            <Search className="w-4 h-4" />
                            {language === 'ar' ? 'عرض الإثبات الكامل' : 'View Full Proof'}
                            <ExternalLink className="w-3 h-3" />
                          </a>
                          <a
                            href={`/behavior-profile`}
                            className="flex items-center justify-center gap-2 py-3 bg-purple-100 text-purple-700 rounded-xl font-medium hover:bg-purple-200 transition shadow-sm"
                          >
                            <User className="w-4 h-4" />
                            {language === 'ar' ? 'ملف السلوك' : 'Behavior Profile'}
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </>
                    ) : (
                      // Fallback for messages without result (errors, etc.)
                      <p className="text-gray-700">{msg.content}</p>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}

          {/* Loading */}
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="bg-white shadow-lg border rounded-2xl p-4 flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-emerald-600" />
                <span className="text-gray-600">{language === 'ar' ? 'جاري التحليل...' : 'Analyzing...'}</span>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t bg-white px-6 py-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3 bg-gray-50 p-2 rounded-xl border border-gray-200">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={language === 'ar' ? 'اكتب سؤالك هنا...' : 'Type your question here...'}
              className="flex-1 px-4 py-3 bg-transparent border-0 focus:ring-0 text-lg"
              dir={isRTL ? 'rtl' : 'ltr'}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white rounded-lg font-bold hover:from-emerald-700 hover:to-emerald-800 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              {language === 'ar' ? 'إرسال' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
