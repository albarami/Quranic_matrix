"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, 
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, 
  PolarRadiusAxis, Radar, Legend
} from "recharts";
import {
  Search, BookOpen, Network, Layers, FileText, Hash, Brain,
  ChevronRight, Loader2, AlertCircle, TrendingUp, Users,
  Heart, Clock, MapPin, Scale, ArrowRight, ExternalLink,
  BarChart3, Globe, Sparkles
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

const TAFSIR_COLORS: Record<string, string> = {
  ibn_kathir: '#10b981',
  tabari: '#3b82f6',
  qurtubi: '#8b5cf6',
  saadi: '#f59e0b',
  jalalayn: '#ef4444',
};

const TAFSIR_NAMES: Record<string, { ar: string; en: string }> = {
  ibn_kathir: { ar: 'ابن كثير', en: 'Ibn Kathir' },
  tabari: { ar: 'الطبري', en: 'Tabari' },
  qurtubi: { ar: 'القرطبي', en: 'Qurtubi' },
  saadi: { ar: 'السعدي', en: 'Saadi' },
  jalalayn: { ar: 'الجلالين', en: 'Jalalayn' },
};

const BEHAVIOR_NAMES: Record<string, { ar: string; en: string }> = {
  inner_state: { ar: 'الحالة الداخلية', en: 'Inner State' },
  speech_act: { ar: 'الفعل الكلامي', en: 'Speech Act' },
  relational_act: { ar: 'الفعل العلائقي', en: 'Relational Act' },
  physical_act: { ar: 'الفعل الجسدي', en: 'Physical Act' },
  trait_disposition: { ar: 'السمة والاستعداد', en: 'Trait/Disposition' },
};

interface BehaviorProfile {
  behavior: string;
  arabic_name: string;
  summary: {
    total_verses: number;
    total_spans: number;
    total_tafsir: number;
    total_surahs: number;
    coverage_percentage: number;
  };
  verses: Array<{
    surah: number;
    surah_name: string;
    ayah: number;
    text: string;
    agent: string;
    agent_referent: string;
    evaluation: string;
    deontic: string;
    behavior_form: string;
  }>;
  tafsir: Record<string, Array<{ surah: number; ayah: number; text: string }>>;
  graph: {
    related_behaviors: string[];
    verses: any[];
    connections: any[];
  };
  dimensions: Record<string, Record<string, number>>;
  surah_distribution: Array<{ surah: string; count: number }>;
  vocabulary: {
    primary_term: string;
    roots: string[];
    derivatives: string[];
    related_concepts: string[];
  };
  similar_behaviors: Array<{ behavior: string; similarity: number }>;
  processing_time_ms: number;
}

interface BehaviorListItem {
  name: string;
  count: number;
}

type TabId = 'verses' | 'tafsir' | 'graph' | 'dimensions' | 'surahs' | 'vocabulary' | 'embeddings';

const TABS: Array<{ id: TabId; labelAr: string; labelEn: string; icon: any }> = [
  { id: 'verses', labelAr: 'الآيات', labelEn: 'Verses', icon: BookOpen },
  { id: 'tafsir', labelAr: 'التفاسير', labelEn: 'Tafsir', icon: FileText },
  { id: 'graph', labelAr: 'الشبكة', labelEn: 'Graph', icon: Network },
  { id: 'dimensions', labelAr: 'الأبعاد', labelEn: 'Dimensions', icon: Layers },
  { id: 'surahs', labelAr: 'السور', labelEn: 'Surahs', icon: Globe },
  { id: 'vocabulary', labelAr: 'المفردات', labelEn: 'Vocabulary', icon: Hash },
  { id: 'embeddings', labelAr: 'التشابه', labelEn: 'Similarity', icon: Brain },
];

export default function BehaviorProfilePage() {
  const { language, isRTL } = useLanguage();
  const [searchQuery, setSearchQuery] = useState("");
  const [profile, setProfile] = useState<BehaviorProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('verses');
  const [behaviorList, setBehaviorList] = useState<BehaviorListItem[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [activeTafsir, setActiveTafsir] = useState('ibn_kathir');

  // Load behavior list on mount
  useEffect(() => {
    const loadBehaviors = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/behavior/list`);
        if (res.ok) {
          const data = await res.json();
          setBehaviorList(data.behaviors || []);
        }
      } catch (e) {
        console.error("Failed to load behaviors:", e);
      }
    };
    loadBehaviors();
  }, []);

  const loadProfile = useCallback(async (behavior: string) => {
    if (!behavior.trim()) return;
    
    setLoading(true);
    setError(null);
    setShowSuggestions(false);
    
    try {
      const res = await fetch(`${BACKEND_URL}/api/behavior/profile/${encodeURIComponent(behavior)}`);
      if (!res.ok) throw new Error(`Failed to load profile: ${res.status}`);
      
      const data = await res.json();
      setProfile(data);
      setActiveTab('verses');
    } catch (e: any) {
      setError(e.message || "Failed to load behavior profile");
      setProfile(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadProfile(searchQuery);
  };

  const handleSuggestionClick = (behavior: string) => {
    setSearchQuery(behavior);
    loadProfile(behavior);
  };

  // Filter suggestions based on search query
  const filteredSuggestions = behaviorList
    .filter(b => b.name.includes(searchQuery) || searchQuery.includes(b.name))
    .slice(0, 10);

  // Build graph data for visualization
  const graphData = profile ? {
    nodes: [
      { id: profile.behavior, group: 'main', val: 20 },
      ...profile.graph.related_behaviors.slice(0, 15).map((b, i) => ({
        id: b, group: 'related', val: 10
      })),
      ...(profile.similar_behaviors || []).slice(0, 10).map((b, i) => ({
        id: b.behavior, group: 'similar', val: 8
      }))
    ],
    links: [
      ...profile.graph.related_behaviors.slice(0, 15).map(b => ({
        source: profile.behavior, target: b, type: 'related'
      })),
      ...(profile.similar_behaviors || []).slice(0, 10).map(b => ({
        source: profile.behavior, target: b.behavior, type: 'similar'
      }))
    ]
  } : { nodes: [], links: [] };

  // Prepare radar chart data for dimensions
  const radarData = profile ? Object.entries(profile.dimensions)
    .filter(([_, values]) => Object.keys(values).length > 0)
    .map(([dim, values]) => ({
      dimension: dim,
      value: Object.values(values).reduce((a, b) => a + b, 0),
      fullMark: profile.summary.total_spans
    })) : [];

  return (
    <div className={`min-h-[calc(100vh-64px)] bg-gradient-to-b from-purple-50 to-white ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Hero Header */}
      <div className="bg-gradient-to-br from-purple-800 via-purple-900 to-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-white/10 rounded-xl">
              <Layers className="w-8 h-8 text-purple-300" />
            </div>
            <div>
              <span className="text-purple-300 text-sm font-medium">
                {language === "ar" ? "أداة تحليل السلوك الشامل" : "Comprehensive Behavior Analysis Tool"}
              </span>
            </div>
          </div>
          
          <h1 className="text-4xl font-bold mb-4">
            {language === "ar" ? "ملف السلوك" : "Behavior Profile"}
          </h1>
          
          <p className="text-purple-200 max-w-3xl text-lg leading-relaxed mb-8">
            {language === "ar" 
              ? "أدخل اسم السلوك واحصل على كل ما يتعلق به: الآيات، التفاسير، الشبكة، الأبعاد الإحدى عشر، السور، المفردات، والتشابه الدلالي."
              : "Enter a behavior name and get everything related to it: verses, tafsir, graph, 11 dimensions, surahs, vocabulary, and semantic similarity."}
          </p>

          {/* Search Form */}
          <form onSubmit={handleSubmit} className="relative max-w-2xl">
            <div className="flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setShowSuggestions(true);
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  placeholder={language === "ar" ? "أدخل السلوك: الكبر، الصبر، الإيمان..." : "Enter behavior: pride, patience, faith..."}
                  className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/10 backdrop-blur border border-white/20 text-white placeholder-purple-300 focus:outline-none focus:ring-2 focus:ring-purple-400 text-lg"
                  dir={isRTL ? 'rtl' : 'ltr'}
                />
                
                {/* Suggestions Dropdown */}
                <AnimatePresence>
                  {showSuggestions && searchQuery && filteredSuggestions.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-200 overflow-hidden z-50"
                    >
                      {filteredSuggestions.map((b, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handleSuggestionClick(b.name)}
                          className="w-full px-4 py-3 text-left hover:bg-purple-50 flex items-center justify-between text-gray-800 border-b border-gray-100 last:border-0"
                        >
                          <span className="font-medium">
                            {BEHAVIOR_NAMES[b.name] 
                              ? (language === "ar" ? BEHAVIOR_NAMES[b.name].ar : BEHAVIOR_NAMES[b.name].en)
                              : b.name}
                          </span>
                          <span className="text-sm text-gray-500">{b.count} {language === "ar" ? "آية" : "verses"}</span>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              
              <button
                type="submit"
                disabled={loading || !searchQuery.trim()}
                className="px-8 py-4 bg-white text-purple-800 rounded-xl font-bold hover:bg-purple-100 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
                {language === "ar" ? "تحليل" : "Analyze"}
              </button>
            </div>
          </form>

          {/* Quick Suggestions - Use actual behavior categories */}
          <div className="flex flex-wrap gap-2 mt-4">
            {[
              { en: 'inner_state', ar: 'الحالة الداخلية' },
              { en: 'speech_act', ar: 'الفعل الكلامي' },
              { en: 'relational_act', ar: 'الفعل العلائقي' },
              { en: 'physical_act', ar: 'الفعل الجسدي' },
              { en: 'trait_disposition', ar: 'السمة والاستعداد' },
            ].map((b) => (
              <button
                key={b.en}
                onClick={() => handleSuggestionClick(b.en)}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-full text-sm transition"
              >
                {language === 'ar' ? b.ar : b.en}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3 text-red-700">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="max-w-7xl mx-auto px-6 py-20 text-center">
          <Loader2 className="w-12 h-12 animate-spin text-purple-600 mx-auto mb-4" />
          <p className="text-gray-600 text-lg">
            {language === "ar" ? "جاري تحليل السلوك..." : "Analyzing behavior..."}
          </p>
        </div>
      )}

      {/* Profile Results */}
      {profile && !loading && (
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl shadow-lg p-5 text-center border-t-4 border-purple-500"
            >
              <BookOpen className="w-6 h-6 text-purple-600 mx-auto mb-2" />
              <p className="text-3xl font-bold text-gray-800">{profile.summary.total_verses}</p>
              <p className="text-sm text-gray-500">{language === "ar" ? "آيات" : "Verses"}</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-xl shadow-lg p-5 text-center border-t-4 border-blue-500"
            >
              <FileText className="w-6 h-6 text-blue-600 mx-auto mb-2" />
              <p className="text-3xl font-bold text-gray-800">{profile.summary.total_tafsir}</p>
              <p className="text-sm text-gray-500">{language === "ar" ? "تفاسير" : "Tafsir"}</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-xl shadow-lg p-5 text-center border-t-4 border-emerald-500"
            >
              <Network className="w-6 h-6 text-emerald-600 mx-auto mb-2" />
              <p className="text-3xl font-bold text-gray-800">{profile.graph.related_behaviors.length}</p>
              <p className="text-sm text-gray-500">{language === "ar" ? "علاقات" : "Relations"}</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-xl shadow-lg p-5 text-center border-t-4 border-amber-500"
            >
              <Globe className="w-6 h-6 text-amber-600 mx-auto mb-2" />
              <p className="text-3xl font-bold text-gray-800">{profile.summary.total_surahs}</p>
              <p className="text-sm text-gray-500">{language === "ar" ? "سور" : "Surahs"}</p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white rounded-xl shadow-lg p-5 text-center border-t-4 border-rose-500"
            >
              <TrendingUp className="w-6 h-6 text-rose-600 mx-auto mb-2" />
              <p className="text-3xl font-bold text-gray-800">{profile.summary.coverage_percentage}%</p>
              <p className="text-sm text-gray-500">{language === "ar" ? "تغطية" : "Coverage"}</p>
            </motion.div>
          </div>

          {/* Tabs */}
          <div className="flex flex-wrap gap-2 mb-6 bg-white rounded-xl p-2 shadow-sm">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
                    activeTab === tab.id
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-600 hover:bg-purple-50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {language === "ar" ? tab.labelAr : tab.labelEn}
                </button>
              );
            })}
          </div>

          {/* Tab Content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-white rounded-2xl shadow-lg p-6"
            >
              {/* Verses Tab */}
              {activeTab === 'verses' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-purple-600" />
                    {language === "ar" ? `جميع الآيات المرتبطة بـ "${profile.behavior}" (${profile.verses.length})` : `All Verses for "${profile.behavior}" (${profile.verses.length})`}
                  </h3>
                  <div className="space-y-4 max-h-[600px] overflow-y-auto">
                    {profile.verses.map((verse, i) => (
                      <div key={i} className="bg-gray-50 rounded-xl p-4 border-r-4 border-purple-500">
                        <p className="text-lg text-gray-800 leading-relaxed mb-3">{verse.text}</p>
                        <div className="flex flex-wrap gap-2 text-sm">
                          <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded">
                            📍 {verse.surah_name || `Surah ${verse.surah}`}:{verse.ayah}
                          </span>
                          {verse.agent && (
                            <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded">
                              👤 {verse.agent}
                            </span>
                          )}
                          {verse.evaluation && (
                            <span className={`px-2 py-1 rounded ${
                              verse.evaluation.includes('praise') || verse.evaluation.includes('ممدوح')
                                ? 'bg-green-100 text-green-700'
                                : verse.evaluation.includes('blame') || verse.evaluation.includes('مذموم')
                                ? 'bg-red-100 text-red-700'
                                : 'bg-gray-100 text-gray-700'
                            }`}>
                              ⚖️ {verse.evaluation}
                            </span>
                          )}
                          {verse.behavior_form && (
                            <span className="px-2 py-1 bg-amber-100 text-amber-700 rounded">
                              🏷️ {verse.behavior_form}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tafsir Tab */}
              {activeTab === 'tafsir' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-600" />
                    {language === "ar" ? `التفاسير المرتبطة بـ "${profile.behavior}"` : `Tafsir for "${profile.behavior}"`}
                  </h3>
                  
                  {/* Tafsir Source Tabs */}
                  <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
                    {Object.keys(TAFSIR_NAMES).map((source) => (
                      <button
                        key={source}
                        onClick={() => setActiveTafsir(source)}
                        className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition ${
                          activeTafsir === source
                            ? 'text-white'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                        style={activeTafsir === source ? { backgroundColor: TAFSIR_COLORS[source] } : {}}
                      >
                        {language === "ar" ? TAFSIR_NAMES[source].ar : TAFSIR_NAMES[source].en}
                        <span className="ml-2 text-xs opacity-75">({profile.tafsir[source]?.length || 0})</span>
                      </button>
                    ))}
                  </div>
                  
                  <div className="space-y-4 max-h-[500px] overflow-y-auto">
                    {(profile.tafsir[activeTafsir] || []).map((t, i) => (
                      <div key={i} className="bg-gray-50 rounded-xl p-4 border-l-4" style={{ borderColor: TAFSIR_COLORS[activeTafsir] }}>
                        <p className="text-sm text-gray-500 mb-2">📍 {t.surah}:{t.ayah}</p>
                        <p className="text-gray-800 leading-relaxed">{t.text}</p>
                      </div>
                    ))}
                    {(profile.tafsir[activeTafsir] || []).length === 0 && (
                      <p className="text-gray-500 text-center py-8">
                        {language === "ar" ? "لا توجد تفاسير متاحة" : "No tafsir available"}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Graph Tab */}
              {activeTab === 'graph' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Network className="w-5 h-5 text-emerald-600" />
                    {language === "ar" ? `شبكة علاقات "${profile.behavior}"` : `Relationship Network for "${profile.behavior}"`}
                  </h3>
                  
                  <div className="h-[500px] bg-gray-900 rounded-xl overflow-hidden">
                    <ForceGraph2D
                      graphData={graphData}
                      nodeLabel="id"
                      nodeColor={(node: any) => 
                        node.group === 'main' ? '#8b5cf6' :
                        node.group === 'related' ? '#10b981' : '#3b82f6'
                      }
                      nodeVal={(node: any) => node.val}
                      linkColor={() => 'rgba(255,255,255,0.3)'}
                      backgroundColor="#111827"
                      width={800}
                      height={500}
                    />
                  </div>
                  
                  <div className="flex gap-4 mt-4 justify-center">
                    <span className="flex items-center gap-2 text-sm">
                      <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                      {language === "ar" ? "السلوك الرئيسي" : "Main Behavior"}
                    </span>
                    <span className="flex items-center gap-2 text-sm">
                      <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
                      {language === "ar" ? "سلوكيات مرتبطة" : "Related Behaviors"}
                    </span>
                    <span className="flex items-center gap-2 text-sm">
                      <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                      {language === "ar" ? "سلوكيات متشابهة" : "Similar Behaviors"}
                    </span>
                  </div>
                </div>
              )}

              {/* Dimensions Tab */}
              {activeTab === 'dimensions' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-amber-600" />
                    {language === "ar" ? `الأبعاد الإحدى عشر لـ "${profile.behavior}"` : `11 Dimensions for "${profile.behavior}"`}
                  </h3>
                  
                  <div className="grid lg:grid-cols-2 gap-6">
                    {/* Radar Chart */}
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h4 className="font-medium text-gray-700 mb-3 text-center">
                        {language === "ar" ? "نظرة عامة على الأبعاد" : "Dimensions Overview"}
                      </h4>
                      <ResponsiveContainer width="100%" height={300}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="#e5e7eb" />
                          <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
                          <PolarRadiusAxis angle={30} domain={[0, 'auto']} />
                          <Radar
                            name="Count"
                            dataKey="value"
                            stroke="#8b5cf6"
                            fill="#8b5cf6"
                            fillOpacity={0.5}
                          />
                          <Tooltip />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    {/* Dimension Details */}
                    <div className="space-y-4 max-h-[400px] overflow-y-auto">
                      {Object.entries(profile.dimensions).map(([dim, values]) => {
                        if (Object.keys(values).length === 0) return null;
                        return (
                          <div key={dim} className="bg-gray-50 rounded-xl p-4">
                            <h4 className="font-medium text-gray-800 mb-2 capitalize">{dim}</h4>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(values).map(([key, count]) => (
                                <span key={key} className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                                  {key}: {count}
                                </span>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* Surahs Tab */}
              {activeTab === 'surahs' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Globe className="w-5 h-5 text-amber-600" />
                    {language === "ar" ? `توزيع "${profile.behavior}" على السور` : `"${profile.behavior}" Distribution by Surah`}
                  </h3>
                  
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={profile.surah_distribution.slice(0, 15)} layout="vertical">
                      <XAxis type="number" />
                      <YAxis dataKey="surah" type="category" width={120} tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Vocabulary Tab */}
              {activeTab === 'vocabulary' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Hash className="w-5 h-5 text-teal-600" />
                    {language === "ar" ? `المفردات المرتبطة بـ "${profile.behavior}"` : `Vocabulary for "${profile.behavior}"`}
                  </h3>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gray-50 rounded-xl p-5">
                      <h4 className="font-medium text-gray-800 mb-3">
                        {language === "ar" ? "المصطلح الرئيسي" : "Primary Term"}
                      </h4>
                      <p className="text-2xl font-bold text-purple-600">{profile.vocabulary.primary_term}</p>
                    </div>
                    
                    <div className="bg-gray-50 rounded-xl p-5">
                      <h4 className="font-medium text-gray-800 mb-3">
                        {language === "ar" ? "المشتقات" : "Derivatives"}
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {profile.vocabulary.derivatives.map((d, i) => (
                          <span key={i} className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-sm">
                            {d}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Embeddings/Similarity Tab */}
              {activeTab === 'embeddings' && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Brain className="w-5 h-5 text-indigo-600" />
                    {language === "ar" ? `السلوكيات المشابهة لـ "${profile.behavior}"` : `Similar Behaviors to "${profile.behavior}"`}
                  </h3>
                  
                  {profile.similar_behaviors.length > 0 ? (
                    <div className="space-y-3">
                      {profile.similar_behaviors.map((b, i) => (
                        <div key={i} className="flex items-center gap-4 bg-gray-50 rounded-xl p-4">
                          <div className="flex-1">
                            <p className="font-medium text-gray-800">{b.behavior}</p>
                          </div>
                          <div className="w-32">
                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-indigo-500 rounded-full"
                                style={{ width: `${b.similarity}%` }}
                              />
                            </div>
                          </div>
                          <span className="text-sm font-medium text-indigo-600 w-12 text-right">
                            {b.similarity}%
                          </span>
                          <button
                            onClick={() => handleSuggestionClick(b.behavior)}
                            className="p-2 hover:bg-indigo-100 rounded-lg transition"
                          >
                            <ArrowRight className="w-4 h-4 text-indigo-600" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-center py-8">
                      {language === "ar" ? "لا توجد بيانات تشابه متاحة" : "No similarity data available"}
                    </p>
                  )}
                </div>
              )}
            </motion.div>
          </AnimatePresence>

          {/* Processing Time */}
          <p className="text-center text-sm text-gray-500 mt-4">
            {language === "ar" 
              ? `تم التحليل في ${profile.processing_time_ms.toFixed(0)} مللي ثانية`
              : `Analyzed in ${profile.processing_time_ms.toFixed(0)}ms`}
          </p>
        </div>
      )}

      {/* Empty State */}
      {!profile && !loading && !error && (
        <div className="max-w-7xl mx-auto px-6 py-20 text-center">
          <Layers className="w-16 h-16 text-purple-300 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            {language === "ar" ? "أدخل سلوكاً للتحليل" : "Enter a Behavior to Analyze"}
          </h2>
          <p className="text-gray-600 max-w-lg mx-auto">
            {language === "ar"
              ? "اكتب اسم السلوك في مربع البحث أعلاه للحصول على ملف شامل يتضمن جميع الآيات والتفاسير والعلاقات والأبعاد."
              : "Type a behavior name in the search box above to get a comprehensive profile including all verses, tafsir, relationships, and dimensions."}
          </p>
        </div>
      )}
    </div>
  );
}
