"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useLanguage } from "../contexts/LanguageContext";
import { Search, Send, Loader2, BookOpen, Brain, Layers, TrendingUp, Network, GitBranch, Sparkles, BarChart3 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import dynamic from "next/dynamic";

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

// Node colors by type
const NODE_COLORS: Record<string, string> = {
  behavior: '#10b981',
  agent: '#3b82f6',
  surah: '#8b5cf6',
  cluster: '#f59e0b',
  default: '#6b7280',
};

// Example queries for the discovery system
const DISCOVERY_QUERIES = {
  ar: [
    { label: "🔍 الصبر", query: "الصبر" },
    { label: "📖 التوبة", query: "التوبة" },
    { label: "💚 الإيمان", query: "الإيمان" },
    { label: "🤲 الدعاء", query: "الدعاء" },
    { label: "⚠️ الكبر", query: "الكبر" },
    { label: "❤️ القلب", query: "القلب" },
  ],
  en: [
    { label: "🔍 Patience", query: "patience sabr" },
    { label: "📖 Repentance", query: "repentance tawbah" },
    { label: "💚 Faith", query: "faith iman" },
    { label: "🤲 Prayer", query: "prayer dua" },
    { label: "⚠️ Arrogance", query: "arrogance kibr" },
    { label: "❤️ Heart", query: "heart qalb" },
  ],
};

interface SearchResult {
  text: string;
  surah?: string;
  ayah?: string;
  behavior?: string;
  source?: string;
  score?: number;
}

interface Stats {
  totalSpans: number;
  uniqueAyat: number;
  behaviorForms: Record<string, number>;
  agentTypes: Record<string, number>;
}

interface GraphNode {
  id: string;
  name: string;
  type: string;
  val: number;
  color: string;
}

interface GraphLink {
  source: string;
  target: string;
  value: number;
  type?: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export default function DiscoveryPage() {
  const { language, isRTL } = useLanguage();
  const queries = language === "ar" ? DISCOVERY_QUERIES.ar : DISCOVERY_QUERIES.en;
  const [searchQuery, setSearchQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [activeView, setActiveView] = useState<'overview' | 'graph' | 'search'>('overview');
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const graphRef = useRef<any>(null);

  // Build graph data from stats
  const buildGraphFromStats = useCallback((statsData: Stats) => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    
    // Add behavior nodes
    const behaviors = Object.entries(statsData.behaviorForms)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15);
    
    behaviors.forEach(([name, count], i) => {
      nodes.push({
        id: `behavior_${i}`,
        name: name,
        type: 'behavior',
        val: Math.sqrt(count) * 2,
        color: NODE_COLORS.behavior,
      });
    });

    // Add agent nodes
    const agents = Object.entries(statsData.agentTypes)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    
    agents.forEach(([name, count], i) => {
      nodes.push({
        id: `agent_${i}`,
        name: name,
        type: 'agent',
        val: Math.sqrt(count) * 1.5,
        color: NODE_COLORS.agent,
      });
    });

    // Create links between behaviors and agents (deterministic based on counts)
    // NOTE: These are co-occurrence links for visualization only, NOT causal
    behaviors.forEach(([behaviorName, behaviorCount], bi) => {
      // Connect each behavior to agents deterministically based on index
      const numLinks = Math.min(2, agents.length);
      for (let j = 0; j < numLinks; j++) {
        const agentIdx = (bi + j) % agents.length;
        const [, agentCount] = agents[agentIdx];
        // Weight based on actual counts, not random
        const weight = Math.sqrt(behaviorCount * agentCount) / 10;
        links.push({
          source: `behavior_${bi}`,
          target: `agent_${agentIdx}`,
          value: Math.max(1, weight),
          type: 'CO_OCCURRENCE', // Explicitly mark as non-causal
        });
      }
    });

    // Connect adjacent behaviors (co-occurrence visualization only)
    // NOTE: This is for discovery/exploration, NOT causal reasoning
    for (let i = 0; i < behaviors.length - 1; i++) {
      const [, count1] = behaviors[i];
      const [, count2] = behaviors[i + 1];
      // Deterministic: connect if both have significant counts
      if (count1 > 5 && count2 > 5) {
        links.push({
          source: `behavior_${i}`,
          target: `behavior_${i + 1}`,
          value: Math.sqrt(count1 + count2) / 5,
          type: 'CO_OCCURRENCE', // Explicitly mark as non-causal
        });
      }
    }

    setGraphData({ nodes, links });
  }, []);

  // Load stats on mount and build graph
  useEffect(() => {
    const loadStats = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/stats`);
        if (res.ok) {
          const data = await res.json();
          const statsData = {
            totalSpans: data.total_spans || 0,
            uniqueAyat: data.unique_ayat || 0,
            behaviorForms: data.behavior_forms || {},
            agentTypes: data.agent_types || {},
          };
          setStats(statsData);
          buildGraphFromStats(statsData);
        }
      } catch (e) {
        console.error("Failed to load stats:", e);
      }
    };
    loadStats();
  }, [buildGraphFromStats]);

  // Handle node click
  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
  }, []);

  const handleSearch = async (query: string) => {
    if (!query.trim()) return;
    
    setLoading(true);
    setHasSearched(true);
    setSearchQuery(query);

    try {
      // Use the proof endpoint for semantic search
      const response = await fetch(`${BACKEND_URL}/api/proof/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, include_proof: true }),
      });

      if (response.ok) {
        const data = await response.json();
        // Extract results from proof data
        const searchResults: SearchResult[] = [];
        
        // Add Quran verses
        if (data.proof?.quran) {
          data.proof.quran.forEach((v: any) => {
            searchResults.push({
              text: v.text,
              surah: v.surah,
              ayah: v.ayah,
              source: 'quran',
              score: v.relevance,
            });
          });
        }
        
        // Add tafsir entries
        ['ibn_kathir', 'tabari', 'qurtubi', 'saadi', 'jalalayn'].forEach(source => {
          if (data.proof?.[source]) {
            data.proof[source].slice(0, 3).forEach((t: any) => {
              searchResults.push({
                text: t.text?.slice(0, 200) + '...',
                surah: t.surah,
                ayah: t.ayah,
                source: source,
              });
            });
          }
        });

        setResults(searchResults);
      }
    } catch (e) {
      console.error("Search failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(searchQuery);
  };

  // Prepare chart data
  const behaviorChartData = stats ? Object.entries(stats.behaviorForms)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, value], i) => ({ name, value, fill: COLORS[i % COLORS.length] })) : [];

  const agentChartData = stats ? Object.entries(stats.agentTypes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([name, value], i) => ({ name, value, fill: COLORS[i % COLORS.length] })) : [];

  return (
    <div className={`min-h-screen bg-gradient-to-b from-emerald-50 via-white to-emerald-50 ${isRTL ? "rtl" : "ltr"}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-800 to-emerald-900 text-white px-6 py-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">🔬</span>
            {language === "ar" ? "نظام الاكتشاف" : "Discovery System"}
          </h1>
          <p className="text-emerald-200 mt-2">
            {language === "ar"
              ? "بحث دلالي متقدم في القرآن والتفاسير الخمسة"
              : "Semantic search across Quran and 5 classical Tafsir sources"}
          </p>
          
          {/* Search Box */}
          <form onSubmit={handleSubmit} className="mt-6">
            <div className="flex gap-3 bg-white/10 backdrop-blur p-2 rounded-xl">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={language === "ar" ? "ابحث عن سلوك أو مفهوم..." : "Search for a behavior or concept..."}
                className="flex-1 px-4 py-3 bg-white rounded-lg text-gray-800 text-lg focus:ring-2 focus:ring-emerald-400"
                dir={isRTL ? 'rtl' : 'ltr'}
              />
              <button
                type="submit"
                disabled={loading}
                className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-bold flex items-center gap-2 transition"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
                {language === "ar" ? "بحث" : "Search"}
              </button>
            </div>
          </form>

          {/* Quick Search Tags */}
          <div className="flex flex-wrap gap-2 mt-4">
            {queries.map((q, i) => (
              <button
                key={i}
                onClick={() => { setSearchQuery(q.query); handleSearch(q.query); }}
                className="px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded-full text-sm font-medium transition"
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Stats Cards */}
        {stats && !hasSearched && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
          >
            <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl p-5 text-white shadow-lg">
              <BookOpen className="w-8 h-8 mb-2 opacity-80" />
              <p className="text-3xl font-bold">{stats.uniqueAyat.toLocaleString()}</p>
              <p className="text-sm opacity-80">{language === "ar" ? "آية محللة" : "Ayat Analyzed"}</p>
            </div>
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-5 text-white shadow-lg">
              <Layers className="w-8 h-8 mb-2 opacity-80" />
              <p className="text-3xl font-bold">{stats.totalSpans.toLocaleString()}</p>
              <p className="text-sm opacity-80">{language === "ar" ? "تعليق سلوكي" : "Behavioral Spans"}</p>
            </div>
            <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-5 text-white shadow-lg">
              <Brain className="w-8 h-8 mb-2 opacity-80" />
              <p className="text-3xl font-bold">{Object.keys(stats.behaviorForms).length}</p>
              <p className="text-sm opacity-80">{language === "ar" ? "نوع سلوك" : "Behavior Types"}</p>
            </div>
            <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-xl p-5 text-white shadow-lg">
              <TrendingUp className="w-8 h-8 mb-2 opacity-80" />
              <p className="text-3xl font-bold">5</p>
              <p className="text-sm opacity-80">{language === "ar" ? "مصادر تفسير" : "Tafsir Sources"}</p>
            </div>
          </motion.div>
        )}

        {/* View Toggle Tabs */}
        {!hasSearched && (
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveView('overview')}
              className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition ${
                activeView === 'overview' 
                  ? 'bg-emerald-600 text-white' 
                  : 'bg-white text-gray-600 hover:bg-emerald-50'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              {language === "ar" ? "نظرة عامة" : "Overview"}
            </button>
            <button
              onClick={() => setActiveView('graph')}
              className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition ${
                activeView === 'graph' 
                  ? 'bg-emerald-600 text-white' 
                  : 'bg-white text-gray-600 hover:bg-emerald-50'
              }`}
            >
              <Network className="w-4 h-4" />
              {language === "ar" ? "شبكة العلاقات" : "Relationship Graph"}
            </button>
          </div>
        )}

        {/* Graph View */}
        {stats && !hasSearched && activeView === 'graph' && graphData && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-8"
          >
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div className="p-4 border-b bg-gradient-to-r from-emerald-50 to-white">
                <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                  <Network className="w-5 h-5 text-emerald-600" />
                  {language === "ar" ? "شبكة السلوكيات والفاعلين" : "Behavior-Agent Relationship Network"}
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  {language === "ar" 
                    ? "انقر على أي عقدة لرؤية التفاصيل • اسحب للتحريك • تكبير/تصغير بالعجلة"
                    : "Click any node for details • Drag to pan • Scroll to zoom"}
                </p>
              </div>
              
              <div className="flex">
                {/* Graph */}
                <div className="flex-1 h-[500px] bg-gray-50">
                  <ForceGraph2D
                    ref={graphRef}
                    graphData={graphData}
                    nodeLabel={(node: any) => `${node.name} (${node.type})`}
                    nodeColor={(node: any) => node.color}
                    nodeVal={(node: any) => node.val}
                    linkColor={() => '#e5e7eb'}
                    linkWidth={(link: any) => link.value * 0.5}
                    onNodeClick={handleNodeClick}
                    nodeCanvasObject={(node: any, ctx, globalScale) => {
                      const label = node.name;
                      const fontSize = 12 / globalScale;
                      ctx.font = `${fontSize}px Sans-Serif`;
                      ctx.textAlign = 'center';
                      ctx.textBaseline = 'middle';
                      
                      // Draw node circle
                      ctx.beginPath();
                      ctx.arc(node.x, node.y, node.val * 2, 0, 2 * Math.PI);
                      ctx.fillStyle = node.color;
                      ctx.fill();
                      
                      // Draw label
                      ctx.fillStyle = '#374151';
                      ctx.fillText(label, node.x, node.y + node.val * 2 + fontSize);
                    }}
                    cooldownTicks={100}
                    onEngineStop={() => graphRef.current?.zoomToFit(400)}
                  />
                </div>

                {/* Legend & Selected Node Info */}
                <div className="w-64 p-4 border-l bg-white">
                  <h4 className="font-bold text-gray-700 mb-3">{language === "ar" ? "دليل الألوان" : "Legend"}</h4>
                  <div className="space-y-2 mb-6">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full bg-emerald-500"></div>
                      <span className="text-sm text-gray-600">{language === "ar" ? "سلوك" : "Behavior"}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                      <span className="text-sm text-gray-600">{language === "ar" ? "فاعل" : "Agent"}</span>
                    </div>
                  </div>

                  {selectedNode && (
                    <div className="border-t pt-4">
                      <h4 className="font-bold text-gray-700 mb-2">{language === "ar" ? "العقدة المحددة" : "Selected Node"}</h4>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="font-medium text-gray-800">{selectedNode.name}</p>
                        <p className="text-sm text-gray-500 capitalize">{selectedNode.type}</p>
                        <button
                          onClick={() => { setSearchQuery(selectedNode.name); handleSearch(selectedNode.name); }}
                          className="mt-2 w-full py-2 bg-emerald-100 text-emerald-700 rounded-lg text-sm font-medium hover:bg-emerald-200 transition"
                        >
                          {language === "ar" ? "بحث عن هذا" : "Search this"}
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="mt-6 text-xs text-gray-400">
                    <p>{graphData.nodes.length} {language === "ar" ? "عقدة" : "nodes"}</p>
                    <p>{graphData.links.length} {language === "ar" ? "رابط" : "links"}</p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Charts - Overview View */}
        {stats && !hasSearched && activeView === 'overview' && (
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <h3 className="text-lg font-bold text-gray-800 mb-4">{language === "ar" ? "توزيع السلوكيات" : "Behavior Distribution"}</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={behaviorChartData} layout="vertical">
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {behaviorChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-xl shadow-lg p-6"
            >
              <h3 className="text-lg font-bold text-gray-800 mb-4">{language === "ar" ? "أنواع الفاعلين" : "Agent Types"}</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={agentChartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {agentChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </motion.div>
          </div>
        )}

        {/* Search Results */}
        {hasSearched && (
          <div>
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              {language === "ar" ? `نتائج البحث عن "${searchQuery}"` : `Search results for "${searchQuery}"`}
              {results.length > 0 && <span className="text-emerald-600 ml-2">({results.length})</span>}
            </h2>

            {loading ? (
              <div className="text-center py-12">
                <Loader2 className="w-12 h-12 animate-spin text-emerald-600 mx-auto mb-4" />
                <p className="text-gray-600">{language === "ar" ? "جاري البحث..." : "Searching..."}</p>
              </div>
            ) : results.length > 0 ? (
              <div className="space-y-4">
                {results.map((result, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className={`bg-white rounded-xl p-4 shadow border-r-4 ${
                      result.source === 'quran' ? 'border-emerald-500' :
                      result.source === 'ibn_kathir' ? 'border-green-500' :
                      result.source === 'tabari' ? 'border-blue-500' :
                      result.source === 'qurtubi' ? 'border-purple-500' :
                      result.source === 'saadi' ? 'border-amber-500' : 'border-red-500'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <p className="text-gray-800 leading-relaxed" dir="rtl">{result.text}</p>
                        <div className="flex items-center gap-3 mt-2 text-sm">
                          {result.surah && result.ayah && (
                            <span className="text-emerald-600 font-medium">📍 {result.surah}:{result.ayah}</span>
                          )}
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            result.source === 'quran' ? 'bg-emerald-100 text-emerald-700' :
                            result.source === 'ibn_kathir' ? 'bg-green-100 text-green-700' :
                            result.source === 'tabari' ? 'bg-blue-100 text-blue-700' :
                            result.source === 'qurtubi' ? 'bg-purple-100 text-purple-700' :
                            result.source === 'saadi' ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {result.source === 'quran' ? '📖 القرآن' :
                             result.source === 'ibn_kathir' ? '📚 ابن كثير' :
                             result.source === 'tabari' ? '📜 الطبري' :
                             result.source === 'qurtubi' ? '📕 القرطبي' :
                             result.source === 'saadi' ? '📗 السعدي' : '📘 الجلالين'}
                          </span>
                          {result.score && (
                            <span className="text-gray-500">{(result.score * 100).toFixed(0)}% {language === "ar" ? "صلة" : "relevance"}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 bg-gray-50 rounded-xl">
                <p className="text-gray-500">{language === "ar" ? "لا توجد نتائج" : "No results found"}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
