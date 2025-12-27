"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import dynamic from "next/dynamic";
import {
  BookOpen,
  Search,
  Filter,
  Grid,
  List,
  ChevronRight,
  X,
  BarChart3,
  Eye,
  Brain,
  MessageSquare,
  Sparkles,
  FileText,
  ArrowRight,
  Network,
  Send,
  Loader2,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const CHART_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

// Node colors for graph
const NODE_COLORS: Record<string, string> = {
  behavior: '#10b981',
  agent: '#3b82f6',
  surah: '#8b5cf6',
  default: '#6b7280',
};

interface GraphNode {
  id: string;
  name: string;
  type: string;
  val: number;
  color: string;
  count?: number;
}

interface GraphLink {
  source: string;
  target: string;
  value: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

// Surah interface for type safety
interface SurahData {
  number: number;
  name: string;
  nameEn: string;
  ayat: number;
  spans: number;
  coverage: number;
}

// Top behaviors - loaded from backend
interface BehaviorData {
  id: string;
  label: string;
  count: number;
  color: string;
}

const BEHAVIOR_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"];

function getCoverageColor(coverage: number) {
  if (coverage >= 90) return "bg-emerald-500";
  if (coverage >= 70) return "bg-emerald-400";
  if (coverage >= 50) return "bg-emerald-300";
  return "bg-emerald-200";
}

export default function ExplorerPage() {
  const { t, isRTL } = useLanguage();
  const [view, setView] = useState<"grid" | "list" | "graph">("grid");
  const [selectedSurah, setSelectedSurah] = useState<SurahData | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const graphRef = useRef<any>(null);
  const [filterBehavior, setFilterBehavior] = useState<string | null>(null);
  const [surahs, setSurahs] = useState<SurahData[]>([]);
  const [behaviors, setBehaviors] = useState<BehaviorData[]>([]);
  const [stats, setStats] = useState({
    totalSpans: 0,
    uniqueAyat: 0,
    totalSurahs: 0,
    avgCoverage: 0,
  });

  // Build graph data from stats - meaningful behavior-agent relationships
  const buildGraphFromStats = useCallback((behaviorForms: Record<string, number>, agentTypes: Record<string, number>) => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    
    // Behavior name translations
    const behaviorNames: Record<string, string> = {
      'inner_state': 'الحالة الداخلية',
      'speech_act': 'الفعل الكلامي',
      'relational_act': 'الفعل العلائقي',
      'physical_act': 'الفعل الجسدي',
      'trait_disposition': 'السمة',
    };
    
    // Agent name translations
    const agentNames: Record<string, string> = {
      'AGT_ALLAH': 'الله',
      'AGT_BELIEVER': 'المؤمن',
      'AGT_DISBELIEVER': 'الكافر',
      'AGT_HUMAN_GENERAL': 'الإنسان',
      'AGT_PROPHET': 'النبي',
      'AGT_HYPOCRITE': 'المنافق',
    };
    
    // Add behavior nodes (center)
    const behaviorsList = Object.entries(behaviorForms)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    const totalBehaviors = behaviorsList.reduce((sum, [, c]) => sum + (c as number), 0);
    
    behaviorsList.forEach(([name, count], i) => {
      const displayName = behaviorNames[name] || name.replace(/_/g, ' ');
      nodes.push({
        id: `behavior_${i}`,
        name: displayName,
        type: 'behavior',
        val: Math.max(8, Math.sqrt(count as number) * 3),
        color: '#10b981',
        count: count as number,
      });
    });

    // Add agent nodes (outer ring)
    const agentsList = Object.entries(agentTypes)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);
    
    agentsList.forEach(([name, count], i) => {
      const displayName = agentNames[name] || name.replace('AGT_', '');
      nodes.push({
        id: `agent_${i}`,
        name: displayName,
        type: 'agent',
        val: Math.max(6, Math.sqrt(count as number) * 2),
        color: '#3b82f6',
        count: count as number,
      });
    });

    // Create meaningful links - each behavior connects to agents based on co-occurrence weight
    behaviorsList.forEach(([bName, bCount], bi) => {
      agentsList.forEach(([aName, aCount], ai) => {
        // Weight based on relative frequency
        const weight = Math.min(bCount as number, aCount as number) / totalBehaviors * 10;
        if (weight > 0.5) {
          links.push({
            source: `behavior_${bi}`,
            target: `agent_${ai}`,
            value: weight,
          });
        }
      });
    });

    setGraphData({ nodes, links });
  }, []);

  // Load real data from backend on mount
  useEffect(() => {
    loadRealData();
  }, []);

  const loadRealData = async () => {
    try {
      const [statsRes, surahsRes] = await Promise.all([
        fetch(`${BACKEND_URL}/stats`),
        fetch(`${BACKEND_URL}/surahs`),
      ]);

      if (statsRes.ok) {
        const statsData = await statsRes.json();
        const forms = statsData.behavior_forms || {};
        const agents = statsData.agent_types || {};
        const behaviorList: BehaviorData[] = Object.entries(forms).map(([key, count], i) => ({
          id: key,
          label: key.replace(/_/g, " "),
          count: count as number,
          color: BEHAVIOR_COLORS[i % BEHAVIOR_COLORS.length],
        }));
        setBehaviors(behaviorList.sort((a, b) => b.count - a.count));
        setStats((prev) => ({
          ...prev,
          totalSpans: statsData.total_spans || 0,
          uniqueAyat: statsData.unique_ayat || 0,
        }));
        // Build graph data
        buildGraphFromStats(forms, agents);
      }

      if (surahsRes.ok) {
        const surahData = await surahsRes.json();
        const surahList: SurahData[] = (surahData.surahs || []).map((s: any) => {
          const totalAyat = s.total_ayat || s.unique_ayat || 0;
          const coverage =
            s.coverage_pct ?? (totalAyat ? Math.round((s.unique_ayat / totalAyat) * 100) : 0);
          const name = s.surah_name || `Surah ${s.surah}`;
          return {
            number: s.surah,
            name,
            nameEn: `Surah ${s.surah}`,
            ayat: totalAyat,
            spans: s.spans || 0,
            coverage,
          };
        });

        const avgCoverage = surahList.length
          ? Math.round(
              surahList.reduce((sum, s) => sum + (s.coverage || 0), 0) / surahList.length
            )
          : 0;

        setSurahs(surahList);
        setStats((prev) => ({
          ...prev,
          totalSurahs: surahData.total_surahs || surahList.length,
          avgCoverage,
        }));
      }
    } catch (error) {
      console.error("Failed to load data:", error);
    }
  };

  const filteredSurahs = surahs.filter(
    (s) =>
      s.name.includes(searchQuery) ||
      s.nameEn.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.number.toString().includes(searchQuery)
  );

  // Surah detail data loaded from backend
  const [surahDetails, setSurahDetails] = useState<{
    behaviors: Array<{ name: string; count: number }>;
    agents: Array<{ name: string; count: number }>;
    ayat: Array<{ surah: number; ayah: number; text: string; behaviors: string[] }>;
  } | null>(null);

  const loadSurahDetails = async (surah: SurahData) => {
    setSelectedSurah(surah);
    setIsLoading(true);
    setSurahDetails(null);

    try {
      // Load surah-specific data from backend
      const response = await fetch(`${BACKEND_URL}/api/proof/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: `Analyze behaviors in Surah ${surah.number}`,
          include_proof: true 
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        // Extract behavior distribution from proof data
        const behaviorCounts: Record<string, number> = {};
        const agentCounts: Record<string, number> = {};
        
        // Count behaviors from taxonomy if available
        if (data.proof?.taxonomy?.behaviors) {
          data.proof.taxonomy.behaviors.forEach((b: any) => {
            const form = b.behavior_form || b.form || 'unknown';
            behaviorCounts[form] = (behaviorCounts[form] || 0) + 1;
            const agent = b.agent_type || b.agent || 'unknown';
            agentCounts[agent] = (agentCounts[agent] || 0) + 1;
          });
        }

        // Extract sample ayat from Quran results
        const sampleAyat = (data.proof?.quran || []).slice(0, 4).map((v: any) => ({
          surah: v.surah,
          ayah: v.ayah,
          text: v.text,
          behaviors: v.behaviors || [],
        }));

        setSurahDetails({
          behaviors: Object.entries(behaviorCounts)
            .map(([name, count]) => ({ name, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 8),
          agents: Object.entries(agentCounts)
            .map(([name, count]) => ({ name, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 6),
          ayat: sampleAyat,
        });
      }
    } catch (error) {
      console.error("Failed to load surah details:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <div className={`min-h-[calc(100vh-64px)] bg-gray-50 ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
        {/* Header */}
        <div className="bg-white border-b border-gray-200 sticky top-0 z-20">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                  <BookOpen className="w-7 h-7 text-emerald-600" />
                  {t("explorer.title")}
                </h1>
                <p className="text-gray-600">
                  {t("explorer.subtitle")}
                </p>
              </div>

              <div className="flex items-center gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder={t("explorer.search_placeholder")}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg w-64 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>

                {/* View toggle */}
                <div className="flex items-center bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setView("grid")}
                    className={`p-2 rounded-md ${view === "grid" ? "bg-white shadow" : ""}`}
                    title="Grid View"
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setView("list")}
                    className={`p-2 rounded-md ${view === "list" ? "bg-white shadow" : ""}`}
                    title="List View"
                  >
                    <List className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setView("graph")}
                    className={`p-2 rounded-md ${view === "graph" ? "bg-white shadow" : ""}`}
                    title="Relationship Graph"
                  >
                    <Network className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Behavior filter chips */}
            <div className="flex items-center gap-2 mt-4 overflow-x-auto pb-2">
              <span className="text-sm text-gray-500 flex-shrink-0">{t("explorer.filter_by_behavior")}:</span>
              {behaviors.map((behavior) => (
                <button
                  key={behavior.id}
                  onClick={() =>
                    setFilterBehavior(filterBehavior === behavior.id ? null : behavior.id)
                  }
                  className={`flex-shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    filterBehavior === behavior.id
                      ? "bg-emerald-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {behavior.label}
                  <span className="ml-1 opacity-70">({behavior.count})</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex gap-8">
            {/* Surah Grid/List */}
            <div className={`${selectedSurah ? "w-1/2" : "w-full"} transition-all`}>
              {/* Stats bar */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { labelKey: "explorer.total_surahs", value: stats.totalSurahs.toString(), icon: BookOpen },
                  { labelKey: "explorer.annotated_ayat", value: stats.uniqueAyat.toLocaleString(), icon: Eye },
                  { labelKey: "explorer.behavioral_spans", value: stats.totalSpans.toLocaleString(), icon: Brain },
                  { labelKey: "explorer.avg_coverage", value: `${stats.avgCoverage}%`, icon: BarChart3 },
                ].map((stat) => (
                  <div
                    key={stat.labelKey}
                    className="bg-white rounded-xl p-4 border border-gray-200"
                  >
                    <stat.icon className="w-5 h-5 text-emerald-600 mb-2" />
                    <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                    <div className="text-sm text-gray-500">{t(stat.labelKey)}</div>
                  </div>
                ))}
              </div>

              {/* Grid view */}
              {view === "grid" && (
                <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-1">
                  {filteredSurahs.map((surah) => (
                    <motion.button
                      key={surah.number}
                      whileHover={{ scale: 1.1, zIndex: 10 }}
                      onClick={() => loadSurahDetails(surah)}
                      className={`aspect-square rounded-lg flex flex-col items-center justify-center text-xs font-medium transition-all cursor-pointer relative group ${getCoverageColor(
                        surah.coverage
                      )} ${
                        selectedSurah?.number === surah.number
                          ? "ring-2 ring-emerald-600 ring-offset-2"
                          : ""
                      }`}
                      title={`${surah.nameEn} - ${surah.coverage}% coverage`}
                    >
                      <span className="text-white font-bold">{surah.number}</span>
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-20">
                        {surah.nameEn}
                        <br />
                        {surah.coverage}% coverage
                      </div>
                    </motion.button>
                  ))}
                </div>
              )}

              {/* List view */}
              {view === "list" && (
                <div className="space-y-2">
                  {filteredSurahs.slice(0, 20).map((surah) => (
                    <motion.button
                      key={surah.number}
                      whileHover={{ x: 4 }}
                      onClick={() => loadSurahDetails(surah)}
                      className={`w-full flex items-center gap-4 p-4 bg-white rounded-xl border transition-all hover:border-emerald-300 hover:shadow-md ${
                        selectedSurah?.number === surah.number
                          ? "border-emerald-500 shadow-md"
                          : "border-gray-200"
                      }`}
                    >
                      <div
                        className={`w-12 h-12 rounded-xl flex items-center justify-center text-white font-bold ${getCoverageColor(
                          surah.coverage
                        )}`}
                      >
                        {surah.number}
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-semibold text-gray-900">{surah.nameEn}</div>
                        <div className="text-sm text-gray-500 font-arabic">{surah.name}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {surah.spans} {t("explorer.spans")}
                        </div>
                        <div className="text-xs text-gray-500">{surah.ayat} {t("explorer.ayat")}</div>
                      </div>
                      <div className="w-24">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="text-gray-500">{t("explorer.coverage")}</span>
                          <span className="font-medium">{surah.coverage}%</span>
                        </div>
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full"
                            style={{ width: `${surah.coverage}%` }}
                          />
                        </div>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    </motion.button>
                  ))}
                </div>
              )}

              {/* Graph view */}
              {view === "graph" && graphData && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-white rounded-xl shadow-lg overflow-hidden"
                >
                  <div className="p-4 border-b bg-gradient-to-r from-emerald-50 to-white">
                    <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                      <Network className="w-5 h-5 text-emerald-600" />
                      {t("explorer.relationship_graph") || "Behavior-Agent Relationship Network"}
                    </h3>
                    <p className="text-sm text-gray-500 mt-1">
                      {t("explorer.graph_instructions") || "Click any node for details • Drag to pan • Scroll to zoom"}
                    </p>
                  </div>
                  
                  <div className="flex">
                    {/* Graph */}
                    <div className="flex-1 h-[500px] bg-gradient-to-br from-slate-900 to-slate-800">
                      <ForceGraph2D
                        ref={graphRef}
                        graphData={graphData}
                        nodeLabel={(node: any) => `${node.name} (${node.type})`}
                        nodeColor={(node: any) => node.color}
                        nodeVal={(node: any) => node.val}
                        linkColor={() => 'rgba(255,255,255,0.15)'}
                        linkWidth={(link: any) => Math.max(1, link.value * 0.3)}
                        onNodeClick={(node: any) => setSelectedNode(node)}
                        nodeCanvasObject={(node: any, ctx, globalScale) => {
                          const label = node.name;
                          const fontSize = Math.max(10, 14 / globalScale);
                          const nodeSize = Math.max(6, node.val * 1.5);
                          
                          // Draw glow effect
                          ctx.beginPath();
                          ctx.arc(node.x, node.y, nodeSize + 4, 0, 2 * Math.PI);
                          ctx.fillStyle = node.color + '40';
                          ctx.fill();
                          
                          // Draw node circle
                          ctx.beginPath();
                          ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
                          ctx.fillStyle = node.color;
                          ctx.fill();
                          ctx.strokeStyle = '#fff';
                          ctx.lineWidth = 1.5;
                          ctx.stroke();
                          
                          // Draw label with background
                          ctx.font = `bold ${fontSize}px Arial, sans-serif`;
                          ctx.textAlign = 'center';
                          ctx.textBaseline = 'middle';
                          const textWidth = ctx.measureText(label).width;
                          
                          // Label background
                          ctx.fillStyle = 'rgba(0,0,0,0.7)';
                          ctx.fillRect(node.x - textWidth/2 - 4, node.y + nodeSize + 4, textWidth + 8, fontSize + 4);
                          
                          // Label text
                          ctx.fillStyle = '#fff';
                          ctx.fillText(label, node.x, node.y + nodeSize + fontSize/2 + 6);
                        }}
                        cooldownTicks={100}
                        d3AlphaDecay={0.02}
                        d3VelocityDecay={0.3}
                        onEngineStop={() => graphRef.current?.zoomToFit(400, 50)}
                      />
                    </div>

                    {/* Legend & Selected Node Info */}
                    <div className="w-72 p-4 border-l bg-white overflow-y-auto max-h-[500px]">
                      <h4 className="font-bold text-gray-700 mb-3">{t("explorer.legend") || "دليل الألوان"}</h4>
                      <div className="space-y-2 mb-4">
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-4 rounded-full bg-emerald-500"></div>
                          <span className="text-sm text-gray-600">{t("explorer.behavior") || "السلوك"}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                          <span className="text-sm text-gray-600">{t("explorer.agent") || "الفاعل"}</span>
                        </div>
                      </div>

                      {/* Node Statistics */}
                      <div className="border-t pt-4 mb-4">
                        <h4 className="font-bold text-gray-700 mb-2">إحصائيات الشبكة</h4>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-emerald-50 rounded-lg p-2 text-center">
                            <p className="text-lg font-bold text-emerald-600">{graphData.nodes.filter(n => n.type === 'behavior').length}</p>
                            <p className="text-xs text-gray-500">سلوكيات</p>
                          </div>
                          <div className="bg-blue-50 rounded-lg p-2 text-center">
                            <p className="text-lg font-bold text-blue-600">{graphData.nodes.filter(n => n.type === 'agent').length}</p>
                            <p className="text-xs text-gray-500">فاعلين</p>
                          </div>
                        </div>
                        <div className="mt-2 bg-gray-50 rounded-lg p-2 text-center">
                          <p className="text-lg font-bold text-gray-600">{graphData.links.length}</p>
                          <p className="text-xs text-gray-500">علاقات</p>
                        </div>
                      </div>

                      {/* All Nodes List */}
                      <div className="border-t pt-4">
                        <h4 className="font-bold text-gray-700 mb-2">العقد</h4>
                        <div className="space-y-1 max-h-40 overflow-y-auto">
                          {graphData.nodes.map((node, i) => (
                            <button
                              key={i}
                              onClick={() => setSelectedNode(node)}
                              className={`w-full text-right px-2 py-1.5 rounded text-sm flex items-center justify-between hover:bg-gray-100 ${selectedNode?.id === node.id ? 'bg-gray-100' : ''}`}
                            >
                              <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${node.type === 'behavior' ? 'bg-emerald-500' : 'bg-blue-500'}`} />
                                <span className="text-gray-700">{node.name}</span>
                              </div>
                              <span className="text-xs text-gray-400">{(node as any).count?.toLocaleString()}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      {selectedNode && (
                        <div className="border-t pt-4 mt-4">
                          <h4 className="font-bold text-gray-700 mb-2">{t("explorer.selected_node") || "العقدة المحددة"}</h4>
                          <div className={`rounded-lg p-3 ${selectedNode.type === 'behavior' ? 'bg-emerald-50' : 'bg-blue-50'}`}>
                            <p className="font-bold text-gray-800 text-lg">{selectedNode.name}</p>
                            <p className="text-sm text-gray-500">{selectedNode.type === 'behavior' ? 'سلوك' : 'فاعل'}</p>
                            {(selectedNode as any).count && (
                              <p className="text-2xl font-bold mt-2 text-gray-800">{(selectedNode as any).count.toLocaleString()}</p>
                            )}
                            <a
                              href={`/proof?q=${encodeURIComponent(selectedNode.name)}`}
                              className="mt-3 w-full py-2 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition flex items-center justify-center gap-2"
                            >
                              <Search className="w-4 h-4" />
                              {t("explorer.search_this") || "عرض الإثبات"}
                            </a>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Legend */}
              <div className="mt-6 flex items-center gap-6 text-sm text-gray-600">
                <span className="font-medium">{t("explorer.coverage")}:</span>
                {[
                  { label: "90%+", color: "bg-emerald-500" },
                  { label: "70-89%", color: "bg-emerald-400" },
                  { label: "50-69%", color: "bg-emerald-300" },
                  { label: "<50%", color: "bg-emerald-200" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded ${item.color}`} />
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Detail Panel */}
            <AnimatePresence>
              {selectedSurah && (
                <motion.div
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 50 }}
                  className="w-1/2 bg-white rounded-2xl border border-gray-200 overflow-hidden"
                >
                  {/* Header */}
                  <div className="bg-gradient-to-r from-emerald-600 to-emerald-700 text-white p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-4xl font-arabic mb-2">{selectedSurah.name}</div>
                        <h2 className="text-2xl font-bold">{selectedSurah.nameEn}</h2>
                        <p className="text-emerald-200">Surah {selectedSurah.number}</p>
                      </div>
                      <button
                        onClick={() => setSelectedSurah(null)}
                        className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* Content - Native Charts */}
                  <div className="p-6 max-h-[calc(100vh-300px)] overflow-y-auto custom-scrollbar">
                    {isLoading ? (
                      <div className="flex items-center justify-center h-64">
                        <div className="text-center">
                          <div className="flex items-center gap-2 justify-center mb-4">
                            <Sparkles className="w-6 h-6 text-emerald-500 animate-pulse" />
                            <span className="text-gray-600">{t("explorer.generating")}</span>
                          </div>
                          <div className="flex gap-1 justify-center">
                            {[0, 1, 2].map((i) => (
                              <div
                                key={i}
                                className="w-2 h-2 rounded-full bg-emerald-500 streaming-dot"
                                style={{ animationDelay: `${i * 0.2}s` }}
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : surahDetails ? (
                      <div className="space-y-6">
                        {/* Stats Cards */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl p-4 text-white">
                            <p className="text-3xl font-bold">{selectedSurah.ayat}</p>
                            <p className="text-sm opacity-80">{t("explorer.ayat")}</p>
                          </div>
                          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-4 text-white">
                            <p className="text-3xl font-bold">{selectedSurah.spans}</p>
                            <p className="text-sm opacity-80">{t("explorer.spans")}</p>
                          </div>
                          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-4 text-white">
                            <p className="text-3xl font-bold">{selectedSurah.coverage}%</p>
                            <p className="text-sm opacity-80">{t("explorer.coverage")}</p>
                          </div>
                        </div>

                        {/* Behavior Distribution Chart */}
                        {surahDetails.behaviors.length > 0 && (
                          <div className="bg-gray-50 rounded-xl p-4">
                            <h4 className="font-bold text-gray-800 mb-3">{t("explorer.behavior_distribution")}</h4>
                            <ResponsiveContainer width="100%" height={200}>
                              <BarChart data={surahDetails.behaviors} layout="vertical">
                                <XAxis type="number" />
                                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11 }} />
                                <Tooltip />
                                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                                  {surahDetails.behaviors.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* Agent Distribution Pie */}
                        {surahDetails.agents.length > 0 && (
                          <div className="bg-gray-50 rounded-xl p-4">
                            <h4 className="font-bold text-gray-800 mb-3">{t("explorer.agent_distribution")}</h4>
                            <ResponsiveContainer width="100%" height={180}>
                              <PieChart>
                                <Pie
                                  data={surahDetails.agents}
                                  cx="50%"
                                  cy="50%"
                                  innerRadius={40}
                                  outerRadius={70}
                                  paddingAngle={5}
                                  dataKey="count"
                                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                >
                                  {surahDetails.agents.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                                  ))}
                                </Pie>
                                <Tooltip />
                              </PieChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* Sample Ayat */}
                        {surahDetails.ayat.length > 0 && (
                          <div>
                            <h4 className="font-bold text-gray-800 mb-3">{t("explorer.notable_ayat")}</h4>
                            <div className="space-y-3">
                              {surahDetails.ayat.map((ayah, i) => (
                                <motion.div
                                  key={i}
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ delay: i * 0.1 }}
                                  className="bg-white rounded-lg p-4 border-r-4 border-emerald-500"
                                >
                                  <p className="text-gray-800 leading-relaxed" dir="rtl">{ayah.text}</p>
                                  <p className="text-sm text-emerald-600 mt-2">📍 {ayah.surah}:{ayah.ayah}</p>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex gap-3">
                          <a
                            href={`/proof?q=Surah ${selectedSurah.number} behaviors`}
                            className="flex-1 py-3 bg-emerald-100 text-emerald-700 rounded-lg font-medium text-center hover:bg-emerald-200 transition flex items-center justify-center gap-2"
                          >
                            <FileText className="w-4 h-4" />
                            {t("explorer.view_full_proof")}
                          </a>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-gray-500">
                        {t("explorer.select_surah")}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
  );
}
