"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Network, Info, ZoomIn, Maximize2, AlertCircle, RefreshCw } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import { SemanticGraph, EdgeTypeFilter, ALL_EDGE_TYPES } from "../components/graph";
import {
  EdgeType,
  EDGE_COLORS,
  NODE_COLORS,
  NODE_TYPE_LABELS,
} from "@/lib/semantic-graph";
import { useGraph } from "@/lib/api/hooks";

export default function GraphPage() {
  const { language, isRTL } = useLanguage();
  const [selectedEdgeTypes, setSelectedEdgeTypes] = useState<EdgeType[]>(ALL_EDGE_TYPES);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // Fetch graph data from API
  const { data: graphData, isLoading, error, refetch } = useGraph();

  // Extract nodes and edges from API response
  const nodes = useMemo(() => graphData?.nodes || [], [graphData]);
  const edges = useMemo(() => graphData?.edges || [], [graphData]);
  const stats = useMemo(() => graphData?.statistics || {
    total_nodes: 0,
    total_edges: 0,
    edges_by_type: {}
  }, [graphData]);

  // Get selected node data
  const selectedNodeData = useMemo(() => {
    if (!selectedNode || !nodes.length) return null;
    return nodes.find(n => n.id === selectedNode) || null;
  }, [selectedNode, nodes]);

  // Get connected edges for selected node
  const connectedEdges = useMemo(() => {
    if (!selectedNode || !edges.length) return [];
    return edges.filter(e => e.source === selectedNode || e.target === selectedNode);
  }, [selectedNode, edges]);

  // Filter connected edges by selected types
  const filteredConnectedEdges = useMemo(() =>
    connectedEdges.filter((e) => selectedEdgeTypes.includes(e.type as EdgeType)),
    [connectedEdges, selectedEdgeTypes]
  );

  // Get node by ID helper
  const getNodeById = (id: string) => nodes.find(n => n.id === id);

  // Calculate node counts by type
  const nodesByType = useMemo(() => {
    const counts: Record<string, number> = {};
    nodes.forEach(n => {
      counts[n.type] = (counts[n.type] || 0) + 1;
    });
    return counts;
  }, [nodes]);

  // Error state
  if (error) {
    return (
      <div className={`min-h-screen bg-slate-900 text-white flex items-center justify-center ${isRTL ? "rtl" : "ltr"}`}>
        <div className="text-center max-w-md mx-auto p-8">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">
            {isRTL ? "خطأ في تحميل الشبكة" : "Failed to Load Graph"}
          </h2>
          <p className="text-slate-400 mb-6">
            {isRTL
              ? "تأكد من تشغيل الخادم الخلفي على المنفذ 8000"
              : "Make sure the backend server is running on port 8000"}
          </p>
          <button
            onClick={() => refetch()}
            className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            {isRTL ? "إعادة المحاولة" : "Retry"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-slate-900 text-white ${isRTL ? "rtl" : "ltr"}`}>
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            {/* Title */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg">
                <Network className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  {isRTL ? "مستكشف الشبكة الدلالية" : "Semantic Graph Explorer"}
                </h1>
                <p className="text-slate-400 text-sm">
                  {isRTL
                    ? `تصور ${stats.total_edges.toLocaleString()} علاقة سلوكية عبر 7 أنواع روابط`
                    : `Visualizing ${stats.total_edges.toLocaleString()} behavioral relationships across 7 edge types`}
                </p>
              </div>
            </div>

            {/* Stats */}
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400">
                  {isLoading ? "..." : stats.total_nodes.toLocaleString()}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "عقد" : "Nodes"}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {isLoading ? "..." : stats.total_edges.toLocaleString()}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "روابط" : "Edges"}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {Object.keys(stats.edges_by_type || {}).length || 7}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "أنواع" : "Types"}
                </div>
              </div>
            </div>
          </div>

          {/* Data source indicator */}
          <div className={`mt-4 flex items-center gap-2 text-xs px-3 py-2 rounded-lg border w-fit ${
            isLoading
              ? "text-amber-400 bg-amber-900/30 border-amber-700/50"
              : "text-emerald-400 bg-emerald-900/30 border-emerald-700/50"
          }`}>
            <Info className="w-4 h-4" />
            {isLoading
              ? (isRTL ? "جاري تحميل البيانات من الخادم..." : "Loading data from backend...")
              : (isRTL
                  ? `متصل بالخادم: ${stats.total_nodes} عقدة، ${stats.total_edges.toLocaleString()} رابط`
                  : `Connected to backend: ${stats.total_nodes} nodes, ${stats.total_edges.toLocaleString()} edges`)}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="space-y-6">
            {/* Edge Type Filter */}
            <EdgeTypeFilter
              selected={selectedEdgeTypes}
              onChange={setSelectedEdgeTypes}
              language={language}
              counts={stats.edges_by_type}
            />

            {/* Selected Node Info */}
            {selectedNodeData && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-slate-800 rounded-lg p-4 border border-slate-700"
              >
                <h3 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: NODE_COLORS[selectedNodeData.type] || "#64748b" }}
                  />
                  {isRTL ? "العقدة المحددة" : "Selected Node"}
                </h3>

                <div className="space-y-3">
                  <div>
                    <div className="text-2xl font-arabic text-white" dir="rtl">
                      {selectedNodeData.labelAr || selectedNodeData.label_ar || selectedNodeData.id}
                    </div>
                    <div className="text-slate-300">
                      {selectedNodeData.label || selectedNodeData.label_en || selectedNodeData.id}
                    </div>
                    <div className="text-xs text-slate-500 font-mono mt-1">
                      {selectedNodeData.id}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className="px-2 py-1 rounded text-xs"
                      style={{
                        backgroundColor: (NODE_COLORS[selectedNodeData.type] || "#64748b") + "30",
                        color: NODE_COLORS[selectedNodeData.type] || "#64748b",
                      }}
                    >
                      {NODE_TYPE_LABELS[selectedNodeData.type]
                        ? (isRTL ? NODE_TYPE_LABELS[selectedNodeData.type].ar : NODE_TYPE_LABELS[selectedNodeData.type].en)
                        : selectedNodeData.type}
                    </span>
                  </div>

                  {/* Connected edges */}
                  <div className="pt-3 border-t border-slate-700">
                    <div className="text-xs text-slate-400 mb-2">
                      {isRTL ? "الروابط المتصلة:" : "Connected edges:"}{" "}
                      <span className="text-white">{filteredConnectedEdges.length}</span>
                    </div>
                    <div className="space-y-1 max-h-40 overflow-y-auto">
                      {filteredConnectedEdges.slice(0, 10).map((edge, i) => {
                        const otherNodeId =
                          edge.source === selectedNode ? edge.target : edge.source;
                        const otherNode = getNodeById(otherNodeId);
                        return (
                          <div
                            key={i}
                            className="text-xs flex items-center gap-2 py-1 hover:bg-slate-700/50 rounded px-1 cursor-pointer"
                            onClick={() => setSelectedNode(otherNodeId)}
                          >
                            <span
                              className="w-2 h-2 rounded-full flex-shrink-0"
                              style={{ backgroundColor: EDGE_COLORS[edge.type as EdgeType] || "#64748b" }}
                            />
                            <span className="text-slate-400 truncate flex-1">
                              {edge.type.replace("_", " ")}
                            </span>
                            <span className="text-slate-300 truncate">
                              {otherNode?.label || otherNode?.label_en || otherNodeId}
                            </span>
                          </div>
                        );
                      })}
                      {filteredConnectedEdges.length > 10 && (
                        <div className="text-xs text-slate-500 pt-1">
                          +{filteredConnectedEdges.length - 10} {isRTL ? "المزيد" : "more"}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setSelectedNode(null)}
                  className="mt-4 text-xs text-slate-400 hover:text-white transition-colors"
                >
                  {isRTL ? "إلغاء التحديد" : "Clear selection"}
                </button>
              </motion.div>
            )}

            {/* Node Types Legend */}
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <h3 className="text-sm font-medium text-slate-300 mb-3">
                {isRTL ? "أنواع العقد" : "Node Types"}
              </h3>
              <div className="space-y-2">
                {(
                  Object.entries(NODE_TYPE_LABELS) as [
                    keyof typeof NODE_TYPE_LABELS,
                    { en: string; ar: string }
                  ][]
                ).map(([type, labels]) => (
                  <div key={type} className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: NODE_COLORS[type] }}
                    />
                    <span className="text-slate-300 text-sm">
                      {isRTL ? labels.ar : labels.en}
                    </span>
                    <span className="text-slate-500 text-xs ml-auto">
                      {nodesByType[type] || 0}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Graph */}
          <div className="lg:col-span-3">
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <SemanticGraph
                width={900}
                height={650}
                selectedNode={selectedNode}
                onNodeClick={setSelectedNode}
                filterEdgeTypes={selectedEdgeTypes}
                nodes={nodes}
                edges={edges}
                isLoading={isLoading}
              />
            </motion.div>

            {/* Graph Controls Info */}
            <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1">
                <ZoomIn className="w-3 h-3" />
                {isRTL ? "تكبير: عجلة الماوس" : "Zoom: Mouse wheel"}
              </span>
              <span className="flex items-center gap-1">
                <Maximize2 className="w-3 h-3" />
                {isRTL ? "تحريك: سحب" : "Pan: Drag"}
              </span>
              <span>
                {isRTL ? "انقر على عقدة للتفاصيل" : "Click node for details"}
              </span>
            </div>
          </div>
        </div>

        {/* Edge Statistics */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          {ALL_EDGE_TYPES.map((type) => {
            const count = stats.edges_by_type?.[type] || 0;
            return (
              <div
                key={type}
                className={`bg-slate-800 rounded-lg p-4 border border-slate-700 ${
                  selectedEdgeTypes.includes(type) ? "ring-1 ring-emerald-500/50" : "opacity-50"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: EDGE_COLORS[type] }}
                  />
                  <span className="text-xs text-slate-400 truncate">
                    {type.replace("_", " ")}
                  </span>
                </div>
                <div className="text-xl font-bold text-white">
                  {isLoading ? "..." : count.toLocaleString()}
                </div>
              </div>
            );
          })}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <span>{isRTL ? "QBM الشبكة الدلالية" : "QBM Semantic Graph"}</span>
              <span className="px-2 py-0.5 bg-blue-900/50 text-blue-400 rounded text-xs">
                v2.0
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span>
                {stats.total_edges.toLocaleString()}{" "}
                {isRTL ? "رابط إجمالي" : "total edges"}
              </span>
              <span>•</span>
              <span>
                {Object.keys(stats.edges_by_type || {}).length || 7} {isRTL ? "أنواع علاقات" : "relationship types"}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
