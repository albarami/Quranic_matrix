"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Network, Info, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import { SemanticGraph, EdgeTypeFilter, ALL_EDGE_TYPES } from "../components/graph";
import {
  EdgeType,
  EDGE_COLORS,
  EDGE_COUNTS,
  NODE_COLORS,
  NODE_TYPE_LABELS,
  GRAPH_TOTALS,
} from "@/lib/semantic-graph";
import {
  SAMPLE_NODES,
  SAMPLE_EDGES,
  SAMPLE_STATS,
  getNodeById,
  getConnectedEdges,
} from "@/lib/semantic-graph-data";

export default function GraphPage() {
  const { language, isRTL } = useLanguage();
  const [selectedEdgeTypes, setSelectedEdgeTypes] = useState<EdgeType[]>(ALL_EDGE_TYPES);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const selectedNodeData = selectedNode ? getNodeById(selectedNode) : null;
  const connectedEdges = selectedNode ? getConnectedEdges(selectedNode) : [];

  // Filter connected edges by selected types
  const filteredConnectedEdges = connectedEdges.filter((e) =>
    selectedEdgeTypes.includes(e.type)
  );

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
                    ? "تصور 4,460 علاقة سلوكية عبر 7 أنواع روابط"
                    : "Visualizing 4,460 behavioral relationships across 7 edge types"}
                </p>
              </div>
            </div>

            {/* Stats */}
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400">
                  {GRAPH_TOTALS.totalNodes}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "عقد" : "Nodes"}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {GRAPH_TOTALS.totalEdges.toLocaleString()}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "روابط" : "Edges"}
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {GRAPH_TOTALS.edgeTypes}
                </div>
                <div className="text-xs text-slate-400">
                  {isRTL ? "أنواع" : "Types"}
                </div>
              </div>
            </div>
          </div>

          {/* Sample indicator */}
          <div className="mt-4 flex items-center gap-2 text-xs text-amber-400 bg-amber-900/30 px-3 py-2 rounded-lg border border-amber-700/50 w-fit">
            <Info className="w-4 h-4" />
            {isRTL
              ? `عرض عينة تمثيلية: ${SAMPLE_STATS.totalNodes} عقدة، ${SAMPLE_STATS.totalEdges} رابط`
              : `Showing representative sample: ${SAMPLE_STATS.totalNodes} nodes, ${SAMPLE_STATS.totalEdges} edges`}
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
                    style={{ backgroundColor: NODE_COLORS[selectedNodeData.type] }}
                  />
                  {isRTL ? "العقدة المحددة" : "Selected Node"}
                </h3>

                <div className="space-y-3">
                  <div>
                    <div className="text-2xl font-arabic text-white" dir="rtl">
                      {selectedNodeData.labelAr}
                    </div>
                    <div className="text-slate-300">{selectedNodeData.label}</div>
                    <div className="text-xs text-slate-500 font-mono mt-1">
                      {selectedNodeData.id}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className="px-2 py-1 rounded text-xs"
                      style={{
                        backgroundColor: NODE_COLORS[selectedNodeData.type] + "30",
                        color: NODE_COLORS[selectedNodeData.type],
                      }}
                    >
                      {isRTL
                        ? NODE_TYPE_LABELS[selectedNodeData.type].ar
                        : NODE_TYPE_LABELS[selectedNodeData.type].en}
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
                              style={{ backgroundColor: EDGE_COLORS[edge.type] }}
                            />
                            <span className="text-slate-400 truncate flex-1">
                              {edge.type.replace("_", " ")}
                            </span>
                            <span className="text-slate-300 truncate">
                              {otherNode?.label}
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
                      {SAMPLE_STATS.nodesByType[type]}
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
                nodes={SAMPLE_NODES}
                edges={SAMPLE_EDGES}
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
          {(Object.entries(EDGE_COUNTS) as [EdgeType, number][]).map(
            ([type, count]) => (
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
                  {count.toLocaleString()}
                </div>
              </div>
            )
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <span>{isRTL ? "QBM الشبكة الدلالية" : "QBM Semantic Graph"}</span>
              <span className="px-2 py-0.5 bg-blue-900/50 text-blue-400 rounded text-xs">
                v1.0
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span>
                {GRAPH_TOTALS.totalEdges.toLocaleString()}{" "}
                {isRTL ? "رابط إجمالي" : "total edges"}
              </span>
              <span>•</span>
              <span>
                7 {isRTL ? "أنواع علاقات" : "relationship types"}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
