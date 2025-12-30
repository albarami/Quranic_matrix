"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { EdgeType, NODE_COLORS, EDGE_COLORS, NodeType } from "@/lib/semantic-graph";

// Dynamic import to avoid SSR issues with canvas-based library
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-slate-900 rounded-xl">
      <div className="text-slate-400 animate-pulse">Loading graph...</div>
    </div>
  ),
});

// Flexible node type that handles both API format and legacy format
interface FlexibleGraphNode {
  id: string;
  label?: string;
  label_en?: string;
  labelAr?: string;
  label_ar?: string;
  type: NodeType;
  group?: number;
}

// Flexible edge type
interface FlexibleGraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
}

interface SemanticGraphProps {
  width?: number;
  height?: number;
  selectedNode?: string | null;
  onNodeClick?: (nodeId: string) => void;
  filterEdgeTypes?: EdgeType[];
  nodes: FlexibleGraphNode[];
  edges: FlexibleGraphEdge[];
  isLoading?: boolean;
}

export function SemanticGraph({
  width = 800,
  height = 600,
  selectedNode,
  onNodeClick,
  filterEdgeTypes,
  nodes,
  edges,
  isLoading = false,
}: SemanticGraphProps) {
  const graphRef = useRef<any>();
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [isClient, setIsClient] = useState(false);

  // Only render on client
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Filter edges by type if specified
  const filteredEdges = filterEdgeTypes
    ? edges.filter((e) => filterEdgeTypes.includes(e.type))
    : edges;

  // Get connected nodes from filtered edges
  const connectedNodeIds = new Set(
    filteredEdges.flatMap((e) => [e.source, e.target])
  );

  const filteredNodes = nodes.filter((n) => connectedNodeIds.has(n.id));

  // Normalize node data (handle both API and legacy formats)
  const normalizeNode = (n: FlexibleGraphNode) => ({
    id: n.id,
    label: n.label || n.label_en || n.id,
    labelAr: n.labelAr || n.label_ar || n.id,
    type: n.type,
    group: n.group ?? 0,
  });

  // Graph data format for react-force-graph
  const graphData = {
    nodes: filteredNodes.map((n) => {
      const normalized = normalizeNode(n);
      return {
        ...normalized,
        color:
          selectedNode === n.id
            ? "#10b981"
            : hoveredNode === n.id
            ? "#fbbf24"
            : NODE_COLORS[n.type] || "#64748b",
      };
    }),
    links: filteredEdges.map((e) => ({
      source: e.source,
      target: e.target,
      type: e.type,
      color: EDGE_COLORS[e.type] || "#64748b",
    })),
  };

  const handleNodeClick = useCallback(
    (node: any) => {
      if (onNodeClick) onNodeClick(node.id);
      // Center on node with animation
      if (graphRef.current) {
        graphRef.current.centerAt(node.x, node.y, 1000);
        graphRef.current.zoom(2, 1000);
      }
    },
    [onNodeClick]
  );

  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node?.id || null);
  }, []);

  // Show loading state
  if (!isClient || isLoading || !nodes.length) {
    return (
      <div
        className="bg-slate-900 rounded-xl border border-slate-700 flex items-center justify-center flex-col gap-3"
        style={{ width, height }}
      >
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        <div className="text-slate-400">
          {isLoading ? "Loading semantic graph..." : "Preparing visualization..."}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 overflow-hidden relative">
      <ForceGraph2D
        ref={graphRef}
        width={width}
        height={height}
        graphData={graphData}
        nodeLabel={(node: any) => `${node.labelAr}\n${node.label}`}
        nodeColor={(node: any) => node.color}
        nodeRelSize={8}
        linkColor={(link: any) => link.color}
        linkWidth={2}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={100}
        backgroundColor="#0f172a"
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          // Draw label below node
          const label = node.label;
          const fontSize = Math.max(10 / globalScale, 3);
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.fillStyle = "#94a3b8";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(label, node.x, node.y + 10 / globalScale);
        }}
      />

      {/* Graph info overlay */}
      <div className="absolute bottom-4 left-4 bg-slate-800/90 px-3 py-2 rounded-lg text-xs text-slate-400">
        {filteredNodes.length} nodes • {filteredEdges.length} edges
      </div>
    </div>
  );
}
