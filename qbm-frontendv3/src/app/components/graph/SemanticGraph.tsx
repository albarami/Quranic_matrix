"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { GraphNode, GraphEdge, EdgeType, NODE_COLORS, EDGE_COLORS } from "@/lib/semantic-graph";
import { SAMPLE_NODES, SAMPLE_EDGES } from "@/lib/semantic-graph-data";

// Dynamic import to avoid SSR issues with canvas-based library
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-slate-900 rounded-xl">
      <div className="text-slate-400 animate-pulse">Loading graph...</div>
    </div>
  ),
});

interface SemanticGraphProps {
  width?: number;
  height?: number;
  selectedNode?: string | null;
  onNodeClick?: (nodeId: string) => void;
  filterEdgeTypes?: EdgeType[];
  nodes?: GraphNode[];
  edges?: GraphEdge[];
}

export function SemanticGraph({
  width = 800,
  height = 600,
  selectedNode,
  onNodeClick,
  filterEdgeTypes,
  nodes = SAMPLE_NODES,
  edges = SAMPLE_EDGES,
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

  // Graph data format for react-force-graph
  const graphData = {
    nodes: filteredNodes.map((n) => ({
      id: n.id,
      label: n.label,
      labelAr: n.labelAr,
      type: n.type,
      group: n.group,
      color:
        selectedNode === n.id
          ? "#10b981"
          : hoveredNode === n.id
          ? "#fbbf24"
          : NODE_COLORS[n.type],
    })),
    links: filteredEdges.map((e) => ({
      source: e.source,
      target: e.target,
      type: e.type,
      color: EDGE_COLORS[e.type],
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

  if (!isClient) {
    return (
      <div
        className="bg-slate-900 rounded-xl border border-slate-700 flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="text-slate-400 animate-pulse">Loading graph...</div>
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
