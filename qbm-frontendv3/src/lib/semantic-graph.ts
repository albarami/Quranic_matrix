// Semantic Graph Types and Constants for QBM
// Based on the actual backend semantic_graph_v2.json with 4,460 edges

export interface GraphNode {
  id: string;
  label: string;
  labelAr: string;
  type: NodeType;
  group: number;  // For coloring by type
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
}

export type NodeType = "behavior" | "agent" | "organ" | "heart_state" | "consequence";

export type EdgeType =
  | "COMPLEMENTS"
  | "CAUSES"
  | "OPPOSITE_OF"
  | "STRENGTHENS"
  | "CONDITIONAL_ON"
  | "PREVENTS"
  | "LEADS_TO";

// Edge colors by relationship type
export const EDGE_COLORS: Record<EdgeType, string> = {
  COMPLEMENTS: "#3b82f6",     // Blue
  CAUSES: "#ef4444",          // Red
  OPPOSITE_OF: "#f97316",     // Orange
  STRENGTHENS: "#22c55e",     // Green
  CONDITIONAL_ON: "#a855f7",  // Purple
  PREVENTS: "#6b7280",        // Gray
  LEADS_TO: "#06b6d4"         // Cyan
};

// Actual edge counts from the backend (4,460 total)
export const EDGE_COUNTS: Record<EdgeType, number> = {
  COMPLEMENTS: 2943,
  CAUSES: 584,
  OPPOSITE_OF: 367,
  STRENGTHENS: 318,
  CONDITIONAL_ON: 119,
  PREVENTS: 106,
  LEADS_TO: 23
};

// Edge type labels for display
export const EDGE_LABELS: Record<EdgeType, { en: string; ar: string }> = {
  COMPLEMENTS: { en: "Complements", ar: "يُكمِّل" },
  CAUSES: { en: "Causes", ar: "يُسبِّب" },
  OPPOSITE_OF: { en: "Opposite Of", ar: "نقيض" },
  STRENGTHENS: { en: "Strengthens", ar: "يُعزِّز" },
  CONDITIONAL_ON: { en: "Conditional On", ar: "مشروط بـ" },
  PREVENTS: { en: "Prevents", ar: "يمنع" },
  LEADS_TO: { en: "Leads To", ar: "يؤدي إلى" }
};

// Node colors by type
export const NODE_COLORS: Record<NodeType, string> = {
  behavior: "#10b981",      // Emerald
  agent: "#3b82f6",         // Blue
  organ: "#a855f7",         // Purple
  heart_state: "#ec4899",   // Pink
  consequence: "#f59e0b"    // Amber
};

// Node type labels
export const NODE_TYPE_LABELS: Record<NodeType, { en: string; ar: string }> = {
  behavior: { en: "Behavior", ar: "سلوك" },
  agent: { en: "Agent", ar: "فاعل" },
  organ: { en: "Organ", ar: "عضو" },
  heart_state: { en: "Heart State", ar: "حالة القلب" },
  consequence: { en: "Consequence", ar: "عاقبة" }
};

// Total graph statistics
export const GRAPH_TOTALS = {
  totalEdges: 4460,
  totalNodes: 155,  // From canonical_entities.json
  edgeTypes: 7,
  nodeTypes: 5
};
