// Sample Semantic Graph Data for QBM
// Representative sample demonstrating all 7 edge types
// Full graph has 4,460 edges - this sample shows ~50 key relationships

import { GraphNode, GraphEdge, EdgeType, EDGE_COLORS } from "./semantic-graph";

// Sample Nodes (39 total covering all entity types)
export const SAMPLE_NODES: GraphNode[] = [
  // Positive Behaviors (group 1)
  { id: "BEH_PATIENCE", label: "Patience", labelAr: "الصبر", type: "behavior", group: 1 },
  { id: "BEH_GRATITUDE", label: "Gratitude", labelAr: "الشكر", type: "behavior", group: 1 },
  { id: "BEH_PRAYER", label: "Prayer", labelAr: "الصلاة", type: "behavior", group: 1 },
  { id: "BEH_TRUTHFULNESS", label: "Truthfulness", labelAr: "الصدق", type: "behavior", group: 1 },
  { id: "BEH_TRUST", label: "Trust in Allah", labelAr: "التوكل", type: "behavior", group: 1 },
  { id: "BEH_CHARITY", label: "Charity", labelAr: "الصدقة", type: "behavior", group: 1 },
  { id: "BEH_FASTING", label: "Fasting", labelAr: "الصيام", type: "behavior", group: 1 },
  { id: "BEH_REPENTANCE", label: "Repentance", labelAr: "التوبة", type: "behavior", group: 1 },
  { id: "BEH_REMEMBRANCE", label: "Remembrance", labelAr: "الذكر", type: "behavior", group: 1 },
  { id: "BEH_HUMILITY", label: "Humility", labelAr: "التواضع", type: "behavior", group: 1 },
  { id: "BEH_SINCERITY", label: "Sincerity", labelAr: "الإخلاص", type: "behavior", group: 1 },
  { id: "BEH_RIGHTEOUSNESS", label: "Righteousness", labelAr: "الصلاح", type: "behavior", group: 1 },
  { id: "BEH_FEAR_ALLAH", label: "Fear of Allah", labelAr: "الخوف", type: "behavior", group: 1 },
  { id: "BEH_TAQWA", label: "God-Consciousness", labelAr: "التقوى", type: "behavior", group: 1 },
  { id: "BEH_JUSTICE", label: "Justice", labelAr: "العدل", type: "behavior", group: 1 },

  // Negative Behaviors (group 2)
  { id: "BEH_ARROGANCE", label: "Arrogance", labelAr: "الكبر", type: "behavior", group: 2 },
  { id: "BEH_ENVY", label: "Envy", labelAr: "الحسد", type: "behavior", group: 2 },
  { id: "BEH_LYING", label: "Lying", labelAr: "الكذب", type: "behavior", group: 2 },
  { id: "BEH_HYPOCRISY", label: "Hypocrisy", labelAr: "النفاق", type: "behavior", group: 2 },
  { id: "BEH_MISERLINESS", label: "Miserliness", labelAr: "البخل", type: "behavior", group: 2 },
  { id: "BEH_IMPATIENCE", label: "Impatience", labelAr: "العجلة", type: "behavior", group: 2 },
  { id: "BEH_INGRATITUDE", label: "Ingratitude", labelAr: "الكفران", type: "behavior", group: 2 },
  { id: "BEH_HEEDLESSNESS", label: "Heedlessness", labelAr: "الغفلة", type: "behavior", group: 2 },
  { id: "BEH_DESPAIR", label: "Despair", labelAr: "اليأس", type: "behavior", group: 2 },
  { id: "BEH_OPPRESSION", label: "Oppression", labelAr: "الظلم", type: "behavior", group: 2 },

  // Agents (group 3)
  { id: "AGT_BELIEVER", label: "Believer", labelAr: "المؤمن", type: "agent", group: 3 },
  { id: "AGT_DISBELIEVER", label: "Disbeliever", labelAr: "الكافر", type: "agent", group: 3 },
  { id: "AGT_HYPOCRITE", label: "Hypocrite", labelAr: "المنافق", type: "agent", group: 3 },
  { id: "AGT_PROPHET", label: "Prophet", labelAr: "النبي", type: "agent", group: 3 },
  { id: "AGT_RIGHTEOUS", label: "Righteous", labelAr: "الصالح", type: "agent", group: 3 },

  // Organs (group 4)
  { id: "ORG_HEART", label: "Heart", labelAr: "القلب", type: "organ", group: 4 },
  { id: "ORG_TONGUE", label: "Tongue", labelAr: "اللسان", type: "organ", group: 4 },
  { id: "ORG_EYE", label: "Eye", labelAr: "العين", type: "organ", group: 4 },
  { id: "ORG_HAND", label: "Hand", labelAr: "اليد", type: "organ", group: 4 },

  // Heart States (group 5)
  { id: "HRT_TRANQUILITY", label: "Tranquility", labelAr: "الطمأنينة", type: "heart_state", group: 5 },
  { id: "HRT_HARDNESS", label: "Hardness", labelAr: "القسوة", type: "heart_state", group: 5 },
  { id: "HRT_SOFTNESS", label: "Softness", labelAr: "اللين", type: "heart_state", group: 5 },
  { id: "HRT_FEAR", label: "Fear/Awe", labelAr: "الخشية", type: "heart_state", group: 5 },

  // Consequences (group 6)
  { id: "CON_JANNAH", label: "Paradise", labelAr: "الجنة", type: "consequence", group: 6 },
  { id: "CON_JAHANNAM", label: "Hellfire", labelAr: "جهنم", type: "consequence", group: 6 },
  { id: "CON_FORGIVENESS", label: "Forgiveness", labelAr: "المغفرة", type: "consequence", group: 6 },
  { id: "CON_GUIDANCE", label: "Guidance", labelAr: "الهداية", type: "consequence", group: 6 },
  { id: "CON_MERCY", label: "Mercy", labelAr: "الرحمة", type: "consequence", group: 6 },
  { id: "CON_PUNISHMENT", label: "Punishment", labelAr: "العذاب", type: "consequence", group: 6 },
];

// Sample Edges demonstrating all 7 relationship types
export const SAMPLE_EDGES: GraphEdge[] = [
  // COMPLEMENTS - behaviors that mutually reinforce (most common: 2,943 total)
  { source: "BEH_PATIENCE", target: "BEH_GRATITUDE", type: "COMPLEMENTS" },
  { source: "BEH_PRAYER", target: "BEH_REMEMBRANCE", type: "COMPLEMENTS" },
  { source: "BEH_CHARITY", target: "BEH_HUMILITY", type: "COMPLEMENTS" },
  { source: "BEH_FASTING", target: "BEH_PATIENCE", type: "COMPLEMENTS" },
  { source: "BEH_TRUST", target: "BEH_PATIENCE", type: "COMPLEMENTS" },
  { source: "BEH_REPENTANCE", target: "BEH_HUMILITY", type: "COMPLEMENTS" },
  { source: "BEH_TRUTHFULNESS", target: "BEH_TRUST", type: "COMPLEMENTS" },
  { source: "BEH_TAQWA", target: "BEH_PRAYER", type: "COMPLEMENTS" },
  { source: "BEH_SINCERITY", target: "BEH_PRAYER", type: "COMPLEMENTS" },
  { source: "BEH_JUSTICE", target: "BEH_TRUTHFULNESS", type: "COMPLEMENTS" },
  { source: "AGT_BELIEVER", target: "BEH_PATIENCE", type: "COMPLEMENTS" },
  { source: "AGT_BELIEVER", target: "BEH_PRAYER", type: "COMPLEMENTS" },
  { source: "AGT_BELIEVER", target: "BEH_TAQWA", type: "COMPLEMENTS" },
  { source: "AGT_PROPHET", target: "BEH_TRUTHFULNESS", type: "COMPLEMENTS" },
  { source: "AGT_RIGHTEOUS", target: "BEH_RIGHTEOUSNESS", type: "COMPLEMENTS" },
  { source: "ORG_HEART", target: "BEH_HUMILITY", type: "COMPLEMENTS" },
  { source: "ORG_TONGUE", target: "BEH_TRUTHFULNESS", type: "COMPLEMENTS" },
  { source: "ORG_TONGUE", target: "BEH_REMEMBRANCE", type: "COMPLEMENTS" },
  { source: "ORG_HAND", target: "BEH_CHARITY", type: "COMPLEMENTS" },

  // CAUSES - direct causal relationships (584 total)
  { source: "BEH_PATIENCE", target: "HRT_TRANQUILITY", type: "CAUSES" },
  { source: "BEH_REMEMBRANCE", target: "HRT_TRANQUILITY", type: "CAUSES" },
  { source: "BEH_ARROGANCE", target: "HRT_HARDNESS", type: "CAUSES" },
  { source: "BEH_PRAYER", target: "CON_FORGIVENESS", type: "CAUSES" },
  { source: "BEH_REPENTANCE", target: "CON_FORGIVENESS", type: "CAUSES" },
  { source: "BEH_HYPOCRISY", target: "CON_JAHANNAM", type: "CAUSES" },
  { source: "BEH_TAQWA", target: "CON_JANNAH", type: "CAUSES" },
  { source: "BEH_OPPRESSION", target: "CON_PUNISHMENT", type: "CAUSES" },
  { source: "BEH_CHARITY", target: "CON_MERCY", type: "CAUSES" },
  { source: "BEH_FEAR_ALLAH", target: "HRT_SOFTNESS", type: "CAUSES" },

  // OPPOSITE_OF - antithetical behaviors (367 total)
  { source: "BEH_PATIENCE", target: "BEH_IMPATIENCE", type: "OPPOSITE_OF" },
  { source: "BEH_GRATITUDE", target: "BEH_INGRATITUDE", type: "OPPOSITE_OF" },
  { source: "BEH_HUMILITY", target: "BEH_ARROGANCE", type: "OPPOSITE_OF" },
  { source: "BEH_TRUTHFULNESS", target: "BEH_LYING", type: "OPPOSITE_OF" },
  { source: "BEH_CHARITY", target: "BEH_MISERLINESS", type: "OPPOSITE_OF" },
  { source: "HRT_TRANQUILITY", target: "HRT_HARDNESS", type: "OPPOSITE_OF" },
  { source: "HRT_SOFTNESS", target: "HRT_HARDNESS", type: "OPPOSITE_OF" },
  { source: "CON_JANNAH", target: "CON_JAHANNAM", type: "OPPOSITE_OF" },
  { source: "AGT_BELIEVER", target: "AGT_DISBELIEVER", type: "OPPOSITE_OF" },
  { source: "BEH_JUSTICE", target: "BEH_OPPRESSION", type: "OPPOSITE_OF" },

  // STRENGTHENS - one enhances another (318 total)
  { source: "BEH_FASTING", target: "BEH_PATIENCE", type: "STRENGTHENS" },
  { source: "BEH_REMEMBRANCE", target: "BEH_FEAR_ALLAH", type: "STRENGTHENS" },
  { source: "BEH_CHARITY", target: "BEH_GRATITUDE", type: "STRENGTHENS" },
  { source: "BEH_PRAYER", target: "BEH_HUMILITY", type: "STRENGTHENS" },
  { source: "BEH_TAQWA", target: "BEH_SINCERITY", type: "STRENGTHENS" },
  { source: "BEH_REPENTANCE", target: "BEH_RIGHTEOUSNESS", type: "STRENGTHENS" },

  // CONDITIONAL_ON - dependency relationships (119 total)
  { source: "BEH_PRAYER", target: "BEH_SINCERITY", type: "CONDITIONAL_ON" },
  { source: "BEH_CHARITY", target: "BEH_SINCERITY", type: "CONDITIONAL_ON" },
  { source: "CON_FORGIVENESS", target: "BEH_REPENTANCE", type: "CONDITIONAL_ON" },
  { source: "CON_JANNAH", target: "BEH_TAQWA", type: "CONDITIONAL_ON" },
  { source: "CON_GUIDANCE", target: "BEH_SINCERITY", type: "CONDITIONAL_ON" },

  // PREVENTS - blocking relationships (106 total)
  { source: "BEH_REMEMBRANCE", target: "BEH_HEEDLESSNESS", type: "PREVENTS" },
  { source: "BEH_PATIENCE", target: "BEH_DESPAIR", type: "PREVENTS" },
  { source: "BEH_PRAYER", target: "BEH_OPPRESSION", type: "PREVENTS" },
  { source: "BEH_TAQWA", target: "BEH_ARROGANCE", type: "PREVENTS" },
  { source: "BEH_FEAR_ALLAH", target: "BEH_HYPOCRISY", type: "PREVENTS" },

  // LEADS_TO - sequential progression (23 total)
  { source: "BEH_REPENTANCE", target: "BEH_RIGHTEOUSNESS", type: "LEADS_TO" },
  { source: "BEH_PATIENCE", target: "CON_JANNAH", type: "LEADS_TO" },
  { source: "BEH_GRATITUDE", target: "CON_MERCY", type: "LEADS_TO" },
  { source: "BEH_TAQWA", target: "CON_GUIDANCE", type: "LEADS_TO" },
];

// Computed statistics for the sample
export const SAMPLE_STATS = {
  totalNodes: SAMPLE_NODES.length,
  totalEdges: SAMPLE_EDGES.length,
  nodesByType: {
    behavior: SAMPLE_NODES.filter(n => n.type === "behavior").length,
    agent: SAMPLE_NODES.filter(n => n.type === "agent").length,
    organ: SAMPLE_NODES.filter(n => n.type === "organ").length,
    heart_state: SAMPLE_NODES.filter(n => n.type === "heart_state").length,
    consequence: SAMPLE_NODES.filter(n => n.type === "consequence").length,
  },
  edgesByType: {
    COMPLEMENTS: SAMPLE_EDGES.filter(e => e.type === "COMPLEMENTS").length,
    CAUSES: SAMPLE_EDGES.filter(e => e.type === "CAUSES").length,
    OPPOSITE_OF: SAMPLE_EDGES.filter(e => e.type === "OPPOSITE_OF").length,
    STRENGTHENS: SAMPLE_EDGES.filter(e => e.type === "STRENGTHENS").length,
    CONDITIONAL_ON: SAMPLE_EDGES.filter(e => e.type === "CONDITIONAL_ON").length,
    PREVENTS: SAMPLE_EDGES.filter(e => e.type === "PREVENTS").length,
    LEADS_TO: SAMPLE_EDGES.filter(e => e.type === "LEADS_TO").length,
  }
};

// Helper function to get node by ID
export function getNodeById(id: string): GraphNode | undefined {
  return SAMPLE_NODES.find(n => n.id === id);
}

// Helper function to get edges connected to a node
export function getConnectedEdges(nodeId: string): GraphEdge[] {
  return SAMPLE_EDGES.filter(e => e.source === nodeId || e.target === nodeId);
}

// Helper function to filter edges by type
export function filterEdgesByType(types: EdgeType[]): GraphEdge[] {
  return SAMPLE_EDGES.filter(e => types.includes(e.type));
}

// Export all for easy access
export const ALL_NODES = SAMPLE_NODES;
export const ALL_EDGES = SAMPLE_EDGES;
