// QBM Backend API Response Types
// These types define the contract between frontend and backend

// ============================================================================
// CORE QUERY TYPES
// ============================================================================

export interface QueryResponse {
  answer: string;
  question: string;
  question_class: string;           // e.g., "GRAPH_CAUSAL"
  planner_used: string;             // e.g., "CausalChainPlanner"
  trigger_patterns: string[];       // Words that triggered classification
  plan_steps: PlanStep[];           // Execution pipeline
  proof: {
    quran: QuranRef[];
    ibn_kathir: TafsirRef[];
    tabari: TafsirRef[];
    qurtubi: TafsirRef[];
    saadi: TafsirRef[];
    jalalayn: TafsirRef[];
    baghawi?: TafsirRef[];
    muyassar?: TafsirRef[];
    graph: GraphContext;
    embeddings: EmbeddingContext;
    rag_retrieval: RAGContext;
    taxonomy: TaxonomyClassification;
    statistics: ProofStatistics;
  };
  validation: {
    score: number;
    passed: boolean;
    missing: string[];
    checks: Record<string, boolean>;
  };
  processing_time_ms: number;
  resolved_entities: ResolvedEntity[];
  debug?: {
    intent?: string;
    retrieval_mode?: string;
    fallback_used?: boolean;
    resolved_entities?: ResolvedEntity[];
  };
}

export interface PlanStep {
  step_number: number;
  action: string;
  component: string;
  status: "completed" | "skipped" | "failed";
  duration_ms: number;
  result_summary?: string;
}

export interface ResolvedEntity {
  original_term: string;      // What user typed (Arabic)
  term?: string;              // Alias for original_term
  canonical_id: string;       // e.g., "AGT_BELIEVER"
  canonical: string;          // Alias for canonical_id
  canonical_ar: string;       // e.g., "المؤمن"
  canonical_en: string;       // e.g., "Believer"
  entity_type: string;        // e.g., "agent"
  type?: string;              // Alias for entity_type
  confidence: number;
  matched_via: "exact" | "synonym" | "root";
}

// ============================================================================
// QURAN & TAFSIR TYPES
// ============================================================================

export interface QuranRef {
  surah: string;
  ayah: string;
  text: string;
  relevance: number;
}

export interface TafsirRef {
  surah: string;
  ayah: string;
  text: string;
}

export interface AyahResponse {
  surah: number;
  ayah: number;
  text_ar: string;
  text_ar_clean: string;      // Without tashkeel
  surah_name_ar: string;
  surah_name_en: string;
  revelation: "makki" | "madani";
  juz: number;
  hizb: number;
  total_ayat_in_surah: number;
}

export interface TafsirResponse {
  surah: number;
  ayah: number;
  sources: {
    ibn_kathir?: TafsirText;
    tabari?: TafsirText;
    qurtubi?: TafsirText;
    saadi?: TafsirText;
    jalalayn?: TafsirText;
    baghawi?: TafsirText;
    muyassar?: TafsirText;
  };
}

export interface TafsirText {
  text_ar: string;
  source_name_ar: string;
  source_name_en: string;
  page_ref?: string;
}

export interface SurahMetadata {
  number: number;
  name_ar: string;
  name_en: string;
  name_transliteration: string;
  total_ayat: number;
  revelation: "makki" | "madani";
  revelation_order: number;
  juz_start: number;
}

// ============================================================================
// GRAPH TYPES
// ============================================================================

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  statistics: {
    total_nodes: number;
    total_edges: number;
    edges_by_type: Record<string, number>;
  };
}

export interface GraphNode {
  id: string;
  label: string;           // English label
  labelAr: string;         // Arabic label
  label_ar?: string;       // Alias
  label_en?: string;       // Alias
  type: NodeType;
  metadata?: Record<string, unknown>;
}

export type NodeType = "behavior" | "agent" | "organ" | "heart_state" | "consequence";

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
  evidence?: string[];
}

export type EdgeType =
  | "COMPLEMENTS"
  | "CAUSES"
  | "OPPOSITE_OF"
  | "STRENGTHENS"
  | "CONDITIONAL_ON"
  | "PREVENTS"
  | "LEADS_TO";

export interface GraphContext {
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: string[][];
}

// ============================================================================
// ENTITY TYPES
// ============================================================================

export interface CanonicalEntitiesResponse {
  behaviors: CanonicalEntity[];
  agents: CanonicalEntity[];
  organs: CanonicalEntity[];
  heart_states: CanonicalEntity[];
  consequences: CanonicalEntity[];
  total_entities: number;
  total_synonyms: number;
}

export interface CanonicalEntity {
  id: string;
  ar: string;
  en: string;
  roots: string[];
  synonyms: string[];
  occurrences: number;
}

export interface EntityResolutionResponse {
  term: string;
  resolved: boolean;
  canonical?: {
    id: string;
    ar: string;
    en: string;
    type: string;
  };
  matched_via?: "exact" | "synonym" | "root";
  confidence?: number;
  suggestions?: CanonicalEntity[];
}

// ============================================================================
// ANNOTATION TYPES
// ============================================================================

export interface AnnotationsResponse {
  surah: number;
  ayah: number;
  annotations: Annotation[];
}

export interface Annotation {
  id: string;
  behavior: EntityRef;
  agent?: EntityRef;
  organ?: EntityRef;
  heart_state?: EntityRef;
  consequence?: EntityRef;
  axes_11: Record<string, string>;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface EntityRef {
  id: string;
  ar: string;
  en: string;
}

export interface SearchParams {
  behavior?: string;
  agent?: string;
  organ?: string;
  surah?: number;
  revelation?: "makki" | "madani";
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  results: Annotation[];
  total: number;
  limit: number;
  offset: number;
}

// ============================================================================
// EMBEDDING TYPES
// ============================================================================

export interface EmbeddingContext {
  similarities: SimilarityPair[];
  clusters: ClusterInfo[];
  nearest_neighbors: NearestNeighbor[];
}

export interface SimilarityPair {
  concept1: string;
  concept2: string;
  score: number;
}

export interface ClusterInfo {
  cluster_id: number;
  behaviors: string[];
  centroid_label: string;
}

export interface NearestNeighbor {
  behavior_id: string;
  behavior_ar: string;
  behavior_en: string;
  distance: number;
}

export interface SimilarityResponse {
  query_behavior: string;
  similar: {
    behavior_id: string;
    behavior_ar: string;
    behavior_en: string;
    similarity_score: number;
  }[];
}

// ============================================================================
// RAG & TAXONOMY TYPES
// ============================================================================

export interface RAGContext {
  query: string;
  retrieved_docs: RetrievedDoc[];
  sources_breakdown: Record<string, number>;
}

export interface RetrievedDoc {
  text: string;
  source: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface TaxonomyClassification {
  behaviors: BehaviorClassification[];
  dimensions: Record<string, string>;
}

export interface BehaviorClassification {
  id?: string;
  name: string;
  name_en?: string;
  category?: string;
  evaluation?: string;
}

export interface ProofStatistics {
  counts: Record<string, number>;
  percentages: Record<string, number>;
}

// ============================================================================
// STATISTICS TYPES
// ============================================================================

export interface StatsResponse {
  total_ayat: number;
  annotated_ayat: number;
  total_annotations: number;
  behaviors_count: number;
  agents_count: number;
  organs_count: number;
  heart_states_count?: number;
  consequences_count?: number;
  graph_nodes: number;
  graph_edges: number;
  edges_by_type: Record<string, number>;
  tafsir_sources: number;
  synonyms_count: number;
  benchmark_score: string;  // "200/200"
}

// ============================================================================
// METRICS TYPES (Truth Layer)
// ============================================================================

export interface MetricsResponse {
  schema_version: string;
  generated_at: string;
  build_version: string;
  source_files: string[];
  status: "ready" | "pending" | "error";
  metrics: {
    totals: {
      spans: number;
      unique_verse_keys: number;
      tafsir_sources_count: number;
    };
    agent_distribution: MetricsDistribution;
    behavior_forms: MetricsDistribution;
    evaluations: MetricsDistribution;
  };
}

export interface MetricsDistribution {
  total: number;
  unique_values: number;
  items: MetricsDistributionItem[];
  percentage_sum: number;
}

export interface MetricsDistributionItem {
  key: string;
  label_ar: string;
  count: number;
  percentage: number;
}

// ============================================================================
// RECENT SPANS TYPES
// ============================================================================

export interface RecentSpansResponse {
  spans: SpanSummary[];
}

export interface SpanSummary {
  span_id?: string;
  reference?: {
    surah?: number;
    ayah?: number;
    surah_name?: string;
  };
  behavior_form?: string;
  agent?: {
    type?: string;
  };
  annotator?: string;
  annotated_at?: string;
}

// ============================================================================
// EXTENDED STATS TYPES (Dashboard format)
// ============================================================================

export interface DashboardStatsResponse {
  total_ayat?: number;
  unique_ayat?: number;
  total_spans?: number;
  unique_surahs?: number;
  coverage_pct?: number;
  dataset_tier?: string;
  behavior_forms?: Record<string, number>;
  agent_types?: Record<string, number>;
  evaluations?: Record<string, number>;
  top_surahs?: Array<{ surah: number; spans: number }>;
}

// ============================================================================
// BEHAVIOR PROFILE TYPES
// ============================================================================

export interface BehaviorListItem {
  name: string;
  count: number;
}

export interface BehaviorListResponse {
  behaviors: BehaviorListItem[];
}

export interface BehaviorProfileResponse {
  behavior: string;
  arabic_name: string;
  summary: {
    total_verses: number;
    total_spans: number;
    total_tafsir: number;
    total_surahs: number;
    coverage_percentage: number;
  };
  verses: BehaviorVerse[];
  tafsir: Record<string, BehaviorTafsirEntry[]>;
  graph: {
    related_behaviors: string[];
    verses: unknown[];
    connections: unknown[];
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

export interface BehaviorVerse {
  surah: number;
  surah_name: string;
  ayah: number;
  text: string;
  agent: string;
  agent_referent: string;
  evaluation: string;
  deontic: string;
  behavior_form: string;
}

export interface BehaviorTafsirEntry {
  surah: number;
  ayah: number;
  text: string;
}
