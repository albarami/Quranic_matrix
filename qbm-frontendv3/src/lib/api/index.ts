"use client";

// QBM API Module
// Re-exports all API-related functionality

export { qbmClient, QBMClient, QBMApiError } from "./qbm-client";

export {
  // Query hooks
  useStats,
  useGraph,
  useGraphSubset,
  useCanonicalEntities,
  useEntityResolution,
  useBatchEntityResolution,
  useAyah,
  useSurah,
  useSurahs,
  useTafsir,
  useAnnotations,
  useSearchAnnotations,
  useCreateAnnotation,
  useQBMQuery,
  useSimilarBehaviors,
  useComputeSimilarity,
  useHealthCheck,
  // Prefetch utilities
  usePrefetchAyah,
  usePrefetchTafsir,
  // Query keys for manual cache operations
  queryKeys,
} from "./hooks";

export type {
  // Core types
  QueryResponse,
  PlanStep,
  ResolvedEntity,
  // Quran & Tafsir
  QuranRef,
  TafsirRef,
  AyahResponse,
  TafsirResponse,
  TafsirText,
  SurahMetadata,
  // Graph
  GraphResponse,
  GraphNode,
  GraphEdge,
  NodeType,
  EdgeType,
  GraphContext,
  // Entities
  CanonicalEntitiesResponse,
  CanonicalEntity,
  EntityResolutionResponse,
  // Annotations
  AnnotationsResponse,
  Annotation,
  EntityRef,
  SearchParams,
  SearchResponse,
  // Embeddings
  EmbeddingContext,
  SimilarityPair,
  ClusterInfo,
  NearestNeighbor,
  SimilarityResponse,
  // RAG & Taxonomy
  RAGContext,
  RetrievedDoc,
  TaxonomyClassification,
  BehaviorClassification,
  ProofStatistics,
  // Statistics
  StatsResponse,
} from "./types";
