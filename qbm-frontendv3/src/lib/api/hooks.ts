// React Query Hooks for QBM Backend API
// Provides caching, error handling, and loading states

"use client";

import {
  useQuery,
  useMutation,
  useQueryClient,
  UseQueryOptions,
} from "@tanstack/react-query";
import { qbmClient } from "./qbm-client";
import type {
  StatsResponse,
  GraphResponse,
  CanonicalEntitiesResponse,
  TafsirResponse,
  AyahResponse,
  AnnotationsResponse,
  QueryResponse,
  EntityResolutionResponse,
  SimilarityResponse,
  SearchParams,
  SearchResponse,
  SurahMetadata,
} from "./types";

// ============================================================================
// QUERY KEYS
// Centralized query key management for cache invalidation
// ============================================================================

export const queryKeys = {
  stats: ["stats"] as const,
  graph: ["graph"] as const,
  graphSubset: (nodeIds: string[], depth: number) =>
    ["graph", "subset", nodeIds.join(","), depth] as const,
  entities: ["entities"] as const,
  ayah: (surah: number, ayah: number) => ["ayah", surah, ayah] as const,
  surah: (surah: number) => ["surah", surah] as const,
  surahs: ["surahs"] as const,
  tafsir: (surah: number, ayah: number, source?: string) =>
    ["tafsir", surah, ayah, source] as const,
  annotations: (surah: number, ayah: number) =>
    ["annotations", surah, ayah] as const,
  searchAnnotations: (params: SearchParams) =>
    ["annotations", "search", params] as const,
  similar: (behaviorId: string, topK: number) =>
    ["similar", behaviorId, topK] as const,
} as const;

// ============================================================================
// STATISTICS
// ============================================================================

export function useStats(
  options?: Omit<UseQueryOptions<StatsResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: () => qbmClient.getStats(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

// ============================================================================
// GRAPH
// ============================================================================

export function useGraph(
  options?: Omit<UseQueryOptions<GraphResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.graph,
    queryFn: () => qbmClient.getGraph(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

export function useGraphSubset(
  nodeIds: string[],
  depth: number = 1,
  options?: Omit<UseQueryOptions<GraphResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.graphSubset(nodeIds, depth),
    queryFn: () => qbmClient.getGraphSubset(nodeIds, depth),
    enabled: nodeIds.length > 0,
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

// ============================================================================
// ENTITIES
// ============================================================================

export function useCanonicalEntities(
  options?: Omit<
    UseQueryOptions<CanonicalEntitiesResponse>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: queryKeys.entities,
    queryFn: () => qbmClient.getCanonicalEntities(),
    staleTime: 30 * 60 * 1000, // 30 minutes (rarely changes)
    ...options,
  });
}

export function useEntityResolution() {
  return useMutation({
    mutationFn: (term: string) => qbmClient.resolveEntity(term),
  });
}

export function useBatchEntityResolution() {
  return useMutation({
    mutationFn: (terms: string[]) => qbmClient.resolveEntities(terms),
  });
}

// ============================================================================
// QURAN & TAFSIR
// ============================================================================

export function useAyah(
  surah: number,
  ayah: number,
  options?: Omit<UseQueryOptions<AyahResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.ayah(surah, ayah),
    queryFn: () => qbmClient.getAyah(surah, ayah),
    enabled: surah > 0 && ayah > 0,
    staleTime: Infinity, // Quran text never changes
    ...options,
  });
}

export function useSurah(
  surah: number,
  options?: Omit<UseQueryOptions<AyahResponse[]>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.surah(surah),
    queryFn: () => qbmClient.getSurah(surah),
    enabled: surah > 0,
    staleTime: Infinity,
    ...options,
  });
}

export function useSurahs(
  options?: Omit<UseQueryOptions<SurahMetadata[]>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.surahs,
    queryFn: () => qbmClient.getSurahs(),
    staleTime: Infinity,
    ...options,
  });
}

export function useTafsir(
  surah: number,
  ayah: number,
  source?: string,
  options?: Omit<UseQueryOptions<TafsirResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.tafsir(surah, ayah, source),
    queryFn: () => qbmClient.getTafsir(surah, ayah, source),
    enabled: surah > 0 && ayah > 0,
    staleTime: Infinity, // Tafsir text never changes
    ...options,
  });
}

// ============================================================================
// ANNOTATIONS
// ============================================================================

export function useAnnotations(
  surah: number,
  ayah: number,
  options?: Omit<UseQueryOptions<AnnotationsResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.annotations(surah, ayah),
    queryFn: () => qbmClient.getAnnotations(surah, ayah),
    enabled: surah > 0 && ayah > 0,
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options,
  });
}

export function useSearchAnnotations(
  params: SearchParams,
  options?: Omit<UseQueryOptions<SearchResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.searchAnnotations(params),
    queryFn: () => qbmClient.searchAnnotations(params),
    staleTime: 2 * 60 * 1000,
    ...options,
  });
}

export function useCreateAnnotation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      surah,
      ayah,
      data,
    }: {
      surah: number;
      ayah: number;
      data: Parameters<typeof qbmClient.createAnnotation>[2];
    }) => qbmClient.createAnnotation(surah, ayah, data),
    onSuccess: (_, { surah, ayah }) => {
      // Invalidate annotations cache for this ayah
      queryClient.invalidateQueries({
        queryKey: queryKeys.annotations(surah, ayah),
      });
    },
  });
}

// ============================================================================
// QUERY (Main Research Query)
// ============================================================================

export function useQBMQuery() {
  return useMutation({
    mutationFn: ({
      question,
      includeProof = true,
    }: {
      question: string;
      includeProof?: boolean;
    }) => qbmClient.query(question, includeProof),
  });
}

// ============================================================================
// EMBEDDINGS
// ============================================================================

export function useSimilarBehaviors(
  behaviorId: string,
  topK: number = 10,
  options?: Omit<UseQueryOptions<SimilarityResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.similar(behaviorId, topK),
    queryFn: () => qbmClient.getSimilarBehaviors(behaviorId, topK),
    enabled: !!behaviorId,
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

export function useComputeSimilarity() {
  return useMutation({
    mutationFn: ({
      behavior1,
      behavior2,
    }: {
      behavior1: string;
      behavior2: string;
    }) => qbmClient.computeSimilarity(behavior1, behavior2),
  });
}

// ============================================================================
// HEALTH CHECK
// ============================================================================

export function useHealthCheck(
  options?: Omit<
    UseQueryOptions<{ status: string; version: string }>,
    "queryKey" | "queryFn"
  >
) {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => qbmClient.healthCheck(),
    staleTime: 30 * 1000, // 30 seconds
    retry: 1,
    ...options,
  });
}

// ============================================================================
// PREFETCH UTILITIES
// ============================================================================

export function usePrefetchAyah() {
  const queryClient = useQueryClient();

  return (surah: number, ayah: number) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.ayah(surah, ayah),
      queryFn: () => qbmClient.getAyah(surah, ayah),
      staleTime: Infinity,
    });
  };
}

export function usePrefetchTafsir() {
  const queryClient = useQueryClient();

  return (surah: number, ayah: number) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.tafsir(surah, ayah),
      queryFn: () => qbmClient.getTafsir(surah, ayah),
      staleTime: Infinity,
    });
  };
}
