// QBM Backend API Client
// Centralized API communication layer

import type {
  QueryResponse,
  StatsResponse,
  GraphResponse,
  TafsirResponse,
  AyahResponse,
  CanonicalEntitiesResponse,
  EntityResolutionResponse,
  AnnotationsResponse,
  SearchParams,
  SearchResponse,
  SimilarityResponse,
  SurahMetadata,
  MetricsResponse,
  RecentSpansResponse,
  DashboardStatsResponse,
} from "./types";

const QBM_BACKEND_URL =
  process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

export class QBMClient {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;

  constructor(baseUrl: string = QBM_BACKEND_URL) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      "Content-Type": "application/json",
    };
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new QBMApiError(
          errorData.detail || `API error: ${response.status}`,
          response.status,
          endpoint
        );
      }

      return response.json();
    } catch (error) {
      if (error instanceof QBMApiError) {
        throw error;
      }
      throw new QBMApiError(
        `Network error: ${(error as Error).message}`,
        0,
        endpoint
      );
    }
  }

  // =========================================================================
  // CORE QUERY
  // =========================================================================

  async query(question: string, includeProof: boolean = true): Promise<QueryResponse> {
    return this.fetch<QueryResponse>("/api/proof/query", {
      method: "POST",
      body: JSON.stringify({ question, include_proof: includeProof }),
    });
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  async getStats(): Promise<StatsResponse> {
    return this.fetch<StatsResponse>("/api/stats");
  }

  async getMetrics(): Promise<MetricsResponse> {
    return this.fetch<MetricsResponse>("/api/metrics/overview");
  }

  async getDashboardStats(): Promise<DashboardStatsResponse> {
    return this.fetch<DashboardStatsResponse>("/stats");
  }

  async getRecentSpans(limit: number = 5): Promise<RecentSpansResponse> {
    return this.fetch<RecentSpansResponse>(`/spans/recent?limit=${limit}`);
  }

  // =========================================================================
  // GRAPH
  // =========================================================================

  async getGraph(): Promise<GraphResponse> {
    return this.fetch<GraphResponse>("/api/graph");
  }

  async getGraphSubset(
    nodeIds: string[],
    depth: number = 1
  ): Promise<GraphResponse> {
    const params = new URLSearchParams({
      nodes: nodeIds.join(","),
      depth: depth.toString(),
    });
    return this.fetch<GraphResponse>(`/api/graph/subset?${params}`);
  }

  // =========================================================================
  // ENTITIES
  // =========================================================================

  async getCanonicalEntities(): Promise<CanonicalEntitiesResponse> {
    return this.fetch<CanonicalEntitiesResponse>("/api/entities");
  }

  async resolveEntity(arabicTerm: string): Promise<EntityResolutionResponse> {
    return this.fetch<EntityResolutionResponse>("/api/resolve", {
      method: "POST",
      body: JSON.stringify({ term: arabicTerm }),
    });
  }

  async resolveEntities(terms: string[]): Promise<EntityResolutionResponse[]> {
    return this.fetch<EntityResolutionResponse[]>("/api/resolve/batch", {
      method: "POST",
      body: JSON.stringify({ terms }),
    });
  }

  // =========================================================================
  // QURAN & TAFSIR
  // =========================================================================

  async getAyah(surah: number, ayah: number): Promise<AyahResponse> {
    return this.fetch<AyahResponse>(`/api/quran/${surah}/${ayah}`);
  }

  async getSurah(surah: number): Promise<AyahResponse[]> {
    return this.fetch<AyahResponse[]>(`/api/quran/${surah}`);
  }

  async getSurahs(): Promise<SurahMetadata[]> {
    return this.fetch<SurahMetadata[]>("/api/surahs");
  }

  async getTafsir(
    surah: number,
    ayah: number,
    source?: string
  ): Promise<TafsirResponse> {
    const endpoint = source
      ? `/api/tafsir/${surah}/${ayah}?source=${source}`
      : `/api/tafsir/${surah}/${ayah}`;
    return this.fetch<TafsirResponse>(endpoint);
  }

  async getTafsirComparison(surah: number, ayah: number): Promise<any> {
    return this.fetch<any>(`/tafsir/compare/${surah}/${ayah}`);
  }

  // =========================================================================
  // ANNOTATIONS
  // =========================================================================

  async getAnnotations(surah: number, ayah: number): Promise<AnnotationsResponse> {
    return this.fetch<AnnotationsResponse>(`/api/annotations/${surah}/${ayah}`);
  }

  async searchAnnotations(params: SearchParams): Promise<SearchResponse> {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    return this.fetch<SearchResponse>(`/api/annotations/search?${searchParams}`);
  }

  async createAnnotation(
    surah: number,
    ayah: number,
    data: Partial<{
      behavior_id: string;
      agent_id: string;
      organ_id: string;
      axes_11: Record<string, string>;
      notes: string;
    }>
  ): Promise<{ id: string }> {
    return this.fetch<{ id: string }>(`/api/annotations/${surah}/${ayah}`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // =========================================================================
  // EMBEDDINGS
  // =========================================================================

  async getSimilarBehaviors(
    behaviorId: string,
    topK: number = 10
  ): Promise<SimilarityResponse> {
    return this.fetch<SimilarityResponse>(
      `/api/embeddings/similar/${behaviorId}?k=${topK}`
    );
  }

  async computeSimilarity(
    behavior1: string,
    behavior2: string
  ): Promise<{ similarity: number }> {
    return this.fetch<{ similarity: number }>("/api/embeddings/similarity", {
      method: "POST",
      body: JSON.stringify({ behavior1, behavior2 }),
    });
  }

  // =========================================================================
  // HEALTH CHECK
  // =========================================================================

  async healthCheck(): Promise<{ status: string; version: string }> {
    return this.fetch<{ status: string; version: string }>("/health");
  }
}

// Custom error class for API errors
export class QBMApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public endpoint: string
  ) {
    super(message);
    this.name = "QBMApiError";
  }
}

// Singleton instance for convenient import
export const qbmClient = new QBMClient();

// Re-export types for convenience
export type { QueryResponse, StatsResponse, GraphResponse } from "./types";
