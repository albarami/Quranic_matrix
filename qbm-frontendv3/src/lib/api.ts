'use client';

import { z } from 'zod';

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || 'http://localhost:8000';

// =============================================================================
// Zod Schemas for API Response Validation
// =============================================================================

export const ProofEvidenceSchema = z.object({
  surah: z.union([z.string(), z.number()]).optional(),
  ayah: z.union([z.string(), z.number()]).optional(),
  text: z.string().optional(),
  relevance: z.number().optional(),
  score: z.number().optional(),
});

export const GraphEvidenceSchema = z.object({
  nodes: z.array(z.any()).optional(),
  edges: z.array(z.any()).optional(),
  paths: z.array(z.any()).optional(),
});

export const DebugTraceSchema = z.object({
  query_type: z.string().optional(),
  fallback_used: z.boolean().optional(),
  retrieval_mode: z.string().optional(),
  graph_backend: z.string().optional(),
  processing_time_ms: z.number().optional(),
});

export const ProofResponseSchema = z.object({
  question: z.string(),
  answer: z.string(),
  proof: z.object({
    quran: z.array(ProofEvidenceSchema).optional(),
    ibn_kathir: z.array(ProofEvidenceSchema).optional(),
    tabari: z.array(ProofEvidenceSchema).optional(),
    qurtubi: z.array(ProofEvidenceSchema).optional(),
    saadi: z.array(ProofEvidenceSchema).optional(),
    jalalayn: z.array(ProofEvidenceSchema).optional(),
    graph: GraphEvidenceSchema.optional(),
    taxonomy: z.any().optional(),
    statistics: z.any().optional(),
  }).optional(),
  validation: z.object({
    score: z.number(),
    passed: z.boolean(),
    missing: z.array(z.string()).optional(),
    checks: z.record(z.boolean()).optional(),
  }).optional(),
  debug: DebugTraceSchema.optional(),
  processing_time_ms: z.number().optional(),
});

export const GenomeStatusSchema = z.object({
  status: z.string(),
  version: z.string(),
  source_versions: z.object({
    canonical_entities: z.string(),
    semantic_graph: z.string(),
  }),
  statistics: z.object({
    canonical_behaviors: z.number(),
    canonical_agents: z.number(),
    canonical_organs: z.number().optional(),
    canonical_heart_states: z.number(),
    canonical_consequences: z.number(),
    semantic_edges: z.number(),
  }),
  endpoints: z.record(z.string()),
});

export const GenomeExportSchema = z.object({
  version: z.string(),
  mode: z.string(),
  checksum: z.string(),
  source_versions: z.object({
    canonical_entities: z.string(),
    semantic_graph: z.string(),
  }),
  statistics: z.any(),
  behaviors: z.array(z.any()),
  agents: z.array(z.any()),
  organs: z.array(z.any()).optional(),
  heart_states: z.array(z.any()),
  consequences: z.array(z.any()),
  semantic_edges: z.array(z.any()).optional(),
  axes: z.any().optional(),
});

export const ReviewSchema = z.object({
  id: z.number(),
  span_id: z.string().nullable().optional(),
  edge_id: z.string().nullable().optional(),
  chunk_id: z.string().nullable().optional(),
  surah: z.number().nullable().optional(),
  ayah: z.number().nullable().optional(),
  verse_key: z.string().nullable().optional(),
  reviewer_id: z.string(),
  reviewer_name: z.string().nullable().optional(),
  status: z.string(),
  rating: z.number().nullable().optional(),
  comment: z.string().nullable().optional(),
  corrections: z.any().nullable().optional(),
  review_type: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  history: z.array(z.any()).optional(),
});

export const ReviewsListSchema = z.object({
  total: z.number(),
  limit: z.number(),
  offset: z.number(),
  reviews: z.array(ReviewSchema),
});

export const ReviewsStatusSchema = z.object({
  status: z.string(),
  backend: z.string(),
  statistics: z.object({
    total_reviews: z.number(),
    by_status: z.record(z.number()),
    by_type: z.record(z.number()),
  }),
  endpoints: z.record(z.string()),
});

// =============================================================================
// API Client Functions
// =============================================================================

export async function fetchProofQuery(question: string, options?: {
  mode?: 'summary' | 'full';
  per_ayah?: boolean;
  max_chunks_per_source?: number;
}) {
  const response = await fetch(`${BACKEND_URL}/api/proof/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      include_proof: true,
      ...options,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  const data = await response.json();
  return ProofResponseSchema.parse(data);
}

export async function fetchGenomeStatus() {
  const response = await fetch(`${BACKEND_URL}/api/genome/status`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const data = await response.json();
  return GenomeStatusSchema.parse(data);
}

export async function fetchGenomeExport(mode: 'full' | 'light' = 'light') {
  const response = await fetch(`${BACKEND_URL}/api/genome/export?mode=${mode}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const data = await response.json();
  return GenomeExportSchema.parse(data);
}

export async function fetchReviewsStatus() {
  const response = await fetch(`${BACKEND_URL}/api/reviews/status`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const data = await response.json();
  return ReviewsStatusSchema.parse(data);
}

export async function fetchReviews(filters?: {
  status?: string;
  review_type?: string;
  limit?: number;
  offset?: number;
}) {
  const params = new URLSearchParams();
  if (filters?.status) params.set('status', filters.status);
  if (filters?.review_type) params.set('review_type', filters.review_type);
  if (filters?.limit) params.set('limit', filters.limit.toString());
  if (filters?.offset) params.set('offset', filters.offset.toString());

  const response = await fetch(`${BACKEND_URL}/api/reviews?${params}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const data = await response.json();
  return ReviewsListSchema.parse(data);
}

export async function createReview(review: {
  span_id?: string;
  edge_id?: string;
  chunk_id?: string;
  surah?: number;
  ayah?: number;
  reviewer_id: string;
  reviewer_name?: string;
  rating?: number;
  comment?: string;
}) {
  const response = await fetch(`${BACKEND_URL}/api/reviews`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  const data = await response.json();
  return ReviewSchema.parse(data);
}

export async function updateReview(
  reviewId: number,
  updates: { status?: string; rating?: number; comment?: string },
  actorId: string
) {
  const response = await fetch(
    `${BACKEND_URL}/api/reviews/${reviewId}?actor_id=${encodeURIComponent(actorId)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  const data = await response.json();
  return ReviewSchema.parse(data);
}

export async function fetchReview(reviewId: number) {
  const response = await fetch(`${BACKEND_URL}/api/reviews/${reviewId}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  const data = await response.json();
  return ReviewSchema.parse(data);
}

// Export types
export type ProofResponse = z.infer<typeof ProofResponseSchema>;
export type GenomeStatus = z.infer<typeof GenomeStatusSchema>;
export type GenomeExport = z.infer<typeof GenomeExportSchema>;
export type Review = z.infer<typeof ReviewSchema>;
export type ReviewsList = z.infer<typeof ReviewsListSchema>;
export type ReviewsStatus = z.infer<typeof ReviewsStatusSchema>;
