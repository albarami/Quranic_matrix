/**
 * Metrics Store - Fetches and caches metrics from /api/metrics/overview
 * 
 * This is the ONLY source of truth for metrics in the frontend.
 * The LLM is NOT allowed to be the source of any numbers.
 */

export interface MetricItem {
  key: string;
  label_ar?: string;
  count: number;
  percentage: number;
}

export interface Distribution {
  total: number;
  unique_values: number;
  items: MetricItem[];
  percentage_sum: number;
}

export interface MetricsTotals {
  spans: number;
  unique_verse_keys: number;
  tafsir_sources_count: number;
}

export interface TruthMetrics {
  schema_version: string;
  generated_at: string;
  build_version: string;
  source_files: string[];
  status: string;
  checksum: string;
  metrics: {
    totals: MetricsTotals;
    agent_distribution: Distribution;
    behavior_forms: Distribution;
    evaluations: Distribution;
    systemic_distribution?: Distribution;
    deontic_signals?: Distribution;
  };
}

export interface MetricsError {
  error: string;
  status?: string;
  how_to_fix?: string;
}

// Cache for metrics
let cachedMetrics: TruthMetrics | null = null;
let cacheTimestamp: number = 0;
const CACHE_TTL_MS = 60000; // 1 minute cache

/**
 * Fetch metrics from the backend /api/metrics/overview endpoint.
 * Returns cached data if available and fresh.
 */
export async function fetchMetrics(apiBaseUrl: string): Promise<TruthMetrics | MetricsError> {
  // Check cache
  if (cachedMetrics && Date.now() - cacheTimestamp < CACHE_TTL_MS) {
    return cachedMetrics;
  }

  try {
    const response = await fetch(`${apiBaseUrl}/api/metrics/overview`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (response.status === 503) {
      const errorData = await response.json();
      return errorData as MetricsError;
    }

    if (!response.ok) {
      return {
        error: `HTTP ${response.status}`,
        status: 'error',
        how_to_fix: 'Check if backend is running',
      };
    }

    const data = await response.json();
    
    // Validate it's real data
    if (data.status !== 'ready') {
      return {
        error: 'metrics_not_ready',
        status: data.status,
        how_to_fix: 'run: python scripts/build_truth_metrics_v1.py',
      };
    }

    // Cache it
    cachedMetrics = data as TruthMetrics;
    cacheTimestamp = Date.now();
    
    return cachedMetrics;
  } catch (error) {
    return {
      error: 'fetch_failed',
      status: 'error',
      how_to_fix: `Backend unreachable: ${error instanceof Error ? error.message : 'unknown'}`,
    };
  }
}

/**
 * Check if a response is an error
 */
export function isMetricsError(data: TruthMetrics | MetricsError): data is MetricsError {
  return 'error' in data;
}

/**
 * Clear the metrics cache (useful for testing)
 */
export function clearMetricsCache(): void {
  cachedMetrics = null;
  cacheTimestamp = 0;
}
