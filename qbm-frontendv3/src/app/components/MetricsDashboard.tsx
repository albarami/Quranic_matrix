"use client";

import { useEffect, useState } from "react";
import { fetchMetrics, isMetricsError, TruthMetrics, MetricsError } from "@/lib/metricsStore";
import { MetricCard } from "./MetricCard";
import { DistributionTable } from "./DistributionTable";
import { DistributionChart } from "./DistributionChart";
import { AlertCircle, RefreshCw, Database } from "lucide-react";

interface MetricsDashboardProps {
  apiBaseUrl: string;
}

export function MetricsDashboard({ apiBaseUrl }: MetricsDashboardProps) {
  const [metrics, setMetrics] = useState<TruthMetrics | null>(null);
  const [error, setError] = useState<MetricsError | null>(null);
  const [loading, setLoading] = useState(true);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    
    const result = await fetchMetrics(apiBaseUrl);
    
    if (isMetricsError(result)) {
      setError(result);
      setMetrics(null);
    } else {
      setMetrics(result);
      setError(null);
    }
    
    setLoading(false);
  };

  useEffect(() => {
    loadMetrics();
  }, [apiBaseUrl]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-emerald-500" />
        <span className="ml-2 text-gray-600">Loading metrics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-red-800">Data Unavailable</h3>
            <p className="text-red-700 mt-1">{error.error}</p>
            {error.how_to_fix && (
              <div className="mt-3 bg-red-100 rounded-lg p-3">
                <p className="text-sm font-mono text-red-800">{error.how_to_fix}</p>
              </div>
            )}
            <button
              onClick={loadMetrics}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return null;
  }

  const { totals, agent_distribution, behavior_forms, evaluations } = metrics.metrics;

  return (
    <div className="space-y-6">
      {/* Header with metadata */}
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
        <div className="flex items-center gap-2 text-emerald-700">
          <Database className="w-5 h-5" />
          <span className="font-semibold">QBM Truth Metrics</span>
          <span className="text-sm opacity-70">v{metrics.schema_version}</span>
        </div>
        <div className="mt-2 text-sm text-emerald-600 flex flex-wrap gap-4">
          <span>Generated: {new Date(metrics.generated_at).toLocaleString()}</span>
          <span>Build: {metrics.build_version}</span>
          <span>Checksum: {metrics.checksum.slice(0, 8)}...</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Total Spans"
          titleAr="إجمالي النطاقات"
          value={totals.spans}
          subtitle="Behavioral annotations"
          color="emerald"
        />
        <MetricCard
          title="Unique Verses"
          titleAr="الآيات الفريدة"
          value={totals.unique_verse_keys}
          subtitle="Covered verse keys"
          color="blue"
        />
        <MetricCard
          title="Tafsir Sources"
          titleAr="مصادر التفسير"
          value={totals.tafsir_sources_count}
          subtitle="Ibn Kathir, Tabari, etc."
          color="amber"
        />
      </div>

      {/* Agent Distribution - Chart and Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DistributionChart
          title="Agent Distribution"
          titleAr="توزيع أنواع الفاعلين"
          items={agent_distribution.items}
          type="donut"
        />
        <DistributionTable
          title="Agent Distribution"
          titleAr="توزيع أنواع الفاعلين"
          items={agent_distribution.items}
          showArabicLabels={true}
        />
      </div>

      {/* Behavior Forms and Evaluations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DistributionTable
          title="Behavior Forms"
          titleAr="أشكال السلوك"
          items={behavior_forms.items}
          showArabicLabels={true}
        />
        <DistributionTable
          title="Evaluations"
          titleAr="التقييمات"
          items={evaluations.items}
          showArabicLabels={true}
        />
      </div>

      {/* Data integrity notice */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-600">
        <strong>Data Integrity:</strong> All metrics are computed from canonical data at build time.
        The LLM is not the source of these numbers. Checksum verified.
      </div>
    </div>
  );
}
