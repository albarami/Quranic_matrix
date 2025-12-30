"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { Loader2, AlertCircle, Database } from "lucide-react";
import { useMetrics } from "@/lib/api/hooks";

const COLORS = [
  "#059669", // emerald-600
  "#0891b2", // cyan-600
  "#7c3aed", // violet-600
  "#db2777", // pink-600
  "#ea580c", // orange-600
  "#ca8a04", // yellow-600
  "#4f46e5", // indigo-600
];

export function MetricsPanel() {
  const { data: metrics, isLoading: loading, error: queryError } = useMetrics();

  // Check if metrics are ready
  const error = queryError?.message ||
    (metrics && metrics.status !== "ready"
      ? "Metrics not ready. Run: python scripts/build_truth_metrics_v1.py"
      : null);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-8 h-8 animate-spin text-emerald-600" />
        <span className="ml-3 text-gray-600">جاري تحميل الإحصائيات...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium text-red-800">خطأ في تحميل الإحصائيات</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!metrics) return null;

  const agentData = metrics.metrics.agent_distribution.items.map((item, idx) => ({
    name: item.label_ar,
    value: item.count,
    percentage: item.percentage,
    key: item.key,
    fill: COLORS[idx % COLORS.length],
  }));

  const behaviorData = metrics.metrics.behavior_forms.items;
  const evaluationData = metrics.metrics.evaluations.items;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 space-y-6">
      {/* Header with source info */}
      <div className="flex items-center justify-between border-b border-gray-100 pb-4">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-emerald-600" />
          <h2 className="text-lg font-semibold text-gray-800">توزيع أنواع الفاعلين في القرآن الكريم</h2>
        </div>
        <div className="text-xs text-gray-500">
          المصدر: truth_metrics_v1 | {metrics.metrics.totals.spans.toLocaleString()} نطاق
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-emerald-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-emerald-700">
            {metrics.metrics.totals.spans.toLocaleString()}
          </div>
          <div className="text-sm text-emerald-600">إجمالي النطاقات</div>
        </div>
        <div className="bg-cyan-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-cyan-700">
            {metrics.metrics.totals.unique_verse_keys.toLocaleString()}
          </div>
          <div className="text-sm text-cyan-600">الآيات الفريدة</div>
        </div>
        <div className="bg-violet-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-violet-700">
            {metrics.metrics.totals.tafsir_sources_count}
          </div>
          <div className="text-sm text-violet-600">مصادر التفسير</div>
        </div>
      </div>

      {/* Agent Distribution - Pie Chart */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold text-gray-700 mb-3">توزيع الفاعلين (مخطط دائري)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={agentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ percentage }) => `${percentage}%`}
                >
                  {agentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [
                    `${value.toLocaleString()} (${agentData.find(d => d.name === name)?.percentage}%)`,
                    name
                  ]}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Agent Distribution - Table */}
        <div>
          <h3 className="font-semibold text-gray-700 mb-3">جدول التوزيع</h3>
          <div className="overflow-hidden rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-emerald-50">
                <tr>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-emerald-800">الفاعل</th>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-emerald-800">النوع</th>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-emerald-800">العدد</th>
                  <th className="px-4 py-2 text-right text-xs font-semibold text-emerald-800">النسبة</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {metrics.metrics.agent_distribution.items.map((item, idx) => (
                  <tr key={item.key} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                    <td className="px-4 py-2 text-sm text-gray-900">{item.label_ar}</td>
                    <td className="px-4 py-2 text-xs text-gray-500 font-mono">{item.key}</td>
                    <td className="px-4 py-2 text-sm text-gray-900 font-medium">{item.count.toLocaleString()}</td>
                    <td className="px-4 py-2 text-sm">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                        {item.percentage}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Behavior Forms Table */}
      <div>
        <h3 className="font-semibold text-gray-700 mb-3">أشكال السلوك</h3>
        <div className="overflow-hidden rounded-lg border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-cyan-50">
              <tr>
                <th className="px-4 py-2 text-right text-xs font-semibold text-cyan-800">الشكل</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-cyan-800">النوع</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-cyan-800">العدد</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-cyan-800">النسبة</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-100">
              {behaviorData.map((item, idx) => (
                <tr key={item.key} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                  <td className="px-4 py-2 text-sm text-gray-900">{item.label_ar}</td>
                  <td className="px-4 py-2 text-xs text-gray-500 font-mono">{item.key}</td>
                  <td className="px-4 py-2 text-sm text-gray-900 font-medium">{item.count.toLocaleString()}</td>
                  <td className="px-4 py-2 text-sm">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-cyan-100 text-cyan-800">
                      {item.percentage}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Evaluations Table */}
      <div>
        <h3 className="font-semibold text-gray-700 mb-3">التقييمات</h3>
        <div className="overflow-hidden rounded-lg border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-violet-50">
              <tr>
                <th className="px-4 py-2 text-right text-xs font-semibold text-violet-800">التقييم</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-violet-800">النوع</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-violet-800">العدد</th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-violet-800">النسبة</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-100">
              {evaluationData.map((item, idx) => (
                <tr key={item.key} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                  <td className="px-4 py-2 text-sm text-gray-900">{item.label_ar}</td>
                  <td className="px-4 py-2 text-xs text-gray-500 font-mono">{item.key}</td>
                  <td className="px-4 py-2 text-sm text-gray-900 font-medium">{item.count.toLocaleString()}</td>
                  <td className="px-4 py-2 text-sm">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-800">
                      {item.percentage}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer with metadata */}
      <div className="text-xs text-gray-400 text-center pt-4 border-t border-gray-100">
        البيانات من: {metrics.source_files[0]?.split(/[/\\]/).pop()} | 
        الإصدار: {metrics.build_version} | 
        التاريخ: {new Date(metrics.generated_at).toLocaleDateString("ar-SA")}
      </div>
    </div>
  );
}

// Utility function to detect metric-intent queries
export function isMetricIntentQuery(text: string): boolean {
  const patterns = [
    /توزيع.*الفاعلين/,
    /توزيع.*أنواع/,
    /مخطط.*دائري/,
    /إحصائيات/,
    /نسب.*الفاعلين/,
    /كم.*عدد/,
    /agent.*distribution/i,
    /pie.*chart/i,
    /statistics/i,
    /how.*many/i,
  ];
  return patterns.some(p => p.test(text));
}
