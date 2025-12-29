"use client";

import { useOnAction } from "@thesysai/genui-sdk";

interface ClusterTheme {
  behavior: string;
  count: number;
}

interface ClusterCardProps {
  clusterId: number;
  size: number;
  themes: ClusterTheme[];
  silhouetteScore?: number;
}

// Static color configs to avoid Tailwind purging
const CLUSTER_COLORS = [
  { border: "border-emerald-200", bg: "bg-emerald-100", text: "text-emerald-700", bar: "bg-emerald-500", btn: "text-emerald-600 hover:text-emerald-700 border-emerald-200 hover:bg-emerald-50" },
  { border: "border-blue-200", bg: "bg-blue-100", text: "text-blue-700", bar: "bg-blue-500", btn: "text-blue-600 hover:text-blue-700 border-blue-200 hover:bg-blue-50" },
  { border: "border-purple-200", bg: "bg-purple-100", text: "text-purple-700", bar: "bg-purple-500", btn: "text-purple-600 hover:text-purple-700 border-purple-200 hover:bg-purple-50" },
  { border: "border-pink-200", bg: "bg-pink-100", text: "text-pink-700", bar: "bg-pink-500", btn: "text-pink-600 hover:text-pink-700 border-pink-200 hover:bg-pink-50" },
  { border: "border-amber-200", bg: "bg-amber-100", text: "text-amber-700", bar: "bg-amber-500", btn: "text-amber-600 hover:text-amber-700 border-amber-200 hover:bg-amber-50" },
  { border: "border-teal-200", bg: "bg-teal-100", text: "text-teal-700", bar: "bg-teal-500", btn: "text-teal-600 hover:text-teal-700 border-teal-200 hover:bg-teal-50" },
  { border: "border-indigo-200", bg: "bg-indigo-100", text: "text-indigo-700", bar: "bg-indigo-500", btn: "text-indigo-600 hover:text-indigo-700 border-indigo-200 hover:bg-indigo-50" },
  { border: "border-rose-200", bg: "bg-rose-100", text: "text-rose-700", bar: "bg-rose-500", btn: "text-rose-600 hover:text-rose-700 border-rose-200 hover:bg-rose-50" },
  { border: "border-cyan-200", bg: "bg-cyan-100", text: "text-cyan-700", bar: "bg-cyan-500", btn: "text-cyan-600 hover:text-cyan-700 border-cyan-200 hover:bg-cyan-50" },
  { border: "border-orange-200", bg: "bg-orange-100", text: "text-orange-700", bar: "bg-orange-500", btn: "text-orange-600 hover:text-orange-700 border-orange-200 hover:bg-orange-50" },
  { border: "border-lime-200", bg: "bg-lime-100", text: "text-lime-700", bar: "bg-lime-500", btn: "text-lime-600 hover:text-lime-700 border-lime-200 hover:bg-lime-50" },
  { border: "border-violet-200", bg: "bg-violet-100", text: "text-violet-700", bar: "bg-violet-500", btn: "text-violet-600 hover:text-violet-700 border-violet-200 hover:bg-violet-50" },
  { border: "border-fuchsia-200", bg: "bg-fuchsia-100", text: "text-fuchsia-700", bar: "bg-fuchsia-500", btn: "text-fuchsia-600 hover:text-fuchsia-700 border-fuchsia-200 hover:bg-fuchsia-50" },
  { border: "border-sky-200", bg: "bg-sky-100", text: "text-sky-700", bar: "bg-sky-500", btn: "text-sky-600 hover:text-sky-700 border-sky-200 hover:bg-sky-50" },
  { border: "border-red-200", bg: "bg-red-100", text: "text-red-700", bar: "bg-red-500", btn: "text-red-600 hover:text-red-700 border-red-200 hover:bg-red-50" },
];

export function ClusterCard({
  clusterId,
  size,
  themes,
  silhouetteScore,
}: ClusterCardProps) {
  const onAction = useOnAction();

  const formatBehavior = (b: string) =>
    b.replace("BEH_", "").replace(/_/g, " ");

  const colors = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

  return (
    <div className={`bg-white rounded-lg border-2 ${colors.border} p-4 hover:shadow-lg transition-all`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-8 h-8 ${colors.bg} ${colors.text} rounded-full flex items-center justify-center font-bold`}>
            {clusterId + 1}
          </span>
          <div>
            <h4 className="font-semibold text-gray-900">
              المجموعة {clusterId + 1}
            </h4>
            <span className="text-xs text-gray-500">
              {size.toLocaleString()} تعليق
            </span>
          </div>
        </div>
        {silhouetteScore !== undefined && (
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
            Score: {silhouetteScore.toFixed(2)}
          </span>
        )}
      </div>

      {/* Top Themes */}
      <div className="space-y-2 mb-3">
        {themes.slice(0, 3).map((theme, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full ${colors.bar}`}
                style={{
                  width: `${Math.min(100, (theme.count / themes[0].count) * 100)}%`,
                }}
              />
            </div>
            <span className="text-xs text-gray-600 w-24 truncate">
              {formatBehavior(theme.behavior)}
            </span>
            <span className="text-xs text-gray-400 w-12 text-right">
              {theme.count}
            </span>
          </div>
        ))}
      </div>

      {/* Action */}
      <button
        onClick={() =>
          onAction(
            `استكشاف المجموعة ${clusterId + 1}`,
            `Show detailed samples from cluster ${clusterId}`
          )
        }
        className={`w-full py-2 text-sm font-medium border rounded-lg transition-colors ${colors.btn}`}
      >
        استكشاف المجموعة →
      </button>
    </div>
  );
}
