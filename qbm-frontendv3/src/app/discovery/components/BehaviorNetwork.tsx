"use client";

import { useOnAction } from "@thesysai/genui-sdk";

interface BehaviorPair {
  behavior_1: string;
  behavior_2: string;
  cooccurrence_count: number;
}

interface BehaviorNetworkProps {
  pairs: BehaviorPair[];
  title?: string;
}

export function BehaviorNetwork({ pairs, title }: BehaviorNetworkProps) {
  const onAction = useOnAction();

  // Format behavior name for display
  const formatBehavior = (b: string) =>
    b.replace("BEH_", "").replace(/_/g, " ").toLowerCase();

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span>🔗</span> {title}
        </h3>
      )}

      <div className="space-y-2">
        {pairs.slice(0, 10).map((pair, i) => (
          <div
            key={i}
            className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 transition-colors"
          >
            {/* Rank */}
            <span className="w-6 h-6 bg-emerald-100 text-emerald-700 rounded-full flex items-center justify-center text-xs font-bold">
              {i + 1}
            </span>

            {/* Behavior 1 */}
            <button
              onClick={() =>
                onAction(
                  `تحليل ${formatBehavior(pair.behavior_1)}`,
                  `Analyze behavior: ${pair.behavior_1}`
                )
              }
              className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-sm hover:bg-blue-100 transition-colors"
            >
              {formatBehavior(pair.behavior_1)}
            </button>

            {/* Connection */}
            <div className="flex-1 flex items-center">
              <div className="flex-1 h-0.5 bg-gradient-to-r from-blue-200 to-purple-200" />
              <span className="px-2 text-xs text-gray-500 bg-white">
                {pair.cooccurrence_count}×
              </span>
              <div className="flex-1 h-0.5 bg-gradient-to-r from-purple-200 to-pink-200" />
            </div>

            {/* Behavior 2 */}
            <button
              onClick={() =>
                onAction(
                  `تحليل ${formatBehavior(pair.behavior_2)}`,
                  `Analyze behavior: ${pair.behavior_2}`
                )
              }
              className="bg-purple-50 text-purple-700 px-2 py-1 rounded text-sm hover:bg-purple-100 transition-colors"
            >
              {formatBehavior(pair.behavior_2)}
            </button>
          </div>
        ))}
      </div>

      {pairs.length > 10 && (
        <button
          onClick={() =>
            onAction("عرض المزيد من الأنماط", "Show more co-occurrence patterns")
          }
          className="mt-4 w-full py-2 text-sm text-emerald-600 hover:text-emerald-700 font-medium border border-emerald-200 rounded-lg hover:bg-emerald-50 transition-colors"
        >
          عرض {pairs.length - 10} نمط إضافي →
        </button>
      )}
    </div>
  );
}
