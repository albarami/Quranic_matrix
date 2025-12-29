"use client";

import { MetricItem } from "@/lib/metricsStore";

interface DistributionTableProps {
  title: string;
  titleAr?: string;
  items: MetricItem[];
  showArabicLabels?: boolean;
}

export function DistributionTable({
  title,
  titleAr,
  items,
  showArabicLabels = true,
}: DistributionTableProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
        <h3 className="font-semibold text-gray-800">
          {titleAr && <span className="block text-right text-emerald-700 mb-1">{titleAr}</span>}
          {title}
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 text-sm text-gray-600">
              {showArabicLabels && <th className="px-4 py-2 text-right">العربية</th>}
              <th className="px-4 py-2 text-left">Key</th>
              <th className="px-4 py-2 text-right">Count</th>
              <th className="px-4 py-2 text-right">%</th>
              <th className="px-4 py-2 text-left w-32">Distribution</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, idx) => (
              <tr
                key={item.key}
                className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                {showArabicLabels && (
                  <td className="px-4 py-2 text-right font-arabic text-gray-700">
                    {item.label_ar || item.key}
                  </td>
                )}
                <td className="px-4 py-2 text-left font-mono text-sm text-gray-600">
                  {item.key}
                </td>
                <td className="px-4 py-2 text-right font-semibold text-gray-800">
                  {item.count.toLocaleString()}
                </td>
                <td className="px-4 py-2 text-right text-gray-600">
                  {item.percentage.toFixed(2)}%
                </td>
                <td className="px-4 py-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-emerald-500 h-2 rounded-full"
                      style={{ width: `${Math.min(item.percentage, 100)}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
