"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { MetricItem } from "@/lib/metricsStore";

interface DistributionChartProps {
  title: string;
  titleAr?: string;
  items: MetricItem[];
  type?: "pie" | "donut";
}

const COLORS = [
  "#10b981", // emerald-500
  "#ef4444", // red-500
  "#3b82f6", // blue-500
  "#6b7280", // gray-500
  "#f59e0b", // amber-500
  "#8b5cf6", // violet-500
  "#ec4899", // pink-500
];

export function DistributionChart({
  title,
  titleAr,
  items,
  type = "donut",
}: DistributionChartProps) {
  const data = items.map((item, idx) => ({
    name: item.label_ar || item.key,
    nameEn: item.key,
    value: item.count,
    percentage: item.percentage,
    fill: COLORS[idx % COLORS.length],
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-800">{item.name}</p>
          <p className="text-sm text-gray-600">{item.nameEn}</p>
          <p className="text-emerald-600 font-bold">
            {item.value.toLocaleString()} ({item.percentage.toFixed(2)}%)
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="mb-4">
        <h3 className="font-semibold text-gray-800">
          {titleAr && <span className="block text-right text-emerald-700 mb-1">{titleAr}</span>}
          {title}
        </h3>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={type === "donut" ? 50 : 0}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
              labelLine={false}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 flex flex-wrap gap-2 justify-center">
        {data.slice(0, 5).map((item, idx) => (
          <div key={idx} className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: item.fill }}
            />
            <span>{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
