"use client";

interface MetricCardProps {
  title: string;
  titleAr?: string;
  value: number | string;
  subtitle?: string;
  format?: "number" | "percent" | "text";
  color?: "emerald" | "blue" | "amber" | "red" | "gray";
}

const colorClasses = {
  emerald: "bg-emerald-50 border-emerald-200 text-emerald-700",
  blue: "bg-blue-50 border-blue-200 text-blue-700",
  amber: "bg-amber-50 border-amber-200 text-amber-700",
  red: "bg-red-50 border-red-200 text-red-700",
  gray: "bg-gray-50 border-gray-200 text-gray-700",
};

export function MetricCard({
  title,
  titleAr,
  value,
  subtitle,
  format = "number",
  color = "emerald",
}: MetricCardProps) {
  const formatValue = () => {
    if (format === "percent" && typeof value === "number") {
      return `${value.toFixed(1)}%`;
    }
    if (format === "number" && typeof value === "number") {
      return value.toLocaleString();
    }
    return value;
  };

  return (
    <div className={`rounded-xl border-2 p-4 ${colorClasses[color]}`}>
      <div className="text-sm font-medium opacity-80">
        {titleAr && <span className="block text-right mb-1">{titleAr}</span>}
        {title}
      </div>
      <div className="text-3xl font-bold mt-2">{formatValue()}</div>
      {subtitle && (
        <div className="text-xs mt-1 opacity-70">{subtitle}</div>
      )}
    </div>
  );
}
