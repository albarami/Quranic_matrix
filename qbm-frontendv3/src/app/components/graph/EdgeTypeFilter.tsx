"use client";

import { EdgeType, EDGE_COLORS, EDGE_COUNTS, EDGE_LABELS } from "@/lib/semantic-graph";

interface EdgeTypeFilterProps {
  selected: EdgeType[];
  onChange: (types: EdgeType[]) => void;
  language?: string;
  counts?: Record<string, number>;  // Dynamic counts from API
}

const ALL_EDGE_TYPES: EdgeType[] = [
  "COMPLEMENTS",
  "CAUSES",
  "OPPOSITE_OF",
  "STRENGTHENS",
  "CONDITIONAL_ON",
  "PREVENTS",
  "LEADS_TO",
];

export function EdgeTypeFilter({ selected, onChange, language = "en", counts }: EdgeTypeFilterProps) {
  const isRTL = language === "ar";

  // Use API counts if provided, otherwise fall back to static counts
  const getCount = (type: EdgeType): number => {
    if (counts && counts[type] !== undefined) {
      return counts[type];
    }
    return EDGE_COUNTS[type];
  };

  const toggleType = (type: EdgeType) => {
    if (selected.includes(type)) {
      onChange(selected.filter((t) => t !== type));
    } else {
      onChange([...selected, type]);
    }
  };

  const selectAll = () => onChange(ALL_EDGE_TYPES);
  const clearAll = () => onChange([]);

  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
      <h3 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
        {isRTL ? "أنواع الروابط" : "Edge Types"}
      </h3>

      <div className="space-y-2">
        {ALL_EDGE_TYPES.map((type) => {
          const isActive = selected.includes(type);
          const label = EDGE_LABELS[type];
          const count = getCount(type);

          return (
            <label
              key={type}
              className={`flex items-center gap-3 cursor-pointer p-2 rounded transition-colors ${
                isActive ? "bg-slate-700/50" : "hover:bg-slate-700/30"
              }`}
            >
              <input
                type="checkbox"
                checked={isActive}
                onChange={() => toggleType(type)}
                className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              />
              <span
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: EDGE_COLORS[type] }}
              />
              <span className={`text-sm flex-1 ${isActive ? "text-white" : "text-slate-400"}`}>
                {isRTL ? label.ar : label.en}
              </span>
              <span className="text-xs text-slate-500 font-mono">
                {count.toLocaleString()}
              </span>
            </label>
          );
        })}
      </div>

      {/* Quick actions */}
      <div className="flex gap-3 mt-4 pt-4 border-t border-slate-700">
        <button
          onClick={selectAll}
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          {isRTL ? "تحديد الكل" : "Select All"}
        </button>
        <button
          onClick={clearAll}
          className="text-xs text-slate-400 hover:text-slate-300 transition-colors"
        >
          {isRTL ? "مسح الكل" : "Clear All"}
        </button>
      </div>

      {/* Stats */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <div className="text-xs text-slate-500">
          {isRTL ? "المحدد:" : "Selected:"}{" "}
          <span className="text-white font-medium">
            {selected.length}/{ALL_EDGE_TYPES.length}
          </span>{" "}
          {isRTL ? "أنواع" : "types"}
        </div>
        <div className="text-xs text-slate-500 mt-1">
          {isRTL ? "إجمالي الروابط:" : "Total edges:"}{" "}
          <span className="text-emerald-400 font-medium">
            {selected
              .reduce((sum, type) => sum + getCount(type), 0)
              .toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

export { ALL_EDGE_TYPES };
