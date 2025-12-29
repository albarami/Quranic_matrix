"use client";

import { useOnAction } from "@thesysai/genui-sdk";

interface AyahCardProps {
  surah: number;
  ayah: number;
  surahName?: string;
  textAr: string;
  behavior?: string;
  score?: number;
  source?: string;
}

export function AyahCard({
  surah,
  ayah,
  surahName,
  textAr,
  behavior,
  score,
  source,
}: AyahCardProps) {
  const onAction = useOnAction();

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded text-sm font-medium">
            {surahName || `سورة ${surah}`}
          </span>
          <span className="text-gray-500 text-sm">
            {surah}:{ayah}
          </span>
        </div>
        {score !== undefined && (
          <span className="text-xs text-gray-400">
            {(score * 100).toFixed(1)}% match
          </span>
        )}
      </div>

      {/* Arabic Text */}
      <p
        className="text-xl leading-loose text-gray-900 font-arabic mb-3"
        dir="rtl"
        style={{ fontFamily: "'Amiri', 'Traditional Arabic', serif" }}
      >
        {textAr}
      </p>

      {/* Metadata */}
      <div className="flex items-center gap-2 flex-wrap">
        {behavior && (
          <span className="bg-blue-50 text-blue-700 px-2 py-0.5 rounded text-xs">
            {behavior.replace("BEH_", "").replace(/_/g, " ")}
          </span>
        )}
        {source && (
          <span className="bg-purple-50 text-purple-700 px-2 py-0.5 rounded text-xs">
            {source}
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2 mt-3 pt-3 border-t border-gray-100">
        <button
          onClick={() =>
            onAction(
              "عرض التفسير",
              `Show tafsir for Surah ${surah}, Ayah ${ayah}`
            )
          }
          className="text-sm text-emerald-600 hover:text-emerald-700 font-medium"
        >
          📖 التفسير
        </button>
        <button
          onClick={() =>
            onAction(
              "آيات مشابهة",
              `Find verses similar to Surah ${surah}, Ayah ${ayah}`
            )
          }
          className="text-sm text-blue-600 hover:text-blue-700 font-medium"
        >
          🔗 مشابهة
        </button>
        <button
          onClick={() =>
            onAction(
              "تحليل السلوك",
              `Analyze the behaviors in Surah ${surah}, Ayah ${ayah}`
            )
          }
          className="text-sm text-purple-600 hover:text-purple-700 font-medium"
        >
          🎯 تحليل
        </button>
      </div>
    </div>
  );
}
