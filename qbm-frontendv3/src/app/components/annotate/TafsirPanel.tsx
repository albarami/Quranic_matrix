"use client";

import { BookOpen, ExternalLink, AlertCircle, RefreshCw, Loader2 } from "lucide-react";
import { useTafsir } from "@/lib/api/hooks";

const TAFSIR_SOURCES = [
  { id: "ibn_kathir", name: "Ibn Kathir", ar: "ابن كثير", color: "emerald" },
  { id: "tabari", name: "Tabari", ar: "الطبري", color: "blue" },
  { id: "qurtubi", name: "Qurtubi", ar: "القرطبي", color: "purple" },
  { id: "saadi", name: "Sa'di", ar: "السعدي", color: "amber" },
  { id: "jalalayn", name: "Jalalayn", ar: "الجلالين", color: "red" },
  { id: "baghawi", name: "Baghawi", ar: "البغوي", color: "cyan" },
  { id: "muyassar", name: "Muyassar", ar: "الميسر", color: "rose" }
];

interface TafsirPanelProps {
  surah: number;
  ayah: number;
  active: string;
  onTabChange: (id: string) => void;
  language: string;
}

export function TafsirPanel({ surah, ayah, active, onTabChange, language }: TafsirPanelProps) {
  const { data: tafsirData, isLoading, error, refetch } = useTafsir(surah, ayah);
  const isRTL = language === "ar";

  // Get the active tafsir text from API response
  const getActiveTafsirText = (): string => {
    if (!tafsirData?.sources) return "";
    const sourceData = tafsirData.sources[active as keyof typeof tafsirData.sources];
    return sourceData?.text_ar || "";
  };

  const tafsirText = getActiveTafsirText();
  const activeTafsir = TAFSIR_SOURCES.find(t => t.id === active);

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-900/50 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-blue-400" />
          <h2 className="font-semibold text-white">
            {isRTL ? "لوحة التفسير" : "Tafsir Panel"}
          </h2>
          <span className="text-slate-500 text-sm">
            (7 {isRTL ? "مصادر" : "sources"})
          </span>
        </div>
        <span className="text-slate-400 text-sm font-mono">{surah}:{ayah}</span>
      </div>

      {/* Tabs */}
      <div className="flex overflow-x-auto border-b border-slate-700 bg-slate-900/30">
        {TAFSIR_SOURCES.map((source) => (
          <button
            key={source.id}
            onClick={() => onTabChange(source.id)}
            className={`px-4 py-3 text-sm whitespace-nowrap transition-colors flex-shrink-0 ${
              active === source.id
                ? "bg-emerald-600 text-white"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <span className="font-arabic">{source.ar}</span>
            <span className="ml-2 text-xs opacity-70">({source.name})</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div
        className={`p-6 h-48 overflow-y-auto text-lg font-arabic leading-relaxed text-slate-200 ${isLoading ? 'animate-pulse' : ''}`}
        dir="rtl"
      >
        {error ? (
          <div className="h-full flex flex-col items-center justify-center gap-3">
            <AlertCircle className="w-8 h-8 text-red-400" />
            <span className="text-red-400">{isRTL ? "فشل تحميل التفسير" : "Failed to load tafsir"}</span>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-2 px-3 py-1.5 bg-red-900/30 hover:bg-red-900/50 rounded-lg text-sm text-red-400 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              {isRTL ? "إعادة المحاولة" : "Retry"}
            </button>
          </div>
        ) : isLoading ? (
          <div className="h-full flex items-center justify-center gap-2">
            <Loader2 className="w-5 h-5 text-slate-500 animate-spin" />
            <span className="text-slate-500">{isRTL ? "جاري التحميل..." : "Loading..."}</span>
          </div>
        ) : tafsirText ? (
          <p className="whitespace-pre-wrap">{tafsirText}</p>
        ) : (
          <div className="h-full flex items-center justify-center">
            <span className="text-slate-500">
              {isRTL
                ? `لا يوجد تفسير ${activeTafsir?.ar} متاح لهذه الآية`
                : `No ${activeTafsir?.name} tafsir available for this ayah`}
            </span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 bg-slate-900/50 border-t border-slate-700 flex items-center justify-between">
        <span className="text-xs text-slate-400">
          {isRTL ? "المصدر:" : "Source:"} {activeTafsir?.name} — {surah}:{ayah}
        </span>
        <button className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1">
          <ExternalLink className="w-3 h-3" />
          {isRTL ? "عرض المزيد" : "View Full"}
        </button>
      </div>
    </div>
  );
}

export { TAFSIR_SOURCES };
