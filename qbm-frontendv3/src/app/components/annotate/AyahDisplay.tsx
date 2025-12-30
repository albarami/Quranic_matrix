"use client";

import { BookOpen, MapPin, AlertCircle, RefreshCw } from "lucide-react";
import { SURAH_DATA } from "./AyahNavigator";
import { useAyah } from "@/lib/api/hooks";

interface AyahDisplayProps {
  surah: number;
  ayah: number;
  language: string;
}

export function AyahDisplay({ surah, ayah, language }: AyahDisplayProps) {
  const { data: ayahData, isLoading, error, refetch } = useAyah(surah, ayah);

  const currentSurah = SURAH_DATA.find(s => s.number === surah);
  // Use revelation from API data
  const isMakki = ayahData?.revelation === "makki";
  const isRTL = language === "ar";

  // Error state
  if (error) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-red-700/50">
        <div className="flex items-center justify-center gap-4 text-red-400">
          <AlertCircle className="w-6 h-6" />
          <span>{isRTL ? "فشل تحميل الآية" : "Failed to load ayah"}</span>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-2 px-3 py-1.5 bg-red-900/30 hover:bg-red-900/50 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            {isRTL ? "إعادة المحاولة" : "Retry"}
          </button>
        </div>
      </div>
    );
  }

  // Use API data for surah name if available
  const surahNameAr = ayahData?.surah_name_ar || currentSurah?.name;
  const surahNameEn = ayahData?.surah_name_en || currentSurah?.nameEn;
  const totalAyat = ayahData?.total_ayat_in_surah || currentSurah?.ayat;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="text-emerald-400 font-arabic text-lg">{surahNameAr}</span>
            <span className="text-slate-400 ml-2">({surahNameEn})</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className={`px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-1.5 ${
            isMakki
              ? "bg-amber-900/50 text-amber-300 border border-amber-700"
              : "bg-blue-900/50 text-blue-300 border border-blue-700"
          }`}>
            <MapPin className="w-3 h-3" />
            {isMakki ? (isRTL ? "مكية" : "Makki") : (isRTL ? "مدنية" : "Madani")}
          </span>
          <span className="text-slate-400 text-sm">
            {isRTL ? `آية ${ayah} من ${totalAyat}` : `Ayah ${ayah} of ${totalAyat}`}
          </span>
        </div>
      </div>

      {/* Arabic Text */}
      <div
        className={`text-3xl font-arabic leading-[2.5] text-white text-center py-8 px-6 bg-slate-900/50 rounded-lg border border-slate-700 ${isLoading ? 'animate-pulse' : ''}`}
        dir="rtl"
      >
        {isLoading ? (
          <div className="h-20 flex items-center justify-center">
            <span className="text-slate-500">{isRTL ? "جاري التحميل..." : "Loading..."}</span>
          </div>
        ) : (
          ayahData?.text_ar || `[${surah}:${ayah}]`
        )}
      </div>

      {/* Reference */}
      <div className="text-center mt-4 flex items-center justify-center gap-4">
        <span className="text-emerald-400 font-mono text-lg">
          {surah}:{ayah}
        </span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-500 text-sm">
          {isRTL ? "سورة" : "Surah"} {surahNameAr} - {isRTL ? "آية" : "Ayah"} {ayah}
        </span>
        {ayahData?.juz && (
          <>
            <span className="text-slate-600">|</span>
            <span className="text-slate-500 text-sm">
              {isRTL ? "جزء" : "Juz"} {ayahData.juz}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
