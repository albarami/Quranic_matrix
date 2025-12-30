"use client";

import { useState, useEffect } from "react";
import { BookOpen, MapPin } from "lucide-react";
import { SURAH_DATA } from "./AyahNavigator";

interface AyahDisplayProps {
  surah: number;
  ayah: number;
  language: string;
}

// Revelation type mapping (simplified - Makki vs Madani)
const MAKKI_SURAHS = [
  1, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29,
  30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52,
  53, 54, 56, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83,
  84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103,
  104, 105, 106, 107, 108, 109, 111, 112, 113, 114
];

// Sample ayat for demo - in production this would come from API
const SAMPLE_AYAT: Record<string, string> = {
  "2:255": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ ۚ لَا تَأْخُذُهُۥ سِنَةٌ وَلَا نَوْمٌ ۚ لَّهُۥ مَا فِى ٱلسَّمَٰوَٰتِ وَمَا فِى ٱلْأَرْضِ ۗ مَن ذَا ٱلَّذِى يَشْفَعُ عِندَهُۥٓ إِلَّا بِإِذْنِهِۦ ۚ يَعْلَمُ مَا بَيْنَ أَيْدِيهِمْ وَمَا خَلْفَهُمْ ۖ وَلَا يُحِيطُونَ بِشَىْءٍ مِّنْ عِلْمِهِۦٓ إِلَّا بِمَا شَآءَ ۚ وَسِعَ كُرْسِيُّهُ ٱلسَّمَٰوَٰتِ وَٱلْأَرْضَ ۖ وَلَا يَـُٔودُهُۥ حِفْظُهُمَا ۚ وَهُوَ ٱلْعَلِىُّ ٱلْعَظِيمُ",
  "1:1": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
  "1:2": "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ",
  "1:3": "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
  "1:4": "مَٰلِكِ يَوْمِ ٱلدِّينِ",
  "1:5": "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
  "1:6": "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ",
  "1:7": "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
  "2:1": "الٓمٓ",
  "2:2": "ذَٰلِكَ ٱلْكِتَٰبُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى لِّلْمُتَّقِينَ",
  "2:3": "ٱلَّذِينَ يُؤْمِنُونَ بِٱلْغَيْبِ وَيُقِيمُونَ ٱلصَّلَوٰةَ وَمِمَّا رَزَقْنَٰهُمْ يُنفِقُونَ",
  "3:1": "الٓمٓ",
  "3:2": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ",
  "112:1": "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
  "112:2": "ٱللَّهُ ٱلصَّمَدُ",
  "112:3": "لَمْ يَلِدْ وَلَمْ يُولَدْ",
  "112:4": "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ",
};

export function AyahDisplay({ surah, ayah, language }: AyahDisplayProps) {
  const [ayahText, setAyahText] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const currentSurah = SURAH_DATA.find(s => s.number === surah);
  const isMakki = MAKKI_SURAHS.includes(surah);
  const isRTL = language === "ar";

  useEffect(() => {
    setLoading(true);
    // Check if we have sample data
    const key = `${surah}:${ayah}`;
    if (SAMPLE_AYAT[key]) {
      setAyahText(SAMPLE_AYAT[key]);
      setLoading(false);
    } else {
      // Simulate API call with placeholder
      setTimeout(() => {
        setAyahText(`[${isRTL ? "آية" : "Ayah"} ${surah}:${ayah} - ${isRTL ? "نص الآية سيُحمل من الواجهة الخلفية" : "Text will be loaded from backend"}]`);
        setLoading(false);
      }, 300);
    }
  }, [surah, ayah, isRTL]);

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="text-emerald-400 font-arabic text-lg">{currentSurah?.name}</span>
            <span className="text-slate-400 ml-2">({currentSurah?.nameEn})</span>
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
            {isRTL ? `آية ${ayah} من ${currentSurah?.ayat}` : `Ayah ${ayah} of ${currentSurah?.ayat}`}
          </span>
        </div>
      </div>

      {/* Arabic Text */}
      <div
        className={`text-3xl font-arabic leading-[2.5] text-white text-center py-8 px-6 bg-slate-900/50 rounded-lg border border-slate-700 ${loading ? 'animate-pulse' : ''}`}
        dir="rtl"
      >
        {loading ? (
          <div className="h-20 flex items-center justify-center">
            <span className="text-slate-500">{isRTL ? "جاري التحميل..." : "Loading..."}</span>
          </div>
        ) : (
          ayahText
        )}
      </div>

      {/* Reference */}
      <div className="text-center mt-4 flex items-center justify-center gap-4">
        <span className="text-emerald-400 font-mono text-lg">
          {surah}:{ayah}
        </span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-500 text-sm">
          {isRTL ? "سورة" : "Surah"} {currentSurah?.name} - {isRTL ? "آية" : "Ayah"} {ayah}
        </span>
      </div>
    </div>
  );
}
