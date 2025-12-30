"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

interface AyahNavigatorProps {
  surah: number;
  ayah: number;
  onNavigate: (surah: number, ayah: number) => void;
  language: string;
}

// Complete list of all 114 surahs with their ayah counts
const SURAH_DATA = [
  { number: 1, name: "الفاتحة", nameEn: "Al-Fatiha", ayat: 7 },
  { number: 2, name: "البقرة", nameEn: "Al-Baqarah", ayat: 286 },
  { number: 3, name: "آل عمران", nameEn: "Aal-Imran", ayat: 200 },
  { number: 4, name: "النساء", nameEn: "An-Nisa", ayat: 176 },
  { number: 5, name: "المائدة", nameEn: "Al-Ma'idah", ayat: 120 },
  { number: 6, name: "الأنعام", nameEn: "Al-An'am", ayat: 165 },
  { number: 7, name: "الأعراف", nameEn: "Al-A'raf", ayat: 206 },
  { number: 8, name: "الأنفال", nameEn: "Al-Anfal", ayat: 75 },
  { number: 9, name: "التوبة", nameEn: "At-Tawbah", ayat: 129 },
  { number: 10, name: "يونس", nameEn: "Yunus", ayat: 109 },
  { number: 11, name: "هود", nameEn: "Hud", ayat: 123 },
  { number: 12, name: "يوسف", nameEn: "Yusuf", ayat: 111 },
  { number: 13, name: "الرعد", nameEn: "Ar-Ra'd", ayat: 43 },
  { number: 14, name: "إبراهيم", nameEn: "Ibrahim", ayat: 52 },
  { number: 15, name: "الحجر", nameEn: "Al-Hijr", ayat: 99 },
  { number: 16, name: "النحل", nameEn: "An-Nahl", ayat: 128 },
  { number: 17, name: "الإسراء", nameEn: "Al-Isra", ayat: 111 },
  { number: 18, name: "الكهف", nameEn: "Al-Kahf", ayat: 110 },
  { number: 19, name: "مريم", nameEn: "Maryam", ayat: 98 },
  { number: 20, name: "طه", nameEn: "Ta-Ha", ayat: 135 },
  { number: 21, name: "الأنبياء", nameEn: "Al-Anbiya", ayat: 112 },
  { number: 22, name: "الحج", nameEn: "Al-Hajj", ayat: 78 },
  { number: 23, name: "المؤمنون", nameEn: "Al-Mu'minun", ayat: 118 },
  { number: 24, name: "النور", nameEn: "An-Nur", ayat: 64 },
  { number: 25, name: "الفرقان", nameEn: "Al-Furqan", ayat: 77 },
  { number: 26, name: "الشعراء", nameEn: "Ash-Shu'ara", ayat: 227 },
  { number: 27, name: "النمل", nameEn: "An-Naml", ayat: 93 },
  { number: 28, name: "القصص", nameEn: "Al-Qasas", ayat: 88 },
  { number: 29, name: "العنكبوت", nameEn: "Al-Ankabut", ayat: 69 },
  { number: 30, name: "الروم", nameEn: "Ar-Rum", ayat: 60 },
  { number: 31, name: "لقمان", nameEn: "Luqman", ayat: 34 },
  { number: 32, name: "السجدة", nameEn: "As-Sajdah", ayat: 30 },
  { number: 33, name: "الأحزاب", nameEn: "Al-Ahzab", ayat: 73 },
  { number: 34, name: "سبأ", nameEn: "Saba", ayat: 54 },
  { number: 35, name: "فاطر", nameEn: "Fatir", ayat: 45 },
  { number: 36, name: "يس", nameEn: "Ya-Sin", ayat: 83 },
  { number: 37, name: "الصافات", nameEn: "As-Saffat", ayat: 182 },
  { number: 38, name: "ص", nameEn: "Sad", ayat: 88 },
  { number: 39, name: "الزمر", nameEn: "Az-Zumar", ayat: 75 },
  { number: 40, name: "غافر", nameEn: "Ghafir", ayat: 85 },
  { number: 41, name: "فصلت", nameEn: "Fussilat", ayat: 54 },
  { number: 42, name: "الشورى", nameEn: "Ash-Shura", ayat: 53 },
  { number: 43, name: "الزخرف", nameEn: "Az-Zukhruf", ayat: 89 },
  { number: 44, name: "الدخان", nameEn: "Ad-Dukhan", ayat: 59 },
  { number: 45, name: "الجاثية", nameEn: "Al-Jathiyah", ayat: 37 },
  { number: 46, name: "الأحقاف", nameEn: "Al-Ahqaf", ayat: 35 },
  { number: 47, name: "محمد", nameEn: "Muhammad", ayat: 38 },
  { number: 48, name: "الفتح", nameEn: "Al-Fath", ayat: 29 },
  { number: 49, name: "الحجرات", nameEn: "Al-Hujurat", ayat: 18 },
  { number: 50, name: "ق", nameEn: "Qaf", ayat: 45 },
  { number: 51, name: "الذاريات", nameEn: "Adh-Dhariyat", ayat: 60 },
  { number: 52, name: "الطور", nameEn: "At-Tur", ayat: 49 },
  { number: 53, name: "النجم", nameEn: "An-Najm", ayat: 62 },
  { number: 54, name: "القمر", nameEn: "Al-Qamar", ayat: 55 },
  { number: 55, name: "الرحمن", nameEn: "Ar-Rahman", ayat: 78 },
  { number: 56, name: "الواقعة", nameEn: "Al-Waqi'ah", ayat: 96 },
  { number: 57, name: "الحديد", nameEn: "Al-Hadid", ayat: 29 },
  { number: 58, name: "المجادلة", nameEn: "Al-Mujadilah", ayat: 22 },
  { number: 59, name: "الحشر", nameEn: "Al-Hashr", ayat: 24 },
  { number: 60, name: "الممتحنة", nameEn: "Al-Mumtahanah", ayat: 13 },
  { number: 61, name: "الصف", nameEn: "As-Saff", ayat: 14 },
  { number: 62, name: "الجمعة", nameEn: "Al-Jumu'ah", ayat: 11 },
  { number: 63, name: "المنافقون", nameEn: "Al-Munafiqun", ayat: 11 },
  { number: 64, name: "التغابن", nameEn: "At-Taghabun", ayat: 18 },
  { number: 65, name: "الطلاق", nameEn: "At-Talaq", ayat: 12 },
  { number: 66, name: "التحريم", nameEn: "At-Tahrim", ayat: 12 },
  { number: 67, name: "الملك", nameEn: "Al-Mulk", ayat: 30 },
  { number: 68, name: "القلم", nameEn: "Al-Qalam", ayat: 52 },
  { number: 69, name: "الحاقة", nameEn: "Al-Haqqah", ayat: 52 },
  { number: 70, name: "المعارج", nameEn: "Al-Ma'arij", ayat: 44 },
  { number: 71, name: "نوح", nameEn: "Nuh", ayat: 28 },
  { number: 72, name: "الجن", nameEn: "Al-Jinn", ayat: 28 },
  { number: 73, name: "المزمل", nameEn: "Al-Muzzammil", ayat: 20 },
  { number: 74, name: "المدثر", nameEn: "Al-Muddaththir", ayat: 56 },
  { number: 75, name: "القيامة", nameEn: "Al-Qiyamah", ayat: 40 },
  { number: 76, name: "الإنسان", nameEn: "Al-Insan", ayat: 31 },
  { number: 77, name: "المرسلات", nameEn: "Al-Mursalat", ayat: 50 },
  { number: 78, name: "النبأ", nameEn: "An-Naba", ayat: 40 },
  { number: 79, name: "النازعات", nameEn: "An-Nazi'at", ayat: 46 },
  { number: 80, name: "عبس", nameEn: "Abasa", ayat: 42 },
  { number: 81, name: "التكوير", nameEn: "At-Takwir", ayat: 29 },
  { number: 82, name: "الانفطار", nameEn: "Al-Infitar", ayat: 19 },
  { number: 83, name: "المطففين", nameEn: "Al-Mutaffifin", ayat: 36 },
  { number: 84, name: "الانشقاق", nameEn: "Al-Inshiqaq", ayat: 25 },
  { number: 85, name: "البروج", nameEn: "Al-Buruj", ayat: 22 },
  { number: 86, name: "الطارق", nameEn: "At-Tariq", ayat: 17 },
  { number: 87, name: "الأعلى", nameEn: "Al-A'la", ayat: 19 },
  { number: 88, name: "الغاشية", nameEn: "Al-Ghashiyah", ayat: 26 },
  { number: 89, name: "الفجر", nameEn: "Al-Fajr", ayat: 30 },
  { number: 90, name: "البلد", nameEn: "Al-Balad", ayat: 20 },
  { number: 91, name: "الشمس", nameEn: "Ash-Shams", ayat: 15 },
  { number: 92, name: "الليل", nameEn: "Al-Layl", ayat: 21 },
  { number: 93, name: "الضحى", nameEn: "Ad-Duha", ayat: 11 },
  { number: 94, name: "الشرح", nameEn: "Ash-Sharh", ayat: 8 },
  { number: 95, name: "التين", nameEn: "At-Tin", ayat: 8 },
  { number: 96, name: "العلق", nameEn: "Al-Alaq", ayat: 19 },
  { number: 97, name: "القدر", nameEn: "Al-Qadr", ayat: 5 },
  { number: 98, name: "البينة", nameEn: "Al-Bayyinah", ayat: 8 },
  { number: 99, name: "الزلزلة", nameEn: "Az-Zalzalah", ayat: 8 },
  { number: 100, name: "العاديات", nameEn: "Al-Adiyat", ayat: 11 },
  { number: 101, name: "القارعة", nameEn: "Al-Qari'ah", ayat: 11 },
  { number: 102, name: "التكاثر", nameEn: "At-Takathur", ayat: 8 },
  { number: 103, name: "العصر", nameEn: "Al-Asr", ayat: 3 },
  { number: 104, name: "الهمزة", nameEn: "Al-Humazah", ayat: 9 },
  { number: 105, name: "الفيل", nameEn: "Al-Fil", ayat: 5 },
  { number: 106, name: "قريش", nameEn: "Quraysh", ayat: 4 },
  { number: 107, name: "الماعون", nameEn: "Al-Ma'un", ayat: 7 },
  { number: 108, name: "الكوثر", nameEn: "Al-Kawthar", ayat: 3 },
  { number: 109, name: "الكافرون", nameEn: "Al-Kafirun", ayat: 6 },
  { number: 110, name: "النصر", nameEn: "An-Nasr", ayat: 3 },
  { number: 111, name: "المسد", nameEn: "Al-Masad", ayat: 5 },
  { number: 112, name: "الإخلاص", nameEn: "Al-Ikhlas", ayat: 4 },
  { number: 113, name: "الفلق", nameEn: "Al-Falaq", ayat: 5 },
  { number: 114, name: "الناس", nameEn: "An-Nas", ayat: 6 },
];

export function AyahNavigator({ surah, ayah, onNavigate, language }: AyahNavigatorProps) {
  const currentSurah = SURAH_DATA.find(s => s.number === surah);
  const isRTL = language === "ar";

  const goToPrev = () => {
    if (ayah > 1) {
      onNavigate(surah, ayah - 1);
    } else if (surah > 1) {
      const prevSurah = SURAH_DATA.find(s => s.number === surah - 1);
      if (prevSurah) onNavigate(surah - 1, prevSurah.ayat);
    }
  };

  const goToNext = () => {
    if (currentSurah && ayah < currentSurah.ayat) {
      onNavigate(surah, ayah + 1);
    } else if (surah < 114) {
      onNavigate(surah + 1, 1);
    }
  };

  return (
    <div className="flex items-center gap-4">
      {/* Surah Selector */}
      <select
        className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
        value={surah}
        onChange={(e) => onNavigate(Number(e.target.value), 1)}
      >
        {SURAH_DATA.map((s) => (
          <option key={s.number} value={s.number}>
            {s.number}. {s.name} ({s.nameEn})
          </option>
        ))}
      </select>

      {/* Ayah Input */}
      <div className="flex items-center gap-2">
        <span className="text-slate-400">{isRTL ? "آية:" : "Ayah:"}</span>
        <input
          type="number"
          min={1}
          max={currentSurah?.ayat || 286}
          value={ayah}
          onChange={(e) => {
            const val = Number(e.target.value);
            if (val >= 1 && val <= (currentSurah?.ayat || 286)) {
              onNavigate(surah, val);
            }
          }}
          className="w-20 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-center"
        />
        <span className="text-slate-500">/ {currentSurah?.ayat}</span>
      </div>

      {/* Nav Buttons */}
      <div className="flex gap-1">
        <button
          onClick={goToPrev}
          disabled={surah === 1 && ayah === 1}
          className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={isRTL ? "السابق" : "Previous"}
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        <button
          onClick={goToNext}
          disabled={surah === 114 && ayah === (currentSurah?.ayat || 6)}
          className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={isRTL ? "التالي" : "Next"}
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

export { SURAH_DATA };
