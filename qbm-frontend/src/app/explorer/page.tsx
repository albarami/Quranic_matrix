"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { C1Component, ThemeProvider } from "@thesysai/genui-sdk";
import {
  BookOpen,
  Search,
  Grid,
  List,
  ChevronRight,
  X,
  BarChart3,
  Eye,
  Brain,
  Sparkles,
} from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

// Surah data with coverage info
const SURAHS = [
  { number: 1, name: "الفاتحة", nameEn: "Al-Fatiha", ayat: 7, spans: 12, coverage: 100 },
  { number: 2, name: "البقرة", nameEn: "Al-Baqarah", ayat: 286, spans: 542, coverage: 95 },
  { number: 3, name: "آل عمران", nameEn: "Ali 'Imran", ayat: 200, spans: 387, coverage: 90 },
  { number: 4, name: "النساء", nameEn: "An-Nisa", ayat: 176, spans: 298, coverage: 88 },
  { number: 5, name: "المائدة", nameEn: "Al-Ma'idah", ayat: 120, spans: 234, coverage: 92 },
  { number: 6, name: "الأنعام", nameEn: "Al-An'am", ayat: 165, spans: 189, coverage: 75 },
  { number: 7, name: "الأعراف", nameEn: "Al-A'raf", ayat: 206, spans: 267, coverage: 78 },
  { number: 8, name: "الأنفال", nameEn: "Al-Anfal", ayat: 75, spans: 134, coverage: 85 },
  { number: 9, name: "التوبة", nameEn: "At-Tawbah", ayat: 129, spans: 198, coverage: 82 },
  { number: 10, name: "يونس", nameEn: "Yunus", ayat: 109, spans: 145, coverage: 70 },
  // ... more surahs
].concat(
  Array.from({ length: 104 }, (_, i) => ({
    number: i + 11,
    name: `سورة ${i + 11}`,
    nameEn: `Surah ${i + 11}`,
    ayat: Math.floor(Math.random() * 100) + 10,
    spans: Math.floor(Math.random() * 150) + 20,
    coverage: Math.floor(Math.random() * 40) + 60,
  }))
);

// Top behaviors in dataset
const TOP_BEHAVIORS = [
  { id: "BEH_BELIEF", nameAr: "الإيمان", nameEn: "Belief", count: 847, color: "#10b981" },
  { id: "BEH_PATIENCE", nameAr: "الصبر", nameEn: "Patience", count: 423, color: "#3b82f6" },
  { id: "BEH_GRATITUDE", nameAr: "الشكر", nameEn: "Gratitude", count: 312, color: "#f59e0b" },
  { id: "BEH_PRAYER", nameAr: "الصلاة", nameEn: "Prayer", count: 287, color: "#8b5cf6" },
  { id: "BEH_CHARITY", nameAr: "الصدقة", nameEn: "Charity", count: 198, color: "#ec4899" },
  { id: "BEH_TRUTHFULNESS", nameAr: "الصدق", nameEn: "Truthfulness", count: 167, color: "#06b6d4" },
];

function getCoverageColor(coverage: number) {
  if (coverage >= 90) return "bg-emerald-500";
  if (coverage >= 70) return "bg-emerald-400";
  if (coverage >= 50) return "bg-emerald-300";
  return "bg-emerald-200";
}

export default function ExplorerPage() {
  const { t, language, isRTL } = useLanguage();
  const [view, setView] = useState<"grid" | "list">("grid");
  const [selectedSurah, setSelectedSurah] = useState<typeof SURAHS[0] | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [c1Response, setC1Response] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [filterBehavior, setFilterBehavior] = useState<string | null>(null);

  const filteredSurahs = SURAHS.filter(
    (s) =>
      s.name.includes(searchQuery) ||
      s.nameEn.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.number.toString().includes(searchQuery)
  );

  const loadSurahDetails = async (surah: typeof SURAHS[0]) => {
    setSelectedSurah(surah);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            {
              role: "user",
              content: `Generate a detailed exploration view for Surah ${surah.nameEn} (${surah.name}), Surah ${surah.number}.

Show:
1. **Surah Overview Card**
   - Name in Arabic and English
   - Total ayat: ${surah.ayat}
   - Behavioral spans: ${surah.spans}
   - Coverage: ${surah.coverage}%

2. **Behavior Distribution** (horizontal bar chart)
   - Show top 8 behavior concepts found in this surah
   - Use emerald/green color palette
   - Include Arabic and English labels

3. **Agent Breakdown** (pie chart)
   - Distribution of agent types in this surah

4. **Notable Ayat** (interactive cards)
   - Show 3-4 ayat with significant behavioral content
   - Include Arabic text, translation
   - Show behavior tags

5. **Action Buttons**
   - "View all ${surah.spans} annotations"
   - "Compare with similar surahs"
   - "Export surah data"

Make it visually rich and scholarly.`,
            },
          ],
        }),
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value);
        setC1Response(accumulated);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ThemeProvider>
      <div className="min-h-[calc(100vh-64px)] bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 sticky top-0 z-20">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                  <BookOpen className="w-7 h-7 text-emerald-600" />
                  {t.explorer.title}
                </h1>
                <p className="text-gray-600">
                  {t.explorer.subtitle}
                </p>
              </div>

              <div className="flex items-center gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder={t.explorer.search}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg w-64 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>

                {/* View toggle */}
                <div className="flex items-center bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setView("grid")}
                    className={`p-2 rounded-md ${view === "grid" ? "bg-white shadow" : ""}`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setView("list")}
                    className={`p-2 rounded-md ${view === "list" ? "bg-white shadow" : ""}`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Behavior filter chips */}
            <div className="flex items-center gap-2 mt-4 overflow-x-auto pb-2">
              <span className="text-sm text-gray-500 flex-shrink-0">{t.explorer.filterBy}</span>
              {TOP_BEHAVIORS.map((behavior) => (
                <button
                  key={behavior.id}
                  onClick={() =>
                    setFilterBehavior(filterBehavior === behavior.id ? null : behavior.id)
                  }
                  className={`flex-shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    filterBehavior === behavior.id
                      ? "bg-emerald-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {behavior.nameEn}
                  <span className="ml-1 opacity-70">({behavior.count})</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex gap-8">
            {/* Surah Grid/List */}
            <div className={`${selectedSurah ? "w-1/2" : "w-full"} transition-all`}>
              {/* Stats bar */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: t.explorer.totalSurahs, value: "114", icon: BookOpen },
                  { label: t.explorer.annotatedAyat, value: "6,236", icon: Eye },
                  { label: t.explorer.behavioralSpans, value: "15,847", icon: Brain },
                  { label: t.explorer.avgCoverage, value: "85%", icon: BarChart3 },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="bg-white rounded-xl p-4 border border-gray-200"
                  >
                    <stat.icon className="w-5 h-5 text-emerald-600 mb-2" />
                    <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                    <div className="text-sm text-gray-500">{stat.label}</div>
                  </div>
                ))}
              </div>

              {/* Grid view */}
              {view === "grid" ? (
                <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-1">
                  {filteredSurahs.map((surah) => (
                    <motion.button
                      key={surah.number}
                      whileHover={{ scale: 1.1, zIndex: 10 }}
                      onClick={() => loadSurahDetails(surah)}
                      className={`aspect-square rounded-lg flex flex-col items-center justify-center text-xs font-medium transition-all cursor-pointer relative group ${getCoverageColor(
                        surah.coverage
                      )} ${
                        selectedSurah?.number === surah.number
                          ? "ring-2 ring-emerald-600 ring-offset-2"
                          : ""
                      }`}
                      title={`${surah.nameEn} - ${surah.coverage}% coverage`}
                    >
                      <span className="text-white font-bold">{surah.number}</span>
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-20">
                        {surah.nameEn}
                        <br />
                        {surah.coverage}% coverage
                      </div>
                    </motion.button>
                  ))}
                </div>
              ) : (
                /* List view */
                <div className="space-y-2">
                  {filteredSurahs.slice(0, 20).map((surah) => (
                    <motion.button
                      key={surah.number}
                      whileHover={{ x: 4 }}
                      onClick={() => loadSurahDetails(surah)}
                      className={`w-full flex items-center gap-4 p-4 bg-white rounded-xl border transition-all hover:border-emerald-300 hover:shadow-md ${
                        selectedSurah?.number === surah.number
                          ? "border-emerald-500 shadow-md"
                          : "border-gray-200"
                      }`}
                    >
                      <div
                        className={`w-12 h-12 rounded-xl flex items-center justify-center text-white font-bold ${getCoverageColor(
                          surah.coverage
                        )}`}
                      >
                        {surah.number}
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-semibold text-gray-900">{surah.nameEn}</div>
                        <div className="text-sm text-gray-500 font-arabic">{surah.name}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {surah.spans} spans
                        </div>
                        <div className="text-xs text-gray-500">{surah.ayat} ayat</div>
                      </div>
                      <div className="w-24">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="text-gray-500">Coverage</span>
                          <span className="font-medium">{surah.coverage}%</span>
                        </div>
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full"
                            style={{ width: `${surah.coverage}%` }}
                          />
                        </div>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    </motion.button>
                  ))}
                </div>
              )}

              {/* Legend */}
              <div className="mt-6 flex items-center gap-6 text-sm text-gray-600">
                <span className="font-medium">{t.explorer.coverage}:</span>
                {[
                  { label: "90%+", color: "bg-emerald-500" },
                  { label: "70-89%", color: "bg-emerald-400" },
                  { label: "50-69%", color: "bg-emerald-300" },
                  { label: "<50%", color: "bg-emerald-200" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded ${item.color}`} />
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Detail Panel */}
            <AnimatePresence>
              {selectedSurah && (
                <motion.div
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 50 }}
                  className="w-1/2 bg-white rounded-2xl border border-gray-200 overflow-hidden"
                >
                  {/* Header */}
                  <div className="bg-gradient-to-r from-emerald-600 to-emerald-700 text-white p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-4xl font-arabic mb-2">{selectedSurah.name}</div>
                        <h2 className="text-2xl font-bold">{selectedSurah.nameEn}</h2>
                        <p className="text-emerald-200">Surah {selectedSurah.number}</p>
                      </div>
                      <button
                        onClick={() => setSelectedSurah(null)}
                        className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6 max-h-[calc(100vh-300px)] overflow-y-auto custom-scrollbar">
                    {isLoading && !c1Response ? (
                      <div className="flex items-center justify-center h-64">
                        <div className="text-center">
                          <div className="flex items-center gap-2 justify-center mb-4">
                            <Sparkles className="w-6 h-6 text-emerald-500 animate-pulse" />
                            <span className="text-gray-600">Generating insights...</span>
                          </div>
                          <div className="flex gap-1 justify-center">
                            {[0, 1, 2].map((i) => (
                              <div
                                key={i}
                                className="w-2 h-2 rounded-full bg-emerald-500 streaming-dot"
                                style={{ animationDelay: `${i * 0.2}s` }}
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <C1Component
                        c1Response={c1Response}
                        isStreaming={isLoading}
                      />
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}
