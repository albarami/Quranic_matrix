"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { BookOpen, Edit3, CheckCircle, BarChart3 } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import {
  AyahNavigator,
  AyahDisplay,
  TafsirPanel,
  AnnotationForm,
  ExistingAnnotations,
  SURAH_DATA
} from "../components/annotate";

export default function AnnotatePage() {
  const { language, isRTL } = useLanguage();
  const [currentSurah, setCurrentSurah] = useState(2);
  const [currentAyah, setCurrentAyah] = useState(255);
  const [activeTafsir, setActiveTafsir] = useState("ibn_kathir");

  const handleNavigate = (surah: number, ayah: number) => {
    setCurrentSurah(surah);
    setCurrentAyah(ayah);
  };

  const handleSkipToNext = () => {
    const surahData = SURAH_DATA.find(s => s.number === currentSurah);
    if (surahData && currentAyah < surahData.ayat) {
      setCurrentAyah(currentAyah + 1);
    } else if (currentSurah < 114) {
      setCurrentSurah(currentSurah + 1);
      setCurrentAyah(1);
    }
  };

  // Stats for the header
  const stats = {
    totalAnnotations: 6236,
    todayAnnotations: 0,
    currentProgress: `${currentSurah}:${currentAyah}`
  };

  return (
    <div className={`min-h-screen bg-slate-900 text-white ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            {/* Title */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center shadow-lg">
                <Edit3 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  {isRTL ? "منصة التعليق التوضيحي" : "Annotator Workbench"}
                </h1>
                <p className="text-slate-400 text-sm">
                  {isRTL ? "تصنيف السلوكيات القرآنية عبر 11 محور" : "Classify Quranic behaviors across 11 axes"}
                </p>
              </div>
            </div>

            {/* Navigation */}
            <AyahNavigator
              surah={currentSurah}
              ayah={currentAyah}
              onNavigate={handleNavigate}
              language={language}
            />
          </div>

          {/* Stats Bar */}
          <div className="flex flex-wrap items-center gap-6 mt-4 pt-4 border-t border-slate-700">
            <div className="flex items-center gap-2 text-sm">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              <span className="text-slate-400">{isRTL ? "إجمالي التعليقات:" : "Total Annotations:"}</span>
              <span className="text-white font-semibold">{stats.totalAnnotations.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <CheckCircle className="w-4 h-4 text-blue-400" />
              <span className="text-slate-400">{isRTL ? "اليوم:" : "Today:"}</span>
              <span className="text-white font-semibold">{stats.todayAnnotations}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <BookOpen className="w-4 h-4 text-purple-400" />
              <span className="text-slate-400">{isRTL ? "الموقع الحالي:" : "Current Position:"}</span>
              <span className="text-white font-mono">{stats.currentProgress}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Ayah Display */}
        <motion.div
          key={`${currentSurah}:${currentAyah}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <AyahDisplay
            surah={currentSurah}
            ayah={currentAyah}
            language={language}
          />
        </motion.div>

        {/* Tafsir Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <TafsirPanel
            surah={currentSurah}
            ayah={currentAyah}
            active={activeTafsir}
            onTabChange={setActiveTafsir}
            language={language}
          />
        </motion.div>

        {/* Annotation Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <AnnotationForm
            surah={currentSurah}
            ayah={currentAyah}
            language={language}
            onSkip={handleSkipToNext}
            onSave={(annotation) => {
              console.log("Annotation saved:", annotation);
              // In production, this would call the API
            }}
          />
        </motion.div>

        {/* Existing Annotations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <ExistingAnnotations
            surah={currentSurah}
            ayah={currentAyah}
            language={language}
            onEdit={(annotation) => {
              console.log("Edit annotation:", annotation);
            }}
            onDelete={(id) => {
              console.log("Delete annotation:", id);
            }}
          />
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <span>{isRTL ? "منصة QBM للتعليق التوضيحي" : "QBM Annotation Platform"}</span>
              <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 rounded text-xs">
                v1.0
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span>{isRTL ? "11 محور تصنيف" : "11 Classification Axes"}</span>
              <span>•</span>
              <span>{isRTL ? "7 مصادر تفسير" : "7 Tafsir Sources"}</span>
              <span>•</span>
              <span>114 {isRTL ? "سورة" : "Surahs"}</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
