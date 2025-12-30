"use client";

import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Book,
  Heart,
  Users,
  Hand,
  Brain,
  Award,
  ChevronRight,
  Hash,
  Globe,
  Sparkles,
  Filter,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import {
  EntityType,
  CanonicalEntity,
  ENTITY_COUNTS,
  ENTITY_TYPE_LABELS,
  ALL_ENTITIES,
  getEntitiesByType,
  searchEntities,
} from "@/lib/canonical-entities";

// Icon mapping for entity types
const ENTITY_ICONS: Record<EntityType, any> = {
  behavior: Sparkles,
  agent: Users,
  organ: Hand,
  heart_state: Heart,
  consequence: Award,
};

// Color mapping for entity types (Tailwind classes)
const ENTITY_COLORS: Record<EntityType, { bg: string; text: string; border: string; light: string }> = {
  behavior: { bg: "bg-emerald-500", text: "text-emerald-600", border: "border-emerald-500", light: "bg-emerald-50" },
  agent: { bg: "bg-blue-500", text: "text-blue-600", border: "border-blue-500", light: "bg-blue-50" },
  organ: { bg: "bg-purple-500", text: "text-purple-600", border: "border-purple-500", light: "bg-purple-50" },
  heart_state: { bg: "bg-pink-500", text: "text-pink-600", border: "border-pink-500", light: "bg-pink-50" },
  consequence: { bg: "bg-amber-500", text: "text-amber-600", border: "border-amber-500", light: "bg-amber-50" },
};

export default function SynonymsPage() {
  const { language, isRTL, t } = useLanguage();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<EntityType | "all">("all");

  // Filter entities based on search and active tab
  const filteredEntities = useMemo(() => {
    return searchEntities(searchQuery, activeTab);
  }, [searchQuery, activeTab]);

  // Tab data
  const tabs: { id: EntityType | "all"; count: number }[] = [
    { id: "all", count: ENTITY_COUNTS.total },
    { id: "behavior", count: ENTITY_COUNTS.behavior },
    { id: "agent", count: ENTITY_COUNTS.agent },
    { id: "organ", count: ENTITY_COUNTS.organ },
    { id: "heart_state", count: ENTITY_COUNTS.heart_state },
    { id: "consequence", count: ENTITY_COUNTS.consequence },
  ];

  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Hero Header */}
      <div className="bg-gradient-to-br from-emerald-800 via-emerald-700 to-teal-800 text-white">
        <div className="max-w-7xl mx-auto px-4 py-12">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 bg-emerald-600/30 px-4 py-2 rounded-full mb-4"
            >
              <Book className="w-4 h-4" />
              <span className="text-sm font-medium">
                {language === "ar" ? "مستكشف المترادفات" : "Synonym Explorer"}
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-4xl md:text-5xl font-bold mb-4 font-arabic"
            >
              {language === "ar" ? "القاموس السلوكي القرآني" : "Quranic Behavioral Lexicon"}
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="text-emerald-100 max-w-2xl mx-auto text-lg"
            >
              {language === "ar"
                ? "اكتشف 720+ مرادفاً عربياً عبر 155 كياناً قرآنياً"
                : "Explore 720+ Arabic synonyms across 155 Quranic entities"}
            </motion.p>
          </div>

          {/* Statistics Cards */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mt-8"
          >
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-3xl font-bold">{ENTITY_COUNTS.total}</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "كيانات" : "Entities"}
              </div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-3xl font-bold">{ENTITY_COUNTS.totalSynonyms}+</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "مرادفات" : "Synonyms"}
              </div>
            </div>
            <div className="bg-emerald-600/30 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-2xl font-bold">{ENTITY_COUNTS.behavior}</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "سلوكيات" : "Behaviors"}
              </div>
            </div>
            <div className="bg-blue-600/30 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-2xl font-bold">{ENTITY_COUNTS.agent}</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "فاعلون" : "Agents"}
              </div>
            </div>
            <div className="bg-purple-600/30 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-2xl font-bold">{ENTITY_COUNTS.organ}</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "أعضاء" : "Organs"}
              </div>
            </div>
            <div className="bg-pink-600/30 backdrop-blur-sm rounded-xl p-4 text-center">
              <div className="text-2xl font-bold">{ENTITY_COUNTS.heart_state + ENTITY_COUNTS.consequence}</div>
              <div className="text-emerald-200 text-sm">
                {language === "ar" ? "أحوال وعواقب" : "States & Outcomes"}
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Search Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="relative mb-6"
        >
          <Search className={`absolute top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 ${isRTL ? 'right-4' : 'left-4'}`} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={language === "ar" ? "ابحث بالعربية أو الإنجليزية أو الجذر..." : "Search by Arabic, English, or root..."}
            className={`w-full py-4 rounded-xl border border-slate-200 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-lg ${isRTL ? 'pr-12 pl-4' : 'pl-12 pr-4'}`}
            dir={isRTL ? "rtl" : "ltr"}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className={`absolute top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 ${isRTL ? 'left-4' : 'right-4'}`}
            >
              ×
            </button>
          )}
        </motion.div>

        {/* Category Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="flex flex-wrap gap-2 mb-8"
        >
          {tabs.map((tab) => {
            const label = ENTITY_TYPE_LABELS[tab.id];
            const Icon = tab.id === "all" ? Globe : ENTITY_ICONS[tab.id as EntityType];
            const isActive = activeTab === tab.id;
            const colors = tab.id === "all" ? null : ENTITY_COLORS[tab.id as EntityType];

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full font-medium transition-all ${
                  isActive
                    ? tab.id === "all"
                      ? "bg-slate-800 text-white"
                      : `${colors?.bg} text-white`
                    : "bg-white border border-slate-200 text-slate-600 hover:border-slate-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{language === "ar" ? label.ar : label.en}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  isActive ? "bg-white/20" : "bg-slate-100"
                }`}>
                  {tab.count}
                </span>
              </button>
            );
          })}
        </motion.div>

        {/* Results Count */}
        <div className="mb-6 text-slate-600">
          {language === "ar"
            ? `عرض ${filteredEntities.length} من ${activeTab === "all" ? ENTITY_COUNTS.total : tabs.find(t => t.id === activeTab)?.count} كيان`
            : `Showing ${filteredEntities.length} of ${activeTab === "all" ? ENTITY_COUNTS.total : tabs.find(t => t.id === activeTab)?.count} entities`}
        </div>

        {/* Entity Cards Grid */}
        <motion.div
          layout
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <AnimatePresence mode="popLayout">
            {filteredEntities.map((entity) => (
              <EntityCard key={entity.id} entity={entity} language={language} isRTL={isRTL} />
            ))}
          </AnimatePresence>
        </motion.div>

        {/* Empty State */}
        {filteredEntities.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <Search className="w-16 h-16 text-slate-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-600 mb-2">
              {language === "ar" ? "لا توجد نتائج" : "No Results Found"}
            </h3>
            <p className="text-slate-500">
              {language === "ar"
                ? "جرب البحث بكلمات مختلفة أو غير الفئة"
                : "Try different search terms or change the category"}
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
}

// Entity Card Component
function EntityCard({
  entity,
  language,
  isRTL
}: {
  entity: CanonicalEntity;
  language: string;
  isRTL: boolean;
}) {
  const colors = ENTITY_COLORS[entity.type];
  const Icon = ENTITY_ICONS[entity.type];
  const typeLabel = ENTITY_TYPE_LABELS[entity.type];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className={`bg-white rounded-xl border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow`}
    >
      {/* Card Header */}
      <div className={`${colors.light} px-4 py-3 border-b ${colors.border} border-opacity-20`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 ${colors.bg} rounded-lg flex items-center justify-center`}>
              <Icon className="w-4 h-4 text-white" />
            </div>
            <div>
              <span className={`text-xs font-medium ${colors.text}`}>
                {language === "ar" ? typeLabel.ar : typeLabel.en}
              </span>
              {entity.category && (
                <span className="text-xs text-slate-400 block">
                  {entity.category}
                </span>
              )}
            </div>
          </div>
          <code className="text-xs text-slate-400 font-mono bg-slate-100 px-2 py-1 rounded">
            {entity.id}
          </code>
        </div>
      </div>

      {/* Card Body */}
      <div className="p-4">
        {/* Arabic & English Names */}
        <div className="mb-4">
          <h3 className="text-2xl font-bold text-slate-800 font-arabic mb-1" dir="rtl">
            {entity.ar}
          </h3>
          <p className="text-slate-600">{entity.en}</p>
        </div>

        {/* Root */}
        <div className="mb-4">
          <div className="text-xs text-slate-500 mb-1 flex items-center gap-1">
            <Hash className="w-3 h-3" />
            {language === "ar" ? "الجذر" : "Root"}
          </div>
          <div className="flex flex-wrap gap-1">
            {entity.roots.map((root, idx) => (
              <span
                key={idx}
                className="inline-block bg-slate-100 text-slate-700 px-2 py-1 rounded text-sm font-arabic"
                dir="rtl"
              >
                {root}
              </span>
            ))}
          </div>
        </div>

        {/* Synonyms */}
        <div className="mb-3">
          <div className="text-xs text-slate-500 mb-2 flex items-center gap-1">
            <Sparkles className="w-3 h-3" />
            {language === "ar" ? "المرادفات" : "Synonyms"} ({entity.synonyms.length})
          </div>
          <div className="flex flex-wrap gap-1" dir="rtl">
            {entity.synonyms.slice(0, 8).map((syn, idx) => (
              <span
                key={idx}
                className={`inline-block ${colors.light} ${colors.text} px-2 py-1 rounded text-sm font-arabic`}
              >
                {syn}
              </span>
            ))}
            {entity.synonyms.length > 8 && (
              <span className="inline-block bg-slate-100 text-slate-500 px-2 py-1 rounded text-sm">
                +{entity.synonyms.length - 8}
              </span>
            )}
          </div>
        </div>

        {/* Occurrences */}
        {entity.occurrences && (
          <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-500">
              {language === "ar" ? "الورود في القرآن" : "Quranic Occurrences"}
            </span>
            <span className={`${colors.text} font-semibold`}>
              {entity.occurrences}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
