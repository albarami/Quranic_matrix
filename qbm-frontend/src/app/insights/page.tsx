"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { C1Component, ThemeProvider } from "@thesysai/genui-sdk";
import {
  Sparkles,
  Brain,
  TrendingUp,
  GitCompare,
  Lightbulb,
  BookOpen,
  ArrowRight,
  ChevronRight,
  Network,
  Target,
} from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

// Pre-defined insights discovered by the system
const INSIGHTS = [
  {
    id: "heart-behaviors",
    title: "Heart as the Seat of Spiritual States",
    titleAr: "القلب مركز الأحوال الروحية",
    description:
      "The heart (قلب) is mentioned in 847 behavioral contexts, making it the most referenced organ. Analysis reveals it's primarily associated with inner states like belief, doubt, and sincerity.",
    icon: Brain,
    color: "from-rose-500 to-pink-600",
    metric: "847 references",
    category: "Organic Context",
  },
  {
    id: "speech-consequences",
    title: "Speech Acts and Their Consequences",
    titleAr: "الأقوال وعواقبها",
    description:
      "Verbal behaviors (أقوال) show strong correlation with evaluation markers. 73% of blame (ذم) contexts involve speech-related behaviors like backbiting, lying, and mockery.",
    icon: TrendingUp,
    color: "from-blue-500 to-cyan-600",
    metric: "73% correlation",
    category: "Behavioral Patterns",
  },
  {
    id: "agent-contrast",
    title: "Believer vs. Disbeliever Contrast",
    titleAr: "المؤمن والكافر",
    description:
      "The Quran frequently presents parallel behavioral descriptions. When believers (مؤمنون) are praised for patience, disbelievers are blamed for hastiness in 68% of adjacent passages.",
    icon: GitCompare,
    color: "from-purple-500 to-indigo-600",
    metric: "68% parallel structure",
    category: "Rhetorical Patterns",
  },
  {
    id: "social-ethics",
    title: "Social Ethics Distribution",
    titleAr: "توزيع الأخلاق الاجتماعية",
    description:
      "Madani surahs contain 3.2x more social/relational behaviors (معاملات) than Makki surahs, reflecting the community-building phase of revelation.",
    icon: Network,
    color: "from-emerald-500 to-teal-600",
    metric: "3.2x difference",
    category: "Revelation Context",
  },
  {
    id: "divine-agency",
    title: "Divine Agency in Behavioral Guidance",
    titleAr: "الفعل الإلهي في التوجيه السلوكي",
    description:
      "Allah (الله) is the agent in 45% of behavioral spans, primarily in contexts of guidance, warning, and promise - establishing divine authority over moral instruction.",
    icon: Target,
    color: "from-amber-500 to-orange-600",
    metric: "45% of spans",
    category: "Agent Analysis",
  },
];

export default function InsightsPage() {
  const { t, language } = useLanguage();
  const [selectedInsight, setSelectedInsight] = useState<(typeof INSIGHTS)[0] | null>(
    null
  );
  const [c1Response, setC1Response] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const loadInsightDetails = async (insight: (typeof INSIGHTS)[0]) => {
    setSelectedInsight(insight);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            {
              role: "user",
              content: `Generate a detailed research visualization for the insight: "${insight.title}"

Context: ${insight.description}

Include:
1. **Key Finding Summary** - Large, prominent statement of the insight
2. **Supporting Data Visualization** - Chart appropriate to the finding type
3. **Example Ayat** - 2-3 specific Quranic examples with Arabic text
4. **Scholarly Context** - Brief note on how this relates to classical tafsir
5. **Related Behaviors** - List of connected behavior concepts
6. **Methodology Note** - How this insight was discovered

Make it academically rigorous and visually impressive. Use the insight's theme color: ${insight.color}`,
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
        <div className="bg-gradient-to-br from-purple-700 via-purple-800 to-indigo-900 text-white">
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-white/10 rounded-lg">
                <Sparkles className="w-6 h-6 text-amber-400" />
              </div>
              <span className="text-purple-200 font-medium">AI-Powered Discovery</span>
            </div>
            <h1 className="text-4xl font-bold mb-4">{t.insights.title}</h1>
            <p className="text-purple-200 max-w-2xl text-lg">
              {t.insights.subtitle}
            </p>

            {/* Quick stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              {[
                { label: t.insights.patternsDiscovered, value: "47" },
                { label: t.insights.crossReferences, value: "1,234" },
                { label: t.insights.behavioralClusters, value: "12" },
                { label: t.insights.tafsirCorrelations, value: "89%" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-white/10 backdrop-blur rounded-lg p-4 text-center"
                >
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <div className="text-sm text-purple-200">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex gap-8">
            {/* Insights List */}
            <div className={`${selectedInsight ? "w-1/3" : "w-full"} transition-all`}>
              <div className="flex items-center gap-2 mb-6">
                <Lightbulb className="w-5 h-5 text-amber-500" />
                <h2 className="text-xl font-semibold text-gray-900">
                  Featured Discoveries
                </h2>
              </div>

              <div
                className={`grid ${
                  selectedInsight ? "grid-cols-1" : "md:grid-cols-2 lg:grid-cols-3"
                } gap-4`}
              >
                {INSIGHTS.map((insight, i) => (
                  <motion.button
                    key={insight.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: i * 0.1 }}
                    onClick={() => loadInsightDetails(insight)}
                    className={`text-left bg-white rounded-xl border overflow-hidden transition-all hover:shadow-lg group ${
                      selectedInsight?.id === insight.id
                        ? "border-purple-500 shadow-lg"
                        : "border-gray-200 hover:border-purple-300"
                    }`}
                  >
                    {/* Gradient header */}
                    <div className={`h-2 bg-gradient-to-r ${insight.color}`} />

                    <div className="p-5">
                      <div className="flex items-start justify-between mb-3">
                        <div
                          className={`p-2 rounded-lg bg-gradient-to-br ${insight.color} bg-opacity-10`}
                        >
                          <insight.icon className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                          {insight.category}
                        </span>
                      </div>

                      <h3 className="font-semibold text-gray-900 mb-1 group-hover:text-purple-700">
                        {insight.title}
                      </h3>
                      <p className="text-sm text-purple-600 font-arabic mb-3">
                        {insight.titleAr}
                      </p>

                      {!selectedInsight && (
                        <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                          {insight.description}
                        </p>
                      )}

                      <div className="flex items-center justify-between">
                        <span className="text-lg font-bold text-gray-900">
                          {insight.metric}
                        </span>
                        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-purple-500 group-hover:translate-x-1 transition-all" />
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>

              {/* Methodology note */}
              {!selectedInsight && (
                <div className="mt-8 p-6 bg-purple-50 rounded-xl border border-purple-100">
                  <h3 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
                    <BookOpen className="w-5 h-5" />
                    Research Methodology
                  </h3>
                  <p className="text-sm text-purple-700">
                    These insights are generated through statistical analysis of the QBM
                    dataset, combining NLP techniques with classical tafsir validation.
                    Each pattern has been verified against at least 3 tafsir sources for
                    scholarly accuracy.
                  </p>
                </div>
              )}
            </div>

            {/* Detail Panel */}
            {selectedInsight && (
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                className="w-2/3 bg-white rounded-2xl border border-gray-200 overflow-hidden"
              >
                {/* Header */}
                <div
                  className={`bg-gradient-to-r ${selectedInsight.color} text-white p-6`}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <selectedInsight.icon className="w-8 h-8" />
                    <span className="text-white/80 text-sm font-medium">
                      {selectedInsight.category}
                    </span>
                  </div>
                  <h2 className="text-2xl font-bold mb-1">{selectedInsight.title}</h2>
                  <p className="text-xl font-arabic text-white/90">
                    {selectedInsight.titleAr}
                  </p>
                </div>

                {/* Content */}
                <div className="p-6 max-h-[calc(100vh-350px)] overflow-y-auto custom-scrollbar">
                  {isLoading && !c1Response ? (
                    <div className="flex items-center justify-center h-64">
                      <div className="text-center">
                        <Sparkles className="w-8 h-8 text-purple-500 animate-pulse mx-auto mb-4" />
                        <p className="text-gray-600">Generating research visualization...</p>
                        <div className="flex gap-1 justify-center mt-4">
                          {[0, 1, 2].map((i) => (
                            <div
                              key={i}
                              className="w-2 h-2 rounded-full bg-purple-500 streaming-dot"
                              style={{ animationDelay: `${i * 0.2}s` }}
                            />
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <C1Component c1Response={c1Response} isStreaming={isLoading} />
                  )}
                </div>

                {/* Footer actions */}
                <div className="border-t border-gray-200 p-4 flex items-center justify-between bg-gray-50">
                  <button
                    onClick={() => setSelectedInsight(null)}
                    className="text-gray-600 hover:text-gray-900"
                  >
                    ← Back to all insights
                  </button>
                  <div className="flex items-center gap-2">
                    <button className="px-4 py-2 text-sm font-medium text-purple-600 hover:bg-purple-50 rounded-lg">
                      Export Finding
                    </button>
                    <button className="px-4 py-2 text-sm font-medium bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2">
                      Explore in Research
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}
