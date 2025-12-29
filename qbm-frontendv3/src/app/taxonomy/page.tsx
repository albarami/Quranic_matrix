"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";
import {
  Heart,
  Eye,
  Hand,
  MessageCircle,
  Brain,
  Home,
  Briefcase,
  Users,
  Sun,
  Moon,
  Sunrise,
  Sunset,
  ThumbsUp,
  ThumbsDown,
  Scale,
  Globe,
  Sparkles,
  ChevronRight,
  BookOpen,
  Search,
  ArrowRight,
  Layers,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

const CHART_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

// Bouzidani's 5-Axis Taxonomy Structure
interface TaxonomyAxis {
  id: string;
  nameAr: string;
  nameEn: string;
  descriptionAr: string;
  descriptionEn: string;
  icon: any;
  color: string;
  gradient: string;
  categories: TaxonomyCategory[];
}

interface TaxonomyCategory {
  id: string;
  nameAr: string;
  nameEn: string;
  icon: any;
  examples?: string[];
}

const TAXONOMY_AXES: TaxonomyAxis[] = [
  {
    id: "organic",
    nameAr: "التصنيف العضوي البيولوجي",
    nameEn: "Organic Classification",
    descriptionAr: "تصنيف السلوكيات حسب الأعضاء الجسدية والحالات الداخلية",
    descriptionEn: "Classification of behaviors by bodily organs and internal states",
    icon: Heart,
    color: "text-rose-600",
    gradient: "from-rose-500 to-pink-600",
    categories: [
      { id: "heart", nameAr: "القلب", nameEn: "Heart (Qalb)", icon: Heart, examples: ["الإيمان", "الخشوع", "الحب"] },
      { id: "tongue", nameAr: "اللسان", nameEn: "Tongue (Lisan)", icon: MessageCircle, examples: ["الذكر", "الدعاء", "الغيبة"] },
      { id: "eye", nameAr: "العين", nameEn: "Eye (Ayn)", icon: Eye, examples: ["النظر", "البصيرة", "غض البصر"] },
      { id: "hand", nameAr: "اليد", nameEn: "Hand (Yad)", icon: Hand, examples: ["الصدقة", "العمل", "البطش"] },
      { id: "internal", nameAr: "الباطن", nameEn: "Internal (Batin)", icon: Brain, examples: ["النية", "الفكر", "العاطفة"] },
    ],
  },
  {
    id: "situational",
    nameAr: "التصنيف الموضعي",
    nameEn: "Situational Classification",
    descriptionAr: "تصنيف السلوكيات حسب السياق والموضع",
    descriptionEn: "Classification by context and situation",
    icon: Globe,
    color: "text-blue-600",
    gradient: "from-blue-500 to-cyan-600",
    categories: [
      { id: "self", nameAr: "النفس", nameEn: "Self (Nafs)", icon: Heart, examples: ["التزكية", "المجاهدة", "الصبر"] },
      { id: "horizons", nameAr: "الآفاق", nameEn: "Horizons (Afaq)", icon: Globe, examples: ["التفكر", "التدبر", "الاعتبار"] },
      { id: "creator", nameAr: "الخالق", nameEn: "Creator (Khaliq)", icon: Sparkles, examples: ["العبادة", "التوكل", "الخشية"] },
      { id: "universe", nameAr: "الكون", nameEn: "Universe (Kawn)", icon: Sun, examples: ["الاستخلاف", "الإعمار", "الحفظ"] },
      { id: "life", nameAr: "الحياة", nameEn: "Life (Hayat)", icon: Layers, examples: ["الإحسان", "العدل", "الرحمة"] },
    ],
  },
  {
    id: "systemic",
    nameAr: "التصنيف النسقي",
    nameEn: "Systemic Classification",
    descriptionAr: "تصنيف السلوكيات حسب النظام الاجتماعي",
    descriptionEn: "Classification by social system context",
    icon: Users,
    color: "text-purple-600",
    gradient: "from-purple-500 to-indigo-600",
    categories: [
      { id: "home", nameAr: "البيت", nameEn: "Home (Bayt)", icon: Home, examples: ["بر الوالدين", "صلة الرحم", "التربية"] },
      { id: "work", nameAr: "العمل", nameEn: "Work (Amal)", icon: Briefcase, examples: ["الإتقان", "الأمانة", "الصدق"] },
      { id: "public", nameAr: "المجتمع", nameEn: "Public (Mujtama)", icon: Users, examples: ["الأمر بالمعروف", "النهي عن المنكر", "التعاون"] },
    ],
  },
  {
    id: "temporal",
    nameAr: "التصنيف الزماني",
    nameEn: "Temporal Classification",
    descriptionAr: "تصنيف السلوكيات حسب الوقت",
    descriptionEn: "Classification by time of day",
    icon: Sun,
    color: "text-amber-600",
    gradient: "from-amber-500 to-orange-600",
    categories: [
      { id: "morning", nameAr: "الصباح", nameEn: "Morning (Sabah)", icon: Sunrise, examples: ["صلاة الفجر", "أذكار الصباح"] },
      { id: "noon", nameAr: "الظهر", nameEn: "Noon (Zuhr)", icon: Sun, examples: ["صلاة الظهر", "القيلولة"] },
      { id: "afternoon", nameAr: "العصر", nameEn: "Afternoon (Asr)", icon: Sunset, examples: ["صلاة العصر", "الاستغفار"] },
      { id: "night", nameAr: "الليل", nameEn: "Night (Layl)", icon: Moon, examples: ["قيام الليل", "التهجد", "أذكار النوم"] },
    ],
  },
  {
    id: "evaluation",
    nameAr: "التصنيف التقييمي",
    nameEn: "Evaluation Classification",
    descriptionAr: "تصنيف السلوكيات حسب الحكم الشرعي",
    descriptionEn: "Classification by moral/religious evaluation",
    icon: Scale,
    color: "text-emerald-600",
    gradient: "from-emerald-500 to-teal-600",
    categories: [
      { id: "praise", nameAr: "مدح", nameEn: "Praise (Madh)", icon: ThumbsUp, examples: ["الصدق", "الأمانة", "الإحسان"] },
      { id: "blame", nameAr: "ذم", nameEn: "Blame (Dhamm)", icon: ThumbsDown, examples: ["الكذب", "الخيانة", "الظلم"] },
      { id: "neutral", nameAr: "سواء", nameEn: "Neutral (Sawa)", icon: Scale, examples: ["المباحات", "العادات"] },
    ],
  },
];

export default function TaxonomyPage() {
  const { language, isRTL } = useLanguage();
  const [selectedAxis, setSelectedAxis] = useState<TaxonomyAxis | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<TaxonomyCategory | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load stats from backend
  useEffect(() => {
    const loadStats = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/stats`);
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch (e) {
        console.error("Failed to load stats:", e);
      } finally {
        setIsLoading(false);
      }
    };
    loadStats();
  }, []);

  // Prepare radar chart data for taxonomy overview
  const radarData = TAXONOMY_AXES.map((axis, i) => ({
    axis: language === "ar" ? axis.nameAr.split(" ")[1] : axis.nameEn.split(" ")[0],
    value: (5 - i) * 20 + Math.random() * 20, // Simulated - would come from real data
    fullMark: 100,
  }));

  return (
    <div className={`min-h-[calc(100vh-64px)] bg-gradient-to-b from-gray-50 to-white ${isRTL ? 'rtl' : 'ltr'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Hero Header */}
      <div className="bg-gradient-to-br from-emerald-800 via-emerald-900 to-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-white/10 rounded-xl">
              <Layers className="w-8 h-8 text-emerald-300" />
            </div>
            <div>
              <span className="text-emerald-300 text-sm font-medium">
                {language === "ar" ? "البناء التصنيفي البوزيداني" : "Bouzidani's Taxonomic Framework"}
              </span>
            </div>
          </div>
          
          <h1 className="text-4xl font-bold mb-4">
            {language === "ar" ? "مصفوفة التصنيف القرآني لسلوك الإنسان" : "Quranic Human Behavior Classification Matrix"}
          </h1>
          
          <p className="text-emerald-200 max-w-3xl text-lg leading-relaxed">
            {language === "ar" 
              ? "إطار علمي شامل لتصنيف السلوكيات البشرية في القرآن الكريم عبر خمسة محاور رئيسية: العضوي، الموضعي، النسقي، الزماني، والتقييمي."
              : "A comprehensive scholarly framework for classifying human behaviors in the Holy Quran across five main axes: Organic, Situational, Systemic, Temporal, and Evaluative."}
          </p>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-8">
            {TAXONOMY_AXES.map((axis, i) => (
              <motion.button
                key={axis.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                onClick={() => setSelectedAxis(axis)}
                className={`bg-white/10 backdrop-blur rounded-xl p-4 text-center hover:bg-white/20 transition-all ${
                  selectedAxis?.id === axis.id ? 'ring-2 ring-white' : ''
                }`}
              >
                <axis.icon className="w-6 h-6 mx-auto mb-2 text-emerald-300" />
                <div className="text-lg font-bold">{axis.categories.length}</div>
                <div className="text-xs text-emerald-200">
                  {language === "ar" ? axis.nameAr.split(" ")[1] : axis.nameEn.split(" ")[0]}
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Taxonomy Overview - Radar Chart */}
        {!selectedAxis && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Radar Chart */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <Layers className="w-5 h-5 text-emerald-600" />
                  {language === "ar" ? "نظرة عامة على المحاور" : "Axes Overview"}
                </h2>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#e5e7eb" />
                    <PolarAngleAxis dataKey="axis" tick={{ fontSize: 12 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} />
                    <Radar
                      name="Coverage"
                      dataKey="value"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.5}
                    />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Framework Description */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-emerald-600" />
                  {language === "ar" ? "عن الإطار التصنيفي" : "About the Framework"}
                </h2>
                <div className="prose prose-emerald max-w-none">
                  <p className="text-gray-600 leading-relaxed">
                    {language === "ar"
                      ? "يقدم هذا الإطار التصنيفي نظرة شاملة لفهم السلوك البشري كما ورد في القرآن الكريم. يعتمد على خمسة محاور متكاملة تغطي الجوانب الجسدية والنفسية والاجتماعية والزمانية والأخلاقية للسلوك."
                      : "This taxonomic framework provides a comprehensive view for understanding human behavior as described in the Holy Quran. It is based on five integrated axes covering the physical, psychological, social, temporal, and moral aspects of behavior."}
                  </p>
                  <div className="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-100">
                    <p className="text-sm text-emerald-700 font-medium">
                      {language === "ar"
                        ? "💡 انقر على أي محور أعلاه لاستكشاف فئاته بالتفصيل"
                        : "💡 Click on any axis above to explore its categories in detail"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Axes Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {TAXONOMY_AXES.map((axis, i) => (
            <motion.div
              key={axis.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`bg-white rounded-2xl shadow-lg overflow-hidden cursor-pointer transition-all hover:shadow-xl ${
                selectedAxis?.id === axis.id ? 'ring-2 ring-emerald-500' : ''
              }`}
              onClick={() => setSelectedAxis(selectedAxis?.id === axis.id ? null : axis)}
            >
              {/* Axis Header */}
              <div className={`bg-gradient-to-r ${axis.gradient} text-white p-5`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <axis.icon className="w-8 h-8" />
                    <div>
                      <h3 className="font-bold text-lg">
                        {language === "ar" ? axis.nameAr : axis.nameEn}
                      </h3>
                      <p className="text-sm opacity-80">
                        {axis.categories.length} {language === "ar" ? "فئات" : "categories"}
                      </p>
                    </div>
                  </div>
                  <ChevronRight className={`w-5 h-5 transition-transform ${
                    selectedAxis?.id === axis.id ? 'rotate-90' : ''
                  }`} />
                </div>
              </div>

              {/* Axis Description */}
              <div className="p-5">
                <p className="text-gray-600 text-sm mb-4">
                  {language === "ar" ? axis.descriptionAr : axis.descriptionEn}
                </p>

                {/* Categories Preview */}
                <div className="flex flex-wrap gap-2">
                  {axis.categories.slice(0, 3).map((cat) => (
                    <span
                      key={cat.id}
                      className={`px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700`}
                    >
                      {language === "ar" ? cat.nameAr : cat.nameEn}
                    </span>
                  ))}
                  {axis.categories.length > 3 && (
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-500">
                      +{axis.categories.length - 3}
                    </span>
                  )}
                </div>
              </div>

              {/* Expanded Categories */}
              <AnimatePresence>
                {selectedAxis?.id === axis.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-gray-100"
                  >
                    <div className="p-5 bg-gray-50">
                      <h4 className="font-bold text-gray-800 mb-3">
                        {language === "ar" ? "الفئات" : "Categories"}
                      </h4>
                      <div className="space-y-3">
                        {axis.categories.map((cat) => (
                          <div
                            key={cat.id}
                            className="bg-white rounded-xl p-4 border border-gray-200 hover:border-emerald-300 transition-colors"
                          >
                            <div className="flex items-center gap-3 mb-2">
                              <div className={`p-2 rounded-lg bg-gradient-to-br ${axis.gradient}`}>
                                <cat.icon className="w-4 h-4 text-white" />
                              </div>
                              <div>
                                <h5 className="font-semibold text-gray-800">
                                  {language === "ar" ? cat.nameAr : cat.nameEn}
                                </h5>
                              </div>
                            </div>
                            {cat.examples && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {cat.examples.map((ex, i) => (
                                  <span
                                    key={i}
                                    className="px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded text-xs"
                                  >
                                    {ex}
                                  </span>
                                ))}
                              </div>
                            )}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                window.location.href = `/proof?q=${encodeURIComponent(language === "ar" ? cat.nameAr : cat.nameEn)}`;
                              }}
                              className="mt-3 flex items-center gap-1 text-sm text-emerald-600 hover:text-emerald-700 font-medium"
                            >
                              <Search className="w-3 h-3" />
                              {language === "ar" ? "استكشف في النظام" : "Explore in System"}
                              <ArrowRight className="w-3 h-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>

        {/* Methodology Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-12 bg-gradient-to-r from-emerald-50 to-white rounded-2xl p-8 border border-emerald-100"
        >
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-emerald-600" />
            {language === "ar" ? "المنهجية العلمية" : "Scientific Methodology"}
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white rounded-xl p-5 shadow-sm">
              <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center mb-3">
                <span className="text-xl">📖</span>
              </div>
              <h3 className="font-bold text-gray-800 mb-2">
                {language === "ar" ? "المصادر الأصلية" : "Primary Sources"}
              </h3>
              <p className="text-sm text-gray-600">
                {language === "ar"
                  ? "القرآن الكريم مع خمسة تفاسير كلاسيكية: ابن كثير، الطبري، القرطبي، السعدي، الجلالين"
                  : "Holy Quran with five classical tafsirs: Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn"}
              </p>
            </div>
            <div className="bg-white rounded-xl p-5 shadow-sm">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-3">
                <span className="text-xl">🔬</span>
              </div>
              <h3 className="font-bold text-gray-800 mb-2">
                {language === "ar" ? "التحليل الحاسوبي" : "Computational Analysis"}
              </h3>
              <p className="text-sm text-gray-600">
                {language === "ar"
                  ? "معالجة اللغة الطبيعية والتعلم الآلي لاستخراج وتصنيف السلوكيات"
                  : "NLP and machine learning for behavior extraction and classification"}
              </p>
            </div>
            <div className="bg-white rounded-xl p-5 shadow-sm">
              <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mb-3">
                <span className="text-xl">✅</span>
              </div>
              <h3 className="font-bold text-gray-800 mb-2">
                {language === "ar" ? "التحقق العلمي" : "Scholarly Validation"}
              </h3>
              <p className="text-sm text-gray-600">
                {language === "ar"
                  ? "مراجعة وتحقق من قبل علماء متخصصين في الدراسات القرآنية"
                  : "Review and validation by scholars specializing in Quranic studies"}
              </p>
            </div>
          </div>
        </motion.div>

        {/* Stats Footer */}
        {stats && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
            className="mt-8 text-center text-sm text-gray-500"
          >
            <p>
              {language === "ar"
                ? `تم تحليل ${stats.total_spans?.toLocaleString() || 0} تعليق سلوكي عبر ${stats.unique_ayat?.toLocaleString() || 0} آية`
                : `Analyzed ${stats.total_spans?.toLocaleString() || 0} behavioral annotations across ${stats.unique_ayat?.toLocaleString() || 0} ayat`}
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
}
