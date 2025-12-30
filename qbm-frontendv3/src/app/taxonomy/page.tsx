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
  MapPin,
  Clock,
  UserCircle,
  Zap,
  Shield,
  Award,
  Link2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Building,
} from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

const CHART_COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#14b8a6', '#ec4899', '#6366f1'];

// Bouzidani's Complete 11-Axis Taxonomy Structure
interface TaxonomyAxis {
  id: string;
  axisNumber: number;
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

// Complete 11-Axis Bouzidani Framework from canonical_entities.json
const TAXONOMY_AXES: TaxonomyAxis[] = [
  {
    id: "organic",
    axisNumber: 1,
    nameAr: "التصنيف العضوي",
    nameEn: "Organic Classification",
    descriptionAr: "تصنيف السلوكيات حسب الأعضاء الجسدية",
    descriptionEn: "Classification by bodily organs",
    icon: Heart,
    color: "text-rose-600",
    gradient: "from-rose-500 to-pink-600",
    categories: [
      { id: "external", nameAr: "ظاهري", nameEn: "External", icon: Hand },
      { id: "internal", nameAr: "باطني", nameEn: "Internal", icon: Brain },
      { id: "heart", nameAr: "قلبي", nameEn: "Heart-based", icon: Heart },
      { id: "tongue", nameAr: "لساني", nameEn: "Tongue-based", icon: MessageCircle },
      { id: "limbs", nameAr: "جوارحي", nameEn: "Limbs-based", icon: Hand },
    ],
  },
  {
    id: "situational",
    axisNumber: 2,
    nameAr: "التصنيف الموقفي",
    nameEn: "Situational Classification",
    descriptionAr: "تصنيف السلوكيات حسب السياق",
    descriptionEn: "Classification by context",
    icon: Globe,
    color: "text-blue-600",
    gradient: "from-blue-500 to-cyan-600",
    categories: [
      { id: "self", nameAr: "النفس", nameEn: "Self", icon: UserCircle },
      { id: "horizons", nameAr: "الآفاق", nameEn: "Horizons", icon: Globe },
      { id: "creator", nameAr: "الخالق", nameEn: "Creator", icon: Sparkles },
      { id: "universe", nameAr: "الكون", nameEn: "Universe", icon: Sun },
      { id: "life", nameAr: "الحياة", nameEn: "Life", icon: Layers },
    ],
  },
  {
    id: "systemic",
    axisNumber: 3,
    nameAr: "التصنيف النسقي",
    nameEn: "Systemic Classification",
    descriptionAr: "تصنيف السلوكيات حسب النظام الاجتماعي",
    descriptionEn: "Classification by social system",
    icon: Users,
    color: "text-purple-600",
    gradient: "from-purple-500 to-indigo-600",
    categories: [
      { id: "home", nameAr: "البيت", nameEn: "Home", icon: Home },
      { id: "work", nameAr: "العمل", nameEn: "Work", icon: Briefcase },
      { id: "public", nameAr: "مكان عام", nameEn: "Public", icon: Users },
      { id: "mosque", nameAr: "المسجد", nameEn: "Mosque", icon: Building },
    ],
  },
  {
    id: "spatial",
    axisNumber: 4,
    nameAr: "التصنيف المكاني",
    nameEn: "Spatial Classification",
    descriptionAr: "تصنيف السلوكيات حسب المكان",
    descriptionEn: "Classification by place type",
    icon: MapPin,
    color: "text-cyan-600",
    gradient: "from-cyan-500 to-teal-600",
    categories: [
      { id: "sacred", nameAr: "مقدس", nameEn: "Sacred", icon: Sparkles },
      { id: "ordinary", nameAr: "عادي", nameEn: "Ordinary", icon: Home },
      { id: "private", nameAr: "خاص", nameEn: "Private", icon: Shield },
      { id: "public", nameAr: "عام", nameEn: "Public", icon: Globe },
    ],
  },
  {
    id: "temporal",
    axisNumber: 5,
    nameAr: "التصنيف الزماني",
    nameEn: "Temporal Classification",
    descriptionAr: "تصنيف السلوكيات حسب الزمن",
    descriptionEn: "Classification by time dimension",
    icon: Clock,
    color: "text-amber-600",
    gradient: "from-amber-500 to-orange-600",
    categories: [
      { id: "dunya", nameAr: "الدنيا", nameEn: "This World", icon: Sun },
      { id: "akhira", nameAr: "الآخرة", nameEn: "Hereafter", icon: Moon },
      { id: "both", nameAr: "كلاهما", nameEn: "Both", icon: Clock },
    ],
  },
  {
    id: "agent",
    axisNumber: 6,
    nameAr: "التصنيف الفاعلي",
    nameEn: "Agent Classification",
    descriptionAr: "تصنيف السلوكيات حسب الفاعل",
    descriptionEn: "Classification by actor type",
    icon: UserCircle,
    color: "text-indigo-600",
    gradient: "from-indigo-500 to-violet-600",
    categories: [
      { id: "individual", nameAr: "فردي", nameEn: "Individual", icon: UserCircle },
      { id: "collective", nameAr: "جماعي", nameEn: "Collective", icon: Users },
      { id: "divine", nameAr: "إلهي", nameEn: "Divine", icon: Sparkles },
    ],
  },
  {
    id: "source",
    axisNumber: 7,
    nameAr: "التصنيف المصدري",
    nameEn: "Source Classification",
    descriptionAr: "تصنيف السلوكيات حسب المصدر",
    descriptionEn: "Classification by behavior source",
    icon: Zap,
    color: "text-yellow-600",
    gradient: "from-yellow-500 to-amber-600",
    categories: [
      { id: "nafs", nameAr: "النفس", nameEn: "Self/Nafs", icon: Heart },
      { id: "shaytan", nameAr: "الشيطان", nameEn: "Satan", icon: AlertTriangle },
      { id: "divine", nameAr: "إلهي", nameEn: "Divine Inspiration", icon: Sparkles },
    ],
  },
  {
    id: "evaluation",
    axisNumber: 8,
    nameAr: "التصنيف التقييمي",
    nameEn: "Evaluation Classification",
    descriptionAr: "تصنيف السلوكيات حسب الحكم الشرعي",
    descriptionEn: "Classification by religious ruling",
    icon: Scale,
    color: "text-emerald-600",
    gradient: "from-emerald-500 to-teal-600",
    categories: [
      { id: "wajib", nameAr: "واجب", nameEn: "Obligatory", icon: CheckCircle },
      { id: "mustahab", nameAr: "مستحب", nameEn: "Recommended", icon: ThumbsUp },
      { id: "mubah", nameAr: "مباح", nameEn: "Permissible", icon: Scale },
      { id: "makruh", nameAr: "مكروه", nameEn: "Disliked", icon: ThumbsDown },
      { id: "haram", nameAr: "حرام", nameEn: "Forbidden", icon: XCircle },
    ],
  },
  {
    id: "heart_impact",
    axisNumber: 9,
    nameAr: "تأثير القلب",
    nameEn: "Heart Impact",
    descriptionAr: "تأثير السلوك على القلب",
    descriptionEn: "Impact of behavior on the heart",
    icon: Heart,
    color: "text-pink-600",
    gradient: "from-pink-500 to-rose-600",
    categories: [
      { id: "purifies", nameAr: "يزكي", nameEn: "Purifies", icon: Sparkles },
      { id: "hardens", nameAr: "يقسي", nameEn: "Hardens", icon: Shield },
      { id: "softens", nameAr: "يلين", nameEn: "Softens", icon: Heart },
      { id: "seals", nameAr: "يختم", nameEn: "Seals", icon: XCircle },
    ],
  },
  {
    id: "consequence",
    axisNumber: 10,
    nameAr: "العاقبة",
    nameEn: "Consequence",
    descriptionAr: "عواقب السلوك",
    descriptionEn: "Outcomes of behavior",
    icon: Award,
    color: "text-orange-600",
    gradient: "from-orange-500 to-red-600",
    categories: [
      { id: "reward", nameAr: "ثواب", nameEn: "Reward", icon: Award },
      { id: "punishment", nameAr: "عقاب", nameEn: "Punishment", icon: AlertTriangle },
      { id: "neutral", nameAr: "محايد", nameEn: "Neutral", icon: Scale },
    ],
  },
  {
    id: "relationships",
    axisNumber: 11,
    nameAr: "العلاقات",
    nameEn: "Relationships",
    descriptionAr: "علاقات السلوك بالسلوكيات الأخرى",
    descriptionEn: "Behavior relationships",
    icon: Link2,
    color: "text-violet-600",
    gradient: "from-violet-500 to-purple-600",
    categories: [
      { id: "causal", nameAr: "سببية", nameEn: "Causal", icon: ArrowRight },
      { id: "opposite", nameAr: "تضادية", nameEn: "Opposite", icon: XCircle },
      { id: "strengthens", nameAr: "تعزيزية", nameEn: "Strengthening", icon: ThumbsUp },
      { id: "conditional", nameAr: "شرطية", nameEn: "Conditional", icon: AlertTriangle },
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

  // Prepare radar chart data for 11-axis taxonomy overview
  // Static example values representing typical behavior distribution across axes
  const AXIS_EXAMPLE_VALUES: Record<string, number> = {
    organic: 85,
    situational: 72,
    systemic: 65,
    spatial: 58,
    temporal: 90,
    agent: 78,
    source: 45,
    evaluation: 95,
    heart_impact: 82,
    consequence: 88,
    relationships: 70,
  };

  const radarData = TAXONOMY_AXES.map((axis) => ({
    axis: language === "ar" ? axis.nameAr.replace("التصنيف ", "").replace("تأثير ", "") : axis.nameEn.split(" ")[0],
    axisAr: axis.nameAr,
    value: AXIS_EXAMPLE_VALUES[axis.id] || 50,
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
              ? "إطار علمي شامل لتصنيف السلوكيات البشرية في القرآن الكريم عبر إحدى عشر محوراً: العضوي، الموقفي، النسقي، المكاني، الزماني، الفاعلي، المصدري، التقييمي، تأثير القلب، العاقبة، والعلاقات."
              : "A comprehensive scholarly framework for classifying human behaviors in the Holy Quran across 11 axes: Organic, Situational, Systemic, Spatial, Temporal, Agent, Source, Evaluation, Heart Impact, Consequence, and Relationships."}
          </p>

          {/* Quick Stats - 11 Axes Grid */}
          <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-11 gap-3 mt-8">
            {TAXONOMY_AXES.map((axis, i) => (
              <motion.button
                key={axis.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => setSelectedAxis(axis)}
                className={`bg-white/10 backdrop-blur rounded-xl p-3 text-center hover:bg-white/20 transition-all ${
                  selectedAxis?.id === axis.id ? 'ring-2 ring-white' : ''
                }`}
              >
                <axis.icon className="w-5 h-5 mx-auto mb-1 text-emerald-300" />
                <div className="text-sm font-bold">{axis.categories.length}</div>
                <div className="text-[10px] text-emerald-200 truncate">
                  {language === "ar" ? axis.nameAr.replace("التصنيف ", "") : axis.nameEn.split(" ")[0]}
                </div>
                <div className="text-[9px] text-emerald-400 font-mono">#{axis.axisNumber}</div>
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
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="65%">
                    <PolarGrid stroke="#e5e7eb" />
                    <PolarAngleAxis
                      dataKey="axis"
                      tick={{ fontSize: 10, fill: '#4b5563' }}
                      tickLine={false}
                    />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 100]}
                      tick={{ fontSize: 9, fill: '#9ca3af' }}
                    />
                    <Radar
                      name={language === "ar" ? "التغطية" : "Coverage"}
                      dataKey="value"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.4}
                      strokeWidth={2}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                    />
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
