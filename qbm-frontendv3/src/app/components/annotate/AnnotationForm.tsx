"use client";

import { useState } from "react";
import { Save, Trash2, ChevronRight, Sparkles, Users, Hand, Award, Layers } from "lucide-react";
import {
  SAMPLE_BEHAVIORS,
  SAMPLE_AGENTS,
  SAMPLE_ORGANS,
  SAMPLE_CONSEQUENCES,
} from "@/lib/canonical-entities";

// 11-Axis Classification Options
const AXES_11 = [
  {
    id: 1,
    en: "Organic",
    ar: "العضوي",
    options: [
      { value: "external", en: "External", ar: "ظاهري" },
      { value: "internal", en: "Internal", ar: "باطني" },
      { value: "heart", en: "Heart", ar: "قلبي" },
      { value: "tongue", en: "Tongue", ar: "لساني" },
      { value: "limbs", en: "Limbs", ar: "جوارحي" },
    ]
  },
  {
    id: 2,
    en: "Situational",
    ar: "الموقفي",
    options: [
      { value: "self", en: "Self", ar: "النفس" },
      { value: "horizons", en: "Horizons", ar: "الآفاق" },
      { value: "creator", en: "Creator", ar: "الخالق" },
      { value: "universe", en: "Universe", ar: "الكون" },
      { value: "life", en: "Life", ar: "الحياة" },
    ]
  },
  {
    id: 3,
    en: "Systemic",
    ar: "النسقي",
    options: [
      { value: "home", en: "Home", ar: "البيت" },
      { value: "work", en: "Work", ar: "العمل" },
      { value: "public", en: "Public", ar: "المجتمع" },
      { value: "mosque", en: "Mosque", ar: "المسجد" },
    ]
  },
  {
    id: 4,
    en: "Spatial",
    ar: "المكاني",
    options: [
      { value: "sacred", en: "Sacred", ar: "مقدس" },
      { value: "ordinary", en: "Ordinary", ar: "عادي" },
      { value: "private", en: "Private", ar: "خاص" },
      { value: "public", en: "Public", ar: "عام" },
    ]
  },
  {
    id: 5,
    en: "Temporal",
    ar: "الزماني",
    options: [
      { value: "dunya", en: "This World", ar: "الدنيا" },
      { value: "akhira", en: "Hereafter", ar: "الآخرة" },
      { value: "both", en: "Both", ar: "كلاهما" },
    ]
  },
  {
    id: 6,
    en: "Agent",
    ar: "الفاعلي",
    options: [
      { value: "individual", en: "Individual", ar: "فردي" },
      { value: "collective", en: "Collective", ar: "جماعي" },
      { value: "divine", en: "Divine", ar: "إلهي" },
    ]
  },
  {
    id: 7,
    en: "Source",
    ar: "المصدري",
    options: [
      { value: "nafs", en: "Self/Nafs", ar: "النفس" },
      { value: "satan", en: "Satan", ar: "الشيطان" },
      { value: "divine", en: "Divine Inspiration", ar: "إلهام إلهي" },
    ]
  },
  {
    id: 8,
    en: "Evaluation",
    ar: "التقييمي",
    options: [
      { value: "wajib", en: "Obligatory", ar: "واجب" },
      { value: "mustahab", en: "Recommended", ar: "مستحب" },
      { value: "mubah", en: "Permissible", ar: "مباح" },
      { value: "makruh", en: "Disliked", ar: "مكروه" },
      { value: "haram", en: "Forbidden", ar: "حرام" },
    ]
  },
  {
    id: 9,
    en: "Heart Impact",
    ar: "تأثير القلب",
    options: [
      { value: "purifies", en: "Purifies", ar: "يزكي" },
      { value: "hardens", en: "Hardens", ar: "يقسي" },
      { value: "softens", en: "Softens", ar: "يلين" },
      { value: "seals", en: "Seals", ar: "يختم" },
    ]
  },
  {
    id: 10,
    en: "Consequence",
    ar: "العاقبة",
    options: [
      { value: "reward", en: "Reward", ar: "ثواب" },
      { value: "punishment", en: "Punishment", ar: "عقاب" },
      { value: "neutral", en: "Neutral", ar: "محايد" },
    ]
  },
  {
    id: 11,
    en: "Relationships",
    ar: "العلاقات",
    options: [
      { value: "causal", en: "Causal", ar: "سببي" },
      { value: "opposite", en: "Opposite", ar: "متضاد" },
      { value: "strengthening", en: "Strengthening", ar: "معزز" },
      { value: "conditional", en: "Conditional", ar: "شرطي" },
    ]
  }
];

interface AnnotationFormProps {
  surah: number;
  ayah: number;
  language: string;
  onSave?: (annotation: AnnotationData) => void;
  onSkip?: () => void;
}

interface AnnotationData {
  surah: number;
  ayah: number;
  behavior: string;
  agent: string;
  organ: string;
  consequence: string;
  axes: Record<number, string>;
  notes: string;
}

export function AnnotationForm({ surah, ayah, language, onSave, onSkip }: AnnotationFormProps) {
  const isRTL = language === "ar";
  const [annotation, setAnnotation] = useState<AnnotationData>({
    surah,
    ayah,
    behavior: "",
    agent: "",
    organ: "",
    consequence: "",
    axes: {},
    notes: ""
  });

  const handleSave = () => {
    if (!annotation.behavior) {
      alert(isRTL ? "يرجى اختيار سلوك" : "Please select a behavior");
      return;
    }
    console.log("Saving annotation:", annotation);
    if (onSave) onSave(annotation);
    // Clear form after save
    setAnnotation({
      surah,
      ayah,
      behavior: "",
      agent: "",
      organ: "",
      consequence: "",
      axes: {},
      notes: ""
    });
  };

  const handleClear = () => {
    setAnnotation({
      surah,
      ayah,
      behavior: "",
      agent: "",
      organ: "",
      consequence: "",
      axes: {},
      notes: ""
    });
  };

  const handleSkip = () => {
    if (onSkip) onSkip();
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-2 h-8 rounded-full bg-emerald-500"></div>
        <h2 className="text-xl font-semibold text-white">
          {isRTL ? "تعليق توضيحي جديد" : "New Annotation"}
        </h2>
        <span className="text-slate-400 text-sm font-mono ml-auto">{surah}:{ayah}</span>
      </div>

      {/* Primary Selectors */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Behavior */}
        <div>
          <label className="flex items-center gap-2 text-sm text-slate-400 mb-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            {isRTL ? "السلوك" : "Behavior"}
          </label>
          <select
            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            value={annotation.behavior}
            onChange={(e) => setAnnotation({ ...annotation, behavior: e.target.value })}
          >
            <option value="">{isRTL ? "اختر السلوك..." : "Select behavior..."}</option>
            {SAMPLE_BEHAVIORS.map((b) => (
              <option key={b.id} value={b.id}>{b.ar} ({b.en})</option>
            ))}
          </select>
        </div>

        {/* Agent */}
        <div>
          <label className="flex items-center gap-2 text-sm text-slate-400 mb-2">
            <Users className="w-4 h-4 text-blue-400" />
            {isRTL ? "الفاعل" : "Agent"}
          </label>
          <select
            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={annotation.agent}
            onChange={(e) => setAnnotation({ ...annotation, agent: e.target.value })}
          >
            <option value="">{isRTL ? "اختر الفاعل..." : "Select agent..."}</option>
            {SAMPLE_AGENTS.map((a) => (
              <option key={a.id} value={a.id}>{a.ar} ({a.en})</option>
            ))}
          </select>
        </div>

        {/* Organ */}
        <div>
          <label className="flex items-center gap-2 text-sm text-slate-400 mb-2">
            <Hand className="w-4 h-4 text-purple-400" />
            {isRTL ? "العضو" : "Organ"}
          </label>
          <select
            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
            value={annotation.organ}
            onChange={(e) => setAnnotation({ ...annotation, organ: e.target.value })}
          >
            <option value="">{isRTL ? "اختر العضو..." : "Select organ..."}</option>
            {SAMPLE_ORGANS.map((o) => (
              <option key={o.id} value={o.id}>{o.ar} ({o.en})</option>
            ))}
          </select>
        </div>

        {/* Consequence */}
        <div>
          <label className="flex items-center gap-2 text-sm text-slate-400 mb-2">
            <Award className="w-4 h-4 text-amber-400" />
            {isRTL ? "العاقبة" : "Consequence"}
          </label>
          <select
            className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
            value={annotation.consequence}
            onChange={(e) => setAnnotation({ ...annotation, consequence: e.target.value })}
          >
            <option value="">{isRTL ? "اختر العاقبة..." : "Select consequence..."}</option>
            {SAMPLE_CONSEQUENCES.map((c) => (
              <option key={c.id} value={c.id}>{c.ar} ({c.en})</option>
            ))}
          </select>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-700 my-6"></div>

      {/* 11-Axis Classification */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Layers className="w-5 h-5 text-indigo-400" />
          <h3 className="text-lg font-medium text-white">
            {isRTL ? "التصنيف الإحدى عشري" : "11-Axis Classification"}
          </h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
          {AXES_11.map((axis) => (
            <div key={axis.id}>
              <label className="block text-xs text-slate-400 mb-1.5">
                <span className="text-slate-500 font-mono">#{axis.id}</span>{" "}
                <span className="font-arabic">{axis.ar}</span>
                <span className="text-slate-500 ml-1">({axis.en})</span>
              </label>
              <select
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                value={annotation.axes[axis.id] || ""}
                onChange={(e) => setAnnotation({
                  ...annotation,
                  axes: { ...annotation.axes, [axis.id]: e.target.value }
                })}
              >
                <option value="">--</option>
                {axis.options.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {isRTL ? opt.ar : opt.en}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
      </div>

      {/* Notes */}
      <div className="mb-6">
        <label className="block text-sm text-slate-400 mb-2">
          {isRTL ? "ملاحظات علمية" : "Scholarly Notes"}
        </label>
        <textarea
          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white h-24 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500"
          placeholder={isRTL ? "ملاحظات إضافية، مراجع، أو ملاحظات..." : "Additional notes, references, or observations..."}
          value={annotation.notes}
          onChange={(e) => setAnnotation({ ...annotation, notes: e.target.value })}
          dir={isRTL ? "rtl" : "ltr"}
        />
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-slate-700">
        <button
          onClick={handleClear}
          className="flex items-center gap-2 px-4 py-2 text-slate-400 hover:text-white transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          {isRTL ? "مسح النموذج" : "Clear Form"}
        </button>
        <div className="flex gap-3">
          <button
            onClick={handleSkip}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
          >
            {isRTL ? "تخطي إلى الآية التالية" : "Skip to Next Ayah"}
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-6 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 transition-colors"
          >
            <Save className="w-4 h-4" />
            {isRTL ? "حفظ التعليق" : "Save Annotation"}
          </button>
        </div>
      </div>
    </div>
  );
}

export { AXES_11 };
