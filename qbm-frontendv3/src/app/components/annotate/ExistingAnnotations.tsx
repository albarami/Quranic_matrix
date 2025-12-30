"use client";

import { useState } from "react";
import { Edit2, Trash2, ChevronDown, ChevronUp, Sparkles, Users, Hand, Award } from "lucide-react";
import { AXES_11 } from "./AnnotationForm";

interface Annotation {
  id: number;
  behavior: { id: string; ar: string; en: string };
  agent: { id: string; ar: string; en: string };
  organ: { id: string; ar: string; en: string };
  consequence?: { id: string; ar: string; en: string };
  axes: Record<number, string>;
  notes?: string;
  createdAt?: string;
}

interface ExistingAnnotationsProps {
  surah: number;
  ayah: number;
  language: string;
  onEdit?: (annotation: Annotation) => void;
  onDelete?: (id: number) => void;
}

// Sample annotations for demo
const SAMPLE_ANNOTATIONS: Record<string, Annotation[]> = {
  "2:255": [
    {
      id: 1,
      behavior: { id: "BEH_SPI_FAITH", ar: "الإيمان", en: "Faith" },
      agent: { id: "AGT_BELIEVER", ar: "المؤمن", en: "Believer" },
      organ: { id: "ORG_HEART", ar: "القلب", en: "Heart" },
      consequence: { id: "CSQ_GUIDANCE", ar: "الهداية", en: "Guidance" },
      axes: { 1: "heart", 5: "both", 8: "wajib", 9: "purifies", 10: "reward" },
      notes: "آية الكرسي - أعظم آية في القرآن",
      createdAt: "2024-12-28"
    },
    {
      id: 2,
      behavior: { id: "BEH_SPI_TAQWA", ar: "التقوى", en: "God-Consciousness" },
      agent: { id: "AGT_BELIEVER", ar: "المؤمن", en: "Believer" },
      organ: { id: "ORG_HEART", ar: "القلب", en: "Heart" },
      axes: { 1: "heart", 8: "mustahab", 9: "purifies" },
      createdAt: "2024-12-28"
    }
  ],
  "1:5": [
    {
      id: 3,
      behavior: { id: "BEH_SPI_PRAYER", ar: "الصلاة", en: "Prayer" },
      agent: { id: "AGT_BELIEVER", ar: "المؤمن", en: "Believer" },
      organ: { id: "ORG_HEART", ar: "القلب", en: "Heart" },
      consequence: { id: "CSQ_GUIDANCE", ar: "الهداية", en: "Guidance" },
      axes: { 1: "heart", 2: "creator", 8: "wajib" },
      createdAt: "2024-12-27"
    }
  ]
};

export function ExistingAnnotations({ surah, ayah, language, onEdit, onDelete }: ExistingAnnotationsProps) {
  const isRTL = language === "ar";
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const key = `${surah}:${ayah}`;
  const annotations = SAMPLE_ANNOTATIONS[key] || [];

  const getAxisLabel = (axisId: number, value: string): string => {
    const axis = AXES_11.find(a => a.id === axisId);
    if (!axis) return value;
    const option = axis.options.find(o => o.value === value);
    return option ? (isRTL ? option.ar : option.en) : value;
  };

  if (annotations.length === 0) {
    return (
      <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-8 text-center">
        <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
          <Sparkles className="w-8 h-8 text-slate-500" />
        </div>
        <h3 className="text-lg font-medium text-slate-300 mb-2">
          {isRTL ? `لا توجد تعليقات بعد للآية ${surah}:${ayah}` : `No annotations yet for ${surah}:${ayah}`}
        </h3>
        <p className="text-slate-500 text-sm">
          {isRTL ? "أنشئ أول تعليق توضيحي أعلاه." : "Create the first annotation above."}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-2 h-8 rounded-full bg-blue-500"></div>
        <h2 className="text-xl font-semibold text-white">
          {isRTL ? "التعليقات الموجودة" : "Existing Annotations"}
        </h2>
        <span className="px-2 py-0.5 bg-blue-900/50 text-blue-300 rounded-full text-sm">
          {annotations.length}
        </span>
      </div>

      {/* Annotations List */}
      <div className="space-y-4">
        {annotations.map((ann, idx) => (
          <div
            key={ann.id}
            className="bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden"
          >
            {/* Main Row */}
            <div
              className="p-4 cursor-pointer hover:bg-slate-700/30 transition-colors"
              onClick={() => setExpandedId(expandedId === ann.id ? null : ann.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-slate-500 text-sm font-mono">#{idx + 1}</span>
                    {ann.createdAt && (
                      <span className="text-slate-500 text-xs">{ann.createdAt}</span>
                    )}
                  </div>

                  {/* Entity Chain */}
                  <div className="flex flex-wrap items-center gap-2" dir={isRTL ? "rtl" : "ltr"}>
                    <span className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-900/50 text-emerald-300 rounded-lg text-sm">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span className="font-arabic">{ann.behavior.ar}</span>
                    </span>
                    <span className="text-slate-500">→</span>
                    <span className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-900/50 text-blue-300 rounded-lg text-sm">
                      <Users className="w-3.5 h-3.5" />
                      <span className="font-arabic">{ann.agent.ar}</span>
                    </span>
                    <span className="text-slate-500">→</span>
                    <span className="flex items-center gap-1.5 px-2.5 py-1 bg-purple-900/50 text-purple-300 rounded-lg text-sm">
                      <Hand className="w-3.5 h-3.5" />
                      <span className="font-arabic">{ann.organ.ar}</span>
                    </span>
                    {ann.consequence && (
                      <>
                        <span className="text-slate-500">→</span>
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-900/50 text-amber-300 rounded-lg text-sm">
                          <Award className="w-3.5 h-3.5" />
                          <span className="font-arabic">{ann.consequence.ar}</span>
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onEdit) onEdit(ann);
                    }}
                    className="p-2 text-slate-400 hover:text-white hover:bg-slate-600 rounded transition-colors"
                    title={isRTL ? "تعديل" : "Edit"}
                  >
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onDelete && confirm(isRTL ? "هل أنت متأكد من الحذف؟" : "Are you sure you want to delete?")) {
                        onDelete(ann.id);
                      }
                    }}
                    className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-900/30 rounded transition-colors"
                    title={isRTL ? "حذف" : "Delete"}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  {expandedId === ann.id ? (
                    <ChevronUp className="w-5 h-5 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-slate-400" />
                  )}
                </div>
              </div>
            </div>

            {/* Expanded Details */}
            {expandedId === ann.id && (
              <div className="px-4 pb-4 border-t border-slate-700 pt-4 bg-slate-900/30">
                {/* Axes */}
                {Object.keys(ann.axes).length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-xs text-slate-500 mb-2 uppercase tracking-wide">
                      {isRTL ? "تصنيف المحاور" : "Axis Classification"}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(ann.axes).map(([axisId, value]) => {
                        const axis = AXES_11.find(a => a.id === Number(axisId));
                        return (
                          <span
                            key={axisId}
                            className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs"
                          >
                            <span className="text-slate-500">#{axisId}</span>{" "}
                            {axis ? (isRTL ? axis.ar : axis.en) : `Axis ${axisId}`}:{" "}
                            <span className="text-white">{getAxisLabel(Number(axisId), value)}</span>
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Notes */}
                {ann.notes && (
                  <div>
                    <h4 className="text-xs text-slate-500 mb-2 uppercase tracking-wide">
                      {isRTL ? "ملاحظات" : "Notes"}
                    </h4>
                    <p className="text-slate-300 text-sm font-arabic" dir="rtl">{ann.notes}</p>
                  </div>
                )}

                {/* IDs for debugging */}
                <div className="mt-4 pt-4 border-t border-slate-700">
                  <div className="flex flex-wrap gap-2 text-xs text-slate-500 font-mono">
                    <span>behavior: {ann.behavior.id}</span>
                    <span>|</span>
                    <span>agent: {ann.agent.id}</span>
                    <span>|</span>
                    <span>organ: {ann.organ.id}</span>
                    {ann.consequence && (
                      <>
                        <span>|</span>
                        <span>consequence: {ann.consequence.id}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
