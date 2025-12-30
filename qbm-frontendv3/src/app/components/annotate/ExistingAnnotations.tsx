"use client";

import { useState } from "react";
import { Edit2, Trash2, ChevronDown, ChevronUp, Sparkles, Users, Hand, Award, Heart, AlertCircle, RefreshCw, Loader2 } from "lucide-react";
import { AXES_11 } from "./AnnotationForm";
import { useAnnotations } from "@/lib/api/hooks";
import type { Annotation as APIAnnotation } from "@/lib/api/types";

// Extended annotation type for compatibility with UI
interface Annotation {
  id: string;
  behavior: { id: string; ar: string; en: string };
  agent?: { id: string; ar: string; en: string };
  organ?: { id: string; ar: string; en: string };
  heart_state?: { id: string; ar: string; en: string };
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
  onDelete?: (id: string) => void;
}

// Transform API annotation to local format
function transformAnnotation(apiAnn: APIAnnotation): Annotation {
  // Convert axes_11 to numbered axes
  const axes: Record<number, string> = {};
  if (apiAnn.axes_11) {
    Object.entries(apiAnn.axes_11).forEach(([key, value]) => {
      // Extract axis number from key (e.g., "axis_1" -> 1)
      const match = key.match(/axis_(\d+)/);
      if (match) {
        axes[parseInt(match[1])] = value;
      }
    });
  }

  return {
    id: apiAnn.id,
    behavior: apiAnn.behavior,
    agent: apiAnn.agent,
    organ: apiAnn.organ,
    heart_state: apiAnn.heart_state,
    consequence: apiAnn.consequence,
    axes,
    notes: apiAnn.notes,
    createdAt: apiAnn.created_at?.split("T")[0], // Format date
  };
}

export function ExistingAnnotations({ surah, ayah, language, onEdit, onDelete }: ExistingAnnotationsProps) {
  const isRTL = language === "ar";
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: annotationsData, isLoading, error, refetch } = useAnnotations(surah, ayah);

  // Transform API annotations to local format
  const annotations: Annotation[] = (annotationsData?.annotations || []).map(transformAnnotation);

  const getAxisLabel = (axisId: number, value: string): string => {
    const axis = AXES_11.find(a => a.id === axisId);
    if (!axis) return value;
    const option = axis.options.find(o => o.value === value);
    return option ? (isRTL ? option.ar : option.en) : value;
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-8 text-center">
        <div className="flex items-center justify-center gap-3 text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>{isRTL ? "جاري تحميل التعليقات..." : "Loading annotations..."}</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-slate-800/50 rounded-xl border border-red-700/50 p-8 text-center">
        <div className="flex flex-col items-center gap-3">
          <AlertCircle className="w-8 h-8 text-red-400" />
          <span className="text-red-400">{isRTL ? "فشل تحميل التعليقات" : "Failed to load annotations"}</span>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-2 px-4 py-2 bg-red-900/30 hover:bg-red-900/50 rounded-lg text-red-400 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            {isRTL ? "إعادة المحاولة" : "Retry"}
          </button>
        </div>
      </div>
    );
  }

  // Empty state
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
                    {ann.agent && (
                      <>
                        <span className="text-slate-500">→</span>
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-900/50 text-blue-300 rounded-lg text-sm">
                          <Users className="w-3.5 h-3.5" />
                          <span className="font-arabic">{ann.agent.ar}</span>
                        </span>
                      </>
                    )}
                    {ann.organ && (
                      <>
                        <span className="text-slate-500">→</span>
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-purple-900/50 text-purple-300 rounded-lg text-sm">
                          <Hand className="w-3.5 h-3.5" />
                          <span className="font-arabic">{ann.organ.ar}</span>
                        </span>
                      </>
                    )}
                    {ann.heart_state && (
                      <>
                        <span className="text-slate-500">→</span>
                        <span className="flex items-center gap-1.5 px-2.5 py-1 bg-pink-900/50 text-pink-300 rounded-lg text-sm">
                          <Heart className="w-3.5 h-3.5" />
                          <span className="font-arabic">{ann.heart_state.ar}</span>
                        </span>
                      </>
                    )}
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
                    {ann.agent && (
                      <>
                        <span>|</span>
                        <span>agent: {ann.agent.id}</span>
                      </>
                    )}
                    {ann.organ && (
                      <>
                        <span>|</span>
                        <span>organ: {ann.organ.id}</span>
                      </>
                    )}
                    {ann.heart_state && (
                      <>
                        <span>|</span>
                        <span>heart_state: {ann.heart_state.id}</span>
                      </>
                    )}
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
