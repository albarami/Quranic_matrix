"use client";

import { useState, useEffect } from "react";
import { BookOpen, ExternalLink } from "lucide-react";

const TAFSIR_SOURCES = [
  { id: "ibn_kathir", name: "Ibn Kathir", ar: "ابن كثير", color: "emerald" },
  { id: "tabari", name: "Tabari", ar: "الطبري", color: "blue" },
  { id: "qurtubi", name: "Qurtubi", ar: "القرطبي", color: "purple" },
  { id: "saadi", name: "Sa'di", ar: "السعدي", color: "amber" },
  { id: "jalalayn", name: "Jalalayn", ar: "الجلالين", color: "red" },
  { id: "baghawi", name: "Baghawi", ar: "البغوي", color: "cyan" },
  { id: "muyassar", name: "Muyassar", ar: "الميسر", color: "rose" }
];

// Sample tafsir data for demo
const SAMPLE_TAFSIR: Record<string, Record<string, string>> = {
  "2:255": {
    ibn_kathir: `قوله تعالى: {الله لا إله إلا هو} أي: المعبود بحق، لا معبود سواه في السماوات والأرض. {الحي القيوم} أي: الحي في نفسه الذي لا يموت أبداً، القيوم: القائم بنفسه المقيم لغيره، فجميع الموجودات مفتقرة إليه وهو غني عنها، لا قوام لها بدون أمره كما قال تعالى: {ومن آياته أن تقوم السماء والأرض بأمره}.

{لا تأخذه سنة ولا نوم} السنة: النعاس، قال ابن عباس: السنة: النعاس والنوم هو النوم.`,
    tabari: `يعني تعالى ذكره: الله الذي لا تصلح العبادة إلا له المستوجب على خلقه الألوهية والعبودية. {الحي} الذي له الحياة الدائمة والبقاء، الذي لا أول لوجوده ولا آخر، الحي الذي لا يموت. {القيوم} القائم على كل شيء، الحافظ لكل شيء.`,
    qurtubi: `قوله تعالى: {الله لا إله إلا هو الحي القيوم} هذه آية الكرسي سيدة آي القرآن. ثبت في الصحيح عن أبي بن كعب أن رسول الله صلى الله عليه وسلم سأله: أي آية في كتاب الله أعظم؟ قال: الله ورسوله أعلم. فرددها مراراً، ثم قال: آية الكرسي. قال: "ليهنك العلم أبا المنذر".`,
    saadi: `{اللَّهُ لَا إِلَهَ إِلَّا هُوَ} أي: المألوه المعبود، الذي لا تنبغي العبادة إلا له؛ لأنه المتفرد بالكمال من جميع الوجوه، الذي له الأسماء الحسنى، والصفات الكاملة العليا.`,
    jalalayn: `{الله لا إله} أي لا معبود بحق في الوجود {إلا هو الحي} الدائم البقاء {القيوم} القائم بتدبير خلقه {لا تأخذه سنة} نعاس {ولا نوم} لأنهما نقص`,
    baghawi: `قوله تعالى: {اللَّهُ لَا إِلَهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ} الحي: الباقي الذي لا يموت، والقيوم: القائم بتدبير الخلق وحفظهم، والقيوم على وزن فيعول من القيام.`,
    muyassar: `الله الذي لا معبود بحق سواه، الحي الذي له الحياة الكاملة، القيوم القائم على كل شيء، لا يأخذه نعاس ولا نوم، له كل ما في السماوات والأرض، لا يشفع أحد عنده إلا بإذنه.`
  },
  "1:1": {
    ibn_kathir: `{بسم الله} أي: أبدأ بتسمية الله وذكره قبل كل شيء، {الرحمن الرحيم} اسمان مشتقان من الرحمة.`,
    tabari: `قال أبو جعفر: معنى {بسم الله} بالله، وباسم الله أبدأ.`,
    qurtubi: `البسملة آية من الفاتحة على قول الجمهور، وهي أول كل سورة عند الشافعي.`,
    saadi: `أي: أبتدئ بكل اسم لله تعالى، لأن لفظ {اسم} مفرد مضاف، فيعم جميع الأسماء الحسنى.`,
    jalalayn: `{بسم الله الرحمن الرحيم} أي أبتدئ بتسمية الله، أي أتبرك باسمه في ابتداء قراءتي.`,
    baghawi: `افتتح الله كتابه بالبسملة، وهي آية كاملة عند الشافعي.`,
    muyassar: `أبتدئ قراءة القرآن باسم الله مستعينًا به، الرحمن الرحيم صفتان من صفات الله تعالى.`
  }
};

interface TafsirPanelProps {
  surah: number;
  ayah: number;
  active: string;
  onTabChange: (id: string) => void;
  language: string;
}

export function TafsirPanel({ surah, ayah, active, onTabChange, language }: TafsirPanelProps) {
  const [tafsirText, setTafsirText] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const isRTL = language === "ar";

  useEffect(() => {
    setLoading(true);
    const key = `${surah}:${ayah}`;

    // Check sample data first
    if (SAMPLE_TAFSIR[key] && SAMPLE_TAFSIR[key][active]) {
      setTafsirText(SAMPLE_TAFSIR[key][active]);
      setLoading(false);
    } else {
      // Simulate API call
      setTimeout(() => {
        const activeTafsir = TAFSIR_SOURCES.find(t => t.id === active);
        setTafsirText(
          isRTL
            ? `[تفسير ${activeTafsir?.ar} للآية ${surah}:${ayah} - سيُحمل من الواجهة الخلفية]`
            : `[${activeTafsir?.name} tafsir for ${surah}:${ayah} - Will be loaded from backend]`
        );
        setLoading(false);
      }, 300);
    }
  }, [surah, ayah, active, isRTL]);

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-900/50 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-blue-400" />
          <h2 className="font-semibold text-white">
            {isRTL ? "لوحة التفسير" : "Tafsir Panel"}
          </h2>
          <span className="text-slate-500 text-sm">
            (7 {isRTL ? "مصادر" : "sources"})
          </span>
        </div>
        <span className="text-slate-400 text-sm font-mono">{surah}:{ayah}</span>
      </div>

      {/* Tabs */}
      <div className="flex overflow-x-auto border-b border-slate-700 bg-slate-900/30">
        {TAFSIR_SOURCES.map((source) => (
          <button
            key={source.id}
            onClick={() => onTabChange(source.id)}
            className={`px-4 py-3 text-sm whitespace-nowrap transition-colors flex-shrink-0 ${
              active === source.id
                ? "bg-emerald-600 text-white"
                : "text-slate-400 hover:text-white hover:bg-slate-700"
            }`}
          >
            <span className="font-arabic">{source.ar}</span>
            <span className="ml-2 text-xs opacity-70">({source.name})</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div
        className={`p-6 h-48 overflow-y-auto text-lg font-arabic leading-relaxed text-slate-200 ${loading ? 'animate-pulse' : ''}`}
        dir="rtl"
      >
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <span className="text-slate-500">{isRTL ? "جاري التحميل..." : "Loading..."}</span>
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{tafsirText}</p>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 bg-slate-900/50 border-t border-slate-700 flex items-center justify-between">
        <span className="text-xs text-slate-400">
          {isRTL ? "المصدر:" : "Source:"} {TAFSIR_SOURCES.find(s => s.id === active)?.name} — {surah}:{ayah}
        </span>
        <button className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1">
          <ExternalLink className="w-3 h-3" />
          {isRTL ? "عرض المزيد" : "View Full"}
        </button>
      </div>
    </div>
  );
}

export { TAFSIR_SOURCES };
