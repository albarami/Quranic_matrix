"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Copy, Check, BookOpen, ExternalLink } from "lucide-react";
import { generateQBMCitation, copyToClipboard } from "@/lib/export-utils";
import { useLanguage } from "../../contexts/LanguageContext";

interface CitationPreviewProps {
  surah: number;
  ayah: number;
  behavior: {
    id: string;
    ar: string;
    en: string;
  };
  validationScore?: number;
}

export function CitationPreview({
  surah,
  ayah,
  behavior,
  validationScore,
}: CitationPreviewProps) {
  const { isRTL } = useLanguage();
  const [copied, setCopied] = useState(false);
  const [format, setFormat] = useState<"bibtex" | "apa" | "mla">("bibtex");

  const bibtexCitation = generateQBMCitation(surah, ayah, behavior.en);

  // Generate APA format
  const apaCitation = `Quranic Behavioral Matrix (QBM). (${new Date().getFullYear()}). Behavioral Analysis of Quran ${surah}:${ayah} - ${behavior.en}. QBM Research Platform. https://qbm.research.platform`;

  // Generate MLA format
  const mlaCitation = `"Behavioral Analysis of Quran ${surah}:${ayah} - ${behavior.en}." Quranic Behavioral Matrix (QBM), ${new Date().getFullYear()}, qbm.research.platform.`;

  const citations = {
    bibtex: bibtexCitation,
    apa: apaCitation,
    mla: mlaCitation,
  };

  const handleCopy = async () => {
    const success = await copyToClipboard(citations[format]);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
    >
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-violet-50 to-purple-50 border-b border-violet-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-violet-600" />
            <h3 className="font-semibold text-gray-800">
              {isRTL ? "الاقتباس الأكاديمي" : "Academic Citation"}
            </h3>
          </div>
          {validationScore && (
            <span className="text-xs px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full font-medium">
              {isRTL ? "مُتحقق منه" : "Validated"} {validationScore}%
            </span>
          )}
        </div>
      </div>

      {/* Citation Info */}
      <div className="p-4">
        <div className="flex flex-wrap items-center gap-2 mb-4 text-sm">
          <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded font-medium">
            {isRTL ? "السورة" : "Surah"} {surah}
          </span>
          <span className="px-2 py-1 bg-amber-100 text-amber-700 rounded font-medium">
            {isRTL ? "الآية" : "Ayah"} {ayah}
          </span>
          <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded font-medium">
            {behavior.ar}
          </span>
        </div>

        {/* Format Tabs */}
        <div className="flex border-b border-gray-200 mb-4">
          {(["bibtex", "apa", "mla"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                format === f
                  ? "text-violet-600 border-b-2 border-violet-600 bg-violet-50"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
              }`}
            >
              {f.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Citation Text */}
        <div className="relative">
          <pre
            className="p-4 bg-gray-50 rounded-lg text-sm font-mono text-gray-700 overflow-x-auto whitespace-pre-wrap border border-gray-200"
            dir="ltr"
          >
            {citations[format]}
          </pre>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className={`absolute top-2 ${isRTL ? "left-2" : "right-2"} p-2 rounded-lg transition-all ${
              copied
                ? "bg-emerald-100 text-emerald-600"
                : "bg-white text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            } border border-gray-200 shadow-sm`}
          >
            {copied ? (
              <Check className="w-4 h-4" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Copy Status */}
        {copied && (
          <motion.p
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-emerald-600 mt-2 text-center font-medium"
          >
            {isRTL ? "تم نسخ الاقتباس!" : "Citation copied to clipboard!"}
          </motion.p>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>
            {isRTL
              ? "QBM مصفوفة السلوك القرآني"
              : "Quranic Behavioral Matrix (QBM)"}
          </span>
          <a
            href="#"
            className="flex items-center gap-1 text-violet-600 hover:text-violet-700"
          >
            {isRTL ? "دليل الاقتباس" : "Citation Guide"}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>
    </motion.div>
  );
}
