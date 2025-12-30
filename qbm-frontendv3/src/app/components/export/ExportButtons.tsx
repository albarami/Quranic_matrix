"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Download,
  FileText,
  FileJson,
  Printer,
  Copy,
  Check,
  ChevronDown,
  BookOpen,
} from "lucide-react";
import {
  generateQBMCitation,
  downloadBibTeX,
  downloadJSON,
  generateProofMarkdown,
  downloadMarkdown,
  printProofPage,
  copyToClipboard,
  ProofExportData,
} from "@/lib/export-utils";
import { useLanguage } from "../../contexts/LanguageContext";

interface ExportButtonsProps {
  proofData?: ProofExportData;
  disabled?: boolean;
}

type ExportFormat = "bibtex" | "json" | "markdown" | "print";

export function ExportButtons({ proofData, disabled = false }: ExportButtonsProps) {
  const { isRTL } = useLanguage();
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [lastExport, setLastExport] = useState<ExportFormat | null>(null);

  const handleExport = async (format: ExportFormat) => {
    if (!proofData) return;

    setLastExport(format);

    switch (format) {
      case "bibtex":
        const bibtex = generateQBMCitation(
          proofData.surah,
          proofData.ayah,
          proofData.behavior.en
        );
        downloadBibTeX(
          bibtex,
          `qbm_${proofData.surah}_${proofData.ayah}.bib`
        );
        break;

      case "json":
        downloadJSON(
          proofData,
          `qbm_proof_${proofData.surah}_${proofData.ayah}.json`
        );
        break;

      case "markdown":
        const markdown = generateProofMarkdown(proofData);
        downloadMarkdown(
          markdown,
          `qbm_proof_${proofData.surah}_${proofData.ayah}.md`
        );
        break;

      case "print":
        printProofPage();
        break;
    }

    setIsOpen(false);
  };

  const handleCopyBibTeX = async () => {
    if (!proofData) return;

    const bibtex = generateQBMCitation(
      proofData.surah,
      proofData.ayah,
      proofData.behavior.en
    );

    const success = await copyToClipboard(bibtex);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const exportOptions = [
    {
      id: "bibtex" as ExportFormat,
      icon: BookOpen,
      label: isRTL ? "BibTeX للمراجع" : "BibTeX Citation",
      description: isRTL ? "للأوراق الأكاديمية" : "For academic papers",
      color: "text-violet-500",
      bgColor: "bg-violet-50 hover:bg-violet-100",
    },
    {
      id: "json" as ExportFormat,
      icon: FileJson,
      label: isRTL ? "JSON بيانات" : "JSON Data",
      description: isRTL ? "بيانات منظمة" : "Structured data",
      color: "text-amber-500",
      bgColor: "bg-amber-50 hover:bg-amber-100",
    },
    {
      id: "markdown" as ExportFormat,
      icon: FileText,
      label: isRTL ? "Markdown تقرير" : "Markdown Report",
      description: isRTL ? "تقرير قابل للقراءة" : "Readable report",
      color: "text-blue-500",
      bgColor: "bg-blue-50 hover:bg-blue-100",
    },
    {
      id: "print" as ExportFormat,
      icon: Printer,
      label: isRTL ? "طباعة / PDF" : "Print / PDF",
      description: isRTL ? "حفظ كـ PDF" : "Save as PDF",
      color: "text-emerald-500",
      bgColor: "bg-emerald-50 hover:bg-emerald-100",
    },
  ];

  return (
    <div className="relative inline-block">
      <div className="flex items-center gap-2">
        {/* Main Export Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          disabled={disabled || !proofData}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all ${
            disabled || !proofData
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-emerald-600 text-white hover:bg-emerald-700 shadow-md hover:shadow-lg"
          }`}
        >
          <Download className="w-4 h-4" />
          <span>{isRTL ? "تصدير" : "Export"}</span>
          <ChevronDown
            className={`w-4 h-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
          />
        </button>

        {/* Quick Copy BibTeX Button */}
        <button
          onClick={handleCopyBibTeX}
          disabled={disabled || !proofData}
          className={`flex items-center gap-2 px-3 py-2.5 rounded-lg transition-all ${
            disabled || !proofData
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-violet-100 text-violet-700 hover:bg-violet-200"
          }`}
          title={isRTL ? "نسخ BibTeX" : "Copy BibTeX"}
        >
          {copied ? (
            <>
              <Check className="w-4 h-4 text-emerald-600" />
              <span className="text-emerald-600 text-sm">
                {isRTL ? "تم النسخ!" : "Copied!"}
              </span>
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              <span className="text-sm">BibTeX</span>
            </>
          )}
        </button>
      </div>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Menu */}
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className={`absolute ${isRTL ? "left-0" : "right-0"} mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-200 z-50 overflow-hidden`}
            >
              <div className="p-2">
                <div className="text-xs text-gray-500 px-3 py-2 font-medium uppercase tracking-wider">
                  {isRTL ? "صيغ التصدير" : "Export Formats"}
                </div>

                {exportOptions.map((option) => {
                  const Icon = option.icon;
                  return (
                    <button
                      key={option.id}
                      onClick={() => handleExport(option.id)}
                      className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors ${option.bgColor}`}
                    >
                      <div className={`p-2 rounded-lg bg-white shadow-sm`}>
                        <Icon className={`w-4 h-4 ${option.color}`} />
                      </div>
                      <div className={`flex-1 ${isRTL ? "text-right" : "text-left"}`}>
                        <div className="font-medium text-gray-800">
                          {option.label}
                        </div>
                        <div className="text-xs text-gray-500">
                          {option.description}
                        </div>
                      </div>
                      {lastExport === option.id && (
                        <Check className="w-4 h-4 text-emerald-500" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Footer */}
              <div className="border-t border-gray-100 px-4 py-3 bg-gray-50">
                <p className="text-xs text-gray-500 text-center">
                  {isRTL
                    ? "جميع التصديرات تتضمن بيانات التحقق"
                    : "All exports include validation data"}
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
