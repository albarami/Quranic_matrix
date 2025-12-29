"use client";

import { MetricsDashboard } from "../components/MetricsDashboard";
import { useLanguage } from "../contexts/LanguageContext";
import Link from "next/link";
import { ArrowLeft, Database } from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000";

export default function MetricsPage() {
  const { language } = useLanguage();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-gray-600 hover:text-emerald-600 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>{language === "ar" ? "الرئيسية" : "Home"}</span>
              </Link>
              <div className="h-6 w-px bg-gray-300" />
              <div className="flex items-center gap-2">
                <Database className="w-6 h-6 text-emerald-600" />
                <h1 className="text-xl font-bold text-gray-800">
                  {language === "ar" ? "مقاييس الحقيقة" : "Truth Metrics"}
                </h1>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-800">
            {language === "ar" ? "إحصائيات QBM الرسمية" : "Official QBM Statistics"}
          </h2>
          <p className="text-gray-600 mt-1">
            {language === "ar"
              ? "جميع الأرقام محسوبة من البيانات الأساسية - لا تخمينات، لا قيم مشفرة"
              : "All numbers computed from canonical data - no guesses, no hardcoded values"}
          </p>
        </div>

        <MetricsDashboard apiBaseUrl={BACKEND_URL} />
      </main>
    </div>
  );
}
