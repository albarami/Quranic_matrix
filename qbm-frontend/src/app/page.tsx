import Link from "next/link";
import { BookOpen, LayoutDashboard, PenTool, Database, FileText, GitBranch } from "lucide-react";

export default function HomePage() {
  return (
    <main className="max-w-6xl mx-auto px-4 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Quranic Behavioral Matrix
        </h1>
        <p className="text-xl text-gray-600 mb-2">
          مصفوفة التصنيف القرآني للسلوك البشري
        </p>
        <p className="text-gray-500 max-w-2xl mx-auto">
          A structured dataset of Quranic behavioral classifications grounded in 
          Islamic scholarship. Explore behaviors, contexts, and tafsir across all 
          6,236 ayat of the Holy Quran.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {[
          { label: "Ayat", value: "6,236", sub: "Complete Quran" },
          { label: "Spans", value: "15,000+", sub: "Behavioral annotations" },
          { label: "Tafsir", value: "5", sub: "Classical sources" },
          { label: "Concepts", value: "80+", sub: "Behavior types" },
        ].map((stat) => (
          <div
            key={stat.label}
            className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 text-center"
          >
            <div className="text-3xl font-bold text-emerald-600">{stat.value}</div>
            <div className="text-gray-900 font-medium">{stat.label}</div>
            <div className="text-gray-500 text-sm">{stat.sub}</div>
          </div>
        ))}
      </div>

      {/* Main Actions */}
      <div className="grid md:grid-cols-3 gap-6 mb-16">
        <Link
          href="/research"
          className="group bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:border-emerald-300 hover:shadow-md transition-all"
        >
          <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-emerald-200 transition-colors">
            <BookOpen className="text-emerald-600" size={24} />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Research Assistant
          </h2>
          <p className="text-gray-600 mb-4">
            Ask natural language questions about Quranic behaviors. Explore tafsir,
            find patterns, and discover connections.
          </p>
          <span className="text-emerald-600 font-medium group-hover:underline">
            Start Exploring →
          </span>
        </Link>

        <Link
          href="/dashboard"
          className="group bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:border-emerald-300 hover:shadow-md transition-all"
        >
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
            <LayoutDashboard className="text-blue-600" size={24} />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Project Dashboard
          </h2>
          <p className="text-gray-600 mb-4">
            View annotation progress, coverage statistics, quality metrics, and
            team activity in real-time.
          </p>
          <span className="text-blue-600 font-medium group-hover:underline">
            View Dashboard →
          </span>
        </Link>

        <Link
          href="/annotate"
          className="group bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:border-emerald-300 hover:shadow-md transition-all"
        >
          <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-amber-200 transition-colors">
            <PenTool className="text-amber-600" size={24} />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Annotator Workbench
          </h2>
          <p className="text-gray-600 mb-4">
            Annotate ayat with behavioral classifications. Access tafsir, view
            guidelines, and submit annotations.
          </p>
          <span className="text-amber-600 font-medium group-hover:underline">
            Start Annotating →
          </span>
        </Link>
      </div>

      {/* Framework Info */}
      <div className="bg-emerald-50 rounded-xl p-8 border border-emerald-100">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          Based on Dr. Bouzidani's Framework
        </h2>
        <p className="text-gray-600 mb-6">
          This project implements the five-context behavioral classification matrix
          from "السلوك البشري في سياقه القرآني" (Human Behavior in the Quranic Context).
        </p>
        
        <div className="grid md:grid-cols-5 gap-4">
          {[
            { ar: "العضوي", en: "Organic", desc: "Body organs involved" },
            { ar: "الموضعي", en: "Situational", desc: "Internal/external" },
            { ar: "النسقي", en: "Systemic", desc: "Social contexts" },
            { ar: "المكاني", en: "Spatial", desc: "Location context" },
            { ar: "الزماني", en: "Temporal", desc: "Time context" },
          ].map((ctx) => (
            <div key={ctx.en} className="bg-white rounded-lg p-4 text-center">
              <div className="text-lg font-arabic text-emerald-700 mb-1">{ctx.ar}</div>
              <div className="font-medium text-gray-900">{ctx.en}</div>
              <div className="text-sm text-gray-500">{ctx.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer Links */}
      <div className="mt-16 pt-8 border-t border-gray-200">
        <div className="flex flex-wrap justify-center gap-6 text-gray-500">
          <a href="#" className="flex items-center gap-2 hover:text-emerald-600">
            <Database size={16} />
            <span>API Docs</span>
          </a>
          <a href="#" className="flex items-center gap-2 hover:text-emerald-600">
            <FileText size={16} />
            <span>Coding Manual</span>
          </a>
          <a href="#" className="flex items-center gap-2 hover:text-emerald-600">
            <GitBranch size={16} />
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </main>
  );
}
