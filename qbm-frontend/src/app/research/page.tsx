"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { Search, BookOpen, BarChart3, Users } from "lucide-react";

const C1Chat = dynamic(
  () => import("@thesysai/genui-sdk").then((mod) => mod.C1Chat),
  { ssr: false, loading: () => <div className="flex-1 flex items-center justify-center"><p className="text-gray-500">Loading chat...</p></div> }
);

const EXAMPLE_PROMPTS = [
  { icon: Search, label: "Behaviors of the heart", prompt: "Show me all behaviors related to the heart (قلب) in the Quran" },
  { icon: BookOpen, label: "Tafsir comparison", prompt: "Compare tafsir for Ayat al-Kursi (2:255)" },
  { icon: BarChart3, label: "Project statistics", prompt: "Show annotation statistics and coverage for the QBM project" },
  { icon: Users, label: "Social behaviors", prompt: "Find behaviors involving social relationships in Surah Al-Hujurat" },
];

export default function ResearchPage() {
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);

  return (
    <div className="h-[calc(100vh-64px)] flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-emerald-800 text-white px-6 py-4">
        <h1 className="text-2xl font-bold">Research Assistant</h1>
        <p className="text-emerald-200">
          Explore the Quranic Behavioral Matrix with natural language queries
        </p>
      </div>

      {/* Example Prompts */}
      <div className="bg-emerald-50 border-b border-emerald-100 px-6 py-3">
        <p className="text-sm text-emerald-700 mb-2 font-medium">Try these example queries:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_PROMPTS.map((item, i) => (
            <button
              key={i}
              onClick={() => setSelectedPrompt(item.prompt)}
              className="flex items-center gap-2 px-3 py-1.5 bg-white border border-emerald-200 rounded-full text-sm text-emerald-700 hover:bg-emerald-100 transition-colors"
            >
              <item.icon size={14} />
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* C1 Chat Interface */}
      <div className="flex-1 overflow-hidden" suppressHydrationWarning>
        <C1Chat
          apiUrl="/api/chat"
          agentName="QBM Research Assistant"
          formFactor="full-page"
        />
      </div>
    </div>
  );
}
