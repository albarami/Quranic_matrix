"use client";

import { C1Chat } from "@thesysai/genui-sdk";

export default function ResearchPage() {
  return (
    <div className="h-[calc(100vh-64px)] flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-emerald-800 text-white px-6 py-4">
        <h1 className="text-2xl font-bold">Research Assistant</h1>
        <p className="text-emerald-200">
          Explore the Quranic Behavioral Matrix with natural language queries
        </p>
      </div>

      {/* C1 Chat Interface */}
      <div className="flex-1 overflow-hidden">
        <C1Chat
          apiUrl="/api/chat"
          placeholder="Ask about Quranic behaviors... e.g., 'Show me patience-related behaviors in Surah Al-Baqarah'"
          welcomeMessage={`
## مرحباً بك في مساعد البحث القرآني

Welcome to the **QBM Research Assistant**. I can help you explore the Quranic Behavioral Matrix.

### Try asking:

**Behavioral Queries:**
- "Show me all behaviors related to the heart (قلب)"
- "Find speech acts (أقوال) in Surah Al-Hujurat"
- "What behaviors involve the tongue (لسان)?"

**Tafsir Exploration:**
- "Compare tafsir for Ayat al-Kursi (2:255)"
- "Show Ibn Kathir's explanation of 49:12"
- "What do the scholars say about backbiting?"

**Statistics & Patterns:**
- "Which surahs have the most behavioral annotations?"
- "Show annotation progress for the project"
- "What are the most common behavior concepts?"

**Contextual Analysis:**
- "Find behaviors in the domestic context"
- "Show temporal behaviors (day/night)"
- "What behaviors involve social relationships?"
          `}
        />
      </div>
    </div>
  );
}
