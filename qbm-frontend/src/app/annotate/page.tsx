"use client";

import { useState } from "react";
import { C1Component, ThemeProvider } from "@thesysai/genui-sdk";
import { Search, ChevronRight, BookOpen } from "lucide-react";

// Sample ayat for quick navigation
const SAMPLE_AYAT = [
  { surah: 2, ayah: 255, name: "Ayat al-Kursi", nameAr: "آية الكرسي" },
  { surah: 49, ayah: 12, name: "Backbiting", nameAr: "الغيبة" },
  { surah: 24, ayah: 30, name: "Lowering Gaze", nameAr: "غض البصر" },
  { surah: 31, ayah: 18, name: "Arrogance", nameAr: "الكبر" },
  { surah: 17, ayah: 23, name: "Parents", nameAr: "الوالدين" },
  { surah: 4, ayah: 36, name: "Good Treatment", nameAr: "الإحسان" },
  { surah: 49, ayah: 11, name: "Mockery", nameAr: "السخرية" },
  { surah: 2, ayah: 183, name: "Fasting", nameAr: "الصيام" },
];

export default function AnnotatePage() {
  const [c1Response, setC1Response] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedAyah, setSelectedAyah] = useState<{
    surah: number;
    ayah: number;
    name?: string;
  } | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const loadAnnotationWorkbench = async (surah: number, ayah: number) => {
    setIsLoading(true);
    setSelectedAyah({ surah, ayah });

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            {
              role: "user",
              content: `Generate an annotation workbench for Surah ${surah}, Ayah ${ayah}.

Layout should include:

1. **Ayah Display** (top, prominent)
   - Full Arabic text with tashkeel in large font
   - Surah name and ayah number
   - Translation below

2. **Tafsir Panel** (scrollable)
   - Tabs for: Ibn Kathir | Tabari | Qurtubi | Sa'di
   - Arabic text with RTL
   - Highlight key behavioral phrases

3. **Existing Annotations** (if any)
   - Show any existing behavioral spans
   - Status badges (Gold/Silver/Draft)

4. **New Annotation Form**
   - Span selection (highlight text)
   - Behavior concept dropdown (BEH_* options)
   - Agent type selector
   - Organ selector (if applicable)
   - Support type (direct/indirect)
   - Confidence slider
   - Notes textarea
   - Submit button

5. **Quick Tags** (sidebar)
   - Common behavior buttons for fast tagging
   - Grouped by category (speech, physical, inner state)

6. **Navigation**
   - Previous/Next ayah buttons
   - Jump to specific reference

Make it functional and professional. Use emerald/amber color theme for annotation UI.`,
            },
          ],
        }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value);
        setC1Response(accumulated);
      }
    } catch (error) {
      console.error("Workbench load error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAction = async (action: any) => {
    console.log("Annotation action:", action);

    // Handle form submissions
    if (action.payload?.annotation) {
      console.log("Submitting annotation:", action.payload.annotation);
      // TODO: Submit to backend
      alert("Annotation submitted! (Demo mode)");
      return;
    }

    // Handle navigation
    if (action.llmFriendlyMessage) {
      setIsLoading(true);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [{ role: "user", content: action.llmFriendlyMessage }],
          }),
        });

        if (!response.body) throw new Error("No response body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          accumulated += decoder.decode(value);
          setC1Response(accumulated);
        }
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // Parse search query (e.g., "2:255" or "49:12")
    const match = searchQuery.match(/^(\d+):(\d+)$/);
    if (match) {
      loadAnnotationWorkbench(parseInt(match[1]), parseInt(match[2]));
    }
  };

  return (
    <ThemeProvider>
      <div className="h-[calc(100vh-64px)] flex">
        {/* Sidebar */}
        <aside className="w-72 bg-white border-r border-gray-200 flex flex-col">
          {/* Search */}
          <div className="p-4 border-b border-gray-200">
            <form onSubmit={handleSearch}>
              <div className="relative">
                <Search
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                  size={18}
                />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Jump to ayah (e.g., 2:255)"
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                />
              </div>
            </form>
          </div>

          {/* Quick Navigation */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Quick Navigation
              </h3>
              <div className="space-y-1">
                {SAMPLE_AYAT.map((ayah) => (
                  <button
                    key={`${ayah.surah}:${ayah.ayah}`}
                    onClick={() => loadAnnotationWorkbench(ayah.surah, ayah.ayah)}
                    className={`w-full text-left p-3 rounded-lg transition-colors flex items-center justify-between group ${
                      selectedAyah?.surah === ayah.surah &&
                      selectedAyah?.ayah === ayah.ayah
                        ? "bg-amber-100 text-amber-800"
                        : "hover:bg-gray-100"
                    }`}
                  >
                    <div>
                      <div className="font-medium">
                        {ayah.surah}:{ayah.ayah}
                      </div>
                      <div className="text-sm text-gray-500">{ayah.name}</div>
                      <div className="text-sm text-gray-400 font-arabic">
                        {ayah.nameAr}
                      </div>
                    </div>
                    <ChevronRight
                      size={18}
                      className="text-gray-400 group-hover:text-gray-600"
                    />
                  </button>
                ))}
              </div>
            </div>

            {/* Pending Tasks */}
            <div className="p-4 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Your Queue
              </h3>
              <div className="text-center text-gray-400 py-4">
                <BookOpen size={32} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">No pending tasks</p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-amber-600">0</div>
                <div className="text-xs text-gray-500">Today</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-600">0</div>
                <div className="text-xs text-gray-500">Total</div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-gray-50">
          {selectedAyah ? (
            <div className="p-6">
              <C1Component
                c1Response={c1Response}
                isStreaming={isLoading}
                onAction={handleAction}
              />
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-20 h-20 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <BookOpen size={40} className="text-amber-600" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                  Select an Ayah to Begin
                </h2>
                <p className="text-gray-500 mb-6">
                  Choose an ayah from the sidebar or search for a specific
                  reference to start annotating.
                </p>
                <p className="text-sm text-gray-400">
                  Tip: Use the format "surah:ayah" (e.g., 2:255) to jump directly
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}
