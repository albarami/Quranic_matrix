"use client";

import { useState, useEffect } from "react";
import { C1Component, ThemeProvider } from "@thesysai/genui-sdk";
import { RefreshCw } from "lucide-react";

export default function DashboardPage() {
  const [c1Response, setC1Response] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            {
              role: "user",
              content: `Generate a comprehensive QBM project dashboard with the following sections:

1. **Overall Progress** (prominent at top)
   - Total ayat: 6,236
   - Show a large circular progress indicator
   - Annotated vs pending count

2. **Annotation Statistics** (charts)
   - Pie chart: Gold vs Silver vs Research tier distribution
   - Bar chart: Top 10 surahs by span count
   
3. **Behavioral Coverage**
   - Horizontal bar chart: Most common behavior concepts (top 15)
   - Show both English and Arabic labels

4. **Quality Metrics**
   - IAA (Inter-annotator agreement) score display
   - Confidence distribution

5. **Recent Activity**
   - Table showing last 10 annotations with:
     - Ayah reference
     - Behavior type
     - Annotator
     - Timestamp
   - Make rows clickable

6. **Quick Actions** (buttons at bottom)
   - "View Full Report"
   - "Export Data"
   - "Annotator Leaderboard"

Use emerald/green color theme. Make it visually clean and professional.
Include Arabic text where appropriate (behavior names, surah names).`,
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

      setLastUpdated(new Date());
    } catch (error) {
      console.error("Dashboard load error:", error);
      setC1Response("Error loading dashboard. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAction = async (action: any) => {
    console.log("Dashboard action:", action);
    
    // Handle drill-down actions
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

  return (
    <ThemeProvider>
      <div className="min-h-[calc(100vh-64px)] bg-gray-50">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-700 to-blue-800 text-white px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Project Dashboard</h1>
              <p className="text-blue-200">
                QBM annotation progress and statistics
              </p>
            </div>
            <div className="flex items-center gap-4">
              {lastUpdated && (
                <span className="text-blue-200 text-sm">
                  Updated: {lastUpdated.toLocaleTimeString()}
                </span>
              )}
              <button
                onClick={loadDashboard}
                disabled={isLoading}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2 rounded-lg transition-colors"
              >
                <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Dashboard Content */}
        <div className="max-w-7xl mx-auto p-6">
          {isLoading && !c1Response ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <RefreshCw size={48} className="animate-spin text-blue-500 mx-auto mb-4" />
                <p className="text-gray-600">Loading dashboard...</p>
              </div>
            </div>
          ) : (
            <C1Component
              c1Response={c1Response}
              isStreaming={isLoading}
              onAction={handleAction}
            />
          )}
        </div>
      </div>
    </ThemeProvider>
  );
}
