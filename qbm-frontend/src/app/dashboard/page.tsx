"use client";

import { useState, useEffect } from "react";
import { RefreshCw, BookOpen, Users, BarChart3, FileText } from "lucide-react";

interface Stats {
  total_spans: number;
  unique_surahs: number;
  unique_ayat: number;
  agent_types: Record<string, number>;
  behavior_forms: Record<string, number>;
  evaluations: Record<string, number>;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/stats");
      if (!response.ok) throw new Error("Failed to fetch stats");
      const data = await response.json();
      setStats(data);
    } catch (err) {
      setError("Could not connect to backend. Make sure the API is running on localhost:8000");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const coverage = stats ? ((stats.unique_ayat / 6236) * 100).toFixed(1) : 0;

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-700 to-blue-800 text-white px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Project Dashboard</h1>
            <p className="text-blue-200">QBM annotation progress and statistics</p>
          </div>
          <button
            onClick={loadStats}
            disabled={isLoading}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2 rounded-lg transition-colors"
          >
            <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <RefreshCw size={48} className="animate-spin text-blue-500" />
          </div>
        ) : stats ? (
          <>
            {/* Coverage Card */}
            <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-700">Quran Coverage</h2>
                  <p className="text-4xl font-bold text-emerald-600 mt-2">{coverage}%</p>
                  <p className="text-gray-500 mt-1">{stats.unique_ayat.toLocaleString()} of 6,236 ayat annotated</p>
                </div>
                <div className="w-32 h-32 relative">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="64" cy="64" r="56" stroke="#e5e7eb" strokeWidth="12" fill="none" />
                    <circle cx="64" cy="64" r="56" stroke="#10b981" strokeWidth="12" fill="none"
                      strokeDasharray={`${Number(coverage) * 3.52} 352`} strokeLinecap="round" />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-2xl font-bold text-emerald-600">
                    {coverage}%
                  </span>
                </div>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <StatCard icon={FileText} label="Total Spans" value={stats.total_spans.toLocaleString()} color="blue" />
              <StatCard icon={BookOpen} label="Surahs Covered" value={`${stats.unique_surahs} / 114`} color="emerald" />
              <StatCard icon={Users} label="Agent Types" value={Object.keys(stats.agent_types).length.toString()} color="purple" />
              <StatCard icon={BarChart3} label="Behavior Forms" value={Object.keys(stats.behavior_forms).length.toString()} color="orange" />
            </div>

            {/* Distribution Tables */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DistributionCard title="Agent Types" data={stats.agent_types} />
              <DistributionCard title="Behavior Forms" data={stats.behavior_forms} />
              <DistributionCard title="Evaluations" data={stats.evaluations} />
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: { icon: any; label: string; value: string; color: string }) {
  const colors: Record<string, string> = {
    blue: "bg-blue-50 text-blue-600",
    emerald: "bg-emerald-50 text-emerald-600",
    purple: "bg-purple-50 text-purple-600",
    orange: "bg-orange-50 text-orange-600",
  };
  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <div className={`w-10 h-10 rounded-lg ${colors[color]} flex items-center justify-center mb-3`}>
        <Icon size={20} />
      </div>
      <p className="text-2xl font-bold text-gray-800">{value}</p>
      <p className="text-gray-500 text-sm">{label}</p>
    </div>
  );
}

function DistributionCard({ title, data }: { title: string; data: Record<string, number> }) {
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = sorted[0]?.[1] || 1;
  return (
    <div className="bg-white rounded-xl shadow-sm border p-4">
      <h3 className="font-semibold text-gray-700 mb-3">{title}</h3>
      <div className="space-y-2">
        {sorted.slice(0, 6).map(([key, val]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-28 truncate" title={key}>{key.replace("AGT_", "")}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-2">
              <div className="bg-emerald-500 h-2 rounded-full" style={{ width: `${(val / max) * 100}%` }} />
            </div>
            <span className="text-xs text-gray-600 w-12 text-right">{val.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
