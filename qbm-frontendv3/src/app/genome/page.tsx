'use client';

import { useState, useEffect } from 'react';
import { 
  Download, 
  CheckCircle, 
  Database, 
  GitBranch, 
  Heart, 
  Users, 
  Activity,
  FileJson,
  Copy,
  Check,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { fetchGenomeStatus, fetchGenomeExport, type GenomeStatus, type GenomeExport } from '@/lib/api';

export default function GenomePage() {
  const [status, setStatus] = useState<GenomeStatus | null>(null);
  const [exportData, setExportData] = useState<GenomeExport | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchGenomeStatus();
      setStatus(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load genome status');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (mode: 'full' | 'light') => {
    try {
      setExporting(true);
      setError(null);
      const data = await fetchGenomeExport(mode);
      setExportData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to export genome');
    } finally {
      setExporting(false);
    }
  };

  const downloadJson = () => {
    if (!exportData) return;
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qbm_genome_${exportData.mode}_${exportData.version}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyChecksum = () => {
    if (!exportData?.checksum) return;
    navigator.clipboard.writeText(exportData.checksum);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading Q25 Genome...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-50 via-white to-emerald-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-emerald-800 mb-3 flex items-center justify-center gap-3">
            <Database className="w-10 h-10" />
            Q25 Quranic Behavioral Genome
          </h1>
          <p className="text-lg text-gray-600">
            Canonical entity registry with evidence-backed semantic relationships
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {/* Status Card */}
        {status && (
          <div className="bg-white rounded-xl shadow-lg border border-emerald-200 p-6 mb-8">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-emerald-600" />
                <div>
                  <h2 className="text-xl font-bold text-gray-800">Genome Status</h2>
                  <p className="text-sm text-gray-500">Version {status.version}</p>
                </div>
              </div>
              <span className="px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full font-medium">
                {status.status.toUpperCase()}
              </span>
            </div>

            {/* Statistics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
              <StatCard
                icon={<Activity className="w-5 h-5" />}
                label="Behaviors"
                value={status.statistics.canonical_behaviors}
                color="emerald"
              />
              <StatCard
                icon={<Users className="w-5 h-5" />}
                label="Agents"
                value={status.statistics.canonical_agents}
                color="blue"
              />
              <StatCard
                icon={<Heart className="w-5 h-5" />}
                label="Heart States"
                value={status.statistics.canonical_heart_states}
                color="rose"
              />
              <StatCard
                icon={<Activity className="w-5 h-5" />}
                label="Organs"
                value={status.statistics.canonical_organs || 0}
                color="amber"
              />
              <StatCard
                icon={<AlertCircle className="w-5 h-5" />}
                label="Consequences"
                value={status.statistics.canonical_consequences}
                color="purple"
              />
              <StatCard
                icon={<GitBranch className="w-5 h-5" />}
                label="Semantic Edges"
                value={status.statistics.semantic_edges}
                color="indigo"
              />
            </div>

            {/* Source Versions */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-700 mb-2">Source Versions</h3>
              <div className="flex gap-4 text-sm">
                <span className="px-3 py-1 bg-white rounded border">
                  canonical_entities: v{status.source_versions.canonical_entities}
                </span>
                <span className="px-3 py-1 bg-white rounded border">
                  semantic_graph: v{status.source_versions.semantic_graph}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Export Actions */}
        <div className="bg-white rounded-xl shadow-lg border border-emerald-200 p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <FileJson className="w-6 h-6 text-emerald-600" />
            Export Genome
          </h2>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <button
              onClick={() => handleExport('light')}
              disabled={exporting}
              className="p-4 border-2 border-emerald-200 rounded-xl hover:border-emerald-400 hover:bg-emerald-50 transition text-left"
            >
              <div className="flex items-center gap-3 mb-2">
                <Download className="w-5 h-5 text-emerald-600" />
                <span className="font-bold text-gray-800">Light Export</span>
              </div>
              <p className="text-sm text-gray-600">
                Metadata only - entity IDs, counts, no evidence payloads. Fast and small.
              </p>
            </button>

            <button
              onClick={() => handleExport('full')}
              disabled={exporting}
              className="p-4 border-2 border-blue-200 rounded-xl hover:border-blue-400 hover:bg-blue-50 transition text-left"
            >
              <div className="flex items-center gap-3 mb-2">
                <Download className="w-5 h-5 text-blue-600" />
                <span className="font-bold text-gray-800">Full Export</span>
              </div>
              <p className="text-sm text-gray-600">
                Complete genome with all evidence, provenance (chunk_id, verse_key, char_start/end).
              </p>
            </button>
          </div>

          {exporting && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-8 h-8 text-emerald-600 animate-spin" />
              <span className="ml-3 text-gray-600">Exporting genome...</span>
            </div>
          )}
        </div>

        {/* Export Result */}
        {exportData && (
          <div className="bg-white rounded-xl shadow-lg border border-emerald-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800">Export Result</h2>
              <div className="flex gap-2">
                <button
                  onClick={downloadJson}
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download JSON
                </button>
              </div>
            </div>

            {/* Checksum */}
            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-400 mb-1">SHA256 Checksum</p>
                  <code className="text-emerald-400 font-mono text-sm break-all">
                    {exportData.checksum}
                  </code>
                </div>
                <button
                  onClick={copyChecksum}
                  className="p-2 hover:bg-gray-800 rounded transition"
                >
                  {copied ? (
                    <Check className="w-5 h-5 text-emerald-400" />
                  ) : (
                    <Copy className="w-5 h-5 text-gray-400" />
                  )}
                </button>
              </div>
            </div>

            {/* Export Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-2xl font-bold text-emerald-600">{exportData.behaviors.length}</p>
                <p className="text-sm text-gray-600">Behaviors</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-2xl font-bold text-blue-600">{exportData.agents.length}</p>
                <p className="text-sm text-gray-600">Agents</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-2xl font-bold text-rose-600">{exportData.heart_states.length}</p>
                <p className="text-sm text-gray-600">Heart States</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-2xl font-bold text-indigo-600">
                  {exportData.semantic_edges?.length || 0}
                </p>
                <p className="text-sm text-gray-600">Edges</p>
              </div>
            </div>

            {/* Mode Badge */}
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span>Mode:</span>
              <span className={`px-2 py-1 rounded ${
                exportData.mode === 'full' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
              }`}>
                {exportData.mode}
              </span>
              <span>Version:</span>
              <span className="px-2 py-1 bg-gray-100 rounded">{exportData.version}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ 
  icon, 
  label, 
  value, 
  color 
}: { 
  icon: React.ReactNode; 
  label: string; 
  value: number; 
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    emerald: 'bg-emerald-50 text-emerald-600',
    blue: 'bg-blue-50 text-blue-600',
    rose: 'bg-rose-50 text-rose-600',
    amber: 'bg-amber-50 text-amber-600',
    purple: 'bg-purple-50 text-purple-600',
    indigo: 'bg-indigo-50 text-indigo-600',
  };

  return (
    <div className={`rounded-xl p-4 ${colorClasses[color]}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-xs font-medium opacity-80">{label}</span>
      </div>
      <p className="text-2xl font-bold">{value.toLocaleString()}</p>
    </div>
  );
}
