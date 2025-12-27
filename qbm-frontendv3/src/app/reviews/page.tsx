'use client';

import { useState, useEffect } from 'react';
import { 
  ClipboardCheck, 
  Plus, 
  Filter, 
  CheckCircle, 
  XCircle, 
  Clock,
  History,
  User,
  MessageSquare,
  Star,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { 
  fetchReviewsStatus, 
  fetchReviews, 
  createReview, 
  updateReview,
  fetchReview,
  type ReviewsStatus, 
  type Review 
} from '@/lib/api';

export default function ReviewsPage() {
  const [status, setStatus] = useState<ReviewsStatus | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedReview, setSelectedReview] = useState<Review | null>(null);
  const [filters, setFilters] = useState({
    status: '',
    review_type: '',
  });

  // Form state
  const [formData, setFormData] = useState({
    edge_id: '',
    chunk_id: '',
    span_id: '',
    surah: '',
    ayah: '',
    reviewer_id: 'scholar_default',
    reviewer_name: '',
    rating: 0,
    comment: '',
  });

  useEffect(() => {
    loadData();
  }, [filters]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [statusData, reviewsData] = await Promise.all([
        fetchReviewsStatus(),
        fetchReviews({
          status: filters.status || undefined,
          review_type: filters.review_type || undefined,
          limit: 50,
        }),
      ]);
      
      setStatus(statusData);
      setReviews(reviewsData.reviews);
    } catch (err: any) {
      setError(err.message || 'Failed to load reviews');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateReview = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      await createReview({
        edge_id: formData.edge_id || undefined,
        chunk_id: formData.chunk_id || undefined,
        span_id: formData.span_id || undefined,
        surah: formData.surah ? parseInt(formData.surah) : undefined,
        ayah: formData.ayah ? parseInt(formData.ayah) : undefined,
        reviewer_id: formData.reviewer_id,
        reviewer_name: formData.reviewer_name || undefined,
        rating: formData.rating || undefined,
        comment: formData.comment || undefined,
      });
      
      setShowCreateForm(false);
      setFormData({
        edge_id: '',
        chunk_id: '',
        span_id: '',
        surah: '',
        ayah: '',
        reviewer_id: 'scholar_default',
        reviewer_name: '',
        rating: 0,
        comment: '',
      });
      loadData();
    } catch (err: any) {
      setError(err.message || 'Failed to create review');
    }
  };

  const handleUpdateStatus = async (reviewId: number, newStatus: string) => {
    try {
      setError(null);
      await updateReview(reviewId, { status: newStatus }, 'scholar_default');
      loadData();
      if (selectedReview?.id === reviewId) {
        const updated = await fetchReview(reviewId);
        setSelectedReview(updated);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to update review');
    }
  };

  const loadReviewDetails = async (reviewId: number) => {
    try {
      const review = await fetchReview(reviewId);
      setSelectedReview(review);
    } catch (err: any) {
      setError(err.message || 'Failed to load review details');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved':
        return <CheckCircle className="w-4 h-4 text-emerald-600" />;
      case 'rejected':
        return <XCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-amber-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved':
        return 'bg-emerald-100 text-emerald-700';
      case 'rejected':
        return 'bg-red-100 text-red-700';
      default:
        return 'bg-amber-100 text-amber-700';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'edge':
        return 'bg-blue-100 text-blue-700';
      case 'chunk':
        return 'bg-purple-100 text-purple-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  if (loading && !reviews.length) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading reviews...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-50 via-white to-emerald-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-emerald-800 flex items-center gap-3">
              <ClipboardCheck className="w-8 h-8" />
              Scholar Reviews
            </h1>
            <p className="text-gray-600 mt-1">
              Review and validate QBM annotations, edges, and evidence chunks
            </p>
          </div>
          <button
            onClick={() => setShowCreateForm(true)}
            className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Review
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {/* Status Cards */}
        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 mb-8">
            <StatCard label="Total" value={status.statistics.total_reviews} color="gray" />
            <StatCard label="Pending" value={status.statistics.by_status.pending || 0} color="amber" />
            <StatCard label="Approved" value={status.statistics.by_status.approved || 0} color="emerald" />
            <StatCard label="Rejected" value={status.statistics.by_status.rejected || 0} color="red" />
            <StatCard label="Spans" value={status.statistics.by_type.span || 0} color="gray" />
            <StatCard label="Edges" value={status.statistics.by_type.edge || 0} color="blue" />
            <StatCard label="Chunks" value={status.statistics.by_type.chunk || 0} color="purple" />
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-xl shadow border border-gray-200 p-4 mb-6">
          <div className="flex items-center gap-4">
            <Filter className="w-5 h-5 text-gray-400" />
            <select
              value={filters.status}
              onChange={(e) => setFilters({ ...filters, status: e.target.value })}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
            <select
              value={filters.review_type}
              onChange={(e) => setFilters({ ...filters, review_type: e.target.value })}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">All Types</option>
              <option value="span">Span</option>
              <option value="edge">Edge</option>
              <option value="chunk">Chunk</option>
            </select>
          </div>
        </div>

        <div className="flex gap-6">
          {/* Reviews List */}
          <div className={`${selectedReview ? 'w-1/2' : 'w-full'} transition-all`}>
            <div className="bg-white rounded-xl shadow border border-gray-200 overflow-hidden">
              <div className="p-4 border-b bg-gray-50">
                <h2 className="font-bold text-gray-800">Reviews ({reviews.length})</h2>
              </div>
              
              {reviews.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  <ClipboardCheck className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No reviews found</p>
                </div>
              ) : (
                <div className="divide-y">
                  {reviews.map((review) => (
                    <div
                      key={review.id}
                      onClick={() => loadReviewDetails(review.id)}
                      className={`p-4 hover:bg-gray-50 cursor-pointer transition ${
                        selectedReview?.id === review.id ? 'bg-emerald-50' : ''
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            {getStatusIcon(review.status)}
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(review.status)}`}>
                              {review.status}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${getTypeColor(review.review_type)}`}>
                              {review.review_type}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {review.comment || 'No comment'}
                          </p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                            <span className="flex items-center gap-1">
                              <User className="w-3 h-3" />
                              {review.reviewer_name || review.reviewer_id}
                            </span>
                            {review.verse_key && (
                              <span>📍 {review.verse_key}</span>
                            )}
                            {review.rating && (
                              <span className="flex items-center gap-1">
                                <Star className="w-3 h-3 text-amber-500" />
                                {review.rating}/5
                              </span>
                            )}
                          </div>
                        </div>
                        <span className="text-xs text-gray-400">
                          #{review.id}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Review Detail Panel */}
          {selectedReview && (
            <div className="w-1/2">
              <div className="bg-white rounded-xl shadow border border-gray-200 overflow-hidden sticky top-4">
                <div className="p-4 border-b bg-gradient-to-r from-emerald-50 to-white">
                  <div className="flex items-center justify-between">
                    <h2 className="font-bold text-gray-800">Review #{selectedReview.id}</h2>
                    <button
                      onClick={() => setSelectedReview(null)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      ×
                    </button>
                  </div>
                </div>

                <div className="p-4 space-y-4">
                  {/* Status & Type */}
                  <div className="flex items-center gap-2">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(selectedReview.status)}`}>
                      {selectedReview.status}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getTypeColor(selectedReview.review_type)}`}>
                      {selectedReview.review_type}
                    </span>
                  </div>

                  {/* Reference */}
                  <div className="bg-gray-50 rounded-lg p-3">
                    <h4 className="text-xs font-medium text-gray-500 mb-2">Reference</h4>
                    {selectedReview.edge_id && (
                      <p className="text-sm"><strong>Edge:</strong> {selectedReview.edge_id}</p>
                    )}
                    {selectedReview.chunk_id && (
                      <p className="text-sm"><strong>Chunk:</strong> {selectedReview.chunk_id}</p>
                    )}
                    {selectedReview.span_id && (
                      <p className="text-sm"><strong>Span:</strong> {selectedReview.span_id}</p>
                    )}
                    {selectedReview.verse_key && (
                      <p className="text-sm"><strong>Verse:</strong> {selectedReview.verse_key}</p>
                    )}
                  </div>

                  {/* Comment */}
                  {selectedReview.comment && (
                    <div>
                      <h4 className="text-xs font-medium text-gray-500 mb-1">Comment</h4>
                      <p className="text-gray-700">{selectedReview.comment}</p>
                    </div>
                  )}

                  {/* Rating */}
                  {selectedReview.rating && (
                    <div className="flex items-center gap-2">
                      <h4 className="text-xs font-medium text-gray-500">Rating:</h4>
                      <div className="flex">
                        {[1, 2, 3, 4, 5].map((star) => (
                          <Star
                            key={star}
                            className={`w-5 h-5 ${
                              star <= selectedReview.rating! ? 'text-amber-400 fill-amber-400' : 'text-gray-300'
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Reviewer */}
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <User className="w-4 h-4" />
                    <span>{selectedReview.reviewer_name || selectedReview.reviewer_id}</span>
                  </div>

                  {/* Actions */}
                  {selectedReview.status === 'pending' && (
                    <div className="flex gap-2 pt-4 border-t">
                      <button
                        onClick={() => handleUpdateStatus(selectedReview.id, 'approved')}
                        className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition flex items-center justify-center gap-2"
                      >
                        <CheckCircle className="w-4 h-4" />
                        Approve
                      </button>
                      <button
                        onClick={() => handleUpdateStatus(selectedReview.id, 'rejected')}
                        className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition flex items-center justify-center gap-2"
                      >
                        <XCircle className="w-4 h-4" />
                        Reject
                      </button>
                    </div>
                  )}

                  {/* History */}
                  {selectedReview.history && selectedReview.history.length > 0 && (
                    <div className="pt-4 border-t">
                      <h4 className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
                        <History className="w-4 h-4" />
                        History
                      </h4>
                      <div className="space-y-2">
                        {selectedReview.history.map((h: any, i: number) => (
                          <div key={i} className="text-xs bg-gray-50 rounded p-2">
                            <span className="font-medium">{h.action}</span>
                            {h.old_status && h.new_status && (
                              <span className="text-gray-500">
                                : {h.old_status} → {h.new_status}
                              </span>
                            )}
                            <span className="text-gray-400 ml-2">{h.timestamp}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Create Review Modal */}
        {showCreateForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-xl w-full max-w-lg mx-4">
              <div className="p-4 border-b flex items-center justify-between">
                <h2 className="font-bold text-gray-800">Create New Review</h2>
                <button
                  onClick={() => setShowCreateForm(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>

              <form onSubmit={handleCreateReview} className="p-4 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Edge ID</label>
                    <input
                      type="text"
                      value={formData.edge_id}
                      onChange={(e) => setFormData({ ...formData, edge_id: e.target.value })}
                      placeholder="e.g., BEH_PATIENCE_CAUSES_CSQ_JANNAH"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Chunk ID</label>
                    <input
                      type="text"
                      value={formData.chunk_id}
                      onChange={(e) => setFormData({ ...formData, chunk_id: e.target.value })}
                      placeholder="e.g., ibn_kathir_2_255_001"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Surah</label>
                    <input
                      type="number"
                      min="1"
                      max="114"
                      value={formData.surah}
                      onChange={(e) => setFormData({ ...formData, surah: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Ayah</label>
                    <input
                      type="number"
                      min="1"
                      value={formData.ayah}
                      onChange={(e) => setFormData({ ...formData, ayah: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Reviewer Name</label>
                  <input
                    type="text"
                    value={formData.reviewer_name}
                    onChange={(e) => setFormData({ ...formData, reviewer_name: e.target.value })}
                    placeholder="Your name"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Rating</label>
                  <div className="flex gap-2">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        type="button"
                        onClick={() => setFormData({ ...formData, rating: star })}
                        className="p-1"
                      >
                        <Star
                          className={`w-6 h-6 ${
                            star <= formData.rating ? 'text-amber-400 fill-amber-400' : 'text-gray-300'
                          }`}
                        />
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Comment</label>
                  <textarea
                    value={formData.comment}
                    onChange={(e) => setFormData({ ...formData, comment: e.target.value })}
                    rows={3}
                    placeholder="Your review comments..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <div className="flex gap-2 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowCreateForm(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition"
                  >
                    Create Review
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses: Record<string, string> = {
    gray: 'bg-gray-100 text-gray-700',
    amber: 'bg-amber-100 text-amber-700',
    emerald: 'bg-emerald-100 text-emerald-700',
    red: 'bg-red-100 text-red-700',
    blue: 'bg-blue-100 text-blue-700',
    purple: 'bg-purple-100 text-purple-700',
  };

  return (
    <div className={`rounded-xl p-3 text-center ${colorClasses[color]}`}>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-xs font-medium opacity-80">{label}</p>
    </div>
  );
}
