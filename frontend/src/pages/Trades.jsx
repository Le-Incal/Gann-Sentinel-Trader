import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  CheckCircle, 
  XCircle,
  AlertTriangle,
  Filter,
  Search
} from 'lucide-react'
import api from '../api'
import { formatCurrency, formatDate, formatTimeAgo, truncate } from '../utils/format'

const statusConfig = {
  'filled': { icon: CheckCircle, label: 'Filled', color: 'badge-green' },
  'pending_approval': { icon: Clock, label: 'Pending', color: 'badge-yellow' },
  'approved': { icon: CheckCircle, label: 'Approved', color: 'badge-blue' },
  'rejected': { icon: XCircle, label: 'Rejected', color: 'badge-red' },
  'cancelled': { icon: XCircle, label: 'Cancelled', color: 'badge-gray' },
  'failed': { icon: AlertTriangle, label: 'Failed', color: 'badge-red' },
}

function TradeCard({ trade }) {
  const [expanded, setExpanded] = useState(false)
  const status = statusConfig[trade.status] || statusConfig['pending_approval']
  const StatusIcon = status.icon
  const isBuy = trade.side?.toLowerCase() === 'buy'
  
  return (
    <div className="card hover:border-gst-accent/50 transition-colors">
      <div className="flex items-start justify-between">
        {/* Left side - Trade info */}
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-lg ${isBuy ? 'bg-green-900/30' : 'bg-red-900/30'}`}>
            {isBuy ? (
              <TrendingUp className="w-6 h-6 text-gst-green" />
            ) : (
              <TrendingDown className="w-6 h-6 text-gst-red" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`font-bold text-lg ${isBuy ? 'text-gst-green' : 'text-gst-red'}`}>
                {trade.side?.toUpperCase()}
              </span>
              <span className="text-white font-bold text-lg">{trade.ticker}</span>
              <span className={`badge ${status.color}`}>
                <StatusIcon className="w-3 h-3 mr-1" />
                {status.label}
              </span>
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-400">
              <span>{trade.quantity} shares</span>
              <span>•</span>
              <span>{trade.order_type}</span>
              {trade.limit_price && (
                <>
                  <span>•</span>
                  <span>Limit: {formatCurrency(trade.limit_price)}</span>
                </>
              )}
            </div>
            {trade.fill_price && (
              <p className="text-sm text-gray-300 mt-1">
                Filled at <span className="text-white font-medium">{formatCurrency(trade.fill_price)}</span>
              </p>
            )}
          </div>
        </div>

        {/* Right side - Time and actions */}
        <div className="text-right">
          <p className="text-gray-400 text-sm">{formatTimeAgo(trade.created_at)}</p>
          <p className="text-gray-500 text-xs">{formatDate(trade.created_at)}</p>
          {trade.conviction_score && (
            <p className={`text-sm mt-2 ${
              trade.conviction_score >= 80 ? 'text-gst-green' :
              trade.conviction_score >= 60 ? 'text-gst-yellow' : 'text-gray-400'
            }`}>
              Conviction: {trade.conviction_score}
            </p>
          )}
        </div>
      </div>

      {/* Thesis */}
      {trade.thesis && (
        <div className="mt-4 pt-4 border-t border-gst-border">
          <button 
            onClick={() => setExpanded(!expanded)}
            className="text-sm text-gst-accent hover:text-gst-accent-hover transition-colors"
          >
            {expanded ? 'Hide thesis' : 'Show thesis'}
          </button>
          {expanded && (
            <p className="mt-2 text-gray-300 text-sm leading-relaxed">
              {trade.thesis}
            </p>
          )}
        </div>
      )}

      {/* Rejection reason */}
      {trade.rejection_reason && (
        <div className="mt-4 pt-4 border-t border-gst-border">
          <p className="text-red-400 text-sm">
            <span className="font-medium">Rejection reason:</span> {trade.rejection_reason}
          </p>
        </div>
      )}
    </div>
  )
}

export default function Trades() {
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')

  const { data: tradesData, isLoading, error } = useQuery({
    queryKey: ['trades'],
    queryFn: () => api.getTrades(100),
  })

  const { data: pendingData } = useQuery({
    queryKey: ['pendingTrades'],
    queryFn: api.getPendingTrades,
  })

  const trades = tradesData?.trades || []
  const pendingTrades = pendingData?.trades || []

  // Filter trades
  const filteredTrades = trades.filter(trade => {
    if (filter !== 'all' && trade.status !== filter) return false
    if (search && !trade.ticker?.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gst-accent"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card border-red-800/50">
        <div className="flex items-center gap-3 text-red-400">
          <AlertTriangle className="w-6 h-6" />
          <div>
            <p className="font-medium">Failed to load trades</p>
            <p className="text-sm text-red-400/70">{error.message}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Trade History</h2>
          <p className="text-gray-400 mt-1">View all executed and pending trades</p>
        </div>
        {pendingTrades.length > 0 && (
          <div className="badge badge-yellow text-base px-4 py-2">
            <Clock className="w-4 h-4 mr-2" />
            {pendingTrades.length} pending approval{pendingTrades.length > 1 ? 's' : ''}
          </div>
        )}
      </div>

      {/* Pending Trades Alert */}
      {pendingTrades.length > 0 && (
        <div className="card bg-yellow-900/20 border-yellow-800/50">
          <div className="flex items-start gap-3">
            <Clock className="w-5 h-5 text-yellow-400 mt-0.5" />
            <div>
              <h3 className="text-yellow-400 font-medium">Trades Awaiting Approval</h3>
              <p className="text-gray-400 text-sm mt-1">
                Use Telegram to approve or reject pending trades. Send <code className="bg-gst-darker px-1 rounded">/pending</code> to view and <code className="bg-gst-darker px-1 rounded">/approve [ID]</code> to approve.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
          <input
            type="text"
            placeholder="Search by ticker..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-gst-card border border-gst-border rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-gst-accent"
          />
        </div>

        {/* Status filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-500" />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-4 py-2.5 bg-gst-card border border-gst-border rounded-lg text-white focus:outline-none focus:border-gst-accent"
          >
            <option value="all">All Trades</option>
            <option value="filled">Filled</option>
            <option value="pending_approval">Pending</option>
            <option value="rejected">Rejected</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
      </div>

      {/* Trade List */}
      <div className="space-y-4">
        {filteredTrades.length > 0 ? (
          filteredTrades.map((trade) => (
            <TradeCard key={trade.id} trade={trade} />
          ))
        ) : (
          <div className="card text-center py-12">
            <TrendingUp className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No trades found</p>
            <p className="text-gray-500 text-sm mt-1">
              {search || filter !== 'all' 
                ? 'Try adjusting your filters' 
                : 'Trades will appear here when the AI committee makes decisions'}
            </p>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      {trades.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card text-center">
            <p className="stat-value text-white">{trades.length}</p>
            <p className="stat-label">Total Trades</p>
          </div>
          <div className="card text-center">
            <p className="stat-value text-gst-green">
              {trades.filter(t => t.status === 'filled').length}
            </p>
            <p className="stat-label">Filled</p>
          </div>
          <div className="card text-center">
            <p className="stat-value text-gst-yellow">
              {trades.filter(t => t.status === 'pending_approval').length}
            </p>
            <p className="stat-label">Pending</p>
          </div>
          <div className="card text-center">
            <p className="stat-value text-gst-red">
              {trades.filter(t => t.status === 'rejected').length}
            </p>
            <p className="stat-label">Rejected</p>
          </div>
        </div>
      )}
    </div>
  )
}
