import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { 
  Activity, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  DollarSign,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react'
import api from '../api'
import { formatDate, formatTimeAgo, formatCurrency } from '../utils/format'

const statusConfig = {
  'complete': { icon: CheckCircle, color: 'text-gst-green', bg: 'bg-green-900/30', label: 'Complete' },
  'started': { icon: Clock, color: 'text-gst-yellow', bg: 'bg-yellow-900/30', label: 'In Progress' },
  'failed': { icon: XCircle, color: 'text-gst-red', bg: 'bg-red-900/30', label: 'Failed' },
  'error': { icon: AlertTriangle, color: 'text-gst-red', bg: 'bg-red-900/30', label: 'Error' },
}

const decisionConfig = {
  'long': { icon: TrendingUp, color: 'text-gst-green', bg: 'bg-green-900/30' },
  'short': { icon: TrendingDown, color: 'text-gst-red', bg: 'bg-red-900/30' },
  'hold': { icon: Minus, color: 'text-gray-400', bg: 'bg-gray-800/30' },
}

function ScanCycleCard({ cycle, isExpanded, onToggle }) {
  const status = statusConfig[cycle.status] || statusConfig['started']
  const StatusIcon = status.icon
  
  const decision = decisionConfig[cycle.final_decision?.toLowerCase()] || decisionConfig['hold']
  const DecisionIcon = decision.icon

  // Parse metadata
  const metadata = cycle.metadata || {}
  const costTracking = metadata.cost_tracking || {}
  
  return (
    <div className="card">
      {/* Header */}
      <button 
        onClick={onToggle}
        className="w-full flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className={`p-2 rounded-lg ${status.bg}`}>
            <StatusIcon className={`w-5 h-5 ${status.color}`} />
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-white font-medium">
                {cycle.cycle_type === 'scheduled' ? 'Scheduled Scan' : 
                 cycle.cycle_type === 'manual' ? 'Manual Scan' : 
                 cycle.cycle_type === 'check' ? 'Ticker Check' : 'Scan Cycle'}
              </span>
              <span className={`badge ${
                status.color.replace('text-', 'badge-').replace('gst-', '')
              }`}>
                {status.label}
              </span>
              {cycle.final_decision && (
                <span className={`badge ${
                  cycle.final_decision.toLowerCase() === 'long' ? 'badge-green' :
                  cycle.final_decision.toLowerCase() === 'short' ? 'badge-red' : 'badge-gray'
                }`}>
                  {cycle.final_decision.toUpperCase()}
                  {cycle.final_conviction && ` (${cycle.final_conviction}%)`}
                </span>
              )}
            </div>
            <p className="text-gray-500 text-sm">
              {formatTimeAgo(cycle.timestamp_utc)} • 
              {cycle.proposals_count || 0} proposals • 
              {cycle.duration_seconds ? `${cycle.duration_seconds.toFixed(1)}s` : 'N/A'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {costTracking.total_cost_usd > 0 && (
            <span className="text-gray-400 text-sm">
              {formatCurrency(costTracking.total_cost_usd)}
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="mt-6 pt-6 border-t border-gst-border">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Cycle Details */}
            <div>
              <h4 className="text-white font-medium mb-3">Cycle Details</h4>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-gray-400">Cycle ID</dt>
                  <dd className="text-white font-mono text-xs">{cycle.id?.slice(0, 8)}...</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Type</dt>
                  <dd className="text-white">{cycle.cycle_type}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Started</dt>
                  <dd className="text-white">{formatDate(cycle.timestamp_utc)}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Duration</dt>
                  <dd className="text-white">{cycle.duration_seconds ? `${cycle.duration_seconds.toFixed(1)} seconds` : 'N/A'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Proposals</dt>
                  <dd className="text-white">{cycle.proposals_count || 0}</dd>
                </div>
                {cycle.restart_count > 0 && (
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Restarts</dt>
                    <dd className="text-yellow-400">{cycle.restart_count}</dd>
                  </div>
                )}
              </dl>
            </div>

            {/* Decision */}
            <div>
              <h4 className="text-white font-medium mb-3">Final Decision</h4>
              {cycle.final_decision ? (
                <div className={`p-4 rounded-lg ${decision.bg} border ${
                  cycle.final_decision.toLowerCase() === 'long' ? 'border-green-800' :
                  cycle.final_decision.toLowerCase() === 'short' ? 'border-red-800' : 'border-gray-700'
                }`}>
                  <div className="flex items-center gap-3">
                    <DecisionIcon className={`w-8 h-8 ${decision.color}`} />
                    <div>
                      <p className={`text-xl font-bold ${decision.color}`}>
                        {cycle.final_decision.toUpperCase()}
                      </p>
                      {cycle.final_conviction && (
                        <p className="text-gray-400 text-sm">
                          Conviction: {cycle.final_conviction}%
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500">No decision recorded</p>
              )}
            </div>
          </div>

          {/* Cost Breakdown */}
          {costTracking.by_source && Object.keys(costTracking.by_source).length > 0 && (
            <div className="mt-6">
              <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                <DollarSign className="w-4 h-4 text-gst-accent" />
                API Cost Breakdown
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(costTracking.by_source).map(([source, data]) => (
                  <div key={source} className="p-3 bg-gst-darker rounded-lg">
                    <p className="text-gray-400 text-sm capitalize">{source}</p>
                    <p className="text-white font-medium">
                      {formatCurrency(data.cost_usd || 0)}
                    </p>
                    <p className="text-gray-500 text-xs">
                      {(data.tokens || 0).toLocaleString()} tokens
                    </p>
                  </div>
                ))}
              </div>
              <div className="mt-3 flex justify-between items-center text-sm">
                <span className="text-gray-400">Total Cost</span>
                <span className="text-white font-medium">
                  {formatCurrency(costTracking.total_cost_usd || 0)}
                </span>
              </div>
            </div>
          )}

          {/* Error if any */}
          {cycle.error && (
            <div className="mt-6 p-4 bg-red-900/20 border border-red-800/50 rounded-lg">
              <p className="text-red-400 text-sm">{cycle.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ScanCycles() {
  const [expandedId, setExpandedId] = useState(null)

  const { data: cyclesData, isLoading, error } = useQuery({
    queryKey: ['scanCycles'],
    queryFn: () => api.getScanCycles(50),
  })

  const { data: costData } = useQuery({
    queryKey: ['costSummary'],
    queryFn: () => api.getCostSummary(7),
  })

  const cycles = cyclesData?.scan_cycles || []
  const costSummary = costData || {}

  // Calculate stats
  const completedCycles = cycles.filter(c => c.status === 'complete').length
  const failedCycles = cycles.filter(c => c.status === 'failed' || c.status === 'error').length
  const avgDuration = cycles.length > 0 
    ? cycles.reduce((sum, c) => sum + (c.duration_seconds || 0), 0) / cycles.length 
    : 0

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
            <p className="font-medium">Failed to load scan cycles</p>
            <p className="text-sm text-red-400/70">{error.message}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Scan Cycles</h2>
        <p className="text-gray-400 mt-1">
          Monitor the AI committee's scheduled and manual analysis cycles
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="stat-value text-white">{cycles.length}</p>
          <p className="stat-label">Total Cycles</p>
        </div>
        <div className="card text-center">
          <p className="stat-value text-gst-green">{completedCycles}</p>
          <p className="stat-label">Completed</p>
        </div>
        <div className="card text-center">
          <p className="stat-value text-gst-yellow">{avgDuration.toFixed(1)}s</p>
          <p className="stat-label">Avg Duration</p>
        </div>
        <div className="card text-center">
          <p className="stat-value text-gst-accent">
            {formatCurrency(costSummary.total_cost_usd || 0)}
          </p>
          <p className="stat-label">7-Day API Cost</p>
        </div>
      </div>

      {/* Cycles List */}
      <div className="space-y-4">
        {cycles.length > 0 ? (
          cycles.map((cycle) => (
            <ScanCycleCard 
              key={cycle.id} 
              cycle={cycle}
              isExpanded={expandedId === cycle.id}
              onToggle={() => setExpandedId(expandedId === cycle.id ? null : cycle.id)}
            />
          ))
        ) : (
          <div className="card text-center py-12">
            <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No scan cycles recorded</p>
            <p className="text-gray-500 text-sm mt-1">
              Cycles run automatically at 9:35 AM and 12:30 PM ET
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
