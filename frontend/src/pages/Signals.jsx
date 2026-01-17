import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { 
  Radio, 
  AlertTriangle,
  Filter,
  ExternalLink,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap
} from 'lucide-react'
import api from '../api'
import { formatDate, formatTimeAgo, truncate } from '../utils/format'

const sourceConfig = {
  'fred': { label: 'FRED', color: 'badge-blue', icon: '📊' },
  'polymarket': { label: 'Polymarket', color: 'badge-purple', icon: '🎯' },
  'grok': { label: 'Grok', color: 'badge-yellow', icon: '🤖' },
  'technical': { label: 'Technical', color: 'badge-green', icon: '📈' },
  'event': { label: 'Event', color: 'badge-red', icon: '📅' },
}

const signalTypeConfig = {
  'macro': { icon: TrendingUp, color: 'text-blue-400' },
  'sentiment': { icon: Zap, color: 'text-yellow-400' },
  'technical': { icon: TrendingUp, color: 'text-green-400' },
  'event': { icon: Clock, color: 'text-red-400' },
  'prediction': { icon: TrendingUp, color: 'text-purple-400' },
}

function SignalCard({ signal }) {
  const [expanded, setExpanded] = useState(false)
  const data = signal.data || signal
  const source = sourceConfig[signal.source?.toLowerCase()] || { label: signal.source, color: 'badge-gray', icon: '📡' }
  const signalType = signalTypeConfig[signal.signal_type?.toLowerCase()] || signalTypeConfig['macro']
  const SignalIcon = signalType.icon

  // Extract meaningful data from the signal
  const headline = data.headline || data.title || data.summary?.slice(0, 100) || signal.signal_type
  const description = data.description || data.summary || data.analysis
  const tickers = data.asset_scope?.tickers || (signal.ticker ? [signal.ticker] : [])
  const sentiment = data.sentiment || data.direction

  return (
    <div className="card hover:border-gst-accent/50 transition-colors">
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className={`p-3 rounded-lg bg-gst-darker ${signalType.color}`}>
          <span className="text-2xl">{source.icon}</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <span className={`badge ${source.color}`}>{source.label}</span>
                {signal.signal_type && (
                  <span className="badge badge-gray">{signal.signal_type}</span>
                )}
                {tickers.map(ticker => (
                  <span key={ticker} className="badge badge-blue">{ticker}</span>
                ))}
                {sentiment && (
                  <span className={`badge ${
                    sentiment === 'bullish' || sentiment === 'positive' ? 'badge-green' :
                    sentiment === 'bearish' || sentiment === 'negative' ? 'badge-red' : 'badge-gray'
                  }`}>
                    {sentiment}
                  </span>
                )}
              </div>
              <h3 className="text-white font-medium">
                {truncate(headline, 150)}
              </h3>
            </div>
            <div className="text-right shrink-0">
              <p className="text-gray-400 text-sm">{formatTimeAgo(signal.timestamp_utc || signal.created_at)}</p>
            </div>
          </div>

          {/* Expandable description */}
          {description && (
            <div className="mt-3">
              <p className="text-gray-400 text-sm">
                {expanded ? description : truncate(description, 200)}
              </p>
              {description.length > 200 && (
                <button 
                  onClick={() => setExpanded(!expanded)}
                  className="text-sm text-gst-accent hover:text-gst-accent-hover mt-1"
                >
                  {expanded ? 'Show less' : 'Show more'}
                </button>
              )}
            </div>
          )}

          {/* Metadata */}
          <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
            {signal.staleness_seconds !== undefined && (
              <span>Staleness: {signal.staleness_seconds}s</span>
            )}
            {data.confidence && (
              <span>Confidence: {Math.round(data.confidence * 100)}%</span>
            )}
            {data.source_url && (
              <a 
                href={data.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-gst-accent hover:text-gst-accent-hover"
              >
                Source <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Signals() {
  const [sourceFilter, setSourceFilter] = useState('all')
  const [typeFilter, setTypeFilter] = useState('all')

  const { data: signalsData, isLoading, error } = useQuery({
    queryKey: ['signals'],
    queryFn: () => api.getSignals(100),
  })

  const signals = signalsData?.signals || []

  // Get unique sources and types for filters
  const sources = [...new Set(signals.map(s => s.source).filter(Boolean))]
  const types = [...new Set(signals.map(s => s.signal_type).filter(Boolean))]

  // Filter signals
  const filteredSignals = signals.filter(signal => {
    if (sourceFilter !== 'all' && signal.source?.toLowerCase() !== sourceFilter.toLowerCase()) return false
    if (typeFilter !== 'all' && signal.signal_type?.toLowerCase() !== typeFilter.toLowerCase()) return false
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
            <p className="font-medium">Failed to load signals</p>
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
        <h2 className="text-2xl font-bold text-white">Signal History</h2>
        <p className="text-gray-400 mt-1">
          Real-time signals from FRED, Polymarket, Grok, and Technical scanners
        </p>
      </div>

      {/* Source Overview */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {Object.entries(sourceConfig).map(([key, config]) => {
          const count = signals.filter(s => s.source?.toLowerCase() === key).length
          return (
            <button
              key={key}
              onClick={() => setSourceFilter(sourceFilter === key ? 'all' : key)}
              className={`card text-center transition-colors ${
                sourceFilter === key ? 'border-gst-accent' : ''
              }`}
            >
              <span className="text-3xl">{config.icon}</span>
              <p className="text-white font-semibold mt-2">{count}</p>
              <p className="text-gray-400 text-sm">{config.label}</p>
            </button>
          )
        })}
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-500" />
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            className="px-4 py-2.5 bg-gst-card border border-gst-border rounded-lg text-white focus:outline-none focus:border-gst-accent"
          >
            <option value="all">All Sources</option>
            {sources.map(source => (
              <option key={source} value={source}>{source}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="px-4 py-2.5 bg-gst-card border border-gst-border rounded-lg text-white focus:outline-none focus:border-gst-accent"
          >
            <option value="all">All Types</option>
            {types.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {(sourceFilter !== 'all' || typeFilter !== 'all') && (
          <button
            onClick={() => { setSourceFilter('all'); setTypeFilter('all'); }}
            className="text-gst-accent hover:text-gst-accent-hover text-sm"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Signals List */}
      <div className="space-y-4">
        {filteredSignals.length > 0 ? (
          filteredSignals.map((signal) => (
            <SignalCard key={signal.id} signal={signal} />
          ))
        ) : (
          <div className="card text-center py-12">
            <Radio className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No signals found</p>
            <p className="text-gray-500 text-sm mt-1">
              Signals from scanners will appear here in real-time
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
