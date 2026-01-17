import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { 
  MessageSquare, 
  Bell, 
  AlertTriangle, 
  CheckCircle, 
  Info,
  Clock,
  Filter,
  Search,
  RefreshCw
} from 'lucide-react'
import api from '../api'
import { formatTimeAgo, formatDate } from '../utils/format'

// Message type configurations
const MESSAGE_TYPES = {
  trade_signal: { 
    icon: Bell, 
    color: 'text-lime-600', 
    bg: 'bg-lime-50',
    border: 'border-lime-200',
    label: 'Trade Signal' 
  },
  trade_executed: { 
    icon: CheckCircle, 
    color: 'text-green-600', 
    bg: 'bg-green-50',
    border: 'border-green-200',
    label: 'Trade Executed' 
  },
  trade_rejected: { 
    icon: AlertTriangle, 
    color: 'text-red-500', 
    bg: 'bg-red-50',
    border: 'border-red-200',
    label: 'Trade Rejected' 
  },
  system: { 
    icon: Info, 
    color: 'text-schwab-600', 
    bg: 'bg-schwab-50',
    border: 'border-schwab-200',
    label: 'System' 
  },
  scan_complete: { 
    icon: RefreshCw, 
    color: 'text-purple-600', 
    bg: 'bg-purple-50',
    border: 'border-purple-200',
    label: 'Scan Complete' 
  },
  alert: { 
    icon: AlertTriangle, 
    color: 'text-amber-600', 
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    label: 'Alert' 
  },
}

// Individual message component
function MessageCard({ message }) {
  const config = MESSAGE_TYPES[message.type] || MESSAGE_TYPES.system
  const Icon = config.icon
  
  return (
    <div className={`card ${config.border} hover:shadow-md transition-shadow`}>
      <div className="flex gap-4">
        <div className={`p-3 rounded-xl ${config.bg} shrink-0`}>
          <Icon className={`w-5 h-5 ${config.color}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className={`text-xs font-semibold ${config.color} uppercase tracking-wide`}>
                {config.label}
              </span>
              <h3 className="text-gray-900 font-semibold mt-1">{message.title}</h3>
            </div>
            <span className="text-gray-400 text-xs whitespace-nowrap flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatTimeAgo(message.timestamp)}
            </span>
          </div>
          <p className="text-gray-600 text-sm mt-2 leading-relaxed">{message.content}</p>
          {message.metadata && (
            <div className="mt-3 flex flex-wrap gap-2">
              {message.metadata.ticker && (
                <span className="badge badge-blue">{message.metadata.ticker}</span>
              )}
              {message.metadata.action && (
                <span className={`badge ${message.metadata.action === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                  {message.metadata.action}
                </span>
              )}
              {message.metadata.confidence && (
                <span className="badge badge-gray">{Math.round(message.metadata.confidence * 100)}% confidence</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Filter button component
function FilterButton({ active, onClick, children, count }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
        active 
          ? 'bg-lime-500 text-white shadow-sm' 
          : 'text-gray-600 hover:bg-gray-100'
      }`}
    >
      {children}
      {count !== undefined && (
        <span className={`ml-2 px-1.5 py-0.5 rounded text-xs ${
          active ? 'bg-lime-600' : 'bg-gray-200 text-gray-600'
        }`}>
          {count}
        </span>
      )}
    </button>
  )
}

export default function Messages() {
  const [filter, setFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')

  // For now, generate sample messages since we don't have a messages API yet
  // This would be replaced with a real API call
  const sampleMessages = [
    {
      id: 1,
      type: 'scan_complete',
      title: 'Scheduled Scan Complete',
      content: 'Morning scan completed at 9:35 AM ET. Analyzed 27 event types across market data. 2 potential signals identified for committee review.',
      timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      metadata: {}
    },
    {
      id: 2,
      type: 'trade_signal',
      title: 'New Trade Signal: ARRY',
      content: 'AI Committee has reached consensus on ARRY. Grok identified narrative momentum from solar energy policy discussions. Perplexity confirmed recent earnings beat. Committee vote: 3-1 in favor.',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
      metadata: { ticker: 'ARRY', action: 'BUY', confidence: 0.78 }
    },
    {
      id: 3,
      type: 'trade_executed',
      title: 'Trade Executed: ARRY',
      content: 'Successfully purchased 1,136 shares of ARRY at $9.82 per share. Total position value: $11,155.52. Order filled via Alpaca.',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 3).toISOString(),
      metadata: { ticker: 'ARRY', action: 'BUY' }
    },
    {
      id: 4,
      type: 'system',
      title: 'System Started',
      content: 'Gann Sentinel Trader v3.0 started successfully. All AI models connected. Portfolio synced from Alpaca. Ready for trading.',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
      metadata: {}
    },
    {
      id: 5,
      type: 'alert',
      title: 'Market Volatility Alert',
      content: 'VIX has risen above 20. Risk engine has increased caution thresholds. Position sizing reduced by 25% until volatility normalizes.',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
      metadata: {}
    },
    {
      id: 6,
      type: 'scan_complete',
      title: 'Midday Scan Complete',
      content: 'Afternoon scan completed at 12:30 PM ET. No actionable signals identified. All positions within risk parameters.',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString(),
      metadata: {}
    },
  ]

  // Filter messages
  const filteredMessages = sampleMessages.filter(msg => {
    if (filter !== 'all' && msg.type !== filter) return false
    if (searchQuery && !msg.title.toLowerCase().includes(searchQuery.toLowerCase()) && 
        !msg.content.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  // Count by type
  const counts = {
    all: sampleMessages.length,
    trade_signal: sampleMessages.filter(m => m.type === 'trade_signal').length,
    trade_executed: sampleMessages.filter(m => m.type === 'trade_executed').length,
    scan_complete: sampleMessages.filter(m => m.type === 'scan_complete').length,
    alert: sampleMessages.filter(m => m.type === 'alert').length,
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Message Center</h2>
          <p className="text-gray-500 mt-1">System notifications, trade alerts, and scan results</p>
        </div>
        <div className="text-right">
          <p className="text-gray-500 text-sm">{filteredMessages.length} messages</p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="card">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search messages..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-lime-500/20 focus:border-lime-500"
            />
          </div>
          
          {/* Filter buttons */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            <FilterButton active={filter === 'all'} onClick={() => setFilter('all')} count={counts.all}>
              All
            </FilterButton>
            <FilterButton active={filter === 'trade_signal'} onClick={() => setFilter('trade_signal')} count={counts.trade_signal}>
              Signals
            </FilterButton>
            <FilterButton active={filter === 'trade_executed'} onClick={() => setFilter('trade_executed')} count={counts.trade_executed}>
              Executed
            </FilterButton>
            <FilterButton active={filter === 'scan_complete'} onClick={() => setFilter('scan_complete')} count={counts.scan_complete}>
              Scans
            </FilterButton>
            <FilterButton active={filter === 'alert'} onClick={() => setFilter('alert')} count={counts.alert}>
              Alerts
            </FilterButton>
          </div>
        </div>
      </div>

      {/* Messages List */}
      <div className="space-y-4">
        {filteredMessages.length > 0 ? (
          filteredMessages.map((message) => (
            <MessageCard key={message.id} message={message} />
          ))
        ) : (
          <div className="card text-center py-12">
            <MessageSquare className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No messages found</p>
            <p className="text-gray-400 text-sm mt-1">
              {searchQuery ? 'Try a different search term' : 'Messages will appear here as the system operates'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
