import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { 
  MessageSquare, 
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  User,
  ThumbsUp,
  ThumbsDown,
  Minus,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react'
import api from '../api'
import { formatDate, formatTimeAgo } from '../utils/format'

// AI analyst configuration
const analystConfig = {
  'grok': { 
    name: 'Grok', 
    role: 'Narrative Momentum', 
    color: 'border-yellow-500', 
    bg: 'bg-yellow-900/20',
    textColor: 'text-yellow-400',
    avatar: '🔮'
  },
  'perplexity': { 
    name: 'Perplexity', 
    role: 'External Reality', 
    color: 'border-blue-500', 
    bg: 'bg-blue-900/20',
    textColor: 'text-blue-400',
    avatar: '🔍'
  },
  'chatgpt': { 
    name: 'ChatGPT', 
    role: 'Sentiment & Bias', 
    color: 'border-green-500', 
    bg: 'bg-green-900/20',
    textColor: 'text-green-400',
    avatar: '🧠'
  },
  'claude': { 
    name: 'Claude', 
    role: 'Technical Validator', 
    color: 'border-purple-500', 
    bg: 'bg-purple-900/20',
    textColor: 'text-purple-400',
    avatar: '📊'
  },
  'chair': { 
    name: 'Chair', 
    role: 'Synthesizer', 
    color: 'border-red-500', 
    bg: 'bg-red-900/20',
    textColor: 'text-red-400',
    avatar: '👔'
  },
}

function VoteBadge({ vote }) {
  if (!vote) return null
  
  const action = vote.action?.toUpperCase()
  
  if (action === 'LONG' || action === 'BUY') {
    return (
      <span className="badge badge-green">
        <ThumbsUp className="w-3 h-3 mr-1" />
        LONG {vote.ticker} ({vote.confidence}%)
      </span>
    )
  }
  
  if (action === 'SHORT' || action === 'SELL') {
    return (
      <span className="badge badge-red">
        <ThumbsDown className="w-3 h-3 mr-1" />
        SHORT {vote.ticker} ({vote.confidence}%)
      </span>
    )
  }
  
  return (
    <span className="badge badge-gray">
      <Minus className="w-3 h-3 mr-1" />
      HOLD ({vote.confidence || 0}%)
    </span>
  )
}

function DebateTurn({ turn }) {
  const speaker = turn.speaker?.toLowerCase() || 'unknown'
  const config = analystConfig[speaker] || {
    name: turn.speaker || 'Unknown',
    role: 'Analyst',
    color: 'border-gray-500',
    bg: 'bg-gray-900/20',
    textColor: 'text-gray-400',
    avatar: '🤖'
  }
  
  const vote = turn.vote_action ? {
    action: turn.vote_action,
    ticker: turn.vote_ticker,
    confidence: turn.vote_confidence
  } : null

  // Parse agreements/disagreements if they're strings
  let agreements = turn.agreements
  let disagreements = turn.disagreements
  
  if (typeof agreements === 'string') {
    try { agreements = JSON.parse(agreements) } catch { agreements = [] }
  }
  if (typeof disagreements === 'string') {
    try { disagreements = JSON.parse(disagreements) } catch { disagreements = [] }
  }

  return (
    <div className={`border-l-4 ${config.color} pl-4 py-3`}>
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-full ${config.bg} flex items-center justify-center text-xl`}>
          {config.avatar}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`font-semibold ${config.textColor}`}>{config.name}</span>
            <span className="text-gray-500 text-sm">{config.role}</span>
            <span className="text-gray-600 text-sm">• Round {turn.round}</span>
            {turn.changed_mind === 1 && (
              <span className="badge badge-yellow">Changed Position</span>
            )}
          </div>
          
          {/* Message */}
          {turn.message && (
            <p className="text-gray-300 mt-2 whitespace-pre-wrap">{turn.message}</p>
          )}

          {/* Agreements/Disagreements */}
          <div className="flex flex-wrap gap-2 mt-3">
            {agreements && agreements.length > 0 && (
              <div className="flex items-center gap-1 text-sm text-green-400">
                <CheckCircle className="w-4 h-4" />
                Agrees with: {agreements.join(', ')}
              </div>
            )}
            {disagreements && disagreements.length > 0 && (
              <div className="flex items-center gap-1 text-sm text-red-400">
                <XCircle className="w-4 h-4" />
                Disagrees with: {disagreements.join(', ')}
              </div>
            )}
          </div>

          {/* Vote */}
          {vote && (
            <div className="mt-3">
              <VoteBadge vote={vote} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function DebateSession({ debate, isExpanded, onToggle }) {
  const turns = debate.turns || []
  
  // Calculate vote summary
  const votes = turns.filter(t => t.vote_action).reduce((acc, turn) => {
    const action = turn.vote_action?.toUpperCase()
    if (action === 'LONG' || action === 'BUY') acc.long++
    else if (action === 'SHORT' || action === 'SELL') acc.short++
    else acc.hold++
    return acc
  }, { long: 0, short: 0, hold: 0 })

  const totalVotes = votes.long + votes.short + votes.hold
  const outcome = votes.long > votes.short ? 'LONG' : 
                  votes.short > votes.long ? 'SHORT' : 'HOLD'

  return (
    <div className="card">
      {/* Header - Always visible */}
      <button 
        onClick={onToggle}
        className="w-full flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className={`p-2 rounded-lg ${
            outcome === 'LONG' ? 'bg-green-900/30' :
            outcome === 'SHORT' ? 'bg-red-900/30' : 'bg-gray-800/30'
          }`}>
            <MessageSquare className={`w-5 h-5 ${
              outcome === 'LONG' ? 'text-gst-green' :
              outcome === 'SHORT' ? 'text-gst-red' : 'text-gray-400'
            }`} />
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2">
              <span className="text-white font-medium">
                Debate Session
              </span>
              <span className={`badge ${
                outcome === 'LONG' ? 'badge-green' :
                outcome === 'SHORT' ? 'badge-red' : 'badge-gray'
              }`}>
                {outcome}
              </span>
            </div>
            <p className="text-gray-500 text-sm">
              {formatTimeAgo(debate.created_at)} • {turns.length} turns • 
              Votes: {votes.long} Long / {votes.short} Short / {votes.hold} Hold
            </p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="mt-6 pt-6 border-t border-gst-border space-y-4">
          {turns.length > 0 ? (
            turns.map((turn, idx) => (
              <DebateTurn key={turn.id || idx} turn={turn} />
            ))
          ) : (
            <p className="text-gray-500 text-center py-4">No debate turns recorded</p>
          )}
        </div>
      )}
    </div>
  )
}

export default function Debates() {
  const [expandedId, setExpandedId] = useState(null)

  const { data: debatesData, isLoading, error } = useQuery({
    queryKey: ['debates'],
    queryFn: () => api.getDebates(50),
  })

  const debates = debatesData?.debates || []

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
            <p className="font-medium">Failed to load debates</p>
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
        <h2 className="text-2xl font-bold text-white">AI Committee Debates</h2>
        <p className="text-gray-400 mt-1">
          Watch the multi-agent investment committee deliberate on trading decisions
        </p>
      </div>

      {/* How it works */}
      <div className="card bg-gradient-to-r from-gst-card to-gst-darker">
        <h3 className="text-white font-semibold mb-3">How the Committee Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {Object.entries(analystConfig).filter(([k]) => k !== 'chair').map(([key, config]) => (
            <div key={key} className={`p-3 rounded-lg ${config.bg} border ${config.color}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xl">{config.avatar}</span>
                <span className={`font-medium ${config.textColor}`}>{config.name}</span>
              </div>
              <p className="text-gray-400 text-sm">{config.role}</p>
            </div>
          ))}
        </div>
        <p className="text-gray-500 text-sm mt-4">
          Each analyst speaks twice per debate. Disagreement triggers deeper analysis. 
          Majority vote wins, with Chair tie-breaking on 2-2 splits.
        </p>
      </div>

      {/* Debates List */}
      <div className="space-y-4">
        {debates.length > 0 ? (
          debates.map((debate) => (
            <DebateSession 
              key={debate.id} 
              debate={debate}
              isExpanded={expandedId === debate.id}
              onToggle={() => setExpandedId(expandedId === debate.id ? null : debate.id)}
            />
          ))
        ) : (
          <div className="card text-center py-12">
            <MessageSquare className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No debates recorded yet</p>
            <p className="text-gray-500 text-sm mt-1">
              Committee deliberations will appear here after scan cycles
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
