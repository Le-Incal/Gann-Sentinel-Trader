import { useQuery } from '@tanstack/react-query'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Briefcase,
  Activity,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  PauseCircle,
  Wallet
} from 'lucide-react'
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import api from '../api'
import { formatCurrency, formatPercent, formatDate, formatTimeAgo } from '../utils/format'

// Stat card component
function StatCard({ icon: Icon, label, value, change, changeType, iconColor }) {
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <p className="stat-label">{label}</p>
          <p className="stat-value mt-1">{value}</p>
          {change !== undefined && (
            <p className={`text-sm mt-1 flex items-center gap-1 ${
              changeType === 'positive' ? 'text-gst-green' : 
              changeType === 'negative' ? 'text-gst-red' : 'text-gray-400'
            }`}>
              {changeType === 'positive' && <TrendingUp className="w-4 h-4" />}
              {changeType === 'negative' && <TrendingDown className="w-4 h-4" />}
              {change}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${iconColor || 'bg-gst-accent/20'}`}>
          <Icon className={`w-6 h-6 ${iconColor ? iconColor.replace('bg-', 'text-').replace('/20', '') : 'text-gst-accent'}`} />
        </div>
      </div>
    </div>
  )
}

// Position row component
function PositionRow({ position }) {
  const pnlPositive = position.unrealized_pnl >= 0
  
  return (
    <tr className="table-row">
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-gst-accent/20 flex items-center justify-center">
            <span className="text-gst-accent font-semibold text-sm">
              {position.ticker?.slice(0, 2)}
            </span>
          </div>
          <div>
            <p className="text-white font-medium">{position.ticker}</p>
            <p className="text-gray-500 text-xs">{position.quantity} shares</p>
          </div>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <p className="text-white">{formatCurrency(position.current_price)}</p>
        <p className="text-gray-500 text-xs">Avg: {formatCurrency(position.avg_entry_price)}</p>
      </td>
      <td className="px-4 py-3 text-right">
        <p className="text-white">{formatCurrency(position.market_value)}</p>
      </td>
      <td className="px-4 py-3 text-right">
        <p className={pnlPositive ? 'text-gst-green' : 'text-gst-red'}>
          {pnlPositive ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
        </p>
        <p className={`text-xs ${pnlPositive ? 'text-gst-green' : 'text-gst-red'}`}>
          {pnlPositive ? '+' : ''}{formatPercent(position.unrealized_pnl_pct)}
        </p>
      </td>
    </tr>
  )
}

// Recent trade row
function TradeRow({ trade }) {
  const statusConfig = {
    'filled': { icon: CheckCircle, color: 'text-gst-green', bg: 'bg-green-900/30' },
    'pending_approval': { icon: Clock, color: 'text-gst-yellow', bg: 'bg-yellow-900/30' },
    'rejected': { icon: XCircle, color: 'text-gst-red', bg: 'bg-red-900/30' },
    'cancelled': { icon: XCircle, color: 'text-gray-400', bg: 'bg-gray-800/30' },
  }
  
  const status = statusConfig[trade.status] || statusConfig['pending_approval']
  const StatusIcon = status.icon
  
  return (
    <div className="flex items-center justify-between py-3 border-b border-gst-border last:border-0">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${status.bg}`}>
          <StatusIcon className={`w-4 h-4 ${status.color}`} />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className={`font-medium ${trade.side === 'buy' ? 'text-gst-green' : 'text-gst-red'}`}>
              {trade.side?.toUpperCase()}
            </span>
            <span className="text-white font-medium">{trade.ticker}</span>
            <span className="text-gray-400">× {trade.quantity}</span>
          </div>
          <p className="text-gray-500 text-xs">{formatTimeAgo(trade.created_at)}</p>
        </div>
      </div>
      <div className="text-right">
        <span className={`badge ${
          trade.status === 'filled' ? 'badge-green' :
          trade.status === 'pending_approval' ? 'badge-yellow' :
          trade.status === 'rejected' ? 'badge-red' : 'badge-gray'
        }`}>
          {trade.status?.replace('_', ' ')}
        </span>
      </div>
    </div>
  )
}

// Custom tooltip for chart
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  
  return (
    <div className="bg-gst-darker border border-gst-border rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-sm">{label}</p>
      <p className="text-white font-semibold">{formatCurrency(payload[0].value)}</p>
    </div>
  )
}

export default function Dashboard() {
  // Fetch dashboard data
  const { data: dashboard, isLoading, error } = useQuery({
    queryKey: ['dashboard'],
    queryFn: api.getDashboard,
  })

  // Fetch portfolio history for chart
  const { data: historyData } = useQuery({
    queryKey: ['portfolioHistory'],
    queryFn: () => api.getPortfolioHistory(30),
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
            <p className="font-medium">Failed to load dashboard</p>
            <p className="text-sm text-red-400/70">{error.message}</p>
          </div>
        </div>
      </div>
    )
  }

  const portfolio = dashboard?.portfolio || {}
  // Positions can be in portfolio.positions (from Railway) or dashboard.positions (from local)
  const positions = portfolio?.positions || dashboard?.positions || []
  const recentTrades = dashboard?.recent_trades || []
  const pendingCount = dashboard?.pending_trades?.length || 0

  // Calculate total unrealized P&L from positions if not in portfolio
  const totalUnrealizedPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0)
  
  // Determine daily P&L status (use unrealized if daily not available)
  const dailyPnl = portfolio.daily_pnl || totalUnrealizedPnl || 0
  const dailyPnlPct = portfolio.daily_pnl_pct || (positions[0]?.unrealized_pnl_pct * 100) || 0
  const dailyPnlType = dailyPnl >= 0 ? 'positive' : 'negative'

  // Mock chart data if no history (replace with real data)
  const chartData = historyData?.history || [
    { date: '2026-01-10', value: 98000 },
    { date: '2026-01-11', value: 98500 },
    { date: '2026-01-12', value: 99200 },
    { date: '2026-01-13', value: 98800 },
    { date: '2026-01-14', value: 100500 },
    { date: '2026-01-15', value: 101200 },
    { date: '2026-01-16', value: 100800 },
    { date: '2026-01-17', value: portfolio.total_value || 100000 },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Portfolio Dashboard</h2>
          <p className="text-gray-400 mt-1">Real-time view of your AI trading system</p>
        </div>
        <div className="text-right">
          <p className="text-gray-400 text-sm">Last updated</p>
          <p className="text-white font-mono">{formatDate(new Date())}</p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={DollarSign}
          label="Total Portfolio Value"
          value={formatCurrency(portfolio.total_value || 0)}
          change={`${positions.length} position${positions.length !== 1 ? 's' : ''}`}
          iconColor="bg-gst-accent/20"
        />
        <StatCard
          icon={Wallet}
          label="Cash"
          value={formatCurrency(portfolio.cash || 0)}
          change={`${((portfolio.cash / portfolio.total_value) * 100 || 0).toFixed(1)}% of portfolio`}
          iconColor="bg-green-500/20"
        />
        <StatCard
          icon={Briefcase}
          label="Positions Value"
          value={formatCurrency(portfolio.positions_value || 0)}
          change={`${((portfolio.positions_value / portfolio.total_value) * 100 || 0).toFixed(1)}% invested`}
          iconColor="bg-purple-500/20"
        />
        <StatCard
          icon={dailyPnl >= 0 ? TrendingUp : TrendingDown}
          label="Unrealized P&L"
          value={`${dailyPnl >= 0 ? '+' : ''}${formatCurrency(dailyPnl)}`}
          change={`${dailyPnl >= 0 ? '+' : ''}${dailyPnlPct.toFixed(2)}%`}
          changeType={dailyPnlType}
          iconColor={dailyPnl >= 0 ? "bg-green-500/20" : "bg-red-500/20"}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio Chart */}
        <div className="lg:col-span-2 card">
          <h3 className="card-header">
            <Activity className="w-5 h-5 text-gst-accent" />
            Portfolio Performance (30 Days)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis 
                  dataKey="date" 
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis 
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<ChartTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorValue)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Trades */}
        <div className="card">
          <h3 className="card-header">
            <TrendingUp className="w-5 h-5 text-gst-green" />
            Recent Trades
          </h3>
          <div className="space-y-0 -mx-6 -mb-6 px-6 pb-4 max-h-64 overflow-y-auto">
            {recentTrades.length > 0 ? (
              recentTrades.slice(0, 5).map((trade) => (
                <TradeRow key={trade.id} trade={trade} />
              ))
            ) : (
              <p className="text-gray-500 text-center py-8">No recent trades</p>
            )}
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="card">
        <h3 className="card-header">
          <Briefcase className="w-5 h-5 text-purple-400" />
          Current Positions
        </h3>
        {positions.length > 0 ? (
          <div className="overflow-x-auto -mx-6 -mb-6">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gst-border">
                  <th className="px-4 py-3 font-medium">Asset</th>
                  <th className="px-4 py-3 font-medium text-right">Price</th>
                  <th className="px-4 py-3 font-medium text-right">Value</th>
                  <th className="px-4 py-3 font-medium text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <PositionRow key={position.ticker} position={position} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No open positions</p>
            <p className="text-gray-500 text-sm mt-1">Positions will appear here when trades are executed</p>
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="card bg-gradient-to-r from-gst-card to-gst-darker">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-500/20 rounded-lg">
              <CheckCircle className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <h3 className="text-white font-semibold">System Status: Operational</h3>
              <p className="text-gray-400 text-sm">Multi-Agent Consensus Architecture (MACA) v3.0.0</p>
            </div>
          </div>
          <div className="text-right text-sm">
            <p className="text-gray-400">Next scheduled scan</p>
            <p className="text-white font-mono">9:35 AM ET / 12:30 PM ET</p>
          </div>
        </div>
      </div>
    </div>
  )
}
