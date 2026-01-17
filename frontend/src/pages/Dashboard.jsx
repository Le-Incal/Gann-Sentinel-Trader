import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
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
  Wallet,
  Radio
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
  
  // Format label - could be a time string or date
  const displayLabel = label?.includes(':') ? label : 
    (label ? new Date(label).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '')
  
  return (
    <div className="bg-gst-darker border border-gst-border rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-sm">{displayLabel}</p>
      <p className="text-white font-semibold">{formatCurrency(payload[0].value)}</p>
    </div>
  )
}

// Time period options
const TIME_PERIODS = [
  { key: 'live', label: 'Live', days: 0 },
  { key: '1D', label: '1D', days: 1 },
  { key: '1W', label: '1W', days: 7 },
  { key: '1M', label: '1M', days: 30 },
  { key: '3M', label: '3M', days: 90 },
  { key: '6M', label: '6M', days: 180 },
  { key: '1Y', label: '1Y', days: 365 },
  { key: 'ALL', label: 'All', days: 9999 },
]

// Check if market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)
function isMarketOpen() {
  const now = new Date()
  
  // Convert to ET (approximate - doesn't handle DST perfectly)
  const etOffset = -5 // EST
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000)
  const et = new Date(utc + (3600000 * etOffset))
  
  const day = et.getDay() // 0 = Sunday, 6 = Saturday
  const hours = et.getHours()
  const minutes = et.getMinutes()
  const timeInMinutes = hours * 60 + minutes
  
  // Market hours: 9:30 AM (570 min) to 4:00 PM (960 min)
  const marketOpen = 9 * 60 + 30  // 9:30 AM = 570
  const marketClose = 16 * 60     // 4:00 PM = 960
  
  // Check if weekday and within market hours
  const isWeekday = day >= 1 && day <= 5
  const isDuringHours = timeInMinutes >= marketOpen && timeInMinutes < marketClose
  
  return isWeekday && isDuringHours
}

// Time period selector component
function TimePeriodSelector({ selected, onSelect }) {
  const marketOpen = isMarketOpen()
  
  return (
    <div className="flex items-center gap-1 bg-gst-darker rounded-lg p-1">
      {TIME_PERIODS.map((period) => {
        const isLive = period.key === 'live'
        const isDisabled = isLive && !marketOpen
        
        return (
          <button
            key={period.key}
            onClick={() => !isDisabled && onSelect(period)}
            disabled={isDisabled}
            title={isDisabled ? 'Market is closed' : ''}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              isDisabled
                ? 'text-gray-600 cursor-not-allowed'
                : selected.key === period.key
                  ? 'bg-gst-accent text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gst-card'
            }`}
          >
            {isLive && (
              <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 ${
                marketOpen ? 'bg-green-500 animate-pulse' : 'bg-gray-600'
              }`} />
            )}
            {period.label}
          </button>
        )
      })}
    </div>
  )
}

export default function Dashboard() {
  // Time period state - default to 1D if market closed, otherwise 1M
  const [timePeriod, setTimePeriod] = useState(() => {
    return isMarketOpen() ? TIME_PERIODS[0] : TIME_PERIODS[3] // Live if open, else 1M
  })
  
  // Fetch dashboard data
  const { data: dashboard, isLoading, error } = useQuery({
    queryKey: ['dashboard'],
    queryFn: api.getDashboard,
  })

  // Fetch portfolio history for chart based on selected time period
  const { data: historyData } = useQuery({
    queryKey: ['portfolioHistory', timePeriod.days],
    queryFn: () => api.getPortfolioHistory(timePeriod.days || 30),
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

  // Generate chart data based on time period
  const generateChartData = () => {
    const currentValue = portfolio.total_value || 100000
    
    if (timePeriod.key === 'live') {
      // Live: Show intraday data (minute by minute for market hours)
      const now = new Date()
      const marketOpen = new Date(now)
      marketOpen.setHours(9, 30, 0, 0) // 9:30 AM
      
      const data = []
      const minutesSinceOpen = Math.max(0, Math.floor((now - marketOpen) / (1000 * 60)))
      const intervals = Math.min(minutesSinceOpen, 390) // Max 6.5 hours of trading
      
      // Generate data points every 5 minutes - nearly flat line
      for (let i = 0; i <= intervals; i += 5) {
        const time = new Date(marketOpen.getTime() + i * 60 * 1000)
        data.push({
          date: time.toISOString(),
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
          value: currentValue // Flat line - no mock variance
        })
      }
      
      // Add current value as last point
      if (data.length > 0) {
        data[data.length - 1].value = currentValue
      } else {
        data.push({ date: now.toISOString(), time: 'Now', value: currentValue })
      }
      
      return data
    }
    
    // For other periods, use API data or generate mock data
    if (historyData?.history) return historyData.history
    
    // Show flat line at current value until real historical data is available
    const days = Math.min(timePeriod.days || 30, 365)
    const data = []
    
    for (let i = days; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      data.push({
        date: date.toISOString().split('T')[0],
        value: currentValue // Flat line - no mock data
      })
    }
    
    return data
  }
  
  // Simple hash function for consistent random-looking data
  const hash = (str) => {
    let h = 0
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) - h) + str.charCodeAt(i)
      h |= 0
    }
    return Math.abs(h)
  }
  
  const chartData = generateChartData()
  const isLiveMode = timePeriod.key === 'live'

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
          <div className="flex items-center justify-between mb-4">
            <h3 className="card-header mb-0">
              <Activity className="w-5 h-5 text-gst-accent" />
              Portfolio Performance
            </h3>
            <TimePeriodSelector selected={timePeriod} onSelect={setTimePeriod} />
          </div>
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
                  dataKey={isLiveMode ? "time" : "date"}
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  tickFormatter={(val) => {
                    if (isLiveMode) return val // Already formatted as 24h time
                    return new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                  }}
                  interval={isLiveMode ? Math.floor(chartData.length / 6) : 'preserveEnd'}
                  minTickGap={isLiveMode ? 30 : 20}
                  tickMargin={6}
                />
                <YAxis 
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                  domain={['auto', 'auto']}
                  tickMargin={6}
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
