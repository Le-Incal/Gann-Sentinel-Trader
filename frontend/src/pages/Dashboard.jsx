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
  LineChart,
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts'
import api from '../api'
import { formatCurrency, formatPercent, formatDate, formatTimeAgo } from '../utils/format'

// Stat card component
function StatCard({ icon: Icon, label, value, change, changeType, iconColor, iconTextColor }) {
  return (
    <div className="card group">
      <div className="flex items-start justify-between">
        <div>
          <p className="stat-label">{label}</p>
          <p className="stat-value mt-1">{value}</p>
          {change !== undefined && (
            <p className={`text-sm mt-2 flex items-center gap-1 font-medium ${
              changeType === 'positive' ? 'text-lime-600' : 
              changeType === 'negative' ? 'text-red-500' : 'text-gray-500'
            }`}>
              {changeType === 'positive' && <TrendingUp className="w-4 h-4" />}
              {changeType === 'negative' && <TrendingDown className="w-4 h-4" />}
              {change}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-xl ${iconColor || 'bg-lime-100'} transition-all group-hover:scale-110`}>
          <Icon className={`w-6 h-6 ${iconTextColor || 'text-lime-600'}`} />
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
      <td className="px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-lime-100 to-schwab-100 flex items-center justify-center border border-lime-200">
            <span className="text-lime-700 font-bold text-sm">
              {position.ticker?.slice(0, 2)}
            </span>
          </div>
          <div>
            <p className="text-gray-900 font-semibold">{position.ticker}</p>
            <p className="text-gray-500 text-xs">{position.quantity} shares</p>
          </div>
        </div>
      </td>
      <td className="px-4 py-4 text-right">
        <p className="text-gray-900 font-medium">{formatCurrency(position.current_price)}</p>
        <p className="text-gray-500 text-xs">Avg: {formatCurrency(position.avg_entry_price)}</p>
      </td>
      <td className="px-4 py-4 text-right">
        <p className="text-gray-900 font-medium">{formatCurrency(position.market_value)}</p>
      </td>
      <td className="px-4 py-4 text-right">
        <p className={`font-semibold ${pnlPositive ? 'text-lime-600' : 'text-red-500'}`}>
          {pnlPositive ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
        </p>
        <p className={`text-xs font-medium ${pnlPositive ? 'text-lime-600' : 'text-red-500'}`}>
          {pnlPositive ? '+' : ''}{formatPercent(position.unrealized_pnl_pct)}
        </p>
      </td>
    </tr>
  )
}

// Recent trade row
function TradeRow({ trade }) {
  const statusConfig = {
    'filled': { icon: CheckCircle, color: 'text-lime-600', bg: 'bg-lime-50' },
    'pending_approval': { icon: Clock, color: 'text-amber-600', bg: 'bg-amber-50' },
    'rejected': { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50' },
    'cancelled': { icon: XCircle, color: 'text-gray-400', bg: 'bg-gray-50' },
  }
  
  const status = statusConfig[trade.status] || statusConfig['pending_approval']
  const StatusIcon = status.icon
  
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${status.bg}`}>
          <StatusIcon className={`w-4 h-4 ${status.color}`} />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className={`font-medium ${trade.side === 'buy' ? 'text-lime-600' : 'text-red-500'}`}>
              {trade.side?.toUpperCase()}
            </span>
            <span className="text-gray-900 font-medium">{trade.ticker}</span>
            <span className="text-gray-500">× {trade.quantity}</span>
          </div>
          <p className="text-gray-400 text-xs">{formatTimeAgo(trade.created_at)}</p>
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
  
  const colors = {
    portfolio: '#00c853',  // Lime green
    spy: '#00a3e0',        // Schwab blue
    qqq: '#f59e0b'         // Warning orange
  }
  
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 shadow-lg">
      <p className="text-gray-600 text-sm mb-2 font-medium">{displayLabel}</p>
      {payload.map((entry, idx) => (
        <div key={idx} className="flex items-center justify-between gap-4 text-sm">
          <span style={{ color: colors[entry.dataKey] || entry.color }} className="font-medium">
            {entry.dataKey === 'portfolio' ? 'Portfolio' : entry.dataKey.toUpperCase()}
          </span>
          <span className={entry.value >= 0 ? 'text-lime-600 font-semibold' : 'text-red-500 font-semibold'}>
            {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(2)}%
          </span>
        </div>
      ))}
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
    <div className="flex items-center gap-1 bg-gray-50 border border-gray-200 rounded-xl p-1">
      {TIME_PERIODS.map((period) => {
        const isLive = period.key === 'live'
        const isDisabled = isLive && !marketOpen
        
        return (
          <button
            key={period.key}
            onClick={() => !isDisabled && onSelect(period)}
            disabled={isDisabled}
            title={isDisabled ? 'Market is closed' : ''}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all duration-200 ${
              isDisabled
                ? 'text-gray-300 cursor-not-allowed'
                : selected.key === period.key
                  ? 'bg-lime-500 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white'
            }`}
          >
            {isLive && (
              <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${
                marketOpen ? 'bg-lime-400 animate-pulse' : 'bg-gray-300'
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
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-lime-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card border-red-200 bg-red-50">
        <div className="flex items-center gap-3 text-red-600">
          <AlertTriangle className="w-6 h-6" />
          <div>
            <p className="font-medium">Failed to load dashboard</p>
            <p className="text-sm text-red-500">{error.message}</p>
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

  // Simple hash function for consistent benchmark data
  const hash = (str) => {
    let h = 0
    for (let i = 0; i < str.length; i++) {
      h = ((h << 5) - h) + str.charCodeAt(i)
      h |= 0
    }
    return Math.abs(h)
  }

  // Generate benchmark data (SPY, QQQ) - simulated realistic movements
  const generateBenchmarkData = (ticker, days, startValue = 100) => {
    const data = []
    let value = startValue
    
    // Simulate realistic daily returns based on ticker
    const avgReturn = ticker === 'SPY' ? 0.0003 : 0.0004 // QQQ slightly more volatile
    const volatility = ticker === 'SPY' ? 0.008 : 0.012
    
    for (let i = days; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      
      // Use hash for consistent "random" data
      const seed = hash(date.toISOString() + ticker)
      const dailyReturn = avgReturn + ((seed % 1000) / 1000 - 0.5) * volatility * 2
      
      if (i < days) {
        value = value * (1 + dailyReturn)
      }
      
      data.push({
        date: date.toISOString().split('T')[0],
        value: value
      })
    }
    
    return data
  }

  // Generate chart data based on time period with benchmarks
  const generateChartData = () => {
    const currentValue = portfolio.total_value || 100000
    const days = Math.min(timePeriod.days || 30, 365)
    
    if (timePeriod.key === 'live') {
      // Live: Show intraday data
      const now = new Date()
      const marketOpen = new Date(now)
      marketOpen.setHours(9, 30, 0, 0)
      
      const data = []
      const minutesSinceOpen = Math.max(0, Math.floor((now - marketOpen) / (1000 * 60)))
      const intervals = Math.min(minutesSinceOpen, 390)
      
      for (let i = 0; i <= intervals; i += 5) {
        const time = new Date(marketOpen.getTime() + i * 60 * 1000)
        const seed = hash(time.toISOString())
        
        // Small intraday movements for benchmarks
        const spyChange = ((seed % 100) - 50) / 5000
        const qqqChange = ((seed % 120) - 60) / 4000
        
        data.push({
          date: time.toISOString(),
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
          portfolio: 0, // Flat at 0% for now
          spy: spyChange * 100,
          qqq: qqqChange * 100
        })
      }
      
      if (data.length === 0) {
        data.push({ date: now.toISOString(), time: 'Now', portfolio: 0, spy: 0, qqq: 0 })
      }
      
      return data
    }
    
    // Historical data - normalize to percentage change from start
    const spyData = generateBenchmarkData('SPY', days)
    const qqqData = generateBenchmarkData('QQQ', days)
    
    const spyStart = spyData[0]?.value || 100
    const qqqStart = qqqData[0]?.value || 100
    const portfolioStart = currentValue // Assume flat for now until real history
    
    const data = []
    for (let i = 0; i <= days; i++) {
      const date = new Date()
      date.setDate(date.getDate() - (days - i))
      
      data.push({
        date: date.toISOString().split('T')[0],
        portfolio: 0, // Flat until real data - 0% change
        spy: ((spyData[i]?.value || spyStart) / spyStart - 1) * 100,
        qqq: ((qqqData[i]?.value || qqqStart) / qqqStart - 1) * 100
      })
    }
    
    return data
  }
  
  const chartData = generateChartData()
  const isLiveMode = timePeriod.key === 'live'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Portfolio Dashboard</h2>
          <p className="text-gray-500 mt-1">Real-time view of your AI trading system</p>
        </div>
        <div className="text-right">
          <p className="text-gray-500 text-sm">Last updated</p>
          <p className="text-gray-900 font-mono">{formatDate(new Date())}</p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={DollarSign}
          label="Total Portfolio Value"
          value={formatCurrency(portfolio.total_value || 0)}
          change={`${positions.length} position${positions.length !== 1 ? 's' : ''}`}
          iconColor="bg-lime-100"
          iconTextColor="text-lime-600"
        />
        <StatCard
          icon={Wallet}
          label="Cash"
          value={formatCurrency(portfolio.cash || 0)}
          change={`${((portfolio.cash / portfolio.total_value) * 100 || 0).toFixed(1)}% available`}
          iconColor="bg-schwab-100"
          iconTextColor="text-schwab-600"
        />
        <StatCard
          icon={Briefcase}
          label="Positions Value"
          value={formatCurrency(portfolio.positions_value || 0)}
          change={`${((portfolio.positions_value / portfolio.total_value) * 100 || 0).toFixed(1)}% invested`}
          iconColor="bg-purple-100"
          iconTextColor="text-purple-600"
        />
        <StatCard
          icon={dailyPnl >= 0 ? TrendingUp : TrendingDown}
          label="Unrealized P&L"
          value={`${dailyPnl >= 0 ? '+' : ''}${formatCurrency(dailyPnl)}`}
          change={`${dailyPnl >= 0 ? '+' : ''}${dailyPnlPct.toFixed(2)}%`}
          changeType={dailyPnlType}
          iconColor={dailyPnl >= 0 ? "bg-lime-100" : "bg-red-100"}
          iconTextColor={dailyPnl >= 0 ? "text-lime-600" : "text-red-500"}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio Chart */}
        <div className="lg:col-span-2 card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="card-header mb-0">
              <Activity className="w-5 h-5 text-lime-600" />
              Portfolio Performance
            </h3>
            <TimePeriodSelector selected={timePeriod} onSelect={setTimePeriod} />
          </div>
          {/* Legend */}
          <div className="flex items-center justify-end gap-6 mb-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-lime-500 rounded-full"></span>
              <span className="text-gray-600 font-medium">Portfolio</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-schwab-500 rounded-full opacity-70"></span>
              <span className="text-gray-600 font-medium">SPY</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-amber-500 rounded-full opacity-70"></span>
              <span className="text-gray-600 font-medium">QQQ</span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey={isLiveMode ? "time" : "date"}
                  stroke="#94a3b8"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  tickFormatter={(val) => {
                    if (isLiveMode) return val
                    return new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                  }}
                  interval={isLiveMode ? Math.floor(chartData.length / 6) : 'preserveEnd'}
                  minTickGap={isLiveMode ? 30 : 20}
                  tickMargin={6}
                />
                <YAxis 
                  stroke="#94a3b8"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`}
                  domain={['auto', 'auto']}
                  tickMargin={6}
                />
                <Tooltip content={<ChartTooltip />} />
                {/* Zero line */}
                <Line 
                  type="monotone" 
                  dataKey={() => 0} 
                  stroke="#cbd5e1" 
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  isAnimationActive={false}
                />
                {/* Portfolio - lime green brand color */}
                <Line 
                  type="monotone" 
                  dataKey="portfolio" 
                  stroke="#00c853" 
                  strokeWidth={2.5}
                  dot={false}
                  name="Portfolio"
                />
                {/* SPY - Schwab blue */}
                <Line 
                  type="monotone" 
                  dataKey="spy" 
                  stroke="#00a3e0" 
                  strokeWidth={1.5}
                  dot={false}
                  name="SPY"
                  strokeDasharray="5 2"
                  opacity={0.7}
                />
                {/* QQQ - amber orange */}
                <Line 
                  type="monotone" 
                  dataKey="qqq" 
                  stroke="#f59e0b" 
                  strokeWidth={1.5}
                  dot={false}
                  name="QQQ"
                  strokeDasharray="5 2"
                  opacity={0.7}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Trades */}
        <div className="card">
          <h3 className="card-header">
            <TrendingUp className="w-5 h-5 text-lime-600" />
            Recent Trades
          </h3>
          <div className="space-y-0 -mx-6 -mb-6 px-6 pb-4 max-h-64 overflow-y-auto">
            {recentTrades.length > 0 ? (
              recentTrades.slice(0, 5).map((trade) => (
                <TradeRow key={trade.id} trade={trade} />
              ))
            ) : (
              <p className="text-gray-400 text-center py-8">No recent trades</p>
            )}
          </div>
        </div>
      </div>

      {/* Positions Table */}
      <div className="card">
        <h3 className="card-header">
          <Briefcase className="w-5 h-5 text-purple-600" />
          Current Positions
        </h3>
        {positions.length > 0 ? (
          <div className="overflow-x-auto -mx-6 -mb-6">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-500 text-sm border-b border-gray-200">
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
            <Briefcase className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No open positions</p>
            <p className="text-gray-400 text-sm mt-1">Positions will appear here when trades are executed</p>
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="card bg-gradient-to-r from-lime-50 to-schwab-50 border-lime-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-lime-100 rounded-xl">
              <CheckCircle className="w-6 h-6 text-lime-600" />
            </div>
            <div>
              <h3 className="text-gray-900 font-semibold">System Status: <span className="text-lime-600">Operational</span></h3>
              <p className="text-gray-500 text-sm">Multi-Agent Consensus Architecture (MACA) v3.0</p>
            </div>
          </div>
          <div className="text-right text-sm">
            <p className="text-gray-500">Next scheduled scan</p>
            <p className="text-schwab-600 font-mono font-medium">9:35 AM ET / 12:30 PM ET</p>
          </div>
        </div>
      </div>
    </div>
  )
}
