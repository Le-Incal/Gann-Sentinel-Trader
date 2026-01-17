import { Outlet, NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, 
  TrendingUp, 
  Radio, 
  MessageSquare, 
  Activity,
  ExternalLink,
  Github
} from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/trades', icon: TrendingUp, label: 'Trades' },
  { to: '/signals', icon: Radio, label: 'Signals' },
  { to: '/debates', icon: MessageSquare, label: 'AI Debates' },
  { to: '/scans', icon: Activity, label: 'Scan Cycles' },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-gst-bg">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-gst-border sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-white border border-gray-200 flex items-center justify-center shadow-sm overflow-hidden p-1">
                <img src="/logo.svg" alt="Gann Sentinel" className="w-full h-full object-contain" />
              </div>
              <div>
                <h1 className="text-gray-900 font-semibold text-lg tracking-tight">
                  Gann Sentinel
                </h1>
                <p className="text-gray-500 text-xs font-medium">AI Trading System</p>
              </div>
            </div>

            {/* Live indicator */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 bg-lime-50 border border-lime-200 rounded-full">
                <span className="w-2 h-2 bg-lime-500 rounded-full animate-pulse"></span>
                <span className="text-lime-700 text-sm font-semibold">Live</span>
              </div>
              <a 
                href="https://github.com/Le-Incal/Gann-Sentinel-Trader" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-gray-700 transition-colors p-2 hover:bg-gray-100 rounded-lg"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-1 -mb-px">
            {navItems.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-5 py-3 text-sm font-medium rounded-t-xl transition-all duration-200 ${
                    isActive
                      ? 'text-lime-600 bg-lime-50 border-b-2 border-lime-500'
                      : 'text-gray-500 hover:text-gray-900 hover:bg-gray-50'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-gst-border mt-auto py-8 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white border border-gray-200 flex items-center justify-center overflow-hidden p-0.5">
                <img src="/logo.svg" alt="Gann Sentinel" className="w-full h-full object-contain" />
              </div>
              <div>
                <p className="text-gray-700 text-sm font-medium">
                  Gann Sentinel Trader v3.0
                </p>
                <p className="text-gray-400 text-xs">
                  Multi-Agent Consensus Architecture
                </p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm">
              <span className="flex items-center gap-2 text-gray-500">
                <span>Trade approvals via</span>
                <span className="text-schwab-600 font-medium">Telegram</span>
              </span>
              <a 
                href="https://gann-sentinel-trader-production.up.railway.app/health"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-gray-500 hover:text-lime-600 transition-colors"
              >
                API Status
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
