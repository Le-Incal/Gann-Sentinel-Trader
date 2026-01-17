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
    <div className="min-h-screen bg-gst-dark">
      {/* Header */}
      <header className="bg-gst-darker border-b border-gst-border sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-gst-accent to-gst-purple flex items-center justify-center">
                <span className="text-white font-bold text-lg">G</span>
              </div>
              <div>
                <h1 className="text-white font-semibold text-lg">Gann Sentinel Trader</h1>
                <p className="text-gray-500 text-xs">AI-Powered Multi-Agent Trading System</p>
              </div>
            </div>

            {/* Live indicator */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-green-900/30 border border-green-800/50 rounded-full">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-green-400 text-sm font-medium">Live</span>
              </div>
              <a 
                href="https://github.com/yourusername/gann-sentinel-trader" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
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
                  `flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? 'text-gst-accent border-gst-accent'
                      : 'text-gray-400 border-transparent hover:text-gray-200 hover:border-gray-600'
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
      <footer className="border-t border-gst-border mt-auto py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
            <p>
              Gann Sentinel Trader v3.0.0 — Committee-Based Multi-Agent Architecture
            </p>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <span className="text-gray-600">Trade approvals via</span>
                <span className="text-gst-accent">Telegram</span>
              </span>
              <a 
                href="https://gann-sentinel-trader-production.up.railway.app/health"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-gray-300 transition-colors"
              >
                API Status
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
