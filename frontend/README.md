# GST Dashboard - Frontend

A React-based public dashboard for the Gann Sentinel Trader system.

## Features

- **Portfolio Dashboard** - Real-time portfolio value, positions, and P&L charts
- **Trade History** - View all executed and pending trades
- **AI Debates** - Watch the multi-agent investment committee deliberate
- **Signals Browser** - Explore signals from FRED, Polymarket, Grok, and Technical scanners
- **Scan Cycles** - Monitor scheduled and manual analysis cycles with cost tracking

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts
- **TanStack Query** - Data fetching
- **React Router** - Navigation

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Setup

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

The dev server runs on `http://localhost:3000` and proxies API requests to `http://localhost:8080`.

### Environment Variables

Create a `.env` file for production:

```env
# Backend API URL (your Railway deployment)
VITE_API_URL=https://gann-sentinel-trader-production.up.railway.app
```

For local development, leave `VITE_API_URL` empty - the Vite dev server proxies `/api` requests automatically.

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repo to Vercel
2. Set the root directory to `frontend`
3. Add environment variable:
   - `VITE_API_URL` = `https://gann-sentinel-trader-production.up.railway.app`
4. Deploy!

Vercel will automatically:
- Detect the Vite framework
- Run `npm run build`
- Serve from `dist/`

### Manual Build

```bash
npm run build
```

Output is in `dist/` - serve with any static file server.

## API Integration

The dashboard uses public API endpoints that don't require authentication:

| Endpoint | Description |
|----------|-------------|
| `/api/public/dashboard` | Portfolio overview |
| `/api/public/portfolio` | Current portfolio state |
| `/api/public/positions` | Open positions |
| `/api/public/trades` | Trade history |
| `/api/public/signals` | Scanner signals |
| `/api/public/scan_cycles` | MACA scan cycles |
| `/api/public/debates` | AI committee debates |
| `/api/public/costs` | API cost tracking |

## Architecture

```
src/
├── main.jsx          # Entry point
├── App.jsx           # Router setup
├── api.js            # API client
├── index.css         # Tailwind + custom styles
├── components/
│   └── Layout.jsx    # App shell with navigation
├── pages/
│   ├── Dashboard.jsx # Portfolio overview
│   ├── Trades.jsx    # Trade history
│   ├── Signals.jsx   # Signal browser
│   ├── Debates.jsx   # AI debates viewer
│   └── ScanCycles.jsx # Scan monitoring
└── utils/
    └── format.js     # Formatting utilities
```

## Notes

- This is a **view-only** dashboard - trade approvals are handled via Telegram
- Data refreshes every 30 seconds automatically
- The dashboard is designed for public viewing (no sensitive data exposed)
