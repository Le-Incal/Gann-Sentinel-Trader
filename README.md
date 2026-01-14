# Gann Sentinel Trader (GST)

**AI-Powered Autonomous Trading System**

An experimental trading system that combines multiple AI agents for market analysis and decision-making. Built on the philosophy of "ANCHOR in history, ORIENT toward future."

[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-blueviolet)](https://railway.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Private-red.svg)]()

---

## Overview

GST is a sophisticated trading system that:

- **Scans multiple data sources** for market signals (sentiment, macro, predictions, events)
- **Leverages 4 AI systems** to generate and validate investment theses
- **Applies strict risk management** before any trade recommendation
- **Requires human approval** via Telegram for all trades
- **Learns from outcomes** to improve future decisions

> ⚠️ **Disclaimer:** Trading involves substantial risk of loss. This is an experimental system and nothing here constitutes financial advice. Only trade what you can afford to lose.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GANN SENTINEL TRADER v2.4.2                         │
│                    "ANCHOR in history, ORIENT toward future"                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│    SCANNERS     │           │   AI COUNCIL    │           │    EXECUTORS    │
│  (Data Input)   │           │     (MACA)      │           │    (Output)     │
└─────────────────┘           └─────────────────┘           └─────────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│ • Grok          │           │ • Grok Thesis   │           │ • Risk Engine   │
│ • FRED          │    ───►   │ • Perplexity    │    ───►   │ • Alpaca        │
│ • Polymarket    │           │ • ChatGPT       │           │ • Telegram      │
│ • Technical     │           │ • Claude        │           │                 │
│ • Event (27)    │           │   (Synthesis)   │           │                 │
└─────────────────┘           └─────────────────┘           └─────────────────┘
```

---

## Key Features

### Multi-Agent Consensus Architecture (MACA)

Four AI systems work together to reduce bias and improve decision quality:

| AI | Role | Specialty |
|----|------|-----------|
| **Grok** | Signal Generator | Social sentiment, X/Twitter trends |
| **Perplexity** | Researcher | Fundamental analysis, citations |
| **ChatGPT** | Pattern Finder | Technical patterns, risk scenarios |
| **Claude** | Senior Trader | Synthesis, final decision |

### Signal Scanners

| Scanner | Source | Signals |
|---------|--------|---------|
| **Grok** | xAI API | Sentiment, catalysts, trending narratives |
| **FRED** | Federal Reserve | Macro indicators (yields, CPI, GDP, unemployment) |
| **Polymarket** | Prediction Markets | 17 investment categories with probability tracking |
| **Technical** | Alpaca | 5-year price history, support/resistance, trends |
| **Event** | Grok (parsed) | 27 corporate event types (FDA, M&A, insider buying) |

### Risk Management

- Position size limits (max 20% per position)
- Daily loss limits (max 3% drawdown)
- Sector concentration limits (max 40%)
- Liquidity requirements (min $1M daily volume)
- Human approval required for all trades

### Smart Scheduling

- 2 scans per day (9:35 AM, 12:30 PM ET)
- No weekend scans
- Manual `/scan` and `/check` always available
- 75% cost reduction vs hourly scanning

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/scan` | Run full MACA scan cycle |
| `/check [TICKER]` | Analyze specific stock |
| `/status` | Portfolio and system health |
| `/positions` | Current open positions |
| `/pending` | Trades awaiting approval |
| `/approve [ID]` | Approve pending trade |
| `/reject [ID]` | Reject pending trade |
| `/history [N]` | Last N trades |
| `/export [csv/parquet]` | Export data |
| `/cost [days]` | API cost summary |
| `/logs` | View recent activity |
| `/stop` | Emergency halt |
| `/resume` | Resume trading |
| `/help` | Show all commands |

---

## Project Structure

```
gann-sentinel-trader/
├── agent.py                    # Main orchestrator
├── config.py                   # Configuration
├── learning_engine.py          # Performance tracking
│
├── scanners/
│   ├── temporal.py             # Shared time framework
│   ├── grok_scanner.py         # Sentiment/catalysts
│   ├── fred_scanner.py         # Macro data
│   ├── polymarket_scanner.py   # Prediction markets
│   ├── technical_scanner.py    # Chart analysis
│   └── event_scanner.py        # Corporate events
│
├── analyzers/
│   ├── claude_analyst.py       # Claude analysis
│   ├── claude_maca_extension.py
│   ├── perplexity_analyst.py   
│   └── chatgpt_analyst.py      
│
├── core/
│   └── maca_orchestrator.py    # 4-phase MACA cycle
│
├── executors/
│   ├── risk_engine.py          # Risk validation
│   └── alpaca_executor.py      # Trade execution
│
├── notifications/
│   └── telegram_bot.py         # Bot interface
│
├── storage/
│   └── database.py             # SQLite database
│
├── api/
│   └── logs_api.py             # HTTP API
│
└── docs/
    ├── GST_MASTER_FRAMEWORK.md # Full documentation
    ├── MACA_SPEC_v1.md         # MACA architecture
    └── PHASE2_DEPLOYMENT_GUIDE.md
```

---

## Environment Variables

### Required

```bash
# AI APIs
XAI_API_KEY=           # Grok
ANTHROPIC_API_KEY=     # Claude

# Trading
ALPACA_API_KEY=        
ALPACA_SECRET_KEY=     
ALPACA_PAPER=true      # Use paper trading

# Notifications
TELEGRAM_BOT_TOKEN=    
TELEGRAM_CHAT_ID=      
```

### Optional (MACA)

```bash
MACA_ENABLED=true
PERPLEXITY_API_KEY=    # Perplexity Sonar Pro
OPENAI_API_KEY=        # GPT-4o
```

### Optional (Logs API)

```bash
LOGS_API_TOKEN=        # For remote monitoring
```

---

## Installation

### Prerequisites

- Python 3.11+
- Railway account (or other hosting)
- API keys for all services

### Local Development

```bash
# Clone repository
git clone https://github.com/Le-Incal/Gann-Sentinel-Trader.git
cd Gann-Sentinel-Trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run
python agent.py
```

### Railway Deployment

1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main

---

## API Endpoints

The Logs API provides remote monitoring:

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Service health |
| `GET /api/status` | Token | Full system status |
| `GET /api/logs` | Token | Telegram history |
| `GET /api/errors` | Token | System errors |
| `GET /api/signals` | Token | Recent signals |

---

## The 27 Event Types

GST monitors corporate events that historically move stock prices:

**Leadership:** CEO exits, appointments, insider buying/selling

**Capital Allocation:** Buybacks, dividend changes

**Regulatory:** FDA approvals/rejections, DOJ investigations

**Index Changes:** S&P 500 additions/removals

**External Pressure:** Activist investors, short seller reports

**Contracts:** Government contracts, major partnerships

**Corporate Actions:** M&A, spinoffs, bankruptcies

---

## Philosophy

### ANCHOR in History, ORIENT Toward Future

GST combines historical pattern recognition with forward-looking catalyst analysis:

1. **Historical Context** - "When has this happened before?"
2. **Forward Catalysts** - "What events are coming?"
3. **Second-Order Thinking** - "Who benefits that isn't obvious?"

### Example: SpaceX IPO

```
Signal: "SpaceX IPO expected H2 2026"

First-Order: "Buy SpaceX" → Can't, it's private

Second-Order: 
→ SpaceX IPO brings attention to space sector
→ Investors comparison shop for public alternatives
→ Rocket Lab (RKLB) is most comparable public company
→ Trade: BUY RKLB ahead of SpaceX IPO
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.4.2 | Jan 2026 | Full MACA for scheduled scans, sports filter fix |
| 2.4.1 | Jan 2026 | Trade blocker visibility |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.2.0 | Jan 2026 | MACA for /check command |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 1.0.0 | Dec 2025 | Initial release |

---

## Documentation

- [Master Framework](docs/GST_MASTER_FRAMEWORK.md) - Complete system documentation
- [MACA Specification](docs/MACA_SPEC_v1.md) - Multi-Agent architecture details
- [Deployment Guide](docs/PHASE2_DEPLOYMENT_GUIDE.md) - Setup instructions

---

## Contributing

This is a private project. Contact the maintainer for access.

---

## License

Private - All rights reserved.

---

## Acknowledgments

Built with:
- [Anthropic Claude](https://anthropic.com) - AI synthesis and decision-making
- [xAI Grok](https://x.ai) - Real-time sentiment analysis
- [Perplexity](https://perplexity.ai) - Research and citations
- [OpenAI GPT-4](https://openai.com) - Pattern recognition
- [Alpaca](https://alpaca.markets) - Trading execution
- [Polymarket](https://polymarket.com) - Prediction market data
- [FRED](https://fred.stlouisfed.org) - Economic data

---

*"The market is a device for transferring money from the impatient to the patient." - Warren Buffett*
