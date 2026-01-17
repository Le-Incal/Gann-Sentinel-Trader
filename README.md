# Gann Sentinel Trader (GST)

**AI-Powered Multi-Agent Trading Decision System**

A sophisticated trading system that combines multiple AI agents in a committee-based architecture for market analysis and decision-making. Built on the philosophy of "ANCHOR in history, ORIENT toward future."

[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-blueviolet)](https://railway.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-3.1.1-green.svg)]()

---

## Overview

GST is a committee-based trading decision system that:

- **Collects signals** from 5 data sources (FRED, Polymarket, Grok, Technical, Events)
- **Leverages 3 AI analysts** who independently generate investment theses
- **Enables visible debate** where analysts review each other's proposals and vote
- **Uses Claude as Senior Trader** to synthesize and make final decisions
- **Applies strict risk management** before any trade recommendation
- **Requires human approval** via Telegram for all trades
- **Learns from outcomes** to improve future decisions

> ⚠️ **Disclaimer:** Trading involves substantial risk of loss. This is an experimental system and nothing here constitutes financial advice. Only trade what you can afford to lose.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GANN SENTINEL TRADER v3.1.1                         │
│                    "ANCHOR in history, ORIENT toward future"                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────┼─────────────────────────────────────┐
│                            SIGNAL COLLECTION                               │
├─────────────────────────────────────┼─────────────────────────────────────┤
│  FRED (7 macro)  │  Polymarket (investment categories)  │  Events (27)   │
│  Technical (IEX) │  Grok (sentiment)                    │                │
└─────────────────────────────────────┼─────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI COUNCIL (3 Analysts)                           │
├──────────────────────┬──────────────────────┬──────────────────────────────┤
│   🐦 GROK            │   🎯 PERPLEXITY      │   🧠 CHATGPT                 │
│   Narrative Momentum │   External Reality   │   Sentiment & Bias          │
│   X/Twitter trends   │   Web facts, filings │   Market psychology         │
└──────────────────────┴──────────────────────┴──────────────────────────────┘
                                      │
                              Each produces thesis
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEBATE LAYER                                   │
│            Analysts review proposals, vote, may change position             │
│                    (Skipped if unanimous HOLD)                              │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     🧠 CLAUDE (Senior Trader / Chair)                       │
│         Synthesizes theses → Checks technicals → Makes final decision       │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTION                                      │
├──────────────────────┬──────────────────────┬──────────────────────────────┤
│   Risk Engine        │   Human Approval     │   Alpaca (Paper Trading)     │
│   Position limits    │   via Telegram       │   Order execution            │
└──────────────────────┴──────────────────────┴──────────────────────────────┘
```

---

## Key Features

### Multi-Agent Consensus Architecture (MACA)

Three AI analysts work independently, then debate and vote:

| AI | Role | Specialty |
|----|------|-----------|
| **Grok** | Narrative Momentum Analyst | Social sentiment, X/Twitter trends, emerging narratives |
| **Perplexity** | External Reality Analyst | Web facts, SEC filings, earnings (6-hour recency) |
| **ChatGPT** | Sentiment & Bias Analyst | Market psychology, cognitive bias detection |
| **Claude** | Senior Trader (Chair) | Synthesis, technical validation, final decision |

### Signal Scanners

| Scanner | Source | Signals |
|---------|--------|---------|
| **FRED** | Federal Reserve | 7 macro indicators (yields, CPI, GDP, unemployment) |
| **Polymarket** | Prediction Markets | 12 investment categories (Fed, AI, Crypto, Defense, etc.) |
| **Technical** | Alpaca (IEX) | Chart analysis, support/resistance, market state |
| **Event** | Grok API | 27 corporate event types (FDA, M&A, insider activity) |
| **Grok** | xAI API | Sentiment, catalysts, trending narratives |

### Polymarket Investment Categories

Only investment-relevant prediction markets are scanned:

| Category | Keywords | Trading Relevance |
|----------|----------|-------------------|
| Federal Reserve | fed, fomc, rate cut, powell | Interest rate sensitive stocks |
| Inflation | cpi, inflation, prices | TIPS, commodities |
| AI Sector | ai, artificial intelligence | NVDA, MSFT, GOOGL |
| Crypto Policy | bitcoin, crypto, sec | COIN, MSTR, miners |
| Defense | defense spending, pentagon | LMT, RTX, NOC |
| Energy | oil, gas, renewable, opec | XLE, XOP, clean energy |
| Elections | us election, congress, senate | Policy uncertainty plays |

*Sports betting, foreign elections, and non-investment markets are automatically filtered out.*

### Risk Management

- Position size limits (max 20% per position)
- Daily loss limits (max 3% drawdown)
- Sector concentration limits (max 40%)
- Liquidity requirements (min $1M daily volume)
- Human approval required for all trades

### Smart Scheduling

- 2 scans per day (9:35 AM, 12:30 PM ET)
- Weekend-aware (Event Scanner looks back 72-96 hours)
- Manual `/scan` and `/check` always available
- 75% cost reduction vs hourly scanning

---

## Telegram Output

GST sends a **3-part message** for each scan:

### Message 1: Signal Inventory + AI Council
```
📊 Signals Collected: 53

📈 FRED (7): Treasury yields, CPI, unemployment
🎲 Polymarket (45): AI models, crypto, Fed policy
📅 Events (0): No events (weekend)
📊 Technical (1): TSLA chart analyzed

🐦 GROK: HOLD (80/100) - Bullish outlook, no specific catalyst
🎯 PERPLEXITY: HOLD (0/100) - No recent catalysts found
🧠 CHATGPT: HOLD (0/100) - Mixed macro signals
```

### Message 2: Debate Summary
```
🗣️ MACA Debate (IC Minutes)
✅ Unanimous Agreement: HOLD
Reason: Unanimous HOLD at proposal stage — debate skipped
```

### Message 3: Claude's Decision + Buttons
```
🧠 CLAUDE'S SYNTHESIS (Senior Trader)
Decision: NO_TRADE
Conviction: 20/100

💤 NO TRADE
Reason: All analysts recommend HOLD...

💰 PORTFOLIO
  Equity: $0.00
  Cash: $88,867.16
  Positions: 1

[Status] [Pending] [Scan] [Logs] [Help]  ← Interactive buttons
```

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
├── agent.py                    # Main orchestrator - START HERE
├── config.py                   # Configuration
├── learning_engine.py          # Performance tracking
│
├── scanners/
│   ├── temporal.py             # Shared time framework
│   ├── grok_scanner.py         # Sentiment/catalysts
│   ├── fred_scanner.py         # Macro data (7 series)
│   ├── polymarket_scanner.py   # Prediction markets (whitelist filter)
│   ├── technical_scanner.py    # Chart analysis (IEX data)
│   └── event_scanner.py        # Corporate events (weekend-aware)
│
├── analyzers/
│   ├── perplexity_analyst.py   # 6-hour recency requirement
│   ├── chatgpt_analyst.py      # Sentiment analysis
│   ├── claude_analyst.py       # Legacy analyst
│   └── claude_chair.py         # Senior Trader / Synthesizer
│
├── core/
│   └── maca_orchestrator.py    # Committee debate + synthesis
│
├── executors/
│   ├── risk_engine.py          # Risk validation
│   └── alpaca_executor.py      # Trade execution
│
├── notifications/
│   └── telegram_bot.py         # 3-part messages + buttons
│
├── storage/
│   └── database.py             # SQLite database
│
├── api/
│   └── logs_api.py             # HTTP API for monitoring
│
└── docs/
    ├── GST_MASTER_FRAMEWORK.md
    ├── MACA_SPEC_v1.md
    └── DEPLOYMENT_GUIDE_PHASE2.md
```

---

## Environment Variables

### Required

```bash
# AI APIs
XAI_API_KEY=           # Grok (sentiment + events)
ANTHROPIC_API_KEY=     # Claude (synthesis)
PERPLEXITY_API_KEY=    # Perplexity (research)
OPENAI_API_KEY=        # ChatGPT (sentiment)

# Trading
ALPACA_API_KEY=        
ALPACA_SECRET_KEY=     
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Notifications
TELEGRAM_BOT_TOKEN=    
TELEGRAM_CHAT_ID=      

# Data
FRED_API_KEY=          # Federal Reserve data
```

### Optional

```bash
LOGS_API_TOKEN=        # For remote monitoring
LOG_LEVEL=INFO
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

## The 27 Event Types

GST monitors corporate events that historically move stock prices:

**Leadership:** CEO exits, appointments, insider buying/selling

**Capital Allocation:** Buybacks, dividend changes, special dividends

**Regulatory:** FDA approvals/rejections, DOJ investigations, lawsuits

**Index Changes:** S&P 500 additions/removals, rebalancing

**External Pressure:** Activist investors, short seller reports, proxy fights

**Contracts:** Government contracts, major partnerships, contract losses

**Corporate Actions:** M&A, spinoffs, bankruptcies, debt restructuring

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
| 3.1.1 | Jan 17, 2026 | Polymarket whitelist, Technical IEX fix, weekend events, Telegram buttons |
| 3.1.0 | Jan 17, 2026 | Signal inventory display, analyst formatting fixes |
| 3.0.0 | Jan 2026 | Committee-based debate architecture, Claude as Senior Trader |
| 2.4.3 | Jan 2026 | Trade execution pipeline fixes |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 1.0.0 | Dec 2025 | Initial release |

---

## Documentation

- [Master Framework](docs/GST_MASTER_FRAMEWORK.md) - Complete system documentation
- [MACA Specification](docs/MACA_SPEC_v1.md) - Multi-Agent architecture details
- [Deployment Guide](docs/DEPLOYMENT_GUIDE_PHASE2.md) - Setup instructions
- [CLAUDE.md](CLAUDE.md) - Context for Claude Code

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
- [Alpaca](https://alpaca.markets) - Trading execution (IEX data)
- [Polymarket](https://polymarket.com) - Prediction market data
- [FRED](https://fred.stlouisfed.org) - Economic data

---

*"The market is a device for transferring money from the impatient to the patient." - Warren Buffett*
