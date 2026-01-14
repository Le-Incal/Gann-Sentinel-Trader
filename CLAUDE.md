# CLAUDE.md - Gann Sentinel Trader (GST)

> This file provides context for Claude Code. Read this first when working on GST.

## Quick Reference

| Item | Value |
|------|-------|
| Version | 2.4.2 |
| Status | Production (Paper Trading) |
| Deployment | Railway (auto-deploy from GitHub main) |
| URL | https://gann-sentinel-trader-production.up.railway.app |
| Logs API Token | `QzHBtENzt-sYeLXKSUzEN_v6VREwfEnGaqpoQVmOBWE` |

---

## 1. What Is GST?

An AI-powered autonomous trading system that combines multiple AI agents for market analysis and decision-making. Currently running in Alpaca paper trading mode with Telegram-based human approval for all trades.

**Core Philosophy: "ANCHOR in history, ORIENT toward future"**

The system combines historical pattern recognition with forward-looking catalyst analysis. It practices second-order thinking to find non-obvious opportunities.

**Example:**
```
Signal: "SpaceX IPO expected H2 2026"
First-Order: "Buy SpaceX" → Can't, it's private
Second-Order: SpaceX IPO → attention to space sector → investors comparison shop
             → Rocket Lab (RKLB) is most comparable public company
             → Trade: BUY RKLB ahead of SpaceX IPO
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GANN SENTINEL TRADER v2.4.2                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│    SCANNERS     │           │   AI COUNCIL    │           │    EXECUTORS    │
│  (Data Input)   │           │     (MACA)      │           │    (Output)     │
└─────────────────┘           └─────────────────┘           └─────────────────┘
│ • Grok          │           │ • Grok Thesis   │           │ • Risk Engine   │
│ • FRED          │    ───►   │ • Perplexity    │    ───►   │ • Alpaca        │
│ • Polymarket    │           │ • ChatGPT       │           │ • Telegram      │
│ • Technical     │           │ • Claude        │           │                 │
│ • Event (27)    │           │   (Synthesis)   │           │                 │
└─────────────────┘           └─────────────────┘           └─────────────────┘
```

### MACA (Multi-Agent Consensus Architecture)

Four AI systems work together:

| AI | Role | Specialty |
|----|------|-----------|
| **Grok** | Signal Generator | Social sentiment, X/Twitter trends, live search |
| **Perplexity** | Researcher | Fundamental analysis, citations, financial data |
| **ChatGPT** | Pattern Finder | Technical patterns, risk scenarios, market structure |
| **Claude** | Senior Trader (CIO) | Synthesis, final decision, conviction scoring |

**MACA Flow:**
1. Phase 1: Grok, Perplexity, ChatGPT generate theses (parallel)
2. Phase 2: Claude synthesizes all theses
3. Phase 3: Peer review (if conviction ≥ 80)
4. Phase 4: Final decision → Risk Engine → Human Approval

---

## 3. File Structure

```
gann-sentinel-trader/
├── agent.py                 # Main orchestrator - START HERE
├── temporal.py              # Shared time framework (market hours, holidays)
├── database.py              # SQLite persistence layer
│
├── SCANNERS (Data Input)
│   ├── grok_scanner.py      # xAI Grok with live_search
│   ├── fred_scanner.py      # Federal Reserve macro data
│   ├── polymarket_scanner.py # Prediction markets (17 categories)
│   ├── technical_scanner.py # 5-year chart analysis via Alpaca
│   └── event_scanner.py     # 27 corporate event types
│
├── ANALYZERS (AI Council)
│   ├── claude_analyst.py    # Claude synthesis + decisions
│   ├── claude_maca_extension.py # MACA-specific Claude logic
│   ├── chatgpt_analyst.py   # GPT-4o analysis
│   └── maca_orchestrator.py # 4-phase MACA cycle
│
├── EXECUTORS
│   ├── risk_engine.py       # Position sizing, limits, validation
│   └── telegram_bot.py      # Bot interface + approval workflow
│
├── UTILITIES
│   ├── learning_engine.py   # Performance tracking + adaptation
│   ├── logs_api.py          # HTTP API for remote monitoring
│   └── exporter.py          # CSV/Parquet export
│
└── DOCS
    ├── GST_MASTER_FRAMEWORK.md  # Complete system documentation
    ├── MACA_SPEC_v1.md          # MACA architecture details
    └── PHASE2_DEPLOYMENT_GUIDE.md
```

---

## 4. Key Concepts

### Signal Categories

| Scanner | Source | Signals |
|---------|--------|---------|
| Grok | xAI API | Sentiment, catalysts, trending narratives |
| FRED | Federal Reserve | Yields, CPI, GDP, unemployment |
| Polymarket | Prediction Markets | 17 investment categories (excludes sports/entertainment) |
| Technical | Alpaca | Support/resistance, trends, volume |
| Event | Grok (parsed) | 27 corporate event types |

### The 27 Event Types

Leadership changes, insider buying/selling, buybacks, dividends, FDA approvals, DOJ investigations, S&P 500 changes, activist investors, short seller reports, government contracts, M&A, spinoffs, bankruptcies, and more.

### Risk Parameters

```python
CONVICTION_THRESHOLD = 80      # Min score to trigger trade
MAX_POSITION_SIZE_PCT = 20     # Max 20% per position
DAILY_LOSS_LIMIT_PCT = 3       # Max 3% daily drawdown
SECTOR_CONCENTRATION_PCT = 40  # Max 40% in one sector
MIN_LIQUIDITY = 1_000_000      # $1M daily volume minimum
DEFAULT_STOP_LOSS_PCT = 8      # 8% stop loss
```

---

## 5. Development Workflow

### Philosophy: "Build It Lean"

1. Complete Phase 1 before adding Phase 2 features
2. Observe production behavior before adding complexity
3. Avoid premature optimization

### TDD Approach

```bash
# 1. Write tests first
# 2. Run tests, confirm they fail
# 3. Implement until tests pass
# 4. Commit
```

### Deployment

```bash
# Auto-deploys on push to main
git add .
git commit -m "description"
git push origin main

# Verify
curl https://gann-sentinel-trader-production.up.railway.app/health
```

### Logs API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Service health |
| `GET /api/status` | Token | Full system status |
| `GET /api/logs` | Token | Telegram history |
| `GET /api/errors` | Token | System errors |
| `GET /api/signals` | Token | Recent signals |

---

## 6. Telegram Commands

| Command | Description |
|---------|-------------|
| `/scan` | Run full MACA scan cycle |
| `/check [TICKER]` | Analyze specific stock with MACA |
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

---

## 7. Environment Variables

### Required
```bash
XAI_API_KEY=           # Grok
ANTHROPIC_API_KEY=     # Claude
ALPACA_API_KEY=        
ALPACA_SECRET_KEY=     
ALPACA_PAPER=true      # Paper trading mode
TELEGRAM_BOT_TOKEN=    
TELEGRAM_CHAT_ID=      
```

### Optional (MACA)
```bash
MACA_ENABLED=true
PERPLEXITY_API_KEY=    # Perplexity Sonar Pro
OPENAI_API_KEY=        # GPT-4o
```

### Optional (Monitoring)
```bash
LOGS_API_TOKEN=        # For remote monitoring
LOG_LEVEL=INFO
```

---

## 8. Current State (v2.4.2)

### What's Working
- Full MACA for both scheduled scans and `/check` command
- Smart scheduling (2x daily: 9:35 AM, 12:30 PM ET)
- 27 event type detection
- Learning Engine tracking outcomes
- Sports/entertainment filter for Polymarket
- Trade blocker visibility in Telegram
- Logs API for remote monitoring

### Smart Schedule
```
Scheduled Scans: 9:35 AM ET, 12:30 PM ET (weekdays only)
Manual Commands: /scan and /check always available
Cost Reduction: ~75% vs hourly scanning
```

### Known Considerations
- Paper trading only (Alpaca)
- Human approval required for all trades
- SQLite database (Railway persistent volume)

---

## 9. Safety Architecture

```
Signal Sources → Grok/FRED/Polymarket/Technical/Event
       ↓
AI Council → Grok + Perplexity + ChatGPT generate theses
       ↓
Claude → Synthesizes, assigns conviction score
       ↓
Risk Engine → Position sizing, limits, validation
       ↓
Human Gate → Telegram approval required
       ↓
Execution → Alpaca paper trading
```

**Key Safety Principles:**
1. Multiple AI sources prevent echo chambers
2. Risk Engine has final authority on position sizing
3. Human approval required for ALL trades
4. Paper trading until system proves itself

---

## 10. Common Tasks

### Add a new ticker to watchlist
Edit `config.py` or pass via environment variable.

### Debug a failed scan
```bash
# Check Railway logs
# Or use Logs API:
curl -H "Authorization: Bearer TOKEN" \
  https://gann-sentinel-trader-production.up.railway.app/api/errors
```

### Test locally
```bash
python -m pytest tests/ -v
```

### Check system status
```bash
curl https://gann-sentinel-trader-production.up.railway.app/api/status \
  -H "Authorization: Bearer QzHBtENzt-sYeLXKSUzEN_v6VREwfEnGaqpoQVmOBWE"
```

---

## 11. Documentation Links

- **Master Framework:** `GST_MASTER_FRAMEWORK.md` - Complete system docs
- **MACA Spec:** `MACA_SPEC_v1.md` - Multi-agent architecture
- **Deployment:** `PHASE2_DEPLOYMENT_GUIDE.md` - Setup instructions
- **Signal Spec:** `GROK_SIGNAL_AGENT_SPEC_v1_1.md` - Grok scanner details
- **Forward System:** `FORWARD_PREDICTIVE_SYSTEM_v2_1.md` - Predictive methodology

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.4.2 | Jan 2026 | Full MACA for scheduled scans, analysis.id fix |
| 2.4.1 | Jan 2026 | Trade blocker visibility |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.2.0 | Jan 2026 | MACA for /check command |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 1.0.0 | Dec 2025 | Initial release |

---

## 13. Working Agreements

1. **Test-Driven Development** - Write tests first, implement second
2. **Lean Philosophy** - Complete current phase before adding features
3. **Safety First** - Human approval gate is non-negotiable
4. **Observability** - Log everything, make debugging easy
5. **Forward-Looking** - ANCHOR in history, ORIENT toward future

---

*Last Updated: January 14, 2026*
*Maintainer: Kyle + Claude*
