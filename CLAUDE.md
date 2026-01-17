# CLAUDE.md - Gann Sentinel Trader (GST)

> This file provides context for Claude Code. Read this first when working on GST.

## Quick Reference

| Item | Value |
|------|-------|
| Version | 3.1.1 |
| Status | Production (Paper Trading) |
| Deployment | Railway (auto-deploy from GitHub main) |
| URL | https://gann-sentinel-trader-production.up.railway.app |
| Logs API Token | `QzHBtENzt-sYeLXKSUzEN_v6VREwfEnGaqpoQVmOBWE` |

---

## 1. What Is GST?

A **committee-based, multi-agent trading decision system** with:
- 3 AI analysts generating independent theses (Grok, Perplexity, ChatGPT)
- Claude as Senior Trader / Synthesizer (makes final decisions)
- Visible debate and voting
- Whitelist-filtered market signals
- Explainable Telegram output with interactive buttons
- HOLD as a first-class outcome

**Core Philosophy: "ANCHOR in history, ORIENT toward future"**

This system is **not** a black-box trading bot. It prioritizes clarity, discipline, auditability, and trust over speed or trade frequency.

**Example:**
```
Signal: "SpaceX IPO expected H2 2026"
First-Order: "Buy SpaceX" → Can't, it's private
Second-Order: SpaceX IPO → attention to space sector → investors comparison shop
             → Rocket Lab (RKLB) is most comparable public company
             → Trade: BUY RKLB ahead of SpaceX IPO
```

---

## 2. Architecture Overview (v3.1.1)

```
Signal Collection (5 sources)
       ↓
┌──────────────────────────────────────────────────┐
│  FRED (7) │ Polymarket (whitelist) │ Events (27) │
│  Technical (IEX) │ Grok (sentiment)              │
└──────────────────────────────────────────────────┘
       ↓
AI Analysts (3 independent theses)
  • Grok: Narrative Momentum
  • Perplexity: External Reality (6hr recency)
  • ChatGPT: Sentiment & Bias
       ↓
Debate Layer (if disagreement)
       ↓
Claude (Senior Trader)
  • Synthesizes theses
  • Validates against technicals
  • Makes final decision
       ↓
Risk Engine (hard constraints)
       ↓
Telegram (3-part message + buttons)
       ↓
Human Approval Required
       ↓
Alpaca Execution (paper trading)
```

---

## 3. Model Roles (STRICT)

Each AI model has a **single epistemic role**. No model may operate outside its role.

### Scanners (No Opinions)
| Scanner | Source | Purpose |
|---------|--------|---------|
| FRED Scanner | Federal Reserve | 7 macroeconomic indicators |
| Polymarket Scanner | Prediction Markets | 12 investment categories (WHITELIST - no sports) |
| Event Scanner | Grok API | 27 corporate event types (weekend-aware) |
| Technical Scanner | Alpaca (IEX) | Chart analysis, market state, support/resistance |

Scanners **must not** propose trades or opinions.

### Analysts (Each produces a thesis)

| Model | Role | Purpose |
|-------|------|---------|
| **Grok** | Narrative Momentum Analyst | Detect emerging/accelerating narratives from X/Twitter |
| **Perplexity** | External Reality Analyst | Web facts, filings, earnings (6-HOUR RECENCY REQUIRED) |
| **ChatGPT** | Sentiment & Cognitive Bias Analyst | Market psychology & bias detection |

### Synthesizer / Senior Trader

- Implemented as **Claude Chair** (analyzers/claude_chair.py)
- Does NOT generate new signals or browse
- Weighs analyst theses + debate outcomes
- Validates against technical analysis
- Breaks ties when analysts disagree
- Produces the **Final Investment Thesis**
- Defaults to HOLD when ambiguity remains

---

## 4. Signal Sources

### FRED (7 Macro Indicators)
- 10-Year Treasury Yield (DGS10)
- 2-Year Treasury Yield (DGS2)
- 10Y-2Y Spread (T10Y2Y)
- Unemployment Rate (UNRATE)
- CPI (CPIAUCSL)
- Fed Funds Rate (FEDFUNDS)
- GDP (GDP)

### Polymarket (12 Investment Categories - WHITELIST)
| Category | Keywords | Tickers |
|----------|----------|---------|
| FEDERAL_RESERVE | fed, fomc, rate cut, powell | Rate-sensitive |
| INFLATION | cpi, inflation, prices | TIPS, commodities |
| RECESSION | recession, gdp, contraction | Defensive |
| TRADE_POLICY | tariff, trade war, sanctions | Import/export |
| CHINA_RISK | china, taiwan, xi | Supply chain, tech |
| AI_SECTOR | ai, artificial intelligence | NVDA, MSFT, GOOGL |
| SEMICONDUCTOR | chip, semiconductor, nvidia | SMH, SOXX |
| CRYPTO_POLICY | bitcoin, crypto, sec | COIN, MSTR |
| ENERGY_POLICY | oil, gas, renewable, opec | XLE, XOP |
| DEBT_CEILING | debt ceiling, default | TLT, financials |
| ELECTION | us election, congress, senate | Policy plays |
| DEFENSE | defense spending, pentagon | LMT, RTX, NOC |

**EXCLUDED (NOT_RELEVANT):** Sports betting, foreign elections, entertainment, price guessing

### Event Scanner (27 Corporate Events)
- Weekend-aware: Saturday=72hr, Sunday=96hr, Monday=72hr lookback
- Leadership: CEO exits/appointments, insider buying/selling
- Capital: Buybacks, dividends
- Regulatory: FDA, DOJ
- Index: S&P 500 changes
- External: Activists, short sellers
- Corporate: M&A, spinoffs, bankruptcy

### Technical Scanner (Alpaca IEX)
- Uses FREE IEX data feed (not paid SIP)
- Market state classification (Trending/Range/Transitional)
- Support/resistance levels
- Liquidity sweep detection
- Trade hypothesis generation with R-multiple

---

## 5. Telegram Output Format

### 3-Part Message Sequence

**Message 1: Signal Inventory + AI Council**
```
📊 Signals Collected: 53
📈 FRED (7): [summaries]
🎲 Polymarket (45): [summaries]
📅 Events (0): No events (weekend)
📊 Technical (1): 1 chart analyzed

🐦 GROK: Recommendation, Signals, Conviction, Thesis
🎯 PERPLEXITY: Recommendation, Signals, Conviction, Thesis
🧠 CHATGPT: Recommendation, Signals, Conviction, Thesis
```

**Message 2: Debate Summary**
```
🗣️ MACA Debate (IC Minutes)
Cycle: [UUID]
✅ Unanimous Agreement: HOLD
Reason: [why debate was skipped or outcome]
```

**Message 3: Claude's Decision + Buttons**
```
🕯 CHART ANALYSIS (if technical signals)
🧠 CLAUDE'S SYNTHESIS (Senior Trader)
Decision: NO_TRADE | TRADE
Conviction: X/100 [bar]
Final thesis: [summary]
Why now: [detail]
Invalidation: [conditions]

💤 NO TRADE / 🟢 TRADE PENDING
💰 PORTFOLIO snapshot

[Status] [Pending] [Scan] [Logs] [Help] ← Interactive buttons
```

---

## 6. File Structure

```
gann-sentinel-trader/
├── agent.py                 # Main orchestrator - START HERE
├── config.py                # All configuration
├── learning_engine.py       # Performance tracking
│
├── SCANNERS (Data Input)
│   ├── grok_scanner.py      # xAI Grok sentiment
│   ├── fred_scanner.py      # Federal Reserve (7 series)
│   ├── polymarket_scanner.py # Whitelist filtering (12 categories)
│   ├── technical_scanner.py # Alpaca IEX charts
│   └── event_scanner.py     # 27 event types (weekend-aware)
│
├── ANALYZERS (AI Council)
│   ├── perplexity_analyst.py   # 6-hour recency requirement
│   ├── chatgpt_analyst.py      # Sentiment analysis
│   ├── claude_chair.py         # Senior Trader / Synthesizer
│   └── claude_analyst.py       # Legacy (backwards compat)
│
├── CORE
│   └── maca_orchestrator.py # Committee debate + synthesis
│
├── STORAGE
│   └── database.py          # SQLite
│
├── EXECUTORS
│   ├── risk_engine.py       # Position sizing, limits
│   └── alpaca_executor.py   # Alpaca trading API
│
├── NOTIFICATIONS
│   └── telegram_bot.py      # 3-part messages + buttons
│
├── API
│   └── logs_api.py          # HTTP API for monitoring
│
└── DOCS
    ├── CHANGELOG.md
    ├── GST_MASTER_FRAMEWORK.md
    └── MACA_SPEC_v1.md
```

---

## 7. Environment Variables

```bash
# Required
XAI_API_KEY=           # Grok (sentiment + events)
ANTHROPIC_API_KEY=     # Claude (synthesis)
PERPLEXITY_API_KEY=    # Perplexity Sonar Pro
OPENAI_API_KEY=        # GPT-4o (sentiment)
ALPACA_API_KEY=        # Trading + IEX data
ALPACA_SECRET_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
FRED_API_KEY=          # Macro data

# Optional
LOGS_API_TOKEN=        # Remote monitoring
LOG_LEVEL=INFO
```

---

## 8. Telegram Commands

| Command | Description |
|---------|-------------|
| `/scan` | Run full committee cycle |
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

---

## 9. Current State (v3.1.1)

### What's Working ✓
- **3 AI analysts** generating independent theses
- **Claude as Senior Trader** synthesizing and deciding
- **Polymarket whitelist filtering** (12 investment categories, no sports)
- **Technical Scanner** using free IEX data
- **Event Scanner** with weekend-aware lookback
- **Perplexity** with 6-hour recency requirement
- **3-part Telegram messages** with interactive buttons
- **Signal inventory** showing all collected signals
- Smart scheduling (2x daily: 9:35 AM, 12:30 PM ET)
- Learning Engine tracking outcomes

### Smart Schedule
```
Scheduled Scans: 9:35 AM ET, 12:30 PM ET (weekdays only)
Manual Commands: /scan and /check always available
Weekend: Event Scanner extends lookback to 72-96 hours
```

### Known Considerations
- Paper trading only (Alpaca)
- Human approval required for all trades
- SQLite database (Railway persistent volume)

---

## 10. Safety Architecture

```
Signal Sources → FRED/Polymarket/Technical/Event (data only, filtered)
       ↓
AI Analysts → Grok + Perplexity + ChatGPT (independent theses)
       ↓
Debate Layer → Review each other's proposals, vote
       ↓
Claude (Senior Trader) → Synthesize, validate, decide
       ↓
Risk Engine → Position sizing, limits, validation
       ↓
Human Gate → Telegram approval required
       ↓
Execution → Alpaca paper trading
```

**Key Safety Principles:**
1. **Whitelist filtering** - Only investment-relevant signals
2. **Multiple AI sources** - 3 independent theses reduce bias
3. **Visible disagreement** - Debate transcripts logged
4. **Roles are sacred** - No AI operates outside its role
5. **HOLD is success** - Capital preservation > opportunity
6. **Human approval** - Required for ALL trades

---

## 11. Design Principles (DO NOT VIOLATE)

1. HOLD is a success outcome
2. Disagreement is information
3. Roles are sacred
4. Explainability > frequency
5. Capital preservation > opportunity

**Optimize for:** clarity, discipline, auditability, trust

**DO NOT optimize for:** speed, trade frequency, cleverness

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.1.1 | Jan 17, 2026 | Polymarket whitelist, Technical IEX, weekend events, Telegram buttons |
| 3.1.0 | Jan 17, 2026 | Signal inventory display, analyst formatting, Perplexity recency |
| 3.0.0 | Jan 2026 | Committee-based architecture, Claude as Senior Trader |
| 2.4.3 | Jan 2026 | Trade execution pipeline fixes |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 1.0.0 | Dec 2025 | Initial release |

---

## 13. Recent Fixes (v3.1.1)

### Technical Scanner
- **Issue:** Alpaca SIP data requires paid subscription
- **Fix:** Now uses free IEX data feed via `DataFeed.IEX`

### Event Scanner
- **Issue:** 0 events on weekends (corporate events happen on business days)
- **Fix:** Weekend-aware lookback (Sat=72hr, Sun=96hr, Mon=72hr)

### Polymarket Scanner
- **Issue:** Sports betting signals appearing (Navy Midshipmen, etc.)
- **Fix:** Whitelist-based filtering with 12 investment categories + exclusion patterns

### Telegram Messages
- **Issue:** Buttons not appearing, messages truncated
- **Fix:** 4000-char truncation, section-level error handling, safe type conversions

### Perplexity Analyst
- **Issue:** Returning old/stale data
- **Fix:** 6-hour recency requirement in prompt

---

## 14. Working Agreements

1. **Safety First** - Human approval gate is non-negotiable
2. **Observability** - Log everything, make debugging easy
3. **Defensive Coding** - Handle None values, type errors gracefully
4. **Forward-Looking** - ANCHOR in history, ORIENT toward future
5. **Lean Philosophy** - Complete current phase before adding features

---

*Last Updated: January 17, 2026*
*Maintainer: Kyle + Claude*
