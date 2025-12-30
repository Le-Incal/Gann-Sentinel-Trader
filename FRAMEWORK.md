# Gann Sentinel Trader Framework

**Version:** 1.0.0  
**Status:** Phase 1 - Build It Lean  
**Last Updated:** December 29, 2025  

---

## Executive Summary

Gann Sentinel Trader is an autonomous trading agent that combines Grok's real-time sentiment analysis with Claude's strategic reasoning to identify high-conviction position trades. The system operates with human approval gates, comprehensive logging, and strict risk controls.

**North Star:** Deliver an end-to-end system that can ingest signals (Grok sentiment, FRED macro, Polymarket probabilities), run strategies, paper trade by default, and (only when armed) execute via Alpaca—while logging everything for audit and iteration.

---

## 1. Core Philosophy

### What We Believe

1. **Sentiment precedes price.** Crowd psychology on X/Twitter often signals moves before they appear in price action.

2. **Second-order effects create opportunity.** The best trades aren't the obvious ones (SpaceX IPO → SpaceX stock) but the adjacent ones (SpaceX IPO → Rocket Lab benefits from sector attention).

3. **High conviction, low frequency.** We'd rather make 10 trades at 80%+ conviction than 100 trades at 50% conviction.

4. **Log everything.** Every decision, signal, and outcome gets recorded. We can't improve what we don't measure.

5. **Start lean, add complexity only when justified.** We don't know what the agent will be good or bad at. Baseline first, then iterate.

### What This Is NOT

- **Not a high-frequency trading system.** We're position trading (days to months), not scalping.
- **Not a black box.** Every trade has a thesis, and Claude explains its reasoning.
- **Not infallible.** We expect losses. Risk controls exist to ensure no single loss is catastrophic.
- **Not financial advice.** This is an experimental system. Trade only what we can afford to lose.

---

## 2. Phased Roadmap

### Phase 1: Build It Lean (Current)

**Timeline:** Now  
**Goal:** Working end-to-end system with paper trading

| Component | Implementation |
|-----------|----------------|
| Sentiment Brain | Grok API (x_search, web_search) |
| Reasoning Brain | Claude API (claude-sonnet-4-20250514) |
| Execution | Alpaca (paper trading first) |
| Macro Data | FRED API |
| Predictions | Polymarket API |
| Notifications | Telegram Bot |
| Hosting | Railway |
| Version Control | GitHub |
| Storage | SQLite |

**Constraints:**
- No paid market intelligence APIs
- No pattern library (just raw logging)
- No economic regime logic
- Approval gate ON (human confirms every trade)

**Success Criteria:**
- System runs daily without crashes
- Generates at least 1 signal per week
- Approval flow works smoothly
- All trades logged with full context

---

### Phase 2: Add Complexity If Needed

**Timeline:** 1-2 months after Phase 1 baseline  
**Trigger:** Observed gaps in signal quality or missed opportunities

**Potential Additions:**
- Targeted paid APIs if specific signals are missing
- Pattern library based on observed successes/failures
- Economic regime detection (expansion, contraction, crisis)
- Sector rotation logic
- Correlation tracking between signals and outcomes

**Only Add If:**
- We can identify a specific gap the addition would fill
- The cost is justified by expected improvement
- We can measure whether it actually helps

---

### Phase 3: Sophistication

**Timeline:** Only if Phase 1-2 are working  
**Trigger:** Consistent profitability and clear understanding of what works

**Potential Additions:**
- Full pattern library with historical backreferences
- LevelFields or similar event data provider
- Multi-strategy framework (momentum, mean reversion, event-driven)
- Automated regime-based position sizing
- Portfolio-level risk optimization

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GANN SENTINEL TRADER                                │
│                         Architecture Overview                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                      DATA LAYER                                  │     │
│    │                                                                  │     │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │     │
│    │  │   GROK   │  │   FRED   │  │POLYMARKET│  │    ALPACA    │    │     │
│    │  │          │  │          │  │          │  │              │    │     │
│    │  │• X Search│  │• Treasury│  │• Fed odds│  │• Price data  │    │     │
│    │  │• Web News│  │• GDP/CPI │  │• Election│  │• Positions   │    │     │
│    │  │• Trends  │  │• Jobs    │  │• Events  │  │• Account     │    │     │
│    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │     │
│    │       │             │             │               │             │     │
│    └───────┼─────────────┼─────────────┼───────────────┼─────────────┘     │
│            │             │             │               │                    │
│            ▼             ▼             ▼               ▼                    │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                    SIGNAL PROCESSOR                              │     │
│    │                                                                  │     │
│    │  • Normalize data formats                                       │     │
│    │  • Apply staleness policies                                     │     │
│    │  • Tag with timestamps and sources                              │     │
│    │  • Store raw signals in SQLite                                  │     │
│    └────────────────────────────┬────────────────────────────────────┘     │
│                                 │                                           │
│                                 ▼                                           │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                      CLAUDE ANALYST                              │     │
│    │                    (Reasoning Engine)                            │     │
│    │                                                                  │     │
│    │  • Synthesize signals into thesis                               │     │
│    │  • Evaluate bull/bear cases                                     │     │
│    │  • Score conviction (0-100)                                     │     │
│    │  • Generate trade recommendation                                │     │
│    │  • Define exit triggers                                         │     │
│    └────────────────────────────┬────────────────────────────────────┘     │
│                                 │                                           │
│                                 ▼                                           │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                     RISK ENGINE                                  │     │
│    │                                                                  │     │
│    │  • Position size limits (max 25% per position)                  │     │
│    │  • Daily loss limits (5% max drawdown)                          │     │
│    │  • Concentration checks                                         │     │
│    │  • Correlation screening                                        │     │
│    │  • Pass/Fail gate                                               │     │
│    └────────────────────────────┬────────────────────────────────────┘     │
│                                 │                                           │
│                    ┌────────────┴────────────┐                             │
│                    ▼                         ▼                              │
│    ┌────────────────────────┐  ┌────────────────────────┐                  │
│    │     APPROVAL GATE      │  │     AUTO-EXECUTE       │                  │
│    │      (Default ON)      │  │      (Toggle OFF)      │                  │
│    │                        │  │                        │                  │
│    │  → Telegram message    │  │  → Direct to Alpaca    │                  │
│    │  → Human approves      │  │  → Log execution       │                  │
│    │  → Then execute        │  │                        │                  │
│    └───────────┬────────────┘  └───────────┬────────────┘                  │
│                │                           │                                │
│                └─────────────┬─────────────┘                               │
│                              ▼                                              │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                    ALPACA EXECUTOR                               │     │
│    │                                                                  │     │
│    │  • Submit orders (market/limit)                                 │     │
│    │  • Track fills                                                  │     │
│    │  • Update positions                                             │     │
│    │  • Handle errors                                                │     │
│    └────────────────────────────┬────────────────────────────────────┘     │
│                                 │                                           │
│                                 ▼                                           │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                      SQLITE DATABASE                             │     │
│    │                                                                  │     │
│    │  • signals (all raw signals received)                           │     │
│    │  • analyses (Claude's reasoning for each)                       │     │
│    │  • trades (orders, fills, P&L)                                  │     │
│    │  • positions (current holdings)                                 │     │
│    │  • portfolio_snapshots (daily state)                            │     │
│    │  • errors (all failures and exceptions)                         │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Grok Scanner (Sentiment Brain)

**Model:** `grok-4-1-fast-reasoning`

**Responsibilities:**
- Scan X/Twitter for sentiment on watchlist stocks
- Detect emerging narratives and momentum shifts
- Surface breaking news relevant to positions
- Track influential accounts and viral content

**API Calls Used:**
- `x_search`: Twitter/X sentiment ($5/1k calls)
- `web_search`: News and web content ($5/1k calls)

**Output Format:**
```json
{
  "signal_type": "sentiment",
  "source": "x_search",
  "ticker": "RKLB",
  "sentiment_score": 0.72,
  "sentiment_direction": "bullish",
  "volume_change": "+340%",
  "key_narratives": [
    "Neutron rocket progress",
    "SpaceX IPO spillover"
  ],
  "influential_posts": [...],
  "timestamp_utc": "2025-01-15T14:30:00Z",
  "staleness_seconds": 3600
}
```

**Staleness Policy:** 1 hour for sentiment, 4 hours for news

---

### 4.2 Claude Analyst (Reasoning Brain)

**Model:** `claude-sonnet-4-20250514`

**Responsibilities:**
- Synthesize signals from all sources
- Build investment thesis
- Evaluate bull/bear cases
- Score conviction (0-100)
- Generate trade recommendations
- Define entry/exit criteria

**Input:** Structured signals from Grok, FRED, Polymarket, Alpaca

**Output Format:**
```json
{
  "analysis_id": "uuid",
  "timestamp_utc": "2025-01-15T14:45:00Z",
  "ticker": "RKLB",
  "recommendation": "BUY",
  "conviction_score": 85,
  "thesis": "SpaceX IPO speculation driving sector attention...",
  "bull_case": "...",
  "bear_case": "...",
  "position_size_pct": 20,
  "entry_price_target": 22.50,
  "stop_loss_pct": 15,
  "take_profit_triggers": [...],
  "thesis_breakers": [...],
  "time_horizon": "weeks",
  "signals_used": [...]
}
```

**Conviction Thresholds:**
- 80-100: Trade (subject to approval gate)
- 60-79: Log, watch, no trade
- Below 60: No action

---

### 4.3 FRED Integration (Macro Data)

**API:** FRED API (free)

**Data Series Tracked:**
| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| DGS10 | 10-Year Treasury Yield | Daily |
| DGS2 | 2-Year Treasury Yield | Daily |
| T10Y2Y | 10Y-2Y Spread | Daily |
| UNRATE | Unemployment Rate | Monthly |
| CPIAUCSL | CPI (Inflation) | Monthly |
| GDP | Gross Domestic Product | Quarterly |
| FEDFUNDS | Federal Funds Rate | Daily |

**Staleness Policy:** 24 hours for daily, 7 days for monthly/quarterly

---

### 4.4 Polymarket Integration (Predictions)

**API:** Polymarket public API (free)

**Markets Tracked:**
- Federal Reserve rate decisions
- Major election outcomes (if market-relevant)
- Geopolitical events (tariffs, conflicts)
- Specific company events

**Output Format:**
```json
{
  "market_id": "fed-rate-march-2025",
  "question": "Will the Fed cut rates in March 2025?",
  "probability": 0.52,
  "previous_probability": 0.68,
  "change_24h": -0.16,
  "volume_usd": 850000,
  "timestamp_utc": "2025-01-15T14:00:00Z"
}
```

**Staleness Policy:** 1 hour

---

### 4.5 Alpaca Executor

**Mode:** Paper trading (default), Live trading (requires arming)

**Responsibilities:**
- Submit orders (market and limit)
- Track order status and fills
- Maintain position records
- Report account balance and buying power

**Order Types Supported:**
- Market orders (for immediate execution)
- Limit orders (for price targets)
- Stop orders (for stop-losses)

**Safety Controls:**
- Idempotent order keys (prevent duplicates)
- Order state machine (pending → filled → closed)
- Automatic retry with backoff on transient errors

---

### 4.6 Risk Engine

**Hard Limits (Non-Negotiable):**

| Rule | Limit | Action if Breached |
|------|-------|-------------------|
| Max position size | 25% of portfolio | Reject trade |
| Max positions | 5 concurrent | Reject trade |
| Daily loss limit | 5% of portfolio | Halt trading for day |
| Single trade loss | 15% stop-loss | Auto-exit position |
| Min market cap | $500M | Reject trade |
| Leverage | None allowed | Reject trade |

**Soft Limits (Warnings):**
- Position approaching 20% of portfolio
- 3% daily drawdown (yellow alert)
- High correlation between positions

---

### 4.7 Telegram Bot (Notifications)

**Commands:**
| Command | Action |
|---------|--------|
| `/status` | Current positions and P&L |
| `/pending` | Trades awaiting approval |
| `/approve [id]` | Approve a pending trade |
| `/reject [id]` | Reject a pending trade |
| `/stop` | Kill switch - cancel all, halt trading |
| `/resume` | Resume trading after stop |

**Notification Types:**
- Trade recommendations (awaiting approval)
- Trade executions (confirmed)
- Stop-loss triggers
- Daily summaries
- Error alerts

---

## 5. Daily Operating Rhythm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DAILY RHYTHM                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  04:00 PT │ ASIA CLOSE SCAN                                                │
│           │ • Check Nikkei, Hang Seng, Shanghai movements                  │
│           │ • Note any significant overnight developments                   │
│           │                                                                 │
│  05:00 PT │ EUROPE MID-SESSION SCAN                                        │
│           │ • Check FTSE, DAX, CAC movements                               │
│           │ • ECB/BoE policy signals                                       │
│           │                                                                 │
│  06:00 PT │ PRE-MARKET SYNTHESIS                                           │
│           │ • Grok: Overnight news digest                                  │
│           │ • FRED: Treasury yields, any macro releases                    │
│           │ • Polymarket: Odds shifts overnight                            │
│           │ • Alpaca: Pre-market movers                                    │
│           │                                                                 │
│  06:30 PT │ GROK SENTIMENT SCAN                                            │
│           │ • X sentiment on watchlist stocks                              │
│           │ • Emerging narratives                                          │
│           │ • Influential account activity                                 │
│           │                                                                 │
│  07:00 PT │ CLAUDE ANALYSIS                                                │
│           │ • Synthesize all signals                                       │
│           │ • Generate recommendations                                      │
│           │ • Score conviction                                             │
│           │ • Send to approval gate if actionable                          │
│           │                                                                 │
│  09:30 PT │ MARKET OPEN                                                    │
│           │ • Execute approved trades                                      │
│           │ • Log entries with full thesis                                 │
│           │                                                                 │
│  Throughout│ POSITION MONITORING                                           │
│           │ • Watch for thesis-breaking news                               │
│           │ • Track stop-loss levels                                       │
│           │ • Monitor sentiment shifts on holdings                         │
│           │                                                                 │
│  13:00 PT │ MID-DAY CHECK                                                  │
│           │ • Any major news since open?                                   │
│           │ • Position performance update                                  │
│           │ • Re-scan sentiment if significant moves                       │
│           │                                                                 │
│  16:00 PT │ MARKET CLOSE                                                   │
│           │ • End-of-day summary to Telegram                               │
│           │ • Portfolio snapshot saved                                     │
│           │ • Next-day watchlist generated                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Database Schema

### signals
```sql
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    signal_type TEXT NOT NULL,  -- sentiment, macro, prediction, price
    source TEXT NOT NULL,       -- grok, fred, polymarket, alpaca
    ticker TEXT,
    data JSON NOT NULL,
    timestamp_utc TEXT NOT NULL,
    staleness_seconds INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### analyses
```sql
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    ticker TEXT,
    recommendation TEXT,        -- BUY, SELL, HOLD, NONE
    conviction_score INTEGER,
    thesis TEXT,
    full_analysis JSON NOT NULL,
    signals_used JSON,          -- array of signal IDs
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### trades
```sql
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    analysis_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,         -- buy, sell
    quantity REAL NOT NULL,
    order_type TEXT NOT NULL,   -- market, limit, stop
    limit_price REAL,
    status TEXT NOT NULL,       -- pending_approval, approved, submitted, filled, rejected, cancelled
    alpaca_order_id TEXT,
    fill_price REAL,
    fill_quantity REAL,
    filled_at TEXT,
    thesis TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);
```

### positions
```sql
CREATE TABLE positions (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    current_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    unrealized_pnl_pct REAL,
    thesis TEXT,
    entry_date TEXT,
    stop_loss_price REAL,
    take_profit_price REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### portfolio_snapshots
```sql
CREATE TABLE portfolio_snapshots (
    id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    daily_pnl REAL,
    daily_pnl_pct REAL,
    positions JSON,             -- snapshot of all positions
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### errors
```sql
CREATE TABLE errors (
    id TEXT PRIMARY KEY,
    error_type TEXT NOT NULL,
    component TEXT NOT NULL,    -- grok, claude, alpaca, fred, polymarket
    message TEXT NOT NULL,
    stack_trace TEXT,
    context JSON,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 7. Configuration & Environment Variables

### Required Environment Variables

```bash
# Telegram
TELEGRAM_BOT_TOKEN=           # From @BotFather
TELEGRAM_CHAT_ID=             # Our chat ID

# xAI (Grok)
XAI_API_KEY=                  # From console.x.ai

# Anthropic (Claude)
ANTHROPIC_API_KEY=            # From console.anthropic.com

# Alpaca
ALPACA_API_KEY=               # From Alpaca dashboard
ALPACA_SECRET_KEY=            # From Alpaca dashboard
ALPACA_BASE_URL=              # https://paper-api.alpaca.markets (paper)
                              # https://api.alpaca.markets (live)

# FRED
FRED_API_KEY=                 # From FRED website

# System
MODE=PAPER                    # PAPER or LIVE
APPROVAL_GATE=ON              # ON or OFF
LOG_LEVEL=INFO                # DEBUG, INFO, WARN, ERROR
```

### Configuration Defaults

```python
CONFIG = {
    # Trading
    "max_position_pct": 0.25,           # 25% max per position
    "max_positions": 5,                  # 5 concurrent positions
    "min_conviction": 80,                # Only trade 80+ conviction
    "stop_loss_pct": 0.15,              # 15% stop loss
    "daily_loss_limit_pct": 0.05,       # 5% daily loss halt
    "min_market_cap": 500_000_000,      # $500M minimum
    
    # Models
    "grok_model": "grok-4-1-fast-reasoning",
    "claude_model": "claude-sonnet-4-20250514",
    
    # Staleness (seconds)
    "staleness_sentiment": 3600,         # 1 hour
    "staleness_news": 14400,             # 4 hours
    "staleness_macro": 86400,            # 24 hours
    "staleness_prediction": 3600,        # 1 hour
    
    # Scheduling
    "scan_interval_minutes": 60,         # Full scan every hour
    "position_check_minutes": 15,        # Check positions every 15 min
}
```

---

## 8. Setup Checklist

### Accounts & API Keys

| Service | Status | Key Format | Notes |
|---------|--------|------------|-------|
| Telegram | ✅ Bot created | `TOKEN:HASH` | Regenerate after sharing |
| xAI (Grok) | ✅ Key obtained | `xai-...` | Regenerate after sharing |
| Anthropic | ⏳ Pending | `sk-ant-...` | Need $10-20 credits |
| Alpaca | ⏳ Pending | Key + Secret | Start with paper trading |
| FRED | ⏳ Pending | 32-char string | Free, 120 req/min |

### Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| GitHub repo | ⏳ Pending | Will create with code |
| Railway project | ✅ Connected to GitHub | Will deploy after code |

### Security Reminders

- [ ] Regenerate Telegram bot token (shared in chat)
- [ ] Regenerate xAI API key (shared in chat)
- [ ] Never commit API keys to GitHub (use Railway env vars)
- [ ] Enable 2FA on all accounts

---

## 9. Success Metrics

### Phase 1 Metrics (Baseline)

| Metric | Target | How We Measure |
|--------|--------|----------------|
| System uptime | >95% | Railway logs |
| Signals generated | >7/week | Database count |
| Approval flow latency | <5 min response | Telegram timestamps |
| Trade logging completeness | 100% | Database audit |
| Error rate | <5% of runs | Error table count |

### Phase 2+ Metrics (Performance)

| Metric | Target | How We Measure |
|--------|--------|----------------|
| Win rate | >50% | Trades table |
| Average gain/loss ratio | >1.5 | P&L analysis |
| Max drawdown | <15% | Portfolio snapshots |
| Conviction correlation | Positive | Backtest conviction vs outcome |
| Signal-to-trade ratio | Track only | Understand selectivity |

---

## 10. Competition: Agent vs Human

### Structure

| Parameter | Agent | Human (Kyle) |
|-----------|-------|--------------|
| Starting capital | $5,000 | $5,000 |
| Universe | Any US equity | Any US equity |
| Strategy | Autonomous + approval | AI-assisted position trading |
| Leverage | None | None |
| Timeline | Evergreen (ongoing) | Evergreen (ongoing) |

### Scoring

- **Primary:** Total return percentage
- **Secondary:** Risk-adjusted return (Sharpe-like ratio)
- **Tertiary:** Max drawdown

### Monthly Review

First week of each month:
1. Compare portfolio values
2. Review agent's best/worst trades
3. Identify missed opportunities
4. Decide on Phase 2 additions

---

## 11. Risk Disclosures

**IMPORTANT:** This is an experimental trading system.

1. **Past performance does not guarantee future results.**
2. **We are trading real money (after paper phase). Losses can and will occur.**
3. **This system is not financial advice.**
4. **We should only trade capital we can afford to lose.**
5. **Algorithmic systems can fail in unexpected ways.**
6. **Market conditions can change faster than the system adapts.**

By using this system, we acknowledge these risks and accept full responsibility for any outcomes.

---

## Appendix A: File Structure

```
gann-sentinel-trader/
├── config.py                 # Configuration and environment variables
├── models/
│   ├── signals.py           # Signal data structures
│   ├── analysis.py          # Analysis data structures
│   └── trades.py            # Trade and position structures
├── scanners/
│   ├── grok_scanner.py      # Grok API integration
│   ├── fred_scanner.py      # FRED data integration
│   └── polymarket_scanner.py# Polymarket integration
├── analyzers/
│   └── claude_analyst.py    # Claude reasoning engine
├── executors/
│   ├── risk_engine.py       # Risk checks and limits
│   └── alpaca_executor.py   # Trade execution
├── notifications/
│   └── telegram_bot.py      # Telegram notifications and commands
├── storage/
│   └── database.py          # SQLite operations
├── agent.py                  # Main orchestration loop
├── requirements.txt          # Python dependencies
├── README.md                 # Setup instructions
└── .env.example             # Environment variable template
```

---

## Appendix B: Grok Signal Schema

Based on the Grok Signal Intelligence Agent Spec (v1.1.0), signals follow this structure:

```json
{
  "schema_version": "1.1.0",
  "signals": [
    {
      "signal_id": "<uuid-v4>",
      "category": "<macro|sentiment|policy|event|prediction_market|narrative_shift>",
      "source_type": "<polymarket|fred|news|social|official>",
      "asset_scope": {
        "tickers": ["SYMBOL"],
        "sectors": ["SECTOR"],
        "macro_regions": ["US|EU|ASIA|GLOBAL"],
        "asset_classes": ["EQUITY|FIXED_INCOME|CRYPTO|COMMODITY|FX"]
      },
      "summary": "1-2 sentence factual description",
      "raw_value": {
        "type": "<probability|rate|index|count|price|null>",
        "value": "<number or null>",
        "unit": "<percent|bps|usd|index_points|null>",
        "prior_value": "<number or null>",
        "change": "<number or null>",
        "change_period": "<string, e.g., '24h', '1w'>"
      },
      "evidence": [...],
      "confidence": "<float 0.0-1.0>",
      "directional_bias": "<positive|negative|mixed|unclear>",
      "time_horizon": "<intraday|days|weeks|months|unknown>",
      "novelty": "<new|developing|recurring>",
      "staleness_policy": {
        "max_age_seconds": "<integer>",
        "stale_after_utc": "<ISO-8601>"
      },
      "uncertainties": ["<what is unclear or incomplete>"]
    }
  ],
  "meta": {
    "query_context": "<what was searched>",
    "retrieval_time_utc": "<ISO-8601>",
    "retrieval_errors": [...],
    "known_blindspots": [...],
    "signal_disclaimer": "Signals are informational only."
  }
}
```

---

**End of Framework Document**

*Trading involves substantial risk of loss. Nothing in this document constitutes financial advice.*
