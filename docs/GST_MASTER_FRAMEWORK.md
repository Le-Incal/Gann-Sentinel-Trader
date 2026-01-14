# Gann Sentinel Trader (GST) - Master Framework Document

**Version:** 2.4.2  
**Last Updated:** January 14, 2026  
**Status:** Production (Railway)  
**Repository:** https://github.com/Le-Incal/Gann-Sentinel-Trader.git

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy](#2-core-philosophy)
3. [System Architecture](#3-system-architecture)
4. [Signal Scanners](#4-signal-scanners)
5. [Multi-Agent Consensus Architecture (MACA)](#5-multi-agent-consensus-architecture-maca)
6. [Risk Engine](#6-risk-engine)
7. [Trade Execution](#7-trade-execution)
8. [Telegram Bot Interface](#8-telegram-bot-interface)
9. [Learning Engine](#9-learning-engine)
10. [Smart Scheduling](#10-smart-scheduling)
11. [Logs API](#11-logs-api)
12. [Database Schema](#12-database-schema)
13. [File Structure](#13-file-structure)
14. [Configuration](#14-configuration)
15. [Deployment](#15-deployment)
16. [Cost Analysis](#16-cost-analysis)
17. [Version History](#17-version-history)
18. [Appendix](#18-appendix)

---

## 1. Executive Summary

Gann Sentinel Trader (GST) is an AI-powered autonomous trading system that combines multiple AI agents for market analysis and decision-making. The system follows a safety-first approach with human approval required for all trades.

### Key Capabilities

- **Multi-Source Signal Generation:** 5 scanners (Grok, FRED, Polymarket, Technical, Event)
- **AI Council Analysis:** 4 AI systems generate and synthesize investment theses
- **Risk-First Execution:** Multi-layer risk validation before any trade
- **Human-in-the-Loop:** Telegram-based approval workflow
- **Performance Learning:** Tracks outcomes and adapts over time
- **Full Observability:** Logs API for remote monitoring

### Current State

| Metric | Value |
|--------|-------|
| Portfolio | $100,000 (paper) |
| Trading Mode | Paper Trading (Alpaca) |
| Scan Schedule | 2x daily (9:35 AM, 12:30 PM ET) |
| Conviction Threshold | 80/100 |
| Deployment | Railway (auto-deploy from GitHub) |

---

## 2. Core Philosophy

### ANCHOR in History, ORIENT Toward Future

The system rejects both purely backward-looking analysis and purely predictive approaches. Instead, it combines:

1. **Historical Pattern Recognition** - "When has this happened before, and what followed?"
2. **Forward Catalyst Analysis** - "What events are coming that will move prices?"
3. **Second-Order Thinking** - "Who benefits that isn't obvious?"

### The SpaceX Example

```
Signal: "SpaceX IPO expected H2 2026"

First-Order Thinking (what most do):
  → "I should buy SpaceX" → Can't, it's private

Second-Order Thinking (what GST does):
  → "SpaceX IPO brings attention to entire space sector"
  → "Investors will comparison shop for public alternatives"
  → "Rocket Lab (RKLB) is the most comparable public company"
  → Trade: BUY RKLB ahead of SpaceX IPO announcement
```

### Key Principles

| Principle | Implementation |
|-----------|----------------|
| Safety First | Human approval gate for all trades |
| Lean Development | Phase-constrained, complete Phase 1 before Phase 2 |
| Observability | Full logging, remote API access |
| Mechanical Rules | Conviction thresholds, position sizing formulas |
| Diversified Signals | Multiple AI sources prevent echo chambers |

---

## 3. System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GANN SENTINEL TRADER v2.4.2                         │
│                    "ANCHOR in history, ORIENT toward future"                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│    SCANNERS     │           │    ANALYZERS    │           │    EXECUTORS    │
│  (Data Input)   │           │  (AI Council)   │           │    (Output)     │
└─────────────────┘           └─────────────────┘           └─────────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│ • Grok          │           │ • Grok Thesis   │           │ • Risk Engine   │
│ • FRED          │    ───►   │ • Perplexity    │    ───►   │ • Alpaca        │
│ • Polymarket    │           │ • ChatGPT       │           │ • Telegram      │
│ • Technical     │           │ • Claude        │           │                 │
│ • Event         │           │   (Synthesis)   │           │                 │
└─────────────────┘           └─────────────────┘           └─────────────────┘
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │       DATABASE        │
                          │  (SQLite + Logs API)  │
                          └───────────────────────┘
```

### Data Flow

```
1. SIGNAL COLLECTION (Parallel)
   ├── Grok: Social sentiment, catalysts
   ├── FRED: Macro indicators
   ├── Polymarket: Prediction probabilities
   ├── Technical: Chart patterns, support/resistance
   └── Event: Corporate actions (27 types)
              │
              ▼
2. AI COUNCIL (MACA - Parallel + Sequential)
   ├── Phase 1: Grok, Perplexity, ChatGPT generate theses (parallel)
   ├── Phase 2: Claude synthesizes all theses
   ├── Phase 3: Peer review (if conviction ≥ 80)
   └── Phase 4: Final decision
              │
              ▼
3. RISK VALIDATION
   ├── Position size check (max 20%)
   ├── Daily loss check (max 3%)
   ├── Concentration check (max 40% sector)
   └── Liquidity check (min $1M volume)
              │
              ▼
4. HUMAN APPROVAL (Telegram)
   ├── Trade details displayed
   ├── Inline buttons: [APPROVE] [REJECT]
   └── User makes final call
              │
              ▼
5. EXECUTION (Alpaca)
   ├── Market order submitted
   ├── Stop loss set
   └── Position tracked
```

---

## 4. Signal Scanners

### 4.1 Overview

| Scanner | Source | Data Type | Update Frequency |
|---------|--------|-----------|------------------|
| Grok | xAI API | Social sentiment, news | Real-time |
| FRED | Federal Reserve | Macro indicators | Daily/Weekly |
| Polymarket | Polymarket API | Prediction probabilities | Real-time |
| Technical | Alpaca Data | Price history, patterns | Real-time |
| Event | Grok (parsed) | Corporate events | Real-time |

### 4.2 Grok Scanner

**Purpose:** Capture social sentiment, trending narratives, and retail momentum from X/Twitter and web sources.

**API:** xAI `grok-3-latest` with `live_search` tool

**Signal Types:**
- `sentiment` - Overall market/ticker sentiment
- `catalyst` - Upcoming events that could move prices

**Example Output:**
```json
{
  "signal_id": "uuid-v4",
  "category": "sentiment",
  "source_type": "grok_x",
  "asset_scope": {
    "tickers": ["TSLA"],
    "sectors": ["AUTOMOTIVE", "TECH"]
  },
  "summary": "Strong bullish sentiment on X around FSD breakthrough...",
  "directional_bias": "bullish",
  "confidence": 0.78,
  "time_horizon": "days"
}
```

### 4.3 FRED Scanner

**Purpose:** Monitor macroeconomic indicators that affect market direction.

**Data Series:**

| Series ID | Name | Frequency | Trading Implication |
|-----------|------|-----------|---------------------|
| DGS10 | 10-Year Treasury | Daily | >4.5% = tight conditions |
| DGS2 | 2-Year Treasury | Daily | Fed expectations |
| T10Y2Y | Yield Curve Spread | Daily | <0 = recession signal |
| UNRATE | Unemployment | Monthly | >5% = Fed pivot likely |
| CPIAUCSL | CPI Inflation | Monthly | >3% = restrictive Fed |
| GDP | GDP Growth | Quarterly | Recession indicator |
| FEDFUNDS | Fed Funds Rate | Daily | Current policy stance |

**Forward Context Logic:**
```
If 10Y Yield > 4.5% → directional_bias = "negative" (tight conditions)
If 10Y Yield < 3.5% → directional_bias = "positive" (easing)
If Yield Curve < 0 → directional_bias = "negative" (recession signal)
If CPI > 3% → directional_bias = "negative" (restrictive Fed)
```

### 4.4 Polymarket Scanner

**Purpose:** Extract forward-looking probabilities from prediction markets.

**Investment Categories (17):**

| Category | Keywords | Trading Relevance |
|----------|----------|-------------------|
| FEDERAL_RESERVE | fed, fomc, rate cut, powell | Interest rate sensitive stocks |
| INFLATION | cpi, inflation, prices | TIPS, commodities |
| RECESSION | recession, gdp, contraction | Defensive positioning |
| TRADE_POLICY | tariff, trade war, sanctions | Import/export exposed |
| CHINA_RISK | china, taiwan, xi | Supply chain, tech |
| AI_SECTOR | ai, artificial intelligence | NVDA, MSFT, GOOGL |
| SEMICONDUCTOR | chip, semiconductor, nvidia | SMH, SOXX holdings |
| CRYPTO_POLICY | bitcoin, crypto, sec crypto | COIN, MSTR, miners |

**Filtering:** Sports, entertainment, and non-investment markets are excluded.

**Momentum Tracking:** Flags ±10% probability changes within 24 hours.

### 4.5 Technical Scanner

**Purpose:** Analyze 5-year price history for patterns, support/resistance, and trend state.

**API:** Alpaca Market Data API

**Analysis Components:**

| Component | Description |
|-----------|-------------|
| Market State | TRENDING, RANGING, BREAKOUT, BREAKDOWN |
| Directional Bias | bullish, bearish, neutral |
| Channel Position | % from bottom of historical range |
| Support/Resistance | Key price levels |
| Volume Profile | Relative volume analysis |

**Timeframes:**
- `/scan` command: 1-year daily (fast)
- `/check` command: 5-year weekly (comprehensive)

### 4.6 Event Scanner

**Purpose:** Detect 27 corporate event types that historically move stock prices.

**Event Categories:**

**Leadership (5):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| CEO_EXIT | mixed | varies | varies |
| CEO_APPOINTMENT | mixed | varies | varies |
| INSIDER_BUYING | bullish | +6% | 65% |
| INSIDER_SELLING | bearish | -3% | 55% |

**Capital Allocation (4):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| STOCK_BUYBACK | bullish | +5% | 68% |
| DIVIDEND_INCREASE | bullish | +3% | 72% |
| DIVIDEND_CUT | bearish | -8% | 70% |

**Regulatory (5):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| FDA_APPROVAL | bullish | +15% | 78% |
| FDA_REJECTION | bearish | -25% | 80% |
| FDA_BREAKTHROUGH | bullish | +12% | 75% |

**Index Changes (3):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| SP500_ADDITION | bullish | +8% | 80% |
| SP500_REMOVAL | bearish | -10% | 75% |

**External Pressure (3):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| ACTIVIST_INVESTOR | bullish | +10% | 62% |
| SHORT_SELLER_REPORT | bearish | -15% | 65% |

**Contracts (3):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| GOVERNMENT_CONTRACT | bullish | +7% | 70% |
| MAJOR_PARTNERSHIP | bullish | +6% | 65% |
| CONTRACT_LOSS | bearish | -8% | 68% |

**Corporate Actions (4):**
| Event | Bias | Historical Avg Move | Win Rate |
|-------|------|---------------------|----------|
| BANKRUPTCY_FILING | bearish | -50% | 90% |
| MA_ANNOUNCEMENT | mixed | varies | 52% |

---

## 5. Multi-Agent Consensus Architecture (MACA)

### 5.1 The Problem MACA Solves

The original single-AI architecture surfaced the same stocks repeatedly (TSLA, NVDA, PLTR, MSTR) because Grok scans X/Twitter, which has concentrated retail sentiment around popular tech stocks.

MACA diversifies signal sources to capture different perspectives:
- **Grok:** Social sentiment, trending narratives
- **Perplexity:** Fundamental research, citations
- **ChatGPT:** Pattern recognition, risk scenarios

### 5.2 The AI Council

| AI | Role | Specialty | Model | Cost/1K tokens |
|----|------|-----------|-------|----------------|
| Grok | Signal Generator | Social sentiment | grok-3-latest | $0.005 |
| Perplexity | Researcher | Fundamentals | sonar-pro | $0.003 |
| ChatGPT | Pattern Finder | Technical | gpt-4o | $0.005 |
| Claude | Senior Trader | Synthesis | claude-3-5-sonnet | $0.003 |

### 5.3 The 4-Phase MACA Cycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: PARALLEL THESIS GENERATION                  │
│                              (Async - ~15 seconds)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │    GROK     │      │  PERPLEXITY │      │   CHATGPT   │               │
│    │  (Sentiment)│      │ (Fundamental)│      │  (Patterns) │               │
│    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘               │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ Thesis A    │      │ Thesis B    │      │ Thesis C    │               │
│    │ Ticker: TSLA│      │ Ticker: NVDA│      │ Ticker: NVDA│               │
│    │ Conv: 78    │      │ Conv: 82    │      │ Conv: 85    │               │
│    └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: CLAUDE SYNTHESIS                           │
│                              (~10 seconds)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Claude receives all 3 theses + portfolio context + signals              │
│                                                                             │
│    Analysis:                                                                │
│    - Grok: TSLA sentiment-driven, conviction 78 (below threshold)          │
│    - Perplexity: NVDA fundamental case, conviction 82 ✓                    │
│    - ChatGPT: NVDA technical breakout, conviction 85 ✓                     │
│                                                                             │
│    Synthesis:                                                               │
│    - 2/3 analysts converge on NVDA                                         │
│    - Technical + fundamental alignment = higher confidence                  │
│    - Historical: Similar setups +12% in 30 days                            │
│                                                                             │
│    Output: BUY NVDA, Conviction 83/100                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (Only if conviction ≥ 80)
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3: PEER REVIEW                               │
│                              (~10 seconds)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Grok reviews Claude's synthesis:                                         │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────┐         │
│    │ GROK REVIEW                                                  │         │
│    │                                                              │         │
│    │ Verdict: APPROVE                                             │         │
│    │ Concerns: None significant                                   │         │
│    │ Confidence Adjustment: +2                                    │         │
│    │ Comment: "Strong social momentum confirms thesis"            │         │
│    └─────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: FINAL DECISION                             │
│                              (~5 seconds)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Final Conviction: 85 (83 + 2 from peer review)                          │
│    Proceed: YES                                                             │
│                                                                             │
│    Trade Created:                                                           │
│    - Ticker: NVDA                                                           │
│    - Side: BUY                                                              │
│    - Shares: 37 (12% of $100k = $12k / $324)                               │
│    - Stop Loss: $298 (8% below entry)                                       │
│    - Status: PENDING_APPROVAL                                               │
│                                                                             │
│    → Sent to Telegram for human approval                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 MACA Prompt Structure

Each AI receives identical context:

```
CONTEXT PROVIDED TO ALL AIs:
- Current portfolio positions and P&L
- Available cash for new positions
- Recent FRED macro signals
- Recent Polymarket predictions
- Technical analysis (if available)
- Market context (Learning Engine history)

TASK:
Generate ONE investment thesis with:
- Ticker recommendation
- BUY/SELL/HOLD decision
- Conviction score (0-100)
- Thesis (2-3 sentences)
- Catalyst with timeline
- Stop loss %
- Position size %
```

---

## 6. Risk Engine

### 6.1 Risk Checks

| Check | Rule | Severity | Action if Failed |
|-------|------|----------|------------------|
| Position Size | Max 20% of portfolio | error | Block trade |
| Daily Loss | Max 3% drawdown | error | Block trade |
| Concentration | Max 40% in single sector | warning | Log warning |
| Correlation | Avoid 80%+ correlated | warning | Log warning |
| Liquidity | Min $1M daily volume | error | Block trade |

### 6.2 Trade Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Conviction Threshold | 80 | 0-100 | Below = no trade |
| Max Position Size | 20% | 5-25% | Per position |
| Default Stop Loss | 8% | 5-15% | Below entry |
| Max Daily Trades | 3 | 1-10 | Circuit breaker |
| Approval Required | true | true/false | Human gate |

### 6.3 Position Sizing Formula

```python
position_value = portfolio_equity * (position_size_pct / 100)
shares = int(position_value / current_price)

# Example:
# Equity: $100,000
# Position Size: 12%
# Stock Price: $324
# Position Value: $12,000
# Shares: 37
```

---

## 7. Trade Execution

### 7.1 Alpaca Integration

**Mode:** Paper Trading (sandbox)
**Order Types:** Market orders only (Phase 1)
**Stop Losses:** Set after fill confirmation

### 7.2 Trade Lifecycle

```
1. PENDING_APPROVAL
   └── Waiting for human via Telegram

2. APPROVED
   └── Human clicked [APPROVE]

3. SUBMITTED
   └── Order sent to Alpaca

4. FILLED
   └── Execution confirmed

5. REJECTED (alternate path)
   └── Human clicked [REJECT]

6. CANCELLED (alternate path)
   └── Order failed or cancelled
```

### 7.3 Trade Record Structure

```json
{
  "id": "uuid-v4",
  "analysis_id": "uuid-v4",
  "ticker": "NVDA",
  "side": "BUY",
  "quantity": 37,
  "order_type": "MARKET",
  "status": "PENDING_APPROVAL",
  "thesis": "Technical breakout with fundamental support...",
  "conviction_score": 85,
  "stop_loss_price": 298.00,
  "created_at": "2026-01-14T14:35:00Z",
  "approved_at": null,
  "filled_at": null,
  "fill_price": null,
  "order_id": null
}
```

---

## 8. Telegram Bot Interface

### 8.1 Commands Reference

| Command | Syntax | Description |
|---------|--------|-------------|
| `/scan` | `/scan` | Run full MACA scan cycle |
| `/check` | `/check NVDA` | Analyze specific ticker |
| `/status` | `/status` | Portfolio and system health |
| `/positions` | `/positions` | Current open positions |
| `/history` | `/history 20` | Last N trades |
| `/pending` | `/pending` | Trades awaiting approval |
| `/approve` | `/approve abc123` | Approve trade by ID |
| `/reject` | `/reject abc123` | Reject trade by ID |
| `/export` | `/export csv` | Export data (csv/parquet) |
| `/cost` | `/cost 7` | API costs for last N days |
| `/logs` | `/logs` | Recent activity |
| `/digest` | `/digest` | Send daily summary |
| `/stop` | `/stop` | Emergency halt |
| `/resume` | `/resume` | Resume trading |
| `/help` | `/help` | Show all commands |

### 8.2 MACA Message Format

**Message 1: AI Council Views**
```
========================================
🔍 MACA SCAN - AI COUNCIL
2026-01-14 09:35 UTC
========================================

🐦 GROK
------------------------------
Ticker: TSLA
Action: BUY
Conviction: 78/100
[████████░░]

Thesis: Strong bullish sentiment on X around 
FSD breakthrough. Retail momentum building...

Catalyst: FSD v13 release expected
Horizon: days

🎯 PERPLEXITY
------------------------------
Ticker: NVDA
Action: BUY
Conviction: 82/100
[████████░░] 🟢

Thesis: Datacenter revenue accelerating per 
channel checks. H100 demand exceeding supply...

Catalyst: Earnings Feb 21
Horizon: weeks

🧠 CHATGPT
------------------------------
Ticker: NVDA
Action: BUY
Conviction: 85/100
[█████████░] 🟢

Thesis: Technical breakout from 6-month 
consolidation. Volume confirming move...

Catalyst: Technical breakout
Horizon: days

========================================
🧠 Claude's synthesis follows...
```

**Message 2: Claude Decision**
```
========================================
🕯️ CHART ANALYSIS
----------------------------------------
• NVDA @ $324.50
  State: 📈 TRENDING (bullish, high conf)
  Channel: 72% from bottom
  Verdict: ✅ HYPOTHESIS ALLOWED

========================================
🧠 CLAUDE'S SYNTHESIS (Senior Trader)
========================================

Decision: TRADE
Selected: CHATGPT proposal (confirmed by PERPLEXITY)

Recommendation: BUY NVDA
Conviction: 85/100
[█████████░] 🟢 ACTIONABLE

Thesis: Two analysts converged on NVDA with 
technical + fundamental alignment. Historical 
pattern suggests +12% move in similar setups.

Trade Parameters:
  Stop Loss: 8%
  Position Size: 12%

----------------------------------------
💰 PORTFOLIO
  Equity: $100,000.00
  Cash: $100,000.00
  Positions: 0

----------------------------------------
🔔 TRADE PENDING APPROVAL
Trade ID: abc12345
BUY 37 NVDA @ $324.50

[ ✅ APPROVE ]  [ ❌ REJECT ]
```

### 8.3 Inline Buttons

Trades include clickable buttons:
- **APPROVE** - Executes the trade
- **REJECT** - Cancels the trade

Command shortcuts also available:
- **Status** - Quick status check
- **Pending** - View pending trades
- **Scan** - Trigger manual scan
- **Help** - Show commands

---

## 9. Learning Engine

### 9.1 Purpose

Track trading performance over time to:
1. Compare returns vs SPY benchmark
2. Identify which AI sources are most accurate
3. Provide historical context to Claude
4. Adapt conviction thresholds based on track record

### 9.2 Metrics Tracked

| Metric | Description |
|--------|-------------|
| Total Trades | Count of executed trades |
| Win Rate | % of trades with positive return |
| Avg Return | Mean return per trade |
| Max Drawdown | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted return |
| SPY Benchmark | Comparison vs S&P 500 |
| Signal Accuracy | Per-source hit rate |

### 9.3 Context Injection

Before each Claude analysis, the Learning Engine generates context:

```python
learning_context = {
    "performance_summary": {
        "total_trades": 47,
        "win_rate": 0.62,
        "avg_return": 0.034,
        "vs_spy": "+2.3%"
    },
    "signal_accuracy": {
        "grok": 0.58,
        "perplexity": 0.67,
        "chatgpt": 0.63,
        "events": 0.71
    },
    "recent_trades": [...],
    "sector_performance": {...}
}
```

Claude receives this as part of the portfolio context, enabling it to:
- Weight sources by historical accuracy
- Avoid repeating recent losing patterns
- Calibrate conviction based on track record

---

## 10. Smart Scheduling

### 10.1 Scan Schedule

| Time (ET) | Time (UTC) | Scan Type | Rationale |
|-----------|------------|-----------|-----------|
| 9:35 AM | 14:35 | Morning | 5 min after open, initial price discovery |
| 12:30 PM | 17:30 | Midday | Lunch lull, reassess morning moves |

### 10.2 Schedule Rules

- **Weekdays only** - Markets closed Sat/Sun
- **Market hours** - No overnight scans
- **Manual override** - `/scan` and `/check` always work

### 10.3 Cost Savings

| Mode | Scans/Day | Scans/Month | Monthly Cost |
|------|-----------|-------------|--------------|
| Hourly (old) | 8 | 160 | ~$280 |
| Smart (new) | 2 | 40 | ~$70 |
| **Savings** | **75%** | | **$210/month** |

---

## 11. Logs API

### 11.1 Endpoints

| Endpoint | Auth | Method | Description |
|----------|------|--------|-------------|
| `/health` | No | GET | Service health check |
| `/api/status` | Token | GET | Full system status |
| `/api/logs` | Token | GET | Telegram message history |
| `/api/errors` | Token | GET | System errors |
| `/api/signals` | Token | GET | Recent signals |
| `/api/scan_cycles` | Token | GET | MACA cycle history |

### 11.2 Authentication

```
Header: Authorization: Bearer <token>
  OR
Query: ?token=<token>
```

### 11.3 Example Requests

**Health Check (no auth):**
```bash
curl https://gann-sentinel-trader-production.up.railway.app/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-14T01:12:16.159001+00:00",
  "service": "gst-logs-api"
}
```

**System Status:**
```bash
curl "https://gann-sentinel-trader-production.up.railway.app/api/status?token=YOUR_TOKEN"
```

Response:
```json
{
  "status": "success",
  "timestamp": "2026-01-14T01:09:50.336472+00:00",
  "portfolio": {
    "cash": 100000.0,
    "positions_value": 0.0,
    "total_value": 100000.0
  },
  "positions": [],
  "pending_trades": [],
  "recent_errors": [...]
}
```

### 11.4 Access Details

```
Base URL: https://gann-sentinel-trader-production.up.railway.app
Token: QzHBtENzt-sYeLXKSUzEN_v6VREwfEnGaqpoQVmOBWE
```

---

## 12. Database Schema

### 12.1 Core Tables

**signals**
```sql
CREATE TABLE signals (
    id TEXT PRIMARY KEY,
    category TEXT,
    source_type TEXT,
    ticker TEXT,
    summary TEXT,
    directional_bias TEXT,
    confidence REAL,
    time_horizon TEXT,
    raw_data TEXT,
    created_at TIMESTAMP
);
```

**analyses**
```sql
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    ticker TEXT,
    recommendation TEXT,
    conviction_score INTEGER,
    thesis TEXT,
    position_size_pct REAL,
    stop_loss_pct REAL,
    bull_case TEXT,
    bear_case TEXT,
    created_at TIMESTAMP
);
```

**trades**
```sql
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    analysis_id TEXT,
    ticker TEXT,
    side TEXT,
    quantity INTEGER,
    order_type TEXT,
    status TEXT,
    thesis TEXT,
    conviction_score INTEGER,
    stop_loss_price REAL,
    order_id TEXT,
    fill_price REAL,
    created_at TIMESTAMP,
    approved_at TIMESTAMP,
    filled_at TIMESTAMP
);
```

**portfolio_snapshots**
```sql
CREATE TABLE portfolio_snapshots (
    id TEXT PRIMARY KEY,
    timestamp_utc TIMESTAMP,
    cash REAL,
    positions_value REAL,
    total_value REAL,
    daily_pnl REAL,
    positions TEXT
);
```

### 12.2 MACA Tables

**scan_cycles**
```sql
CREATE TABLE scan_cycles (
    cycle_id TEXT PRIMARY KEY,
    timestamp_utc TIMESTAMP,
    cycle_type TEXT,
    status TEXT,
    duration_seconds REAL,
    proposals_count INTEGER,
    final_decision TEXT
);
```

**ai_proposals**
```sql
CREATE TABLE ai_proposals (
    proposal_id TEXT PRIMARY KEY,
    scan_cycle_id TEXT,
    ai_source TEXT,
    ticker TEXT,
    side TEXT,
    conviction_score INTEGER,
    thesis TEXT,
    timestamp_utc TIMESTAMP
);
```

**ai_reviews**
```sql
CREATE TABLE ai_reviews (
    review_id TEXT PRIMARY KEY,
    scan_cycle_id TEXT,
    reviewer_ai TEXT,
    verdict TEXT,
    concerns TEXT,
    confidence_adjustment INTEGER,
    timestamp_utc TIMESTAMP
);
```

### 12.3 Logging Tables

**telegram_messages**
```sql
CREATE TABLE telegram_messages (
    id INTEGER PRIMARY KEY,
    direction TEXT,
    message_type TEXT,
    content TEXT,
    chat_id TEXT,
    message_id TEXT,
    timestamp_utc TIMESTAMP
);
```

**error_logs**
```sql
CREATE TABLE error_logs (
    id INTEGER PRIMARY KEY,
    error_type TEXT,
    component TEXT,
    message TEXT,
    stack_trace TEXT,
    context TEXT,
    created_at TIMESTAMP
);
```

---

## 13. File Structure

```
gann-sentinel-trader/
├── agent.py                      # Main orchestrator (v2.4.2)
├── config.py                     # Configuration management
├── learning_engine.py            # Performance tracking
├── requirements.txt              # Python dependencies
├── requirements_api.txt          # API server dependencies
│
├── scanners/
│   ├── __init__.py               # Module exports
│   ├── temporal.py               # Shared temporal framework
│   ├── grok_scanner.py           # Grok sentiment/catalysts
│   ├── fred_scanner.py           # FRED macro data
│   ├── polymarket_scanner.py     # Prediction markets
│   ├── technical_scanner.py      # Chart analysis
│   └── event_scanner.py          # Corporate events (27 types)
│
├── analyzers/
│   ├── claude_analyst.py         # Claude analysis engine
│   ├── claude_maca_extension.py  # MACA synthesis capability
│   ├── perplexity_analyst.py     # Perplexity integration
│   └── chatgpt_analyst.py        # ChatGPT integration
│
├── core/
│   └── maca_orchestrator.py      # 4-phase MACA cycle
│
├── executors/
│   ├── risk_engine.py            # Risk validation
│   └── alpaca_executor.py        # Trade execution
│
├── notifications/
│   └── telegram_bot.py           # Bot interface (v2.2.0)
│
├── storage/
│   └── database.py               # SQLite + all tables (v2.1.0)
│
├── api/
│   └── logs_api.py               # HTTP API for remote access
│
├── models/
│   ├── signals.py                # Signal dataclasses
│   ├── analysis.py               # Analysis dataclasses
│   └── trades.py                 # Trade dataclasses
│
├── utils/
│   └── data_exporter.py          # CSV/Parquet export
│
├── docs/
│   ├── MACA_SPEC_v1.md           # MACA architecture doc
│   ├── PHASE2_DEPLOYMENT_GUIDE.md
│   ├── FORWARD_PREDICTIVE_SYSTEM_v2.1.md
│   └── GST_MASTER_FRAMEWORK.md   # This document
│
└── main_with_api.py              # Entry point with API server
```

---

## 14. Configuration

### 14.1 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | Yes | Grok API access |
| `ANTHROPIC_API_KEY` | Yes | Claude API access |
| `PERPLEXITY_API_KEY` | Yes* | Perplexity API (MACA) |
| `OPENAI_API_KEY` | Yes* | ChatGPT API (MACA) |
| `ALPACA_API_KEY` | Yes | Trading + market data |
| `ALPACA_SECRET_KEY` | Yes | Alpaca authentication |
| `ALPACA_PAPER` | Yes | "true" for paper trading |
| `TELEGRAM_BOT_TOKEN` | Yes | Bot authentication |
| `TELEGRAM_CHAT_ID` | Yes | Your chat ID |
| `MACA_ENABLED` | No | "true" to enable 4-AI mode |
| `LOGS_API_TOKEN` | No | API authentication token |
| `LOG_LEVEL` | No | DEBUG/INFO/WARNING/ERROR |

*Required when MACA_ENABLED=true

### 14.2 Config Defaults

```python
# config.py defaults
CONVICTION_THRESHOLD = 80
MAX_POSITION_SIZE_PCT = 20
DEFAULT_STOP_LOSS_PCT = 8
APPROVAL_GATE = True
SCAN_INTERVAL_MINUTES = 60
DAILY_LOSS_LIMIT_PCT = 3
```

### 14.3 Watchlist

Default tickers monitored:
```python
WATCHLIST = [
    "TSLA", "NVDA", "RKLB", "PLTR", "MSTR",
    "COIN", "HOOD", "SOFI", "AMD", "SMCI"
]
```

---

## 15. Deployment

### 15.1 Platform

| Component | Service |
|-----------|---------|
| Hosting | Railway |
| Source Control | GitHub |
| Deployment | Auto-deploy on push |
| Database | SQLite (persistent volume) |
| Logs | Railway logging + Logs API |

### 15.2 URLs

| Environment | URL |
|-------------|-----|
| Production | https://gann-sentinel-trader-production.up.railway.app |
| Health Check | /health |
| Logs API | /api/* |

### 15.3 Deployment Steps

```bash
# 1. Make changes locally
git add .
git commit -m "description of changes"

# 2. Push to GitHub (auto-deploys to Railway)
git push origin main

# 3. Monitor deployment in Railway dashboard

# 4. Verify via health check
curl https://gann-sentinel-trader-production.up.railway.app/health
```

### 15.4 Rollback

```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Or deploy specific commit in Railway dashboard
```

---

## 16. Cost Analysis

### 16.1 API Costs per Scan

| Service | Tokens/Scan | Cost/1K | Cost/Scan |
|---------|-------------|---------|-----------|
| Grok (thesis) | ~2,000 | $0.005 | $0.010 |
| Perplexity | ~1,500 | $0.003 | $0.005 |
| ChatGPT | ~2,000 | $0.005 | $0.010 |
| Claude (synthesis) | ~3,000 | $0.003 | $0.009 |
| Claude (review) | ~1,000 | $0.003 | $0.003 |
| **Total/Scan** | | | **$0.037** |

### 16.2 Monthly Projections

| Mode | Scans/Day | Monthly Scans | API Cost | Total |
|------|-----------|---------------|----------|-------|
| Phase 1 (Claude only) | 2 | 40 | ~$45 | ~$45 |
| Phase 2 (MACA) | 2 | 40 | ~$70 | ~$70 |

### 16.3 Additional Costs

| Service | Cost |
|---------|------|
| Railway Hosting | ~$5/month |
| Alpaca Data | Free (paper) |
| Telegram | Free |

---

## 17. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2025 | Initial Phase 1 release |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 2.1.0 | Jan 2026 | Historical pattern recognition |
| 2.2.0 | Jan 2026 | MACA for /check command |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.4.1 | Jan 2026 | Trade blocker visibility fix |
| **2.4.2** | Jan 2026 | **Full MACA for scheduled scans, analysis.id fix** |

---

## 18. Appendix

### 18.1 Signal Schema (Grok Spec v1.1.0)

```json
{
  "signal_id": "uuid-v4",
  "dedup_hash": "sha256",
  "category": "macro|sentiment|prediction_market|event|technical",
  "source_type": "grok_x|grok_web|fred|polymarket|alpaca|event",
  "asset_scope": {
    "tickers": ["SPY", "NVDA"],
    "sectors": ["TECH"],
    "macro_regions": ["US"],
    "asset_classes": ["EQUITY"]
  },
  "summary": "Forward-looking description...",
  "raw_value": {},
  "confidence": 0.75,
  "confidence_factors": {},
  "directional_bias": "positive|negative|mixed|unclear",
  "time_horizon": "intraday|days|weeks|months",
  "staleness_policy": {
    "max_age_hours": 24,
    "decay_type": "linear"
  },
  "timestamp_utc": "ISO-8601",
  "forward_horizon": "short-term (1 month)",
  "forward_implication": "What this means going forward...",
  "catalyst_date": "2026-02-15"
}
```

### 18.2 Analysis Schema

```json
{
  "analysis_id": "uuid-v4",
  "ticker": "NVDA",
  "recommendation": "BUY|SELL|HOLD",
  "conviction_score": 85,
  "thesis": "Main investment thesis...",
  "bull_case": "Best case scenario...",
  "bear_case": "Risk factors...",
  "position_size_pct": 12,
  "stop_loss_pct": 8,
  "time_horizon": "weeks",
  "catalyst": "Earnings Feb 21",
  "historical_context": {
    "analogous_event": "Similar setup in Nov 2023",
    "outcome": "+15% in 30 days",
    "confidence": 0.72
  },
  "timestamp_utc": "ISO-8601"
}
```

### 18.3 Glossary

| Term | Definition |
|------|------------|
| MACA | Multi-Agent Consensus Architecture |
| Conviction | Confidence score 0-100 for trade recommendation |
| Thesis | Investment rationale explaining the trade |
| Catalyst | Event expected to trigger price movement |
| Stop Loss | Price level to exit losing position |
| Position Size | % of portfolio allocated to trade |
| Approval Gate | Human confirmation required before execution |
| Paper Trading | Simulated trading without real money |

---

## Document Control

| Field | Value |
|-------|-------|
| Document | GST_MASTER_FRAMEWORK.md |
| Version | 2.4.2 |
| Author | Kyle + Claude |
| Created | January 14, 2026 |
| Last Updated | January 14, 2026 |
| Status | Active |
| Classification | Internal |

---

*End of Document*
