# Gann Sentinel Trader - Project Context

## Overview

Autonomous trading system combining multiple AI agents for algorithmic trading:
- **Grok (xAI)**: Real-time sentiment from X/Twitter and web
- **Claude (Anthropic)**: Strategic reasoning and trade recommendations
- **Risk Engine**: Final authority on trade validation
- **Telegram**: Human-in-the-loop approval gate

**Philosophy**: "Build It Lean" - strict phase constraints, safety-first protocols.

## Architecture

```
Scanners (signals) → Claude Analyst (decisions) → Risk Engine (validation) → Alpaca (execution)
     ↓                       ↓                          ↓                        ↓
   Grok               Conviction Score              Position Limits         Paper/Live
   FRED               Thesis + Entry                Stop-Loss               Trading
   Polymarket         Historical Patterns           Daily Loss Limit
   Technical          Business Model Audit          Sector Exposure
```

## Data Sources

| Source | Purpose | Key Config |
|--------|---------|------------|
| **Grok** | X/Twitter sentiment, web news | `XAI_API_KEY`, uses `live_search` tool |
| **FRED** | Macro data (yields, inflation, GDP) | `FRED_API_KEY` |
| **Polymarket** | Prediction market signals | No API key needed |
| **Alpaca** | Price data, trade execution | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| **Claude** | Strategic analysis | `ANTHROPIC_API_KEY`, model: `claude-3-5-sonnet-20241022` |

## Scanners

### Polymarket Scanner (`scanners/polymarket_scanner.py`)

**17 Investment-Focused Categories:**
- FEDERAL_RESERVE, MACRO_ECONOMIC, FISCAL_TREASURY
- TRADE_POLICY, GEOPOLITICAL, CHINA_RISK
- AI_SECTOR, SEMICONDUCTOR, TECH_GIANTS
- SPACE_INDUSTRY, ENERGY_COMMODITIES, HEALTHCARE_BIOTECH
- CEO_EXECUTIVE, REGULATORY_LEGAL, IPO_CAPITAL_MARKETS
- LABOR_IMMIGRATION, CRYPTO_POLICY

**Signal Purpose Classification:**
- `hypothesis_generator` - Novel info that could spark a trade idea
- `sentiment_validator` - Crowd odds to check against existing thesis
- `catalyst_timing` - Markets resolving within 30 days (entry timing)

**Exclusion Filters:**
- Sports (NFL, NBA, soccer/FC teams, betting language)
- Entertainment (awards, reality TV, celebrities)
- Weather predictions
- Pattern matching for European football clubs

### Technical Scanner (`scanners/technical_scanner.py`)

Market state classification → Scenario-based reasoning → R-multiple calculation

**Verdicts:** `no_trade`, `analyze_only`, `hypothesis_allowed`, `escalate`

### FRED Scanner (`scanners/fred_scanner.py`)

Forward-contextualized macro data. Key series: DGS10, DGS2, T10Y2Y, FEDFUNDS, UNRATE, CPIAUCSL

### Grok Scanner (`scanners/grok_scanner.py`)

Deep business intelligence via xAI. Supports platform company analysis (TSLA, AMZN, GOOGL, etc.)

## Claude Analyst (`analyzers/claude_analyst.py`)

**Methodology:** AUDIT → ANCHOR → ORIENT

1. **Business Model Audit** - Map ALL business lines before pattern matching
2. **Historical Pattern Recognition** - Platform transformation patterns (AWS, NVIDIA, MSFT)
3. **Forward-Predictive Reasoning** - Position AHEAD of catalysts

**Conviction Scoring:**
- 80-100: HIGH - Trade allowed
- 60-79: MEDIUM - Watch
- 0-59: LOW - Pass

## Risk Engine (`risk_engine.py`)

Final authority. Validates:
- Position size limits
- Sector concentration
- Daily loss limits
- Stop-loss requirements

## Commands

| Command | Description |
|---------|-------------|
| `/status` | System status |
| `/pending` | Pending trades |
| `/check [TICKER]` | Analyze any stock on-demand |
| `/catalyst` | Hypothetical scenario evaluation |
| `/approve [id]` | Approve pending trade |
| `/reject [id]` | Reject pending trade |

## Deployment

- **Platform**: Railway (auto-deploys from GitHub main branch)
- **Mode**: Paper trading via Alpaca
- **Cycle**: Hourly scans during market hours
- **Notifications**: Telegram

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "No response from Claude" | Invalid model name or API error | Check model name, API key, error logs |
| Polymarket showing sports | Missing exclusion patterns | Pattern-based filtering for FC/AFC teams |
| FRED 0 signals | Weekend/no new data OR missing API key | Normal on weekends; check `FRED_API_KEY` |
| Technical 0 signals | Markets closed | Normal on weekends |
| Grok API 500 | xAI server issue | Usually self-resolves; retry logic in place |

## Development Workflow

1. **Research** - Read files, understand patterns
2. **Plan** - Present options with tradeoffs
3. **Execute** - TDD, commit incrementally

Tests before implementation. Commits are atomic. Push to main auto-deploys.

## Key Files

```
agent.py                    # Main orchestration
analyzers/claude_analyst.py # Strategic reasoning
scanners/
  grok_scanner.py          # X/Twitter + web sentiment
  fred_scanner.py          # Macro economic data
  polymarket_scanner.py    # Prediction markets
  technical_scanner.py     # Chart analysis
risk_engine.py             # Trade validation
notifications/telegram_bot.py # Alerts + approval
database.py                # SQLite storage
```

## Version

Current: v2.0.2 (Polymarket focus + Claude/config alignment)
