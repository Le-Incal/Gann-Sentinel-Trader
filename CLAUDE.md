# CLAUDE.md - Gann Sentinel Trader (GST)

> This file provides context for Claude Code. Read this first when working on GST.

## Quick Reference

| Item | Value |
|------|-------|
| Version | 3.0.0 |
| Status | Production (Paper Trading) |
| Deployment | Railway (auto-deploy from GitHub main) |
| URL | https://gann-sentinel-trader-production.up.railway.app |
| Logs API Token | `QzHBtENzt-sYeLXKSUzEN_v6VREwfEnGaqpoQVmOBWE` |

---

## 1. What Is GST?

A **committee-based, multi-agent trading decision system** with:
- Explicit role separation
- Thesis-first reasoning
- Visible AI debate (2 rounds)
- Majority voting with tie-breaking
- Technical validation as check-and-balance
- Explainable Telegram output
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

## 2. Architecture Overview (v3.0.0)

```
Scanners (data only)
       ↓
Analysts (independent theses)
       ↓
Debate Layer (cross-examination, 2 rounds)
       ↓
Synthesizer / Chair (tie-break + final thesis)
       ↓
Risk Engine (hard constraints)
       ↓
Telegram (explainability + debate transcript)
       ↓
Execution (paper or live)
       ↓
Learning / Logging
```

---

## 3. Model Roles (STRICT)

Each AI model has a **single epistemic role**. No model may operate outside its role.

### Scanners (No Opinions)
| Scanner | Source | Purpose |
|---------|--------|---------|
| FRED Scanner | Federal Reserve | Macroeconomic indicators |
| Polymarket Scanner | Prediction Markets | Market expectations (NO SPORTS) |
| Event Scanner | Grok (parsed) | 27 corporate event types |
| Technical Scanner | Alpaca | OHLCV + indicators ONLY |

Scanners **must not** propose trades or opinions.

### Analysts (Each produces a thesis)

| Model | Role | Purpose |
|-------|------|---------|
| **Grok** | Narrative Momentum Analyst | Detect emerging/accelerating narratives from X/Twitter |
| **Perplexity** | External Reality Analyst | Web facts, filings, earnings, macro news |
| **ChatGPT** | Sentiment & Cognitive Bias Analyst | Market psychology & bias detection |
| **Claude** | Technical Structure Validator | Chart validity ONLY (SUPPORTS/WEAKENS/INVALIDATES) |

Claude **must not**: browse, propose trades independently, or override synthesis.

### Synthesizer / Chair

- Implemented as **ChatGPT Chair** (separate role from ChatGPT Analyst)
- Does NOT generate new signals or browse
- Weighs analyst theses + debate outcomes
- Breaks **2-2 vote ties**
- Produces the **Final Investment Thesis**
- Defaults to HOLD when ambiguity remains

---

## 4. Debate Layer

### Purpose
Enable **visible, logged disagreement** between analysts before synthesis.

### Debate Rules
- Debate occurs **after analysts submit initial theses**
- Every analyst speaks **exactly twice** (2 rounds)
- No new data sources allowed during debate
- Analysts may defend OR revise their position
- Each turn must end with an explicit **vote**

### Debate Trigger
Debate is triggered if:
- Analysts disagree on action (LONG / SHORT / HOLD)
- OR confidence dispersion exceeds threshold
- OR Claude issues a technical INVALIDATION

### Voting & Consensus

| Vote Pattern | Outcome |
|--------------|---------|
| 4–0 | Execute |
| 3–1 | Execute |
| 2–2 | Chair breaks tie |
| 2–1–1 | HOLD |
| Majority HOLD | HOLD |

**Technical Check-and-Balance:** If Claude outputs INVALIDATES, require supermajority (3 of 4) to proceed.

---

## 5. File Structure

```
gann-sentinel-trader/
├── agent.py                 # Main orchestrator - START HERE
├── temporal.py              # Shared time framework (market hours, holidays)
├── config.py                # All configuration including debate flags
│
├── SCANNERS (Data Input)
│   ├── grok_scanner.py      # xAI Grok with live_search + debate()
│   ├── fred_scanner.py      # Federal Reserve macro data
│   ├── polymarket_scanner.py # Prediction markets (17 categories)
│   ├── technical_scanner.py # 5-year chart analysis via Alpaca
│   └── event_scanner.py     # 27 corporate event types
│
├── ANALYZERS (AI Council)
│   ├── perplexity_analyst.py   # External facts + debate()
│   ├── chatgpt_analyst.py      # Sentiment analysis + debate()
│   ├── chatgpt_chair.py        # NEW: Synthesizer/Chair (tie-breaker)
│   ├── claude_technical_validator.py # NEW: Technical validation only
│   └── claude_analyst.py       # Legacy (kept for backwards compatibility)
│
├── CORE
│   └── maca_orchestrator.py # Committee debate + synthesis
│
├── STORAGE
│   └── database.py          # SQLite with debate_sessions + debate_turns
│
├── EXECUTORS
│   ├── risk_engine.py       # Position sizing, limits, validation
│   └── alpaca_executor.py   # Alpaca trading API
│
├── NOTIFICATIONS
│   └── telegram_bot.py      # Bot interface + debate display
│
├── UTILITIES
│   ├── learning_engine.py   # Performance tracking + adaptation
│   ├── logs_api.py          # HTTP API for remote monitoring
│   └── exporter.py          # CSV/Parquet export
│
└── DOCS
    ├── CHANGELOG.md         # Version history and bug fixes
    ├── GST_MASTER_FRAMEWORK.md
    └── MACA_SPEC_v1.md
```

---

## 6. Configuration Flags

### Debate Layer (config.py)
```python
DEBATE_ENABLED = True         # Enable 2-round cross-examination
DEBATE_ROUNDS = 2             # Number of debate rounds
DEBATE_MIN_AVG_CONFIDENCE = 0.60  # Min avg confidence to proceed
TECH_INVALIDATION_SUPERMAJORITY = True  # Require 3/4 to override tech veto
```

### Environment Variables
```bash
# Required
XAI_API_KEY=           # Grok
ANTHROPIC_API_KEY=     # Claude
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# MACA (required for committee mode)
PERPLEXITY_API_KEY=    # Perplexity Sonar Pro
OPENAI_API_KEY=        # GPT-4o (used by ChatGPT Analyst AND Chair)

# Optional
DEBATE_ENABLED=ON      # Default: ON
DEBATE_ROUNDS=2        # Default: 2
LOG_LEVEL=INFO
```

---

## 7. Telegram Commands

| Command | Description |
|---------|-------------|
| `/scan` | Run full committee debate cycle |
| `/check [TICKER]` | Analyze specific stock with full debate |
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

## 8. Current State (v3.0.0)

### What's Working
- **Committee-based debate** with 4 AI analysts
- **2-round cross-examination** before synthesis
- **Majority voting** with Chair tie-breaker
- **Technical check-and-balance** (Claude as validator)
- Trade execution pipeline (scan → debate → approve → Alpaca order)
- Smart scheduling (2x daily: 9:35 AM, 12:30 PM ET)
- 27 event type detection
- Learning Engine tracking outcomes
- Debate transcript in Telegram

### Smart Schedule
```
Scheduled Scans: 9:35 AM ET, 12:30 PM ET (weekdays only)
Manual Commands: /scan and /check always available
```

### Known Considerations
- Paper trading only (Alpaca)
- Human approval required for all trades
- SQLite database (Railway persistent volume)

---

## 9. Safety Architecture

```
Signal Sources → FRED/Polymarket/Technical/Event (data only)
       ↓
AI Analysts → Grok + Perplexity + ChatGPT + Claude (theses)
       ↓
Debate Layer → 2 rounds, each speaks twice
       ↓
Vote Summary → Majority wins, Chair breaks ties
       ↓
Chair Synthesis → Final Investment Thesis
       ↓
Risk Engine → Position sizing, limits, validation
       ↓
Human Gate → Telegram approval required
       ↓
Execution → Alpaca paper trading
```

**Key Safety Principles:**
1. **Multiple AI sources prevent echo chambers** - 4 independent theses
2. **Visible disagreement is information** - Debate transcripts logged
3. **Roles are sacred** - No AI operates outside its role
4. **Technical veto power** - Claude can block weak consensus
5. **HOLD is a success outcome** - Capital preservation > opportunity
6. **Human approval required for ALL trades**

---

## 10. Design Principles (DO NOT VIOLATE)

1. HOLD is a success outcome
2. Disagreement is information
3. Roles are sacred
4. Explainability > frequency
5. Capital preservation > opportunity

**Optimize for:** clarity, discipline, auditability, trust

**DO NOT optimize for:** speed, trade frequency, cleverness

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.0.0 | Jan 2026 | Committee-based debate architecture |
| 2.4.3 | Jan 2026 | Trade execution pipeline fixes |
| 2.4.2 | Jan 2026 | Full MACA for scheduled scans |
| 2.4.0 | Jan 2026 | Learning Engine, Smart Scheduling |
| 2.3.0 | Jan 2026 | Event Scanner (27 types) |
| 2.0.0 | Jan 2026 | Forward-predictive system |
| 1.0.0 | Dec 2025 | Initial release |

---

## 12. Documentation Links

- **Changelog:** `CHANGELOG.md` - Version history and bug fixes
- **Master Framework:** `GST_MASTER_FRAMEWORK.md` - Complete system docs
- **MACA Spec:** `MACA_SPEC_v1.md` - Multi-agent architecture

---

## 13. Working Agreements

1. **Test-Driven Development** - Write tests first, implement second
2. **Lean Philosophy** - Complete current phase before adding features
3. **Safety First** - Human approval gate is non-negotiable
4. **Observability** - Log everything, make debugging easy
5. **Forward-Looking** - ANCHOR in history, ORIENT toward future

---

*Last Updated: January 17, 2026*
*Maintainer: Kyle + Claude*
