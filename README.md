# Gann Sentinel Trader

An autonomous trading agent that combines Grok's real-time sentiment analysis with Claude's strategic reasoning to identify high-conviction position trades.

## ⚠️ DISCLAIMER

**Trading involves substantial risk of loss.** This is an experimental system and nothing here constitutes financial advice. Only trade what you can afford to lose.

## Features

- **Hybrid AI Brain**: Grok scans X/Twitter sentiment and news, Claude reasons about trades
- **Multi-Source Signals**: FRED macro data, Polymarket predictions, social sentiment
- **Risk Management**: Position limits, stop-losses, daily loss limits, conviction thresholds
- **Approval Gate**: Human-in-the-loop for trade approval via Telegram
- **Full Audit Trail**: Every signal, analysis, and trade logged to SQLite
- **Paper Trading**: Test with Alpaca paper trading before going live

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GANN SENTINEL TRADER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   GROK   │  │   FRED   │  │POLYMARKET│  │  ALPACA  │       │
│  │ X Search │  │ Treasury │  │ Fed odds │  │  Prices  │       │
│  │ Web News │  │ GDP/CPI  │  │ Election │  │ Positions│       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                           │                                    │
│                    ┌──────▼──────┐                             │
│                    │   SIGNAL    │                             │
│                    │  PROCESSOR  │                             │
│                    └──────┬──────┘                             │
│                           │                                    │
│                    ┌──────▼──────┐                             │
│                    │   CLAUDE    │                             │
│                    │   ANALYST   │                             │
│                    └──────┬──────┘                             │
│                           │                                    │
│                    ┌──────▼──────┐                             │
│                    │    RISK     │                             │
│                    │   ENGINE    │                             │
│                    └──────┬──────┘                             │
│                           │                                    │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│     ┌────────────────┐       ┌────────────────┐               │
│     │ APPROVAL GATE  │       │ AUTO-EXECUTE   │               │
│     │   (Telegram)   │       │   (Optional)   │               │
│     └───────┬────────┘       └───────┬────────┘               │
│             │                         │                        │
│             └────────────┬────────────┘                        │
│                          ▼                                     │
│                  ┌───────────────┐                             │
│                  │    ALPACA     │                             │
│                  │   EXECUTOR    │                             │
│                  └───────────────┘                             │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/gann-sentinel-trader.git
cd gann-sentinel-trader
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run the Agent

```bash
python agent.py
```

## Configuration

### Required API Keys

| Service | Get Key From | Required |
|---------|--------------|----------|
| Telegram | [@BotFather](https://t.me/botfather) | Yes |
| xAI (Grok) | [console.x.ai](https://console.x.ai) | Yes |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Yes |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Yes |
| FRED | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Yes |

### Environment Variables

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# xAI (Grok)
XAI_API_KEY=xai-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Alpaca
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading first!

# FRED
FRED_API_KEY=your_fred_key

# System
MODE=PAPER              # PAPER or LIVE
APPROVAL_GATE=ON        # ON or OFF
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
```

### Risk Parameters

```bash
MAX_POSITION_PCT=0.25      # Max 25% of portfolio per position
MAX_POSITIONS=5            # Max 5 concurrent positions
MIN_CONVICTION=80          # Only trade with 80+ conviction
STOP_LOSS_PCT=0.15         # 15% stop loss
DAILY_LOSS_LIMIT_PCT=0.05  # 5% daily loss halts trading
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | System and portfolio status |
| `/pending` | List pending trade approvals |
| `/approve [id]` | Approve a pending trade |
| `/reject [id]` | Reject a pending trade |
| `/stop` | Emergency halt (cancel orders, stop trading) |
| `/resume` | Resume trading after halt |
| `/help` | Show help message |

## Project Structure

```
gann-sentinel-trader/
├── agent.py              # Main orchestration loop
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
│
├── models/               # Data structures
│   ├── signals.py        # Signal models
│   ├── analysis.py       # Analysis models
│   └── trades.py         # Trade and position models
│
├── scanners/             # Data ingestion
│   ├── grok_scanner.py   # X/Twitter sentiment via Grok
│   ├── fred_scanner.py   # FRED macro data
│   └── polymarket_scanner.py  # Prediction markets
│
├── analyzers/            # Decision making
│   └── claude_analyst.py # Claude reasoning engine
│
├── executors/            # Trade execution
│   ├── risk_engine.py    # Risk validation
│   └── alpaca_executor.py # Alpaca trading
│
├── notifications/        # Alerts
│   └── telegram_bot.py   # Telegram integration
│
├── storage/              # Persistence
│   └── database.py       # SQLite operations
│
├── data/                 # Generated at runtime
│   └── sentinel.db       # SQLite database
│
└── logs/                 # Generated at runtime
    └── agent.log         # Application logs
```

## How It Works

### 1. Signal Gathering (Every Hour)

- **Grok** scans X/Twitter for sentiment on watchlist stocks
- **Grok** searches web for relevant news
- **FRED** pulls latest macro data (Treasury yields, unemployment, etc.)
- **Polymarket** fetches prediction market probabilities

### 2. Analysis (After Each Scan)

Claude receives all signals and:
- Synthesizes them into a coherent market view
- Identifies potential trading opportunities
- Scores conviction (0-100) for any recommendation
- Only recommends trades with conviction >= 80

### 3. Risk Check

Before any trade, the Risk Engine validates:
- Position size limits
- Maximum positions
- Daily loss limits
- Stop-loss requirements
- Conviction threshold

### 4. Approval (If Gate Enabled)

Trade recommendation sent to Telegram for human approval:
- Review thesis and conviction
- Approve or reject with a simple command
- Trades expire if not approved

### 5. Execution

Approved trades are submitted to Alpaca:
- Market or limit orders
- Stop-loss orders placed automatically
- Fills tracked and logged

### 6. Monitoring

Continuous position monitoring:
- Stop-loss trigger checks
- Thesis breaker detection (via periodic re-analysis)
- Daily P&L tracking

## Trading Philosophy

1. **Sentiment precedes price** - Crowd psychology often signals moves before price action
2. **Second-order effects** - Best trades are adjacent plays (SpaceX IPO → Rocket Lab benefits)
3. **High conviction, low frequency** - Only trade with 80%+ conviction
4. **Thesis-driven** - Every trade has a clear, falsifiable thesis

## Database Schema

### Tables

- `signals` - All raw signals from all sources
- `analyses` - Claude's reasoning for each analysis
- `trades` - Full trade lifecycle
- `positions` - Current holdings
- `portfolio_snapshots` - Daily portfolio state
- `errors` - All failures for debugging

## Deployment (Railway)

1. Connect GitHub repo to Railway
2. Add environment variables in Railway dashboard
3. Railway auto-deploys on push

### Procfile (if needed)

```
web: python agent.py
```

## Development

### Run Tests

```bash
pytest tests/
```

### Format Code

```bash
black .
flake8 .
```

## Phase Roadmap

### Phase 1: Build It Lean (Current)
- Core hybrid system
- Paper trading
- Basic logging
- Human approval gate

### Phase 2: Add Complexity (If Needed)
- Pattern library from observed successes/failures
- Economic regime detection
- Targeted paid APIs if gaps identified

### Phase 3: Sophistication (If Working)
- Full pattern library with backreferences
- Multi-strategy framework
- Portfolio-level optimization

## Security Notes

- **Never commit `.env` to Git**
- Regenerate API keys if accidentally exposed
- Start with paper trading before live
- Keep approval gate ON until comfortable

## License

MIT

## Contributing

This is an experimental personal project. Feel free to fork and adapt for your own use.

---

**Remember: Trading is risky. This system can and will lose money. Only trade what you can afford to lose.**
