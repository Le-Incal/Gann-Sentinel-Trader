# Phase 2 Deployment Guide - MACA Integration

## Overview

This deployment adds:
1. **Telegram Activity Logging** - Full observability of all bot messages
2. **Multi-Agent Consensus Architecture (MACA)** - Grok + Perplexity + ChatGPT with Claude synthesis
3. **New Telegram Commands** - `/logs` and `/export_logs`
4. **Logs API** - HTTP endpoint for remote log access (Claude can read directly)

---

## New Environment Variables

Add these to Railway:

```
# Perplexity API
PERPLEXITY_API_KEY=pplx-xxx
PERPLEXITY_MODEL=sonar-pro

# OpenAI (ChatGPT)
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o

# Logs API (for Claude remote access)
LOGS_API_TOKEN=<generate-a-secure-token>
LOGS_API_PORT=8080

# Existing (already set)
XAI_API_KEY=xxx
ANTHROPIC_API_KEY=xxx
```

### Generate a Secure Token

Use this command to generate a secure token:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Example output: `Kx8mP2vL9nQ4rT7wY1zA3bC6dE8fG0hJ`

---

## Files to Upload

### Replace Existing Files:

| File | Location | Changes |
|------|----------|---------|
| `database.py` | `/storage/database.py` | Added telegram_messages, ai_proposals, ai_reviews, scan_cycles tables |
| `telegram_bot.py` | `/notifications/telegram_bot.py` | Added activity logging + /logs command |

### Add New Files:

| File | Location | Purpose |
|------|----------|---------|
| `perplexity_analyst.py` | `/analyzers/perplexity_analyst.py` | Perplexity Sonar Pro integration |
| `chatgpt_analyst.py` | `/analyzers/chatgpt_analyst.py` | GPT-4o integration |
| `claude_maca_extension.py` | `/analyzers/claude_maca_extension.py` | Synthesis capability for Claude |
| `maca_orchestrator.py` | `/core/maca_orchestrator.py` | Orchestrates 4-phase MACA cycle |
| `logs_api.py` | `/api/logs_api.py` | HTTP endpoint for remote log access |
| `main.py` | `/main.py` | Entry point with API server |
| `MACA_SPEC_v1.md` | `/docs/MACA_SPEC_v1.md` | Architecture documentation |

---

## Directory Structure After Update

```
gann-sentinel-trader/
├── analyzers/
│   ├── claude_analyst.py          (existing)
│   ├── claude_maca_extension.py   (NEW)
│   ├── perplexity_analyst.py      (NEW)
│   └── chatgpt_analyst.py         (NEW)
├── api/
│   └── logs_api.py                (NEW)
├── core/
│   └── maca_orchestrator.py       (NEW)
├── notifications/
│   └── telegram_bot.py            (UPDATED)
├── storage/
│   └── database.py                (UPDATED)
├── docs/
│   └── MACA_SPEC_v1.md            (NEW)
├── main.py                        (NEW - entry point with API)
└── ...
```

---

## Requirements Update

Add to `requirements.txt`:
```
fastapi>=0.109.0
uvicorn>=0.27.0
```

---

## Integration Steps

### Step 1: Upload Files

Upload all files to GitHub via web interface (folder by folder as preferred).

### Step 2: Add Environment Variables

In Railway dashboard → Variables, add the new API keys.

### Step 3: Test Telegram Logging

After deployment, send `/logs` to the bot. You should see activity history.

### Step 4: Enable MACA (Optional)

MACA is not auto-enabled. To activate:

1. Add to `config.py`:
```python
MACA_ENABLED = os.getenv("MACA_ENABLED", "false").lower() == "true"
```

2. Add to Railway environment:
```
MACA_ENABLED=true
```

3. Update `agent.py` to use MACA orchestrator (next phase)

---

## New Telegram Commands

| Command | Description |
|---------|-------------|
| `/logs` | Show last 20 activity log entries |
| `/logs 50` | Show last 50 entries |
| `/export_logs` | Export detailed log entries |
| `/export_logs 100` | Export last 100 entries |

---

## Logs API Setup

The Logs API allows Claude (and other tools) to access GST data remotely.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check (no auth) |
| `GET /api/logs?token=xxx&limit=50` | Recent Telegram activity |
| `GET /api/status?token=xxx` | System status overview |
| `GET /api/errors?token=xxx&limit=20` | Recent errors |
| `GET /api/signals?token=xxx&limit=50` | Recent signals |
| `GET /api/scan_cycles?token=xxx&limit=10` | MACA scan history |

### Railway Configuration

1. **Set the token** in Railway environment:
   ```
   LOGS_API_TOKEN=your-secure-token-here
   ```

2. **Expose the port** in Railway settings:
   - Go to Settings → Networking
   - Add port 8080 (or set `LOGS_API_PORT`)
   - Railway will give you a public URL like: `https://gst-production-xxxx.up.railway.app`

3. **Test the endpoint**:
   ```
   curl "https://your-railway-url/health"
   # Should return: {"status": "healthy", ...}

   curl "https://your-railway-url/api/logs?token=your-token&limit=10"
   # Should return recent logs
   ```

### Integration Options

**Option 1: Modify existing entry point**

Add to the end of your current main/agent startup:
```python
# In agent.py or main.py
import threading
from api.logs_api import app as logs_api_app
import uvicorn

def run_api():
    uvicorn.run(logs_api_app, host="0.0.0.0", port=8080, log_level="warning")

# Start API in background
api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
```

**Option 2: Use the new entry point**

Replace your start command with:
```
python main.py
```

---

## Database Schema Changes

New tables created automatically on startup:

```sql
-- telegram_messages: All bot activity
-- ai_proposals: Multi-agent thesis proposals
-- ai_reviews: Peer review records
-- scan_cycles: MACA scan tracking
```

---

## Version Update

Update version in relevant files:
- `telegram_bot.py`: 2.1.0
- `database.py`: 2.1.0

---

## Rollback Plan

If issues occur:
1. Set `MACA_ENABLED=false` in Railway
2. System falls back to Phase 1 Grok-only mode
3. All new tables are additive (won't break existing data)

---

## Next Steps After Deployment

1. **Test Telegram logging** - `/logs` should work immediately
2. **Verify API keys** - Check Railway logs for connection success
3. **Gradual MACA rollout** - Enable one AI at a time
4. **Monitor costs** - Track API usage in respective dashboards
