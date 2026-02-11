# Changelog

All notable changes to Gann Sentinel Trader are documented in this file.

---

## [Unreleased]

### A) Event scan
- Event scanner was already wired; it runs when `XAI_API_KEY` is set and logs "Event Scanner not configured" when skipped. No code change; confirmed behavior.

### B) Technical scanner – multiple charts
- **agent.py:** Pass full `technical_signals` list to MACA instead of only `technical_signals[0]`.
- **maca_orchestrator.py:** `run_scan_cycle()` now accepts `technical_analysis` as either a single dict or a list of chart dicts. All charts are normalized to a list, the first is used as primary for phase1/phase2, and the full list is added to `signal_inventory.technical_charts` and passed to Telegram so up to 3 charts are shown.

### C) Conviction scoring
- Conviction is defined as strength of the **trade recommendation** (0 = no trade, 1–100 = strength of BUY/SELL). Perplexity and ChatGPT prompts now state this explicitly and require either a concrete trade (ticker, side, conviction_score 1–100) or explicit HOLD (ticker/side null, conviction_score 0). Chair schema now includes `conviction_score` (0–100) in `final_thesis`; orchestrator uses it when present, else derives from `confidence`.

### D) Grok trade recommendation
- **grok_scanner.py:** Market outlook prompt and JSON schema now include optional `recommended_ticker`, `recommended_side` (BUY/SELL), and `recommendation_conviction` (1–100). When present, `_parse_outlook_to_signals()` creates a second GrokSignal with that ticker/side/conviction so the MACA adapter can pick it. Adapter uses `raw_value.unit == "recommendation_conviction"` to set `conviction_score` from the explicit value. `directional_bias` mapping extended to `positive`/`negative` for BUY/SELL.

### E–F) Perplexity and ChatGPT trade recommendation
- **perplexity_analyst.py** and **chatgpt_analyst.py:** Added a strict "RECOMMENDATION RULE" to prompts: output exactly one of (A) concrete trade with ticker, side, conviction_score 1–100, or (B) HOLD with ticker/side null, conviction_score 0 and thesis explaining why.

### G) Claude synthesis
- **claude_chair.py:** Schema now requires `synthesis_summary` (one sentence: "BUY/SELL TICKER at N conviction" or "No trade: &lt;reason&gt;") and `final_thesis.conviction_score` (0–100). Orchestrator passes `synthesis_summary` into synthesis; **telegram_bot.py** displays it at the top of the "CLAUDE'S SYNTHESIS" section.

---

## [2.4.3] - 2026-01-14

### Fixed - Trade Execution Pipeline

This release fixes multiple issues that prevented trades from being created and executed after MACA scans showed actionable signals with conviction ≥80.

#### 1. Trade Constructor Field Names
**File:** `agent.py` (lines 963-974)

**Problem:** Trade creation failed with `Trade.__init__() got an unexpected keyword argument 'id'`

**Root Cause:** The Trade model uses `trade_id` and `stop_price` as field names, but the code was passing `id` and `stop_loss_price`.

**Fix:**
```python
# Before (broken)
trade = Trade(
    id=str(uuid.uuid4()),
    stop_loss_price=current_price * (1 - stop_loss_pct / 100)
)

# After (fixed)
trade = Trade(
    trade_id=str(uuid.uuid4()),
    stop_price=current_price * (1 - stop_loss_pct / 100)
)
```

---

#### 2. Executor Method Mismatch
**File:** `agent.py` (line 1131)

**Problem:** Trade approval failed with `'AlpacaExecutor' object has no attribute 'execute_order'`

**Root Cause:** The approval handler called `execute_order()` but AlpacaExecutor has `submit_order()` which takes a Trade object, not individual parameters.

**Fix:**
```python
# Before (broken)
result = await self.executor.execute_order(
    ticker=trade.get("ticker"),
    side=trade.get("side"),
    quantity=trade.get("quantity"),
    order_type=trade.get("order_type", "market")
)

# After (fixed)
trade = Trade(
    trade_id=trade_id,
    ticker=trade_dict.get("ticker"),
    side=side,
    quantity=trade_dict.get("quantity"),
    # ... other fields
)
result_trade = await self.executor.submit_order(trade)
```

---

#### 3. Database Update Signature
**File:** `agent.py` (line 1160)

**Problem:** `Database.update_trade_status() takes 3 positional arguments but 4 were given`

**Root Cause:** The method signature is `update_trade_status(trade_id, status, **kwargs)` so additional fields must be passed as keyword arguments.

**Fix:**
```python
# Before (broken)
self.db.update_trade_status(trade_id, status, alpaca_order_id)

# After (fixed)
self.db.update_trade_status(trade_id, status, alpaca_order_id=alpaca_order_id)
```

---

#### 4. UUID Type Conversion
**File:** `agent.py` (line 1160)

**Problem:** `Error binding parameter 3: type 'UUID' is not supported`

**Root Cause:** Alpaca returns the order ID as a UUID object, but SQLite requires strings.

**Fix:**
```python
# Before (broken)
alpaca_order_id=result_trade.alpaca_order_id

# After (fixed)
order_id_str = str(result_trade.alpaca_order_id) if result_trade.alpaca_order_id else None
alpaca_order_id=order_id_str
```

---

#### 5. Order Side Case Sensitivity
**File:** `agent.py` (lines 1132-1133)

**Problem:** All trades were submitted as SELL regardless of the intended side.

**Root Cause:** The OrderSide enum stores lowercase values (`"buy"`, `"sell"`) but the approval handler compared against uppercase (`"BUY"`), causing the condition to always fail and default to SELL.

**Fix:**
```python
# Before (broken) - always defaulted to SELL
side_str = trade_dict.get("side", "BUY")
side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

# After (fixed) - case-insensitive comparison
side_str = trade_dict.get("side", "buy").lower()
side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
```

---

### Result

After these fixes, the complete trade pipeline now works:

1. MACA scan generates thesis with conviction ≥80
2. Trade record created with `PENDING_APPROVAL` status
3. Telegram notification shows approve/reject buttons
4. User clicks Approve
5. Trade object reconstructed from database
6. Order submitted to Alpaca with correct side
7. Database updated with Alpaca order ID
8. Confirmation sent via Telegram

**First successful trade:** BUY 187 shares OMC @ market (Order ID: 26493816-49bb-47e1-ab41-ddf0d3c66a1e)

---

## [2.4.2] - 2026-01-14

### Fixed
- Full MACA for scheduled scans
- `analysis.id` reference fix
- Trade blocker visibility in Telegram notifications

### Added
- Debug entry markers for trade creation troubleshooting
- Trade blocker recording for all early return paths

---

## [2.4.1] - 2026-01-13

### Added
- Trade blocker visibility in Telegram messages

---

## [2.4.0] - 2026-01-12

### Added
- Learning Engine for performance tracking
- Smart Scheduling (2x daily: 9:35 AM, 12:30 PM ET)

---

## [2.3.0] - 2026-01-10

### Added
- Event Scanner with 27 corporate event types

---

## [2.2.0] - 2026-01-08

### Added
- MACA integration for `/check` command

---

## [2.0.0] - 2026-01-05

### Added
- Forward-predictive system
- MACA (Multi-Agent Consensus Architecture)
- Second-order thinking methodology

---

## [1.0.0] - 2025-12-15

### Added
- Initial release
- Grok scanner integration
- Alpaca paper trading
- Telegram bot interface
- Basic risk engine

---

*Maintained by Kyle + Claude*
