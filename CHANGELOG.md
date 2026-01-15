# Changelog

All notable changes to Gann Sentinel Trader are documented in this file.

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
