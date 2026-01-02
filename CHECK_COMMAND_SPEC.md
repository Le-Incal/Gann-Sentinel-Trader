# `/check` Command Implementation Spec

## Overview

Add an on-demand stock analysis command that runs full Grok + Claude analysis on any ticker (including pre-IPO) and generates a trade recommendation with approval prompt.

## Command Syntax

```
/check TSLA          # Analyze Tesla
/check SPACEX        # Analyze pre-IPO SpaceX
/check RKLB          # Analyze Rocket Lab
```

## User Flow

1. User sends `/check NVDA`
2. Bot acknowledges: "Analyzing NVDA..."
3. System runs:
   - Grok sentiment scan for ticker
   - Grok catalyst scan for ticker
   - Claude full analysis with historical context
4. Bot returns comprehensive analysis with:
   - Current sentiment summary
   - Historical pattern matches
   - Forward catalysts
   - Conviction score with visual bar
   - Trade recommendation
5. If conviction >= 80 AND ticker is tradeable:
   - Creates pending trade
   - Shows `/approve [id]` prompt
6. If pre-IPO or non-tradeable:
   - Shows "WATCH" recommendation
   - No trade created

---

## File Changes Required

### 1. `agent.py` - Add command handler

```python
# In _process_commands(), add new elif:
elif command == "check":
    await self._handle_check_command(cmd.get("ticker"))

# New method:
async def _handle_check_command(self, ticker: str) -> None:
    """Handle /check [ticker] command - on-demand analysis."""
    if not ticker:
        await self.telegram.send_message(
            "Usage: /check [TICKER]\nExample: /check NVDA",
            parse_mode=None
        )
        return
    
    ticker = ticker.upper().strip()
    
    # Acknowledge
    await self.telegram.send_message(
        f"{EMOJI_SEARCH} Analyzing {ticker}...",
        parse_mode=None
    )
    
    try:
        # Run on-demand analysis
        result = await self._run_ticker_analysis(ticker)
        
        # Send analysis result
        await self.telegram.send_check_result(result)
        
    except Exception as e:
        logger.error(f"Error in check command: {e}")
        await self.telegram.send_message(
            f"{EMOJI_CROSS} Analysis failed: {str(e)[:100]}",
            parse_mode=None
        )


async def _run_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
    """Run full analysis on a single ticker."""
    signals = []
    
    # Get Grok sentiment for this ticker
    try:
        sentiment_signals = await self.grok.scan_sentiment([ticker])
        signals.extend(sentiment_signals)
    except Exception as e:
        logger.error(f"Grok sentiment error for {ticker}: {e}")
    
    # Get Grok catalysts for this ticker
    try:
        catalyst_signals = await self.grok.scan_catalysts(ticker)
        signals.extend(catalyst_signals)
    except Exception as e:
        logger.error(f"Grok catalyst error for {ticker}: {e}")
    
    # Check if ticker is tradeable (has price data)
    is_tradeable = False
    current_price = None
    price_error = None
    
    try:
        quote = await self.executor.get_quote(ticker)
        if "error" not in quote and quote.get("mid"):
            is_tradeable = True
            current_price = quote.get("mid")
    except Exception as e:
        price_error = str(e)
        logger.info(f"{ticker} not tradeable: {e}")
    
    # Get portfolio context
    portfolio = await self.executor.get_portfolio_snapshot()
    positions = await self.executor.get_positions()
    portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
    
    # Run Claude analysis
    analysis = None
    analysis_dict = None
    
    if signals:
        signals_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals]
        
        try:
            analysis = await self.analyst.analyze_signals(
                signals=signals,
                portfolio_context=portfolio_dict,
                watchlist=[ticker]  # Focus on this ticker
            )
            analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
    
    # Build result
    result = {
        "ticker": ticker,
        "is_tradeable": is_tradeable,
        "current_price": current_price,
        "price_error": price_error,
        "signals_count": len(signals),
        "signals": [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals],
        "analysis": analysis_dict,
        "conviction": analysis.conviction_score if analysis else 0,
        "recommendation": analysis.recommendation.value if analysis else "NONE",
        "thesis": analysis.thesis if analysis else "Insufficient data for analysis",
        "historical_context": analysis.historical_context if analysis and hasattr(analysis, 'historical_context') else None,
        "pending_trade_id": None,
    }
    
    # Create trade if actionable
    if (analysis and 
        analysis.is_actionable and 
        is_tradeable and 
        analysis.conviction_score >= 80):
        
        # Run risk checks
        passed, risk_results = self.risk_engine.validate_trade(
            analysis=analysis,
            portfolio=portfolio,
            current_positions=positions
        )
        
        if passed:
            # Calculate position size
            sizing = self.risk_engine.calculate_position_size(
                analysis=analysis,
                portfolio=portfolio,
                current_price=current_price
            )
            
            if sizing["shares"] >= 1:
                # Create trade
                trade = Trade(
                    analysis_id=analysis.analysis_id,
                    ticker=ticker,
                    side=OrderSide.BUY if analysis.recommendation == Recommendation.BUY else OrderSide.SELL,
                    quantity=sizing["shares"],
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    status=TradeStatus.PENDING_APPROVAL,
                    thesis=analysis.thesis,
                    conviction_score=analysis.conviction_score
                )
                
                self.db.save_trade(trade.to_dict())
                result["pending_trade_id"] = trade.trade_id[:8]
        else:
            failed_checks = [r for r in risk_results if not r.passed]
            result["risk_rejection"] = "; ".join([r.message for r in failed_checks])
    
    return result
```

### 2. `telegram_bot.py` - Add command parsing and result formatting

```python
# In process_commands(), add parsing for "check":
elif cmd_text == "check" and args:
    cmd_dict["ticker"] = args[0].upper()

# New method for formatting check results:
async def send_check_result(self, result: Dict[str, Any]) -> bool:
    """Send on-demand analysis result."""
    ticker = result.get("ticker", "???")
    conviction = result.get("conviction", 0)
    is_tradeable = result.get("is_tradeable", False)
    current_price = result.get("current_price")
    recommendation = result.get("recommendation", "NONE")
    thesis = result.get("thesis", "No analysis available")
    pending_trade_id = result.get("pending_trade_id")
    risk_rejection = result.get("risk_rejection")
    historical_context = result.get("historical_context")
    
    # Build conviction bar
    bar = self._build_conviction_bar(conviction)
    
    # Build status line
    if is_tradeable and current_price:
        status = f"Tradeable @ ${current_price:.2f}"
    elif is_tradeable:
        status = "Tradeable (price unavailable)"
    else:
        status = f"Not Tradeable (pre-IPO or unlisted)"
    
    # Build recommendation emoji
    if recommendation == "BUY":
        rec_emoji = EMOJI_BULL
        rec_text = "BUY"
    elif recommendation == "SELL":
        rec_emoji = EMOJI_BEAR
        rec_text = "SELL"
    else:
        rec_emoji = EMOJI_WHITE_CIRCLE
        rec_text = "HOLD/WATCH"
    
    lines = [
        f"{EMOJI_TARGET} ANALYSIS: {ticker}",
        "=" * 30,
        "",
        f"Status: {status}",
        f"Signals: {result.get('signals_count', 0)} gathered",
        "",
        f"{EMOJI_CHART} CONVICTION: {conviction}/100",
        bar,
        "",
        f"{rec_emoji} RECOMMENDATION: {rec_text}",
        "",
        f"{EMOJI_BRAIN} THESIS:",
        thesis[:400],
    ]
    
    # Add historical context if available
    if historical_context:
        lines.extend([
            "",
            f"{EMOJI_SEARCH} HISTORICAL PATTERN:",
            historical_context[:200],
        ])
    
    # Add trade action or status
    lines.append("")
    lines.append("=" * 30)
    
    if pending_trade_id:
        lines.extend([
            f"{EMOJI_BELL} Trade Created - Pending Approval",
            "",
            f"/approve {pending_trade_id}",
            f"/reject {pending_trade_id}",
        ])
    elif risk_rejection:
        lines.extend([
            f"{EMOJI_WARNING} Risk Check Failed:",
            risk_rejection[:100],
        ])
    elif not is_tradeable:
        lines.extend([
            f"{EMOJI_SEARCH} WATCH LIST",
            "Cannot trade (pre-IPO or unlisted)",
            "Monitor for when it becomes available",
        ])
    elif conviction < 80:
        lines.extend([
            f"{EMOJI_ZZZ} No Trade",
            f"Conviction {conviction} below 80 threshold",
        ])
    else:
        lines.extend([
            f"{EMOJI_ZZZ} No Trade",
            "Analysis did not meet criteria",
        ])
    
    message = "\n".join(lines)
    return await self.send_message(message, parse_mode=None)
```

### 3. `agent.py` - Update help command

```python
async def _handle_help_command(self) -> None:
    """Handle /help command."""
    help_text = """Gann Sentinel Commands:

/check [TICKER] - Analyze any stock
/status - System status
/pending - Pending trades
/approve [id] - Approve trade
/reject [id] - Reject trade
/digest - Daily digest
/stop - Emergency halt
/resume - Resume trading
/help - This message

Examples:
  /check NVDA
  /check TSLA
  /check RKLB

Scans run every ~60 minutes
Digest at 4 PM ET daily"""
    await self.telegram.send_message(help_text, parse_mode=None)
```

---

## Response Format Example

### Tradeable Stock with High Conviction

```
🎯 ANALYSIS: NVDA
==============================

Status: Tradeable @ $142.50
Signals: 5 gathered

📊 CONVICTION: 85/100
[████████░░]

🐂 RECOMMENDATION: BUY

🧠 THESIS:
NVDA showing strong momentum with AI demand acceleration. 
Blackwell GPU rollout ahead of schedule. Datacenter revenue 
guidance raised 15%. Technical breakout above $140 resistance 
with volume confirmation.

🔍 HISTORICAL PATTERN:
Similar to AMD breakout Q4 2020 when datacenter narrative 
shifted. That move saw +180% over 12 months.

==============================
🔔 Trade Created - Pending Approval

/approve a1b2c3d4
/reject a1b2c3d4
```

### Pre-IPO Stock

```
🎯 ANALYSIS: SPACEX
==============================

Status: Not Tradeable (pre-IPO or unlisted)
Signals: 3 gathered

📊 CONVICTION: 72/100
[███████░░░]

⚪ RECOMMENDATION: HOLD/WATCH

🧠 THESIS:
SpaceX valuation reportedly at $350B in latest secondary 
round. Starlink subscriber growth exceeding expectations. 
Starship orbital success would be major catalyst. Consider 
public proxies: RKLB (space competitor), GOOGL (Starlink 
investor).

🔍 HISTORICAL PATTERN:
Similar pre-IPO dynamics to Rivian 2021. Space sector 
proxies rallied 40%+ in 6 months before RIVN IPO.

==============================
🔍 WATCH LIST
Cannot trade (pre-IPO or unlisted)
Monitor for when it becomes available
```

### Low Conviction

```
🎯 ANALYSIS: F
==============================

Status: Tradeable @ $10.25
Signals: 4 gathered

📊 CONVICTION: 55/100
[█████░░░░░]

⚪ RECOMMENDATION: HOLD/WATCH

🧠 THESIS:
Ford showing mixed signals. EV transition costs weighing 
on margins but F-150 Lightning demand stable. Union 
contract settled but wage increases will pressure 2025 
earnings. No clear catalyst in near term.

==============================
💤 No Trade
Conviction 55 below 80 threshold
```

---

## Testing Checklist

After deployment:

```
/help                    # Should show /check command
/check NVDA              # Tradeable, should get full analysis
/check TSLA              # Tradeable, volatile ticker
/check SPACEX            # Pre-IPO, should show WATCH
/check STRIPE            # Pre-IPO, should show WATCH
/check RKLB              # In watchlist, should work
/check INVALIDTICKER     # Should handle gracefully
/check                   # Should show usage
```

---

## Implementation Order

1. **telegram_bot.py**: Add command parsing for "check"
2. **telegram_bot.py**: Add `send_check_result()` method
3. **agent.py**: Add `_handle_check_command()` method
4. **agent.py**: Add `_run_ticker_analysis()` method
5. **agent.py**: Update help text
6. Test locally
7. Deploy to Railway

---

## Notes

- Pre-IPO tickers won't have price data from Alpaca, but Grok can still find sentiment and news
- The system will gracefully handle unknown tickers
- Risk engine still validates all trades before creating pending approval
- User must still manually approve any trade recommendations
