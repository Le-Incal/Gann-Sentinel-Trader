"""
Telegram Bot for Gann Sentinel Trader
Handles notifications and command processing for trade approvals and system control.

This implementation uses Unicode escape sequences for all emojis to prevent
encoding issues across different environments.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# EMOJI CONSTANTS - Using Unicode escape sequences to prevent encoding issues
# =============================================================================
EMOJI_ROCKET = "\U0001F680"      # 🚀
EMOJI_STOP = "\U0001F6D1"        # 🛑
EMOJI_CHART = "\U0001F4CA"       # 📊
EMOJI_MONEY = "\U0001F4B0"       # 💰
EMOJI_CHART_UP = "\U0001F4C8"    # 📈
EMOJI_WARNING = "\U000026A0"     # ⚠
EMOJI_CHECK = "\U00002705"       # ✅
EMOJI_CROSS = "\U0000274C"       # ❌
EMOJI_BELL = "\U0001F514"        # 🔔
EMOJI_BRAIN = "\U0001F9E0"       # 🧠
EMOJI_SEARCH = "\U0001F50D"      # 🔍
EMOJI_TARGET = "\U0001F3AF"      # 🎯
EMOJI_HOURGLASS = "\U000023F3"   # ⏳
EMOJI_GREEN_CIRCLE = "\U0001F7E2"  # 🟢
EMOJI_RED_CIRCLE = "\U0001F534"    # 🔴
EMOJI_WHITE_CIRCLE = "\U000026AA"  # ⚪
EMOJI_BIRD = "\U0001F426"        # 🦆 (placeholder for Twitter/X)
EMOJI_ANTENNA = "\U0001F4E1"     # 📡
EMOJI_MEMO = "\U0001F4CB"        # 📋
EMOJI_BULLET = "\U00002022"      # •


class TelegramBot:
    """
    Telegram bot for Gann Sentinel Trader notifications and commands.
    
    Responsibilities:
    - Send trade recommendation notifications
    - Process approval/rejection commands (returns dicts for agent to handle)
    - Provide system status updates
    - Track digest data (scans, signals, decisions)
    - Send comprehensive scan summaries
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN not set - notifications disabled")
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set - notifications disabled")
        
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.last_update_id = 0
        
        # Digest tracking state
        self._scan_start_time: Optional[datetime] = None
        self._source_queries: List[Dict[str, Any]] = []
        self._signals: List[Dict[str, Any]] = []
        self._decisions: List[Dict[str, Any]] = []
        self._system_errors: List[Dict[str, Any]] = []
        self._pending_approvals: List[str] = []
    
    @property
    def is_configured(self) -> bool:
        """Check if bot is properly configured."""
        return bool(self.token and self.chat_id)
    
    # =========================================================================
    # CORE MESSAGING
    # =========================================================================
    
    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = "Markdown",
        disable_notification: bool = False
    ) -> bool:
        """Send a message to Telegram with fallback for parse errors."""
        if not self.is_configured:
            logger.warning("Telegram not configured, skipping message")
            return False
        
        target_chat = chat_id or self.chat_id
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "chat_id": target_chat,
                    "text": text,
                    "disable_notification": disable_notification
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                )
                
                if response.status_code == 200:
                    logger.debug(f"Message sent to {target_chat}")
                    return True
                else:
                    # If Markdown parsing failed, retry without parse_mode
                    if parse_mode and "can't parse" in response.text.lower():
                        logger.warning("Markdown parse failed, retrying without formatting")
                        payload.pop("parse_mode", None)
                        retry_response = await client.post(
                            f"{self.base_url}/sendMessage",
                            json=payload
                        )
                        if retry_response.status_code == 200:
                            return True
                    
                    logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    # =========================================================================
    # COMMAND PROCESSING
    # =========================================================================
    
    async def get_updates(self) -> List[Dict[str, Any]]:
        """Fetch new updates from Telegram."""
        if not self.is_configured:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self.last_update_id + 1,
                        "timeout": 5,
                        "allowed_updates": ["message"]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok") and data.get("result"):
                        updates = data["result"]
                        if updates:
                            self.last_update_id = updates[-1]["update_id"]
                        return updates
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Telegram updates: {e}")
            return []
    
    async def process_commands(self) -> List[Dict[str, Any]]:
        """
        Fetch and parse any pending Telegram commands.
        Returns list of command dicts for agent to handle.
        """
        commands = []
        updates = await self.get_updates()
        
        for update in updates:
            message = update.get("message")
            if not message:
                continue
            
            # Only process messages from our chat
            if str(message.get("chat", {}).get("id")) != str(self.chat_id):
                logger.debug(f"Ignoring message from chat {message.get('chat', {}).get('id')}")
                continue
            
            text = message.get("text", "")
            if not text.startswith("/"):
                continue
            
            # Parse command
            parts = text.split()
            cmd_text = parts[0][1:].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Handle commands with @botname suffix
            if "@" in cmd_text:
                cmd_text = cmd_text.split("@")[0]
            
            # Build command dict based on command type
            cmd_dict = {"command": cmd_text}
            
            if cmd_text == "approve" and args:
                cmd_dict["trade_id"] = args[0]
            
            elif cmd_text == "reject" and args:
                cmd_dict["trade_id"] = args[0]
                cmd_dict["reason"] = " ".join(args[1:]) if len(args) > 1 else "Rejected by user"
            
            commands.append(cmd_dict)
            logger.info(f"Parsed command: {cmd_dict}")
        
        return commands
    
    # =========================================================================
    # DIGEST TRACKING
    # =========================================================================
    
    def record_scan_start(self) -> None:
        """Record when a scan cycle starts."""
        self._scan_start_time = datetime.now(timezone.utc)
        self._source_queries = []
        self._signals = []
        self._decisions = []
        logger.debug("Scan start recorded")
    
    def record_source_query(
        self,
        source: str,
        query: str,
        signals_returned: int,
        error: Optional[str] = None
    ) -> None:
        """Record a source query for digest."""
        self._source_queries.append({
            "source": source,
            "query": query,
            "signals_returned": signals_returned,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Source query recorded: {source} -> {signals_returned} signals")
    
    def record_signal(self, signal: Dict[str, Any]) -> None:
        """Record a signal for digest."""
        self._signals.append(signal)
        sig_id = signal.get('signal_id', 'unknown')
        logger.debug(f"Signal recorded: {sig_id[:8] if sig_id else 'unknown'}")
    
    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision for digest."""
        self._decisions.append({
            **decision,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Decision recorded: {decision.get('decision_type', 'unknown')}")
    
    def record_system_error(self, component: str, error: str) -> None:
        """Record a system error for digest."""
        self._system_errors.append({
            "component": component,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"System error recorded: {component}")
    
    def remove_pending_approval(self, trade_id: str) -> None:
        """Remove a trade from pending approvals list."""
        if trade_id in self._pending_approvals:
            self._pending_approvals.remove(trade_id)
            logger.debug(f"Removed pending approval: {trade_id}")
    
    # =========================================================================
    # COMPREHENSIVE SCAN SUMMARY
    # =========================================================================
    
    async def send_scan_summary(
        self,
        signals: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        portfolio: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a comprehensive scan summary after each scan cycle.
        
        This provides full visibility into:
        - What sources were queried
        - What signals were found
        - Key highlights from each source
        - Claude's analysis and reasoning
        - Trade opportunities considered
        - Conviction score and decision
        """
        now = datetime.now(timezone.utc)
        scan_duration = None
        if self._scan_start_time:
            scan_duration = (now - self._scan_start_time).total_seconds()
        
        msg_parts = []
        
        # Header
        msg_parts.append("=" * 35)
        msg_parts.append(f"{EMOJI_SEARCH} SCAN CYCLE COMPLETE")
        msg_parts.append(f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if scan_duration:
            msg_parts.append(f"Duration: {scan_duration:.1f}s")
        msg_parts.append("=" * 35)
        
        # =====================================================================
        # SOURCES & SIGNAL COUNTS
        # =====================================================================
        msg_parts.append(f"\n{EMOJI_ANTENNA} DATA SOURCES")
        
        total_signals = 0
        errors = []
        
        for query in self._source_queries:
            source = query.get("source", "Unknown")
            count = query.get("signals_returned", 0)
            error = query.get("error")
            total_signals += count
            
            if error:
                msg_parts.append(f"  {EMOJI_CROSS} {source}: ERROR ({error})")
                errors.append(source)
            else:
                msg_parts.append(f"  {EMOJI_CHECK} {source}: {count} signals")
        
        msg_parts.append(f"\nTotal Signals: {total_signals}")
        
        # =====================================================================
        # KEY SIGNALS BY CATEGORY
        # =====================================================================
        if signals:
            msg_parts.append("\n" + "-" * 35)
            msg_parts.append(f"{EMOJI_CHART} KEY SIGNALS")
            
            # Group signals by category/type
            sentiment_signals = []
            macro_signals = []
            prediction_signals = []
            
            for sig in signals:
                sig_type = sig.get("signal_type") or sig.get("category") or ""
                source = sig.get("source") or sig.get("source_type") or ""
                
                if "sentiment" in sig_type.lower() or "grok" in source.lower():
                    sentiment_signals.append(sig)
                elif "macro" in sig_type.lower() or "fred" in source.lower():
                    macro_signals.append(sig)
                elif "prediction" in sig_type.lower() or "polymarket" in source.lower():
                    prediction_signals.append(sig)
            
            # Sentiment/Social signals
            if sentiment_signals:
                msg_parts.append(f"\n{EMOJI_BIRD} SENTIMENT ({len(sentiment_signals)})")
                for sig in sentiment_signals[:3]:
                    summary = sig.get("summary", "")[:100]
                    bias = sig.get("directional_bias", "unclear")
                    confidence = sig.get("confidence", 0)
                    
                    tickers = sig.get("asset_scope", {}).get("tickers", [])
                    ticker_str = ", ".join(tickers[:3]) if tickers else "Market"
                    
                    if bias == "positive":
                        bias_emoji = EMOJI_GREEN_CIRCLE
                    elif bias == "negative":
                        bias_emoji = EMOJI_RED_CIRCLE
                    else:
                        bias_emoji = EMOJI_WHITE_CIRCLE
                    
                    msg_parts.append(f"  {bias_emoji} [{ticker_str}] {summary}")
                    if confidence:
                        msg_parts.append(f"     Confidence: {confidence:.0%}")
            
            # Macro signals
            if macro_signals:
                msg_parts.append(f"\n{EMOJI_CHART_UP} MACRO DATA ({len(macro_signals)})")
                for sig in macro_signals[:4]:
                    summary = sig.get("summary", "")[:80]
                    raw_value = sig.get("raw_value", {})
                    change = raw_value.get("change")
                    
                    if change is not None:
                        msg_parts.append(f"  {EMOJI_BULLET} {summary} (delta: {change:+.2f})")
                    else:
                        msg_parts.append(f"  {EMOJI_BULLET} {summary}")
            
            # Prediction market signals
            if prediction_signals:
                msg_parts.append(f"\n{EMOJI_TARGET} PREDICTIONS ({len(prediction_signals)})")
                sorted_preds = sorted(
                    prediction_signals,
                    key=lambda x: abs(x.get("raw_value", {}).get("change") or 0),
                    reverse=True
                )
                for sig in sorted_preds[:3]:
                    summary = sig.get("summary", "")[:80]
                    raw_value = sig.get("raw_value", {})
                    prob = raw_value.get("value")
                    change = raw_value.get("change")
                    
                    if prob is not None:
                        prob_pct = prob * 100 if prob <= 1 else prob
                        change_str = f" (delta: {change:+.1%})" if change else ""
                        msg_parts.append(f"  {EMOJI_BULLET} {prob_pct:.0f}%{change_str}: {summary}")
                    else:
                        msg_parts.append(f"  {EMOJI_BULLET} {summary}")
        
        # =====================================================================
        # CLAUDE'S ANALYSIS
        # =====================================================================
        msg_parts.append("\n" + "-" * 35)
        msg_parts.append(f"{EMOJI_BRAIN} CLAUDE'S ANALYSIS")
        
        if analysis:
            ticker = analysis.get("ticker")
            recommendation = analysis.get("recommendation", "NONE")
            conviction = analysis.get("conviction_score", 0)
            thesis = analysis.get("thesis", "")
            bull_case = analysis.get("bull_case", "")
            bear_case = analysis.get("bear_case", "")
            time_horizon = analysis.get("time_horizon", "unknown")
            
            # Decision header
            if recommendation in ["BUY", "SELL"] and conviction >= 80:
                decision_emoji = EMOJI_BELL
                decision_text = f"{recommendation} {ticker}"
            elif recommendation in ["BUY", "SELL"]:
                decision_emoji = EMOJI_SEARCH
                decision_text = f"Watching {ticker} ({recommendation})"
            else:
                decision_emoji = EMOJI_WHITE_CIRCLE
                decision_text = "No actionable opportunity"
            
            msg_parts.append(f"\n{decision_emoji} Decision: {decision_text}")
            msg_parts.append(f"Conviction Score: {conviction}/100")
            
            # Conviction bar visualization
            filled = int(conviction / 10)
            empty = 10 - filled
            bar = "#" * filled + "-" * empty
            threshold_note = " <- Threshold: 80" if conviction < 80 else " ACTIONABLE"
            msg_parts.append(f"[{bar}]{threshold_note}")
            
            if ticker:
                msg_parts.append(f"Ticker: {ticker}")
                msg_parts.append(f"Time Horizon: {time_horizon}")
            
            # Thesis
            if thesis:
                msg_parts.append(f"\nThesis:")
                msg_parts.append(f"{thesis[:300]}")
            
            # Bull/Bear cases
            if bull_case:
                msg_parts.append(f"\nBull Case: {bull_case[:150]}...")
            if bear_case:
                msg_parts.append(f"\nBear Case: {bear_case[:150]}...")
            
            # Trade parameters
            if recommendation in ["BUY", "SELL"]:
                entry_price = analysis.get("entry_price_target")
                stop_loss = analysis.get("stop_loss_pct", 0.15)
                position_size = analysis.get("position_size_pct", 0)
                
                msg_parts.append(f"\nTrade Parameters:")
                if entry_price:
                    msg_parts.append(f"  Entry Target: ${entry_price:.2f}")
                msg_parts.append(f"  Stop Loss: {stop_loss:.0%}")
                if position_size:
                    msg_parts.append(f"  Position Size: {position_size:.0%} of portfolio")
        else:
            msg_parts.append(f"\n{EMOJI_CROSS} No analysis generated")
        
        # =====================================================================
        # PORTFOLIO CONTEXT
        # =====================================================================
        if portfolio:
            msg_parts.append("\n" + "-" * 35)
            msg_parts.append(f"{EMOJI_MONEY} PORTFOLIO CONTEXT")
            
            equity = portfolio.get("equity") or portfolio.get("total_value", 0)
            cash = portfolio.get("cash", 0)
            buying_power = portfolio.get("buying_power", 0)
            position_count = portfolio.get("position_count", 0)
            
            msg_parts.append(f"  Equity: ${equity:,.2f}")
            msg_parts.append(f"  Cash: ${cash:,.2f}")
            msg_parts.append(f"  Buying Power: ${buying_power:,.2f}")
            msg_parts.append(f"  Open Positions: {position_count}")
        
        # =====================================================================
        # ERRORS
        # =====================================================================
        if errors or self._system_errors:
            msg_parts.append("\n" + "-" * 35)
            msg_parts.append(f"{EMOJI_WARNING} ERRORS")
            for err in errors:
                msg_parts.append(f"  {EMOJI_BULLET} Source error: {err}")
            for err in self._system_errors[-3:]:
                msg_parts.append(f"  {EMOJI_BULLET} {err.get('component')}: {err.get('error', '')[:50]}")
        
        # Footer
        msg_parts.append("\n" + "=" * 35)
        msg_parts.append("Next scan in ~60 minutes")
        
        # Join and send
        message = "\n".join(msg_parts)
        
        # Telegram has a 4096 character limit
        if len(message) > 4000:
            message = message[:3950] + "\n\n[Message truncated]"
        
        # Send without parse_mode to avoid Markdown issues
        return await self.send_message(message, parse_mode=None)
    
    # =========================================================================
    # NOTIFICATION METHODS
    # =========================================================================
    
    async def send_error_alert(self, component: str, error: str) -> bool:
        """Send error notification."""
        message = (
            f"{EMOJI_WARNING} ERROR: {component}\n\n"
            f"{error[:500]}"
        )
        return await self.send_message(message, parse_mode=None)
    
    async def send_trade_alert(
        self,
        trade_id: str,
        ticker: str,
        side: str,
        quantity: int,
        conviction: int,
        thesis: str
    ) -> bool:
        """Send trade recommendation for approval."""
        short_id = trade_id[:8]
        
        if short_id not in self._pending_approvals:
            self._pending_approvals.append(short_id)
        
        message = (
            f"{EMOJI_BELL} TRADE RECOMMENDATION\n\n"
            f"Ticker: {ticker}\n"
            f"Action: {side.upper()}\n"
            f"Quantity: {quantity} shares\n"
            f"Conviction: {conviction}/100\n\n"
            f"{EMOJI_CHART_UP} THESIS\n"
            f"{thesis[:500]}\n\n"
            f"To approve: /approve {short_id}\n"
            f"To reject: /reject {short_id}"
        )
        return await self.send_message(message, parse_mode=None)
    
    async def send_execution_alert(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        total: float
    ) -> bool:
        """Send notification when trade is executed."""
        message = (
            f"{EMOJI_CHECK} TRADE EXECUTED\n\n"
            f"{side.upper()} {ticker}\n"
            f"Quantity: {quantity}\n"
            f"Price: ${price:.2f}\n"
            f"Total: ${total:.2f}"
        )
        return await self.send_message(message, parse_mode=None)
    
    async def send_stop_loss_alert(
        self,
        ticker: str,
        trigger_price: float,
        loss_pct: float
    ) -> bool:
        """Send notification when stop loss is triggered."""
        message = (
            f"{EMOJI_STOP} STOP LOSS TRIGGERED\n\n"
            f"{ticker}\n"
            f"Trigger Price: ${trigger_price:.2f}\n"
            f"Loss: {loss_pct:.1f}%\n\n"
            f"Position is being closed."
        )
        return await self.send_message(message, parse_mode=None)
    
    async def send_system_status(
        self,
        status: str,
        mode: str,
        approval_gate: bool,
        positions_count: int,
        pending_trades: int
    ) -> bool:
        """Send system status update."""
        gate_status = "ON" if approval_gate else "OFF"
        
        message = (
            f"{EMOJI_CHART} SYSTEM STATUS\n\n"
            f"Status: {status}\n"
            f"Mode: {mode}\n"
            f"Approval Gate: {gate_status}\n"
            f"Open Positions: {positions_count}\n"
            f"Pending Trades: {pending_trades}"
        )
        return await self.send_message(message, parse_mode=None)
    
    async def send_daily_digest(
        self,
        positions: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        pending_approvals: List[Dict[str, Any]]
    ) -> bool:
        """Send the daily digest summary."""
        now = datetime.now(timezone.utc)
        
        msg_parts = [f"{EMOJI_CHART} DAILY DIGEST\n"]
        msg_parts.append(f"{now.strftime('%Y-%m-%d %H:%M UTC')}\n")
        
        # Portfolio summary
        msg_parts.append(f"\n{EMOJI_MONEY} PORTFOLIO")
        total_value = portfolio.get("total_value") or portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)
        
        pnl_emoji = EMOJI_GREEN_CIRCLE if daily_pnl >= 0 else EMOJI_RED_CIRCLE
        msg_parts.append(f"Total Value: ${total_value:,.2f}")
        msg_parts.append(f"Cash: ${cash:,.2f}")
        msg_parts.append(f"Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        
        # Positions
        if positions:
            msg_parts.append(f"\n{EMOJI_CHART_UP} POSITIONS ({len(positions)})")
            for pos in positions[:5]:
                ticker = pos.get("ticker", "N/A")
                qty = pos.get("quantity", 0)
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                pos_emoji = EMOJI_GREEN_CIRCLE if pnl >= 0 else EMOJI_RED_CIRCLE
                msg_parts.append(f"  {EMOJI_BULLET} {ticker}: {qty} shares | {pos_emoji} ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            if len(positions) > 5:
                msg_parts.append(f"  ...and {len(positions) - 5} more")
        else:
            msg_parts.append(f"\n{EMOJI_CHART_UP} POSITIONS\nNo open positions")
        
        # Scan activity
        msg_parts.append(f"\n{EMOJI_SEARCH} SCAN ACTIVITY")
        msg_parts.append(f"Sources queried: {len(self._source_queries)}")
        total_signals = sum(q.get("signals_returned", 0) for q in self._source_queries)
        msg_parts.append(f"Signals gathered: {total_signals}")
        errors = [q for q in self._source_queries if q.get("error")]
        if errors:
            msg_parts.append(f"Errors: {len(errors)}")
        
        # Decisions
        if self._decisions:
            msg_parts.append(f"\n{EMOJI_MEMO} DECISIONS")
            for decision in self._decisions[-3:]:
                dtype = decision.get("decision_type", "UNKNOWN")
                if dtype == "TRADE":
                    details = decision.get("trade_details", {})
                    ticker = details.get("ticker", "N/A")
                    side = details.get("side", "N/A")
                    msg_parts.append(f"  {EMOJI_BULLET} {dtype}: {side} {ticker}")
                else:
                    rationale = decision.get("reasoning", {}).get("rationale", "")[:50]
                    msg_parts.append(f"  {EMOJI_BULLET} {dtype}: {rationale}...")
        
        # Pending approvals
        if pending_approvals:
            msg_parts.append(f"\n{EMOJI_HOURGLASS} PENDING APPROVALS ({len(pending_approvals)})")
            for trade in pending_approvals[:3]:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                trade_id = trade.get("id", "")[:8]
                msg_parts.append(f"  {EMOJI_BULLET} {side} {ticker} ({trade_id})")
        
        # System errors
        if self._system_errors:
            msg_parts.append(f"\n{EMOJI_WARNING} ERRORS ({len(self._system_errors)})")
            for err in self._system_errors[-3:]:
                component = err.get("component", "unknown")
                error_msg = err.get("error", "")[:30]
                msg_parts.append(f"  {EMOJI_BULLET} [{component}] {error_msg}...")
        
        message = "\n".join(msg_parts)
        
        # Reset tracking after sending digest
        self._source_queries = []
        self._signals = []
        self._decisions = []
        self._system_errors = []
        
        return await self.send_message(message, parse_mode=None)


# Convenience function for quick notifications
async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
