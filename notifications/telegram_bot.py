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
EMOJI_BIRD = "\U0001F426"        # 🐦
EMOJI_ANTENNA = "\U0001F4E1"     # 📡
EMOJI_MEMO = "\U0001F4CB"        # 📋
EMOJI_BULLET = "\U00002022"      # •
EMOJI_KEYBOARD = "\U00002328"    # ⌨
EMOJI_BEAR = "\U0001F43B"        # 🐻
EMOJI_BULL = "\U0001F402"        # 🐂


class TelegramBot:
    """
    Telegram bot for Gann Sentinel Trader notifications and commands.
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
        self._risk_rejections: List[Dict[str, Any]] = []
    
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
        """Fetch and parse any pending Telegram commands."""
        commands = []
        updates = await self.get_updates()
        
        for update in updates:
            message = update.get("message")
            if not message:
                continue
            
            if str(message.get("chat", {}).get("id")) != str(self.chat_id):
                continue
            
            text = message.get("text", "")
            if not text.startswith("/"):
                continue
            
            parts = text.split()
            cmd_text = parts[0][1:].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if "@" in cmd_text:
                cmd_text = cmd_text.split("@")[0]
            
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
        self._risk_rejections = []
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
    
    def record_signal(self, signal: Dict[str, Any]) -> None:
        """Record a signal for digest."""
        self._signals.append(signal)
    
    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision for digest."""
        self._decisions.append({
            **decision,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def record_risk_rejection(self, ticker: str, reason: str) -> None:
        """Record a risk engine rejection for inclusion in scan summary."""
        self._risk_rejections.append({
            "ticker": ticker,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Risk rejection recorded: {ticker} - {reason}")
    
    def record_system_error(self, component: str, error: str) -> None:
        """Record a system error for digest."""
        self._system_errors.append({
            "component": component,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def remove_pending_approval(self, trade_id: str) -> None:
        """Remove a trade from pending approvals list."""
        if trade_id in self._pending_approvals:
            self._pending_approvals.remove(trade_id)
    
    # =========================================================================
    # COMPREHENSIVE SCAN SUMMARY (SINGLE CONSOLIDATED MESSAGE)
    # =========================================================================
    
    async def send_scan_summary(
        self,
        signals: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        portfolio: Optional[Dict[str, Any]] = None,
        pending_trade_id: Optional[str] = None
    ) -> bool:
        """
        Send a comprehensive scan summary after each scan cycle.
        This is the ONE message sent after each scan.
        """
        now = datetime.now(timezone.utc)
        scan_duration = None
        if self._scan_start_time:
            scan_duration = (now - self._scan_start_time).total_seconds()
        
        lines = []
        
        # Header
        lines.append("=" * 40)
        lines.append(f"{EMOJI_SEARCH} SCAN CYCLE COMPLETE")
        lines.append(f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if scan_duration:
            lines.append(f"Duration: {scan_duration:.1f}s")
        lines.append("=" * 40)
        
        # =====================================================================
        # SOURCES & SIGNAL COUNTS
        # =====================================================================
        lines.append(f"\n{EMOJI_ANTENNA} DATA SOURCES")
        
        total_signals = 0
        errors = []
        
        for query in self._source_queries:
            source = query.get("source", "Unknown")
            count = query.get("signals_returned", 0)
            error = query.get("error")
            total_signals += count
            
            if error:
                lines.append(f"  {EMOJI_CROSS} {source}: ERROR ({error})")
                errors.append(source)
            else:
                lines.append(f"  {EMOJI_CHECK} {source}: {count} signals")
        
        lines.append(f"\nTotal: {total_signals} signals")
        
        # =====================================================================
        # KEY SIGNALS BY CATEGORY (condensed)
        # =====================================================================
        if signals:
            lines.append("\n" + "-" * 40)
            lines.append(f"{EMOJI_CHART} KEY SIGNALS")
            
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
            
            if sentiment_signals:
                lines.append(f"\n{EMOJI_BIRD} Sentiment: {len(sentiment_signals)} signals")
                for sig in sentiment_signals[:2]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")
            
            if macro_signals:
                lines.append(f"\n{EMOJI_CHART_UP} Macro: {len(macro_signals)} signals")
                for sig in macro_signals[:3]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")
            
            if prediction_signals:
                lines.append(f"\n{EMOJI_TARGET} Predictions: {len(prediction_signals)} signals")
                sorted_preds = sorted(
                    prediction_signals,
                    key=lambda x: abs(x.get("raw_value", {}).get("change") or 0),
                    reverse=True
                )
                for sig in sorted_preds[:3]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")
        
        # =====================================================================
        # CLAUDE'S ANALYSIS (with Bull/Bear cases restored)
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_BRAIN} CLAUDE'S ANALYSIS")
        
        if analysis:
            ticker = analysis.get("ticker")
            recommendation = analysis.get("recommendation", "NONE")
            conviction = analysis.get("conviction_score", 0)
            thesis = analysis.get("thesis", "")
            bull_case = analysis.get("bull_case", "")
            bear_case = analysis.get("bear_case", "")
            time_horizon = analysis.get("time_horizon", "unknown")
            
            # Conviction bar with threshold marker
            filled = int(conviction / 10)
            empty = 10 - filled
            bar = "#" * filled + "-" * empty
            
            # Determine actionability
            is_actionable = conviction >= 80 and recommendation in ["BUY", "SELL"]
            
            lines.append(f"\nConviction: {conviction}/100")
            lines.append(f"[{bar}] {'<-- 80 threshold' if conviction < 80 else ''}")
            
            if is_actionable:
                lines.append(f"{EMOJI_GREEN_CIRCLE} ACTIONABLE (80+)")
            else:
                lines.append(f"{EMOJI_WHITE_CIRCLE} Below threshold")
            
            if ticker and recommendation in ["BUY", "SELL"]:
                lines.append(f"\nRecommendation: {recommendation} {ticker}")
                lines.append(f"Time Horizon: {time_horizon}")
                
                # Thesis
                if thesis:
                    lines.append(f"\nThesis: {thesis[:250]}...")
                
                # Bull Case (restored)
                if bull_case:
                    lines.append(f"\n{EMOJI_BULL} Bull: {bull_case[:150]}...")
                
                # Bear Case (restored)
                if bear_case:
                    lines.append(f"\n{EMOJI_BEAR} Bear: {bear_case[:150]}...")
                
                # Trade Parameters - normalize display
                position_size = analysis.get("position_size_pct", 0)
                stop_loss = analysis.get("stop_loss_pct", 0)
                
                # Normalize for display
                if position_size > 1:
                    position_display = f"{position_size:.0f}%"
                else:
                    position_display = f"{position_size * 100:.0f}%"
                    
                if stop_loss > 1:
                    stop_display = f"{stop_loss:.0f}%"
                else:
                    stop_display = f"{stop_loss * 100:.0f}%"
                
                lines.append(f"\nTrade Parameters:")
                lines.append(f"  Stop Loss: {stop_display}")
                lines.append(f"  Position Size: {position_display}")
            else:
                lines.append(f"\nNo trade recommendation")
        else:
            lines.append(f"\n{EMOJI_CROSS} Analysis not generated")
        
        # =====================================================================
        # TRADE STATUS
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_WARNING} TRADE STATUS")
        
        if self._risk_rejections:
            for rejection in self._risk_rejections:
                lines.append(f"\n{EMOJI_RED_CIRCLE} REJECTED BY RISK ENGINE")
                lines.append(f"Ticker: {rejection['ticker']}")
                lines.append(f"Reason: {rejection['reason']}")
            lines.append(f"\n{EMOJI_BULLET} Trade blocked - no approval needed")
        elif pending_trade_id:
            lines.append(f"\n{EMOJI_BELL} TRADE PENDING APPROVAL")
            lines.append(f"Trade ID: {pending_trade_id}")
            lines.append(f"\n{EMOJI_CHECK} /approve {pending_trade_id}")
            lines.append(f"{EMOJI_CROSS} /reject {pending_trade_id}")
        elif analysis and analysis.get("conviction_score", 0) >= 80:
            lines.append(f"\n{EMOJI_WHITE_CIRCLE} Conviction met but trade not created")
            lines.append("(Check logs for sizing/quote issues)")
        else:
            lines.append(f"\n{EMOJI_WHITE_CIRCLE} No trade - conviction below 80")
        
        # =====================================================================
        # PORTFOLIO SNAPSHOT
        # =====================================================================
        if portfolio:
            lines.append("\n" + "-" * 40)
            lines.append(f"{EMOJI_MONEY} PORTFOLIO")
            
            equity = portfolio.get("equity") or portfolio.get("total_value", 0)
            cash = portfolio.get("cash", 0)
            position_count = portfolio.get("position_count", 0)
            
            lines.append(f"  Equity: ${equity:,.2f}")
            lines.append(f"  Cash: ${cash:,.2f}")
            lines.append(f"  Positions: {position_count}")
        
        # =====================================================================
        # COMMANDS
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_KEYBOARD} COMMANDS")
        lines.append("/status - System status")
        lines.append("/pending - Pending trades")
        lines.append("/approve [id] - Approve trade")
        lines.append("/reject [id] - Reject trade")
        lines.append("/digest - Daily digest")
        lines.append("/stop - Emergency halt")
        
        # Footer
        lines.append("\n" + "=" * 40)
        lines.append("Next scan: ~60 minutes")
        
        message = "\n".join(lines)
        
        if len(message) > 4000:
            message = message[:3950] + "\n\n[Truncated]"
        
        return await self.send_message(message, parse_mode=None)
    
    # =========================================================================
    # OTHER NOTIFICATION METHODS
    # =========================================================================
    
    async def send_error_alert(self, component: str, error: str) -> bool:
        """Send error notification."""
        message = f"{EMOJI_WARNING} ERROR: {component}\n\n{error[:500]}"
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
            f"{EMOJI_BELL} TRADE PENDING APPROVAL\n\n"
            f"Ticker: {ticker}\n"
            f"Action: {side.upper()}\n"
            f"Quantity: {quantity} shares\n"
            f"Conviction: {conviction}/100\n\n"
            f"Thesis: {thesis[:300]}...\n\n"
            f"/approve {short_id}\n"
            f"/reject {short_id}"
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
            f"Trigger: ${trigger_price:.2f}\n"
            f"Loss: {loss_pct:.1f}%"
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
            f"Positions: {positions_count}\n"
            f"Pending: {pending_trades}"
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
        
        lines = [
            "=" * 40,
            f"{EMOJI_CHART} DAILY DIGEST",
            f"{now.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 40
        ]
        
        lines.append(f"\n{EMOJI_MONEY} PORTFOLIO")
        total_value = portfolio.get("total_value") or portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)
        
        pnl_emoji = EMOJI_GREEN_CIRCLE if daily_pnl >= 0 else EMOJI_RED_CIRCLE
        lines.append(f"Total: ${total_value:,.2f}")
        lines.append(f"Cash: ${cash:,.2f}")
        lines.append(f"P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        
        if positions:
            lines.append(f"\n{EMOJI_CHART_UP} POSITIONS ({len(positions)})")
            for pos in positions[:5]:
                ticker = pos.get("ticker", "N/A")
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                pos_emoji = EMOJI_GREEN_CIRCLE if pnl >= 0 else EMOJI_RED_CIRCLE
                lines.append(f"  {EMOJI_BULLET} {ticker}: {pos_emoji} ${pnl:,.2f} ({pnl_pct:+.1f}%)")
        else:
            lines.append(f"\n{EMOJI_CHART_UP} POSITIONS: None")
        
        if pending_approvals:
            lines.append(f"\n{EMOJI_HOURGLASS} PENDING ({len(pending_approvals)})")
            for trade in pending_approvals[:3]:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                trade_id = trade.get("id", "")[:8]
                lines.append(f"  {EMOJI_BULLET} {side} {ticker} - /approve {trade_id}")
        
        lines.append(f"\n{EMOJI_KEYBOARD} /status /pending /help")
        lines.append("=" * 40)
        
        message = "\n".join(lines)
        
        self._source_queries = []
        self._signals = []
        self._decisions = []
        self._system_errors = []
        
        return await self.send_message(message, parse_mode=None)


async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
