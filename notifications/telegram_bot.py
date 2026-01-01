"""
Telegram Bot for Gann Sentinel Trader
Handles notifications and command processing for trade approvals and system control.

This implementation is aligned with agent.py's expectations.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for Gann Sentinel Trader notifications and commands.
    
    Responsibilities:
    - Send trade recommendation notifications
    - Process approval/rejection commands (returns dicts for agent to handle)
    - Provide system status updates
    - Track digest data (scans, signals, decisions)
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
        self._pending_approvals: List[str] = []  # trade_ids awaiting approval
    
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
        parse_mode: str = "Markdown",
        disable_notification: bool = False
    ) -> bool:
        """Send a message to Telegram."""
        if not self.is_configured:
            logger.warning("Telegram not configured, skipping message")
            return False
        
        target_chat = chat_id or self.chat_id
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": target_chat,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_notification": disable_notification
                    }
                )
                
                if response.status_code == 200:
                    logger.debug(f"Message sent to {target_chat}")
                    return True
                else:
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
        
        Returns dicts like:
        - {"command": "status"}
        - {"command": "approve", "trade_id": "abc123"}
        - {"command": "reject", "trade_id": "abc123", "reason": "..."}
        - {"command": "stop"}
        - {"command": "resume"}
        - {"command": "digest"}
        - {"command": "help"}
        - {"command": "pending"}
        - {"command": "catalyst", "description": "SpaceX IPO expected H2 2026"}
        - {"command": "whatif", "description": "Fed cuts rates by 50bps"}
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
            cmd_text = parts[0][1:].lower()  # Remove leading /
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
            
            elif cmd_text in ["catalyst", "whatif"] and args:
                # Capture the full catalyst description
                cmd_dict["description"] = " ".join(args)
            
            commands.append(cmd_dict)
            logger.info(f"Parsed command: {cmd_dict}")
        
        return commands
    
    # =========================================================================
    # DIGEST TRACKING (called by agent to track activity)
    # =========================================================================
    
    def record_scan_start(self) -> None:
        """Record when a scan cycle starts."""
        self._scan_start_time = datetime.now(timezone.utc)
        # Reset tracking for new scan
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
        logger.debug(f"Signal recorded: {signal.get('signal_id', 'unknown')[:8]}")
    
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
    # NOTIFICATION METHODS (called by agent)
    # =========================================================================
    
    async def send_error_alert(self, component: str, error: str) -> bool:
        """Send error notification."""
        message = f"""
âš ï¸ **ERROR: {component}**

{error[:500]}
"""
        return await self.send_message(message)
    
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
        
        # Track pending approval
        if short_id not in self._pending_approvals:
            self._pending_approvals.append(short_id)
        
        message = f"""
ðŸ”” **TRADE RECOMMENDATION**

**Ticker:** {ticker}
**Action:** {side.upper()}
**Quantity:** {quantity} shares
**Conviction:** {conviction}/100

ðŸ“ˆ **THESIS**
{thesis[:500]}

To approve: `/approve {short_id}`
To reject: `/reject {short_id}`
"""
        return await self.send_message(message)
    
    async def send_execution_alert(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        total: float
    ) -> bool:
        """Send notification when trade is executed."""
        message = f"""
âœ… **TRADE EXECUTED**

**{side.upper()} {ticker}**
Quantity: {quantity}
Price: ${price:.2f}
Total: ${total:.2f}
"""
        return await self.send_message(message)
    
    async def send_stop_loss_alert(
        self,
        ticker: str,
        trigger_price: float,
        loss_pct: float
    ) -> bool:
        """Send notification when stop loss is triggered."""
        message = f"""
ðŸ›‘ **STOP LOSS TRIGGERED**

**{ticker}**
Trigger Price: ${trigger_price:.2f}
Loss: {loss_pct:.1f}%

Position is being closed.
"""
        return await self.send_message(message)
    
    async def send_catalyst_analysis(
        self,
        analysis: Dict[str, Any],
        catalyst_query: str
    ) -> bool:
        """
        Send catalyst analysis results with historical context.
        
        This is the response to /catalyst or /whatif commands.
        Shows Claude's reasoning that combines:
        1. Historical pattern recognition
        2. Forward catalyst analysis
        3. Second-order thinking
        """
        now = datetime.now(timezone.utc)
        
        msg_parts = []
        msg_parts.append("🎯 **CATALYST ANALYSIS**")
        msg_parts.append(f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_\n")
        
        # Show the query
        msg_parts.append(f"**Query:** _{catalyst_query}_\n")
        msg_parts.append("-" * 30)
        
        # Check if we have a valid analysis
        recommendation = analysis.get("recommendation", "NONE")
        ticker = analysis.get("ticker")
        conviction = analysis.get("conviction_score", 0)
        
        if recommendation in ["BUY", "SELL"] and ticker and conviction >= 60:
            # We have an actionable or watchable trade
            
            # Decision header with emoji
            if conviction >= 80:
                decision_emoji = "🚨"
                action_note = "ACTIONABLE"
            else:
                decision_emoji = "👀"
                action_note = "WATCHING"
            
            msg_parts.append(f"\n{decision_emoji} **{recommendation} {ticker}** ({action_note})")
            
            # Conviction bar
            filled = int(conviction / 10)
            empty = 10 - filled
            bar = "█" * filled + "░" * empty
            msg_parts.append(f"**Conviction:** `[{bar}]` {conviction}/100")
            
            # =====================================================
            # HISTORICAL CONTEXT (NEW SECTION)
            # =====================================================
            hist_ctx = analysis.get("historical_context")
            if hist_ctx and isinstance(hist_ctx, dict):
                analogous = hist_ctx.get("analogous_event", "")
                period = hist_ctx.get("historical_period", "")
                outcome = hist_ctx.get("historical_outcome", "")
                confidence = hist_ctx.get("pattern_confidence", "")
                rhymes = hist_ctx.get("rhymes_with", "")
                
                if analogous or period:
                    msg_parts.append(f"\n**📚 HISTORICAL PATTERN**")
                    if period:
                        msg_parts.append(f"_\"This reminds me of {period}...\"_")
                    if analogous:
                        msg_parts.append(f"**Analogue:** {analogous[:150]}")
                    if outcome:
                        msg_parts.append(f"**What happened:** {outcome[:150]}")
                    if confidence:
                        conf_emoji = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                        msg_parts.append(f"Pattern confidence: {conf_emoji} {confidence.upper()}")
                    
                    # Key differences
                    diffs = hist_ctx.get("key_differences", [])
                    if diffs and len(diffs) > 0:
                        msg_parts.append(f"**Key differences:** {', '.join(diffs[:2])}")
            
            # Technical & Cycle Context
            tech_ctx = analysis.get("technical_context", "")
            macro_pos = analysis.get("macro_cycle_position", "")
            seasonal = analysis.get("seasonal_factors", "")
            
            if tech_ctx or macro_pos or seasonal:
                msg_parts.append(f"\n**📈 MARKET CONTEXT**")
                if tech_ctx:
                    msg_parts.append(f"Technical: {tech_ctx[:100]}")
                if macro_pos:
                    msg_parts.append(f"Cycle: {macro_pos}")
                if seasonal and seasonal != "none":
                    msg_parts.append(f"Seasonal: {seasonal[:80]}")
            
            # =====================================================
            # FORWARD CATALYST
            # =====================================================
            catalyst = analysis.get("catalyst", "N/A")
            catalyst_date = analysis.get("catalyst_date", "TBD")
            catalyst_horizon = analysis.get("catalyst_horizon", "unknown")
            
            msg_parts.append(f"\n**📅 FORWARD CATALYST**")
            msg_parts.append(f"{catalyst}")
            msg_parts.append(f"Timeline: {catalyst_date} ({catalyst_horizon})")
            
            # Second-order reasoning
            is_primary = analysis.get("primary_beneficiary", True)
            second_order = analysis.get("second_order_rationale", "")
            
            if not is_primary and second_order:
                msg_parts.append(f"\n**🧠 SECOND-ORDER THINKING**")
                msg_parts.append(f"_{second_order[:250]}_")
            
            # =====================================================
            # THESIS (History + Forward Combined)
            # =====================================================
            thesis = analysis.get("thesis", "")
            if thesis:
                msg_parts.append(f"\n**📝 THESIS**")
                msg_parts.append(f"{thesis[:350]}")
            
            # Variant perception
            variant = analysis.get("variant_perception", "")
            if variant:
                msg_parts.append(f"\n**💡 VARIANT PERCEPTION**")
                msg_parts.append(f"_{variant[:180]}_")
            
            # Bull/Bear cases
            bull = analysis.get("bull_case", "")
            bear = analysis.get("bear_case", "")
            
            if bull or bear:
                msg_parts.append(f"\n**⚖️ RISK/REWARD**")
                if bull:
                    msg_parts.append(f"🟢 Bull: {bull[:120]}...")
                if bear:
                    msg_parts.append(f"🔴 Bear: {bear[:120]}...")
            
            # Trade parameters
            position_size = analysis.get("position_size_pct", 0)
            stop_loss = analysis.get("stop_loss_pct", 0.15)
            time_horizon = analysis.get("time_horizon", "unknown")
            entry_price = analysis.get("entry_price_target")
            
            msg_parts.append(f"\n**📊 TRADE PARAMETERS**")
            msg_parts.append(f"Position Size: {position_size:.0%} of portfolio")
            msg_parts.append(f"Stop Loss: {stop_loss:.0%}")
            msg_parts.append(f"Time Horizon: {time_horizon}")
            if entry_price:
                msg_parts.append(f"Entry Target: ${entry_price:.2f}")
            
            # If actionable, show approval instructions
            if conviction >= 80:
                msg_parts.append(f"\n" + "-" * 30)
                msg_parts.append("_This analysis suggests an actionable trade._")
                msg_parts.append("_Run /scan to generate a formal trade recommendation._")
        
        else:
            # No actionable trade found
            msg_parts.append(f"\n**💤 NO ACTIONABLE TRADE**")
            msg_parts.append(f"Conviction: {conviction}/100 (threshold: 80)")
            
            # Still show historical context if available
            hist_ctx = analysis.get("historical_context")
            if hist_ctx and isinstance(hist_ctx, dict):
                period = hist_ctx.get("historical_period", "")
                if period:
                    msg_parts.append(f"\n**Historical note:** Pattern resembles {period}")
            
            # Show reasoning anyway
            thesis = analysis.get("thesis", "No clear opportunity identified.")
            msg_parts.append(f"\n**Analysis:**")
            msg_parts.append(f"_{thesis[:400]}_")
            
            # If there's a ticker being watched
            if ticker:
                msg_parts.append(f"\n**Watching:** {ticker}")
                bear = analysis.get("bear_case", "")
                if bear:
                    msg_parts.append(f"**Concern:** {bear[:200]}")
        
        # Footer
        msg_parts.append(f"\n" + "=" * 30)
        msg_parts.append("_Use /catalyst <description> for more queries_")
        
        message = "\n".join(msg_parts)
        
        # Truncate if too long for Telegram
        if len(message) > 4000:
            message = message[:3950] + "\n\n_[Truncated]_"
        
        return await self.send_message(message)
    
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
        
        message = f"""
ðŸ“Š **SYSTEM STATUS**

Status: {status}
Mode: {mode}
Approval Gate: {gate_status}
Open Positions: {positions_count}
Pending Trades: {pending_trades}
"""
        return await self.send_message(message)
    
    async def send_daily_digest(
        self,
        positions: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        pending_approvals: List[Dict[str, Any]]
    ) -> bool:
        """Send the daily digest summary."""
        now = datetime.now(timezone.utc)
        
        # Build digest message
        msg_parts = ["ðŸ“Š **DAILY DIGEST**\n"]
        msg_parts.append(f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_\n")
        
        # Portfolio summary
        msg_parts.append("\n**ðŸ’° PORTFOLIO**")
        total_value = portfolio.get("total_value") or portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)
        
        pnl_emoji = "ðŸŸ¢" if daily_pnl >= 0 else "ðŸ”´"
        msg_parts.append(f"Total Value: ${total_value:,.2f}")
        msg_parts.append(f"Cash: ${cash:,.2f}")
        msg_parts.append(f"Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        
        # Positions
        if positions:
            msg_parts.append(f"\n**ðŸ“ˆ POSITIONS ({len(positions)})**")
            for pos in positions[:5]:  # Limit to 5
                ticker = pos.get("ticker", "N/A")
                qty = pos.get("quantity", 0)
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                pos_emoji = "ðŸŸ¢" if pnl >= 0 else "ðŸ”´"
                msg_parts.append(f"â€¢ {ticker}: {qty} shares | {pos_emoji} ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            if len(positions) > 5:
                msg_parts.append(f"  _...and {len(positions) - 5} more_")
        else:
            msg_parts.append("\n**ðŸ“ˆ POSITIONS**\nNo open positions")
        
        # Scan activity (from tracked data)
        msg_parts.append(f"\n**ðŸ” SCAN ACTIVITY**")
        msg_parts.append(f"Sources queried: {len(self._source_queries)}")
        total_signals = sum(q.get("signals_returned", 0) for q in self._source_queries)
        msg_parts.append(f"Signals gathered: {total_signals}")
        errors = [q for q in self._source_queries if q.get("error")]
        if errors:
            msg_parts.append(f"Errors: {len(errors)}")
        
        # Decisions
        if self._decisions:
            msg_parts.append(f"\n**ðŸ“‹ DECISIONS**")
            for decision in self._decisions[-3:]:  # Last 3
                dtype = decision.get("decision_type", "UNKNOWN")
                if dtype == "TRADE":
                    details = decision.get("trade_details", {})
                    ticker = details.get("ticker", "N/A")
                    side = details.get("side", "N/A")
                    msg_parts.append(f"â€¢ {dtype}: {side} {ticker}")
                else:
                    rationale = decision.get("reasoning", {}).get("rationale", "")[:50]
                    msg_parts.append(f"â€¢ {dtype}: {rationale}...")
        
        # Pending approvals
        if pending_approvals:
            msg_parts.append(f"\n**â³ PENDING APPROVALS ({len(pending_approvals)})**")
            for trade in pending_approvals[:3]:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                trade_id = trade.get("id", "")[:8]
                msg_parts.append(f"â€¢ {side} {ticker} (`{trade_id}`)")
        
        # System errors
        if self._system_errors:
            msg_parts.append(f"\n**âš ï¸ ERRORS ({len(self._system_errors)})**")
            for err in self._system_errors[-3:]:
                component = err.get("component", "unknown")
                error_msg = err.get("error", "")[:30]
                msg_parts.append(f"â€¢ [{component}] {error_msg}...")
        
        message = "\n".join(msg_parts)
        
        # Reset tracking after sending digest
        self._source_queries = []
        self._signals = []
        self._decisions = []
        self._system_errors = []
        
        return await self.send_message(message)


# Convenience function for quick notifications
async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
