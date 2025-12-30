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
⚠️ **ERROR: {component}**

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
🔔 **TRADE RECOMMENDATION**

**Ticker:** {ticker}
**Action:** {side.upper()}
**Quantity:** {quantity} shares
**Conviction:** {conviction}/100

📈 **THESIS**
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
✅ **TRADE EXECUTED**

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
🛑 **STOP LOSS TRIGGERED**

**{ticker}**
Trigger Price: ${trigger_price:.2f}
Loss: {loss_pct:.1f}%

Position is being closed.
"""
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
📊 **SYSTEM STATUS**

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
        msg_parts = ["📊 **DAILY DIGEST**\n"]
        msg_parts.append(f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_\n")
        
        # Portfolio summary
        msg_parts.append("\n**💰 PORTFOLIO**")
        total_value = portfolio.get("total_value") or portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)
        
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        msg_parts.append(f"Total Value: ${total_value:,.2f}")
        msg_parts.append(f"Cash: ${cash:,.2f}")
        msg_parts.append(f"Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        
        # Positions
        if positions:
            msg_parts.append(f"\n**📈 POSITIONS ({len(positions)})**")
            for pos in positions[:5]:  # Limit to 5
                ticker = pos.get("ticker", "N/A")
                qty = pos.get("quantity", 0)
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                pos_emoji = "🟢" if pnl >= 0 else "🔴"
                msg_parts.append(f"• {ticker}: {qty} shares | {pos_emoji} ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            if len(positions) > 5:
                msg_parts.append(f"  _...and {len(positions) - 5} more_")
        else:
            msg_parts.append("\n**📈 POSITIONS**\nNo open positions")
        
        # Scan activity (from tracked data)
        msg_parts.append(f"\n**🔍 SCAN ACTIVITY**")
        msg_parts.append(f"Sources queried: {len(self._source_queries)}")
        total_signals = sum(q.get("signals_returned", 0) for q in self._source_queries)
        msg_parts.append(f"Signals gathered: {total_signals}")
        errors = [q for q in self._source_queries if q.get("error")]
        if errors:
            msg_parts.append(f"Errors: {len(errors)}")
        
        # Decisions
        if self._decisions:
            msg_parts.append(f"\n**📋 DECISIONS**")
            for decision in self._decisions[-3:]:  # Last 3
                dtype = decision.get("decision_type", "UNKNOWN")
                if dtype == "TRADE":
                    details = decision.get("trade_details", {})
                    ticker = details.get("ticker", "N/A")
                    side = details.get("side", "N/A")
                    msg_parts.append(f"• {dtype}: {side} {ticker}")
                else:
                    rationale = decision.get("reasoning", {}).get("rationale", "")[:50]
                    msg_parts.append(f"• {dtype}: {rationale}...")
        
        # Pending approvals
        if pending_approvals:
            msg_parts.append(f"\n**⏳ PENDING APPROVALS ({len(pending_approvals)})**")
            for trade in pending_approvals[:3]:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                trade_id = trade.get("id", "")[:8]
                msg_parts.append(f"• {side} {ticker} (`{trade_id}`)")
        
        # System errors
        if self._system_errors:
            msg_parts.append(f"\n**⚠️ ERRORS ({len(self._system_errors)})**")
            for err in self._system_errors[-3:]:
                component = err.get("component", "unknown")
                error_msg = err.get("error", "")[:30]
                msg_parts.append(f"• [{component}] {error_msg}...")
        
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
