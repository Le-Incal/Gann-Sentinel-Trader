"""
Gann Sentinel Trader - Telegram Bot
Handles notifications and user commands.
"""

import logging
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any, List

from config import Config

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for notifications and commands.
    """
    
    def __init__(self):
        """Initialize Telegram bot."""
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        
        # Command handlers
        self.command_handlers: Dict[str, Callable] = {}
        
        # Pending approvals
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        
        # Last processed update ID
        self.last_update_id = 0
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured - notifications disabled")
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_preview: bool = True
    ) -> bool:
        """
        Send a message to the configured chat.
        
        Args:
            text: Message text
            parse_mode: Markdown or HTML
            disable_preview: Disable link previews
            
        Returns:
            True if sent successfully
        """
        if not self.token or not self.chat_id:
            logger.debug(f"Telegram message (not sent): {text[:100]}...")
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": disable_preview
                    }
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_trade_alert(
        self,
        trade_id: str,
        ticker: str,
        side: str,
        quantity: float,
        conviction: int,
        thesis: str
    ) -> bool:
        """
        Send a trade alert requiring approval.
        
        Args:
            trade_id: Unique trade ID
            ticker: Stock ticker
            side: BUY or SELL
            quantity: Number of shares
            conviction: Conviction score (0-100)
            thesis: Trade thesis
            
        Returns:
            True if sent successfully
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        short_id = trade_id[:8]
        
        # Store pending approval
        self.pending_approvals[short_id] = {
            "trade_id": trade_id,
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "conviction": conviction,
            "thesis": thesis,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        message = f"""
{emoji} **TRADE ALERT: {side.upper()} {ticker}**

**Conviction:** {conviction}/100
**Quantity:** {quantity:.2f} shares

**Thesis:**
{thesis[:400]}{'...' if len(thesis) > 400 else ''}

---
⏳ Awaiting approval

Reply with:
`/approve {short_id}` - Execute trade
`/reject {short_id}` - Cancel trade
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
        """Send notification that trade was executed."""
        emoji = "✅"
        
        message = f"""
{emoji} **TRADE EXECUTED**

**{side.upper()} {ticker}**
• Quantity: {quantity:.2f} shares
• Price: ${price:.2f}
• Total: ${total:.2f}
"""
        
        return await self.send_message(message)
    
    async def send_stop_loss_alert(
        self,
        ticker: str,
        trigger_price: float,
        loss_pct: float
    ) -> bool:
        """Send stop loss trigger alert."""
        message = f"""
🛑 **STOP LOSS TRIGGERED**

**{ticker}** hit stop loss
• Trigger Price: ${trigger_price:.2f}
• Loss: {loss_pct:.1f}%

Position will be closed.
"""
        
        return await self.send_message(message)
    
    async def send_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        positions: List[Dict[str, Any]],
        trades_today: int
    ) -> bool:
        """Send end of day summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        positions_text = ""
        if positions:
            for pos in positions[:5]:  # Max 5 positions
                ticker = pos.get("ticker", "?")
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                positions_text += f"• {ticker}: {pnl_pct:+.2f}%\n"
        else:
            positions_text = "No open positions"
        
        message = f"""
📊 **DAILY SUMMARY**

**Portfolio Value:** ${portfolio_value:,.2f}
{pnl_emoji} **Today's P&L:** ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)

**Positions:**
{positions_text}

**Trades Today:** {trades_today}

---
_Market closed. See you tomorrow!_
"""
        
        return await self.send_message(message)
    
    async def send_error_alert(
        self,
        component: str,
        error_message: str
    ) -> bool:
        """Send error alert."""
        message = f"""
⚠️ **ERROR ALERT**

**Component:** {component}
**Error:** {error_message}

Check logs for details.
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
        gate_status = "🟢 ON" if approval_gate else "🔴 OFF"
        
        message = f"""
🤖 **SYSTEM STATUS**

**Status:** {status}
**Mode:** {mode}
**Approval Gate:** {gate_status}

**Positions:** {positions_count}
**Pending Trades:** {pending_trades}
"""
        
        return await self.send_message(message)
    
    async def get_updates(self) -> List[Dict[str, Any]]:
        """Get new messages/commands from Telegram."""
        if not self.token:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self.last_update_id + 1,
                        "timeout": 10
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                updates = data.get("result", [])
                
                if updates:
                    self.last_update_id = updates[-1]["update_id"]
                
                return updates
        except Exception as e:
            logger.error(f"Error getting Telegram updates: {e}")
            return []
    
    async def process_commands(self) -> List[Dict[str, Any]]:
        """
        Process incoming commands.
        
        Returns:
            List of processed commands with their results
        """
        updates = await self.get_updates()
        results = []
        
        for update in updates:
            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = message.get("chat", {}).get("id")
            
            # Only process messages from our chat
            if str(chat_id) != str(self.chat_id):
                continue
            
            if text.startswith("/"):
                result = await self._handle_command(text)
                results.append(result)
        
        return results
    
    async def _handle_command(self, command_text: str) -> Dict[str, Any]:
        """Handle a command."""
        parts = command_text.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command == "/status":
            return {"command": "status", "args": args}
        
        elif command == "/pending":
            return {"command": "pending", "args": args}
        
        elif command == "/approve":
            if args:
                trade_id = args[0]
                return {"command": "approve", "trade_id": trade_id}
            else:
                await self.send_message("Usage: `/approve [trade_id]`")
                return {"command": "approve", "error": "No trade ID provided"}
        
        elif command == "/reject":
            if args:
                trade_id = args[0]
                reason = " ".join(args[1:]) if len(args) > 1 else "Rejected by user"
                return {"command": "reject", "trade_id": trade_id, "reason": reason}
            else:
                await self.send_message("Usage: `/reject [trade_id] [reason]`")
                return {"command": "reject", "error": "No trade ID provided"}
        
        elif command == "/stop":
            return {"command": "stop", "args": args}
        
        elif command == "/resume":
            return {"command": "resume", "args": args}
        
        elif command == "/help":
            await self._send_help()
            return {"command": "help", "args": args}
        
        else:
            await self.send_message(f"Unknown command: {command}\nUse /help for available commands.")
            return {"command": "unknown", "text": command_text}
    
    async def _send_help(self) -> None:
        """Send help message."""
        message = """
🤖 **GANN SENTINEL COMMANDS**

**Trading:**
`/approve [id]` - Approve pending trade
`/reject [id]` - Reject pending trade
`/pending` - List pending trades

**Status:**
`/status` - System status
`/help` - This help message

**Emergency:**
`/stop` - HALT all trading
`/resume` - Resume trading
"""
        await self.send_message(message)
    
    def get_pending_approval(self, short_id: str) -> Optional[Dict[str, Any]]:
        """Get a pending approval by short ID."""
        return self.pending_approvals.get(short_id)
    
    def remove_pending_approval(self, short_id: str) -> bool:
        """Remove a pending approval."""
        if short_id in self.pending_approvals:
            del self.pending_approvals[short_id]
            return True
        return False
