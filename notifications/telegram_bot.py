"""
Telegram Bot for Gann Sentinel Trader
Handles notifications and command processing for trade approvals and system control.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class CommandType(Enum):
    STATUS = "status"
    PENDING = "pending"
    APPROVE = "approve"
    REJECT = "reject"
    SCAN = "scan"
    STOP = "stop"
    RESUME = "resume"
    POSITIONS = "positions"
    HISTORY = "history"
    ERRORS = "errors"
    HELP = "help"


@dataclass
class TelegramCommand:
    """Represents a parsed Telegram command."""
    command: CommandType
    args: List[str]
    chat_id: int
    message_id: int
    timestamp: datetime


class TelegramBot:
    """
    Telegram bot for Gann Sentinel Trader notifications and commands.
    
    Responsibilities:
    - Send trade recommendation notifications
    - Process approval/rejection commands
    - Provide system status updates
    - Handle emergency stop/resume
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        database = None,
        risk_engine = None,
        alpaca_executor = None
    ):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.database = database
        self.risk_engine = risk_engine
        self.alpaca_executor = alpaca_executor
        
        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN not set - notifications disabled")
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set - notifications disabled")
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self._command_handlers: Dict[CommandType, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Register command handlers."""
        self._command_handlers = {
            CommandType.STATUS: self._handle_status,
            CommandType.PENDING: self._handle_pending,
            CommandType.APPROVE: self._handle_approve,
            CommandType.REJECT: self._handle_reject,
            CommandType.SCAN: self._handle_scan,
            CommandType.STOP: self._handle_stop,
            CommandType.RESUME: self._handle_resume,
            CommandType.POSITIONS: self._handle_positions,
            CommandType.HISTORY: self._handle_history,
            CommandType.ERRORS: self._handle_errors,
            CommandType.HELP: self._handle_help,
        }
    
    @property
    def is_configured(self) -> bool:
        """Check if bot is properly configured."""
        return bool(self.token and self.chat_id)
    
    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
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
    
    def _parse_command(self, message: Dict[str, Any]) -> Optional[TelegramCommand]:
        """Parse a Telegram message into a command."""
        text = message.get("text", "")
        if not text.startswith("/"):
            return None
        
        parts = text.split()
        cmd_text = parts[0][1:].lower()  # Remove leading /
        args = parts[1:] if len(parts) > 1 else []
        
        # Handle commands with @botname suffix
        if "@" in cmd_text:
            cmd_text = cmd_text.split("@")[0]
        
        try:
            command_type = CommandType(cmd_text)
        except ValueError:
            logger.debug(f"Unknown command: {cmd_text}")
            return None
        
        return TelegramCommand(
            command=command_type,
            args=args,
            chat_id=message["chat"]["id"],
            message_id=message["message_id"],
            timestamp=datetime.now(timezone.utc)
        )
    
    async def process_commands(self) -> List[TelegramCommand]:
        """
        Fetch and process any pending Telegram commands.
        Returns list of processed commands.
        """
        processed = []
        updates = await self.get_updates()
        
        for update in updates:
            message = update.get("message")
            if not message:
                continue
            
            # Only process messages from our chat
            if str(message.get("chat", {}).get("id")) != str(self.chat_id):
                logger.debug(f"Ignoring message from chat {message.get('chat', {}).get('id')}")
                continue
            
            command = self._parse_command(message)
            if command:
                await self._execute_command(command)
                processed.append(command)
        
        return processed
    
    async def _execute_command(self, command: TelegramCommand) -> None:
        """Execute a parsed command."""
        handler = self._command_handlers.get(command.command)
        if handler:
            try:
                await handler(command)
            except Exception as e:
                logger.error(f"Error executing command {command.command}: {e}")
                await self.send_message(f"⚠️ Error executing command: {str(e)}")
        else:
            await self.send_message(f"Unknown command: {command.command.value}")
    
    # === Command Handlers ===
    
    async def _handle_status(self, command: TelegramCommand) -> None:
        """Handle /status command."""
        try:
            status_parts = ["📊 <b>GANN SENTINEL STATUS</b>\n"]
            
            # System status
            mode = os.getenv("MODE", "PAPER")
            approval_gate = os.getenv("APPROVAL_GATE", "ON")
            status_parts.append(f"Mode: {mode}")
            status_parts.append(f"Approval Gate: {approval_gate}")
            
            # Risk engine status
            if self.risk_engine:
                is_halted = getattr(self.risk_engine, 'is_halted', False)
                status_parts.append(f"Trading: {'🔴 HALTED' if is_halted else '🟢 ACTIVE'}")
            
            # Portfolio summary
            if self.database:
                snapshot = self.database.get_latest_portfolio_snapshot()
                if snapshot:
                    status_parts.append(f"\n💰 <b>Portfolio</b>")
                    status_parts.append(f"Total Value: ${snapshot.get('total_value', 0):,.2f}")
                    status_parts.append(f"Cash: ${snapshot.get('cash', 0):,.2f}")
                    daily_pnl = snapshot.get('daily_pnl', 0)
                    daily_pnl_pct = snapshot.get('daily_pnl_pct', 0)
                    pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
                    status_parts.append(f"Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
            
            # Pending trades
            if self.database:
                pending = self.database.get_pending_trades()
                status_parts.append(f"\n⏳ Pending Approvals: {len(pending)}")
            
            await self.send_message("\n".join(status_parts))
            
        except Exception as e:
            logger.error(f"Error in status handler: {e}")
            await self.send_message(f"⚠️ Error fetching status: {str(e)}")
    
    async def _handle_pending(self, command: TelegramCommand) -> None:
        """Handle /pending command."""
        try:
            if not self.database:
                await self.send_message("Database not connected")
                return
            
            pending = self.database.get_pending_trades()
            
            if not pending:
                await self.send_message("✅ No pending trade approvals")
                return
            
            msg_parts = ["⏳ <b>PENDING APPROVALS</b>\n"]
            
            for trade in pending:
                trade_id = trade.get("id", "unknown")[:8]
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                conviction = trade.get("conviction_score", "N/A")
                thesis = trade.get("thesis", "No thesis")[:100]
                
                msg_parts.append(f"\n<b>{side} {ticker}</b> (ID: {trade_id})")
                msg_parts.append(f"Conviction: {conviction}/100")
                msg_parts.append(f"Thesis: {thesis}...")
                msg_parts.append(f"/approve {trade_id}  |  /reject {trade_id}")
            
            await self.send_message("\n".join(msg_parts))
            
        except Exception as e:
            logger.error(f"Error in pending handler: {e}")
            await self.send_message(f"⚠️ Error fetching pending: {str(e)}")
    
    async def _handle_approve(self, command: TelegramCommand) -> None:
        """Handle /approve <id> command."""
        if not command.args:
            await self.send_message("Usage: /approve <trade_id>")
            return
        
        trade_id = command.args[0]
        
        try:
            if self.database:
                # Find trade by partial ID match
                trade = self.database.get_trade_by_partial_id(trade_id)
                
                if not trade:
                    await self.send_message(f"❌ Trade not found: {trade_id}")
                    return
                
                if trade.get("status") != "pending_approval":
                    await self.send_message(f"❌ Trade {trade_id} is not pending (status: {trade.get('status')})")
                    return
                
                # Update status to approved
                self.database.update_trade_status(trade["id"], "approved")
                
                await self.send_message(
                    f"✅ <b>APPROVED</b>\n\n"
                    f"Trade: {trade.get('side', '').upper()} {trade.get('ticker', 'N/A')}\n"
                    f"ID: {trade['id'][:8]}\n\n"
                    f"Trade will execute on next cycle."
                )
            else:
                await self.send_message("Database not connected")
                
        except Exception as e:
            logger.error(f"Error approving trade: {e}")
            await self.send_message(f"⚠️ Error approving trade: {str(e)}")
    
    async def _handle_reject(self, command: TelegramCommand) -> None:
        """Handle /reject <id> command."""
        if not command.args:
            await self.send_message("Usage: /reject <trade_id>")
            return
        
        trade_id = command.args[0]
        
        try:
            if self.database:
                trade = self.database.get_trade_by_partial_id(trade_id)
                
                if not trade:
                    await self.send_message(f"❌ Trade not found: {trade_id}")
                    return
                
                self.database.update_trade_status(trade["id"], "rejected")
                
                await self.send_message(
                    f"🚫 <b>REJECTED</b>\n\n"
                    f"Trade: {trade.get('side', '').upper()} {trade.get('ticker', 'N/A')}\n"
                    f"ID: {trade['id'][:8]}"
                )
            else:
                await self.send_message("Database not connected")
                
        except Exception as e:
            logger.error(f"Error rejecting trade: {e}")
            await self.send_message(f"⚠️ Error rejecting trade: {str(e)}")
    
    async def _handle_scan(self, command: TelegramCommand) -> None:
        """Handle /scan command - triggers manual scan cycle."""
        await self.send_message("🔍 Manual scan triggered. Running analysis cycle...")
        # The actual scan is handled by the agent loop which checks for this flag
        # We just acknowledge here - agent.py should check for scan requests
    
    async def _handle_stop(self, command: TelegramCommand) -> None:
        """Handle /stop command - emergency halt."""
        try:
            if self.risk_engine:
                self.risk_engine.halt_trading("Manual stop via Telegram")
            
            # Try to cancel open orders
            if self.alpaca_executor:
                try:
                    cancelled = await self.alpaca_executor.cancel_all_orders()
                    await self.send_message(
                        f"🛑 <b>TRADING HALTED</b>\n\n"
                        f"All trading suspended.\n"
                        f"Open orders cancelled: {cancelled}\n\n"
                        f"Use /resume to restart."
                    )
                except Exception as e:
                    await self.send_message(
                        f"🛑 <b>TRADING HALTED</b>\n\n"
                        f"Warning: Could not cancel orders: {str(e)}\n\n"
                        f"Use /resume to restart."
                    )
            else:
                await self.send_message(
                    "🛑 <b>TRADING HALTED</b>\n\n"
                    "All trading suspended.\n\n"
                    "Use /resume to restart."
                )
                
        except Exception as e:
            logger.error(f"Error in stop handler: {e}")
            await self.send_message(f"⚠️ Error halting: {str(e)}")
    
    async def _handle_resume(self, command: TelegramCommand) -> None:
        """Handle /resume command - resume trading after halt."""
        try:
            if self.risk_engine:
                self.risk_engine.resume_trading()
            
            await self.send_message(
                "🟢 <b>TRADING RESUMED</b>\n\n"
                "System is now active and accepting trades."
            )
            
        except Exception as e:
            logger.error(f"Error in resume handler: {e}")
            await self.send_message(f"⚠️ Error resuming: {str(e)}")
    
    async def _handle_positions(self, command: TelegramCommand) -> None:
        """Handle /positions command."""
        try:
            if not self.database:
                await self.send_message("Database not connected")
                return
            
            positions = self.database.get_all_positions()
            
            if not positions:
                await self.send_message("📭 No open positions")
                return
            
            msg_parts = ["📈 <b>OPEN POSITIONS</b>\n"]
            
            total_value = 0
            total_pnl = 0
            
            for pos in positions:
                ticker = pos.get("ticker", "N/A")
                qty = pos.get("quantity", 0)
                entry = pos.get("avg_entry_price", 0)
                current = pos.get("current_price", entry)
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                market_value = pos.get("market_value", 0)
                
                total_value += market_value
                total_pnl += pnl
                
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                
                msg_parts.append(f"\n<b>{ticker}</b>")
                msg_parts.append(f"Qty: {qty} @ ${entry:.2f}")
                msg_parts.append(f"Current: ${current:.2f}")
                msg_parts.append(f"P&L: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                
                if pos.get("stop_loss_price"):
                    msg_parts.append(f"Stop: ${pos['stop_loss_price']:.2f}")
            
            total_pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
            msg_parts.append(f"\n<b>Total Value:</b> ${total_value:,.2f}")
            msg_parts.append(f"<b>Total P&L:</b> {total_pnl_emoji} ${total_pnl:,.2f}")
            
            await self.send_message("\n".join(msg_parts))
            
        except Exception as e:
            logger.error(f"Error in positions handler: {e}")
            await self.send_message(f"⚠️ Error fetching positions: {str(e)}")
    
    async def _handle_history(self, command: TelegramCommand) -> None:
        """Handle /history <n> command."""
        try:
            n = int(command.args[0]) if command.args else 5
            n = min(n, 20)  # Cap at 20
            
            if not self.database:
                await self.send_message("Database not connected")
                return
            
            trades = self.database.get_recent_trades(n)
            
            if not trades:
                await self.send_message("📭 No trade history")
                return
            
            msg_parts = [f"📜 <b>LAST {len(trades)} TRADES</b>\n"]
            
            for trade in trades:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                status = trade.get("status", "unknown")
                fill_price = trade.get("fill_price")
                created = trade.get("created_at", "")[:10]
                
                status_emoji = {
                    "filled": "✅",
                    "rejected": "🚫",
                    "cancelled": "❌",
                    "pending_approval": "⏳",
                    "approved": "🟡",
                    "submitted": "📤"
                }.get(status, "❓")
                
                price_str = f" @ ${fill_price:.2f}" if fill_price else ""
                msg_parts.append(f"\n{status_emoji} {side} {ticker}{price_str}")
                msg_parts.append(f"   Status: {status} | {created}")
            
            await self.send_message("\n".join(msg_parts))
            
        except ValueError:
            await self.send_message("Usage: /history <number>")
        except Exception as e:
            logger.error(f"Error in history handler: {e}")
            await self.send_message(f"⚠️ Error fetching history: {str(e)}")
    
    async def _handle_errors(self, command: TelegramCommand) -> None:
        """Handle /errors command."""
        try:
            if not self.database:
                await self.send_message("Database not connected")
                return
            
            errors = self.database.get_recent_errors(5)
            
            if not errors:
                await self.send_message("✅ No recent errors")
                return
            
            msg_parts = ["⚠️ <b>RECENT ERRORS</b>\n"]
            
            for err in errors:
                component = err.get("component", "unknown")
                error_type = err.get("error_type", "unknown")
                message = err.get("message", "No message")[:100]
                created = err.get("created_at", "")[:19]
                
                msg_parts.append(f"\n<b>[{component}]</b> {error_type}")
                msg_parts.append(f"{message}")
                msg_parts.append(f"<i>{created}</i>")
            
            await self.send_message("\n".join(msg_parts))
            
        except Exception as e:
            logger.error(f"Error in errors handler: {e}")
            await self.send_message(f"⚠️ Error fetching errors: {str(e)}")
    
    async def _handle_help(self, command: TelegramCommand) -> None:
        """Handle /help command."""
        help_text = """
🤖 <b>GANN SENTINEL COMMANDS</b>

<b>Status & Info</b>
/status - Portfolio & system status
/positions - Open positions detail
/pending - Pending trade approvals
/history [n] - Last n trades (default 5)
/errors - Recent system errors

<b>Trade Control</b>
/approve [id] - Approve a pending trade
/reject [id] - Reject a pending trade
/scan - Trigger manual analysis cycle

<b>System Control</b>
/stop - Emergency halt all trading
/resume - Resume after halt
/help - Show this message
"""
        await self.send_message(help_text)
    
    # === Notification Methods ===
    
    async def notify_trade_recommendation(self, analysis: Dict[str, Any], trade_id: str) -> bool:
        """Send trade recommendation for approval."""
        try:
            ticker = analysis.get("ticker", "N/A")
            side = analysis.get("recommendation", "N/A")
            conviction = analysis.get("conviction_score", 0)
            thesis = analysis.get("thesis", "No thesis provided")
            bull_case = analysis.get("bull_case", "N/A")
            bear_case = analysis.get("bear_case", "N/A")
            position_size = analysis.get("position_size_pct", 0) * 100
            stop_loss = analysis.get("stop_loss_pct", 0) * 100
            time_horizon = analysis.get("time_horizon", "unknown")
            
            short_id = trade_id[:8]
            
            message = f"""
🔔 <b>TRADE RECOMMENDATION</b>

<b>Ticker:</b> {ticker}
<b>Action:</b> {side}
<b>Conviction:</b> {conviction}/100
<b>Position Size:</b> {position_size:.0f}%

📈 <b>THESIS</b>
{thesis}

✅ <b>BULL CASE</b>
{bull_case}

⚠️ <b>BEAR CASE</b>
{bear_case}

⏰ Time Horizon: {time_horizon}
🛑 Stop Loss: {stop_loss:.0f}%

/approve {short_id}
/reject {short_id}
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trade recommendation: {e}")
            return False
    
    async def notify_trade_executed(self, trade: Dict[str, Any]) -> bool:
        """Send notification when trade is executed."""
        try:
            ticker = trade.get("ticker", "N/A")
            side = trade.get("side", "N/A").upper()
            quantity = trade.get("fill_quantity", trade.get("quantity", 0))
            price = trade.get("fill_price", "N/A")
            thesis = trade.get("thesis", "")[:100]
            
            message = f"""
✅ <b>TRADE EXECUTED</b>

<b>{side} {ticker}</b>
Quantity: {quantity}
Fill Price: ${price}

{thesis}
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending execution notification: {e}")
            return False
    
    async def notify_stop_loss_triggered(self, position: Dict[str, Any], exit_price: float) -> bool:
        """Send notification when stop loss is triggered."""
        try:
            ticker = position.get("ticker", "N/A")
            entry = position.get("avg_entry_price", 0)
            quantity = position.get("quantity", 0)
            pnl = (exit_price - entry) * quantity
            pnl_pct = ((exit_price / entry) - 1) * 100 if entry > 0 else 0
            
            message = f"""
🛑 <b>STOP LOSS TRIGGERED</b>

<b>{ticker}</b>
Entry: ${entry:.2f}
Exit: ${exit_price:.2f}
P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending stop loss notification: {e}")
            return False
    
    async def notify_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send daily portfolio summary."""
        try:
            total_value = summary.get("total_value", 0)
            daily_pnl = summary.get("daily_pnl", 0)
            daily_pnl_pct = summary.get("daily_pnl_pct", 0)
            positions_count = summary.get("positions_count", 0)
            trades_today = summary.get("trades_today", 0)
            signals_processed = summary.get("signals_processed", 0)
            
            pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
            
            message = f"""
📊 <b>DAILY SUMMARY</b>

💰 Portfolio Value: ${total_value:,.2f}
{pnl_emoji} Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)

📈 Open Positions: {positions_count}
🔄 Trades Today: {trades_today}
📡 Signals Processed: {signals_processed}
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    async def notify_error(self, component: str, error: str) -> bool:
        """Send error notification."""
        try:
            message = f"""
⚠️ <b>ERROR: {component}</b>

{error[:500]}
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
            return False
    
    async def notify_circuit_breaker(self, breaker: str, reason: str, reset_time: str) -> bool:
        """Send circuit breaker notification."""
        try:
            message = f"""
🚨 <b>CIRCUIT BREAKER TRIGGERED</b>

<b>Breaker:</b> {breaker}
<b>Reason:</b> {reason}
<b>Reset:</b> {reset_time}

Trading is paused. Use /resume to override (with caution).
"""
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending circuit breaker notification: {e}")
            return False


# Convenience function for quick notifications
async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
