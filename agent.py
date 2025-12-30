"""
Gann Sentinel Trader - Main Agent
Orchestrates the trading system: scan signals, analyze, approve, execute.

DISCLAIMER: Trading involves substantial risk of loss. This is an experimental
system and nothing here constitutes financial advice. Only trade what you can
afford to lose.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from config import Config
from storage.database import Database
from scanners.grok_scanner import GrokScanner
from scanners.fred_scanner import FREDScanner
from scanners.polymarket_scanner import PolymarketScanner
from analyzers.claude_analyst import ClaudeAnalyst
from executors.risk_engine import RiskEngine
from executors.alpaca_executor import AlpacaExecutor
from notifications.telegram_bot import TelegramBot
from models.signals import Signal
from models.analysis import Analysis, Recommendation
from models.trades import Trade, TradeStatus, OrderType, OrderSide, Position

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_PATH / "agent.log")
    ]
)
logger = logging.getLogger(__name__)


class GannSentinelAgent:
    """
    Main trading agent that orchestrates all components.
    """
    
    def __init__(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("INITIALIZING GANN SENTINEL TRADER")
        logger.info("=" * 60)
        
        # Validate configuration
        validation = Config.validate()
        if not validation["valid"]:
            for issue in validation["issues"]:
                logger.error(f"Config issue: {issue}")
            raise ValueError("Invalid configuration - see logs for details")
        
        Config.print_config()
        
        # Initialize components
        self.db = Database()
        self.grok = GrokScanner()
        self.fred = FREDScanner()
        self.polymarket = PolymarketScanner()
        self.analyst = ClaudeAnalyst()
        self.risk_engine = RiskEngine()
        self.executor = AlpacaExecutor()
        self.telegram = TelegramBot()
        
        # Agent state
        self.running = False
        self.last_scan_time: Optional[datetime] = None
        self.watchlist: List[str] = []
        
        # Default watchlist (can be customized)
        self.default_watchlist = [
            "SPY", "QQQ", "IWM",  # Index ETFs
            "NVDA", "AMD", "SMCI",  # Semiconductors
            "TSLA", "RKLB",  # EV/Space
            "AAPL", "MSFT", "GOOGL",  # Big tech
            "COIN", "MSTR",  # Crypto-adjacent
        ]
        
        logger.info("Agent initialized successfully")
    
    async def start(self) -> None:
        """Start the trading agent."""
        logger.info("Starting Gann Sentinel Agent...")
        
        # Send startup notification
        await self.telegram.send_message(
            "🚀 **Gann Sentinel Trader Started**\n\n"
            f"Mode: {Config.MODE}\n"
            f"Approval Gate: {'ON' if Config.APPROVAL_GATE else 'OFF'}\n\n"
            "Use /help for commands"
        )
        
        self.running = True
        self.watchlist = self.default_watchlist.copy()
        
        # Main loop
        while self.running:
            try:
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Error in main cycle: {e}")
                logger.error(traceback.format_exc())
                self.db.log_error(
                    error_type="cycle_error",
                    component="agent",
                    message=str(e),
                    stack_trace=traceback.format_exc()
                )
                await self.telegram.send_error_alert("Agent", str(e))
            
            # Wait before next cycle
            await asyncio.sleep(60)  # Check every minute
    
    async def stop(self) -> None:
        """Stop the trading agent."""
        logger.info("Stopping Gann Sentinel Agent...")
        self.running = False
        
        await self.telegram.send_message(
            "🛑 **Gann Sentinel Trader Stopped**\n\n"
            "Trading halted. Use /resume to restart."
        )
    
    async def _run_cycle(self) -> None:
        """Run one cycle of the trading loop."""
        now = datetime.now(timezone.utc)
        
        # Process any pending Telegram commands
        await self._process_commands()
        
        # Check if it's time for a full scan
        should_scan = (
            self.last_scan_time is None or
            (now - self.last_scan_time).total_seconds() >= Config.SCAN_INTERVAL_MINUTES * 60
        )
        
        if should_scan:
            logger.info("Running full scan cycle...")
            await self._full_scan_cycle()
            self.last_scan_time = now
        
        # Always check positions for stop-loss triggers
        await self._check_positions()
        
        # Process any approved trades
        await self._process_approved_trades()
    
    async def _full_scan_cycle(self) -> None:
        """Run a full signal scan and analysis cycle."""
        signals: List[Signal] = []
        
        # 1. Gather signals from all sources
        logger.info("Gathering signals...")
        
        # Grok sentiment scan
        try:
            sentiment_signals = await self.grok.scan_sentiment(self.watchlist[:5])  # Limit to save API calls
            signals.extend(sentiment_signals)
            logger.info(f"Got {len(sentiment_signals)} sentiment signals from Grok")
        except Exception as e:
            logger.error(f"Error in Grok sentiment scan: {e}")
            self.db.log_error("scan_error", "grok_sentiment", str(e))
        
        # Grok market overview
        try:
            overview_signals = await self.grok.scan_market_overview()
            signals.extend(overview_signals)
            logger.info(f"Got {len(overview_signals)} overview signals from Grok")
        except Exception as e:
            logger.error(f"Error in Grok overview scan: {e}")
            self.db.log_error("scan_error", "grok_overview", str(e))
        
        # FRED macro data
        try:
            macro_signals = await self.fred.scan_all_series()
            signals.extend(macro_signals)
            logger.info(f"Got {len(macro_signals)} macro signals from FRED")
        except Exception as e:
            logger.error(f"Error in FRED scan: {e}")
            self.db.log_error("scan_error", "fred", str(e))
        
        # Polymarket predictions
        try:
            prediction_signals = await self.polymarket.scan_all_markets()
            signals.extend(prediction_signals)
            logger.info(f"Got {len(prediction_signals)} prediction signals from Polymarket")
        except Exception as e:
            logger.error(f"Error in Polymarket scan: {e}")
            self.db.log_error("scan_error", "polymarket", str(e))
        
        # 2. Save all signals
        for signal in signals:
            try:
                self.db.save_signal(signal.to_dict())
            except Exception as e:
                logger.error(f"Error saving signal: {e}")
        
        logger.info(f"Total signals gathered: {len(signals)}")
        
        if not signals:
            logger.warning("No signals gathered - skipping analysis")
            return
        
        # 3. Get portfolio context
        portfolio = await self.executor.get_portfolio_snapshot()
        positions = await self.executor.get_positions()
        
        # Save portfolio snapshot
        self.db.save_snapshot(portfolio.to_dict())
        
        # 4. Run Claude analysis
        logger.info("Running Claude analysis...")
        try:
            analysis = await self.analyst.analyze_signals(
                signals=signals,
                portfolio_context=portfolio.to_dict(),
                watchlist=self.watchlist
            )
            
            # Save analysis
            self.db.save_analysis(analysis.to_dict())
            
            # 5. Check if we have an actionable trade
            if analysis.is_actionable:
                logger.info(f"Actionable trade identified: {analysis.ticker} - {analysis.recommendation.value}")
                await self._handle_trade_recommendation(analysis, portfolio, positions)
            else:
                logger.info(f"No actionable trade. Conviction: {analysis.conviction_score}")
                
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}")
            self.db.log_error("analysis_error", "claude", str(e), traceback.format_exc())
    
    async def _handle_trade_recommendation(
        self,
        analysis: Analysis,
        portfolio,
        positions: List[Position]
    ) -> None:
        """Handle a trade recommendation from Claude."""
        
        # 1. Run risk checks
        passed, risk_results = self.risk_engine.validate_trade(
            analysis=analysis,
            portfolio=portfolio,
            current_positions=positions
        )
        
        if not passed:
            logger.warning(f"Trade rejected by risk engine: {analysis.ticker}")
            failed_checks = [r for r in risk_results if not r.passed and r.severity == "error"]
            reasons = "; ".join([r.message for r in failed_checks])
            
            await self.telegram.send_message(
                f"⚠️ **Trade Rejected by Risk Engine**\n\n"
                f"Ticker: {analysis.ticker}\n"
                f"Reason: {reasons}"
            )
            return
        
        # 2. Get current price
        quote = await self.executor.get_quote(analysis.ticker)
        if "error" in quote:
            logger.error(f"Could not get quote for {analysis.ticker}: {quote['error']}")
            return
        
        current_price = quote["mid"]
        
        # 3. Calculate position size
        sizing = self.risk_engine.calculate_position_size(
            analysis=analysis,
            portfolio=portfolio,
            current_price=current_price
        )
        
        if sizing["shares"] < 1:
            logger.warning(f"Position size too small: {sizing['shares']} shares")
            return
        
        # 4. Create trade object
        trade = Trade(
            analysis_id=analysis.analysis_id,
            ticker=analysis.ticker,
            side=OrderSide.BUY if analysis.recommendation == Recommendation.BUY else OrderSide.SELL,
            quantity=sizing["shares"],
            order_type=OrderType.MARKET if analysis.entry_strategy == "market" else OrderType.LIMIT,
            limit_price=analysis.entry_price_target,
            status=TradeStatus.PENDING_APPROVAL if Config.APPROVAL_GATE else TradeStatus.APPROVED,
            thesis=analysis.thesis,
            conviction_score=analysis.conviction_score
        )
        
        # 5. Save trade
        self.db.save_trade(trade.to_dict())
        
        # 6. Handle based on approval gate setting
        if Config.APPROVAL_GATE:
            # Send for approval
            await self.telegram.send_trade_alert(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                side=trade.side.value,
                quantity=trade.quantity,
                conviction=trade.conviction_score,
                thesis=trade.thesis
            )
            logger.info(f"Trade sent for approval: {trade.trade_id[:8]}")
        else:
            # Auto-execute
            await self._execute_trade(trade)
    
    async def _execute_trade(self, trade: Trade) -> None:
        """Execute a trade."""
        logger.info(f"Executing trade: {trade.ticker} {trade.side.value} {trade.quantity}")
        
        try:
            # Submit to Alpaca
            trade = await self.executor.submit_order(trade)
            
            # Update in database
            self.db.save_trade(trade.to_dict())
            
            if trade.status == TradeStatus.FILLED:
                # Send success notification
                await self.telegram.send_execution_alert(
                    ticker=trade.ticker,
                    side=trade.side.value,
                    quantity=trade.fill_quantity or trade.quantity,
                    price=trade.fill_price or 0,
                    total=(trade.fill_price or 0) * (trade.fill_quantity or trade.quantity)
                )
                
                logger.info(f"Trade executed: {trade.ticker} @ ${trade.fill_price:.2f}")
            
            elif trade.status == TradeStatus.FAILED:
                await self.telegram.send_error_alert(
                    "Trade Execution",
                    f"Failed to execute {trade.ticker}: {trade.rejection_reason}"
                )
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            trade.status = TradeStatus.FAILED
            trade.rejection_reason = str(e)
            self.db.save_trade(trade.to_dict())
            await self.telegram.send_error_alert("Trade Execution", str(e))
    
    async def _check_positions(self) -> None:
        """Check positions for stop-loss triggers."""
        positions = await self.executor.get_positions()
        
        for position in positions:
            # Get stop loss from database if stored
            db_position = self.db.get_position(position.ticker)
            
            if db_position and db_position.get("stop_loss_price"):
                stop_loss = db_position["stop_loss_price"]
                
                if position.current_price and position.current_price <= stop_loss:
                    logger.warning(f"STOP LOSS TRIGGERED: {position.ticker} @ ${position.current_price}")
                    
                    # Calculate loss percentage
                    entry_price = db_position.get("avg_entry_price", position.avg_entry_price)
                    loss_pct = (position.current_price - entry_price) / entry_price * 100
                    
                    # Send alert
                    await self.telegram.send_stop_loss_alert(
                        ticker=position.ticker,
                        trigger_price=position.current_price,
                        loss_pct=loss_pct
                    )
                    
                    # Close position
                    if Config.MODE == "LIVE" or self.executor.is_paper:
                        await self.executor.close_position(position.ticker)
                        logger.info(f"Position closed: {position.ticker}")
    
    async def _process_approved_trades(self) -> None:
        """Process any trades that have been approved."""
        pending_trades = self.db.get_pending_trades()
        
        for trade_data in pending_trades:
            if trade_data["status"] == "approved":
                # Reconstruct trade object
                trade = Trade(
                    trade_id=trade_data["id"],
                    analysis_id=trade_data.get("analysis_id"),
                    ticker=trade_data["ticker"],
                    side=OrderSide(trade_data["side"]),
                    quantity=trade_data["quantity"],
                    order_type=OrderType(trade_data["order_type"]),
                    limit_price=trade_data.get("limit_price"),
                    status=TradeStatus.APPROVED,
                    thesis=trade_data.get("thesis", ""),
                    conviction_score=trade_data.get("conviction_score", 0)
                )
                
                await self._execute_trade(trade)
    
    async def _process_commands(self) -> None:
        """Process Telegram commands."""
        commands = await self.telegram.process_commands()
        
        for cmd in commands:
            command = cmd.get("command")
            
            if command == "status":
                await self._handle_status_command()
            
            elif command == "pending":
                await self._handle_pending_command()
            
            elif command == "approve":
                await self._handle_approve_command(cmd.get("trade_id"))
            
            elif command == "reject":
                await self._handle_reject_command(
                    cmd.get("trade_id"),
                    cmd.get("reason", "Rejected by user")
                )
            
            elif command == "stop":
                await self._handle_stop_command()
            
            elif command == "resume":
                await self._handle_resume_command()
    
    async def _handle_status_command(self) -> None:
        """Handle /status command."""
        portfolio = await self.executor.get_portfolio_snapshot()
        pending_count = len(self.db.get_pending_trades())
        
        await self.telegram.send_system_status(
            status="Running" if self.running else "Stopped",
            mode=Config.MODE,
            approval_gate=Config.APPROVAL_GATE,
            positions_count=portfolio.position_count,
            pending_trades=pending_count
        )
        
        # Also send portfolio status
        await self.telegram.send_message(portfolio.to_telegram_message())
    
    async def _handle_pending_command(self) -> None:
        """Handle /pending command."""
        pending = self.db.get_pending_trades()
        
        if not pending:
            await self.telegram.send_message("No pending trades.")
            return
        
        msg = "**Pending Trades:**\n\n"
        for trade in pending:
            msg += f"• `{trade['id'][:8]}` - {trade['side'].upper()} {trade['ticker']} ({trade['quantity']} shares)\n"
        
        await self.telegram.send_message(msg)
    
    async def _handle_approve_command(self, trade_id: str) -> None:
        """Handle /approve command."""
        if not trade_id:
            await self.telegram.send_message("Usage: `/approve [trade_id]`")
            return
        
        # Find trade
        trade_data = self.db.get_trade(trade_id)
        
        if not trade_data:
            await self.telegram.send_message(f"Trade not found: {trade_id}")
            return
        
        if trade_data["status"] != "pending_approval":
            await self.telegram.send_message(f"Trade {trade_id} is not pending approval (status: {trade_data['status']})")
            return
        
        # Approve
        self.db.update_trade_status(trade_id, "approved")
        await self.telegram.send_message(f"✅ Trade approved: {trade_id[:8]}")
        
        # Remove from pending approvals
        self.telegram.remove_pending_approval(trade_id[:8])
        
        logger.info(f"Trade approved: {trade_id}")
    
    async def _handle_reject_command(self, trade_id: str, reason: str) -> None:
        """Handle /reject command."""
        if not trade_id:
            await self.telegram.send_message("Usage: `/reject [trade_id] [reason]`")
            return
        
        # Find trade
        trade_data = self.db.get_trade(trade_id)
        
        if not trade_data:
            await self.telegram.send_message(f"Trade not found: {trade_id}")
            return
        
        # Reject
        self.db.update_trade_status(trade_id, "rejected", rejection_reason=reason)
        await self.telegram.send_message(f"❌ Trade rejected: {trade_id[:8]}\nReason: {reason}")
        
        # Remove from pending approvals
        self.telegram.remove_pending_approval(trade_id[:8])
        
        logger.info(f"Trade rejected: {trade_id} - {reason}")
    
    async def _handle_stop_command(self) -> None:
        """Handle /stop command - emergency halt."""
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        # Halt risk engine
        self.risk_engine.halt_trading("Emergency stop triggered via Telegram")
        
        # Cancel all open orders
        cancelled = await self.executor.cancel_all_orders()
        
        await self.telegram.send_message(
            f"🛑 **EMERGENCY STOP**\n\n"
            f"Trading halted.\n"
            f"Cancelled {cancelled} orders.\n\n"
            f"Use /resume to restart."
        )
    
    async def _handle_resume_command(self) -> None:
        """Handle /resume command."""
        self.risk_engine.resume_trading()
        
        await self.telegram.send_message(
            "✅ **Trading Resumed**\n\n"
            "System is active again."
        )
        
        logger.info("Trading resumed via Telegram command")


async def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗  █████╗ ███╗   ██╗███╗   ██╗                       ║
    ║  ██╔════╝ ██╔══██╗████╗  ██║████╗  ██║                       ║
    ║  ██║  ███╗███████║██╔██╗ ██║██╔██╗ ██║                       ║
    ║  ██║   ██║██╔══██║██║╚██╗██║██║╚██╗██║                       ║
    ║  ╚██████╔╝██║  ██║██║ ╚████║██║ ╚████║                       ║
    ║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝                       ║
    ║                                                               ║
    ║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗ ║
    ║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║ ║
    ║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║ ║
    ║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║ ║
    ║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███╗║
    ║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══╝║
    ║                                                               ║
    ║               AUTONOMOUS TRADING AGENT                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    DISCLAIMER: Trading involves substantial risk of loss. This is an 
    experimental system. Nothing here constitutes financial advice.
    Only trade what you can afford to lose.
    
    """)
    
    try:
        agent = GannSentinelAgent()
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
