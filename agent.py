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
        self.telegram = TelegramBot(
            token=Config.TELEGRAM_BOT_TOKEN,
            chat_id=Config.TELEGRAM_CHAT_ID
        )
        
        # Agent state
        self.running = False
        self.last_scan_time: Optional[datetime] = None
        self.watchlist: List[str] = []
        
        # Daily digest scheduling
        self.last_digest_time: Optional[datetime] = None
        self.digest_hour_utc = 21  # 9 PM UTC = 4 PM ET / 1 PM PT
        
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
            "ðŸš€ **Gann Sentinel Trader Started**\n\n"
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
            "ðŸ›‘ **Gann Sentinel Trader Stopped**\n\n"
            "Trading halted. Use /resume to restart."
        )
    
    async def _run_cycle(self) -> None:
        """Run one cycle of the trading loop."""
        now = datetime.now(timezone.utc)
        
        # Process any pending Telegram commands
        await self._process_commands()
        
        # Check if it's time for daily digest
        await self._maybe_send_daily_digest(now)
        
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
    
    async def _maybe_send_daily_digest(self, now: datetime) -> None:
        """Send daily digest if it's time."""
        
        # Check if we should send digest
        should_send = False
        
        if self.last_digest_time is None:
            # First run - check if we're past digest hour today
            if now.hour >= self.digest_hour_utc:
                should_send = True
        else:
            # Check if it's a new day and past digest hour
            if (now.date() > self.last_digest_time.date() and 
                now.hour >= self.digest_hour_utc):
                should_send = True
        
        if should_send:
            try:
                await self._send_daily_digest()
                self.last_digest_time = now
                logger.info("Daily digest sent successfully")
            except Exception as e:
                logger.error(f"Failed to send daily digest: {e}")
                self.telegram.record_system_error("daily_digest", str(e))
    
    async def _send_daily_digest(self) -> None:
        """Generate and send the daily digest."""
        # Get current positions and portfolio from Alpaca
        positions = await self.executor.get_positions()
        portfolio = await self.executor.get_portfolio_snapshot()
        pending = self.db.get_pending_trades()
        
        # Convert to dict format for telegram
        positions_data = [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
        portfolio_data = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        
        # Send digest
        await self.telegram.send_daily_digest(
            positions=positions_data,
            portfolio=portfolio_data,
            pending_approvals=pending
        )
    
    async def _full_scan_cycle(self) -> None:
        """Run a full signal scan and analysis cycle."""
        signals: List[Signal] = []
        
        # Record scan start for digest tracking
        self.telegram.record_scan_start()
        
        # 1. Gather signals from all sources
        logger.info("Gathering signals...")
        
        # Grok sentiment scan
        try:
            sentiment_signals = await self.grok.scan_sentiment(self.watchlist[:5])  # Limit to save API calls
            signals.extend(sentiment_signals)
            logger.info(f"Got {len(sentiment_signals)} sentiment signals from Grok")
            
            # Record for digest
            self.telegram.record_source_query(
                source="Grok X Search",
                query=f"sentiment: {', '.join(self.watchlist[:5])}",
                signals_returned=len(sentiment_signals),
                error=None
            )
            for signal in sentiment_signals:
                self.telegram.record_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
                
        except Exception as e:
            logger.error(f"Error in Grok sentiment scan: {e}")
            self.db.log_error("scan_error", "grok_sentiment", str(e))
            self.telegram.record_source_query(
                source="Grok X Search",
                query=f"sentiment: {', '.join(self.watchlist[:5])}",
                signals_returned=0,
                error=str(type(e).__name__)
            )
        
        # Grok market overview
        try:
            overview_signals = await self.grok.scan_market_overview()
            signals.extend(overview_signals)
            logger.info(f"Got {len(overview_signals)} overview signals from Grok")
            
            # Record for digest
            self.telegram.record_source_query(
                source="Grok Web Search",
                query="market overview",
                signals_returned=len(overview_signals),
                error=None
            )
            for signal in overview_signals:
                self.telegram.record_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
                
        except Exception as e:
            logger.error(f"Error in Grok overview scan: {e}")
            self.db.log_error("scan_error", "grok_overview", str(e))
            self.telegram.record_source_query(
                source="Grok Web Search",
                query="market overview",
                signals_returned=0,
                error=str(type(e).__name__)
            )
        
        # FRED macro data
        try:
            macro_signals = await self.fred.scan_all_series()
            signals.extend(macro_signals)
            logger.info(f"Got {len(macro_signals)} macro signals from FRED")
            
            # Record for digest
            fred_series = ["DGS10", "DGS2", "UNRATE", "CPIAUCSL", "GDP", "FEDFUNDS", "T10Y2Y"]
            self.telegram.record_source_query(
                source="FRED",
                query=", ".join(fred_series),
                signals_returned=len(macro_signals),
                error=None
            )
            for signal in macro_signals:
                self.telegram.record_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
                
        except Exception as e:
            logger.error(f"Error in FRED scan: {e}")
            self.db.log_error("scan_error", "fred", str(e))
            self.telegram.record_source_query(
                source="FRED",
                query="macro series",
                signals_returned=0,
                error=str(type(e).__name__)
            )
        
        # Polymarket predictions
        try:
            prediction_signals = await self.polymarket.scan_all_markets()
            signals.extend(prediction_signals)
            logger.info(f"Got {len(prediction_signals)} prediction signals from Polymarket")
            
            # Record for digest
            self.telegram.record_source_query(
                source="Polymarket",
                query="fed rates, economic events",
                signals_returned=len(prediction_signals),
                error=None
            )
            for signal in prediction_signals:
                self.telegram.record_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
                
        except Exception as e:
            logger.error(f"Error in Polymarket scan: {e}")
            self.db.log_error("scan_error", "polymarket", str(e))
            self.telegram.record_source_query(
                source="Polymarket",
                query="fed rates, economic events",
                signals_returned=0,
                error=str(type(e).__name__)
            )
        
        # 2. Save all signals
        for signal in signals:
            try:
                self.db.save_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
            except Exception as e:
                logger.error(f"Error saving signal: {e}")
        
        logger.info(f"Total signals gathered: {len(signals)}")
        
        # Convert signals to dict for summary
        signals_dict = [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals]
        
        if not signals:
            logger.warning("No signals gathered - skipping analysis")
            # Record no-trade decision for digest
            self.telegram.record_decision({
                "decision_type": "NO_TRADE",
                "reasoning": {"rationale": "No signals gathered"}
            })
            
            # Still send scan summary even with no signals
            await self.telegram.send_scan_summary(
                signals=signals_dict,
                analysis=None,
                portfolio=None
            )
            return
        
        # 3. Get portfolio context
        portfolio = await self.executor.get_portfolio_snapshot()
        positions = await self.executor.get_positions()
        
        # Save portfolio snapshot
        portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        self.db.save_snapshot(portfolio_dict)
        
        # 4. Run Claude analysis
        logger.info("Running Claude analysis...")
        analysis = None
        analysis_dict = None
        
        try:
            analysis = await self.analyst.analyze_signals(
                signals=signals,
                portfolio_context=portfolio_dict,
                watchlist=self.watchlist
            )
            
            # Convert to dict for summary
            analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
            
            # Save analysis
            self.db.save_analysis(analysis_dict)
            
            # 5. Check if we have an actionable trade
            if analysis.is_actionable:
                logger.info(f"Actionable trade identified: {analysis.ticker} - {analysis.recommendation.value}")
                
                # Record trade decision for digest
                self.telegram.record_decision({
                    "decision_type": "TRADE",
                    "trade_details": {
                        "ticker": analysis.ticker,
                        "side": analysis.recommendation.value,
                        "conviction_score": analysis.conviction_score
                    },
                    "reasoning": {"rationale": analysis.thesis},
                    "status": "pending_approval" if Config.APPROVAL_GATE else "approved"
                })
                
                await self._handle_trade_recommendation(analysis, portfolio, positions)
            else:
                logger.info(f"No actionable trade. Conviction: {analysis.conviction_score}")
                
                # Record no-trade decision for digest
                self.telegram.record_decision({
                    "decision_type": "NO_TRADE",
                    "trade_details": {
                        "ticker": getattr(analysis, 'ticker', None),
                        "conviction_score": analysis.conviction_score
                    },
                    "reasoning": {"rationale": f"Conviction {analysis.conviction_score} below threshold"}
                })
                
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}")
            self.db.log_error("analysis_error", "claude", str(e), traceback.format_exc())
            self.telegram.record_system_error("claude_analyst", str(e))
            
            # Record failed analysis for digest
            self.telegram.record_decision({
                "decision_type": "NO_TRADE",
                "reasoning": {"rationale": f"Analysis error: {str(e)[:50]}"}
            })
        
        # 6. SEND COMPREHENSIVE SCAN SUMMARY
        # This is the key addition - send full visibility after every scan
        try:
            await self.telegram.send_scan_summary(
                signals=signals_dict,
                analysis=analysis_dict,
                portfolio=portfolio_dict
            )
            logger.info("Scan summary sent to Telegram")
        except Exception as e:
            logger.error(f"Failed to send scan summary: {e}")
            # Don't let summary failure break the cycle
    
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
                f"âš ï¸ **Trade Rejected by Risk Engine**\n\n"
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
            
            elif command == "digest":
                await self._handle_digest_command()
            
            elif command in ["catalyst", "whatif"]:
                await self._handle_catalyst_command(cmd.get("description"))
            
            elif command == "help":
                await self._handle_help_command()
    
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
            msg += f"â€¢ `{trade['id'][:8]}` - {trade['side'].upper()} {trade['ticker']} ({trade['quantity']} shares)\n"
        
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
        await self.telegram.send_message(f"âœ… Trade approved: {trade_id[:8]}")
        
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
        await self.telegram.send_message(f"âŒ Trade rejected: {trade_id[:8]}\nReason: {reason}")
        
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
            f"ðŸ›‘ **EMERGENCY STOP**\n\n"
            f"Trading halted.\n"
            f"Cancelled {cancelled} orders.\n\n"
            f"Use /resume to restart."
        )
    
    async def _handle_resume_command(self) -> None:
        """Handle /resume command."""
        self.risk_engine.resume_trading()
        
        await self.telegram.send_message(
            "âœ… **Trading Resumed**\n\n"
            "System is active again."
        )
        
        logger.info("Trading resumed via Telegram command")
    
    async def _handle_digest_command(self) -> None:
        """Handle /digest command - send daily digest immediately."""
        logger.info("Manual digest requested via Telegram")
        
        try:
            await self._send_daily_digest()
            logger.info("Manual digest sent successfully")
        except Exception as e:
            logger.error(f"Failed to send manual digest: {e}")
            await self.telegram.send_message(f"âŒ Failed to generate digest: {str(e)[:100]}")
    
    async def _handle_catalyst_command(self, description: str) -> None:
        """
        Handle /catalyst or /whatif command.
        
        This is the manual query interface for forward-predictive analysis.
        User can ask: "/catalyst SpaceX IPO expected H2 2026"
        And get back Claude's second-order analysis of that catalyst.
        """
        if not description:
            await self.telegram.send_message(
                "**Usage:**\n"
                "`/catalyst <description>`\n\n"
                "**Examples:**\n"
                "• `/catalyst SpaceX IPO expected H2 2026`\n"
                "• `/catalyst Fed cuts rates by 50bps in March`\n"
                "• `/whatif NVIDIA beats earnings by 20%`\n"
                "• `/whatif China announces new chip restrictions`"
            )
            return
        
        logger.info(f"Catalyst query: {description}")
        
        # Send thinking indicator
        await self.telegram.send_message(
            f"🧠 **Analyzing catalyst...**\n\n"
            f"_\"{description}\"_\n\n"
            f"Applying second-order thinking..."
        )
        
        try:
            # Get portfolio context
            portfolio = await self.executor.get_portfolio_snapshot()
            portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
            
            # Run the catalyst analysis through Claude
            analysis = await self.analyst.analyze_specific_catalyst(
                catalyst_description=description,
                catalyst_date=None,  # Let Claude infer from description
                portfolio_context=portfolio_dict
            )
            
            # Convert to dict for telegram
            analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
            
            # Save analysis to database
            self.db.save_analysis(analysis_dict)
            
            # Send the formatted response
            await self.telegram.send_catalyst_analysis(
                analysis=analysis_dict,
                catalyst_query=description
            )
            
            logger.info(f"Catalyst analysis complete: {analysis.ticker} @ {analysis.conviction_score}")
            
        except Exception as e:
            logger.error(f"Error in catalyst analysis: {e}")
            await self.telegram.send_message(
                f"❌ **Error analyzing catalyst**\n\n"
                f"_{str(e)[:200]}_\n\n"
                f"Please try again."
            )
    
    async def _handle_help_command(self) -> None:
        """Handle /help command."""
        help_text = """
**Gann Sentinel Commands:**

**📊 Status & Monitoring:**
/status - Portfolio & system status
/pending - List pending trade approvals
/digest - Send daily digest now

**✅ Trade Approval:**
/approve [id] - Approve a pending trade
/reject [id] - Reject a pending trade

**🎯 Catalyst Analysis (NEW!):**
/catalyst <description> - Analyze any catalyst
/whatif <scenario> - Same as /catalyst

**Examples:**
• `/catalyst SpaceX IPO H2 2026`
• `/whatif Fed cuts 50bps in March`
• `/catalyst NVIDIA beats earnings`

**🛑 System Control:**
/stop - Emergency halt (cancels all orders)
/resume - Resume trading after stop
/help - Show this message

**Automatic Features:**
• Hourly scan summaries
• Daily digest at 4 PM ET (9 PM UTC)
"""
        await self.telegram.send_message(help_text)


async def main():
    """Main entry point."""
    print("""
    â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
    â•‘                                                               â•‘
    â•‘   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•— â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—                       â•‘
    â•‘  â–ˆâ–ˆâ•”â•â•â•â•â• â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘                       â•‘
    â•‘  â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘                       â•‘
    â•‘  â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘                       â•‘
    â•‘  â•šâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•”â•â–ˆâ–ˆâ•‘  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘                       â•‘
    â•‘   â•šâ•â•â•â•â•â• â•šâ•â•  â•šâ•â•â•šâ•â•  â•šâ•â•â•â•â•šâ•â•  â•šâ•â•â•â•                       â•‘
    â•‘                                                               â•‘
    â•‘   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ•—   â–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•— â•‘
    â•‘   â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘â•šâ•â•â–ˆâ–ˆâ•”â•â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â•â•â•â–ˆâ–ˆâ•‘ â•‘
    â•‘   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â–ˆâ–ˆâ•— â–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—  â–ˆâ–ˆâ•‘ â•‘
    â•‘   â•šâ•â•â•â•â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â•  â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘â•šâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•”â•â•â•  â–ˆâ–ˆâ•‘ â•‘
    â•‘   â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘   â–ˆâ–ˆâ•‘â–ˆâ–ˆâ•‘ â•šâ–ˆâ–ˆâ–ˆâ–ˆâ•‘â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ•—â–ˆâ–ˆâ–ˆâ•—â•‘
    â•‘   â•šâ•â•â•â•â•â•â•â•šâ•â•â•â•â•â•â•â•šâ•â•  â•šâ•â•â•â•   â•šâ•â•   â•šâ•â•â•šâ•â•  â•šâ•â•â•â•â•šâ•â•â•â•â•â•â•â•šâ•â•â•â•‘
    â•‘                                                               â•‘
    â•‘               AUTONOMOUS TRADING AGENT                        â•‘
    â•‘                                                               â•‘
    â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
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
