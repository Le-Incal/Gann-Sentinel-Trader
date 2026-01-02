"""
Gann Sentinel Trader - Main Agent
Orchestrates the trading system: scan signals, analyze, approve, execute.

Version: 1.1.0 - Added /check command for on-demand stock analysis

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


# =============================================================================
# EMOJI CONSTANTS - Using Unicode escape sequences to prevent encoding issues
# =============================================================================
EMOJI_ROCKET = "\U0001F680"      # 🚀
EMOJI_STOP = "\U0001F6D1"        # 🛑
EMOJI_WARNING = "\U000026A0"     # ⚠
EMOJI_CHECK = "\U00002705"       # ✅
EMOJI_CROSS = "\U0000274C"       # ❌
EMOJI_BULLET = "\U00002022"      # •
EMOJI_SEARCH = "\U0001F50D"      # 🔍


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
        
        # Track pending trade for scan summary
        self._current_pending_trade_id: Optional[str] = None
        
        # Daily digest scheduling
        self.last_digest_time: Optional[datetime] = None
        self.digest_hour_utc = 21  # 9 PM UTC = 4 PM ET / 1 PM PT
        
        # Skip digest on first startup
        self._startup_complete = False
        
        # Default watchlist
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
        
        await self.telegram.send_message(
            f"{EMOJI_ROCKET} Gann Sentinel Trader Started\n\n"
            f"Mode: {Config.MODE}\n"
            f"Approval Gate: {'ON' if Config.APPROVAL_GATE else 'OFF'}\n\n"
            "First scan starting now...\n"
            "Use /help for commands",
            parse_mode=None
        )
        
        self.running = True
        self.watchlist = self.default_watchlist.copy()
        
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
            
            await asyncio.sleep(60)
    
    async def stop(self) -> None:
        """Stop the trading agent."""
        logger.info("Stopping Gann Sentinel Agent...")
        self.running = False
        
        await self.telegram.send_message(
            f"{EMOJI_STOP} Gann Sentinel Trader Stopped\n\n"
            "Use /resume to restart.",
            parse_mode=None
        )
    
    async def _run_cycle(self) -> None:
        """Run one cycle of the trading loop."""
        now = datetime.now(timezone.utc)
        
        await self._process_commands()
        
        if self._startup_complete:
            await self._maybe_send_daily_digest(now)
        
        should_scan = (
            self.last_scan_time is None or
            (now - self.last_scan_time).total_seconds() >= Config.SCAN_INTERVAL_MINUTES * 60
        )
        
        if should_scan:
            logger.info("Running full scan cycle...")
            await self._full_scan_cycle()
            self.last_scan_time = now
            self._startup_complete = True
        
        await self._check_positions()
        await self._process_approved_trades()
    
    async def _maybe_send_daily_digest(self, now: datetime) -> None:
        """Send daily digest if it's time."""
        should_send = False
        
        if self.last_digest_time is None:
            if now.hour >= self.digest_hour_utc:
                should_send = True
        else:
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
        positions = await self.executor.get_positions()
        portfolio = await self.executor.get_portfolio_snapshot()
        pending = self.db.get_pending_trades()
        
        positions_data = [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
        portfolio_data = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        
        await self.telegram.send_daily_digest(
            positions=positions_data,
            portfolio=portfolio_data,
            pending_approvals=pending
        )
    
    async def _full_scan_cycle(self) -> None:
        """Run a full signal scan and analysis cycle."""
        signals: List[Signal] = []
        
        self._current_pending_trade_id = None
        self.telegram.record_scan_start()
        
        logger.info("Gathering signals...")
        
        # Grok sentiment scan
        try:
            sentiment_signals = await self.grok.scan_sentiment(self.watchlist[:5])
            signals.extend(sentiment_signals)
            logger.info(f"Got {len(sentiment_signals)} sentiment signals from Grok")
            
            # Check for errors even if no exception was raised
            grok_error = self.grok.last_error if hasattr(self.grok, 'last_error') else None
            
            self.telegram.record_source_query(
                source="Grok X Search",
                query=f"sentiment: {', '.join(self.watchlist[:5])}",
                signals_returned=len(sentiment_signals),
                error=grok_error if len(sentiment_signals) == 0 else None
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
                error=str(e)[:50]
            )
        
        # Grok market overview
        try:
            overview_signals = await self.grok.scan_market_overview()
            signals.extend(overview_signals)
            logger.info(f"Got {len(overview_signals)} overview signals from Grok")
            
            # Check for errors even if no exception was raised
            grok_error = self.grok.last_error if hasattr(self.grok, 'last_error') else None
            
            self.telegram.record_source_query(
                source="Grok Web Search",
                query="market overview",
                signals_returned=len(overview_signals),
                error=grok_error if len(overview_signals) == 0 else None
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
                error=str(e)[:50]
            )
        
        # FRED macro data
        try:
            macro_signals = await self.fred.scan_all_series()
            signals.extend(macro_signals)
            logger.info(f"Got {len(macro_signals)} macro signals from FRED")
            
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
        
        # Save all signals
        for signal in signals:
            try:
                self.db.save_signal(signal.to_dict() if hasattr(signal, 'to_dict') else signal)
            except Exception as e:
                logger.error(f"Error saving signal: {e}")
        
        logger.info(f"Total signals gathered: {len(signals)}")
        
        signals_dict = [s.to_dict() if hasattr(s, 'to_dict') else s for s in signals]
        
        if not signals:
            logger.warning("No signals gathered - skipping analysis")
            self.telegram.record_decision({
                "decision_type": "NO_TRADE",
                "reasoning": {"rationale": "No signals gathered"}
            })
            
            await self.telegram.send_scan_summary(
                signals=signals_dict,
                analysis=None,
                portfolio=None,
                pending_trade_id=None
            )
            return
        
        # Get portfolio context
        portfolio = await self.executor.get_portfolio_snapshot()
        positions = await self.executor.get_positions()
        
        portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        self.db.save_snapshot(portfolio_dict)
        
        # Run Claude analysis
        logger.info("Running Claude analysis...")
        analysis = None
        analysis_dict = None
        
        try:
            analysis = await self.analyst.analyze_signals(
                signals=signals,
                portfolio_context=portfolio_dict,
                watchlist=self.watchlist
            )
            
            analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
            self.db.save_analysis(analysis_dict)
            
            if analysis.is_actionable:
                logger.info(f"Actionable trade identified: {analysis.ticker} - {analysis.recommendation.value}")
                
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
            
            self.telegram.record_decision({
                "decision_type": "NO_TRADE",
                "reasoning": {"rationale": f"Analysis error: {str(e)[:50]}"}
            })
        
        # Send consolidated scan summary
        try:
            await self.telegram.send_scan_summary(
                signals=signals_dict,
                analysis=analysis_dict,
                portfolio=portfolio_dict,
                pending_trade_id=self._current_pending_trade_id
            )
            logger.info("Scan summary sent to Telegram")
        except Exception as e:
            logger.error(f"Failed to send scan summary: {e}")
    
    async def _handle_trade_recommendation(
        self,
        analysis: Analysis,
        portfolio,
        positions: List[Position]
    ) -> None:
        """Handle a trade recommendation from Claude."""
        
        # Run risk checks
        passed, risk_results = self.risk_engine.validate_trade(
            analysis=analysis,
            portfolio=portfolio,
            current_positions=positions
        )
        
        if not passed:
            logger.warning(f"Trade rejected by risk engine: {analysis.ticker}")
            failed_checks = [r for r in risk_results if not r.passed and r.severity == "error"]
            reasons = "; ".join([r.message for r in failed_checks])
            
            self.telegram.record_risk_rejection(
                ticker=analysis.ticker,
                reason=reasons
            )
            
            logger.info(f"Risk rejection recorded: {analysis.ticker} - {reasons}")
            return
        
        # Get current price
        quote = await self.executor.get_quote(analysis.ticker)
        if "error" in quote:
            error_msg = quote.get("error", "Unknown quote error")
            logger.error(f"Could not get quote for {analysis.ticker}: {error_msg}")
            
            # Record the blocker so it shows in scan summary
            self.telegram.record_trade_blocker(
                blocker_type="Quote Error",
                details=f"{analysis.ticker}: {error_msg}"
            )
            return
        
        current_price = quote.get("mid")
        if not current_price or current_price <= 0:
            logger.error(f"Invalid price for {analysis.ticker}: {current_price}")
            self.telegram.record_trade_blocker(
                blocker_type="Invalid Price",
                details=f"{analysis.ticker}: Price is {current_price}"
            )
            return
        
        # Calculate position size
        sizing = self.risk_engine.calculate_position_size(
            analysis=analysis,
            portfolio=portfolio,
            current_price=current_price
        )
        
        if sizing["shares"] < 1:
            logger.warning(f"Position size too small: {sizing['shares']} shares")
            self.telegram.record_trade_blocker(
                blocker_type="Position Size",
                details=f"Calculated {sizing['shares']} shares (need at least 1)"
            )
            return
        
        # Create trade object
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
        
        self.db.save_trade(trade.to_dict())
        
        if Config.APPROVAL_GATE:
            self._current_pending_trade_id = trade.trade_id[:8]
            logger.info(f"Trade pending approval: {self._current_pending_trade_id}")
        else:
            await self._execute_trade(trade)
    
    async def _execute_trade(self, trade: Trade) -> None:
        """Execute a trade."""
        logger.info(f"Executing trade: {trade.ticker} {trade.side.value} {trade.quantity}")
        
        try:
            trade = await self.executor.submit_order(trade)
            self.db.save_trade(trade.to_dict())
            
            if trade.status == TradeStatus.FILLED:
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
            db_position = self.db.get_position(position.ticker)
            
            if db_position and db_position.get("stop_loss_price"):
                stop_loss = db_position["stop_loss_price"]
                
                if position.current_price and position.current_price <= stop_loss:
                    logger.warning(f"STOP LOSS TRIGGERED: {position.ticker} @ ${position.current_price}")
                    
                    entry_price = db_position.get("avg_entry_price", position.avg_entry_price)
                    loss_pct = (position.current_price - entry_price) / entry_price * 100
                    
                    await self.telegram.send_stop_loss_alert(
                        ticker=position.ticker,
                        trigger_price=position.current_price,
                        loss_pct=loss_pct
                    )
                    
                    if Config.MODE == "LIVE" or self.executor.is_paper:
                        await self.executor.close_position(position.ticker)
                        logger.info(f"Position closed: {position.ticker}")
    
    async def _process_approved_trades(self) -> None:
        """Process any trades that have been approved."""
        pending_trades = self.db.get_pending_trades()
        
        for trade_data in pending_trades:
            if trade_data["status"] == "approved":
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
            elif command == "check":
                await self._handle_check_command(cmd.get("ticker"))
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
    
    async def _handle_pending_command(self) -> None:
        """Handle /pending command."""
        pending = self.db.get_pending_trades()
        
        if not pending:
            await self.telegram.send_message("No pending trades.", parse_mode=None)
            return
        
        msg = "Pending Trades:\n\n"
        for trade in pending:
            msg += f"{EMOJI_BULLET} {trade['id'][:8]} - {trade['side'].upper()} {trade['ticker']} ({trade['quantity']} shares)\n"
        
        await self.telegram.send_message(msg, parse_mode=None)
    
    async def _handle_approve_command(self, trade_id: str) -> None:
        """Handle /approve command."""
        if not trade_id:
            await self.telegram.send_message("Usage: /approve [trade_id]", parse_mode=None)
            return
        
        trade_data = self.db.get_trade(trade_id)
        
        if not trade_data:
            pending = self.db.get_pending_trades()
            for t in pending:
                if t.get("id", "").startswith(trade_id):
                    trade_data = t
                    trade_id = t["id"]
                    break
        
        if not trade_data:
            await self.telegram.send_message(f"Trade not found: {trade_id}", parse_mode=None)
            return
        
        if trade_data["status"] != "pending_approval":
            await self.telegram.send_message(
                f"Trade {trade_id[:8]} is not pending approval (status: {trade_data['status']})",
                parse_mode=None
            )
            return
        
        self.db.update_trade_status(trade_id, "approved")
        await self.telegram.send_message(f"{EMOJI_CHECK} Trade approved: {trade_id[:8]}", parse_mode=None)
        self.telegram.remove_pending_approval(trade_id[:8])
        
        logger.info(f"Trade approved: {trade_id}")
    
    async def _handle_reject_command(self, trade_id: str, reason: str) -> None:
        """Handle /reject command."""
        if not trade_id:
            await self.telegram.send_message("Usage: /reject [trade_id] [reason]", parse_mode=None)
            return
        
        trade_data = self.db.get_trade(trade_id)
        
        if not trade_data:
            pending = self.db.get_pending_trades()
            for t in pending:
                if t.get("id", "").startswith(trade_id):
                    trade_data = t
                    trade_id = t["id"]
                    break
        
        if not trade_data:
            await self.telegram.send_message(f"Trade not found: {trade_id}", parse_mode=None)
            return
        
        self.db.update_trade_status(trade_id, "rejected", rejection_reason=reason)
        await self.telegram.send_message(
            f"{EMOJI_CROSS} Trade rejected: {trade_id[:8]}\nReason: {reason}",
            parse_mode=None
        )
        self.telegram.remove_pending_approval(trade_id[:8])
        
        logger.info(f"Trade rejected: {trade_id} - {reason}")
    
    async def _handle_stop_command(self) -> None:
        """Handle /stop command - emergency halt."""
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        self.risk_engine.halt_trading("Emergency stop triggered via Telegram")
        cancelled = await self.executor.cancel_all_orders()
        
        await self.telegram.send_message(
            f"{EMOJI_STOP} EMERGENCY STOP\n\n"
            f"Trading halted.\n"
            f"Cancelled {cancelled} orders.\n\n"
            f"Use /resume to restart.",
            parse_mode=None
        )
    
    async def _handle_resume_command(self) -> None:
        """Handle /resume command."""
        self.risk_engine.resume_trading()
        
        await self.telegram.send_message(
            f"{EMOJI_CHECK} Trading Resumed\n\n"
            "System is active again.",
            parse_mode=None
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
            await self.telegram.send_message(
                f"{EMOJI_CROSS} Failed to generate digest: {str(e)[:100]}",
                parse_mode=None
            )
    
    async def _handle_check_command(self, ticker: str) -> None:
        """
        Handle /check [ticker] command - on-demand analysis.
        
        Runs full Grok + Claude analysis on any ticker (including pre-IPO)
        and generates a trade recommendation with approval prompt if actionable.
        """
        if not ticker:
            await self.telegram.send_message(
                "Usage: /check [TICKER]\n\nExamples:\n  /check NVDA\n  /check TSLA\n  /check SPACEX",
                parse_mode=None
            )
            return
        
        ticker = ticker.upper().strip()
        logger.info(f"On-demand check requested for: {ticker}")
        
        # Acknowledge the request
        await self.telegram.send_message(
            f"{EMOJI_SEARCH} Analyzing {ticker}...\n\nGathering signals and running analysis.",
            parse_mode=None
        )
        
        try:
            # Run the analysis
            result = await self._run_ticker_analysis(ticker)
            
            # Send formatted result
            await self.telegram.send_check_result(result)
            
            logger.info(f"Check command completed for {ticker}: conviction={result.get('conviction', 0)}")
            
        except Exception as e:
            logger.error(f"Error in check command for {ticker}: {e}")
            logger.error(traceback.format_exc())
            await self.telegram.send_message(
                f"{EMOJI_CROSS} Analysis failed for {ticker}\n\nError: {str(e)[:100]}",
                parse_mode=None
            )
    
    async def _run_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Run full analysis on a single ticker.
        
        Returns a dict with:
        - ticker: The ticker symbol
        - is_tradeable: Whether we can execute trades
        - current_price: Current price (if available)
        - signals_count: Number of signals gathered
        - conviction: Conviction score (0-100)
        - recommendation: BUY/SELL/HOLD/NONE
        - thesis: Analysis thesis
        - historical_context: Historical pattern match (if found)
        - pending_trade_id: Trade ID if trade was created
        - risk_rejection: Reason if risk check failed
        """
        signals = []
        
        # Gather Grok sentiment for this ticker
        try:
            sentiment_signals = await self.grok.scan_sentiment([ticker])
            signals.extend(sentiment_signals)
            logger.info(f"Got {len(sentiment_signals)} sentiment signals for {ticker}")
        except Exception as e:
            logger.error(f"Grok sentiment error for {ticker}: {e}")
        
        # Gather Grok catalysts for this ticker
        try:
            catalyst_signals = await self.grok.scan_catalysts(ticker)
            signals.extend(catalyst_signals)
            logger.info(f"Got {len(catalyst_signals)} catalyst signals for {ticker}")
        except Exception as e:
            logger.error(f"Grok catalyst error for {ticker}: {e}")
        
        # Check if ticker is tradeable (has price data from Alpaca)
        is_tradeable = False
        current_price = None
        price_error = None
        
        try:
            quote = await self.executor.get_quote(ticker)
            if "error" not in quote and quote.get("mid"):
                is_tradeable = True
                current_price = quote.get("mid")
                logger.info(f"{ticker} is tradeable @ ${current_price:.2f}")
            else:
                price_error = quote.get("error", "No price data")
                logger.info(f"{ticker} not tradeable: {price_error}")
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
            try:
                analysis = await self.analyst.analyze_signals(
                    signals=signals,
                    portfolio_context=portfolio_dict,
                    watchlist=[ticker]  # Focus analysis on this ticker
                )
                analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
                logger.info(f"Claude analysis complete: conviction={analysis.conviction_score}")
            except Exception as e:
                logger.error(f"Claude analysis error for {ticker}: {e}")
        
        # Build result dict
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
            "thesis": analysis.thesis if analysis else "Insufficient signals for analysis. Try again later or check if ticker is valid.",
            "historical_context": getattr(analysis, 'historical_context', None) if analysis else None,
            "pending_trade_id": None,
            "risk_rejection": None,
        }
        
        # Attempt to create trade if actionable
        if (analysis and 
            analysis.is_actionable and 
            is_tradeable and 
            analysis.conviction_score >= Config.MIN_CONVICTION):
            
            logger.info(f"Trade criteria met for {ticker}, running risk checks...")
            
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
                    logger.info(f"Trade created for {ticker}: {trade.trade_id[:8]}")
                else:
                    logger.info(f"Position size too small for {ticker}: {sizing['shares']} shares")
            else:
                # Risk check failed
                failed_checks = [r for r in risk_results if not r.passed]
                result["risk_rejection"] = "; ".join([r.message for r in failed_checks])
                logger.info(f"Risk check failed for {ticker}: {result['risk_rejection']}")
        
        return result
    
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
  /check SPACEX (pre-IPO)

Scans run every ~60 minutes
Digest at 4 PM ET daily"""
        await self.telegram.send_message(help_text, parse_mode=None)


async def main():
    """Main entry point."""
    print("""
    ================================================================
                                                                   
       GANN SENTINEL TRADER                                        
       AUTONOMOUS TRADING AGENT                                    
                                                                   
    ================================================================
    
    DISCLAIMER: Trading involves substantial risk of loss.
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
