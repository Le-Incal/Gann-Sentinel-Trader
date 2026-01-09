"""
Gann Sentinel Trader - Main Agent
Orchestrates the trading system: scan signals, analyze, approve, execute.

Version: 2.0.1 - Fixed silent trade creation failures
- Added trade blocker recording for quote fetch failures
- Added trade blocker recording for invalid price
- Added trade blocker recording for position size too small
- Now shows WHY trades weren't created in Telegram

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
from scanners.technical_scanner import TechnicalScanner  # NEW
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
EMOJI_ROCKET = "\U0001F680"      # ðŸš€
EMOJI_STOP = "\U0001F6D1"        # ðŸ›‘
EMOJI_WARNING = "\U000026A0"     # âš 
EMOJI_CHECK = "\U00002705"       # âœ…
EMOJI_CROSS = "\U0000274C"       # âŒ
EMOJI_BULLET = "\U00002022"      # â€¢
EMOJI_SEARCH = "\U0001F50D"      # ðŸ”


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
        self.technical = TechnicalScanner()  # NEW - Chart analysis
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
        
        logger.info("All components initialized successfully")
        logger.info(f"Technical Scanner: {'CONFIGURED' if self.technical.is_configured else 'NOT CONFIGURED'}")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    async def start(self) -> None:
        """Start the trading agent."""
        if self.running:
            logger.warning("Agent already running")
            return
        
        self.running = True
        logger.info("Starting Gann Sentinel Agent...")
        
        # Initialize watchlist
        self._initialize_watchlist()
        
        # Start Telegram command listener
        asyncio.create_task(self._telegram_listener())
        
        # Send startup notification
        await self.telegram.send_startup_message()
        
        # Main loop
        while self.running:
            try:
                await self._main_loop_iteration()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                self.db.log_error("main_loop_error", "agent", str(e), traceback.format_exc())
                self.telegram.record_system_error("main_loop", str(e))
            
            # Wait before next iteration
            await asyncio.sleep(60)
    
    async def stop(self) -> None:
        """Stop the trading agent."""
        logger.info("Stopping agent...")
        self.running = False
        await self.telegram.send_message(f"{EMOJI_STOP} Gann Sentinel Agent stopped", parse_mode=None)
    
    def _initialize_watchlist(self) -> None:
        """Initialize the watchlist from config or defaults."""
        default_watchlist = [
            "TSLA", "NVDA", "RKLB", "PLTR", "MSTR",
            "COIN", "HOOD", "SOFI", "AMD", "SMCI"
        ]
        
        config_watchlist = getattr(Config, 'WATCHLIST', None)
        if config_watchlist:
            self.watchlist = config_watchlist
        else:
            self.watchlist = default_watchlist
        
        logger.info(f"Watchlist initialized: {self.watchlist}")
    
    async def _main_loop_iteration(self) -> None:
        """Single iteration of the main loop."""
        now = datetime.now(timezone.utc)
        
        # Check if it's time for a scan
        if self._should_scan(now):
            await self._full_scan_cycle()
            self.last_scan_time = now
        
        # Check if it's time for daily digest
        await self._check_daily_digest(now)
        
        # Check positions for stop-loss/take-profit
        await self._check_positions()
    
    def _should_scan(self, now: datetime) -> bool:
        """Check if we should run a scan cycle."""
        # Market hours check (US market: 9:30 AM - 4:00 PM ET)
        # ET is UTC-5 (standard) or UTC-4 (daylight)
        hour_utc = now.hour
        
        # Rough market hours in UTC: 14:30 - 21:00 (EST) or 13:30 - 20:00 (EDT)
        market_open = 13  # Conservative start
        market_close = 21  # Conservative end
        
        if not (market_open <= hour_utc <= market_close):
            return False
        
        # Check scan interval
        if self.last_scan_time is None:
            return True
        
        elapsed = (now - self.last_scan_time).total_seconds() / 60
        return elapsed >= Config.SCAN_INTERVAL_MINUTES
    
    async def _check_daily_digest(self, now: datetime) -> None:
        """Check if daily digest should be sent."""
        if now.hour == self.digest_hour_utc:
            if self.last_digest_time is None or self.last_digest_time.date() < now.date():
                try:
                    await self._send_daily_digest()
                    self.last_digest_time = now
                except Exception as e:
                    logger.error(f"Error sending daily digest: {e}")
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
        technical_signals: List[Dict[str, Any]] = []  # NEW - Track technical separately
        
        self._current_pending_trade_id = None
        self.telegram.record_scan_start()
        
        logger.info("Gathering signals...")
        
        # Grok sentiment scan
        try:
            sentiment_signals = await self.grok.scan_sentiment(self.watchlist[:5])
            signals.extend(sentiment_signals)
            logger.info(f"Got {len(sentiment_signals)} sentiment signals from Grok")
            
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
        
        # =================================================================
        # NEW: TECHNICAL ANALYSIS SCAN
        # Run on top watchlist tickers
        # =================================================================
        try:
            if self.technical.is_configured:
                tech_tickers = self.watchlist[:5]  # Top 5 from watchlist
                logger.info(f"Running technical analysis on: {tech_tickers}")
                
                for ticker in tech_tickers:
                    try:
                        # Use 1-year daily for hourly scans (faster)
                        tech_signal = await self.technical.scan_ticker(ticker, "1D", 1.0)
                        
                        if tech_signal:
                            signal_dict = tech_signal.to_dict()
                            technical_signals.append(signal_dict)
                            signals.append(tech_signal)  # Add to main signals list
                            
                            # Record for telegram
                            self.telegram.record_signal(signal_dict)
                            self.telegram.record_technical_signal(signal_dict)  # NEW method
                            
                            logger.info(
                                f"Technical {ticker}: {tech_signal.market_state.state.value}, "
                                f"verdict={tech_signal.verdict.value}"
                            )
                    except Exception as e:
                        logger.error(f"Technical scan error for {ticker}: {e}")
                
                self.telegram.record_source_query(
                    source="Technical Scanner",
                    query=f"chart analysis: {', '.join(tech_tickers)}",
                    signals_returned=len(technical_signals),
                    error=None
                )
                
                logger.info(f"Got {len(technical_signals)} technical signals")
            else:
                logger.warning("Technical scanner not configured - skipping")
                self.telegram.record_source_query(
                    source="Technical Scanner",
                    query="chart analysis",
                    signals_returned=0,
                    error="Not configured (Alpaca keys missing)"
                )
                
        except Exception as e:
            logger.error(f"Error in technical scan: {e}")
            self.db.log_error("scan_error", "technical", str(e))
            self.telegram.record_source_query(
                source="Technical Scanner",
                query="chart analysis",
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
                pending_trade_id=None,
                technical_signals=technical_signals  # NEW
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
                pending_trade_id=self._current_pending_trade_id,
                technical_signals=technical_signals  # NEW
            )
        except Exception as e:
            logger.error(f"Error sending scan summary: {e}")
            self.telegram.record_system_error("scan_summary", str(e))
    
    async def _handle_trade_recommendation(
        self,
        analysis: Analysis,
        portfolio: Any,
        positions: List[Position]
    ) -> None:
        """Process a trade recommendation through risk checks."""
        # Run risk checks
        passed, results = self.risk_engine.validate_trade(
            analysis=analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis,
            portfolio=portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio,
            current_positions=[p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
        )
        
        for result in results:
            if not result.passed:
                logger.warning(f"Risk check failed: {result.check_name} - {result.message}")
                self.telegram.record_risk_rejection({
                    "check_name": result.check_name,
                    "message": result.message,
                    "severity": result.severity
                })
        
        if not passed:
            failed_checks = [r for r in results if not r.passed and r.severity == "error"]
            reasons = "; ".join([r.message for r in failed_checks])
            logger.info(f"Trade rejected by risk engine: {reasons}")
            return
        
        # Calculate position size
        portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        position_value = portfolio_dict.get("equity", 100000) * (analysis.position_size_pct / 100)
        
        # Get current price
        try:
            quote = await self.executor.get_quote(analysis.ticker)
            current_price = quote.get("mid", quote.get("last", 0))
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            self.telegram.record_trade_blocker({
                "type": "quote_fetch_failed",
                "details": f"Could not get price for {analysis.ticker}: {str(e)[:100]}"
            })
            return
        
        if current_price <= 0:
            logger.error(f"Invalid price for {analysis.ticker}")
            self.telegram.record_trade_blocker({
                "type": "invalid_price",
                "details": f"Price for {analysis.ticker} is {current_price} (must be > 0)"
            })
            return
        
        shares = int(position_value / current_price)
        if shares <= 0:
            logger.warning(f"Position size too small for {analysis.ticker}")
            self.telegram.record_trade_blocker({
                "type": "position_too_small",
                "details": f"Calculated 0 shares for {analysis.ticker} (value: ${position_value:.2f}, price: ${current_price:.2f})"
            })
            return
        
        # Create trade record
        trade = Trade(
            id=str(uuid.uuid4()),
            analysis_id=analysis.id,
            ticker=analysis.ticker,
            side=OrderSide.BUY if analysis.recommendation == Recommendation.BUY else OrderSide.SELL,
            quantity=shares,
            order_type=OrderType.MARKET,
            status=TradeStatus.PENDING_APPROVAL,
            thesis=analysis.thesis,
            conviction_score=analysis.conviction_score,
            stop_loss_price=current_price * (1 - analysis.stop_loss_pct / 100) if analysis.stop_loss_pct else None
        )
        
        self.db.save_trade(trade.to_dict())
        
        # Store for scan summary
        self._current_pending_trade_id = trade.id
        
        # Send approval request
        await self.telegram.send_trade_approval_request(
            trade=trade.to_dict(),
            analysis=analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis,
            current_price=current_price
        )
    
    async def _check_positions(self) -> None:
        """Check positions for stop-loss and take-profit triggers."""
        try:
            positions = await self.executor.get_positions()
            
            for position in positions:
                pos_dict = position.to_dict() if hasattr(position, 'to_dict') else position
                
                # Check stop loss
                stop_loss = pos_dict.get("stop_loss_price")
                current_price = pos_dict.get("current_price")
                
                if stop_loss and current_price and current_price <= stop_loss:
                    logger.warning(f"Stop loss triggered for {pos_dict.get('ticker')}")
                    await self.telegram.send_message(
                        f"{EMOJI_WARNING} STOP LOSS TRIGGERED\n\n"
                        f"Ticker: {pos_dict.get('ticker')}\n"
                        f"Current: ${current_price:.2f}\n"
                        f"Stop: ${stop_loss:.2f}",
                        parse_mode=None
                    )
                    
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    # =========================================================================
    # TELEGRAM COMMANDS
    # =========================================================================
    
    async def _telegram_listener(self) -> None:
        """Listen for Telegram commands."""
        while self.running:
            try:
                commands = await self.telegram.get_commands()
                
                for cmd in commands:
                    await self._handle_command(cmd)
                    
            except Exception as e:
                logger.error(f"Error in Telegram listener: {e}")
            
            await asyncio.sleep(2)
    
    async def _handle_command(self, cmd: Dict[str, Any]) -> None:
        """Handle a Telegram command."""
        command = cmd.get("command", "").lower()
        
        if command == "status":
            await self._handle_status_command()
        elif command == "pending":
            await self._handle_pending_command()
        elif command == "approve":
            await self._handle_approve_command(cmd.get("trade_id"))
        elif command == "reject":
            await self._handle_reject_command(cmd.get("trade_id"))
        elif command == "stop":
            await self._handle_stop_command()
        elif command == "resume":
            await self._handle_resume_command()
        elif command == "scan":
            await self._handle_scan_command()
        elif command == "digest":
            await self._send_daily_digest()
        elif command == "check":
            await self._handle_check_command(cmd.get("ticker"))
        elif command == "help":
            await self._handle_help_command()
    
    async def _handle_status_command(self) -> None:
        """Handle /status command."""
        try:
            portfolio = await self.executor.get_portfolio_snapshot()
            positions = await self.executor.get_positions()
            pending = self.db.get_pending_trades()
            
            portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
            positions_list = [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
            
            await self.telegram.send_status_message(
                portfolio=portfolio_dict,
                positions=positions_list,
                pending_approvals=pending,
                agent_running=self.running
            )
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await self.telegram.send_message(f"{EMOJI_CROSS} Error getting status: {str(e)[:50]}", parse_mode=None)
    
    async def _handle_pending_command(self) -> None:
        """Handle /pending command."""
        pending = self.db.get_pending_trades()
        
        if not pending:
            await self.telegram.send_message("No pending trade approvals.", parse_mode=None)
            return
        
        for trade in pending:
            analysis = self.db.get_analysis(trade.get("analysis_id"))
            await self.telegram.send_trade_approval_request(
                trade=trade,
                analysis=analysis,
                current_price=trade.get("current_price")
            )
    
    async def _handle_approve_command(self, trade_id: str) -> None:
        """Handle /approve command."""
        if not trade_id:
            await self.telegram.send_message("Usage: /approve [trade_id]", parse_mode=None)
            return
        
        trade = self.db.get_trade(trade_id)
        if not trade:
            await self.telegram.send_message(f"Trade {trade_id} not found", parse_mode=None)
            return
        
        if trade.get("status") != TradeStatus.PENDING_APPROVAL.value:
            await self.telegram.send_message(f"Trade {trade_id} is not pending approval", parse_mode=None)
            return
        
        # Execute trade
        try:
            result = await self.executor.execute_order(
                ticker=trade.get("ticker"),
                side=trade.get("side"),
                quantity=trade.get("quantity"),
                order_type=trade.get("order_type", "market")
            )
            
            if result.get("success"):
                self.db.update_trade_status(trade_id, TradeStatus.SUBMITTED.value, result.get("order_id"))
                await self.telegram.send_message(
                    f"{EMOJI_CHECK} Trade {trade_id} approved and submitted\n"
                    f"Order ID: {result.get('order_id')}",
                    parse_mode=None
                )
            else:
                await self.telegram.send_message(
                    f"{EMOJI_CROSS} Trade execution failed: {result.get('error')}",
                    parse_mode=None
                )
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            await self.telegram.send_message(f"{EMOJI_CROSS} Execution error: {str(e)[:50]}", parse_mode=None)
    
    async def _handle_reject_command(self, trade_id: str) -> None:
        """Handle /reject command."""
        if not trade_id:
            await self.telegram.send_message("Usage: /reject [trade_id]", parse_mode=None)
            return
        
        trade = self.db.get_trade(trade_id)
        if not trade:
            await self.telegram.send_message(f"Trade {trade_id} not found", parse_mode=None)
            return
        
        self.db.update_trade_status(trade_id, TradeStatus.REJECTED.value)
        await self.telegram.send_message(f"{EMOJI_CROSS} Trade {trade_id} rejected", parse_mode=None)
    
    async def _handle_stop_command(self) -> None:
        """Handle /stop command - emergency halt."""
        await self.stop()
    
    async def _handle_resume_command(self) -> None:
        """Handle /resume command."""
        if not self.running:
            asyncio.create_task(self.start())
            await self.telegram.send_message(f"{EMOJI_ROCKET} Agent resumed", parse_mode=None)
        else:
            await self.telegram.send_message("Agent already running", parse_mode=None)
    
    async def _handle_scan_command(self) -> None:
        """Handle /scan command - manual scan trigger."""
        await self.telegram.send_message(f"{EMOJI_SEARCH} Running manual scan...", parse_mode=None)
        await self._full_scan_cycle()
    
    async def _handle_check_command(self, ticker: str) -> None:
        """
        Handle /check [ticker] command - on-demand analysis.
        
        Runs full Grok + Technical + Claude analysis on any ticker
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
            f"{EMOJI_SEARCH} Analyzing {ticker}...\n\nGathering signals, chart analysis, and running Claude analysis.",
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
        - technical_analysis: Chart structure analysis (NEW)
        - pending_trade_id: Trade ID if trade was created
        - risk_rejection: Reason if risk check failed
        """
        signals = []
        technical_data = None  # NEW
        
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
        
        # =================================================================
        # NEW: TECHNICAL ANALYSIS for /check command
        # Use 5-year weekly for comprehensive view
        # =================================================================
        try:
            if self.technical.is_configured:
                logger.info(f"Running technical analysis for {ticker}")
                tech_signal = await self.technical.scan_ticker(ticker, "1W", 5.0)
                
                if tech_signal:
                    technical_data = tech_signal.to_dict()
                    signals.append(tech_signal)  # Add to signals for Claude
                    logger.info(
                        f"Technical {ticker}: state={tech_signal.market_state.state.value}, "
                        f"bias={tech_signal.market_state.bias.value}, "
                        f"verdict={tech_signal.verdict.value}"
                    )
        except Exception as e:
            logger.error(f"Technical analysis error for {ticker}: {e}")
        
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
        portfolio_obj = await self.executor.get_portfolio_snapshot()
        positions = await self.executor.get_positions()
        
        if hasattr(portfolio_obj, 'to_dict'):
            portfolio_dict = portfolio_obj.to_dict()
        elif isinstance(portfolio_obj, dict):
            portfolio_dict = portfolio_obj
        else:
            portfolio_dict = {
                "equity": getattr(portfolio_obj, 'equity', 100000),
                "cash": getattr(portfolio_obj, 'cash', 100000),
                "daily_pnl": getattr(portfolio_obj, 'daily_pnl', 0),
                "position_count": getattr(portfolio_obj, 'position_count', 0),
            }
        
        # Run Claude analysis
        analysis = None
        analysis_dict = None
        
        if signals:
            try:
                analysis = await self.analyst.analyze_signals(
                    signals=signals,
                    portfolio_context=portfolio_dict,
                    watchlist=[ticker]
                )
                analysis_dict = analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
                logger.info(f"Claude analysis complete: conviction={analysis.conviction_score}")
            except Exception as e:
                logger.error(f"Claude analysis error for {ticker}: {e}")
        
        # Build result dict
        recommendation = analysis.recommendation.value if analysis else "NONE"
        
        # Convert SELL to HOLD for /check (we can't sell what we don't own)
        if recommendation == "SELL":
            existing_position = any(
                p.ticker == ticker if hasattr(p, 'ticker') else p.get('ticker') == ticker
                for p in positions
            )
            if not existing_position:
                logger.info(f"Converting SELL to HOLD for {ticker} - /check is BUY-only")
                recommendation = "HOLD"
        
        result = {
            "ticker": ticker,
            "is_tradeable": is_tradeable,
            "current_price": current_price,
            "price_error": price_error,
            "signals_count": len(signals),
            "conviction": analysis.conviction_score if analysis else 0,
            "recommendation": recommendation,
            "thesis": analysis.thesis if analysis else "Insufficient signals for analysis.",
            "historical_context": analysis_dict.get("historical_context") if analysis_dict else None,
            "bull_case": analysis_dict.get("bull_case") if analysis_dict else None,
            "bear_case": analysis_dict.get("bear_case") if analysis_dict else None,
            "technical_analysis": technical_data,  # NEW
            "pending_trade_id": None,
            "risk_rejection": None,
        }
        
        # Only create trade for BUY recommendations on tradeable tickers
        if (is_tradeable and 
            analysis and 
            analysis.conviction_score >= 80 and 
            recommendation == "BUY"):
            
            logger.info(f"Trade criteria met for {ticker}, running risk checks...")
            
            passed, risk_results = self.risk_engine.validate_trade(
                analysis=analysis_dict,
                portfolio=portfolio_dict,
                current_positions=[p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]
            )
            
            if passed:
                # Create trade
                position_value = portfolio_dict.get("equity", 100000) * (analysis.position_size_pct / 100)
                shares = int(position_value / current_price) if current_price > 0 else 0
                
                if shares > 0:
                    trade = Trade(
                        id=str(uuid.uuid4()),
                        analysis_id=analysis.id,
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                        status=TradeStatus.PENDING_APPROVAL,
                        thesis=analysis.thesis,
                        conviction_score=analysis.conviction_score,
                        stop_loss_price=current_price * (1 - analysis.stop_loss_pct / 100) if analysis.stop_loss_pct else None
                    )
                    
                    self.db.save_trade(trade.to_dict())
                    result["pending_trade_id"] = trade.id
                    logger.info(f"Created pending trade {trade.id} for {ticker}")
            else:
                failed_checks = [r for r in risk_results if not r.passed]
                result["risk_rejection"] = "; ".join([r.message for r in failed_checks])
                logger.info(f"Risk check failed for {ticker}: {result['risk_rejection']}")
        
        return result
    
    async def _handle_help_command(self) -> None:
        """Handle /help command."""
        help_text = f"""
{EMOJI_SEARCH} GANN SENTINEL COMMANDS

/status - Portfolio & system status
/pending - List pending approvals
/approve [ID] - Approve a trade
/reject [ID] - Reject a trade
/scan - Run manual scan cycle
/digest - Send daily digest now
/check [TICKER] - Analyze any stock
/stop - Emergency halt
/resume - Resume trading
/help - Show this help

EXAMPLES:
  /check NVDA
  /check TSLA
"""
        await self.telegram.send_message(help_text, parse_mode=None)


# =============================================================================
# IMPORTS FOR TRADE MODEL
# =============================================================================

import uuid


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point."""
    agent = GannSentinelAgent()
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await agent.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        await agent.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
