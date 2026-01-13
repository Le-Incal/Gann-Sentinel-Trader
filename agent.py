"""
Gann Sentinel Trader - Main Agent
Orchestrates the trading system: scan signals, analyze, approve, execute.

Version: 2.4.0 - Event Scanner + MACA + Learning Engine + Smart Scheduling
- Event Scanner: 27 corporate event types (LevelFields-style)
- MACA: Multi-Agent Consensus Architecture for /check command
- Learning Engine tracks performance vs SPY
- Smart scheduling: 2 scans/day (9:35 AM, 12:30 PM ET), no weekends
- Context injection: Claude sees historical performance
- 75% reduction in API costs

DISCLAIMER: Trading involves substantial risk of loss. This is an experimental
system and nothing here constitutes financial advice. Only trade what you can
afford to lose.
"""

import asyncio
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from config import Config
from storage.database import Database
from scanners.grok_scanner import GrokScanner
from scanners.fred_scanner import FREDScanner
from scanners.polymarket_scanner import PolymarketScanner
from scanners.technical_scanner import TechnicalScanner
from scanners.event_scanner import EventScanner
from analyzers.claude_analyst import ClaudeAnalyst
from executors.risk_engine import RiskEngine
from executors.alpaca_executor import AlpacaExecutor
from notifications.telegram_bot import TelegramBot
from models.signals import Signal
from models.analysis import Analysis, Recommendation
from models.trades import Trade, TradeStatus, OrderType, OrderSide, Position
from learning_engine import LearningEngine, SmartScheduler, add_learning_tables

# MACA components (optional)
try:
    from core.maca_orchestrator import MACAOrchestrator
    from analyzers.perplexity_analyst import PerplexityAnalyst
    from analyzers.chatgpt_analyst import ChatGPTAnalyst
    MACA_AVAILABLE = True
except ImportError as e:
    MACA_AVAILABLE = False
    logging.warning(f"MACA components not available: {e}")

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
EMOJI_HOURGLASS = "\U000023F3"   # ⏳


class GannSentinelAgent:
    """
    Main trading agent that orchestrates all components.
    """

    def __init__(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("INITIALIZING GANN SENTINEL TRADER v2.4.0")
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

        # Add learning tables to database
        try:
            add_learning_tables(self.db)
            logger.info("Learning Engine tables initialized")
        except Exception as e:
            logger.warning(f"Could not add learning tables: {e}")

        self.grok = GrokScanner()
        self.fred = FREDScanner()
        self.polymarket = PolymarketScanner()
        self.technical = TechnicalScanner()
        self.event_scanner = EventScanner()
        self.analyst = ClaudeAnalyst()
        self.risk_engine = RiskEngine()
        self.executor = AlpacaExecutor()
        self.telegram = TelegramBot(
            token=Config.TELEGRAM_BOT_TOKEN,
            chat_id=Config.TELEGRAM_CHAT_ID
        )

        # Initialize Learning Engine and Smart Scheduler
        self.learning_engine = LearningEngine(db=self.db, executor=self.executor)
        self.scheduler = SmartScheduler()

        # Initialize MACA Orchestrator if enabled
        self.maca: Optional[MACAOrchestrator] = None
        self.maca_enabled = os.getenv("MACA_ENABLED", "false").lower() == "true"

        if self.maca_enabled and MACA_AVAILABLE:
            try:
                self.perplexity = PerplexityAnalyst()
                self.chatgpt = ChatGPTAnalyst()

                self.maca = MACAOrchestrator(
                    db=self.db,
                    grok=self.grok,
                    perplexity=self.perplexity,
                    chatgpt=self.chatgpt,
                    claude=self.analyst,
                    telegram=self.telegram
                )
                logger.info("MACA Orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MACA: {e}")
                self.maca = None
        elif self.maca_enabled and not MACA_AVAILABLE:
            logger.warning("MACA enabled but components not available")

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
        logger.info(f"Event Scanner: {'CONFIGURED' if self.event_scanner.is_configured else 'NOT CONFIGURED'}")
        logger.info(f"Learning Engine: ENABLED")
        logger.info(f"Smart Scheduling: Morning (9:35 AM ET) + Midday (12:30 PM ET)")
        logger.info(f"MACA: {'ENABLED' if self.maca else 'DISABLED'}")

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
        """
        Check if we should run a scan cycle using Smart Scheduling.

        Smart Schedule:
        - Morning scan: 9:35 AM ET (14:35 UTC)
        - Midday scan: 12:30 PM ET (17:30 UTC)
        - No weekends
        - Manual /scan and /check always work
        """
        return self.scheduler.should_scan(now)

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
        technical_signals: List[Dict[str, Any]] = []

        self._current_pending_trade_id = None
        self.telegram.record_scan_start()

        # Record scan type with scheduler
        now = datetime.now(timezone.utc)
        scan_type = self.scheduler.get_scan_type(now)
        self.scheduler.record_scan(now, scan_type)
        logger.info(f"Starting {scan_type.upper()} scan cycle")

        # Generate learning context for Claude
        learning_context = self.learning_engine.generate_claude_context()
        logger.info(f"Learning context generated: {learning_context.get('performance_summary', {}).get('total_trades', 0)} historical trades")

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
        # TECHNICAL ANALYSIS SCAN
        # Run on top watchlist tickers
        # =================================================================
        try:
            if self.technical.is_configured:
                tech_tickers = self.watchlist[:5]
                logger.info(f"Running technical analysis on: {tech_tickers}")

                for ticker in tech_tickers:
                    try:
                        # Use 1-year daily for hourly scans (faster)
                        tech_signal = await self.technical.scan_ticker(ticker, "1D", 1.0)

                        if tech_signal:
                            signal_dict = tech_signal.to_dict()
                            technical_signals.append(signal_dict)
                            signals.append(tech_signal)

                            # Record for telegram
                            self.telegram.record_signal(signal_dict)
                            self.telegram.record_technical_signal(signal_dict)

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

        # =================================================================
        # EVENT SCANNER - Corporate Events (27 types)
        # =================================================================
        event_signals = []
        try:
            if self.event_scanner.is_configured:
                logger.info("Running event scan (27 event types)...")

                raw_events = await self.event_scanner.scan_market_wide()

                for event in raw_events:
                    signal_dict = event.to_dict()
                    event_signals.append(signal_dict)
                    signals.append(event)

                    # Record for telegram
                    self.telegram.record_signal(signal_dict)

                    logger.info(
                        f"Event: {event.asset_scope['tickers'][0]} - {event.event_type} "
                        f"({event.directional_bias}, conf={event.confidence:.2f})"
                    )

                self.telegram.record_source_query(
                    source="Event Scanner",
                    query="market-wide event scan (27 types)",
                    signals_returned=len(event_signals),
                    error=None
                )

                logger.info(f"Got {len(event_signals)} event signals")
            else:
                logger.warning("Event Scanner not configured - skipping")
                self.telegram.record_source_query(
                    source="Event Scanner",
                    query="corporate events",
                    signals_returned=0,
                    error="Not configured (XAI_API_KEY missing)"
                )

        except Exception as e:
            logger.error(f"Error in event scan: {e}")
            self.db.log_error("scan_error", "event_scanner", str(e))
            self.telegram.record_source_query(
                source="Event Scanner",
                query="corporate events",
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
                technical_signals=technical_signals
            )
            return

        # Get portfolio context
        portfolio = await self.executor.get_portfolio_snapshot()
        positions = await self.executor.get_positions()

        portfolio_dict = portfolio.to_dict() if hasattr(portfolio, 'to_dict') else portfolio
        self.db.save_snapshot(portfolio_dict)

        # Run Claude analysis with learning context
        logger.info("Running Claude analysis with learning context...")
        analysis = None
        analysis_dict = None

        # Enrich portfolio context with learning data
        enriched_portfolio = {
            **portfolio_dict,
            "learning_context": learning_context,
            "learning_summary": self.learning_engine.format_context_for_prompt(learning_context)
        }

        try:
            analysis = await self.analyst.analyze_signals(
                signals=signals,
                portfolio_context=enriched_portfolio,
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
                technical_signals=technical_signals
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
            return

        if current_price <= 0:
            logger.error(f"Invalid price for {analysis.ticker}")
            return

        shares = int(position_value / current_price)
        if shares <= 0:
            logger.warning(f"Position size too small for {analysis.ticker}")
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
        """Handle a Telegram command or callback query."""
        command = cmd.get("command", "").lower()

        # Answer callback query if this came from an inline button
        callback_id = cmd.get("callback_id")
        if callback_id:
            # Answer immediately to remove "loading" state
            await self.telegram.answer_callback_query(callback_id)

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
        elif command == "logs":
            await self._handle_logs_command(cmd.get("count", 20))
        elif command == "positions":
            await self._handle_positions_command()
        elif command == "history":
            await self._handle_history_command(cmd.get("count", 10))
        elif command == "export":
            await self._handle_export_command(cmd.get("format", "csv"))
        elif command == "cost":
            await self._handle_cost_command(cmd.get("days", 7))

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

        When MACA is enabled:
        - Runs Grok, Perplexity, ChatGPT in parallel
        - Claude synthesizes all theses into final recommendation
        - Shows thesis from each AI with conviction scores
        - Tracks API costs per check

        When MACA is disabled:
        - Runs Grok + Technical + Claude analysis
        """
        if not ticker:
            await self.telegram.send_message(
                "Usage: /check [TICKER]\n\nExamples:\n  /check NVDA\n  /check TSLA\n  /check AAPL",
                parse_mode=None
            )
            return

        ticker = ticker.upper().strip()
        logger.info(f"On-demand check requested for: {ticker}")

        # Use MACA if enabled
        if self.maca and self.maca.is_configured:
            await self._handle_maca_check(ticker)
        else:
            await self._handle_standard_check(ticker)

    async def _handle_maca_check(self, ticker: str) -> None:
        """
        Handle /check using MACA - all 4 AIs analyze the ticker.

        Shows:
        1. Each AI's thesis and conviction
        2. Claude's synthesis and final recommendation
        3. Cost tracking for the check
        """
        # Acknowledge with MACA mode
        await self.telegram.send_message(
            f"{EMOJI_SEARCH} MACA Analyzing {ticker}...\n\n"
            f"Running parallel analysis:\n"
            f"  \U0001F426 Grok (sentiment)\n"
            f"  \U0001F3AF Perplexity (fundamentals)\n"
            f"  \U0001F9E0 ChatGPT (patterns)\n"
            f"  \U0001F916 Claude (synthesis)\n\n"
            f"This may take 30-45 seconds...",
            parse_mode=None
        )

        try:
            # Get portfolio context
            portfolio_obj = await self.executor.get_portfolio_snapshot()
            if hasattr(portfolio_obj, 'to_dict'):
                portfolio = portfolio_obj.to_dict()
            elif isinstance(portfolio_obj, dict):
                portfolio = portfolio_obj
            else:
                portfolio = {
                    "equity": getattr(portfolio_obj, 'equity', 100000),
                    "cash": getattr(portfolio_obj, 'cash', 100000),
                    "positions": []
                }

            # Get technical analysis
            technical_data = None
            if self.technical.is_configured:
                try:
                    tech_signal = await self.technical.scan_ticker(ticker, "1W", 5.0)
                    if tech_signal:
                        technical_data = tech_signal.to_dict()
                except Exception as e:
                    logger.warning(f"Technical analysis failed for {ticker}: {e}")

            # Run MACA ticker check
            result = await self.maca.run_ticker_check(
                ticker=ticker,
                portfolio=portfolio,
                fred_signals=[],
                polymarket_signals=[],
                technical_analysis=technical_data
            )

            # Format and send MACA result
            message = self.telegram.format_maca_check_result(result)
            await self.telegram.send_message(message, parse_mode=None, message_type="maca_check")

            # Check if we should create a trade
            synthesis = result.get("synthesis", {})
            conviction = synthesis.get("recommendation", {}).get("conviction_score", 0)
            side = synthesis.get("recommendation", {}).get("side")

            if conviction >= 80 and side == "BUY":
                # Create pending trade for approval
                await self._create_maca_trade(ticker, result, portfolio)

            # Log completion
            cycle_cost = result.get("cycle_cost", {})
            logger.info(f"MACA check completed for {ticker}: "
                       f"conviction={conviction}, "
                       f"cost=${cycle_cost.get('total_cost_usd', 0):.4f}")

        except Exception as e:
            logger.error(f"MACA check failed for {ticker}: {e}")
            logger.error(traceback.format_exc())
            await self.telegram.send_message(
                f"{EMOJI_CROSS} MACA analysis failed for {ticker}\n\nError: {str(e)[:100]}",
                parse_mode=None
            )

    async def _create_maca_trade(
        self,
        ticker: str,
        maca_result: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Optional[str]:
        """Create a pending trade from MACA result."""
        try:
            synthesis = maca_result.get("synthesis", {})
            rec = synthesis.get("recommendation", {})

            conviction = rec.get("conviction_score", 0)
            position_size_pct = rec.get("position_size_pct", 10)
            stop_loss_pct = rec.get("stop_loss_pct", 8)
            thesis = rec.get("thesis", "MACA consensus recommendation")

            # Get current price
            quote = await self.executor.get_quote(ticker)
            current_price = quote.get("mid", 0)

            if current_price <= 0:
                logger.warning(f"Cannot create trade for {ticker}: no price")
                return None

            # Calculate shares
            equity = portfolio.get("equity", 100000)
            position_value = equity * (position_size_pct / 100)
            shares = int(position_value / current_price)

            if shares <= 0:
                return None

            # Create trade
            trade = Trade(
                id=str(uuid.uuid4()),
                analysis_id=maca_result.get("cycle_id"),
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=shares,
                order_type=OrderType.MARKET,
                status=TradeStatus.PENDING_APPROVAL,
                thesis=thesis,
                conviction_score=conviction,
                stop_loss_price=current_price * (1 - stop_loss_pct / 100)
            )

            self.db.save_trade(trade.to_dict())

            # Send approval message
            await self.telegram.send_message(
                f"\n{EMOJI_BULLET} Trade pending approval:\n"
                f"  {trade.side.value} {trade.quantity} {ticker}\n"
                f"  Conviction: {conviction}/100\n"
                f"  ID: {trade.id[:8]}\n\n"
                f"Reply /approve {trade.id[:8]} or /reject {trade.id[:8]}",
                parse_mode=None
            )

            return trade.id

        except Exception as e:
            logger.error(f"Failed to create MACA trade: {e}")
            return None

    async def _handle_standard_check(self, ticker: str) -> None:
        """Handle /check using standard analysis (Grok + Technical + Claude)."""
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
        - technical_analysis: Chart structure analysis
        - pending_trade_id: Trade ID if trade was created
        - risk_rejection: Reason if risk check failed
        """
        signals = []
        technical_data = None

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
        # TECHNICAL ANALYSIS for /check command
        # Use 5-year weekly for comprehensive view
        # =================================================================
        try:
            if self.technical.is_configured:
                logger.info(f"Running technical analysis for {ticker}")
                tech_signal = await self.technical.scan_ticker(ticker, "1W", 5.0)

                if tech_signal:
                    technical_data = tech_signal.to_dict()
                    signals.append(tech_signal)
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
            "technical_analysis": technical_data,
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
/positions - Current open positions
/history - Trade history
/pending - List pending approvals
/approve [ID] - Approve a trade
/reject [ID] - Reject a trade
/scan - Run manual scan cycle
/check [TICKER] - Analyze any stock (MACA)
/export [csv/parquet] - Export data
/cost [days] - API cost summary
/logs - View recent activity
/digest - Send daily digest
/stop - Emergency halt
/resume - Resume trading
/help - Show this help

EXAMPLES:
  /check NVDA
  /history 20
  /export csv
  /cost 30
"""
        await self.telegram.send_message(help_text, parse_mode=None)

    async def _handle_logs_command(self, count: int = 20) -> None:
        """Handle /logs command."""
        try:
            await self.telegram.send_logs_summary(count)
        except Exception as e:
            logger.error(f"Error in logs command: {e}")
            await self.telegram.send_message(f"{EMOJI_CROSS} Error getting logs: {str(e)[:50]}", parse_mode=None)

    async def _handle_positions_command(self) -> None:
        """Handle /positions command - show current open positions."""
        try:
            # Get positions from Alpaca
            positions = await self.executor.get_positions()
            positions_data = [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions]

            # Format and send
            message = self.telegram.format_positions_message(positions_data)
            await self.telegram.send_message(message, parse_mode=None, message_type="positions")

        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await self.telegram.send_message(f"{EMOJI_CROSS} Error getting positions: {str(e)[:50]}", parse_mode=None)

    async def _handle_history_command(self, count: int = 10) -> None:
        """Handle /history command - show trade history."""
        try:
            # Get recent trades from database
            trades = self.db.get_recent_trades(limit=count)

            # Format and send
            message = self.telegram.format_history_message(trades, limit=count)
            await self.telegram.send_message(message, parse_mode=None, message_type="history")

        except Exception as e:
            logger.error(f"Error in history command: {e}")
            await self.telegram.send_message(f"{EMOJI_CROSS} Error getting history: {str(e)[:50]}", parse_mode=None)

    async def _handle_export_command(self, format: str = "csv") -> None:
        """Handle /export command - export data to CSV or Parquet."""
        try:
            from utils.data_exporter import (
                export_trades_csv, export_signals_csv, export_positions_csv,
                export_trades_parquet, generate_export_filename
            )

            format = format.lower()
            if format not in ["csv", "parquet"]:
                format = "csv"

            await self.telegram.send_message(
                f"{EMOJI_HOURGLASS} Generating {format.upper()} export...",
                parse_mode=None
            )

            # Gather data
            trades = self.db.get_recent_trades(limit=500)
            signals = self.db.get_recent_signals(limit=500)
            positions_raw = await self.executor.get_positions()
            positions = [p.to_dict() if hasattr(p, 'to_dict') else p for p in positions_raw]

            if format == "csv":
                # Generate CSV exports
                trades_csv = export_trades_csv(trades)
                signals_csv = export_signals_csv(signals)
                positions_csv = export_positions_csv(positions)

                # Send summary
                summary = (
                    f"{EMOJI_CHECK} CSV Export Ready\n\n"
                    f"Trades: {len(trades)} records\n"
                    f"Signals: {len(signals)} records\n"
                    f"Positions: {len(positions)} records\n\n"
                    f"Use the Logs API to download:\n"
                    f"GET /api/export?token=xxx&format=csv"
                )
                await self.telegram.send_message(summary, parse_mode=None)

            else:
                # Parquet export
                summary = (
                    f"{EMOJI_CHECK} Parquet Export Ready\n\n"
                    f"Trades: {len(trades)} records\n"
                    f"Signals: {len(signals)} records\n"
                    f"Positions: {len(positions)} records\n\n"
                    f"Use the Logs API to download:\n"
                    f"GET /api/export?token=xxx&format=parquet"
                )
                await self.telegram.send_message(summary, parse_mode=None)

        except ImportError as e:
            logger.error(f"Export module not found: {e}")
            await self.telegram.send_message(
                f"{EMOJI_CROSS} Export module not available",
                parse_mode=None
            )
        except Exception as e:
            logger.error(f"Error in export command: {e}")
            await self.telegram.send_message(
                f"{EMOJI_CROSS} Export error: {str(e)[:50]}",
                parse_mode=None
            )

    async def _handle_cost_command(self, days: int = 7) -> None:
        """Handle /cost command - show API cost summary."""
        try:
            # Get cost summary from database
            summary = self.db.get_cost_summary(days=days)

            # Get daily breakdown
            daily = self.db.get_cost_by_day(days=days)
            summary["by_day"] = daily

            # Format and send
            message = self.telegram.format_cost_message(summary)
            await self.telegram.send_message(message, parse_mode=None, message_type="cost_summary")

        except Exception as e:
            logger.error(f"Error in cost command: {e}")
            await self.telegram.send_message(
                f"{EMOJI_CROSS} Error getting cost summary: {str(e)[:50]}",
                parse_mode=None
            )


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
