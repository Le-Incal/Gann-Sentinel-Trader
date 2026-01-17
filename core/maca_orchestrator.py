"""
MACA Orchestrator for Gann Sentinel Trader
Coordinates the Multi-Agent Consensus Architecture 2-phase process.

Version: 2.1.0 - Added Ticker Check + Cleanup
- Phase 1: Parallel thesis generation (Grok, Perplexity, ChatGPT)
- Phase 2: Claude synthesis with direct decision
- Added run_ticker_check() for /check TICKER command
- Removed dead Phase 3/4 code (peer review, final decision)
- Fixed Grok signal parsing (GrokSignal.to_dict(), confidence not conviction)
- Standardized portfolio keys (equity/total_value fallback)
- Per-cycle API cost tracking with aggregate_cycle_costs()

Trade Decision Logic:
- If synthesis.decision_type in ["TRADE", "WATCH"] AND conviction >= 80: proceed = True
- Direct path from synthesis to trade creation
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Tuple

from config import Config

if TYPE_CHECKING:
    from storage.database import Database
    from scanners.grok_scanner import GrokScanner
    from analyzers.perplexity_analyst import PerplexityAnalyst
    from analyzers.chatgpt_analyst import ChatGPTAnalyst
    from analyzers.chatgpt_chair import ChatGPTChair
    from analyzers.claude_technical_validator import ClaudeTechnicalValidator
    from notifications.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class MACAOrchestrator:
    """
    Orchestrates the Multi-Agent Consensus Architecture (MACA) scan cycle.

    Architecture (current):
    - Phase 1: Parallel thesis generation (Grok, Perplexity, ChatGPT)
    - Phase 1b: Committee Debate (2 rounds; each speaker twice; visible log)
    - Phase 2: Chair synthesis (ChatGPT Chair)

    Claude is used as the Technical Validator (check-and-balance), not as chair.
    """

    def __init__(
        self,
        db: "Database",
        grok: "GrokScanner",
        perplexity: "PerplexityAnalyst",
        chatgpt: "ChatGPTAnalyst",
        chair: "ChatGPTChair",
        claude_technical: "ClaudeTechnicalValidator",
        telegram: Optional["TelegramBot"] = None
    ):
        """
        Initialize the MACA orchestrator.

        Args:
            db: Database instance for logging
            grok: Grok scanner for sentiment analysis
            perplexity: Perplexity analyst for fundamental research
            chatgpt: ChatGPT analyst for pattern recognition
            chair: ChatGPT Chair synthesizer
            claude_technical: Claude technical validator (check-and-balance)
            telegram: Optional Telegram bot for notifications
        """
        # Defensive check: ensure db is an instance, not a class
        logger.info(f"MACA init: db type = {type(db)}, is_class = {isinstance(db, type)}")
        if isinstance(db, type):
            logger.warning("MACA received Database class instead of instance - auto-creating instance")
            db = db()
        self.db = db
        self.grok = grok
        self.perplexity = perplexity
        self.chatgpt = chatgpt
        self.chair = chair
        self.claude_technical = claude_technical
        self.telegram = telegram

        # Track API costs per cycle
        self._cycle_costs: Dict[str, Dict[str, Any]] = {}

    @property
    def is_configured(self) -> bool:
        """Check if all AI components are properly configured."""
        components = [self.grok, self.perplexity, self.chatgpt, self.chair, self.claude_technical]
        for component in components:
            if component is None:
                return False
            # Check if component has is_configured attribute
            if hasattr(component, 'is_configured'):
                if not component.is_configured:
                    return False
        return True

    def aggregate_cycle_costs(self) -> Dict[str, Any]:
        """
        Aggregate API costs from current scan cycle.

        Returns:
            Dict with total_tokens, total_cost_usd, and by_source breakdown
        """
        total_tokens = 0
        total_cost = 0.0

        for source, cost_data in self._cycle_costs.items():
            total_tokens += cost_data.get("tokens", 0)
            total_cost += cost_data.get("cost_usd", 0.0)

        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "by_source": self._cycle_costs.copy()
        }

    def reset_cycle_costs(self) -> None:
        """Reset cycle costs for a new scan cycle."""
        self._cycle_costs = {}

    def record_cost(self, source: str, tokens: int, cost_usd: float) -> None:
        """
        Record API cost for a specific AI source.

        Args:
            source: AI source name (grok, perplexity, chatgpt, claude)
            tokens: Number of tokens used
            cost_usd: Cost in USD
        """
        self._cycle_costs[source] = {
            "tokens": tokens,
            "cost_usd": cost_usd
        }

    # Conviction threshold for trade execution
    CONVICTION_THRESHOLD = 80

    # Debate constants
    DEBATE_ROUNDS = 2

    def _build_signal_context(
        self,
        fred_signals: Optional[List[Dict[str, Any]]] = None,
        polymarket_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a compact, analyst-readable signal inventory for prompts.

        This is intentionally short (fits within token budgets) but explicit
        enough that each analyst can:
          - state how many signals were considered
          - rank what mattered
          - explain conflicts
        """

        fred_signals = fred_signals or []
        polymarket_signals = polymarket_signals or []

        def _sig_line(s: Dict[str, Any]) -> str:
            summary = s.get("summary") or s.get("description") or ""
            src = s.get("source") or s.get("source_type") or ""
            conf = s.get("confidence")
            conf_txt = f" (conf={conf:.2f})" if isinstance(conf, (int, float)) else ""
            return f"- [{src}] {summary}{conf_txt}".strip()

        lines: List[str] = []
        lines.append("SIGNAL INVENTORY (for attribution + counts):")
        lines.append(f"- FRED signals: {len(fred_signals)}")
        for s in fred_signals[:6]:
            lines.append(_sig_line(s))
        lines.append(f"- Polymarket signals: {len(polymarket_signals)} (NO sports/entertainment)")
        for s in polymarket_signals[:6]:
            lines.append(_sig_line(s))

        return "\n".join(lines)

    async def run_scan_cycle(
        self,
        portfolio: Dict[str, Any],
        available_cash: float,
        fred_signals: List[Dict[str, Any]],
        polymarket_signals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]] = None,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a simplified 2-phase MACA scan cycle.

        Phase 1: Parallel thesis generation (Grok, Perplexity, ChatGPT)
        Phase 2: Claude synthesis with direct decision

        Args:
            portfolio: Current portfolio state with positions
            available_cash: Cash available for trading
            fred_signals: Macro indicators from FRED
            polymarket_signals: Prediction market data
            technical_analysis: Technical chart analysis
            market_context: Additional market context string

        Returns:
            Result including synthesis and final_decision with proceed_to_execution flag
        """
        # Create scan cycle record
        start_time = datetime.now(timezone.utc)
        cycle_id = self.db.create_scan_cycle({
            "cycle_id": str(uuid.uuid4()),
            "timestamp_utc": start_time.isoformat(),
            "cycle_type": "scheduled",
            "status": "started"
        })

        logger.info(f"Starting MACA scan cycle {cycle_id} (2-phase architecture)")

        try:
            # ================================================================
            # PHASE 1: Parallel thesis generation
            # ================================================================
            proposals = await self._phase1_generate_theses(
                cycle_id=cycle_id,
                portfolio=portfolio,
                available_cash=available_cash,
                market_context=market_context,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals
            )

            phase1_complete = datetime.now(timezone.utc)
            logger.info(f"Phase 1 complete: {len(proposals)} proposals generated")

            # ================================================================
            # PHASE 1B: Debate (optional)
            # ================================================================
            debate, vote_summary, proposals_with_tech, signal_inventory = await self._phase1b_debate(
                cycle_id=cycle_id,
                proposals=proposals,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals,
                technical_analysis=technical_analysis,
            )

            # ================================================================
            # PHASE 2: Chair synthesis
            # ================================================================
            synthesis = await self._phase2_synthesize(
                cycle_id=cycle_id,
                proposals=proposals_with_tech,
                portfolio=portfolio,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals,
                technical_analysis=technical_analysis,
                debate=debate,
                vote_summary=vote_summary,
                signal_inventory=signal_inventory,
            )

            phase2_complete = datetime.now(timezone.utc)

            # ================================================================
            # DEBUG: Log full synthesis response to trace issue
            # ================================================================
            logger.info(f"DEBUG SYNTHESIS: Full response keys = {list(synthesis.keys())}")
            logger.info(f"DEBUG SYNTHESIS: Full response = {synthesis}")

            # Extract conviction from synthesis
            recommendation = synthesis.get("recommendation", {})
            logger.info(f"DEBUG SYNTHESIS: recommendation = {recommendation}")
            logger.info(f"DEBUG SYNTHESIS: recommendation type = {type(recommendation)}")

            conviction = recommendation.get("conviction_score", 0)
            decision_type = synthesis.get("decision_type", "NO_TRADE")

            logger.info(f"DEBUG SYNTHESIS: decision_type = '{decision_type}' (type={type(decision_type)})")
            logger.info(f"DEBUG SYNTHESIS: conviction = {conviction} (type={type(conviction)})")
            logger.info(f"DEBUG SYNTHESIS: CONVICTION_THRESHOLD = {self.CONVICTION_THRESHOLD}")

            logger.info(f"Phase 2 complete: decision_type={decision_type}, conviction={conviction}")

            # ================================================================
            # DIRECT DECISION FROM SYNTHESIS (no Phase 3/4)
            # ================================================================
            # Decision logic: conviction >= 80 AND has valid ticker/side = proceed
            # Note: Claude may return "WATCH" even with high conviction, so we
            # prioritize conviction over decision_type for actionability
            meets_threshold = conviction >= self.CONVICTION_THRESHOLD
            has_ticker = bool(recommendation.get("ticker"))
            has_side = recommendation.get("side") in ["BUY", "SELL"]
            is_actionable = decision_type in ["TRADE", "WATCH"]  # Either TRADE or WATCH can be actionable

            # ------------------------------------------------
            # Consensus failure modes (hard gate)
            # Use debate vote summary as the source of truth.
            # ------------------------------------------------
            consensus = vote_summary or {"hold": False, "reason": ""}

            # Proceed if: high conviction + valid ticker/side + not explicitly NO_TRADE
            proceed = meets_threshold and has_ticker and has_side and is_actionable and not consensus.get("hold", False)

            logger.info(f"DEBUG DECISION: meets_threshold={meets_threshold}, has_ticker={has_ticker}, "
                       f"has_side={has_side}, is_actionable={is_actionable}, proceed={proceed}")

            # If conviction is high but decision_type was WATCH, log a note
            if proceed and decision_type == "WATCH":
                logger.info(f"NOTE: Proceeding with WATCH decision due to high conviction ({conviction})")

            # Build final_decision directly from synthesis
            final_decision = {
                "decision_id": str(uuid.uuid4()),
                "cycle_id": cycle_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "decision_type": decision_type,
                "final_conviction": conviction,
                "proceed_to_execution": proceed,
                "recommendation": recommendation if proceed else None,
                "source": "chair_synthesis",
                "rationale": synthesis.get("rationale", ""),
                "consensus": consensus,
            }

            logger.info(f"DEBUG FINAL_DECISION: proceed_to_execution={final_decision.get('proceed_to_execution')}")
            logger.info(f"DEBUG FINAL_DECISION: recommendation={final_decision.get('recommendation')}")
            logger.info(f"MACA decision: proceed_to_execution={proceed}, conviction={conviction}")

            # Update scan cycle record
            self.db.update_scan_cycle(
                cycle_id=cycle_id,
                status="completed",
                decision_type=decision_type,
                final_conviction=conviction
            )

            # Notify via Telegram ONLY if NOT proceeding to trade
            # If proceeding, agent.py will send notification AFTER creating the trade
            # (so trade_id can be included for approve/reject buttons)
            if self.telegram and not proceed:
                await self._notify_decision(
                    final_decision=final_decision,
                    synthesis=synthesis,
                    proposals=proposals_with_tech,
                    technical_analysis=technical_analysis,
                    portfolio=portfolio
                )

            logger.info(f"MACA cycle {cycle_id} complete: {decision_type} (conviction: {conviction})")

            return {
                "cycle_id": cycle_id,
                "status": "completed",
                "decision_type": decision_type,
                "synthesis": synthesis,
                "proposals": proposals_with_tech,
                "reviews": [],  # No reviews in 2-phase architecture
                "final_decision": final_decision,
                "proceed_to_execution": proceed,
                "timing": {
                    "total_ms": int((phase2_complete - start_time).total_seconds() * 1000),
                    "phase1_ms": int((phase1_complete - start_time).total_seconds() * 1000),
                    "phase2_ms": int((phase2_complete - phase1_complete).total_seconds() * 1000)
                },
                "cycle_cost": self.aggregate_cycle_costs()
            }

        except Exception as e:
            logger.error(f"MACA cycle {cycle_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            self.db.update_scan_cycle(
                cycle_id=cycle_id,
                status="failed",
                decision_type="ERROR",
                final_conviction=0
            )

            return {
                "cycle_id": cycle_id,
                "status": "failed",
                "error": str(e),
                "proceed_to_execution": False,
                "final_decision": None
            }

    async def run_ticker_check(
        self,
        ticker: str,
        portfolio: Dict[str, Any],
        fred_signals: List[Dict[str, Any]],
        polymarket_signals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a MACA check for a specific ticker (used by /check command).

        Same 2-phase architecture as run_scan_cycle() but focused on a single ticker.

        Args:
            ticker: The ticker symbol to analyze
            portfolio: Current portfolio state with positions
            fred_signals: Macro indicators from FRED
            polymarket_signals: Prediction market data
            technical_analysis: Technical chart analysis for the ticker

        Returns:
            Result including synthesis and final_decision with proceed_to_execution flag
        """
        # Create scan cycle record
        start_time = datetime.now(timezone.utc)
        cycle_id = self.db.create_scan_cycle({
            "cycle_id": str(uuid.uuid4()),
            "timestamp_utc": start_time.isoformat(),
            "cycle_type": "ticker_check",
            "status": "started",
            "ticker": ticker
        })

        logger.info(f"Starting MACA ticker check for {ticker} (cycle {cycle_id})")

        try:
            # Get available cash from portfolio
            available_cash = portfolio.get("cash", portfolio.get("available_cash", 100000))

            # Market context focused on specific ticker
            market_context = f"Analyze {ticker} specifically. User requested a detailed check of this ticker."

            # ================================================================
            # PHASE 1: Parallel thesis generation (ticker-focused)
            # ================================================================
            proposals = await self._phase1_generate_theses_for_ticker(
                cycle_id=cycle_id,
                ticker=ticker,
                portfolio=portfolio,
                available_cash=available_cash,
                market_context=market_context
            )

            phase1_complete = datetime.now(timezone.utc)
            logger.info(f"Phase 1 complete for {ticker}: {len(proposals)} proposals generated")

            # ================================================================
            # PHASE 1B: Debate (optional)
            # ================================================================
            debate, vote_summary, proposals_with_tech, signal_inventory = await self._phase1b_debate(
                cycle_id=cycle_id,
                proposals=proposals,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals,
                technical_analysis=technical_analysis,
            )

            # ================================================================
            # PHASE 2: Chair synthesis
            # ================================================================
            synthesis = await self._phase2_synthesize(
                cycle_id=cycle_id,
                proposals=proposals_with_tech,
                portfolio=portfolio,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals,
                technical_analysis=technical_analysis,
                debate=debate,
                vote_summary=vote_summary,
                signal_inventory=signal_inventory,
            )

            phase2_complete = datetime.now(timezone.utc)

            # Extract conviction from synthesis
            recommendation = synthesis.get("recommendation", {})
            conviction = recommendation.get("conviction_score", 0)
            decision_type = synthesis.get("decision_type", "NO_TRADE")

            logger.info(f"Phase 2 complete for {ticker}: decision_type={decision_type}, conviction={conviction}")

            # ================================================================
            # DIRECT DECISION FROM SYNTHESIS
            # ================================================================
            meets_threshold = conviction >= self.CONVICTION_THRESHOLD
            has_ticker = bool(recommendation.get("ticker"))
            has_side = recommendation.get("side") in ["BUY", "SELL"]
            is_actionable = decision_type in ["TRADE", "WATCH"]

            # Use debate vote summary as the source of truth.
            consensus = vote_summary or {"hold": False, "reason": ""}

            proceed = meets_threshold and has_ticker and has_side and is_actionable and not consensus.get("hold", False)

            logger.info(f"Ticker check decision for {ticker}: proceed={proceed}, conviction={conviction}")

            # Build final_decision directly from synthesis
            final_decision = {
                "decision_id": str(uuid.uuid4()),
                "cycle_id": cycle_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "decision_type": decision_type,
                "final_conviction": conviction,
                "proceed_to_execution": proceed,
                "recommendation": recommendation if proceed else None,
                "source": "chair_synthesis",
                "rationale": synthesis.get("rationale", ""),
                "ticker_checked": ticker,
                "consensus": consensus,
            }

            # Update scan cycle record
            self.db.update_scan_cycle(
                cycle_id=cycle_id,
                status="completed",
                decision_type=decision_type,
                final_conviction=conviction
            )

            # Notify via Telegram ONLY if NOT proceeding to trade
            if self.telegram and not proceed:
                await self._notify_decision(
                    final_decision=final_decision,
                    synthesis=synthesis,
                    proposals=proposals_with_tech,
                    technical_analysis=technical_analysis,
                    portfolio=portfolio
                )

            logger.info(f"MACA ticker check {ticker} complete: {decision_type} (conviction: {conviction})")

            return {
                "cycle_id": cycle_id,
                "status": "completed",
                "ticker": ticker,
                "decision_type": decision_type,
                "synthesis": synthesis,
                "proposals": proposals_with_tech,
                "reviews": [],  # No reviews in 2-phase architecture
                "final_decision": final_decision,
                "proceed_to_execution": proceed,
                "timing": {
                    "total_ms": int((phase2_complete - start_time).total_seconds() * 1000),
                    "phase1_ms": int((phase1_complete - start_time).total_seconds() * 1000),
                    "phase2_ms": int((phase2_complete - phase1_complete).total_seconds() * 1000)
                },
                "cycle_cost": self.aggregate_cycle_costs()
            }

        except Exception as e:
            logger.error(f"MACA ticker check {ticker} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            self.db.update_scan_cycle(
                cycle_id=cycle_id,
                status="failed",
                decision_type="ERROR",
                final_conviction=0
            )

            return {
                "cycle_id": cycle_id,
                "status": "failed",
                "ticker": ticker,
                "error": str(e),
                "proceed_to_execution": False,
                "final_decision": None
            }

    async def _phase1_generate_theses_for_ticker(
        self,
        cycle_id: str,
        ticker: str,
        portfolio: Dict[str, Any],
        available_cash: float,
        market_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Generate thesis proposals for a specific ticker.

        Similar to _phase1_generate_theses but focused on a single ticker.
        """
        logger.info(f"Phase 1: Generating theses for {ticker} (cycle {cycle_id})")

        # Prepare portfolio summary
        portfolio_summary = {
            "positions": portfolio.get("positions", []),
            "total_value": portfolio.get("total_value", portfolio.get("equity", 0)),
            "cash": available_cash
        }

        # Ticker-specific market context
        ticker_context = f"{market_context or ''}\n\nFocus on analyzing {ticker} - this is a specific ticker check request."

        tasks = []

        # Grok thesis for specific ticker
        if hasattr(self.grok, 'check_ticker'):
            tasks.append(self._grok_ticker_check_adapter(
                cycle_id=cycle_id,
                ticker=ticker,
                portfolio_summary=portfolio_summary,
                available_cash=available_cash
            ))
        else:
            # Fallback to market overview if no ticker-specific method
            tasks.append(self._grok_thesis_adapter(
                cycle_id=cycle_id,
                portfolio_summary=portfolio_summary,
                available_cash=available_cash,
                market_context=ticker_context
            ))

        # Perplexity thesis - pass ticker context
        tasks.append(self._safe_generate_thesis(
            "perplexity",
            self.perplexity.generate_thesis,
            portfolio_summary=portfolio_summary,
            available_cash=available_cash,
            scan_cycle_id=cycle_id,
            additional_context=f"Focus analysis on {ticker}. User has specifically requested a check on this stock."
        ))

        # ChatGPT thesis - pass ticker context
        tasks.append(self._safe_generate_thesis(
            "chatgpt",
            self.chatgpt.generate_thesis,
            portfolio_summary=portfolio_summary,
            available_cash=available_cash,
            scan_cycle_id=cycle_id,
            market_context=ticker_context
        ))

        # Wait for all with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals = []
        for i, result in enumerate(results):
            source = ["grok", "perplexity", "chatgpt"][i]
            if isinstance(result, Exception):
                logger.error(f"{source} thesis generation for {ticker} failed: {result}")
                proposals.append(self._empty_proposal(cycle_id, source, str(result)))
            else:
                proposals.append(result)
                self.db.save_ai_proposal(result)

        return proposals

    async def _grok_ticker_check_adapter(
        self,
        cycle_id: str,
        ticker: str,
        portfolio_summary: Dict[str, Any],
        available_cash: float
    ) -> Dict[str, Any]:
        """
        Adapter for Grok scanner to check a specific ticker.

        Uses check_ticker() if available, otherwise falls back to market overview.
        """
        try:
            # Try ticker-specific check
            if hasattr(self.grok, 'check_ticker'):
                signals = await asyncio.wait_for(
                    self.grok.check_ticker(ticker),
                    timeout=30.0
                )
            else:
                # Fallback to market overview
                return await self._grok_thesis_adapter(
                    cycle_id=cycle_id,
                    portfolio_summary=portfolio_summary,
                    available_cash=available_cash,
                    market_context=f"Focus on {ticker}"
                )

            # Handle empty signals
            if not signals:
                logger.warning(f"Grok returned no signals for {ticker}")
                return self._empty_proposal(cycle_id, "grok", f"No signals for {ticker}")

            # Convert GrokSignal objects to dicts
            signals_dicts = []
            for s in signals:
                if hasattr(s, 'to_dict'):
                    signals_dicts.append(s.to_dict())
                elif isinstance(s, dict):
                    signals_dicts.append(s)

            if not signals_dicts:
                return self._empty_proposal(cycle_id, "grok", f"No valid signals after conversion for {ticker}")

            # Find best signal for this ticker
            best_signal = max(
                signals_dicts,
                key=lambda s: s.get("confidence", 0),
                default=None
            )

            if not best_signal:
                return self._empty_proposal(cycle_id, "grok", f"No best signal found for {ticker}")

            # Convert confidence (0-1) to conviction_score (0-100)
            confidence_raw = best_signal.get("confidence", 0.5)
            conviction_score = int(confidence_raw * 100)

            # Map directional_bias to side
            bias = best_signal.get("directional_bias", "neutral")
            if bias == "bullish":
                side = "BUY"
            elif bias == "bearish":
                side = "SELL"
            else:
                side = None

            logger.info(f"Grok ticker check for {ticker}: conviction={conviction_score}, side={side}")

            return {
                "schema_version": "1.0.0",
                "proposal_id": str(uuid.uuid4()),
                "ai_source": "grok",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scan_cycle_id": cycle_id,
                "proposal_type": "NEW_POSITION" if side else "NO_OPPORTUNITY",
                "recommendation": {
                    "ticker": ticker,
                    "side": side,
                    "conviction_score": conviction_score,
                    "thesis": best_signal.get("narrative", f"Grok analysis of {ticker}"),
                    "time_horizon": best_signal.get("time_horizon"),
                    "catalyst": best_signal.get("event_type"),
                    "catalyst_deadline": best_signal.get("validity", {}).get("expires_at")
                },
                "supporting_evidence": {
                    "signal_source": best_signal.get("source", "grok"),
                    "event_type": best_signal.get("event_type"),
                    "raw_confidence": confidence_raw,
                    "signals_count": len(signals_dicts)
                },
                "raw_data": best_signal,
                "time_sensitive": best_signal.get("validity", {}).get("requires_immediate_action", False),
                "metadata": {
                    "model": "grok-3-fast-beta",
                    "adapter": "grok_ticker_check_adapter",
                    "ticker_requested": ticker
                }
            }

        except asyncio.TimeoutError:
            logger.warning(f"Grok ticker check for {ticker} timed out")
            return self._empty_proposal(cycle_id, "grok", f"Timeout checking {ticker}")
        except Exception as e:
            logger.error(f"Grok ticker check adapter error for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._empty_proposal(cycle_id, "grok", str(e))

    async def _phase1_generate_theses(
        self,
        cycle_id: str,
        portfolio: Dict[str, Any],
        available_cash: float,
        market_context: Optional[str] = None,
        fred_signals: Optional[List[Dict[str, Any]]] = None,
        polymarket_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Generate thesis proposals from all AI sources in parallel.

        Args:
            cycle_id: Current scan cycle ID
            portfolio: Current portfolio state
            available_cash: Available cash for trading
            market_context: Additional market context

        Returns:
            List of thesis proposals from Grok, Perplexity, and ChatGPT
        """
        logger.info(f"Phase 1: Generating theses for cycle {cycle_id}")

        # Prepare portfolio summary for all AIs
        # Use total_value with equity fallback for consistency (Alpaca returns equity)
        portfolio_summary = {
            "positions": portfolio.get("positions", []),
            "total_value": portfolio.get("total_value", portfolio.get("equity", 0)),
            "cash": available_cash
        }

        # Build shared context so each analyst can explicitly attribute signals.
        signal_context = self._build_signal_context(
            fred_signals=fred_signals,
            polymarket_signals=polymarket_signals,
        )

        combined_context = "\n\n".join([c for c in [market_context, signal_context] if c])

        # Generate theses in parallel with timeout handling
        tasks = []

        # Grok thesis (if available)
        if hasattr(self.grok, 'generate_thesis'):
            tasks.append(self._safe_generate_thesis(
                "grok",
                self.grok.generate_thesis,
                portfolio_summary=portfolio_summary,
                available_cash=available_cash,
                scan_cycle_id=cycle_id,
                market_context=combined_context
            ))
        else:
            # Grok scanner might use different method signature
            tasks.append(self._grok_thesis_adapter(
                cycle_id=cycle_id,
                portfolio_summary=portfolio_summary,
                available_cash=available_cash,
                market_context=market_context
            ))

        # Perplexity thesis
        tasks.append(self._safe_generate_thesis(
            "perplexity",
            self.perplexity.generate_thesis,
            portfolio_summary=portfolio_summary,
            available_cash=available_cash,
            scan_cycle_id=cycle_id
            ,additional_context=signal_context
        ))

        # ChatGPT thesis
        tasks.append(self._safe_generate_thesis(
            "chatgpt",
            self.chatgpt.generate_thesis,
            portfolio_summary=portfolio_summary,
            available_cash=available_cash,
            scan_cycle_id=cycle_id,
            market_context=combined_context,
            additional_context=signal_context
        ))

        # Wait for all with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals = []
        for i, result in enumerate(results):
            source = ["grok", "perplexity", "chatgpt"][i]
            if isinstance(result, Exception):
                logger.error(f"{source} thesis generation failed: {result}")
                proposals.append(self._empty_proposal(cycle_id, source, str(result)))
            else:
                proposals.append(result)
                # Save to database
                self.db.save_ai_proposal(result)

        return proposals

    async def _phase1b_debate(
        self,
        *,
        cycle_id: str,
        proposals: List[Dict[str, Any]],
        fred_signals: List[Dict[str, Any]],
        polymarket_signals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Optional committee debate.

        Returns: (debate_summary, vote_summary, proposals_with_tech, signal_inventory)
        """

        # Signal inventory for explainability/debuggability.
        signal_inventory: Dict[str, Any] = {
            "by_source": {
                "FRED": len(fred_signals or []),
                "Polymarket": len(polymarket_signals or []),
                "Technical": 1 if technical_analysis else 0,
            },
            "total": int(len(fred_signals or []) + len(polymarket_signals or []) + (1 if technical_analysis else 0)),
        }

        if not Config.DEBATE_ENABLED:
            return {}, {"hold": False, "reason": "Debate disabled"}, proposals, signal_inventory

        # Determine a candidate ticker/side from the initial proposals.
        candidate_ticker, candidate_side = self._pick_candidate_from_proposals(proposals)

        # Create DB session for the debate transcript.
        session_id = None
        try:
            session_id = self.db.create_debate_session(cycle_id)
        except Exception as e:
            logger.warning(f"Could not create debate session: {e}")

        # Add technical validator as a committee member (round 0).
        tech_turn0 = await self.claude_technical.initial_vote(
            scan_cycle_id=cycle_id,
            candidate_ticker=candidate_ticker,
            candidate_side=candidate_side,
            technical_analysis=technical_analysis,
        )
        if session_id:
            try:
                self.db.save_debate_turn(session_id, cycle_id, tech_turn0)
            except Exception:
                pass

        tech_proposal = self._technical_turn_to_proposal(cycle_id, tech_turn0)
        proposals_with_tech = list(proposals or []) + [tech_proposal]

        # Seed "own_thesis" payloads.
        base_theses = [self._proposal_to_thesis_stub(p) for p in proposals_with_tech]

        # Track last turn by speaker.
        last_turns: Dict[str, Dict[str, Any]] = {
            "grok": {"speaker": "grok", "round": 0, "vote": self._proposal_to_vote(self._get_by_ai(proposals_with_tech, "grok"))},
            "perplexity": {"speaker": "perplexity", "round": 0, "vote": self._proposal_to_vote(self._get_by_ai(proposals_with_tech, "perplexity"))},
            "chatgpt": {"speaker": "chatgpt", "round": 0, "vote": self._proposal_to_vote(self._get_by_ai(proposals_with_tech, "chatgpt"))},
            "claude_technical": tech_turn0,
        }

        rounds: List[Dict[str, Any]] = []

        # Debate rounds: each speaker speaks once per round.
        for r in range(1, max(1, Config.DEBATE_ROUNDS) + 1):
            round_turns: List[Dict[str, Any]] = []

            # Prepare other theses (latest known) for each speaker.
            for speaker in ["grok", "perplexity", "chatgpt", "claude_technical"]:
                own = self._speaker_own_thesis_stub(speaker, proposals_with_tech, tech_turn0)
                others = [t for t in base_theses if t.get("speaker") != speaker]

                try:
                    if speaker == "grok":
                        turn = await self.grok.debate(scan_cycle_id=cycle_id, round_num=r, own_thesis=own, other_theses=others)
                    elif speaker == "perplexity":
                        turn = await self.perplexity.debate(scan_cycle_id=cycle_id, round_num=r, own_thesis=own, other_theses=others)
                    elif speaker == "chatgpt":
                        turn = await self.chatgpt.debate(scan_cycle_id=cycle_id, round_num=r, own_thesis=own, other_theses=others)
                    else:
                        turn = await self.claude_technical.debate(
                            scan_cycle_id=cycle_id,
                            round_num=r,
                            own_thesis=own,
                            other_theses=others,
                            technical_analysis=technical_analysis,
                        )
                except Exception as e:
                    turn = {
                        "speaker": speaker,
                        "round": r,
                        "message": f"Debate error: {e}",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                    }

                # Normalize and persist.
                turn.setdefault("speaker", speaker)
                turn.setdefault("round", r)
                if session_id:
                    try:
                        self.db.save_debate_turn(session_id, cycle_id, turn)
                    except Exception:
                        pass
                last_turns[speaker] = turn
                round_turns.append(turn)

            rounds.append({"round": r, "turns": round_turns})

        debate_summary = {
            "session_id": session_id,
            "rounds": rounds,
        }

        vote_summary = self._summarize_votes(last_turns, tech_turn0)

        # Telegram: optionally show the debate as its own message.
        if self.telegram and debate_summary.get("rounds"):
            try:
                await self.telegram.send_maca_debate_summary(
                    cycle_id=cycle_id,
                    debate=debate_summary,
                    vote_summary=vote_summary,
                )
            except Exception:
                pass

        return debate_summary, vote_summary, proposals_with_tech, signal_inventory

    def _get_by_ai(self, proposals: List[Dict[str, Any]], ai_source: str) -> Optional[Dict[str, Any]]:
        for p in proposals or []:
            if (p.get("ai_source") or p.get("speaker")) == ai_source:
                return p
            # Some proposals store ai_source under metadata
            if p.get("metadata", {}).get("ai_source") == ai_source:
                return p
        return None

    def _proposal_to_vote(self, proposal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not proposal:
            return {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0}
        rec = proposal.get("recommendation") or {}
        ticker = rec.get("ticker") or proposal.get("ticker")
        side = rec.get("side") or proposal.get("side")
        conviction = rec.get("conviction_score")
        # Some proposals store confidence directly
        conf = proposal.get("confidence")
        if conf is None and isinstance(conviction, (int, float)):
            conf = float(conviction) / 100.0
        action = "HOLD"
        if side in ["BUY", "SELL"] and ticker:
            action = side
        return {
            "action": action,
            "ticker": ticker,
            "side": side if side in ["BUY", "SELL"] else None,
            "confidence": float(conf) if isinstance(conf, (int, float)) else 0.0,
        }

    def _proposal_to_thesis_stub(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        rec = proposal.get("recommendation") or {}
        speaker = proposal.get("ai_source") or proposal.get("speaker") or "unknown"
        return {
            "speaker": speaker,
            "proposal_type": proposal.get("proposal_type"),
            "vote": self._proposal_to_vote(proposal),
            "thesis": rec.get("thesis") or proposal.get("thesis") or "",
            "key_signals": (proposal.get("supporting_evidence") or {}).get("key_signals") or [],
            "signals_count": (proposal.get("supporting_evidence") or {}).get("signals_count"),
        }

    def _technical_turn_to_proposal(self, scan_cycle_id: str, tech_turn: Dict[str, Any]) -> Dict[str, Any]:
        vote = tech_turn.get("vote") or {}
        return {
            "schema_version": "1.0.0",
            "proposal_id": str(uuid.uuid4()),
            "ai_source": "claude_technical",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scan_cycle_id": scan_cycle_id,
            "proposal_type": "TECHNICAL_VALIDATION",
            "recommendation": {
                "ticker": vote.get("ticker"),
                "side": vote.get("action") if vote.get("action") in ["BUY", "SELL"] else None,
                "conviction_score": int(round(float(vote.get("confidence", 0.0)) * 100)),
                "thesis": tech_turn.get("message") or "",
                "time_horizon": "unspecified",
                "catalyst": None,
                "catalyst_deadline": None,
            },
            "supporting_evidence": {
                "technical_verdict": tech_turn.get("verdict"),
                "invalidation": tech_turn.get("invalidation"),
            },
            "raw_data": tech_turn,
            "time_sensitive": False,
            "metadata": {"role": "technical_validator"},
        }

    def _pick_candidate_from_proposals(self, proposals: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        counts: Dict[Tuple[str, str], int] = {}
        for p in proposals or []:
            v = self._proposal_to_vote(p)
            if v.get("action") in ["BUY", "SELL"] and v.get("ticker"):
                key = (v.get("ticker"), v.get("action"))
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None, None
        (ticker, action), _ = max(counts.items(), key=lambda kv: kv[1])
        return ticker, action

    def _speaker_own_thesis_stub(self, speaker: str, proposals_with_tech: List[Dict[str, Any]], tech_turn0: Dict[str, Any]) -> Dict[str, Any]:
        if speaker == "claude_technical":
            return {
                "speaker": "claude_technical",
                "vote": (tech_turn0.get("vote") or {}),
                "thesis": tech_turn0.get("message") or "",
                "technical_verdict": tech_turn0.get("verdict"),
            }
        p = self._get_by_ai(proposals_with_tech, speaker)
        return self._proposal_to_thesis_stub(p or {"ai_source": speaker, "recommendation": {}, "supporting_evidence": {}})

    def _summarize_votes(self, last_turns: Dict[str, Dict[str, Any]], tech_turn0: Dict[str, Any]) -> Dict[str, Any]:
        """Compute majority/tie, confidence aggregates, and hard-gate failure modes."""

        speakers = ["grok", "perplexity", "chatgpt", "claude_technical"]
        votes: List[Dict[str, Any]] = []
        for s in speakers:
            t = last_turns.get(s) or {}
            v = t.get("vote") or {}
            votes.append({
                "speaker": s,
                "action": (v.get("action") or "HOLD").upper(),
                "ticker": v.get("ticker"),
                "side": v.get("side") or (v.get("action") if v.get("action") in ["BUY", "SELL"] else None),
                "confidence": float(v.get("confidence")) if isinstance(v.get("confidence"), (int, float)) else 0.0,
            })

        # Count votes by (action,ticker)
        bucket: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        for v in votes:
            key = (v["action"], v.get("ticker"))
            if key not in bucket:
                bucket[key] = {"count": 0, "confidence_sum": 0.0}
            bucket[key]["count"] += 1
            bucket[key]["confidence_sum"] += float(v.get("confidence", 0.0))

        n = len(votes)
        majority_required = (n // 2) + 1
        top_key = None
        top_count = 0
        for k, d in bucket.items():
            if d["count"] > top_count:
                top_key, top_count = k, d["count"]

        avg_conf = sum(v["confidence"] for v in votes) / max(1, n)

        # Determine failure modes / tie handling
        hold = False
        reason = ""

        # Identify tie: 2-2 split across two distinct keys
        sorted_counts = sorted([d["count"] for d in bucket.values()], reverse=True)
        is_tie_2_2 = (sorted_counts[:2] == [2, 2])
        has_majority = top_count >= majority_required

        if not has_majority and not is_tie_2_2:
            hold = True
            reason = "No majority consensus (fragmented votes)"

        if avg_conf < float(Config.DEBATE_MIN_AVG_CONFIDENCE):
            hold = True
            reason = reason or f"Low average confidence ({avg_conf:.2f})"

        # Technical check-and-balance:
        tech_verdict = (tech_turn0.get("verdict") or "unknown")
        if Config.TECH_INVALIDATION_SUPERMAJORITY and tech_verdict in ["no_trade", "analyze_only"]:
            # Require supermajority (3/4) to proceed with a trade recommendation.
            if top_key and top_key[0] in ["BUY", "SELL"] and top_count < 3:
                hold = True
                reason = reason or f"Technical validator restricts trading (verdict={tech_verdict}); needs supermajority"

        # If it's a clean 2-2 tie, allow chair tie-breaker (do not HOLD here).
        if is_tie_2_2 and not hold:
            reason = "Vote tie (2-2); Chair will break"

        return {
            "n": n,
            "votes": votes,
            "buckets": {f"{k[0]}:{k[1] or 'NA'}": v for k, v in bucket.items()},
            "top": {"action": top_key[0], "ticker": top_key[1], "count": top_count} if top_key else None,
            "avg_confidence": avg_conf,
            "tie_2_2": is_tie_2_2,
            "hold": hold,
            "reason": reason,
            "technical_verdict": tech_verdict,
        }

    async def _safe_generate_thesis(
        self,
        source: str,
        generate_func,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Safely call thesis generation with timeout.

        Args:
            source: AI source name
            generate_func: The generate_thesis function to call
            **kwargs: Arguments to pass to generate_func

        Returns:
            Thesis proposal or error placeholder
        """
        try:
            return await asyncio.wait_for(
                generate_func(**kwargs),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"{source} thesis generation timed out")
            return self._empty_proposal(kwargs.get("scan_cycle_id", ""), source, "Timeout")
        except Exception as e:
            logger.error(f"{source} thesis generation error: {e}")
            return self._empty_proposal(kwargs.get("scan_cycle_id", ""), source, str(e))

    async def _grok_thesis_adapter(
        self,
        cycle_id: str,
        portfolio_summary: Dict[str, Any],
        available_cash: float,
        market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adapter for Grok scanner to match thesis generation interface.

        The Grok scanner returns GrokSignal objects which need to be converted
        to the standard ThesisProposal schema.

        Key conversions:
        - GrokSignal.confidence (0-1) -> conviction_score (0-100)
        - GrokSignal.to_dict() for serialization
        """
        try:
            # Grok scanner uses scan_market_overview for general market thesis
            if not hasattr(self.grok, 'scan_market_overview'):
                return self._empty_proposal(cycle_id, "grok", "No compatible method found")

            signals = await asyncio.wait_for(
                self.grok.scan_market_overview(),
                timeout=30.0
            )

            # Handle empty signals
            if not signals:
                logger.warning("Grok returned no signals")
                return self._empty_proposal(cycle_id, "grok", "No signals returned")

            # Convert GrokSignal objects to dicts
            signals_dicts = []
            for s in signals:
                if hasattr(s, 'to_dict'):
                    signals_dicts.append(s.to_dict())
                elif isinstance(s, dict):
                    signals_dicts.append(s)
                else:
                    logger.warning(f"Unknown signal type: {type(s)}")

            if not signals_dicts:
                return self._empty_proposal(cycle_id, "grok", "No valid signals after conversion")

            # Find highest CONFIDENCE signal (not conviction - GrokSignal uses confidence 0-1)
            best_signal = max(
                signals_dicts,
                key=lambda s: s.get("confidence", 0),
                default=None
            )

            if not best_signal:
                return self._empty_proposal(cycle_id, "grok", "No best signal found")

            # Debug: Log full signal structure to diagnose issues
            logger.info(f"DEBUG Grok best_signal keys: {list(best_signal.keys())}")
            logger.info(f"DEBUG Grok asset_scope: {best_signal.get('asset_scope')}")
            logger.info(f"DEBUG Grok full signal: {best_signal}")

            # Extract ticker from asset_scope (defensive)
            asset_scope = best_signal.get("asset_scope", {})
            if asset_scope is None:
                asset_scope = {}
            tickers = asset_scope.get("tickers", [])
            if tickers is None:
                tickers = []
            ticker = tickers[0] if tickers else None

            logger.info(f"Grok best signal: ticker={ticker}, "
                       f"confidence={best_signal.get('confidence', 0)}, "
                       f"bias={best_signal.get('directional_bias', 'N/A')}")

            # Convert confidence (0-1) to conviction_score (0-100)
            confidence_raw = best_signal.get("confidence", 0.5)
            conviction_score = int(confidence_raw * 100)

            # Map directional_bias to side
            bias = best_signal.get("directional_bias", "neutral")
            if bias == "bullish":
                side = "BUY"
            elif bias == "bearish":
                side = "SELL"
            else:
                side = None

            # Build thesis proposal from Grok signal
            return {
                "schema_version": "1.0.0",
                "proposal_id": str(uuid.uuid4()),
                "ai_source": "grok",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scan_cycle_id": cycle_id,
                "proposal_type": "NEW_POSITION" if ticker and side else "NO_OPPORTUNITY",
                "recommendation": {
                    "ticker": ticker,
                    "side": side,
                    "conviction_score": conviction_score,
                    "thesis": best_signal.get("narrative", "Grok market signal"),
                    "time_horizon": best_signal.get("time_horizon"),
                    "catalyst": best_signal.get("event_type"),
                    "catalyst_deadline": best_signal.get("validity", {}).get("expires_at")
                },
                "supporting_evidence": {
                    "signal_source": best_signal.get("source", "grok"),
                    "event_type": best_signal.get("event_type"),
                    "raw_confidence": confidence_raw,
                    "signals_count": len(signals_dicts)
                },
                "raw_data": best_signal,
                "time_sensitive": best_signal.get("validity", {}).get("requires_immediate_action", False),
                "metadata": {
                    "model": "grok-3-fast-beta",
                    "adapter": "grok_thesis_adapter_v2"
                }
            }

        except asyncio.TimeoutError:
            logger.warning("Grok thesis generation timed out")
            return self._empty_proposal(cycle_id, "grok", "Timeout")
        except Exception as e:
            logger.error(f"Grok adapter error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._empty_proposal(cycle_id, "grok", str(e))

    async def _phase2_synthesize(
        self,
        cycle_id: str,
        proposals: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        fred_signals: List[Dict[str, Any]],
        polymarket_signals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]],
        debate: Optional[Dict[str, Any]] = None,
        vote_summary: Optional[Dict[str, Any]] = None,
        signal_inventory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Phase 2: Chair synthesizes proposals + debate into a final thesis."""

        logger.info(f"Phase 2: Chair synthesis for cycle {cycle_id}")

        try:
            chair_out = await asyncio.wait_for(
                self.chair.synthesize(
                    cycle_id=cycle_id,
                    proposals=proposals,
                    debate=debate,
                    signal_inventory=signal_inventory,
                    technical_analysis=technical_analysis,
                    vote_summary=vote_summary,
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.error("Chair synthesis timed out")
            return self._empty_synthesis(cycle_id, "Chair synthesis timeout")
        except Exception as e:
            logger.error(f"Chair synthesis error: {e}")
            return self._empty_synthesis(cycle_id, str(e))

        # Normalize into the legacy synthesis structure expected by the rest of the pipeline.
        final_thesis = (chair_out or {}).get("final_thesis", {})
        action = (final_thesis.get("action") or "HOLD").upper()
        ticker = final_thesis.get("ticker")
        side = action if action in ["BUY", "SELL"] else None
        confidence = final_thesis.get("confidence")
        conviction_score = int(round(float(confidence) * 100)) if isinstance(confidence, (int, float)) else 0

        decision_type = (chair_out or {}).get("decision_type")
        if not decision_type:
            decision_type = "TRADE" if action in ["BUY", "SELL"] else "NO_TRADE"

        synthesis = {
            "decision_type": decision_type,
            "rationale": (final_thesis.get("description") or "")[:2000],
            "recommendation": {
                "ticker": ticker,
                "side": side,
                "conviction_score": conviction_score,
                "thesis": final_thesis.get("summary") or final_thesis.get("description") or "",
                "time_horizon": final_thesis.get("time_horizon") or "unspecified",
                "catalyst": None,
                "catalyst_deadline": None,
            },
            "final_thesis": final_thesis,
            "committee_notes": (chair_out or {}).get("committee_notes", {}),
            "tie_break_used": bool((chair_out or {}).get("tie_break_used")),
            "vote_summary": vote_summary or {},
            "debate": debate or {},
        }
        return synthesis

    async def _notify_decision(
        self,
        final_decision: Dict[str, Any],
        synthesis: Dict[str, Any],
        proposals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send Telegram notification for MACA decision (2-phase architecture).

        Uses send_maca_scan_summary() for rich display with:
        - AI Council views (all proposals)
        - Chart analysis with technical signals
        - Claude's synthesis decision
        - Inline approve/reject buttons (if actionable)
        """
        try:
            # Build technical signals list for display
            technical_signals = []
            if technical_analysis:
                if isinstance(technical_analysis, dict):
                    if "ticker" in technical_analysis:
                        technical_signals.append(technical_analysis)
                    else:
                        for key, value in technical_analysis.items():
                            if isinstance(value, dict) and "ticker" in value:
                                technical_signals.append(value)

            # Use the MACA scan summary method
            await self.telegram.send_maca_scan_summary(
                proposals=proposals,
                synthesis=synthesis,
                technical_signals=technical_signals,
                portfolio=portfolio or {},
                trade_id=None  # Trade ID comes from agent.py after trade creation
            )

            logger.info(f"MACA notification sent: decision_type={final_decision.get('decision_type')}, "
                       f"conviction={final_decision.get('final_conviction')}")

        except Exception as e:
            logger.error(f"Failed to send MACA decision notification: {e}")
            # Fallback to simple message
            try:
                rec = synthesis.get("recommendation", {})
                ticker = rec.get("ticker", "N/A")
                side = rec.get("side", "N/A")
                conviction = final_decision.get("final_conviction", 0)
                proceed = final_decision.get("proceed_to_execution", False)

                if proceed:
                    message = (
                        f"MACA Trade Signal\n\n"
                        f"Ticker: {ticker}\n"
                        f"Side: {side}\n"
                        f"Conviction: {conviction}/100\n"
                        f"Status: ACTIONABLE\n\n"
                        f"Thesis: {rec.get('thesis', 'N/A')[:200]}"
                    )
                else:
                    message = (
                        f"MACA Cycle Complete\n\n"
                        f"Decision: NO TRADE\n"
                        f"Highest Conviction: {conviction}/100\n"
                        f"Rationale: {synthesis.get('rationale', 'Below threshold')[:200]}"
                    )

                await self.telegram.send_message(
                    message,
                    message_type="maca_decision"
                )
            except Exception as fallback_error:
                logger.error(f"Fallback notification also failed: {fallback_error}")

    def _empty_proposal(self, cycle_id: str, source: str, reason: str) -> Dict[str, Any]:
        """Return empty proposal when generation fails."""
        return {
            "schema_version": "1.0.0",
            "proposal_id": str(uuid.uuid4()),
            "ai_source": source,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scan_cycle_id": cycle_id,
            "proposal_type": "NO_OPPORTUNITY",
            "recommendation": {
                "ticker": None,
                "side": None,
                "conviction_score": 0,
                "thesis": f"Generation failed: {reason}",
                "time_horizon": None,
                "catalyst": None,
                "catalyst_deadline": None
            },
            "rotation_details": {},
            "supporting_evidence": {},
            "raw_data": {"error": reason},
            "time_sensitive": False,
            "metadata": {"error": reason}
        }

    def _empty_synthesis(self, cycle_id: str, reason: str) -> Dict[str, Any]:
        """Return empty synthesis when generation fails."""
        return {
            "schema_version": "1.0.0",
            "synthesis_id": str(uuid.uuid4()),
            "scan_cycle_id": cycle_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decision_type": "NO_TRADE",
            "selected_proposal": {},
            "recommendation": {
                "ticker": None,
                "side": None,
                "conviction_score": 0,
                "thesis": f"Synthesis failed: {reason}",
                "position_size_pct": 0,
                "stop_loss_pct": 0,
                "time_horizon": None
            },
            "cross_validation": {},
            "proposal_evaluation": [],
            "proceed_to_review": False,
            "time_sensitive_override": False,
            "rationale": f"Error: {reason}",
            "error": reason
        }

    async def send_maca_summary(self, maca_result: Dict[str, Any]) -> None:
        """
        Send MACA scan summary to Telegram.

        Extracts components from maca_result and calls send_maca_scan_summary
        with proper parameters including trade_id for approve/reject buttons.
        """
        if not self.telegram or not hasattr(self.telegram, 'send_maca_scan_summary'):
            logger.warning("Telegram bot not configured or missing send_maca_scan_summary method")
            return

        try:
            # Extract components from maca_result
            proposals = maca_result.get("proposals", [])
            synthesis = maca_result.get("synthesis", {})
            final_decision = maca_result.get("final_decision", {})
            portfolio = maca_result.get("portfolio", {})

            # Get trade_id if trade was created
            trade_id = final_decision.get("trade_id")

            # Build technical signals list
            technical_signals = []
            tech_analysis = maca_result.get("technical_analysis", [])
            if isinstance(tech_analysis, list):
                technical_signals = tech_analysis
            elif isinstance(tech_analysis, dict):
                technical_signals = [tech_analysis] if tech_analysis else []

            logger.info(f"Sending MACA summary: trade_id={trade_id}, "
                       f"proceed={final_decision.get('proceed_to_execution')}")

            await self.telegram.send_maca_scan_summary(
                proposals=proposals,
                synthesis=synthesis,
                technical_signals=technical_signals,
                portfolio=portfolio,
                trade_id=trade_id
            )
        except Exception as e:
            logger.error(f"Failed to send MACA summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
