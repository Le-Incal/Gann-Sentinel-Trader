"""
MACA Orchestrator for Gann Sentinel Trader
Coordinates the Multi-Agent Consensus Architecture 4-phase process.

Version: 1.2.0 - Per-Cycle Cost Tracking
- Uses send_maca_scan_summary() for AI Council + Decision display
- Inline buttons for approve/reject
- Enhanced notification with technical signals and portfolio
- Per-cycle API cost tracking with aggregate_cycle_costs()
- is_configured property for component status check

Phases:
1. Parallel thesis generation (Grok, Perplexity, ChatGPT)
2. Claude synthesis
3. Peer review (all three AIs)
4. Final decision (Claude)
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from storage.database import Database
    from scanners.grok_scanner import GrokScanner
    from analyzers.perplexity_analyst import PerplexityAnalyst
    from analyzers.chatgpt_analyst import ChatGPTAnalyst
    from analyzers.claude_maca_extension import ClaudeMACAMixin
    from notifications.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class MACAOrchestrator:
    """
    Orchestrates the Multi-Agent Consensus Architecture (MACA) scan cycle.

    Coordinates Grok, Perplexity, and ChatGPT for thesis generation,
    with Claude as the Chief Investment Officer for synthesis and final decisions.
    """

    def __init__(
        self,
        db: "Database",
        grok: "GrokScanner",
        perplexity: "PerplexityAnalyst",
        chatgpt: "ChatGPTAnalyst",
        claude: "ClaudeMACAMixin",
        telegram: Optional["TelegramBot"] = None
    ):
        """
        Initialize the MACA orchestrator.

        Args:
            db: Database instance for logging
            grok: Grok scanner for sentiment analysis
            perplexity: Perplexity analyst for fundamental research
            chatgpt: ChatGPT analyst for pattern recognition
            claude: Claude analyst with MACA synthesis capability
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
        self.claude = claude
        self.telegram = telegram

        # Track API costs per cycle
        self._cycle_costs: Dict[str, Dict[str, Any]] = {}

    @property
    def is_configured(self) -> bool:
        """Check if all AI components are properly configured."""
        components = [self.grok, self.perplexity, self.chatgpt, self.claude]
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
        Execute a full MACA scan cycle.

        Args:
            portfolio: Current portfolio state with positions
            available_cash: Cash available for trading
            fred_signals: Macro indicators from FRED
            polymarket_signals: Prediction market data
            technical_analysis: Technical chart analysis
            market_context: Additional market context string

        Returns:
            Final decision result including all phase outputs
        """
        # Create scan cycle record
        start_time = datetime.now(timezone.utc)
        cycle_id = self.db.create_scan_cycle({
            "cycle_id": str(uuid.uuid4()),
            "timestamp_utc": start_time.isoformat(),
            "cycle_type": "scheduled",
            "status": "started"
        })

        logger.info(f"Starting MACA scan cycle {cycle_id}")

        try:
            # Phase 1: Parallel thesis generation
            proposals = await self._phase1_generate_theses(
                cycle_id=cycle_id,
                portfolio=portfolio,
                available_cash=available_cash,
                market_context=market_context
            )

            phase1_complete = datetime.now(timezone.utc)
            logger.info(f"Phase 1 complete: {len(proposals)} proposals generated")

            # Phase 2: Claude synthesis
            synthesis = await self._phase2_synthesize(
                cycle_id=cycle_id,
                proposals=proposals,
                portfolio=portfolio,
                fred_signals=fred_signals,
                polymarket_signals=polymarket_signals,
                technical_analysis=technical_analysis
            )

            phase2_complete = datetime.now(timezone.utc)
            logger.info(f"Phase 2 complete: decision_type={synthesis.get('decision_type')}, "
                       f"conviction={synthesis.get('recommendation', {}).get('conviction_score', 0)}")

            # Check if we should proceed to review
            if not synthesis.get("proceed_to_review", False):
                # No trade opportunity above threshold
                self.db.update_scan_cycle(
                    cycle_id=cycle_id,
                    status="completed",
                    decision_type=synthesis.get("decision_type", "NO_TRADE"),
                    final_conviction=synthesis.get("recommendation", {}).get("conviction_score", 0)
                )

                if self.telegram:
                    await self._notify_no_trade(synthesis)

                return {
                    "cycle_id": cycle_id,
                    "status": "completed",
                    "decision_type": synthesis.get("decision_type", "NO_TRADE"),
                    "synthesis": synthesis,
                    "proposals": proposals,
                    "reviews": [],
                    "final_decision": None,
                    "proceed_to_execution": False,
                    "timing": {
                        "total_ms": int((phase2_complete - start_time).total_seconds() * 1000),
                        "phase1_ms": int((phase1_complete - start_time).total_seconds() * 1000),
                        "phase2_ms": int((phase2_complete - phase1_complete).total_seconds() * 1000)
                    }
                }

            # Phase 3: Peer review
            reviews = await self._phase3_peer_review(
                cycle_id=cycle_id,
                synthesis=synthesis
            )

            phase3_complete = datetime.now(timezone.utc)
            approve_count = sum(1 for r in reviews if r.get("verdict") == "APPROVE")
            logger.info(f"Phase 3 complete: {approve_count}/3 approvals")

            # Phase 4: Final decision
            final_decision = await self._phase4_final_decision(
                cycle_id=cycle_id,
                synthesis=synthesis,
                reviews=reviews,
                proposals=proposals,
                technical_analysis=technical_analysis,
                portfolio=portfolio
            )

            phase4_complete = datetime.now(timezone.utc)

            # Update scan cycle record
            self.db.update_scan_cycle(
                cycle_id=cycle_id,
                status="completed",
                decision_type=final_decision.get("decision_type", "NO_TRADE"),
                final_conviction=final_decision.get("final_conviction", 0)
            )

            logger.info(f"MACA cycle {cycle_id} complete: {final_decision.get('decision_type')} "
                       f"(conviction: {final_decision.get('final_conviction', 0)})")

            return {
                "cycle_id": cycle_id,
                "status": "completed",
                "decision_type": final_decision.get("decision_type"),
                "synthesis": synthesis,
                "proposals": proposals,
                "reviews": reviews,
                "final_decision": final_decision,
                "proceed_to_execution": final_decision.get("proceed_to_execution", False),
                "timing": {
                    "total_ms": int((phase4_complete - start_time).total_seconds() * 1000),
                    "phase1_ms": int((phase1_complete - start_time).total_seconds() * 1000),
                    "phase2_ms": int((phase2_complete - phase1_complete).total_seconds() * 1000),
                    "phase3_ms": int((phase3_complete - phase2_complete).total_seconds() * 1000),
                    "phase4_ms": int((phase4_complete - phase3_complete).total_seconds() * 1000)
                }
            }

        except Exception as e:
            logger.error(f"MACA cycle {cycle_id} failed: {e}")

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
                "proceed_to_execution": False
            }

    async def _phase1_generate_theses(
        self,
        cycle_id: str,
        portfolio: Dict[str, Any],
        available_cash: float,
        market_context: Optional[str] = None
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
        portfolio_summary = {
            "positions": portfolio.get("positions", []),
            "total_value": portfolio.get("total_value", 0),
            "cash": available_cash
        }

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
                market_context=market_context
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
        ))

        # ChatGPT thesis
        tasks.append(self._safe_generate_thesis(
            "chatgpt",
            self.chatgpt.generate_thesis,
            portfolio_summary=portfolio_summary,
            available_cash=available_cash,
            scan_cycle_id=cycle_id,
            market_context=market_context
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

        The Grok scanner may have a different interface, so this adapts it
        to the standard ThesisProposal schema.
        """
        try:
            # Grok scanner typically uses scan_sentiment or similar
            if hasattr(self.grok, 'scan_sentiment'):
                sentiment_result = await asyncio.wait_for(
                    self.grok.scan_sentiment(
                        context=market_context or "investment opportunities"
                    ),
                    timeout=30.0
                )

                # Convert sentiment result to thesis format
                return {
                    "schema_version": "1.0.0",
                    "proposal_id": str(uuid.uuid4()),
                    "ai_source": "grok",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scan_cycle_id": cycle_id,
                    "proposal_type": sentiment_result.get("proposal_type", "NO_OPPORTUNITY"),
                    "recommendation": sentiment_result.get("recommendation", {}),
                    "rotation_details": sentiment_result.get("rotation_details", {}),
                    "supporting_evidence": sentiment_result.get("supporting_evidence", {}),
                    "raw_data": sentiment_result.get("raw_data", {}),
                    "time_sensitive": sentiment_result.get("time_sensitive", False),
                    "metadata": {
                        "model": "grok-3-fast-beta",
                        "adapter": "grok_thesis_adapter"
                    }
                }
            else:
                return self._empty_proposal(cycle_id, "grok", "No compatible method found")

        except Exception as e:
            logger.error(f"Grok adapter error: {e}")
            return self._empty_proposal(cycle_id, "grok", str(e))

    async def _phase2_synthesize(
        self,
        cycle_id: str,
        proposals: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        fred_signals: List[Dict[str, Any]],
        polymarket_signals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 2: Claude synthesizes all proposals into a recommendation.

        Args:
            cycle_id: Current scan cycle ID
            proposals: List of thesis proposals from Phase 1
            portfolio: Current portfolio state
            fred_signals: Macro indicators from FRED
            polymarket_signals: Prediction market data
            technical_analysis: Technical chart analysis

        Returns:
            SynthesisDecision from Claude
        """
        logger.info(f"Phase 2: Claude synthesis for cycle {cycle_id}")

        context = {
            "proposals": proposals,
            "portfolio": portfolio,
            "fred_signals": fred_signals,
            "polymarket_signals": polymarket_signals,
            "technical_analysis": technical_analysis
        }

        try:
            synthesis = await asyncio.wait_for(
                self.claude.synthesize_proposals(context, cycle_id),
                timeout=45.0
            )
            return synthesis
        except asyncio.TimeoutError:
            logger.error("Claude synthesis timed out")
            return self._empty_synthesis(cycle_id, "Synthesis timeout")
        except Exception as e:
            logger.error(f"Claude synthesis error: {e}")
            return self._empty_synthesis(cycle_id, str(e))

    async def _phase3_peer_review(
        self,
        cycle_id: str,
        synthesis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: All AIs review Claude's synthesis.

        Args:
            cycle_id: Current scan cycle ID
            synthesis: Claude's synthesis decision

        Returns:
            List of peer reviews from all AIs
        """
        logger.info(f"Phase 3: Peer review for cycle {cycle_id}")

        tasks = [
            self._safe_review(
                "perplexity",
                self.perplexity.review_proposal,
                synthesis=synthesis,
                scan_cycle_id=cycle_id
            ),
            self._safe_review(
                "chatgpt",
                self.chatgpt.review_proposal,
                synthesis=synthesis,
                scan_cycle_id=cycle_id
            )
        ]

        # Grok review if available
        if hasattr(self.grok, 'review_proposal'):
            tasks.append(self._safe_review(
                "grok",
                self.grok.review_proposal,
                synthesis=synthesis,
                scan_cycle_id=cycle_id
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Review {i} failed: {result}")
                reviews.append(self._empty_review(
                    cycle_id,
                    synthesis.get("synthesis_id"),
                    ["perplexity", "chatgpt", "grok"][i],
                    str(result)
                ))
            else:
                reviews.append(result)
                # Save to database
                self.db.save_ai_review(result)

        return reviews

    async def _safe_review(
        self,
        source: str,
        review_func,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Safely call review with timeout.

        Args:
            source: AI source name
            review_func: The review_proposal function to call
            **kwargs: Arguments to pass to review_func

        Returns:
            Review result or error placeholder
        """
        try:
            return await asyncio.wait_for(
                review_func(**kwargs),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"{source} review timed out")
            return self._empty_review(
                kwargs.get("scan_cycle_id", ""),
                kwargs.get("synthesis", {}).get("synthesis_id"),
                source,
                "Timeout"
            )
        except Exception as e:
            logger.error(f"{source} review error: {e}")
            return self._empty_review(
                kwargs.get("scan_cycle_id", ""),
                kwargs.get("synthesis", {}).get("synthesis_id"),
                source,
                str(e)
            )

    async def _phase4_final_decision(
        self,
        cycle_id: str,
        synthesis: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        proposals: Optional[List[Dict[str, Any]]] = None,
        technical_analysis: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Phase 4: Claude makes final decision based on reviews.

        Args:
            cycle_id: Current scan cycle ID
            synthesis: Claude's synthesis from Phase 2
            reviews: Peer reviews from Phase 3
            proposals: Original thesis proposals from Phase 1 (for notification)
            technical_analysis: Technical chart analysis (for notification)
            portfolio: Current portfolio state (for notification)

        Returns:
            Final decision with adjusted conviction
        """
        logger.info(f"Phase 4: Final decision for cycle {cycle_id}")

        # Count approvals and calculate confidence adjustment
        approve_count = sum(1 for r in reviews if r.get("verdict") == "APPROVE")
        total_adjustment = sum(r.get("confidence_adjustment", 0) for r in reviews)

        original_conviction = synthesis.get("recommendation", {}).get("conviction_score", 0)
        adjusted_conviction = max(0, min(100, original_conviction + total_adjustment))

        # Decision matrix based on approval count
        if approve_count == 3:
            proceed = adjusted_conviction >= 80
            decision_note = "Full consensus - proceeding with confidence"
        elif approve_count == 2:
            proceed = adjusted_conviction >= 80
            decision_note = "Majority approval - proceeding with noted minority concern"
        elif approve_count == 1:
            proceed = adjusted_conviction >= 85  # Higher bar with minority approval
            decision_note = "Single approval - reduced confidence, likely no trade"
        else:
            proceed = adjusted_conviction >= 90  # Very high bar to override
            decision_note = "No approvals - requires override or no trade"

        # Collect concerns from reviews
        all_concerns = []
        for review in reviews:
            concerns = review.get("review_details", {}).get("concerns", [])
            all_concerns.extend(concerns)

        final_decision = {
            "cycle_id": cycle_id,
            "decision_type": "TRADE" if proceed else "NO_TRADE",
            "original_conviction": original_conviction,
            "final_conviction": adjusted_conviction,
            "confidence_adjustment": total_adjustment,
            "approval_count": approve_count,
            "total_reviews": len(reviews),
            "decision_note": decision_note,
            "concerns_raised": all_concerns,
            "proceed_to_execution": proceed,
            "recommendation": synthesis.get("recommendation", {}) if proceed else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

        # Notify via Telegram if configured
        if self.telegram:
            await self._notify_final_decision(
                final_decision=final_decision,
                synthesis=synthesis,
                reviews=reviews,
                proposals=proposals or [],
                technical_analysis=technical_analysis,
                portfolio=portfolio or {}
            )

        return final_decision

    async def _notify_no_trade(self, synthesis: Dict[str, Any]) -> None:
        """Send Telegram notification for no-trade decision."""
        try:
            rationale = synthesis.get("rationale", "No compelling opportunities above 80 conviction threshold")
            conviction = synthesis.get("recommendation", {}).get("conviction_score", 0)

            message = (
                f"MACA Cycle Complete\n\n"
                f"Decision: NO TRADE\n"
                f"Highest Conviction: {conviction}/100\n"
                f"Rationale: {rationale}"
            )

            await self.telegram.send_message(
                message,
                message_type="maca_no_trade"
            )
        except Exception as e:
            logger.error(f"Failed to send no-trade notification: {e}")

    async def _notify_final_decision(
        self,
        final_decision: Dict[str, Any],
        synthesis: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        proposals: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send Telegram notification for final decision using MACA display.

        Uses the new send_maca_scan_summary() method for rich display with:
        - AI Council views (all proposals)
        - Chart analysis with technical signals
        - Claude's synthesis decision
        - Inline approve/reject buttons
        """
        try:
            # Build technical signals list for display
            technical_signals = []
            if technical_analysis:
                # technical_analysis might be a single dict or have nested structure
                if isinstance(technical_analysis, dict):
                    # If it has a ticker, it's a single signal
                    if "ticker" in technical_analysis:
                        technical_signals.append(technical_analysis)
                    # If it has tickers as keys, extract them
                    else:
                        for key, value in technical_analysis.items():
                            if isinstance(value, dict) and "ticker" in value:
                                technical_signals.append(value)

            # Get trade_id if this is actionable
            trade_id = None
            if final_decision.get("proceed_to_execution"):
                # Trade ID would be generated by the trade creation process
                # For now, use cycle_id as placeholder - actual trade_id comes from trade creation
                trade_id = final_decision.get("cycle_id", "")[:8]

            # Use the new MACA scan summary method with inline buttons
            await self.telegram.send_maca_scan_summary(
                proposals=proposals,
                synthesis=synthesis,
                technical_signals=technical_signals,
                portfolio=portfolio or {},
                trade_id=trade_id if final_decision.get("proceed_to_execution") else None
            )

        except Exception as e:
            logger.error(f"Failed to send MACA decision notification: {e}")
            # Fallback to simple message
            try:
                rec = synthesis.get("recommendation", {})
                ticker = rec.get("ticker", "N/A")
                side = rec.get("side", "N/A")

                if final_decision.get("proceed_to_execution"):
                    message = (
                        f"MACA Trade Signal\n\n"
                        f"Ticker: {ticker}\n"
                        f"Side: {side}\n"
                        f"Conviction: {final_decision.get('final_conviction')}/100\n"
                        f"Approvals: {final_decision.get('approval_count')}/{final_decision.get('total_reviews')}\n"
                        f"Note: {final_decision.get('decision_note')}\n\n"
                        f"Thesis: {rec.get('thesis', 'N/A')}"
                    )
                else:
                    message = (
                        f"MACA Cycle Complete\n\n"
                        f"Decision: NO TRADE\n"
                        f"Considered: {ticker} ({side})\n"
                        f"Final Conviction: {final_decision.get('final_conviction')}/100\n"
                        f"Approvals: {final_decision.get('approval_count')}/{final_decision.get('total_reviews')}\n"
                        f"Note: {final_decision.get('decision_note')}"
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

    def _empty_review(
        self,
        cycle_id: str,
        proposal_id: str,
        source: str,
        reason: str
    ) -> Dict[str, Any]:
        """Return empty review when review fails."""
        return {
            "schema_version": "1.0.0",
            "review_id": str(uuid.uuid4()),
            "proposal_id": proposal_id,
            "scan_cycle_id": cycle_id,
            "reviewer_ai": source,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "REJECT",
            "confidence_adjustment": -5,
            "review_details": {
                "agrees_with_thesis": False,
                "concerns": [f"Review failed: {reason}"],
                "additional_risks": [],
                "missing_information": [],
                "alternative_view": None
            },
            "validation_checks": {
                "facts_verified": False,
                "timing_appropriate": False,
                "risk_reward_acceptable": False
            },
            "raw_response": f"Error: {reason}"
        }
