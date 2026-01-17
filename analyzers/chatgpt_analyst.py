"""
ChatGPT Analyst for Gann Sentinel Trader
Uses OpenAI GPT-4o API for pattern recognition and risk analysis.

Version: 1.0.0 - Initial MACA Integration
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


class ChatGPTAnalyst:
    """
    ChatGPT-powered analyst for sentiment + cognitive bias analysis.

    Intended role in the multi-agent system:
    - Abstract market psychology from noisy context into structured signals
    - Detect crowding, herding, overconfidence, and narrative exhaustion
    - Stress-test other theses for bias contamination

    This analyst should NOT browse the web or read charts directly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - ChatGPT analyst disabled")

    @property
    def is_configured(self) -> bool:
        """Check if analyst is properly configured."""
        return bool(self.api_key)

    def _generate_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        return str(uuid.uuid4())

    async def generate_thesis(
        self,
        portfolio_summary: Dict[str, Any],
        available_cash: float,
        scan_cycle_id: str,
        market_context: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a thesis proposal using GPT-4o.

        Args:
            portfolio_summary: Current portfolio positions and P&L
            available_cash: Cash available for trading
            scan_cycle_id: ID of current scan cycle
            market_context: Recent market conditions summary
            additional_context: Any additional context

        Returns:
            ThesisProposal schema-compliant dict
        """
        if not self.is_configured:
            return self._empty_proposal(scan_cycle_id, "ChatGPT not configured")

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Format portfolio for prompt
        positions_text = self._format_portfolio(portfolio_summary)

        prompt = f"""You are a Market Sentiment + Cognitive Bias Analyst.

You do NOT browse the web.
You do NOT analyze charts.
You do NOT invent signals.

Your unique strength: translate messy narrative + mixed signals into a disciplined sentiment view and identify bias contamination.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio Positions:
{positions_text}
- Available Cash: ${available_cash:,.2f}
{f"- Market Context: {market_context}" if market_context else ""}

{additional_context or ""}

YOUR TASK:
Propose a single trade OR recommend HOLD based on sentiment regime + cognitive bias.

YOU MUST:
1) List every signal you considered (grouped by source) and provide counts.
2) Rank the top 3 signals by importance.
3) State conflicting signals and why they matter.
4) Provide a clear invalidation condition.
5) If evidence is weak/conflicting → proposal_type = NO_OPPORTUNITY.

OUTPUT:
Return ONLY valid JSON (no markdown) in this exact structure:
{{
  "proposal_type": "NEW_BUY" | "SELL" | "ROTATE" | "NO_OPPORTUNITY",
  "analyst_role": "sentiment_bias",
  "signal_inventory": {{
    "total_signals": 0,
    "by_source": {{"fred": 0, "polymarket": 0, "market_context": 0, "other": 0}}
  }},
  "signals_considered": [
    {{"source": "fred|polymarket|market_context|other", "summary": "what it implies", "weight": 0-1, "confidence": 0-1}}
  ],
  "recommendation": {{
    "ticker": "SYMBOL or null",
    "side": "BUY" | "SELL" | null,
    "conviction_score": 0-100,
    "thesis": "1-3 sentence thesis",
    "thesis_description": "100-200 words explaining why this trade exists NOW from a sentiment/bias view",
    "time_horizon": "days" | "weeks" | "months",
    "catalyst": "sentiment catalyst / narrative trigger",
    "catalyst_deadline": "YYYY-MM-DD or null",
    "invalidation": "what would prove this wrong"
  }},
  "supporting_evidence": {{
    "key_signals": [
      {{"signal_type": "sentiment"|"positioning"|"narrative"|"macro", "summary": "brief", "source": "fred|polymarket|context", "confidence": "high"|"medium"|"low"}}
    ],
    "bull_case": "bull case + probability",
    "bear_case": "bear case + probability",
    "risks": ["risk 1", "risk 2"],
    "bias_flags": ["herding", "overconfidence", "recency", "confirmation"],
    "what_you_might_be_missing": "one contrarian consideration"
  }},
  "time_sensitive": true | false
}}"""

        try:
            start_time = datetime.now(timezone.utc)

            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a market sentiment and cognitive bias analyst. Respond with JSON only."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                )

                latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    return self._empty_proposal(scan_cycle_id, f"API error: {response.status_code}")

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON from response
                parsed = self._parse_json_response(content)

                if not parsed:
                    logger.error(f"Failed to parse ChatGPT response: {content[:500]}")
                    return self._empty_proposal(scan_cycle_id, "Failed to parse response")

                # Build full proposal
                proposal = self._build_proposal(
                    parsed=parsed,
                    scan_cycle_id=scan_cycle_id,
                    latency_ms=latency_ms,
                    raw_response=content,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0)
                )

                logger.info(f"ChatGPT thesis: {parsed.get('recommendation', {}).get('ticker', 'NO_OPPORTUNITY')} "
                           f"conviction={parsed.get('recommendation', {}).get('conviction_score', 0)}")

                return proposal

        except Exception as e:
            logger.error(f"Error generating ChatGPT thesis: {e}")
            return self._empty_proposal(scan_cycle_id, str(e))

    async def review_proposal(
        self,
        synthesis: Dict[str, Any],
        scan_cycle_id: str
    ) -> Dict[str, Any]:
        """
        Review Claude's synthesized proposal.

        Args:
            synthesis: Claude's synthesis decision
            scan_cycle_id: ID of current scan cycle

        Returns:
            PeerReview schema-compliant dict
        """
        if not self.is_configured:
            return self._empty_review(scan_cycle_id, synthesis.get("synthesis_id"), "Not configured")

        recommendation = synthesis.get("recommendation", {})

        prompt = f"""You are reviewing a proposed trade recommendation from a risk/reward perspective.

PROPOSED TRADE:
- Ticker: {recommendation.get('ticker')}
- Side: {recommendation.get('side')}
- Conviction: {recommendation.get('conviction_score')}/100
- Position Size: {recommendation.get('position_size_pct', 'N/A')}%
- Stop Loss: {recommendation.get('stop_loss_pct', 'N/A')}%
- Time Horizon: {recommendation.get('time_horizon')}

THESIS:
{recommendation.get('thesis')}

CROSS-VALIDATION DATA:
{json.dumps(synthesis.get('cross_validation', {}), indent=2)}

YOUR TASK:
1. Evaluate the risk/reward profile of this trade
2. Check if position sizing is appropriate for the conviction level
3. Identify any pattern or scenario risks not mentioned
4. Assess if the stop loss is at a logical level

OUTPUT:
Return ONLY a valid JSON object (no markdown):
{{
  "verdict": "APPROVE" | "REJECT",
  "confidence_adjustment": -10 to +10,
  "review_details": {{
    "agrees_with_thesis": true | false,
    "concerns": ["concern 1", "concern 2"],
    "additional_risks": ["risk not mentioned"],
    "missing_information": ["what should be considered"],
    "alternative_view": "different interpretation if any"
  }},
  "validation_checks": {{
    "facts_verified": true | false,
    "timing_appropriate": true | false,
    "risk_reward_acceptable": true | false
  }},
  "risk_assessment": {{
    "position_size_appropriate": true | false,
    "stop_loss_logical": true | false,
    "expected_value_positive": true | false,
    "correlation_risk": "low" | "medium" | "high"
  }}
}}

Be specific about risk concerns. Quantify where possible."""

        try:
            start_time = datetime.now(timezone.utc)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a risk analyst. Always respond with valid JSON only."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                )

                if response.status_code != 200:
                    logger.error(f"OpenAI review API error: {response.status_code}")
                    return self._empty_review(scan_cycle_id, synthesis.get("synthesis_id"), "API error")

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                parsed = self._parse_json_response(content)

                if not parsed:
                    return self._empty_review(scan_cycle_id, synthesis.get("synthesis_id"), "Parse error")

                return self._build_review(
                    parsed=parsed,
                    scan_cycle_id=scan_cycle_id,
                    proposal_id=synthesis.get("synthesis_id"),
                    raw_response=content
                )

        except Exception as e:
            logger.error(f"Error in ChatGPT review: {e}")
            return self._empty_review(scan_cycle_id, synthesis.get("synthesis_id"), str(e))

    def _format_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """Format portfolio for prompt."""
        positions = portfolio.get("positions", [])
        if not positions:
            return "  No current positions"

        lines = []
        for pos in positions:
            ticker = pos.get("ticker", "???")
            qty = pos.get("quantity", 0)
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            entry = pos.get("avg_entry_price", 0)
            lines.append(f"  - {ticker}: {qty} shares @ ${entry:.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

        return "\n".join(lines)

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from response, handling markdown code blocks."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def _build_proposal(
        self,
        parsed: Dict,
        scan_cycle_id: str,
        latency_ms: int,
        raw_response: str,
        tokens_used: int = 0
    ) -> Dict[str, Any]:
        """Build full proposal from parsed response."""
        proposal_id = self._generate_proposal_id()

        return {
            "schema_version": "1.0.0",
            "proposal_id": proposal_id,
            "ai_source": "chatgpt",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scan_cycle_id": scan_cycle_id,
            "proposal_type": parsed.get("proposal_type", "NO_OPPORTUNITY"),
            "recommendation": parsed.get("recommendation", {}),
            "rotation_details": parsed.get("rotation_details", {}),
            "supporting_evidence": parsed.get("supporting_evidence", {}),
            "raw_data": {
                "risk_analysis": parsed.get("risk_analysis", {}),
                "raw_response": raw_response[:2000]
            },
            "time_sensitive": parsed.get("time_sensitive", False),
            "metadata": {
                "model": self.model,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used
            }
        }

    def _build_review(
        self,
        parsed: Dict,
        scan_cycle_id: str,
        proposal_id: str,
        raw_response: str
    ) -> Dict[str, Any]:
        """Build full review from parsed response."""
        return {
            "schema_version": "1.0.0",
            "review_id": str(uuid.uuid4()),
            "proposal_id": proposal_id,
            "scan_cycle_id": scan_cycle_id,
            "reviewer_ai": "chatgpt",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": parsed.get("verdict", "REJECT"),
            "confidence_adjustment": parsed.get("confidence_adjustment", 0),
            "review_details": parsed.get("review_details", {}),
            "validation_checks": parsed.get("validation_checks", {}),
            "risk_assessment": parsed.get("risk_assessment", {}),
            "raw_response": raw_response
        }

    # ------------------------------------------------------------------
    # Debate Layer
    # ------------------------------------------------------------------
    async def debate(
        self,
        *,
        scan_cycle_id: str,
        round_num: int,
        own_thesis: Dict[str, Any],
        other_theses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Participate in committee debate (role-constrained).

        Returns a structured response that can be logged and displayed.
        """

        if not self.is_configured:
            return {
                "speaker": "chatgpt",
                "round": round_num,
                "message": "ChatGPT not configured",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
            }

        system = """You are participating in an investment committee cross-examination.

You are ChatGPT in the role of Sentiment + Cognitive Bias Analyst.

RULES:
1) Stay within your role: sentiment regime, positioning psychology, bias contamination.
2) Do NOT browse the web. Do NOT analyze charts.
3) Do NOT invent new signals. Only react to provided theses/context.
4) Speak in DELTAS (what changes because of others' theses). Do NOT restate your full memo.
5) You may vote for ANY proposal listed in debate_context.proposal_options.

Output ONLY JSON in this schema:
{
  "claim": "1 sentence: your current position",
  "top_signals": ["exactly 2 short bullets"],
  "counterpoint": "1 sentence: strongest objection you acknowledge",
  "change_my_mind": "1 explicit condition",
  "changed_mind": true|false,
  "vote": {"action": "BUY"|"SELL"|"HOLD", "ticker": "..."|null, "side": "BUY"|"SELL"|null, "confidence": 0.0-1.0}
}
"""

        user = {
            "round": round_num,
            "own_thesis": own_thesis,
            "other_theses": other_theses,
        }

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 900,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            }

            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                if resp.status_code != 200:
                    return {
                        "speaker": "chatgpt",
                        "round": round_num,
                        "message": f"Debate API error {resp.status_code}",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                        "status": "error",
                    }

                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                parsed.update({"speaker": "chatgpt", "round": round_num})
                return parsed

        except Exception as e:
            return {
                "speaker": "chatgpt",
                "round": round_num,
                "message": f"Debate exception: {e}",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
                "status": "error",
            }

    def _empty_proposal(self, scan_cycle_id: str, reason: str) -> Dict[str, Any]:
        """Return empty proposal when generation fails."""
        return {
            "schema_version": "1.0.0",
            "proposal_id": self._generate_proposal_id(),
            "ai_source": "chatgpt",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scan_cycle_id": scan_cycle_id,
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
            "metadata": {"model": self.model, "error": reason}
        }

    def _empty_review(self, scan_cycle_id: str, proposal_id: str, reason: str) -> Dict[str, Any]:
        """Return empty review when review fails."""
        return {
            "schema_version": "1.0.0",
            "review_id": str(uuid.uuid4()),
            "proposal_id": proposal_id,
            "scan_cycle_id": scan_cycle_id,
            "reviewer_ai": "chatgpt",
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
            "risk_assessment": {
                "position_size_appropriate": False,
                "stop_loss_logical": False,
                "expected_value_positive": False,
                "correlation_risk": "high"
            },
            "raw_response": f"Error: {reason}"
        }
