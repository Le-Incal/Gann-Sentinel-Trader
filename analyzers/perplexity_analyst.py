"""
Perplexity Analyst for Gann Sentinel Trader
Uses Perplexity Sonar Pro API for fundamental research and citation-backed analysis.

Version: 1.0.0 - Initial MACA Integration
"""

import os
import json
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


class PerplexityAnalyst:
    """
    Perplexity-powered analyst for fundamental research.

    Strengths:
    - Citation-backed research
    - Real-time web access
    - Financial data integration
    - Deep research capabilities
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar-pro"
    ):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model = model
        self.base_url = "https://api.perplexity.ai"

        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set - Perplexity analyst disabled")

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
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a thesis proposal using Perplexity Sonar Pro.

        Args:
            portfolio_summary: Current portfolio positions and P&L
            available_cash: Cash available for trading
            scan_cycle_id: ID of current scan cycle
            additional_context: Any additional market context

        Returns:
            ThesisProposal schema-compliant dict
        """
        if not self.is_configured:
            return self._empty_proposal(scan_cycle_id, "Perplexity not configured")

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Format portfolio for prompt
        positions_text = self._format_portfolio(portfolio_summary)

        prompt = f"""You are a Fundamental Research Analyst with access to comprehensive financial data.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio Positions:
{positions_text}
- Available Cash: ${available_cash:,.2f}

YOUR TASK:
Research and identify the single highest-conviction investment opportunity based on fundamental analysis.

CONSIDER:
1. Recent earnings, guidance, or financial developments
2. Valuation relative to peers and historical norms
3. Industry trends and competitive positioning
4. Institutional activity and analyst sentiment
5. Whether rotating out of current positions could improve returns

{additional_context or ""}

OUTPUT:
Return ONLY a valid JSON object (no markdown, no explanation) with this structure:
{{
  "proposal_type": "NEW_BUY" | "SELL" | "ROTATE" | "NO_OPPORTUNITY",
  "recommendation": {{
    "ticker": "SYMBOL",
    "side": "BUY" | "SELL",
    "conviction_score": 0-100,
    "thesis": "1-3 sentence investment thesis",
    "time_horizon": "days" | "weeks" | "months",
    "catalyst": "what drives this opportunity",
    "catalyst_deadline": "YYYY-MM-DD or null"
  }},
  "rotation_details": {{
    "sell_ticker": "SYMBOL or null",
    "sell_rationale": "why selling",
    "expected_net_gain": "estimated improvement"
  }},
  "supporting_evidence": {{
    "key_signals": [
      {{
        "signal_type": "fundamental",
        "summary": "brief description",
        "source": "URL or source name",
        "confidence": "high" | "medium" | "low"
      }}
    ],
    "bull_case": "key bullish factors",
    "bear_case": "key bearish factors",
    "risks": ["risk 1", "risk 2"]
  }},
  "time_sensitive": true | false
}}

IMPORTANT:
- Be honest about conviction (don't inflate scores)
- CITE YOUR SOURCES with URLs where possible
- If no compelling opportunity exists, return proposal_type: "NO_OPPORTUNITY"
- For ROTATE proposals, the new opportunity must be significantly better"""

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
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                )

                latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

                if response.status_code != 200:
                    logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                    return self._empty_proposal(scan_cycle_id, f"API error: {response.status_code}")

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Debug: Log raw response before parsing
                logger.info(f"DEBUG Perplexity raw response (first 500 chars): {content[:500]}")

                # Parse JSON from response
                parsed = self._parse_json_response(content)

                if not parsed:
                    logger.error(f"Failed to parse Perplexity response. Full content: {content}")
                    return self._empty_proposal(scan_cycle_id, "Failed to parse response")

                # Build full proposal
                proposal = self._build_proposal(
                    parsed=parsed,
                    scan_cycle_id=scan_cycle_id,
                    latency_ms=latency_ms,
                    raw_response=content
                )

                logger.info(f"Perplexity thesis: {parsed.get('recommendation', {}).get('ticker', 'NO_OPPORTUNITY')} "
                           f"conviction={parsed.get('recommendation', {}).get('conviction_score', 0)}")

                return proposal

        except Exception as e:
            logger.error(f"Error generating Perplexity thesis: {e}")
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

        prompt = f"""You are reviewing a proposed trade recommendation. Your role is to validate or challenge it.

PROPOSED TRADE:
- Ticker: {recommendation.get('ticker')}
- Side: {recommendation.get('side')}
- Conviction: {recommendation.get('conviction_score')}/100
- Time Horizon: {recommendation.get('time_horizon')}

THESIS:
{recommendation.get('thesis')}

CROSS-VALIDATION:
{json.dumps(synthesis.get('cross_validation', {}), indent=2)}

YOUR TASK:
1. Verify the factual claims made using current data
2. Assess whether the thesis is fundamentally sound
3. Identify risks that may have been overlooked
4. Check if the timing and valuation are appropriate

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
  }}
}}

Be specific about any concerns. Cite sources if you find conflicting information."""

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
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Perplexity review API error: {response.status_code}")
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
            logger.error(f"Error in Perplexity review: {e}")
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
            lines.append(f"  - {ticker}: {qty} shares, P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

        return "\n".join(lines)

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from response, handling markdown code blocks and trailing text."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Handle ```json or just ```
            start_idx = 1
            end_idx = -1 if lines[-1].strip() in ["```", ""] else len(lines)
            content = "\n".join(lines[start_idx:end_idx]).strip()
            # Remove trailing ``` if still present
            if content.endswith("```"):
                content = content[:-3].strip()

        # First try: direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Second try: find JSON object boundaries with proper brace matching
        start = content.find("{")
        if start == -1:
            return None

        # Find matching closing brace (handle nested objects)
        depth = 0
        end = -1
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                logger.error(f"JSON parse failed. Extracted: {content[start:end][:200]}")

        return None

    def _build_proposal(
        self,
        parsed: Dict,
        scan_cycle_id: str,
        latency_ms: int,
        raw_response: str
    ) -> Dict[str, Any]:
        """Build full proposal from parsed response."""
        proposal_id = self._generate_proposal_id()

        return {
            "schema_version": "1.0.0",
            "proposal_id": proposal_id,
            "ai_source": "perplexity",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scan_cycle_id": scan_cycle_id,
            "proposal_type": parsed.get("proposal_type", "NO_OPPORTUNITY"),
            "recommendation": parsed.get("recommendation", {}),
            "rotation_details": parsed.get("rotation_details", {}),
            "supporting_evidence": parsed.get("supporting_evidence", {}),
            "raw_data": {
                "search_queries_used": [],
                "sources_consulted": self._extract_sources(parsed),
                "raw_response": raw_response[:2000]  # Truncate for storage
            },
            "time_sensitive": parsed.get("time_sensitive", False),
            "metadata": {
                "model": self.model,
                "latency_ms": latency_ms
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
            "reviewer_ai": "perplexity",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": parsed.get("verdict", "REJECT"),
            "confidence_adjustment": parsed.get("confidence_adjustment", 0),
            "review_details": parsed.get("review_details", {}),
            "validation_checks": parsed.get("validation_checks", {}),
            "raw_response": raw_response
        }

    def _extract_sources(self, parsed: Dict) -> List[str]:
        """Extract source URLs/names from parsed response."""
        sources = []
        evidence = parsed.get("supporting_evidence", {})
        for signal in evidence.get("key_signals", []):
            source = signal.get("source", "")
            if source:
                sources.append(source)
        return sources

    def _empty_proposal(self, scan_cycle_id: str, reason: str) -> Dict[str, Any]:
        """Return empty proposal when generation fails."""
        return {
            "schema_version": "1.0.0",
            "proposal_id": self._generate_proposal_id(),
            "ai_source": "perplexity",
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
            "reviewer_ai": "perplexity",
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
