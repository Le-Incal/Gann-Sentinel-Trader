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

        prompt = f"""You are a Real-Time Web Research Analyst.

Your unique strength: scan the current public web for verifiable facts (news, filings, earnings, data releases) and cite sources.

You do NOT infer sentiment.
You do NOT analyze charts.
You do NOT invent signals.

CURRENT CONTEXT:
- Date: {current_date}
- Portfolio Positions:
{positions_text}
- Available Cash: ${available_cash:,.2f}

{additional_context or ""}

YOUR TASK:
Propose a single trade OR recommend HOLD based on verifiable, time-relevant fundamental catalysts.

YOU MUST:
1) List every external signal you considered and provide counts.
2) Rank the top 3 signals by importance.
3) Cite sources (URLs) for each key claim.
4) State conflicting signals and why they matter.
5) Provide a clear invalidation condition.
6) If evidence is weak/conflicting → proposal_type = NO_OPPORTUNITY.

OUTPUT:
Return ONLY valid JSON (no markdown) in this exact structure:
{{
  "proposal_type": "NEW_BUY" | "SELL" | "ROTATE" | "NO_OPPORTUNITY",
  "analyst_role": "web_facts",
  "signal_inventory": {{
    "total_signals": 0,
    "by_source": {{"news": 0, "filings": 0, "earnings": 0, "macro": 0, "other": 0}}
  }},
  "signals_considered": [
    {{"source": "news|filings|earnings|macro|other", "summary": "what it implies", "url": "https://...", "weight": 0-1, "confidence": 0-1}}
  ],
  "recommendation": {{
    "ticker": "SYMBOL or null",
    "side": "BUY" | "SELL" | null,
    "conviction_score": 0-100,
    "thesis": "1-3 sentence thesis",
    "thesis_description": "100-200 words explaining why this trade exists NOW from a facts/catalyst view",
    "time_horizon": "days" | "weeks" | "months",
    "catalyst": "specific factual catalyst",
    "catalyst_deadline": "YYYY-MM-DD or null",
    "invalidation": "what would prove this wrong"
  }},
  "supporting_evidence": {{
    "key_signals": [
      {{"signal_type": "fundamental"|"event"|"macro", "summary": "brief", "source": "URL or source name", "confidence": "high"|"medium"|"low"}}
    ],
    "bull_case": "bull case + probability",
    "bear_case": "bear case + probability",
    "risks": ["risk 1", "risk 2"]
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
        """Participate in committee debate (role-constrained)."""

        if not self.is_configured:
            return {
                "speaker": "perplexity",
                "round": round_num,
                "message": "Perplexity not configured",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
            }

        system = """You are participating in an investment committee cross-examination.

You are Perplexity in the role of External Reality & Facts Analyst.

RULES:
1) Stay within your role: verifiable facts, catalysts, what is knowable now.
2) Do NOT infer sentiment. Do NOT analyze charts.
3) Do NOT browse for new info in this debate round; react to provided theses only.
4) Speak in DELTAS (what changes because of others' theses). Do NOT restate your full memo.
5) If you vote BUY/SELL, you MUST use the committee candidate ticker (provided in the context). Otherwise vote HOLD.

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
            # Perplexity uses an OpenAI-compatible chat endpoint.
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 900,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            }

            async with httpx.AsyncClient(timeout=45.0) as client:
                # Must hit the OpenAI-compatible chat/completions endpoint.
                resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                if resp.status_code != 200:
                    return {
                        "speaker": "perplexity",
                        "round": round_num,
                        "message": f"Debate API error {resp.status_code}",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                        "status": "error",
                    }

                data = resp.json()
                # Perplexity may return different shapes; accept either.
                content = None
                if isinstance(data, dict):
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content")
                    elif "output" in data:
                        content = data.get("output")
                if not content:
                    content = json.dumps({"message": "Empty debate response", "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0}}, ensure_ascii=False)

                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    return {
                        "speaker": "perplexity",
                        "round": round_num,
                        "message": "Debate parse error",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                        "status": "error",
                    }

                if not isinstance(parsed, dict):
                    parsed = {"message": "Debate non-dict response", "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0}, "changed_mind": False, "status": "error"}

                parsed.update({"speaker": "perplexity", "round": round_num})
                return parsed

        except Exception as e:
            return {
                "speaker": "perplexity",
                "round": round_num,
                "message": f"Debate exception: {e}",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
                "status": "error",
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
