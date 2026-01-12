"""
MACA Extension for Claude Analyst
Adds multi-proposal synthesis capability for Multi-Agent Consensus Architecture.

Version: 1.0.0
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


class ClaudeMACAMixin:
    """
    Mixin that adds MACA synthesis capabilities to ClaudeAnalyst.

    Usage:
        class EnhancedClaudeAnalyst(ClaudeAnalyst, ClaudeMACAMixin):
            pass

    Or call directly:
        synthesis = await claude_maca_synthesize(proposals, context)
    """

    SYNTHESIS_SYSTEM_PROMPT = """You are the Chief Investment Officer (CIO) evaluating multiple investment proposals from your analyst team.

Your role:
1. Evaluate each proposal on its merits
2. Cross-reference against macro data, prediction markets, and technicals
3. Select the best opportunity OR synthesize a new recommendation
4. Apply rigorous risk management

You have FINAL AUTHORITY on all investment decisions. Your analysts provide research, but you make the call.

PRINCIPLES:
- Historical patterns inform but don't dictate
- Multiple confirming signals increase conviction
- Contrarian views deserve extra scrutiny (they might be right)
- Position sizing reflects conviction and risk
- If no opportunity meets standards, say so clearly"""

    async def synthesize_proposals(
        self,
        context: Dict[str, Any],
        scan_cycle_id: str
    ) -> Dict[str, Any]:
        """
        Synthesize multiple AI proposals into a final recommendation.

        Args:
            context: Dict containing:
                - proposals: List of thesis proposals from Grok, Perplexity, ChatGPT
                - portfolio: Current portfolio state
                - fred_signals: Macro indicators
                - polymarket_signals: Prediction market data
                - technical_analysis: Chart analysis
            scan_cycle_id: ID of current scan cycle

        Returns:
            SynthesisDecision schema-compliant dict
        """
        proposals = context.get("proposals", [])
        portfolio = context.get("portfolio", {})
        fred_signals = context.get("fred_signals", [])
        polymarket_signals = context.get("polymarket_signals", [])
        technical = context.get("technical_analysis")

        # Build the synthesis prompt
        prompt = self._build_synthesis_prompt(
            proposals=proposals,
            portfolio=portfolio,
            fred_signals=fred_signals,
            polymarket_signals=polymarket_signals,
            technical=technical
        )

        try:
            # Call Claude API
            api_key = getattr(self, 'api_key', None) or os.getenv("ANTHROPIC_API_KEY")

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 3000,
                        "system": self.SYNTHESIS_SYSTEM_PROMPT,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Claude synthesis API error: {response.status_code}")
                    return self._empty_synthesis(scan_cycle_id, f"API error: {response.status_code}")

                data = response.json()
                content = data["content"][0]["text"]

                # Parse the JSON response
                parsed = self._parse_synthesis_response(content)

                if not parsed:
                    logger.error(f"Failed to parse synthesis response")
                    return self._empty_synthesis(scan_cycle_id, "Parse error")

                # Build full synthesis
                return self._build_synthesis(parsed, scan_cycle_id, proposals)

        except Exception as e:
            logger.error(f"Claude synthesis error: {e}")
            return self._empty_synthesis(scan_cycle_id, str(e))

    def _build_synthesis_prompt(
        self,
        proposals: List[Dict],
        portfolio: Dict,
        fred_signals: List[Dict],
        polymarket_signals: List[Dict],
        technical: Optional[Dict]
    ) -> str:
        """Build the synthesis prompt."""

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Format proposals
        proposals_text = ""
        for i, p in enumerate(proposals, 1):
            source = p.get("ai_source", "Unknown")
            rec = p.get("recommendation", {})
            evidence = p.get("supporting_evidence", {})

            proposals_text += f"""
PROPOSAL {i}: {source.upper()}
- Type: {p.get('proposal_type', 'N/A')}
- Ticker: {rec.get('ticker', 'None')}
- Side: {rec.get('side', 'N/A')}
- Conviction: {rec.get('conviction_score', 0)}/100
- Thesis: {rec.get('thesis', 'N/A')}
- Catalyst: {rec.get('catalyst', 'N/A')}
- Time Horizon: {rec.get('time_horizon', 'N/A')}
- Bull Case: {evidence.get('bull_case', 'N/A')}
- Bear Case: {evidence.get('bear_case', 'N/A')}
- Time Sensitive: {p.get('time_sensitive', False)}
"""

        # Format portfolio
        positions = portfolio.get("positions", [])
        if positions:
            portfolio_text = "\n".join([
                f"  - {p.get('ticker')}: {p.get('quantity')} shares, P&L: ${p.get('unrealized_pnl', 0):,.2f}"
                for p in positions
            ])
        else:
            portfolio_text = "  No current positions"

        # Format FRED signals
        if fred_signals:
            fred_text = "\n".join([
                f"  - {s.get('indicator', s.get('summary', 'N/A')[:60])}"
                for s in fred_signals[:5]
            ])
        else:
            fred_text = "  No macro signals"

        # Format Polymarket signals
        if polymarket_signals:
            poly_text = "\n".join([
                f"  - {s.get('question', s.get('summary', 'N/A')[:60])}: {s.get('probability', 'N/A')}"
                for s in polymarket_signals[:5]
            ])
        else:
            poly_text = "  No prediction signals"

        # Format technical
        if technical:
            tech_text = f"""
  Market State: {technical.get('market_state', {}).get('state', 'N/A')} ({technical.get('market_state', {}).get('bias', 'N/A')})
  Channel Position: {technical.get('trend_channel', {}).get('position_in_channel', 'N/A')}
  Verdict: {technical.get('verdict', 'N/A')}"""
        else:
            tech_text = "  No technical analysis available"

        prompt = f"""DATE: {current_date}

CURRENT PORTFOLIO:
{portfolio_text}
Available Cash: ${portfolio.get('cash', 0):,.2f}

ANALYST PROPOSALS:
{proposals_text}

CROSS-REFERENCE DATA:

MACRO (FRED):
{fred_text}

PREDICTION MARKETS (Polymarket):
{poly_text}

TECHNICAL ANALYSIS:
{tech_text}

YOUR TASK:
1. Evaluate each proposal's merits and weaknesses
2. Check if cross-reference data supports or contradicts the theses
3. Determine if any opportunity meets our 80+ conviction threshold
4. If multiple proposals target the same thesis, combine insights
5. If proposing a ROTATE, calculate if net expected value is positive

OUTPUT:
Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "decision_type": "TRADE" | "NO_TRADE" | "WATCH",
  "selected_proposal": {{
    "proposal_id": "uuid or null",
    "ai_source": "which AI's proposal was selected",
    "modifications": "what you changed, if any"
  }},
  "recommendation": {{
    "ticker": "SYMBOL",
    "side": "BUY" | "SELL",
    "conviction_score": 0-100,
    "thesis": "your refined thesis",
    "position_size_pct": 5-25,
    "stop_loss_pct": 5-15,
    "time_horizon": "days" | "weeks" | "months"
  }},
  "cross_validation": {{
    "fred_alignment": "supports" | "neutral" | "conflicts",
    "polymarket_alignment": "supports" | "neutral" | "conflicts",
    "technical_alignment": "supports" | "neutral" | "conflicts",
    "notes": "specific observations"
  }},
  "proposal_evaluation": [
    {{
      "ai_source": "grok",
      "evaluation": "selected" | "rejected" | "partially_used",
      "reason": "why"
    }},
    {{
      "ai_source": "perplexity",
      "evaluation": "selected" | "rejected" | "partially_used",
      "reason": "why"
    }},
    {{
      "ai_source": "chatgpt",
      "evaluation": "selected" | "rejected" | "partially_used",
      "reason": "why"
    }}
  ],
  "time_sensitive_override": false,
  "rationale": "2-3 sentences explaining your decision"
}}

Remember: If no opportunity meets the 80 conviction threshold, return decision_type: "NO_TRADE" with rationale explaining why."""

        return prompt

    def _parse_synthesis_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from Claude's response."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() in ["```", "```json"] else lines[1:])
            content = content.rstrip("`").strip()

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

        logger.error(f"Could not parse synthesis JSON: {content[:500]}")
        return None

    def _build_synthesis(
        self,
        parsed: Dict,
        scan_cycle_id: str,
        proposals: List[Dict]
    ) -> Dict[str, Any]:
        """Build full synthesis from parsed response."""

        return {
            "schema_version": "1.0.0",
            "synthesis_id": str(uuid.uuid4()),
            "scan_cycle_id": scan_cycle_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decision_type": parsed.get("decision_type", "NO_TRADE"),
            "selected_proposal": parsed.get("selected_proposal", {}),
            "recommendation": parsed.get("recommendation", {}),
            "cross_validation": parsed.get("cross_validation", {}),
            "proposal_evaluation": parsed.get("proposal_evaluation", []),
            "proceed_to_review": parsed.get("decision_type") == "TRADE",
            "time_sensitive_override": parsed.get("time_sensitive_override", False),
            "rationale": parsed.get("rationale", ""),
            "proposals_received": len(proposals)
        }

    def _empty_synthesis(self, scan_cycle_id: str, reason: str) -> Dict[str, Any]:
        """Return empty synthesis when generation fails."""
        return {
            "schema_version": "1.0.0",
            "synthesis_id": str(uuid.uuid4()),
            "scan_cycle_id": scan_cycle_id,
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


# Standalone function for use without mixin
async def claude_maca_synthesize(
    proposals: List[Dict[str, Any]],
    context: Dict[str, Any],
    scan_cycle_id: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Standalone function to synthesize proposals using Claude.

    Can be used without modifying ClaudeAnalyst class.
    """
    mixin = ClaudeMACAMixin()
    mixin.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    full_context = {
        "proposals": proposals,
        **context
    }

    return await mixin.synthesize_proposals(full_context, scan_cycle_id)
