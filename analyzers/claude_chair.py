"""
Claude Chair (Senior Trader) for Gann Sentinel Trader.

Synthesizes analyst theses and debate into a final decision.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

from utils.json_utils import extract_first_json_object

logger = logging.getLogger(__name__)


class ClaudeChair:
    """Claude-based Senior Trader / Synthesizer."""

    SYSTEM_PROMPT = """You are the Senior Trader and committee chair.

You are given:
- Analyst theses (Grok, Perplexity, ChatGPT)
- Debate transcript (2 rounds)
- Vote summary (may include a 1-1-1 tie)
- Technical analysis (precomputed chart structure)

STRICT RULES:
1) Do NOT invent signals or facts.
2) Do NOT browse the web.
3) You MAY use the provided technical_analysis (no new charting).
4) If evidence conflicts materially or is weak, choose HOLD.
5) If vote_summary indicates HOLD (consensus failure), default to HOLD unless
   you can clearly justify a trade with strong evidence and technical support.
6) If vote_summary indicates a 1-1-1 tie, you MUST break the tie.

Return ONLY valid JSON, no markdown."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Claude Chair disabled")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def synthesize(
        self,
        *,
        cycle_id: str,
        proposals: List[Dict[str, Any]],
        debate: Optional[Dict[str, Any]] = None,
        signal_inventory: Optional[Dict[str, Any]] = None,
        technical_analysis: Optional[Dict[str, Any]] = None,
        vote_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize proposals and debate into a final thesis."""

        if not self.is_configured:
            return {
                "decision_type": "NO_TRADE",
                "final_thesis": {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "summary": "Chair not configured",
                    "description": "ANTHROPIC_API_KEY missing; unable to synthesize.",
                    "invalidation": "N/A",
                },
                "tie_break_used": False,
            }

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        payload = {
            "date": date_str,
            "cycle_id": cycle_id,
            "signal_inventory": signal_inventory or {},
            "proposals": proposals or [],
            "technical_analysis": technical_analysis or {},
            "debate": debate or {},
            "vote_summary": vote_summary or {},
        }

        user_prompt = (
            "SYNTHESIZE THE COMMITTEE INPUTS INTO A FINAL THESIS.\n"
            "Return JSON in this schema:\n"
            "{\n"
            "  \"decision_type\": \"TRADE\"|\"WATCH\"|\"NO_TRADE\",\n"
            "  \"tie_break_used\": true|false,\n"
            "  \"final_thesis\": {\n"
            "    \"action\": \"BUY\"|\"SELL\"|\"HOLD\",\n"
            "    \"ticker\": \"...\"|null,\n"
            "    \"time_horizon\": \"intraday\"|\"swing\"|\"multiweek\"|\"unspecified\",\n"
            "    \"confidence\": 0.0-1.0,\n"
            "    \"summary\": \"one sentence\",\n"
            "    \"description\": \"3-8 sentences, investor-grade\",\n"
            "    \"signals_used_count\": {\"total\": 0, \"by_source\": {}},\n"
            "    \"top_signals\": [\"...\"],\n"
            "    \"invalidation\": \"clear condition\"\n"
            "  },\n"
            "  \"committee_notes\": {\n"
            "    \"agreement\": [\"...\"],\n"
            "    \"disagreement\": [\"...\"],\n"
            "    \"why_now\": \"...\"\n"
            "  }\n"
            "}\n\n"
            "INPUTS (JSON):\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": 1800,
                "temperature": 0.2,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.base_url, headers=headers, json=body)
                if resp.status_code != 200:
                    logger.error(f"Claude Chair API error {resp.status_code}: {resp.text[:200]}")
                    return {
                        "decision_type": "NO_TRADE",
                        "final_thesis": {
                            "action": "HOLD",
                            "confidence": 0.0,
                            "summary": "Chair synthesis failed",
                            "description": f"API error {resp.status_code}",
                            "invalidation": "N/A",
                        },
                        "tie_break_used": False,
                    }

                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "")
                parsed, err = extract_first_json_object(text)
                if not parsed:
                    logger.error(f"Claude Chair parse error: {err}")
                    return {
                        "decision_type": "NO_TRADE",
                        "final_thesis": {
                            "action": "HOLD",
                            "confidence": 0.0,
                            "summary": "Chair parse error",
                            "description": f"Parse error: {err}",
                            "invalidation": "N/A",
                        },
                        "tie_break_used": False,
                    }

                return parsed

        except Exception as e:
            logger.error(f"Claude Chair synthesis exception: {e}")
            return {
                "decision_type": "NO_TRADE",
                "final_thesis": {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "summary": "Chair synthesis exception",
                    "description": str(e),
                    "invalidation": "N/A",
                },
                "tie_break_used": False,
            }
