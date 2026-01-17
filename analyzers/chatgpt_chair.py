"""
ChatGPT Chair (Synthesizer) for Gann Sentinel Trader

This is a distinct role from ChatGPTAnalyst.

The Chair:
 - Does NOT discover new signals
 - Does NOT browse the web
 - Does NOT analyze charts directly
 - Synthesizes analyst theses into a final investment thesis
 - Acts as tie-breaker when the committee vote is tied
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)


class ChatGPTChair:
    """ChatGPT-based Investment Committee Chair / Synthesizer."""

    SYSTEM_PROMPT = """You are the Investment Committee Chair.

You are given:
- Analyst theses (narrative, facts, sentiment, technical)
- Optional debate transcript (two rounds)
- Signal inventory counts (FRED, Polymarket, Events, Technical)

STRICT RULES:
1) Do NOT invent signals or facts.
2) Do NOT browse the web.
3) Do NOT analyze charts directly; treat technical analyst output as the only technical input.
4) Be conservative: if evidence conflicts materially or is weak, choose HOLD.

Your tasks:
1) Produce a FINAL INVESTMENT THESIS (or HOLD).
2) Explicitly state why the trade exists NOW.
3) Provide clear invalidation conditions.
4) Provide a confidence score (0-1).
5) If the committee vote is tied, you must select the tie-break outcome.

Return ONLY valid JSON, no markdown."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set - ChatGPT Chair disabled")

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
                    "description": "OPENAI_API_KEY missing; unable to synthesize.",
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
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 1800,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                if resp.status_code != 200:
                    logger.error(f"Chair API error {resp.status_code}: {resp.text[:200]}")
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
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return parsed

        except Exception as e:
            logger.error(f"Chair synthesis exception: {e}")
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
