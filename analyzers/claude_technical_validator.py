"""Claude Technical Validator

Uses Anthropic Claude to validate chart structure using the pre-computed
TechnicalScanner output (no price fetching). This acts as a check-and-balance
against narrative/facts/sentiment theses.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

from utils.json_utils import extract_first_json_object

logger = logging.getLogger(__name__)


class ClaudeTechnicalValidator:
    """Claude-based technical structure validator."""

    SYSTEM_PROMPT = """You are a Technical Structure Validator on an investment committee cross-examination.

You are given a pre-computed technical_analysis payload (market state, structure, scenarios, verdict).

RULES:
1) Do NOT fetch data. Do NOT browse. Do NOT invent indicators.
2) Your job is to judge whether a trade hypothesis is structurally allowed.
3) If structure is unclear or verdict=no_trade, prefer HOLD.
4) Provide a vote (BUY/SELL/HOLD) and confidence 0-1.

Return ONLY JSON in this schema:
{
  "claim": "1 sentence: your current position",
  "top_signals": ["exactly 2 short bullets"],
  "counterpoint": "1 sentence: strongest objection you acknowledge",
  "change_my_mind": "1 explicit condition",
  "verdict": "hypothesis_allowed"|"no_trade"|"analyze_only"|"unknown",
  "invalidation": "clear invalidation",
  "changed_mind": true|false,
  "vote": {"action": "BUY"|"SELL"|"HOLD", "ticker": "..."|null, "side": "BUY"|"SELL"|null, "confidence": 0.0-1.0}
}
"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - ClaudeTechnicalValidator disabled")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def initial_vote(
        self,
        *,
        scan_cycle_id: str,
        candidate_ticker: Optional[str],
        candidate_side: Optional[str],
        technical_analysis: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Produce an initial technical vote for the candidate ticker."""

        if not self.is_configured:
            return {
                "speaker": "claude_technical",
                "round": 0,
                "message": "Claude technical validator not configured",
                "verdict": "unknown",
                "invalidation": "N/A",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
            }

        payload = {
            "scan_cycle_id": scan_cycle_id,
            "candidate": {"ticker": candidate_ticker, "side": candidate_side},
            "technical_analysis": technical_analysis or {},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 900,
                        "temperature": 0.2,
                        "system": self.SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    },
                )

                if resp.status_code != 200:
                    return {
                        "speaker": "claude_technical",
                        "round": 0,
                        "message": f"Technical validator API error {resp.status_code}",
                        "verdict": "unknown",
                        "invalidation": "N/A",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "status": "error",
                    }

                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "")
                parsed, err = extract_first_json_object(text)
                if not parsed:
                    return {
                        "speaker": "claude_technical",
                        "round": 0,
                        "message": f"Technical validator parse error: {err}",
                        "verdict": "unknown",
                        "invalidation": "N/A",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "status": "error",
                    }
                parsed.update({"speaker": "claude_technical", "round": 0})
                return parsed

        except Exception as e:
            return {
                "speaker": "claude_technical",
                "round": 0,
                "message": f"Technical validator exception: {e}",
                "verdict": "unknown",
                "invalidation": "N/A",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
            }

    async def debate(
        self,
        *,
        scan_cycle_id: str,
        round_num: int,
        own_thesis: Dict[str, Any],
        other_theses: List[Dict[str, Any]],
        technical_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Participate in debate rounds (can revise vote)."""

        if not self.is_configured:
            return {
                "speaker": "claude_technical",
                "round": round_num,
                "message": "Claude technical validator not configured",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
            }

        payload = {
            "scan_cycle_id": scan_cycle_id,
            "round": round_num,
            "own_thesis": own_thesis,
            "other_theses": other_theses,
            "technical_analysis": technical_analysis or {},
        }

        # Reuse the system prompt; include instruction to revise if needed.
        system = self.SYSTEM_PROMPT + "\nYou are now debating other analysts. You may defend or revise your vote." 

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 900,
                        "temperature": 0.2,
                        "system": system,
                        "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    },
                )

                if resp.status_code != 200:
                    return {
                        "speaker": "claude_technical",
                        "round": round_num,
                        "message": f"Debate API error {resp.status_code}",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                        "status": "error",
                    }

                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "")
                parsed, err = extract_first_json_object(text)
                if not parsed:
                    return {
                        "speaker": "claude_technical",
                        "round": round_num,
                        "message": f"Debate parse error: {err}",
                        "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                        "changed_mind": False,
                        "status": "error",
                    }
                parsed.update({"speaker": "claude_technical", "round": round_num})
                return parsed

        except Exception as e:
            return {
                "speaker": "claude_technical",
                "round": round_num,
                "message": f"Debate exception: {e}",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
                "status": "error",
            }
