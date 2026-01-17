"""
Gann Sentinel Trader - Event Scanner
LevelFields-style event-driven signal detection via Grok API.

Scans for 27 corporate event types that historically move stock prices.
Uses Grok live_search to find market-wide events in real-time.

Version: 1.0.0
Last Updated: January 2026

Event Categories:
- Leadership: CEO exits, appointments, executive changes, insider activity
- Capital Allocation: Buybacks, dividends, return of capital
- Regulatory: FDA actions, DOJ investigations, lawsuits
- Index Changes: S&P 500 additions/removals, rebalancing
- External Pressure: Activist investors, short sellers, proxy fights
- Contracts: Government contracts, partnerships, contract losses
- Corporate Actions: M&A, spinoffs, bankruptcy, debt restructuring
"""

import os
import uuid
import hashlib
import logging
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPE DEFINITIONS (27 Total)
# =============================================================================

EVENT_TYPES = [
    # Leadership (5)
    "CEO_EXIT",
    "CEO_APPOINTMENT",
    "EXECUTIVE_DEPARTURE",
    "INSIDER_BUYING",
    "INSIDER_SELLING",
    # Capital Allocation (4)
    "STOCK_BUYBACK",
    "DIVIDEND_INCREASE",
    "DIVIDEND_CUT",
    "RETURN_OF_CAPITAL",
    # Regulatory (5)
    "FDA_BREAKTHROUGH",
    "FDA_APPROVAL",
    "FDA_REJECTION",
    "DOJ_INVESTIGATION",
    "CLASS_ACTION_LAWSUIT",
    # Index Changes (3)
    "SP500_ADDITION",
    "SP500_REMOVAL",
    "INDEX_REBALANCING",
    # External Pressure (3)
    "ACTIVIST_INVESTOR",
    "SHORT_SELLER_REPORT",
    "PROXY_FIGHT",
    # Contracts (3)
    "GOVERNMENT_CONTRACT",
    "MAJOR_PARTNERSHIP",
    "CONTRACT_LOSS",
    # Corporate Actions (4)
    "MA_ANNOUNCEMENT",
    "SPINOFF",
    "BANKRUPTCY_FILING",
    "DEBT_RESTRUCTURING",
]


# =============================================================================
# HISTORICAL PATTERNS (LevelFields-derived statistics)
# =============================================================================

EVENT_HISTORICAL_PATTERNS = {
    # Leadership Events
    "CEO_EXIT": {
        "bias": "mixed",
        "avg_move_pct": 0.0,
        "hold_days": 30,
        "win_rate": 0.50,
        "description": "CEO departure - impact depends on successor and context"
    },
    "CEO_APPOINTMENT": {
        "bias": "mixed",
        "avg_move_pct": 0.02,
        "hold_days": 10,
        "win_rate": 0.55,
        "description": "New CEO - market reaction depends on reputation"
    },
    "EXECUTIVE_DEPARTURE": {
        "bias": "mixed",
        "avg_move_pct": -0.01,
        "hold_days": 5,
        "win_rate": 0.48,
        "description": "C-suite departure other than CEO"
    },
    "INSIDER_BUYING": {
        "bias": "bullish",
        "avg_move_pct": 0.06,
        "hold_days": 30,
        "win_rate": 0.65,
        "description": "Significant insider purchases signal confidence"
    },
    "INSIDER_SELLING": {
        "bias": "bearish",
        "avg_move_pct": -0.03,
        "hold_days": 30,
        "win_rate": 0.55,
        "description": "Large insider sales can signal concerns"
    },

    # Capital Allocation Events
    "STOCK_BUYBACK": {
        "bias": "bullish",
        "avg_move_pct": 0.05,
        "hold_days": 10,
        "win_rate": 0.68,
        "description": "Share repurchase programs signal undervaluation"
    },
    "DIVIDEND_INCREASE": {
        "bias": "bullish",
        "avg_move_pct": 0.03,
        "hold_days": 5,
        "win_rate": 0.72,
        "description": "Dividend raises signal financial strength"
    },
    "DIVIDEND_CUT": {
        "bias": "bearish",
        "avg_move_pct": -0.08,
        "hold_days": 5,
        "win_rate": 0.70,
        "description": "Dividend cuts signal financial stress"
    },
    "RETURN_OF_CAPITAL": {
        "bias": "bullish",
        "avg_move_pct": 0.04,
        "hold_days": 10,
        "win_rate": 0.65,
        "description": "Special dividends or capital returns"
    },

    # Regulatory Events
    "FDA_BREAKTHROUGH": {
        "bias": "bullish",
        "avg_move_pct": 0.12,
        "hold_days": 3,
        "win_rate": 0.75,
        "description": "Breakthrough Therapy Designation - fast-track status"
    },
    "FDA_APPROVAL": {
        "bias": "bullish",
        "avg_move_pct": 0.15,
        "hold_days": 5,
        "win_rate": 0.78,
        "description": "Drug/device approval - major catalyst"
    },
    "FDA_REJECTION": {
        "bias": "bearish",
        "avg_move_pct": -0.25,
        "hold_days": 5,
        "win_rate": 0.80,
        "description": "Complete Response Letter or rejection"
    },
    "DOJ_INVESTIGATION": {
        "bias": "bearish",
        "avg_move_pct": -0.10,
        "hold_days": 30,
        "win_rate": 0.60,
        "description": "Department of Justice investigation announced"
    },
    "CLASS_ACTION_LAWSUIT": {
        "bias": "bearish",
        "avg_move_pct": -0.05,
        "hold_days": 10,
        "win_rate": 0.58,
        "description": "Securities class action filed"
    },

    # Index Changes
    "SP500_ADDITION": {
        "bias": "bullish",
        "avg_move_pct": 0.08,
        "hold_days": 5,
        "win_rate": 0.80,
        "description": "S&P 500 inclusion - forced ETF buying"
    },
    "SP500_REMOVAL": {
        "bias": "bearish",
        "avg_move_pct": -0.10,
        "hold_days": 5,
        "win_rate": 0.75,
        "description": "S&P 500 removal - forced ETF selling"
    },
    "INDEX_REBALANCING": {
        "bias": "mixed",
        "avg_move_pct": 0.02,
        "hold_days": 3,
        "win_rate": 0.52,
        "description": "Major index rebalancing event"
    },

    # External Pressure
    "ACTIVIST_INVESTOR": {
        "bias": "bullish",
        "avg_move_pct": 0.10,
        "hold_days": 30,
        "win_rate": 0.62,
        "description": "Activist stake disclosed - potential for change"
    },
    "SHORT_SELLER_REPORT": {
        "bias": "bearish",
        "avg_move_pct": -0.15,
        "hold_days": 3,
        "win_rate": 0.65,
        "description": "Short seller publishes negative research"
    },
    "PROXY_FIGHT": {
        "bias": "mixed",
        "avg_move_pct": 0.05,
        "hold_days": 60,
        "win_rate": 0.55,
        "description": "Proxy contest for board control"
    },

    # Contracts
    "GOVERNMENT_CONTRACT": {
        "bias": "bullish",
        "avg_move_pct": 0.07,
        "hold_days": 5,
        "win_rate": 0.70,
        "description": "Major government contract win"
    },
    "MAJOR_PARTNERSHIP": {
        "bias": "bullish",
        "avg_move_pct": 0.06,
        "hold_days": 10,
        "win_rate": 0.65,
        "description": "Strategic partnership announcement"
    },
    "CONTRACT_LOSS": {
        "bias": "bearish",
        "avg_move_pct": -0.08,
        "hold_days": 5,
        "win_rate": 0.68,
        "description": "Loss of major contract or customer"
    },

    # Corporate Actions
    "MA_ANNOUNCEMENT": {
        "bias": "mixed",
        "avg_move_pct": 0.03,
        "hold_days": 30,
        "win_rate": 0.52,
        "description": "M&A announcement - acquirer often down, target up"
    },
    "SPINOFF": {
        "bias": "mixed",
        "avg_move_pct": 0.05,
        "hold_days": 60,
        "win_rate": 0.58,
        "description": "Corporate spinoff announcement"
    },
    "BANKRUPTCY_FILING": {
        "bias": "bearish",
        "avg_move_pct": -0.50,
        "hold_days": 1,
        "win_rate": 0.90,
        "description": "Chapter 11 or Chapter 7 filing"
    },
    "DEBT_RESTRUCTURING": {
        "bias": "mixed",
        "avg_move_pct": 0.0,
        "hold_days": 30,
        "win_rate": 0.50,
        "description": "Debt restructuring or refinancing"
    },
}


# Source quality ratings
SOURCE_QUALITY = {
    "SEC Filing": 0.95,
    "SEC": 0.95,
    "13D": 0.95,
    "13F": 0.90,
    "8-K": 0.95,
    "FDA.gov": 0.95,
    "FDA": 0.95,
    "Defense.gov": 0.90,
    "Company press release": 0.85,
    "PR Newswire": 0.80,
    "Bloomberg": 0.85,
    "Reuters": 0.85,
    "CNBC": 0.75,
    "WSJ": 0.85,
    "Wall Street Journal": 0.85,
    "Financial Times": 0.85,
    "News": 0.60,
    "Twitter": 0.40,
    "X": 0.40,
    "Unknown": 0.50,
}


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class EventSignal:
    """
    Signal representing a corporate event that may move stock price.
    Conforms to GST signal specification.
    """
    signal_id: str
    dedup_hash: str
    category: str  # Always "corporate_event"
    source_type: str  # Always "event_scanner"
    event_type: str  # One of EVENT_TYPES

    # Asset scope
    asset_scope: Dict[str, List[str]]

    # Signal content
    summary: str
    raw_value: Dict[str, Any]
    evidence: List[Dict[str, Any]]

    # Scoring
    confidence: float
    confidence_factors: Dict[str, float]
    directional_bias: str  # "bullish", "bearish", "mixed"
    time_horizon: str  # "days", "weeks"

    # Metadata
    novelty: str
    staleness_policy: Dict[str, Any]
    uncertainties: List[str]
    timestamp_utc: str

    # Event-specific
    event_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.category,
            "category": self.category,
            "source_type": self.source_type,
            "source": self.source_type,
            "event_type": self.event_type,
            "asset_scope": self.asset_scope,
            "summary": self.summary,
            "raw_value": self.raw_value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "directional_bias": self.directional_bias,
            "time_horizon": self.time_horizon,
            "novelty": self.novelty,
            "staleness_policy": self.staleness_policy,
            "staleness_seconds": self.staleness_policy.get("max_age_seconds", 86400),
            "uncertainties": self.uncertainties,
            "timestamp_utc": self.timestamp_utc,
            "event_metadata": self.event_metadata,
        }


# =============================================================================
# EVENT SCANNER
# =============================================================================

class EventScanner:
    """
    Scanner for corporate events via Grok API.

    Implements LevelFields-style event detection:
    - Scans for 27 event types that historically move stock prices
    - Assigns directional bias based on historical patterns
    - Calculates confidence from source quality and recency
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Event Scanner."""
        self.api_key = api_key if api_key is not None else os.getenv("XAI_API_KEY")

        # Diagnostic logging
        logger.info(f"EventScanner init - XAI_API_KEY present: {bool(self.api_key)}")
        if self.api_key:
            logger.info(f"EventScanner init - API key length: {len(self.api_key)} chars")
        
        if not self.api_key:
            logger.warning("XAI_API_KEY not set - Event Scanner disabled")
        else:
            logger.info("XAI_API_KEY configured for Event Scanner")

        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-3-fast-beta"

        # Deduplication cache
        self._seen_hashes: Dict[str, datetime] = {}

        # Error tracking
        self.last_error: Optional[str] = None
        self.last_raw_response: Optional[str] = None

        logger.info("EventScanner v1.0.0 initialized - 27 event types")

    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return bool(self.api_key)

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_market_wide_prompt(self) -> str:
        """
        Build comprehensive prompt for market-wide event scan.

        Single query covers all 27 event types to minimize API costs.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return f"""You are a financial news analyst scanning for corporate events that move stock prices.

Search the web and news for ANY of these 27 event types that occurred in the last 24 hours (since {today}):

LEADERSHIP CHANGES:
- CEO Exit or resignation
- CEO Appointment
- Executive departures (CFO, COO, etc.)
- Significant insider buying (Form 4 filings)
- Significant insider selling (Form 4 filings)

CAPITAL ALLOCATION:
- Stock buyback announcements
- Dividend increases
- Dividend cuts or suspensions
- Special dividends or return of capital

REGULATORY/LEGAL:
- FDA Breakthrough Therapy Designation
- FDA Approval
- FDA Rejection or Complete Response Letter
- DOJ Investigation announced
- Class action lawsuit filed

INDEX CHANGES:
- S&P 500 additions
- S&P 500 removals
- Major index rebalancing (Russell, NASDAQ-100, etc.)

EXTERNAL PRESSURE:
- Activist investor 13D filing (5%+ stake)
- Short seller report published
- Proxy fight or board contest

CONTRACTS & PARTNERSHIPS:
- Government contract wins
- Major partnership announcements
- Contract losses or customer departures

CORPORATE ACTIONS:
- M&A announcements (merger, acquisition, tender offer)
- Spinoff announcements
- Bankruptcy filings
- Debt restructuring

Respond with valid JSON:
{{
    "scan_date": "{today}",
    "events": [
        {{
            "ticker": "SYMBOL",
            "event_type": "EVENT_TYPE_FROM_LIST_ABOVE",
            "headline": "Brief headline describing the event",
            "event_date": "YYYY-MM-DD",
            "source": "Source name (e.g., SEC Filing, FDA.gov, Bloomberg)",
            "details": "1-2 sentence details"
        }}
    ]
}}

IMPORTANT:
- Only include events from the last 24 hours
- Use EXACT event type names matching the categories above
- Include the ticker symbol for each event
- If no events found, return {{"scan_date": "{today}", "events": []}}
- Focus on US-listed stocks (NYSE, NASDAQ)

JSON only, no other text."""

    def _normalize_event_type(self, raw_type: str) -> str:
        """Normalize event type string to our standard format."""
        # Clean and uppercase
        normalized = raw_type.upper().strip()
        normalized = re.sub(r'[^A-Z0-9_]', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        # Direct matches
        if normalized in EVENT_TYPES:
            return normalized

        # Common variations mapping
        type_mapping = {
            # CEO
            "CEO_RESIGNATION": "CEO_EXIT",
            "CEO_DEPARTURE": "CEO_EXIT",
            "CEO_LEAVES": "CEO_EXIT",
            "CEO_STEPPING_DOWN": "CEO_EXIT",
            "NEW_CEO": "CEO_APPOINTMENT",
            "CEO_HIRE": "CEO_APPOINTMENT",
            "CEO_NAMED": "CEO_APPOINTMENT",

            # Executives
            "CFO_EXIT": "EXECUTIVE_DEPARTURE",
            "COO_EXIT": "EXECUTIVE_DEPARTURE",
            "CTO_EXIT": "EXECUTIVE_DEPARTURE",
            "EXECUTIVE_EXIT": "EXECUTIVE_DEPARTURE",
            "EXECUTIVE_RESIGNATION": "EXECUTIVE_DEPARTURE",

            # Buybacks
            "SHARE_REPURCHASE": "STOCK_BUYBACK",
            "BUYBACK": "STOCK_BUYBACK",
            "REPURCHASE": "STOCK_BUYBACK",
            "SHARE_BUYBACK": "STOCK_BUYBACK",

            # Dividends
            "DIVIDEND_RAISE": "DIVIDEND_INCREASE",
            "DIVIDEND_HIKE": "DIVIDEND_INCREASE",
            "SPECIAL_DIVIDEND": "RETURN_OF_CAPITAL",
            "DIVIDEND_SUSPENSION": "DIVIDEND_CUT",

            # FDA
            "BREAKTHROUGH_THERAPY": "FDA_BREAKTHROUGH",
            "BTD": "FDA_BREAKTHROUGH",
            "FDA_BTD": "FDA_BREAKTHROUGH",
            "DRUG_APPROVAL": "FDA_APPROVAL",
            "CRL": "FDA_REJECTION",
            "COMPLETE_RESPONSE_LETTER": "FDA_REJECTION",

            # Index
            "S_P_500_ADDITION": "SP500_ADDITION",
            "S_P500_ADDITION": "SP500_ADDITION",
            "SP_500_ADDITION": "SP500_ADDITION",
            "S_P_500_REMOVAL": "SP500_REMOVAL",
            "S_P500_REMOVAL": "SP500_REMOVAL",
            "SP_500_REMOVAL": "SP500_REMOVAL",
            "RUSSELL_REBALANCING": "INDEX_REBALANCING",
            "NASDAQ_REBALANCING": "INDEX_REBALANCING",

            # Activist/Shorts
            "13D_FILING": "ACTIVIST_INVESTOR",
            "ACTIVIST_STAKE": "ACTIVIST_INVESTOR",
            "SHORT_REPORT": "SHORT_SELLER_REPORT",
            "SHORT_ATTACK": "SHORT_SELLER_REPORT",

            # Contracts
            "GOV_CONTRACT": "GOVERNMENT_CONTRACT",
            "DEFENSE_CONTRACT": "GOVERNMENT_CONTRACT",
            "DOD_CONTRACT": "GOVERNMENT_CONTRACT",
            "PARTNERSHIP": "MAJOR_PARTNERSHIP",
            "STRATEGIC_PARTNERSHIP": "MAJOR_PARTNERSHIP",
            "CONTRACT_WIN": "GOVERNMENT_CONTRACT",

            # M&A
            "MERGER": "MA_ANNOUNCEMENT",
            "ACQUISITION": "MA_ANNOUNCEMENT",
            "TENDER_OFFER": "MA_ANNOUNCEMENT",
            "TAKEOVER": "MA_ANNOUNCEMENT",
            "SPIN_OFF": "SPINOFF",
            "CHAPTER_11": "BANKRUPTCY_FILING",
            "CHAPTER_7": "BANKRUPTCY_FILING",
            "BANKRUPTCY": "BANKRUPTCY_FILING",
            "DEBT_REFINANCING": "DEBT_RESTRUCTURING",

            # Insider
            "INSIDER_PURCHASE": "INSIDER_BUYING",
            "INSIDER_BUY": "INSIDER_BUYING",
            "INSIDER_SALE": "INSIDER_SELLING",
            "INSIDER_SELL": "INSIDER_SELLING",
            "FORM_4_BUYING": "INSIDER_BUYING",
            "FORM_4_SELLING": "INSIDER_SELLING",

            # Legal
            "DOJ_PROBE": "DOJ_INVESTIGATION",
            "INVESTIGATION": "DOJ_INVESTIGATION",
            "CLASS_ACTION": "CLASS_ACTION_LAWSUIT",
            "LAWSUIT": "CLASS_ACTION_LAWSUIT",
            "SECURITIES_LAWSUIT": "CLASS_ACTION_LAWSUIT",
        }

        if normalized in type_mapping:
            return type_mapping[normalized]

        # Fuzzy matching for partial matches
        for known_type in EVENT_TYPES:
            if known_type in normalized or normalized in known_type:
                return known_type

        # Default to OTHER for unknown types
        logger.warning(f"Unknown event type: {raw_type} -> treating as OTHER")
        return "OTHER"

    # =========================================================================
    # API CALLS
    # =========================================================================

    async def _call_grok_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Grok API with live search enabled."""
        if not self.is_configured:
            self.last_error = "API key not configured"
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "search_parameters": {
                "mode": "auto",
                "return_citations": True,
                "from_date": (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d"),
            },
            "temperature": 0.1,  # Low temperature for factual extraction
        }

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    self.last_error = f"API error: {response.status_code} - {response.text[:200]}"
                    logger.error(self.last_error)
                    return None

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                self.last_raw_response = content

                # Parse JSON from response
                parsed = self._extract_json_from_response(content)

                if parsed is None:
                    logger.warning("Could not parse JSON from Grok response")
                    return {"events": [], "_parse_failed": True, "_raw": content}

                return parsed

        except httpx.TimeoutException:
            self.last_error = "API request timed out (90s)"
            logger.error(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"API call failed: {str(e)}"
            logger.error(self.last_error)
            return None

    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from Grok response."""
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*"events"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try broader JSON pattern
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_dedup_hash(self, event_type: str, ticker: str, headline: str) -> str:
        """Generate deduplication hash for an event."""
        # Normalize inputs
        normalized = f"{event_type}:{ticker.upper()}:{headline.lower().strip()[:80]}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _calculate_confidence(
        self,
        event_type: str,
        source: str,
        hours_since_event: float,
        has_corroboration: bool = False
    ) -> float:
        """
        Calculate confidence score for an event signal.

        Factors:
        - Source quality (SEC filings > news > social)
        - Recency (more recent = higher confidence)
        - Corroboration (multiple sources = higher)
        """
        # Base confidence from source quality
        source_quality = 0.50
        for source_pattern, quality in SOURCE_QUALITY.items():
            if source_pattern.lower() in source.lower():
                source_quality = max(source_quality, quality)
                break

        # Recency factor (1.0 for < 1 hour, decreasing to 0.5 at 24 hours)
        if hours_since_event <= 1:
            recency_factor = 1.0
        elif hours_since_event <= 6:
            recency_factor = 0.95 - (hours_since_event - 1) * 0.03
        elif hours_since_event <= 12:
            recency_factor = 0.80 - (hours_since_event - 6) * 0.02
        elif hours_since_event <= 24:
            recency_factor = 0.68 - (hours_since_event - 12) * 0.015
        else:
            recency_factor = 0.50

        # Corroboration bonus
        corroboration_factor = 1.1 if has_corroboration else 1.0

        # Calculate final confidence
        confidence = source_quality * recency_factor * corroboration_factor

        # Clamp to reasonable range
        return max(0.30, min(0.95, confidence))

    def _calculate_hours_since_event(self, event_date_str: str) -> float:
        """Calculate hours since event occurred."""
        try:
            # Try various date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    event_date = datetime.strptime(event_date_str, fmt)
                    event_date = event_date.replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            else:
                # Default to today if can't parse
                return 12.0

            now = datetime.now(timezone.utc)
            delta = now - event_date
            return max(0, delta.total_seconds() / 3600)

        except Exception:
            return 12.0  # Default to 12 hours

    def _parse_events_response(self, response: Dict[str, Any]) -> List[EventSignal]:
        """Parse Grok response into EventSignal objects."""
        signals = []
        seen_hashes = set()

        events = response.get("events", [])

        if not events:
            logger.info("No events found in response")
            return []

        now = datetime.now(timezone.utc)

        for event in events:
            try:
                ticker = event.get("ticker", "").upper().strip()
                raw_event_type = event.get("event_type", "")
                headline = event.get("headline", "")
                event_date = event.get("event_date", now.strftime("%Y-%m-%d"))
                source = event.get("source", "Unknown")
                details = event.get("details", "")

                # Skip if missing required fields
                if not ticker or not raw_event_type:
                    logger.warning(f"Skipping event with missing ticker/type: {event}")
                    continue

                # Normalize event type
                event_type = self._normalize_event_type(raw_event_type)

                # Skip truly unknown events
                if event_type == "OTHER" and raw_event_type not in EVENT_TYPES:
                    logger.info(f"Skipping unknown event type: {raw_event_type}")
                    continue

                # Generate dedup hash
                dedup_hash = self._generate_dedup_hash(event_type, ticker, headline)

                # Skip duplicates
                if dedup_hash in seen_hashes or dedup_hash in self._seen_hashes:
                    logger.debug(f"Skipping duplicate event: {ticker} {event_type}")
                    continue

                seen_hashes.add(dedup_hash)
                self._seen_hashes[dedup_hash] = now

                # Calculate hours since event
                hours_since = self._calculate_hours_since_event(event_date)

                # Check if actionable (within 24 hours)
                actionable = hours_since <= 24

                # Get historical pattern
                pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {
                    "bias": "mixed",
                    "avg_move_pct": 0.0,
                    "hold_days": 10,
                    "win_rate": 0.50,
                    "description": "Unknown event type"
                })

                # Calculate confidence
                confidence = self._calculate_confidence(
                    event_type=event_type,
                    source=source,
                    hours_since_event=hours_since,
                    has_corroboration=False
                )

                # Build summary
                bias_emoji = "📈" if pattern["bias"] == "bullish" else "📉" if pattern["bias"] == "bearish" else "↔️"
                summary = (
                    f"{ticker}: {headline[:80]}. "
                    f"Historical: {pattern['avg_move_pct']*100:+.1f}% avg move, "
                    f"{pattern['win_rate']*100:.0f}% win rate {bias_emoji}"
                )

                # Determine time horizon based on hold days
                hold_days = pattern.get("hold_days", 10)
                time_horizon = "days" if hold_days <= 10 else "weeks"

                # Create signal
                signal = EventSignal(
                    signal_id=str(uuid.uuid4()),
                    dedup_hash=dedup_hash,
                    category="corporate_event",
                    source_type="event_scanner",
                    event_type=event_type,
                    asset_scope={
                        "tickers": [ticker],
                        "sectors": [],
                        "macro_regions": ["US"],
                        "asset_classes": ["EQUITY"],
                    },
                    summary=summary,
                    raw_value={
                        "event_type": event_type,
                        "event_date": event_date,
                        "historical_win_rate": pattern["win_rate"],
                        "historical_avg_move": pattern["avg_move_pct"],
                        "time_since_event": f"{hours_since:.1f} hours ago",
                        "hold_days": pattern["hold_days"],
                    },
                    evidence=[{
                        "source": source,
                        "source_tier": "official" if confidence > 0.8 else "news",
                        "excerpt": f"{headline}. {details}"[:200],
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=confidence,
                    confidence_factors={
                        "source_quality": self._get_source_quality(source),
                        "recency": max(0.5, 1.0 - hours_since / 48),
                        "corroboration": 1.0,
                    },
                    directional_bias=pattern["bias"],
                    time_horizon=time_horizon,
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 86400,  # 24 hours
                        "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
                    },
                    uncertainties=[
                        f"Historical pattern may not repeat",
                        f"Market conditions may differ from historical average",
                    ],
                    timestamp_utc=now.isoformat(),
                    event_metadata={
                        "catalyst_window": f"1-{pattern['hold_days']} days typical impact",
                        "actionable": actionable,
                        "hours_since_event": hours_since,
                        "pattern_description": pattern.get("description", ""),
                    },
                )

                signals.append(signal)
                logger.info(f"Event detected: {ticker} - {event_type} ({pattern['bias']})")

            except Exception as e:
                logger.error(f"Error parsing event: {e} - {event}")
                continue

        return signals

    def _get_source_quality(self, source: str) -> float:
        """Get source quality score."""
        for pattern, quality in SOURCE_QUALITY.items():
            if pattern.lower() in source.lower():
                return quality
        return 0.50

    # =========================================================================
    # PUBLIC SCAN METHODS
    # =========================================================================

    async def scan_market_wide(self) -> List[EventSignal]:
        """
        Scan for corporate events across all US stocks.

        This is a single comprehensive query covering all 27 event types
        to minimize API costs while maximizing coverage.

        Returns:
            List of EventSignal objects for detected events
        """
        if not self.is_configured:
            logger.warning("Event Scanner not configured - skipping scan")
            logger.warning(f"  - API key present: {bool(self.api_key)}")
            return []

        logger.info("Starting market-wide event scan (27 event types)...")
        logger.info(f"  - Using model: {self.model}")
        logger.info(f"  - Base URL: {self.base_url}")

        prompt = self._build_market_wide_prompt()

        response = await self._call_grok_api(prompt)

        if response is None:
            logger.error(f"Event scan failed: {self.last_error}")
            return []

        # Log raw response info for debugging
        if response.get("_parse_failed"):
            logger.warning(f"JSON parse failed. Raw response (first 500 chars): {response.get('_raw', '')[:500]}")
        
        events_count = len(response.get("events", []))
        logger.info(f"Grok API returned {events_count} raw events")

        signals = self._parse_events_response(response)

        logger.info(f"Event scan complete: {len(signals)} events detected (after filtering)")

        return signals

    async def scan_ticker(self, ticker: str) -> List[EventSignal]:
        """
        Scan for events affecting a specific ticker.

        Args:
            ticker: Stock symbol to scan

        Returns:
            List of EventSignal objects
        """
        if not self.is_configured:
            return []

        # For now, do market-wide scan and filter
        # In future, could optimize with ticker-specific query
        all_signals = await self.scan_market_wide()

        return [s for s in all_signals if ticker.upper() in s.asset_scope.get("tickers", [])]

    def clear_dedup_cache(self):
        """Clear the deduplication cache."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._seen_hashes = {
            h: t for h, t in self._seen_hashes.items()
            if t > cutoff
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("EVENT SCANNER TEST")
        print("=" * 60)

        scanner = EventScanner()

        if scanner.is_configured:
            print(f"\n✓ Scanner configured")
            print(f"  Event types: {len(EVENT_TYPES)}")
            print(f"  Historical patterns: {len(EVENT_HISTORICAL_PATTERNS)}")

            print("\n--- Running market-wide scan ---")
            signals = await scanner.scan_market_wide()

            print(f"\nFound {len(signals)} events:")
            for signal in signals:
                print(f"\n  {signal.asset_scope['tickers'][0]}: {signal.event_type}")
                print(f"    Bias: {signal.directional_bias}")
                print(f"    Confidence: {signal.confidence:.2f}")
                print(f"    Summary: {signal.summary[:100]}...")
        else:
            print("\n✗ Scanner not configured (XAI_API_KEY missing)")
            print("  Set XAI_API_KEY environment variable to test")

    asyncio.run(test())
