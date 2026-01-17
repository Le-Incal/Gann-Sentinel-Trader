"""
Gann Sentinel Trader - Grok Scanner
Deep Business Intelligence via xAI Grok API.

Version: 3.0.0 (Deep Intelligence Update)
Last Updated: January 2026

Change Log:
- 3.0.0: Added deep business intelligence prompts
         - scan_ticker_social(): X/Twitter deep dive on sentiment and narratives
         - scan_ticker_fundamentals(): Web/news business intelligence
         - scan_ticker_deep(): Combined comprehensive analysis
         - Business line mapping, TAM analysis, contrarian signals
- 2.2.0: Fixed xAI API - use search_parameters instead of tools for Live Search
"""

import os
import uuid
import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import httpx

# Import our temporal framework
from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    SignalRelevance,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class GrokModel(Enum):
    """Available Grok models."""
    GROK_FAST = "grok-3-fast-beta"
    GROK_REASONING = "grok-3-mini-fast-beta"


class SignalCategory(Enum):
    """Categories of signals Grok can generate."""
    SENTIMENT = "sentiment"
    NEWS = "news"
    NARRATIVE_SHIFT = "narrative_shift"
    EVENT = "event"
    BUSINESS_INTEL = "business_intel"


# Sector mappings - Updated to recognize platform companies
TICKER_TO_SECTOR = {
    # Platform/Multi-TAM companies (analyze ALL business lines)
    "TSLA": "PLATFORM",   # Auto + Energy + AI/FSD + Robotics + Solar
    "AMZN": "PLATFORM",   # E-commerce + AWS + Advertising + Logistics
    "GOOGL": "PLATFORM",  # Search + Cloud + YouTube + Waymo + DeepMind
    "META": "PLATFORM",   # Social + Advertising + Reality Labs + AI
    "MSFT": "PLATFORM",   # Cloud + Enterprise + Gaming + AI/Copilot
    "AAPL": "PLATFORM",   # Hardware + Services + AI
    
    # Tech (primarily single TAM focus)
    "NVDA": "TECH", "AMD": "TECH", "SMCI": "TECH",
    "INTC": "TECH", "AVGO": "TECH", "MRVL": "TECH",
    
    # Financials
    "JPM": "FINANCIALS", "GS": "FINANCIALS", "MS": "FINANCIALS",
    
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "OXY": "ENERGY",
    
    # Aerospace
    "RKLB": "AEROSPACE", "LMT": "AEROSPACE", "RTX": "AEROSPACE", "BA": "AEROSPACE",
    
    # Crypto
    "COIN": "CRYPTO", "MSTR": "CRYPTO",
    
    # Indexes
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
}

# Platform company business lines (for context in prompts)
PLATFORM_BUSINESS_LINES = {
    "TSLA": [
        "Automotive (EVs, Cybertruck)",
        "Energy Storage (Megapack, Powerwall)",
        "Solar (Solar Roof, panels)",
        "Full Self-Driving (FSD subscription/licensing)",
        "Robotaxi (autonomous ride-hailing)",
        "Optimus (humanoid robotics)",
        "Supercharging Network",
        "AI/Dojo (training infrastructure)",
    ],
    "AMZN": [
        "E-commerce (retail, marketplace)",
        "AWS (cloud infrastructure)",
        "Advertising",
        "Prime (subscription)",
        "Logistics (delivery network)",
        "Devices (Alexa, Fire)",
    ],
    "GOOGL": [
        "Search (advertising)",
        "YouTube (video, advertising)",
        "Google Cloud",
        "Waymo (autonomous vehicles)",
        "DeepMind (AI research)",
        "Android ecosystem",
        "Hardware (Pixel, Nest)",
    ],
    "MSFT": [
        "Azure (cloud)",
        "Office 365 / M365",
        "Windows",
        "LinkedIn",
        "Gaming (Xbox, Activision)",
        "AI/Copilot",
        "GitHub",
    ],
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GrokSignal:
    """Signal extracted from Grok conforming to Grok Spec v1.1.0."""
    signal_id: str
    dedup_hash: str
    category: str
    source_type: str
    asset_scope: Dict[str, List[str]]
    summary: str
    raw_value: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    confidence_factors: Dict[str, float]
    directional_bias: str
    time_horizon: str
    novelty: str
    staleness_policy: Dict[str, Any]
    uncertainties: List[str]
    timestamp_utc: str
    forward_horizon: Optional[str] = None
    catalyst_date: Optional[str] = None
    # New fields for deep intel
    business_lines: Optional[List[Dict[str, Any]]] = None
    contrarian_signals: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.category,
            "category": self.category,
            "source_type": self.source_type,
            "source": self.source_type,
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
            "staleness_seconds": self.staleness_policy.get("max_age_seconds", 3600),
            "uncertainties": self.uncertainties,
            "timestamp_utc": self.timestamp_utc,
            "forward_horizon": self.forward_horizon,
            "catalyst_date": self.catalyst_date,
            "business_lines": self.business_lines,
            "contrarian_signals": self.contrarian_signals,
        }


# =============================================================================
# GROK SCANNER
# =============================================================================

class GrokScanner:
    """
    Scanner for deep business intelligence via xAI Grok API.
    
    Key Methods:
    - scan_ticker_social(): X/Twitter sentiment and narratives
    - scan_ticker_fundamentals(): Web/news business intelligence
    - scan_ticker_deep(): Combined comprehensive analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Grok scanner."""
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        
        if not self.api_key:
            logger.warning("XAI_API_KEY not set - Grok scanning disabled")
        else:
            logger.info(f"XAI_API_KEY configured (length: {len(self.api_key)})")
        
        self.base_url = "https://api.x.ai/v1"
        self.model = GrokModel.GROK_FAST.value
        
        # Initialize temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for deduplication
        self._seen_signals: Dict[str, datetime] = {}
        
        # ERROR TRACKING for diagnostics
        self.last_error: Optional[str] = None
        self.last_raw_response: Optional[str] = None
        
        logger.info("GrokScanner v3.0.0 initialized - Deep Intelligence Mode")
    
    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return bool(self.api_key)
    
    # =========================================================================
    # DEEP INTELLIGENCE PROMPTS
    # =========================================================================
    
    def _build_deep_social_prompt(self, ticker: str) -> str:
        """
        Build prompt for deep X/Twitter analysis.
        
        This prompt asks Grok to:
        1. Map sentiment BY business line (not just overall)
        2. Identify narrative shifts
        3. Surface contrarian signals
        4. Find emerging discussions
        """
        sector = TICKER_TO_SECTOR.get(ticker, "TECH")
        
        # Get known business lines if platform company
        business_lines_hint = ""
        if ticker in PLATFORM_BUSINESS_LINES:
            lines = PLATFORM_BUSINESS_LINES[ticker]
            business_lines_hint = f"""
Known business lines for {ticker}:
{chr(10).join(f'- {line}' for line in lines)}

Analyze sentiment for EACH of these segments separately.
"""
        
        return f"""You are a financial intelligence analyst searching X/Twitter for {ticker}.

{business_lines_hint}

SEARCH X/TWITTER for discussions about {ticker} and provide deep analysis.

Respond with valid JSON:
{{
    "ticker": "{ticker}",
    "overall_sentiment": "bullish" or "bearish" or "mixed",
    "sentiment_score": 0.0 to 1.0,
    
    "business_segments": [
        {{
            "segment": "Name of business line",
            "sentiment": "bullish/bearish/neutral",
            "momentum": "increasing/stable/decreasing",
            "key_discussions": ["What people are saying about this segment"],
            "overlooked": true/false
        }}
    ],
    
    "narrative_shifts": [
        {{
            "old_narrative": "What people used to think",
            "new_narrative": "What's emerging now",
            "trigger": "What caused the shift"
        }}
    ],
    
    "contrarian_signals": {{
        "bull_case_dismissed": "What bulls say that bears dismiss",
        "bear_case_dismissed": "What bears say that bulls dismiss",
        "underappreciated_catalyst": "Catalyst most are ignoring"
    }},
    
    "influential_voices": [
        {{
            "account_type": "analyst/insider/influencer/institutional",
            "stance": "bullish/bearish",
            "key_point": "Their main argument"
        }}
    ],
    
    "upcoming_catalysts_discussed": [
        {{
            "catalyst": "Event name",
            "expected_date": "When",
            "sentiment_into_event": "bullish/bearish/uncertain"
        }}
    ],
    
    "social_momentum": {{
        "volume_trend": "increasing/stable/decreasing",
        "engagement_quality": "high/medium/low",
        "institutional_vs_retail": "institutional-heavy/retail-heavy/balanced"
    }}
}}

IMPORTANT:
- Do NOT give a surface-level "priced in" take
- Identify which business segments are being overlooked
- Find the contrarian angles - what is the crowd missing?
- Report sentiment BY segment, not just overall

JSON only, no other text."""

    def _build_deep_fundamentals_prompt(self, ticker: str) -> str:
        """
        Build prompt for deep web/news business intelligence.
        
        This prompt asks Grok to:
        1. Map ALL business lines and revenue streams
        2. Identify TAM for each segment
        3. Find upcoming catalysts
        4. Surface what's being valued at zero
        """
        sector = TICKER_TO_SECTOR.get(ticker, "TECH")
        
        business_lines_hint = ""
        if ticker in PLATFORM_BUSINESS_LINES:
            lines = PLATFORM_BUSINESS_LINES[ticker]
            business_lines_hint = f"""
Known business lines for {ticker} (verify and expand):
{chr(10).join(f'- {line}' for line in lines)}
"""
        
        return f"""You are a financial analyst researching {ticker} using web and news sources.

{business_lines_hint}

SEARCH WEB AND NEWS for comprehensive business intelligence on {ticker}.

Respond with valid JSON:
{{
    "ticker": "{ticker}",
    "company_type": "platform/single_product/conglomerate",
    
    "business_lines": [
        {{
            "segment": "Business line name",
            "description": "What it does",
            "revenue_contribution": "% of total or 'emerging'",
            "growth_rate": "YoY % or estimate",
            "tam_size": "Total addressable market estimate",
            "competitive_position": "leader/challenger/emerging",
            "current_valuation_treatment": "priced_in/undervalued/zero"
        }}
    ],
    
    "emerging_businesses": [
        {{
            "segment": "Business line not yet material",
            "potential_tam": "TAM if successful",
            "milestone_to_watch": "What proves this out",
            "timeline": "When we'll know more"
        }}
    ],
    
    "upcoming_catalysts": [
        {{
            "event": "Catalyst name",
            "date": "Expected date",
            "impact_potential": "high/medium/low",
            "affected_segments": ["Which business lines"]
        }}
    ],
    
    "regulatory_landscape": {{
        "key_risks": ["Regulatory risks"],
        "potential_tailwinds": ["Regulatory benefits"],
        "upcoming_decisions": ["Pending regulatory events"]
    }},
    
    "competitive_dynamics": {{
        "main_competitors": ["Key competitors"],
        "competitive_moat": "Source of competitive advantage",
        "disruption_risk": "What could disrupt them"
    }},
    
    "analyst_consensus": {{
        "average_rating": "buy/hold/sell",
        "average_target": price,
        "bull_case_target": highest target,
        "bear_case_target": lowest target,
        "key_debate": "What analysts disagree about"
    }},
    
    "what_market_may_be_missing": {{
        "undervalued_segment": "Business line market undervalues",
        "ignored_catalyst": "Catalyst not being discussed",
        "contrarian_thesis": "Non-consensus view worth considering"
    }}
}}

IMPORTANT:
- Map ALL business lines, not just the obvious one
- Identify segments being valued at ZERO by the market
- Find catalysts for the next 6-12 months
- Be specific about TAM sizes

JSON only, no other text."""

    def _build_simple_sentiment_prompt(self, tickers: List[str]) -> str:
        """Build a simpler prompt for quick sentiment scan (backwards compatible)."""
        ticker_str = ", ".join(tickers)
        
        return f"""Analyze market sentiment for these stocks: {ticker_str}

Respond ONLY with valid JSON in this exact format:
{{
    "tickers": [
        {{
            "symbol": "TICKER",
            "sentiment": "bullish" or "bearish" or "neutral",
            "score": 0.5,
            "summary": "One sentence about outlook"
        }}
    ]
}}

Include one entry per ticker. No other text, just the JSON."""

    def _build_simple_outlook_prompt(self) -> str:
        """Build a simpler market outlook prompt."""
        return """Analyze current US stock market outlook.

Respond ONLY with valid JSON:
{
    "outlook": "bullish" or "bearish" or "neutral",
    "confidence": 0.5,
    "summary": "Brief market outlook",
    "key_factors": ["factor 1", "factor 2"]
}

No other text, just the JSON."""

    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _call_grok(
        self,
        system_prompt: str,
        user_message: str,
        use_search: bool = True,
        sources: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the Grok API with enhanced error handling.
        
        Args:
            system_prompt: System instructions
            user_message: User query
            use_search: Whether to enable live search
            sources: Optional list of sources ["x", "web", "news"]
                    If None, uses all sources
        """
        self.last_error = None
        self.last_raw_response = None
        
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        }
        
        if use_search:
            search_params = {
                "mode": "on",  # Force search for deep intel
                "return_citations": True,
            }
            # Add source filter if specified
            if sources:
                search_params["sources"] = sources
            request_body["search_parameters"] = search_params
        
        try:
            source_str = ", ".join(sources) if sources else "all"
            logger.info(f"Calling Grok API (sources: {source_str})...")
            
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                
                logger.info(f"Grok API response status: {response.status_code}")
                
                if response.status_code != 200:
                    self.last_error = f"API error {response.status_code}: {response.text[:200]}"
                    logger.error(self.last_error)
                    return None
                
                data = response.json()
                
                if "choices" not in data or not data["choices"]:
                    self.last_error = "No choices in API response"
                    logger.error(self.last_error)
                    return None
                
                content = data["choices"][0].get("message", {}).get("content", "")
                
                if not content:
                    self.last_error = "Empty content in API response"
                    logger.error(self.last_error)
                    return None
                
                self.last_raw_response = content[:500]
                logger.debug(f"Raw response: {content[:200]}...")
                
                # Parse JSON from response
                parsed = self._extract_json_from_response(content)
                
                if parsed is None:
                    logger.warning("Could not parse JSON from response")
                    return {"_parse_failed": True, "_raw": content}
                
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
        """Extract JSON from Grok's response."""
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
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find array
        array_patterns = [r'\[.*\]']
        for pattern in array_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    arr = json.loads(match.group(0))
                    return {"items": arr}
                except json.JSONDecodeError:
                    continue
        
        return None
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(self, source_type: str, primary_asset: str, summary: str) -> str:
        """Generate deduplication hash."""
        normalized = f"{source_type}:{primary_asset}:{summary.lower().strip()[:100]}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _create_fallback_signal(
        self,
        source_type: str,
        summary: str,
        tickers: List[str] = None
    ) -> GrokSignal:
        """Create a fallback signal when JSON parsing fails."""
        now = datetime.now(timezone.utc)
        signal_id = str(uuid.uuid4())
        
        primary_ticker = tickers[0] if tickers else "MARKET"
        dedup_hash = self._generate_dedup_hash(source_type, primary_ticker, summary)
        
        return GrokSignal(
            signal_id=signal_id,
            dedup_hash=dedup_hash,
            category="sentiment",
            source_type=source_type,
            asset_scope={
                "tickers": tickers or [],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            summary=summary[:200],
            raw_value={
                "type": "index",
                "value": 0.5,
                "unit": "sentiment",
                "prior_value": None,
                "change": None,
                "change_period": None,
            },
            evidence=[{
                "source": source_type,
                "source_tier": "social",
                "excerpt": "Grok analysis (unparsed format)",
                "timestamp_utc": now.isoformat(),
            }],
            confidence=0.30,
            confidence_factors={
                "source_base": 0.30,
                "recency_factor": 1.0,
                "corroboration_factor": 1.0,
            },
            directional_bias="unclear",
            time_horizon="weeks",
            novelty="new",
            staleness_policy={
                "max_age_seconds": 3600,
                "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
            },
            uncertainties=["Response format was not parseable as expected JSON"],
            timestamp_utc=now.isoformat(),
            forward_horizon="short-term",
        )
    
    def _parse_deep_social_to_signal(
        self,
        response: Dict[str, Any],
        ticker: str,
    ) -> Optional[GrokSignal]:
        """Parse deep social analysis into a signal."""
        now = datetime.now(timezone.utc)
        
        if response.get("_parse_failed"):
            raw_content = response.get("_raw", "")
            if raw_content:
                return self._create_fallback_signal("grok_x_deep", raw_content[:200], [ticker])
            return None
        
        try:
            overall_sentiment = response.get("overall_sentiment", "mixed")
            sentiment_score = float(response.get("sentiment_score", 0.5))
            
            # Build comprehensive summary
            business_segments = response.get("business_segments", [])
            contrarian = response.get("contrarian_signals", {})
            catalysts = response.get("upcoming_catalysts_discussed", [])
            
            # Create segment summary
            segment_summaries = []
            for seg in business_segments[:3]:
                seg_name = seg.get("segment", "Unknown")
                seg_sentiment = seg.get("sentiment", "neutral")
                overlooked = seg.get("overlooked", False)
                overlooked_str = " (OVERLOOKED)" if overlooked else ""
                segment_summaries.append(f"{seg_name}: {seg_sentiment}{overlooked_str}")
            
            segment_text = "; ".join(segment_summaries) if segment_summaries else "No segment breakdown"
            
            # Build summary
            summary = f"{ticker}: {overall_sentiment} (score: {sentiment_score:.2f}). Segments: {segment_text}"
            
            if contrarian.get("underappreciated_catalyst"):
                summary += f" | Underappreciated: {contrarian['underappreciated_catalyst'][:50]}"
            
            signal_id = str(uuid.uuid4())
            dedup_hash = self._generate_dedup_hash("grok_x_deep", ticker, summary)
            
            # Determine directional bias
            if overall_sentiment.lower() in ["bullish", "positive"]:
                directional_bias = "positive"
            elif overall_sentiment.lower() in ["bearish", "negative"]:
                directional_bias = "negative"
            else:
                directional_bias = "mixed"
            
            sector = TICKER_TO_SECTOR.get(ticker, "")
            
            return GrokSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="sentiment",
                source_type="grok_x_deep",
                asset_scope={
                    "tickers": [ticker],
                    "sectors": [sector],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                },
                summary=summary[:300],
                raw_value={
                    "type": "index",
                    "value": sentiment_score,
                    "unit": "sentiment_score",
                    "prior_value": None,
                    "change": None,
                    "change_period": None,
                },
                evidence=[{
                    "source": "grok_x_deep_search",
                    "source_tier": "social",
                    "excerpt": f"Deep X analysis with {len(business_segments)} segments analyzed",
                    "timestamp_utc": now.isoformat(),
                }],
                confidence=min(0.65 + (sentiment_score * 0.15), 0.85),
                confidence_factors={
                    "source_base": 0.65,
                    "segment_coverage": len(business_segments) / 5,
                    "contrarian_depth": 1.0 if contrarian else 0.8,
                },
                directional_bias=directional_bias,
                time_horizon="weeks",
                novelty="new",
                staleness_policy={
                    "max_age_seconds": 3600,
                    "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
                },
                uncertainties=[
                    seg.get("segment") for seg in business_segments 
                    if seg.get("overlooked", False)
                ][:3],
                timestamp_utc=now.isoformat(),
                forward_horizon="weeks",
                catalyst_date=catalysts[0].get("expected_date") if catalysts else None,
                business_lines=[
                    {
                        "segment": seg.get("segment"),
                        "sentiment": seg.get("sentiment"),
                        "momentum": seg.get("momentum"),
                        "overlooked": seg.get("overlooked", False),
                    }
                    for seg in business_segments
                ],
                contrarian_signals=contrarian,
            )
            
        except Exception as e:
            logger.error(f"Error parsing deep social response: {e}")
            return None
    
    def _parse_deep_fundamentals_to_signal(
        self,
        response: Dict[str, Any],
        ticker: str,
    ) -> Optional[GrokSignal]:
        """Parse deep fundamentals analysis into a signal."""
        now = datetime.now(timezone.utc)
        
        if response.get("_parse_failed"):
            raw_content = response.get("_raw", "")
            if raw_content:
                return self._create_fallback_signal("grok_web_deep", raw_content[:200], [ticker])
            return None
        
        try:
            company_type = response.get("company_type", "single_product")
            business_lines = response.get("business_lines", [])
            emerging = response.get("emerging_businesses", [])
            catalysts = response.get("upcoming_catalysts", [])
            missing = response.get("what_market_may_be_missing", {})
            
            # Count undervalued segments
            undervalued_count = sum(
                1 for bl in business_lines 
                if bl.get("current_valuation_treatment") in ["undervalued", "zero"]
            )
            
            # Build summary
            total_segments = len(business_lines) + len(emerging)
            summary = f"{ticker}: {company_type} company with {total_segments} business lines. "
            
            if undervalued_count > 0:
                summary += f"{undervalued_count} segments potentially undervalued/zero. "
            
            if missing.get("undervalued_segment"):
                summary += f"Key opportunity: {missing['undervalued_segment'][:50]}. "
            
            if catalysts:
                next_catalyst = catalysts[0]
                summary += f"Next catalyst: {next_catalyst.get('event', 'Unknown')} ({next_catalyst.get('date', 'TBD')})"
            
            signal_id = str(uuid.uuid4())
            dedup_hash = self._generate_dedup_hash("grok_web_deep", ticker, summary)
            
            # Determine directional bias based on undervalued segments
            if undervalued_count >= 2:
                directional_bias = "positive"
            elif undervalued_count == 1:
                directional_bias = "mixed"
            else:
                directional_bias = "unclear"
            
            sector = TICKER_TO_SECTOR.get(ticker, "")
            
            return GrokSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="business_intel",
                source_type="grok_web_deep",
                asset_scope={
                    "tickers": [ticker],
                    "sectors": [sector],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                },
                summary=summary[:300],
                raw_value={
                    "type": "index",
                    "value": undervalued_count / max(len(business_lines), 1),
                    "unit": "undervalued_ratio",
                    "prior_value": None,
                    "change": None,
                    "change_period": None,
                },
                evidence=[{
                    "source": "grok_web_deep_search",
                    "source_tier": "tier1",
                    "excerpt": f"Deep fundamentals: {len(business_lines)} core + {len(emerging)} emerging segments",
                    "timestamp_utc": now.isoformat(),
                }],
                confidence=min(0.70 + (len(business_lines) * 0.03), 0.90),
                confidence_factors={
                    "source_base": 0.70,
                    "business_line_coverage": len(business_lines) / 5,
                    "catalyst_coverage": len(catalysts) / 3,
                },
                directional_bias=directional_bias,
                time_horizon="months",
                novelty="new",
                staleness_policy={
                    "max_age_seconds": 86400,  # 24 hours for fundamentals
                    "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
                },
                uncertainties=[
                    bl.get("segment") for bl in business_lines
                    if bl.get("current_valuation_treatment") == "zero"
                ][:3],
                timestamp_utc=now.isoformat(),
                forward_horizon="months",
                catalyst_date=catalysts[0].get("date") if catalysts else None,
                business_lines=[
                    {
                        "segment": bl.get("segment"),
                        "tam": bl.get("tam_size"),
                        "growth": bl.get("growth_rate"),
                        "valuation": bl.get("current_valuation_treatment"),
                    }
                    for bl in business_lines
                ] + [
                    {
                        "segment": em.get("segment"),
                        "tam": em.get("potential_tam"),
                        "growth": "emerging",
                        "valuation": "zero",
                    }
                    for em in emerging
                ],
                contrarian_signals={
                    "undervalued_segment": missing.get("undervalued_segment"),
                    "ignored_catalyst": missing.get("ignored_catalyst"),
                    "contrarian_thesis": missing.get("contrarian_thesis"),
                },
            )
            
        except Exception as e:
            logger.error(f"Error parsing deep fundamentals response: {e}")
            return None
    
    def _parse_sentiment_to_signals(
        self,
        response: Dict[str, Any],
        tickers: List[str],
    ) -> List[GrokSignal]:
        """Parse Grok sentiment response into signals (backwards compatible)."""
        signals = []
        now = datetime.now(timezone.utc)
        
        if response.get("_parse_failed"):
            raw_content = response.get("_raw", "")
            if raw_content:
                logger.info("Creating fallback signal from raw content")
                summary = raw_content[:150].replace("\n", " ")
                signals.append(self._create_fallback_signal("grok_x", summary, tickers))
            return signals
        
        tickers_data = response.get("tickers", [])
        
        if not tickers_data:
            logger.warning(f"No 'tickers' key in response. Keys found: {list(response.keys())}")
            if "items" in response:
                tickers_data = response["items"]
            elif isinstance(response, list):
                tickers_data = response
        
        for ticker_data in tickers_data:
            try:
                symbol = ticker_data.get("symbol") or ticker_data.get("ticker", "UNKNOWN")
                sentiment = ticker_data.get("sentiment") or ticker_data.get("forward_sentiment", "neutral")
                score = float(ticker_data.get("score") or ticker_data.get("sentiment_score", 0.5))
                summary_text = ticker_data.get("summary") or ticker_data.get("key_expectations", ["No details"])[0]
                
                signal_id = str(uuid.uuid4())
                full_summary = f"{symbol}: {sentiment}. {summary_text}"
                dedup_hash = self._generate_dedup_hash("grok_x", symbol, full_summary)
                
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                sector = TICKER_TO_SECTOR.get(symbol, "")
                
                if sentiment.lower() in ["bullish", "positive"]:
                    directional_bias = "positive"
                elif sentiment.lower() in ["bearish", "negative"]:
                    directional_bias = "negative"
                elif sentiment.lower() == "mixed":
                    directional_bias = "mixed"
                else:
                    directional_bias = "unclear"
                
                signal = GrokSignal(
                    signal_id=signal_id,
                    dedup_hash=dedup_hash,
                    category="sentiment",
                    source_type="grok_x",
                    asset_scope={
                        "tickers": [symbol],
                        "sectors": [sector],
                        "macro_regions": ["US"],
                        "asset_classes": ["EQUITY"],
                    },
                    summary=full_summary[:200],
                    raw_value={
                        "type": "index",
                        "value": score,
                        "unit": "sentiment_score",
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=[{
                        "source": "grok_x_search",
                        "source_tier": "social",
                        "excerpt": summary_text[:100],
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=min(0.55 * (1 + score), 0.80),
                    confidence_factors={
                        "source_base": 0.55,
                        "recency_factor": 1.0,
                        "corroboration_factor": 1.0,
                    },
                    directional_bias=directional_bias,
                    time_horizon="weeks",
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 3600,
                        "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
                    },
                    uncertainties=[],
                    timestamp_utc=now.isoformat(),
                    forward_horizon="weeks",
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error parsing ticker sentiment: {e}")
                continue
        
        return signals
    
    def _parse_outlook_to_signals(self, response: Dict[str, Any]) -> List[GrokSignal]:
        """Parse market outlook response into signals."""
        signals = []
        now = datetime.now(timezone.utc)
        
        if response.get("_parse_failed"):
            return signals
        
        try:
            outlook = response.get("outlook", "neutral")
            confidence = float(response.get("confidence", 0.5))
            summary = response.get("summary", "No summary available")
            key_factors = response.get("key_factors", [])
            
            signal_id = str(uuid.uuid4())
            full_summary = f"Market outlook: {outlook}. {summary}"
            dedup_hash = self._generate_dedup_hash("grok_web", "MARKET", full_summary)
            
            if outlook.lower() in ["bullish", "positive"]:
                directional_bias = "positive"
            elif outlook.lower() in ["bearish", "negative"]:
                directional_bias = "negative"
            else:
                directional_bias = "mixed"
            
            signal = GrokSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="sentiment",
                source_type="grok_web",
                asset_scope={
                    "tickers": [],
                    "sectors": [],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                },
                summary=full_summary[:200],
                raw_value={
                    "type": "index",
                    "value": confidence,
                    "unit": "outlook_confidence",
                    "prior_value": None,
                    "change": None,
                    "change_period": None,
                },
                evidence=[{
                    "source": "grok_web_search",
                    "source_tier": "tier2",
                    "excerpt": "; ".join(key_factors[:3]) if key_factors else summary[:100],
                    "timestamp_utc": now.isoformat(),
                }],
                confidence=min(0.55 * (1 + confidence), 0.80),
                confidence_factors={
                    "source_base": 0.55,
                    "recency_factor": 1.0,
                    "corroboration_factor": 1.0,
                },
                directional_bias=directional_bias,
                time_horizon="weeks",
                novelty="new",
                staleness_policy={
                    "max_age_seconds": 14400,
                    "stale_after_utc": (now + timedelta(hours=4)).isoformat(),
                },
                uncertainties=response.get("risks_to_watch", [])[:3],
                timestamp_utc=now.isoformat(),
                forward_horizon="end of quarter",
            )
            
            signals.append(signal)
            logger.info(f"Created market outlook signal: {outlook}")
            
        except Exception as e:
            logger.error(f"Error parsing market outlook: {e}")
        
        return signals
    
    # =========================================================================
    # PUBLIC INTERFACE - DEEP INTELLIGENCE METHODS
    # =========================================================================
    
    async def scan_ticker_social(self, ticker: str) -> Optional[GrokSignal]:
        """
        Deep X/Twitter analysis for a single ticker.
        
        Returns sentiment BY business segment, narrative shifts,
        contrarian signals, and social momentum.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return None
        
        logger.info(f"Deep social scan for {ticker}")
        
        system_prompt = self._build_deep_social_prompt(ticker)
        user_message = f"Provide deep X/Twitter analysis for {ticker}"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
            sources=["x"],  # X/Twitter only
        )
        
        if not response:
            logger.warning(f"No response from Grok social scan. Error: {self.last_error}")
            return None
        
        signal = self._parse_deep_social_to_signal(response, ticker)
        
        if signal:
            logger.info(f"Generated deep social signal for {ticker}: {signal.directional_bias}")
        
        return signal
    
    async def scan_ticker_fundamentals(self, ticker: str) -> Optional[GrokSignal]:
        """
        Deep web/news analysis for a single ticker.
        
        Returns business line mapping, TAM analysis, catalysts,
        and what the market may be missing.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return None
        
        logger.info(f"Deep fundamentals scan for {ticker}")
        
        system_prompt = self._build_deep_fundamentals_prompt(ticker)
        user_message = f"Provide comprehensive business intelligence for {ticker}"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
            sources=["web", "news"],  # Web and news only
        )
        
        if not response:
            logger.warning(f"No response from Grok fundamentals scan. Error: {self.last_error}")
            return None
        
        signal = self._parse_deep_fundamentals_to_signal(response, ticker)
        
        if signal:
            logger.info(f"Generated deep fundamentals signal for {ticker}: {len(signal.business_lines or [])} business lines")
        
        return signal
    
    async def scan_ticker_deep(self, ticker: str) -> List[GrokSignal]:
        """
        Comprehensive deep scan combining social + fundamentals.
        
        This is the primary method for full business intelligence
        on any ticker, especially platform companies.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return []
        
        logger.info(f"=== DEEP SCAN: {ticker} ===")
        signals = []
        
        # Social analysis (X/Twitter)
        social_signal = await self.scan_ticker_social(ticker)
        if social_signal:
            signals.append(social_signal)
        
        # Fundamentals analysis (Web/News)
        fundamentals_signal = await self.scan_ticker_fundamentals(ticker)
        if fundamentals_signal:
            signals.append(fundamentals_signal)
        
        logger.info(f"Deep scan complete for {ticker}: {len(signals)} signals")
        return signals
    
    # =========================================================================
    # PUBLIC INTERFACE - BACKWARDS COMPATIBLE METHODS
    # =========================================================================
    
    async def scan_sentiment(
        self, 
        tickers: List[str],
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
    ) -> List[GrokSignal]:
        """
        Quick sentiment scan for multiple tickers (backwards compatible).
        
        For deep analysis of a single ticker, use scan_ticker_deep() instead.
        """
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return []
        
        logger.info(f"Scanning sentiment for {tickers}")
        
        system_prompt = self._build_simple_sentiment_prompt(tickers)
        user_message = f"Analyze sentiment for: {', '.join(tickers)}"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning(f"No response from Grok. Last error: {self.last_error}")
            return []
        
        signals = self._parse_sentiment_to_signals(response, tickers)
        logger.info(f"Generated {len(signals)} sentiment signals")
        return signals
    
    async def scan_market_overview(self) -> List[GrokSignal]:
        """Scan for overall market outlook."""
        if not self.is_configured:
            self.last_error = "XAI_API_KEY not configured"
            logger.warning(self.last_error)
            return []
        
        logger.info("Scanning market overview")
        
        system_prompt = self._build_simple_outlook_prompt()
        user_message = "What is the current US stock market outlook?"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning(f"No response from Grok. Last error: {self.last_error}")
            return []
        
        signals = self._parse_outlook_to_signals(response)
        logger.info(f"Generated {len(signals)} market overview signals")
        return signals
    
    async def scan_catalysts(self, ticker: str) -> List[GrokSignal]:
        """Scan for upcoming catalysts for a specific ticker."""
        if not self.is_configured:
            return []
        
        logger.info(f"Scanning catalysts for {ticker}")
        
        system_prompt = f"""Find upcoming events and catalysts for {ticker}.
Respond ONLY with valid JSON:
{{
    "catalysts": [
        {{"event": "...", "date": "...", "impact": "high/medium/low"}}
    ]
}}"""
        
        user_message = f"What are upcoming catalysts for {ticker}?"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            return []
        
        signals = []
        now = datetime.now(timezone.utc)
        
        catalysts = response.get("catalysts", [])
        for catalyst in catalysts[:5]:
            try:
                event = catalyst.get("event", "Unknown event")
                date = catalyst.get("date", "TBD")
                impact = catalyst.get("impact", "medium")
                
                signal_id = str(uuid.uuid4())
                summary = f"{ticker}: {event} ({date})"
                dedup_hash = self._generate_dedup_hash("grok_web", ticker, summary)
                
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                signal = GrokSignal(
                    signal_id=signal_id,
                    dedup_hash=dedup_hash,
                    category="event",
                    source_type="grok_web",
                    asset_scope={
                        "tickers": [ticker],
                        "sectors": [TICKER_TO_SECTOR.get(ticker, "")],
                        "macro_regions": ["US"],
                        "asset_classes": ["EQUITY"],
                    },
                    summary=summary[:200],
                    raw_value={
                        "type": "null",
                        "value": None,
                        "unit": None,
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=[{
                        "source": "grok_web_search",
                        "source_tier": "tier2",
                        "excerpt": event,
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=0.60 if impact == "high" else 0.45,
                    confidence_factors={
                        "source_base": 0.55,
                        "recency_factor": 1.0,
                        "corroboration_factor": 1.0,
                    },
                    directional_bias="unclear",
                    time_horizon="weeks",
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 86400,
                        "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
                    },
                    uncertainties=[f"Date: {date}"],
                    timestamp_utc=now.isoformat(),
                    forward_horizon="weeks",
                    catalyst_date=date,
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error parsing catalyst: {e}")
                continue
        
        return signals

    # ---------------------------------------------------------------------
    # Debate Layer
    # ---------------------------------------------------------------------
    async def debate(
        self,
        *,
        scan_cycle_id: str,
        round_num: int,
        own_thesis: Dict[str, Any],
        other_theses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Participate in committee debate (narrative momentum role).

        Grok is expected to leverage its X/Twitter strengths, but in debate mode
        it MUST NOT fetch new info; it reacts to the provided theses only.
        """

        if not self.is_configured:
            return {
                "speaker": "grok",
                "round": round_num,
                "message": "Grok not configured",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
            }

        system = """You are participating in an investment committee debate.

You are Grok in the role of Narrative Momentum Analyst.

RULES:
1) Stay within role: narrative shifts, attention, momentum, crowd behavior on X.
2) Do NOT browse or search. Do NOT fetch new posts. Only react to provided theses.
3) Do NOT invent signals. You may defend or revise your prior vote.

Output ONLY JSON in this schema:
{
  "message": "2-6 sentences",
  "agreements": ["..."],
  "disagreements": ["..."],
  "changed_mind": true|false,
  "vote": {"action": "BUY"|"SELL"|"HOLD", "ticker": "..."|null, "side": "BUY"|"SELL"|null, "confidence": 0.0-1.0}
}
"""

        user = {
            "scan_cycle_id": scan_cycle_id,
            "round": round_num,
            "own_thesis": own_thesis,
            "other_theses": other_theses,
        }

        parsed = await self._call_grok(system_prompt=system, user_message=json.dumps(user, ensure_ascii=False), use_search=False)
        if not parsed:
            return {
                "speaker": "grok",
                "round": round_num,
                "message": "Grok debate failed",
                "vote": {"action": "HOLD", "ticker": None, "side": None, "confidence": 0.0},
                "changed_mind": False,
            }

        parsed.update({"speaker": "grok", "round": round_num})
        return parsed


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_grok_sentiment(tickers: List[str]) -> List[Dict[str, Any]]:
    """Convenience function to run a Grok sentiment scan."""
    scanner = GrokScanner()
    signals = await scanner.scan_sentiment(tickers)
    return [s.to_dict() for s in signals]


async def scan_ticker_deep(ticker: str) -> List[Dict[str, Any]]:
    """Convenience function to run a deep scan on a single ticker."""
    scanner = GrokScanner()
    signals = await scanner.scan_ticker_deep(ticker)
    return [s.to_dict() for s in signals]


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = GrokScanner()
        
        print("\n" + "="*60)
        print("GROK SCANNER v3.0.0 - DEEP INTELLIGENCE TEST")
        print("="*60)
        
        if scanner.is_configured:
            # Test deep scan on TSLA
            print("\n--- Testing DEEP SCAN on TSLA ---")
            signals = await scanner.scan_ticker_deep("TSLA")
            
            for sig in signals:
                print(f"\nSource: {sig.source_type}")
                print(f"Summary: {sig.summary[:100]}...")
                print(f"Bias: {sig.directional_bias}")
                
                if sig.business_lines:
                    print(f"Business Lines ({len(sig.business_lines)}):")
                    for bl in sig.business_lines[:3]:
                        print(f"  - {bl.get('segment')}: {bl.get('sentiment', bl.get('valuation', 'N/A'))}")
                
                if sig.contrarian_signals:
                    print(f"Contrarian: {sig.contrarian_signals.get('underappreciated_catalyst', 'N/A')[:50]}")
            
            if scanner.last_error:
                print(f"\nLast error: {scanner.last_error}")
        else:
            print("\nXAI_API_KEY not set - skipping live test")
    
    asyncio.run(test())
