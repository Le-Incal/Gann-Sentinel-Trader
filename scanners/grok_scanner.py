"""
Gann Sentinel Trader - Grok Scanner
Forward-looking sentiment and news signal extraction via xAI Grok API.

This scanner uses the temporal awareness framework to ensure all queries
focus on forward-looking sentiment, upcoming catalysts, and future outlooks
rather than historical events.

Version: 2.0.0 (Temporal Awareness Update)
Last Updated: January 2026
"""

import os
import uuid
import hashlib
import logging
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


# Sector mappings for asset scope
TICKER_TO_SECTOR = {
    # Tech
    "NVDA": "TECH", "AMD": "TECH", "SMCI": "TECH", "AAPL": "TECH",
    "MSFT": "TECH", "GOOGL": "TECH", "META": "TECH", "AMZN": "TECH",
    # Financials
    "JPM": "FINANCIALS", "GS": "FINANCIALS", "MS": "FINANCIALS",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "OXY": "ENERGY",
    # Space/Defense
    "RKLB": "AEROSPACE", "LMT": "AEROSPACE", "RTX": "AEROSPACE",
    # Crypto-adjacent
    "COIN": "CRYPTO", "MSTR": "CRYPTO",
    # EVs
    "TSLA": "AUTO",
    # Index ETFs
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
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
    source_type: str  # grok_x or grok_web
    
    # Asset scope
    asset_scope: Dict[str, List[str]]
    
    # Signal content
    summary: str
    raw_value: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    
    # Scoring and metadata
    confidence: float
    confidence_factors: Dict[str, float]
    directional_bias: str
    time_horizon: str
    novelty: str
    
    # Staleness
    staleness_policy: Dict[str, Any]
    
    # Uncertainties
    uncertainties: List[str]
    
    # Timestamps
    timestamp_utc: str
    
    # Forward-looking context
    forward_horizon: Optional[str] = None
    catalyst_date: Optional[str] = None
    
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
        }


# =============================================================================
# GROK SCANNER
# =============================================================================

class GrokScanner:
    """
    Scanner for sentiment and news signals via xAI Grok API.
    
    Uses forward-looking queries to capture:
    - Sentiment about future expectations (not past events)
    - Upcoming catalysts and events
    - Forward outlook and forecasts
    - Narrative shifts that may impact future price action
    
    All queries are temporally aware - they focus on what's COMING,
    not what has HAPPENED.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Grok scanner."""
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        
        if not self.api_key:
            logger.warning("XAI_API_KEY not set - Grok scanning disabled")
        
        self.base_url = "https://api.x.ai/v1"
        self.model = GrokModel.GROK_FAST.value
        
        # Initialize temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for deduplication
        self._seen_signals: Dict[str, datetime] = {}
        
        # Log temporal context
        self.temporal_context.log_context()
        logger.info("GrokScanner initialized with forward-looking temporal awareness")
    
    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return bool(self.api_key)
    
    # =========================================================================
    # TEMPORAL QUERY GENERATION
    # =========================================================================
    
    def _build_forward_sentiment_prompt(
        self, 
        tickers: List[str],
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM
    ) -> str:
        """
        Build a forward-looking sentiment analysis prompt.
        
        Instead of: "What is sentiment on NVDA?"
        We ask: "What are expectations for NVDA over the next month?"
        
        Args:
            tickers: List of tickers to analyze
            horizon: Time horizon for the outlook
            
        Returns:
            System prompt for forward-looking analysis
        """
        horizon_desc = horizon.label
        end_date = self.temporal_context.format_date(
            self.temporal_context.now + timedelta(days=horizon.value)
        )
        
        ticker_str = ", ".join(tickers)
        
        return f"""You are a forward-looking market sentiment analyst. Your job is to analyze 
FUTURE expectations and outlook, not historical events.

CURRENT DATE: {self.temporal_context.format_date(self.temporal_context.now, '%B %d, %Y')}
ANALYSIS HORIZON: {horizon_desc} (through {end_date})

For the following tickers: {ticker_str}

Analyze:
1. FORWARD SENTIMENT: What are market participants expecting over the next {horizon.value} days?
2. UPCOMING CATALYSTS: What events/announcements could move these stocks?
3. NARRATIVE TRAJECTORY: Is the narrative strengthening or weakening?
4. KEY DEBATES: What are bulls vs bears arguing about for the FUTURE?

Focus ONLY on forward-looking information. Ignore past earnings reports unless 
they contain forward guidance. Focus on:
- Analyst price targets and revisions
- Upcoming earnings/events
- Industry trends that will play out
- Sentiment about future prospects

Output format: JSON with structure:
{{
    "tickers": [
        {{
            "symbol": "NVDA",
            "forward_sentiment": "bullish|bearish|neutral|mixed",
            "sentiment_score": 0.0-1.0,
            "key_expectations": ["expectation 1", "expectation 2"],
            "upcoming_catalysts": [
                {{"event": "...", "expected_date": "YYYY-MM-DD", "potential_impact": "high|medium|low"}}
            ],
            "narrative_direction": "strengthening|weakening|stable",
            "bull_thesis": "...",
            "bear_thesis": "...",
            "confidence": 0.0-1.0
        }}
    ],
    "market_context": "brief overall market outlook",
    "analysis_horizon": "{horizon_desc}"
}}
"""
    
    def _build_market_outlook_prompt(self) -> str:
        """Build a prompt for overall market forward outlook."""
        now = self.temporal_context.now
        end_of_month = self.temporal_context.end_of_month
        end_of_quarter = self.temporal_context.end_of_quarter
        
        return f"""You are a forward-looking market analyst. Provide outlook for the NEXT period.

CURRENT DATE: {self.temporal_context.format_date(now, '%B %d, %Y')}

Analyze the forward outlook for:
1. THROUGH END OF MONTH ({self.temporal_context.format_date(end_of_month)})
2. THROUGH END OF QUARTER ({self.temporal_context.format_date(end_of_quarter)})

Focus on:
- Fed policy expectations and FOMC meeting outcomes
- Economic data releases coming up
- Earnings season expectations
- Geopolitical developments that could impact markets
- Sector rotation trends
- Risk sentiment trajectory

Do NOT focus on what has already happened. Focus on what market participants 
EXPECT to happen and what catalysts are upcoming.

Output format: JSON with structure:
{{
    "overall_outlook": "bullish|bearish|neutral|mixed",
    "confidence": 0.0-1.0,
    "key_themes": ["theme 1", "theme 2"],
    "upcoming_catalysts": [
        {{"event": "...", "expected_date": "YYYY-MM-DD", "potential_impact": "..."}}
    ],
    "sector_outlook": {{
        "TECH": "bullish|bearish|neutral",
        "FINANCIALS": "...",
        "ENERGY": "..."
    }},
    "risks_to_watch": ["risk 1", "risk 2"],
    "horizon_end_of_month": "brief outlook",
    "horizon_end_of_quarter": "brief outlook"
}}
"""
    
    def _build_catalyst_search_prompt(self, ticker: str) -> str:
        """Build a prompt for finding upcoming catalysts for a ticker."""
        now = self.temporal_context.now
        
        return f"""Find all upcoming catalysts for {ticker} over the next 90 days.

CURRENT DATE: {self.temporal_context.format_date(now, '%B %d, %Y')}
SEARCH WINDOW: Through {self.temporal_context.format_date(now + timedelta(days=90))}

Search for:
1. Earnings release dates
2. Investor days / conferences
3. Product launches
4. Regulatory decisions (FDA, FTC, etc.)
5. Contract announcements
6. Guidance updates
7. Industry events
8. Analyst days

For each catalyst found, provide:
- Event name
- Expected date (or date range)
- Potential impact on stock
- What the market expects

Output format: JSON with structure:
{{
    "ticker": "{ticker}",
    "catalysts": [
        {{
            "event": "...",
            "expected_date": "YYYY-MM-DD",
            "date_confidence": "confirmed|estimated|rumored",
            "potential_impact": "high|medium|low",
            "market_expectation": "...",
            "source": "..."
        }}
    ],
    "near_term_focus": "what's the key event in next 30 days",
    "medium_term_focus": "what's the key event in 30-90 days"
}}
"""
    
    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _call_grok(
        self,
        system_prompt: str,
        user_message: str,
        use_search: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the Grok API with the given prompts.
        
        Args:
            system_prompt: System prompt defining the task
            user_message: User message with specific query
            use_search: Whether to enable web/X search
            
        Returns:
            Parsed JSON response or None on error
        """
        if not self.is_configured:
            logger.warning("Grok not configured, skipping API call")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Build request with search tools if enabled
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,  # Lower for more consistent analysis
        }
        
        # Add search tools if requested
        if use_search:
            request_body["tools"] = [
                {"type": "web_search"},
                {"type": "x_search"},
            ]
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Try to parse as JSON
                    try:
                        # Clean up markdown code blocks if present
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0]
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0]
                        
                        return json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse Grok response as JSON")
                        return {"raw_content": content}
                else:
                    logger.error(f"Grok API error: {response.status_code}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error("Grok API timeout")
            return None
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return None
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(
        self, 
        source_type: str, 
        primary_asset: str,
        summary: str
    ) -> str:
        """Generate deduplication hash per Grok Spec v1.1.0."""
        normalized = f"{source_type}:{primary_asset}:{summary.lower().strip()[:100]}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _parse_sentiment_to_signals(
        self,
        response: Dict[str, Any],
        horizon: TimeHorizon,
    ) -> List[GrokSignal]:
        """
        Parse Grok sentiment response into standardized signals.
        
        Args:
            response: Parsed JSON from Grok
            horizon: Time horizon that was queried
            
        Returns:
            List of GrokSignal objects
        """
        signals = []
        now = self.temporal_context.now
        
        tickers_data = response.get("tickers", [])
        
        for ticker_data in tickers_data:
            try:
                symbol = ticker_data.get("symbol", "UNKNOWN")
                sentiment = ticker_data.get("forward_sentiment", "neutral")
                sentiment_score = float(ticker_data.get("sentiment_score", 0.5))
                confidence = float(ticker_data.get("confidence", 0.5))
                
                # Generate IDs
                signal_id = str(uuid.uuid4())
                summary = f"{symbol} forward sentiment is {sentiment}. " + \
                         ", ".join(ticker_data.get("key_expectations", [])[:2])
                dedup_hash = self._generate_dedup_hash("grok_x", symbol, summary)
                
                # Check dedup
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                # Build asset scope
                sector = TICKER_TO_SECTOR.get(symbol, "")
                asset_scope = {
                    "tickers": [symbol],
                    "sectors": [sector] if sector else [],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                }
                
                # Directional bias from sentiment
                if sentiment == "bullish":
                    directional_bias = "positive"
                elif sentiment == "bearish":
                    directional_bias = "negative"
                elif sentiment == "mixed":
                    directional_bias = "mixed"
                else:
                    directional_bias = "unclear"
                
                # Confidence factors
                confidence_factors = {
                    "source_base": 0.40,  # Social/sentiment source
                    "recency_factor": 1.0,  # Fresh query
                    "corroboration_factor": 1.0,  # Single source
                }
                final_confidence = min(
                    confidence_factors["source_base"] * 
                    confidence_factors["recency_factor"] * 
                    confidence_factors["corroboration_factor"] *
                    (1 + confidence),  # Boost by Grok's confidence
                    1.0
                )
                
                # Evidence from catalysts
                evidence = []
                for catalyst in ticker_data.get("upcoming_catalysts", [])[:3]:
                    evidence.append({
                        "source": "grok_x_search",
                        "source_tier": "social",
                        "excerpt": f"{catalyst.get('event', 'Catalyst')} - {catalyst.get('potential_impact', 'medium')} impact",
                        "timestamp_utc": now.isoformat(),
                    })
                
                if not evidence:
                    evidence.append({
                        "source": "grok_x_search",
                        "source_tier": "social",
                        "excerpt": f"Forward sentiment analysis for {symbol}",
                        "timestamp_utc": now.isoformat(),
                    })
                
                # Uncertainties
                uncertainties = []
                if confidence < 0.5:
                    uncertainties.append("Low confidence in sentiment reading")
                if not ticker_data.get("upcoming_catalysts"):
                    uncertainties.append("No specific catalysts identified")
                
                # Staleness - sentiment moves fast
                staleness_policy = {
                    "max_age_seconds": 3600,  # 1 hour
                    "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
                }
                
                signal = GrokSignal(
                    signal_id=signal_id,
                    dedup_hash=dedup_hash,
                    category="sentiment",
                    source_type="grok_x",
                    asset_scope=asset_scope,
                    summary=summary[:200],
                    raw_value={
                        "type": "index",
                        "value": sentiment_score,
                        "unit": "sentiment_score",
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=evidence,
                    confidence=final_confidence,
                    confidence_factors=confidence_factors,
                    directional_bias=directional_bias,
                    time_horizon=horizon.signal_horizon_label,
                    novelty="new",
                    staleness_policy=staleness_policy,
                    uncertainties=uncertainties,
                    timestamp_utc=now.isoformat(),
                    forward_horizon=horizon.label,
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error parsing ticker data: {e}")
                continue
        
        return signals
    
    def _parse_market_outlook_to_signals(
        self,
        response: Dict[str, Any],
    ) -> List[GrokSignal]:
        """Parse market outlook response into signals."""
        signals = []
        now = self.temporal_context.now
        
        try:
            outlook = response.get("overall_outlook", "neutral")
            confidence = float(response.get("confidence", 0.5))
            
            # Generate IDs
            signal_id = str(uuid.uuid4())
            summary = f"Market outlook: {outlook}. Key themes: " + \
                     ", ".join(response.get("key_themes", [])[:3])
            dedup_hash = self._generate_dedup_hash("grok_web", "SPY", summary)
            
            if dedup_hash in self._seen_signals:
                return signals
            self._seen_signals[dedup_hash] = now
            
            # Directional bias
            if outlook == "bullish":
                directional_bias = "positive"
            elif outlook == "bearish":
                directional_bias = "negative"
            else:
                directional_bias = "mixed"
            
            # Evidence from catalysts
            evidence = []
            for catalyst in response.get("upcoming_catalysts", [])[:3]:
                evidence.append({
                    "source": "grok_web_search",
                    "source_tier": "tier2",
                    "excerpt": f"{catalyst.get('event', 'Event')} on {catalyst.get('expected_date', 'TBD')}",
                    "timestamp_utc": now.isoformat(),
                })
            
            if not evidence:
                evidence.append({
                    "source": "grok_web_search",
                    "source_tier": "tier2",
                    "excerpt": "Market outlook analysis",
                    "timestamp_utc": now.isoformat(),
                })
            
            confidence_factors = {
                "source_base": 0.55,  # Web news tier-2
                "recency_factor": 1.0,
                "corroboration_factor": 1.0,
            }
            
            staleness_policy = {
                "max_age_seconds": 14400,  # 4 hours
                "stale_after_utc": (now + timedelta(hours=4)).isoformat(),
            }
            
            signal = GrokSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="sentiment",
                source_type="grok_web",
                asset_scope={
                    "tickers": ["SPY", "QQQ"],
                    "sectors": [],
                    "macro_regions": ["US"],
                    "asset_classes": ["EQUITY"],
                },
                summary=summary[:200],
                raw_value={
                    "type": "index",
                    "value": confidence,
                    "unit": "outlook_confidence",
                    "prior_value": None,
                    "change": None,
                    "change_period": None,
                },
                evidence=evidence,
                confidence=min(confidence_factors["source_base"] * (1 + confidence), 1.0),
                confidence_factors=confidence_factors,
                directional_bias=directional_bias,
                time_horizon="weeks",
                novelty="new",
                staleness_policy=staleness_policy,
                uncertainties=response.get("risks_to_watch", [])[:3],
                timestamp_utc=now.isoformat(),
                forward_horizon="end of quarter",
            )
            
            signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error parsing market outlook: {e}")
        
        return signals
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    async def scan_sentiment(
        self, 
        tickers: List[str],
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
    ) -> List[GrokSignal]:
        """
        Scan for forward-looking sentiment on specific tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            horizon: Time horizon for the outlook
            
        Returns:
            List of GrokSignal objects
        """
        if not self.is_configured:
            logger.warning("Grok not configured")
            return []
        
        logger.info(f"Scanning forward sentiment for {tickers} over {horizon.label}")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        # Build forward-looking prompt
        system_prompt = self._build_forward_sentiment_prompt(tickers, horizon)
        
        # Query focuses on expectations, not history
        user_message = self.query_builder.build_sentiment_query(
            " ".join(tickers), 
            horizon
        )
        
        # Call Grok with search enabled
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning("No response from Grok sentiment scan")
            return []
        
        # Parse to signals
        signals = self._parse_sentiment_to_signals(response, horizon)
        
        logger.info(f"Generated {len(signals)} sentiment signals")
        return signals
    
    async def scan_market_overview(self) -> List[GrokSignal]:
        """
        Scan for overall market forward outlook.
        
        Returns:
            List of GrokSignal objects with market overview
        """
        if not self.is_configured:
            logger.warning("Grok not configured")
            return []
        
        logger.info("Scanning market forward outlook")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        # Build forward-looking prompt
        system_prompt = self._build_market_outlook_prompt()
        user_message = self.query_builder.build_market_outlook_query()
        
        # Call Grok
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            logger.warning("No response from Grok market overview")
            return []
        
        # Parse to signals
        signals = self._parse_market_outlook_to_signals(response)
        
        logger.info(f"Generated {len(signals)} market overview signals")
        return signals
    
    async def scan_catalysts(self, ticker: str) -> List[GrokSignal]:
        """
        Scan for upcoming catalysts for a specific ticker.
        
        Args:
            ticker: Ticker symbol to search catalysts for
            
        Returns:
            List of GrokSignal objects for upcoming catalysts
        """
        if not self.is_configured:
            logger.warning("Grok not configured")
            return []
        
        logger.info(f"Scanning upcoming catalysts for {ticker}")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        system_prompt = self._build_catalyst_search_prompt(ticker)
        user_message = f"Find all upcoming catalysts and events for {ticker}"
        
        response = await self._call_grok(
            system_prompt=system_prompt,
            user_message=user_message,
            use_search=True,
        )
        
        if not response:
            return []
        
        # Parse catalysts into event signals
        signals = []
        now = self.temporal_context.now
        
        for catalyst in response.get("catalysts", []):
            try:
                event = catalyst.get("event", "Unknown event")
                expected_date = catalyst.get("expected_date")
                impact = catalyst.get("potential_impact", "medium")
                
                signal_id = str(uuid.uuid4())
                summary = f"{ticker} catalyst: {event} expected {expected_date or 'TBD'}"
                dedup_hash = self._generate_dedup_hash("grok_web", ticker, summary)
                
                if dedup_hash in self._seen_signals:
                    continue
                self._seen_signals[dedup_hash] = now
                
                # Determine time horizon from date
                time_horizon = "unknown"
                if expected_date:
                    try:
                        catalyst_dt = datetime.fromisoformat(expected_date)
                        days_out = (catalyst_dt - now.replace(tzinfo=None)).days
                        time_horizon = self.temporal_context.get_horizon_label(days_out)
                    except (ValueError, TypeError):
                        pass
                
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
                        "type": None,
                        "value": None,
                        "unit": None,
                        "prior_value": None,
                        "change": None,
                        "change_period": None,
                    },
                    evidence=[{
                        "source": "grok_web_search",
                        "source_tier": "tier2",
                        "excerpt": catalyst.get("market_expectation", event),
                        "timestamp_utc": now.isoformat(),
                    }],
                    confidence=0.60 if impact == "high" else 0.45,
                    confidence_factors={
                        "source_base": 0.55,
                        "recency_factor": 1.0,
                        "corroboration_factor": 1.0,
                    },
                    directional_bias="unclear",
                    time_horizon=time_horizon,
                    novelty="new",
                    staleness_policy={
                        "max_age_seconds": 86400,  # 24 hours
                        "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
                    },
                    uncertainties=[
                        f"Date confidence: {catalyst.get('date_confidence', 'estimated')}"
                    ],
                    timestamp_utc=now.isoformat(),
                    forward_horizon=time_horizon,
                    catalyst_date=expected_date,
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error parsing catalyst: {e}")
                continue
        
        logger.info(f"Generated {len(signals)} catalyst signals for {ticker}")
        return signals


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def scan_grok_sentiment(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to run a Grok sentiment scan.
    
    Returns:
        List of signal dictionaries ready for storage
    """
    scanner = GrokScanner()
    signals = await scanner.scan_sentiment(tickers)
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
        print("GROK SCANNER TEST (Forward-Looking)")
        print("="*60)
        
        # Show temporal context
        context = scanner.temporal_context.to_dict()
        print(f"\nReference Time: {context['reference_time']}")
        print(f"Key Dates:")
        for name, date in context["key_dates"].items():
            print(f"  {name}: {date}")
        
        print("\n" + "-"*60)
        print("Testing Query Generation...")
        print("-"*60)
        
        builder = scanner.query_builder
        
        print(f"\nSentiment Query (NVDA):")
        print(f"  {builder.build_sentiment_query('NVDA', TimeHorizon.SHORT_TERM)}")
        
        print(f"\nMarket Outlook Query:")
        print(f"  {builder.build_market_outlook_query()}")
        
        print(f"\nCatalyst Query (AAPL):")
        print(f"  {builder.build_catalyst_query('AAPL')}")
        
        if scanner.is_configured:
            print("\n" + "-"*60)
            print("Testing API (requires XAI_API_KEY)...")
            print("-"*60)
            
            signals = await scanner.scan_market_overview()
            print(f"\nGot {len(signals)} market overview signals")
            for sig in signals[:2]:
                print(f"  - {sig.summary[:80]}...")
    
    asyncio.run(test())
