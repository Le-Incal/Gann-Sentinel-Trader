"""
Gann Sentinel Trader - Polymarket Scanner
Forward-looking prediction market signal extraction.

This scanner uses the shared temporal awareness framework to ensure
we always look projectively (1mo, 3mo, 6mo, 12mo forward).

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

import httpx

# Import shared temporal framework
from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


class MarketCategory(Enum):
    """Categories of markets we care about for trading signals."""
    FED_MONETARY = "fed_monetary"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    TECH_SECTOR = "tech_sector"
    CRYPTO = "crypto"
    ELECTIONS = "elections"
    OTHER = "other"


# Tags and keywords for categorization
CATEGORY_KEYWORDS = {
    MarketCategory.FED_MONETARY: [
        "fed", "federal reserve", "fomc", "rate cut", "rate hike", 
        "interest rate", "powell", "monetary policy", "basis points",
        "quantitative", "tightening", "easing"
    ],
    MarketCategory.ECONOMIC_DATA: [
        "gdp", "inflation", "cpi", "unemployment", "jobs", "payroll",
        "recession", "economy", "economic", "pce", "retail sales",
        "manufacturing", "pmi", "consumer"
    ],
    MarketCategory.GEOPOLITICAL: [
        "tariff", "trade war", "sanctions", "china", "russia", "ukraine",
        "war", "conflict", "nato", "eu", "brexit", "opec"
    ],
    MarketCategory.TECH_SECTOR: [
        "ai", "artificial intelligence", "nvidia", "microsoft", "apple",
        "google", "amazon", "meta", "semiconductor", "chip", "tech"
    ],
    MarketCategory.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "blockchain", "defi", "sec crypto"
    ],
    MarketCategory.ELECTIONS: [
        "election", "president", "congress", "senate", "house",
        "vote", "ballot", "poll", "democrat", "republican"
    ],
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PolymarketSignal:
    """Signal extracted from Polymarket conforming to Grok Spec v1.1.0."""
    signal_id: str
    dedup_hash: str
    category: str  # prediction_market (per spec)
    source_type: str  # polymarket
    
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
    
    # Our additions for internal tracking
    market_category: str  # Our internal categorization
    resolution_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.category,  # For database compatibility
            "category": self.category,
            "source_type": self.source_type,
            "source": self.source_type,  # For database compatibility
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
            "market_category": self.market_category,
            "resolution_date": self.resolution_date,
        }


# =============================================================================
# POLYMARKET SCANNER
# =============================================================================

class PolymarketScanner:
    """
    Scanner for Polymarket prediction markets.
    
    Uses the shared temporal framework to ensure consistent forward-looking
    date windows across all scanners.
    
    Time Windows (from temporal.py):
    - IMMEDIATE: 7 days
    - SHORT_TERM: 30 days
    - MEDIUM_TERM: 90 days  
    - LONG_TERM: 180 days
    - EXTENDED: 365 days
    """
    
    def __init__(self):
        """Initialize the Polymarket scanner."""
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.markets_endpoint = f"{self.gamma_url}/markets"
        self.events_endpoint = f"{self.gamma_url}/events"
        
        # HTTP client settings
        self.timeout = 30.0
        self.max_retries = 3
        
        # Initialize shared temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for seen markets (deduplication)
        self._seen_markets: Dict[str, datetime] = {}
        
        # Log temporal context
        self.temporal_context.log_context()
        logger.info("PolymarketScanner initialized with shared temporal framework")
    
    # =========================================================================
    # DATE WINDOW CALCULATION (delegates to temporal framework)
    # =========================================================================
    
    def _get_current_date(self) -> datetime:
        """Get current UTC datetime from temporal context."""
        return self.temporal_context.now
    
    def _get_date_window(self, horizon: TimeHorizon) -> Tuple[datetime, datetime]:
        """Get date window for a specific horizon."""
        return self.temporal_context.get_window(horizon)
    
    def _format_date_for_api(self, dt: datetime) -> str:
        """Format datetime for Polymarket API (YYYY-MM-DD)."""
        return self.temporal_context.format_date(dt)
    
    # =========================================================================
    # MARKET CATEGORIZATION
    # =========================================================================
    
    def _categorize_market(self, market: Dict[str, Any]) -> MarketCategory:
        """
        Categorize a market based on its title and description.
        
        Args:
            market: Market data from Polymarket API
            
        Returns:
            MarketCategory enum value
        """
        # Get text to analyze
        question = (market.get("question") or "").lower()
        description = (market.get("description") or "").lower()
        title = (market.get("title") or "").lower()
        
        combined_text = f"{question} {description} {title}"
        
        # Check each category
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return category
        
        return MarketCategory.OTHER
    
    def _map_category_to_time_horizon(
        self, 
        category: MarketCategory,
        resolution_date: Optional[datetime]
    ) -> str:
        """Map category and resolution date to time horizon string."""
        if resolution_date:
            horizon = self.temporal_context.classify_future_date(resolution_date)
            if horizon:
                return horizon.signal_horizon_label
            return "unknown"  # Date is in the past
        
        # Default based on category
        if category in [MarketCategory.FED_MONETARY, MarketCategory.ECONOMIC_DATA]:
            return "weeks"
        elif category == MarketCategory.ELECTIONS:
            return "months"
        else:
            return "unknown"
    
    def _category_to_asset_scope(self, category: MarketCategory) -> Dict[str, List[str]]:
        """Map market category to relevant asset scope."""
        scope_map = {
            MarketCategory.FED_MONETARY: {
                "tickers": ["SPY", "TLT", "DX-Y.NYB"],  # S&P, Treasuries, Dollar
                "sectors": ["FINANCIALS"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY", "FIXED_INCOME", "FX"],
            },
            MarketCategory.ECONOMIC_DATA: {
                "tickers": ["SPY", "IWM", "TLT"],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY", "FIXED_INCOME"],
            },
            MarketCategory.GEOPOLITICAL: {
                "tickers": ["VIX", "GLD", "USO"],  # Volatility, Gold, Oil
                "sectors": ["ENERGY", "DEFENSE"],
                "macro_regions": ["GLOBAL"],
                "asset_classes": ["COMMODITY", "EQUITY"],
            },
            MarketCategory.TECH_SECTOR: {
                "tickers": ["QQQ", "NVDA", "SMH"],
                "sectors": ["TECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            MarketCategory.CRYPTO: {
                "tickers": ["COIN", "MSTR"],
                "sectors": ["CRYPTO"],
                "macro_regions": ["GLOBAL"],
                "asset_classes": ["CRYPTO", "EQUITY"],
            },
            MarketCategory.ELECTIONS: {
                "tickers": ["SPY"],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            MarketCategory.OTHER: {
                "tickers": [],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
        }
        
        return scope_map.get(category, scope_map[MarketCategory.OTHER])
    
    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================
    
    def _calculate_confidence(
        self,
        market: Dict[str, Any],
        category: MarketCategory
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate confidence score per Grok Spec v1.1.0 formula.
        
        confidence = source_base × recency_factor × corroboration_factor
        
        Returns:
            Tuple of (final_confidence, confidence_factors_dict)
        """
        # Source base: Polymarket is a tier-2 prediction market
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)
        
        # Higher volume = more reliable signal
        if volume >= 100000:  # >$100k volume
            source_base = 0.75
        elif volume >= 50000:  # $50k-$100k
            source_base = 0.60
        elif volume >= 10000:  # $10k-$50k
            source_base = 0.45
        else:
            source_base = 0.30
        
        # Recency factor based on last trade
        # Since we're querying active markets, assume recent
        recency_factor = 0.95
        
        # Corroboration: Single source (Polymarket)
        corroboration_factor = 1.0
        
        # Calculate final (capped at 1.0)
        confidence = min(source_base * recency_factor * corroboration_factor, 1.0)
        
        factors = {
            "source_base": source_base,
            "recency_factor": recency_factor,
            "corroboration_factor": corroboration_factor,
        }
        
        return confidence, factors
    
    # =========================================================================
    # DIRECTIONAL BIAS DETERMINATION
    # =========================================================================
    
    def _determine_directional_bias(
        self,
        market: Dict[str, Any],
        category: MarketCategory
    ) -> str:
        """
        Determine directional bias based on probability and category.
        
        For Fed rate cuts: Higher probability of cut = positive for equities
        For recession: Higher probability = negative for equities
        etc.
        """
        # Get current probability
        prob = self._extract_probability(market)
        if prob is None:
            return "unclear"
        
        question = (market.get("question") or "").lower()
        
        # Fed rate cuts - cuts are generally positive for equities
        if "rate cut" in question:
            return "positive" if prob >= 0.5 else "negative"
        
        # Fed rate hikes - hikes are generally negative for equities
        if "rate hike" in question or "rate increase" in question:
            return "negative" if prob >= 0.5 else "positive"
        
        # Recession - clearly negative
        if "recession" in question:
            return "negative" if prob >= 0.5 else "positive"
        
        # Tariffs/trade war - generally negative
        if "tariff" in question or "trade war" in question:
            return "negative" if prob >= 0.5 else "mixed"
        
        # Default: significant moves in either direction = mixed
        if prob >= 0.7:
            return "positive"
        elif prob <= 0.3:
            return "negative"
        else:
            return "mixed"
    
    def _extract_probability(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract current probability from market data."""
        # Try various fields where probability might be
        for field in ["outcomePrices", "lastTradePrice", "bestBid", "bestAsk"]:
            if field in market and market[field]:
                try:
                    if isinstance(market[field], list):
                        # Usually [YES_price, NO_price]
                        return float(market[field][0])
                    elif isinstance(market[field], (int, float)):
                        return float(market[field])
                    elif isinstance(market[field], str):
                        return float(market[field])
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_prior_probability(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract 24h prior probability if available."""
        # This would require historical data - may not be in basic API
        # For now, return None
        return None
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(
        self, 
        source_type: str, 
        question: str,
        primary_ticker_or_region: str
    ) -> str:
        """Generate deduplication hash per Grok Spec v1.1.0."""
        normalized = f"{source_type}:{primary_ticker_or_region}:{question.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _market_to_signal(
        self, 
        market: Dict[str, Any],
        window_name: str
    ) -> Optional[PolymarketSignal]:
        """
        Convert a Polymarket market to a standardized signal.
        
        Args:
            market: Raw market data from API
            window_name: Which time window this came from
            
        Returns:
            PolymarketSignal or None if invalid
        """
        try:
            # Extract basic info
            question = market.get("question") or market.get("title") or ""
            if not question:
                logger.debug("Skipping market with no question/title")
                return None
            
            market_id = market.get("id") or market.get("conditionId") or str(uuid.uuid4())
            
            # Categorize
            category = self._categorize_market(market)
            
            # Skip OTHER category if not interesting
            if category == MarketCategory.OTHER:
                logger.debug(f"Skipping OTHER category market: {question[:50]}")
                return None
            
            # Asset scope
            asset_scope = self._category_to_asset_scope(category)
            primary_identifier = (
                asset_scope["tickers"][0] if asset_scope["tickers"]
                else asset_scope["macro_regions"][0] if asset_scope["macro_regions"]
                else "US"
            )
            
            # Generate IDs
            signal_id = str(uuid.uuid4())
            dedup_hash = self._generate_dedup_hash("polymarket", question, primary_identifier)
            
            # Check dedup cache
            if dedup_hash in self._seen_markets:
                last_seen = self._seen_markets[dedup_hash]
                if (self._get_current_date() - last_seen).total_seconds() < 3600:
                    logger.debug(f"Skipping duplicate: {question[:50]}")
                    return None
            
            self._seen_markets[dedup_hash] = self._get_current_date()
            
            # Extract probability
            current_prob = self._extract_probability(market)
            prior_prob = self._extract_prior_probability(market)
            
            # Calculate change if we have both
            change = None
            if current_prob is not None and prior_prob is not None:
                change = current_prob - prior_prob
            
            raw_value = {
                "type": "probability",
                "value": current_prob,
                "unit": "percent",
                "prior_value": prior_prob,
                "change": change,
                "change_period": "24h" if prior_prob else None,
            }
            
            # Parse resolution date
            resolution_date = None
            end_date_str = market.get("endDate") or market.get("endDateIso") or market.get("closedTime")
            if end_date_str:
                try:
                    resolution_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            
            # Time horizon
            time_horizon = self._map_category_to_time_horizon(category, resolution_date)
            
            # Confidence
            confidence, confidence_factors = self._calculate_confidence(market, category)
            
            # Directional bias
            directional_bias = self._determine_directional_bias(market, category)
            
            # Build summary
            prob_str = f"{current_prob*100:.0f}%" if current_prob else "unknown"
            summary = f"Polymarket: {question} Currently at {prob_str}."
            
            # Evidence
            evidence = [{
                "source": f"https://polymarket.com/market/{market_id}",
                "source_tier": "tier2",
                "excerpt": question,
                "timestamp_utc": self._get_current_date().isoformat(),
            }]
            
            # Volume context for uncertainties
            volume = float(market.get("volume", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)
            
            uncertainties = []
            if volume < 50000:
                uncertainties.append(f"Low volume market (${volume:,.0f})")
            if liquidity < 10000:
                uncertainties.append(f"Low liquidity (${liquidity:,.0f})")
            if not resolution_date:
                uncertainties.append("Resolution date unclear")
            
            # Staleness policy - prediction markets are fast-moving
            now = self._get_current_date()
            staleness_policy = {
                "max_age_seconds": 3600,  # 1 hour
                "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
            }
            
            # Novelty
            novelty = "developing" if dedup_hash in self._seen_markets else "new"
            
            return PolymarketSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="prediction_market",
                source_type="polymarket",
                asset_scope=asset_scope,
                summary=summary,
                raw_value=raw_value,
                evidence=evidence,
                confidence=confidence,
                confidence_factors=confidence_factors,
                directional_bias=directional_bias,
                time_horizon=time_horizon,
                novelty=novelty,
                staleness_policy=staleness_policy,
                uncertainties=uncertainties,
                timestamp_utc=now.isoformat(),
                market_category=category.value,
                resolution_date=resolution_date.isoformat() if resolution_date else None,
            )
            
        except Exception as e:
            logger.error(f"Error converting market to signal: {e}")
            return None
    
    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _fetch_markets(
        self,
        end_date_min: str,
        end_date_max: str,
        active: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Polymarket Gamma API with date filtering.
        
        Args:
            end_date_min: Minimum resolution date (YYYY-MM-DD)
            end_date_max: Maximum resolution date (YYYY-MM-DD)
            active: Only fetch active markets
            limit: Max results
            
        Returns:
            List of market dicts
        """
        params = {
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
            "active": str(active).lower(),
            "closed": "false",
            "limit": limit,
            "order": "volume",  # Most active first
            "ascending": "false",
        }
        
        logger.debug(f"Fetching markets: {end_date_min} to {end_date_max}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.markets_endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Fetched {len(data)} markets from Polymarket")
                    return data if isinstance(data, list) else []
                else:
                    logger.error(f"Polymarket API error: {response.status_code}")
                    return []
                    
        except httpx.TimeoutException:
            logger.error("Polymarket API timeout")
            return []
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
            return []
    
    async def _fetch_events_by_tag(
        self,
        tag: str,
        end_date_min: str,
        end_date_max: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fetch events filtered by tag (e.g., 'fed', 'economics').
        
        Args:
            tag: Tag to filter by
            end_date_min: Minimum resolution date
            end_date_max: Maximum resolution date
            limit: Max results
            
        Returns:
            List of event dicts with nested markets
        """
        params = {
            "tag": tag,
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.events_endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"Fetched {len(data)} events for tag '{tag}'")
                    return data if isinstance(data, list) else []
                else:
                    logger.warning(f"Events API error for tag '{tag}': {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Events API error: {e}")
            return []
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    async def scan_all_markets(self) -> List[PolymarketSignal]:
        """
        Scan all relevant Polymarket markets across time windows.
        
        This is the main entry point for the agent.
        
        Returns:
            List of PolymarketSignal objects
        """
        logger.info("Starting Polymarket scan with shared temporal framework...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Use MEDIUM_TERM (3 months) as the main window
        start, end = self._get_date_window(TimeHorizon.MEDIUM_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        logger.info(f"Scanning markets resolving between {start_str} and {end_str}")
        
        # Fetch markets
        markets = await self._fetch_markets(
            end_date_min=start_str,
            end_date_max=end_str,
            limit=100,
        )
        
        # Convert to signals
        for market in markets:
            signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
            if signal:
                signals.append(signal)
        
        # Also try fetching by relevant tags
        relevant_tags = ["fed", "economics", "inflation", "tariffs", "ai"]
        
        for tag in relevant_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=20,
                )
                
                for event in events:
                    # Events have nested markets
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
                        if signal:
                            # Dedupe by hash
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)
                                
            except Exception as e:
                logger.warning(f"Error fetching tag '{tag}': {e}")
                continue
        
        logger.info(f"Polymarket scan complete: {len(signals)} signals generated")
        return signals
    
    async def scan_fed_markets(self) -> List[PolymarketSignal]:
        """
        Specifically scan Fed/monetary policy markets.
        
        Returns:
            List of Fed-related signals
        """
        logger.info("Scanning Fed/monetary policy markets...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Look 6 months out for Fed decisions
        start, end = self._get_date_window(TimeHorizon.LONG_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        # Try Fed-specific tags
        fed_tags = ["fed", "fomc", "federal-reserve", "interest-rates"]
        
        for tag in fed_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=30,
                )
                
                for event in events:
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.LONG_TERM.name)
                        if signal and signal.market_category == MarketCategory.FED_MONETARY.value:
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)
                                
            except Exception as e:
                logger.warning(f"Error fetching Fed tag '{tag}': {e}")
                continue
        
        logger.info(f"Fed market scan complete: {len(signals)} signals")
        return signals
    
    async def scan_economic_events(self) -> List[PolymarketSignal]:
        """
        Scan for economic event markets (GDP, CPI, jobs, etc.).
        
        Returns:
            List of economic event signals
        """
        logger.info("Scanning economic event markets...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Economic data tends to be shorter term
        start, end = self._get_date_window(TimeHorizon.MEDIUM_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        econ_tags = ["economics", "gdp", "inflation", "unemployment", "recession"]
        
        for tag in econ_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=20,
                )
                
                for event in events:
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
                        if signal and signal.market_category == MarketCategory.ECONOMIC_DATA.value:
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)
                                
            except Exception as e:
                logger.warning(f"Error fetching econ tag '{tag}': {e}")
                continue
        
        logger.info(f"Economic events scan complete: {len(signals)} signals")
        return signals


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_polymarket() -> List[Dict[str, Any]]:
    """
    Convenience function to run a full Polymarket scan.
    
    Returns:
        List of signal dictionaries ready for storage
    """
    scanner = PolymarketScanner()
    signals = await scanner.scan_all_markets()
    return [s.to_dict() for s in signals]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = PolymarketScanner()
        
        print("\n" + "="*60)
        print("POLYMARKET SCANNER TEST")
        print("="*60)
        
        # Show date windows
        windows = scanner._calculate_date_windows()
        print("\nForward-Looking Date Windows:")
        for name, (start, end) in windows.items():
            print(f"  {name}: {start.date()} to {end.date()}")
        
        print("\n" + "-"*60)
        print("Scanning markets...")
        print("-"*60)
        
        signals = await scanner.scan_all_markets()
        
        print(f"\nFound {len(signals)} signals:")
        for signal in signals[:5]:
            print(f"\n  Category: {signal.market_category}")
            print(f"  Summary: {signal.summary[:80]}...")
            print(f"  Probability: {signal.raw_value.get('value')}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Bias: {signal.directional_bias}")
            print(f"  Horizon: {signal.time_horizon}")
        
        if len(signals) > 5:
            print(f"\n  ... and {len(signals) - 5} more")
    
    asyncio.run(test())
