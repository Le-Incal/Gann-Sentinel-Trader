"""
Gann Sentinel Trader - Polymarket Scanner
Retrieves prediction market data from Polymarket.
"""

import logging
import httpx
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from config import Config
from models.signals import Signal, SignalType, SignalSource, DirectionalBias, TimeHorizon, AssetScope, RawValue, Evidence

logger = logging.getLogger(__name__)


# Markets we care about for trading
TRACKED_MARKETS = {
    "fed": {
        "keywords": ["federal reserve", "fed rate", "fomc", "interest rate"],
        "description": "Federal Reserve rate decisions",
        "impact": "Rate cuts bullish for equities, rate hikes bearish"
    },
    "recession": {
        "keywords": ["recession", "economic downturn"],
        "description": "Recession probability",
        "impact": "Higher recession probability is bearish"
    },
    "election": {
        "keywords": ["president", "election 2024", "trump", "biden"],
        "description": "US Presidential election outcomes",
        "impact": "Market impact depends on candidate policies"
    },
    "tariffs": {
        "keywords": ["tariff", "trade war", "china trade"],
        "description": "Trade policy and tariffs",
        "impact": "Tariffs generally bearish for affected sectors"
    },
    "crypto": {
        "keywords": ["bitcoin", "ethereum", "crypto", "btc etf"],
        "description": "Cryptocurrency-related predictions",
        "impact": "Crypto sentiment affects risk appetite"
    }
}


class PolymarketScanner:
    """
    Scans Polymarket for relevant prediction market data.
    """
    
    def __init__(self):
        """Initialize Polymarket scanner."""
        # Polymarket has a public API - no key needed for basic queries
        self.base_url = "https://clob.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
    
    async def _fetch_markets(self, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch markets from Polymarket."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Try the gamma API for market discovery
                response = await client.get(
                    f"{self.gamma_url}/markets",
                    params={
                        "limit": limit,
                        "active": "true",
                        "closed": "false"
                    }
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Error fetching from gamma API: {e}")
                return []
    
    async def _fetch_market_by_id(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific market by ID."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.gamma_url}/markets/{market_id}"
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Error fetching market {market_id}: {e}")
                return None
    
    async def scan_all_markets(self) -> List[Signal]:
        """
        Scan all relevant prediction markets.
        
        Returns:
            List of Signal objects
        """
        signals = []
        
        try:
            # Fetch active markets
            markets = await self._fetch_markets(limit=100)
            
            if not markets:
                logger.warning("No markets retrieved from Polymarket")
                return []
            
            # Filter for markets we care about
            for market in markets:
                try:
                    signal = self._market_to_signal(market)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error processing market: {e}")
            
        except Exception as e:
            logger.error(f"Error scanning Polymarket: {e}")
        
        return signals
    
    async def scan_fed_markets(self) -> List[Signal]:
        """Scan specifically for Fed-related markets."""
        return await self._scan_category("fed")
    
    async def scan_election_markets(self) -> List[Signal]:
        """Scan specifically for election-related markets."""
        return await self._scan_category("election")
    
    async def _scan_category(self, category: str) -> List[Signal]:
        """Scan markets for a specific category."""
        if category not in TRACKED_MARKETS:
            return []
        
        signals = []
        category_info = TRACKED_MARKETS[category]
        keywords = category_info["keywords"]
        
        try:
            markets = await self._fetch_markets(limit=100)
            
            for market in markets:
                question = market.get("question", "").lower()
                description = market.get("description", "").lower()
                
                # Check if market matches any keyword
                matches = any(kw in question or kw in description for kw in keywords)
                
                if matches:
                    signal = self._market_to_signal(market, category=category)
                    if signal:
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error scanning {category} markets: {e}")
        
        return signals
    
    def _market_to_signal(
        self,
        market: Dict[str, Any],
        category: Optional[str] = None
    ) -> Optional[Signal]:
        """Convert a Polymarket market to a Signal."""
        try:
            question = market.get("question", "Unknown")
            
            # Get probability/price
            # Polymarket uses different fields depending on the API
            probability = None
            
            # Try different probability fields
            if "outcomePrices" in market:
                prices = market["outcomePrices"]
                if isinstance(prices, list) and len(prices) > 0:
                    probability = float(prices[0])
            elif "bestBid" in market:
                probability = float(market.get("bestBid", 0.5))
            elif "lastTradePrice" in market:
                probability = float(market.get("lastTradePrice", 0.5))
            
            if probability is None:
                probability = 0.5
            
            # Get volume
            volume = market.get("volume", market.get("volumeNum", 0))
            if isinstance(volume, str):
                try:
                    volume = float(volume.replace(",", "").replace("$", ""))
                except:
                    volume = 0
            
            # Determine category if not provided
            if not category:
                category = self._categorize_market(question)
            
            category_info = TRACKED_MARKETS.get(category, {})
            
            # Skip if not a tracked category
            if not category_info and category is None:
                return None
            
            # Determine directional bias
            direction = self._determine_direction(category, probability, question)
            
            # Determine relevant assets
            asset_scope = self._determine_asset_scope(category, question)
            
            signal = Signal(
                signal_type=SignalType.PREDICTION_MARKET,
                source=SignalSource.POLYMARKET,
                asset_scope=asset_scope,
                summary=f"Polymarket: {question} - {probability*100:.1f}% probability",
                raw_value=RawValue(
                    type="probability",
                    value=probability,
                    unit="percent",
                    prior_value=None,  # Would need historical data
                    change=None,
                    change_period=None
                ),
                evidence=[
                    Evidence(
                        source="Polymarket",
                        source_tier="tier2",
                        excerpt=f"Market: {question}. Current probability: {probability*100:.1f}%. Volume: ${volume:,.0f}",
                        timestamp_utc=datetime.now(timezone.utc),
                        url=f"https://polymarket.com"
                    )
                ],
                confidence=self._calculate_confidence(volume),
                directional_bias=direction,
                time_horizon=TimeHorizon.WEEKS,
                novelty="developing",
                timestamp_utc=datetime.now(timezone.utc),
                staleness_seconds=Config.STALENESS_PREDICTION,
                uncertainties=[
                    "Prediction market probabilities reflect bettor consensus, not certainty",
                    f"Market volume: ${volume:,.0f}" if volume > 0 else "Volume data unavailable",
                    category_info.get("impact", "Market impact varies by outcome")
                ]
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting market to signal: {e}")
            return None
    
    def _categorize_market(self, question: str) -> Optional[str]:
        """Categorize a market based on its question."""
        question_lower = question.lower()
        
        for category, info in TRACKED_MARKETS.items():
            if any(kw in question_lower for kw in info["keywords"]):
                return category
        
        return None
    
    def _determine_direction(
        self,
        category: Optional[str],
        probability: float,
        question: str
    ) -> DirectionalBias:
        """Determine directional bias based on category and probability."""
        if category == "recession":
            # Higher recession probability = bearish
            if probability > 0.6:
                return DirectionalBias.NEGATIVE
            elif probability < 0.3:
                return DirectionalBias.POSITIVE
            else:
                return DirectionalBias.MIXED
        
        elif category == "fed":
            # Need to parse if it's about cuts or hikes
            question_lower = question.lower()
            if "cut" in question_lower:
                if probability > 0.6:
                    return DirectionalBias.POSITIVE  # Rate cuts bullish
                elif probability < 0.3:
                    return DirectionalBias.NEGATIVE
            elif "hike" in question_lower or "raise" in question_lower:
                if probability > 0.6:
                    return DirectionalBias.NEGATIVE  # Rate hikes bearish
                elif probability < 0.3:
                    return DirectionalBias.POSITIVE
        
        elif category == "tariffs":
            # Tariffs generally negative for markets
            if probability > 0.6:
                return DirectionalBias.NEGATIVE
            elif probability < 0.3:
                return DirectionalBias.POSITIVE
        
        return DirectionalBias.UNCLEAR
    
    def _determine_asset_scope(
        self,
        category: Optional[str],
        question: str
    ) -> AssetScope:
        """Determine which assets are affected by this prediction."""
        scope = AssetScope(macro_regions=["US"])
        
        if category == "fed":
            scope.asset_classes = ["EQUITY", "FIXED_INCOME"]
            scope.tickers = ["SPY", "TLT", "IWM"]
        elif category == "recession":
            scope.asset_classes = ["EQUITY"]
            scope.sectors = ["FINANCIALS", "CONSUMER_DISCRETIONARY"]
        elif category == "election":
            scope.asset_classes = ["EQUITY"]
        elif category == "tariffs":
            scope.asset_classes = ["EQUITY"]
            scope.sectors = ["INDUSTRIALS", "TECH"]
        elif category == "crypto":
            scope.asset_classes = ["CRYPTO"]
            scope.tickers = ["COIN", "MSTR"]
        
        return scope
    
    def _calculate_confidence(self, volume: float) -> float:
        """Calculate confidence based on market volume."""
        # Higher volume = more confidence in the signal
        if volume >= 1000000:
            return 0.80
        elif volume >= 500000:
            return 0.70
        elif volume >= 100000:
            return 0.60
        elif volume >= 50000:
            return 0.50
        else:
            return 0.40
