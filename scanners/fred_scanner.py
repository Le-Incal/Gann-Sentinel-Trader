"""
Gann Sentinel Trader - FRED Scanner
Forward-contextualized macroeconomic data extraction from FRED.

FRED data is inherently backward-looking (it reports released data), but
this scanner frames the SIGNALS in a forward-looking context:
- What does this reading imply for the next 3-6 months?
- How does it affect Fed policy expectations?
- What are the market implications going forward?

Version: 2.0.0 (Temporal Awareness Update)
Last Updated: January 2026
"""

import os
import uuid
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx

# Import our temporal framework
from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class FREDSeries(Enum):
    """
    Key FRED series we track.
    
    Each series includes:
    - series_id: FRED series identifier
    - name: Human-readable name
    - frequency: How often it's released
    - forward_relevance: Why it matters for forward outlook
    """
    # Treasury Yields
    DGS10 = ("DGS10", "10-Year Treasury Yield", "daily", "Rate expectations, risk appetite")
    DGS2 = ("DGS2", "2-Year Treasury Yield", "daily", "Fed policy expectations")
    T10Y2Y = ("T10Y2Y", "10Y-2Y Spread", "daily", "Recession indicator, curve shape")
    
    # Fed Policy
    FEDFUNDS = ("FEDFUNDS", "Federal Funds Rate", "daily", "Current policy stance")
    
    # Employment
    UNRATE = ("UNRATE", "Unemployment Rate", "monthly", "Labor market health, Fed dual mandate")
    PAYEMS = ("PAYEMS", "Nonfarm Payrolls", "monthly", "Job growth momentum")
    ICSA = ("ICSA", "Initial Jobless Claims", "weekly", "Labor market leading indicator")
    
    # Inflation
    CPIAUCSL = ("CPIAUCSL", "CPI All Items", "monthly", "Inflation, Fed policy driver")
    PCEPI = ("PCEPI", "PCE Price Index", "monthly", "Fed's preferred inflation measure")
    
    # Growth
    GDP = ("GDP", "Real GDP", "quarterly", "Economic growth, recession/expansion")
    
    # Consumer
    UMCSENT = ("UMCSENT", "Consumer Sentiment", "monthly", "Consumer spending outlook")
    RSAFS = ("RSAFS", "Retail Sales", "monthly", "Consumer demand")
    
    @property
    def series_id(self) -> str:
        return self.value[0]
    
    @property
    def display_name(self) -> str:
        return self.value[1]
    
    @property
    def frequency(self) -> str:
        return self.value[2]
    
    @property
    def forward_relevance(self) -> str:
        return self.value[3]


# Mapping series to relevant tickers
SERIES_ASSET_MAPPING = {
    "DGS10": {"tickers": ["TLT", "IEF"], "asset_classes": ["FIXED_INCOME"]},
    "DGS2": {"tickers": ["SHY", "IEI"], "asset_classes": ["FIXED_INCOME"]},
    "T10Y2Y": {"tickers": ["TLT", "SPY"], "asset_classes": ["FIXED_INCOME", "EQUITY"]},
    "FEDFUNDS": {"tickers": ["SPY", "TLT"], "asset_classes": ["EQUITY", "FIXED_INCOME"]},
    "UNRATE": {"tickers": ["SPY", "IWM"], "asset_classes": ["EQUITY"]},
    "PAYEMS": {"tickers": ["SPY", "IWM"], "asset_classes": ["EQUITY"]},
    "ICSA": {"tickers": ["SPY"], "asset_classes": ["EQUITY"]},
    "CPIAUCSL": {"tickers": ["TIP", "TLT", "GLD"], "asset_classes": ["FIXED_INCOME", "COMMODITY"]},
    "PCEPI": {"tickers": ["SPY", "TLT"], "asset_classes": ["EQUITY", "FIXED_INCOME"]},
    "GDP": {"tickers": ["SPY", "IWM"], "asset_classes": ["EQUITY"]},
    "UMCSENT": {"tickers": ["XLY", "XRT"], "asset_classes": ["EQUITY"]},
    "RSAFS": {"tickers": ["XRT", "XLY"], "asset_classes": ["EQUITY"]},
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class FREDSignal:
    """Signal extracted from FRED conforming to Grok Spec v1.1.0."""
    signal_id: str
    dedup_hash: str
    category: str  # "macro"
    source_type: str  # "fred"
    
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
    forward_implication: Optional[str] = None
    next_release: Optional[str] = None
    
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
            "staleness_seconds": self.staleness_policy.get("max_age_seconds", 604800),
            "uncertainties": self.uncertainties,
            "timestamp_utc": self.timestamp_utc,
            "forward_implication": self.forward_implication,
            "next_release": self.next_release,
        }


# =============================================================================
# FRED SCANNER
# =============================================================================

class FREDScanner:
    """
    Scanner for FRED macroeconomic data with forward-looking context.
    
    While FRED data is backward-looking (released data), we frame signals
    in terms of their FORWARD implications:
    - What does this mean for Fed policy over the next 3-6 months?
    - How should markets interpret this for the forward period?
    - What's the trend trajectory suggesting?
    
    All signals include forward_implication context.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the FRED scanner."""
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        
        if not self.api_key:
            logger.warning("FRED_API_KEY not set - FRED scanning disabled")
        
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Initialize temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for deduplication
        self._seen_signals: Dict[str, datetime] = {}
        
        # Track last values for change calculation
        self._last_values: Dict[str, float] = {}
        
        # Log temporal context
        self.temporal_context.log_context()
        logger.info("FREDScanner initialized with forward-looking context")
    
    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return bool(self.api_key)
    
    # =========================================================================
    # FORWARD-LOOKING CONTEXT GENERATION
    # =========================================================================
    
    def _generate_forward_context(
        self,
        series: FREDSeries,
        current_value: float,
        prior_value: Optional[float],
        change: Optional[float],
    ) -> Tuple[str, str]:
        """
        Generate forward-looking context for a FRED data point.
        
        Instead of just "CPI is 3.2%", we add context like:
        "This suggests Fed may stay higher for longer through Q2"
        
        Args:
            series: The FRED series
            current_value: Current reading
            prior_value: Previous reading
            change: Change from prior
            
        Returns:
            Tuple of (forward_implication, directional_bias)
        """
        series_id = series.series_id
        now = self.temporal_context.now
        end_of_quarter = self.temporal_context.format_date(self.temporal_context.end_of_quarter)
        
        # Treasury Yields
        if series_id == "DGS10":
            if current_value > 4.5:
                return (
                    f"Elevated 10Y yield ({current_value:.2f}%) suggests tight financial conditions "
                    f"through {end_of_quarter}. May pressure equity valuations.",
                    "negative"
                )
            elif current_value < 3.5:
                return (
                    f"Low 10Y yield ({current_value:.2f}%) suggests easing conditions ahead. "
                    f"Supportive for risk assets through {end_of_quarter}.",
                    "positive"
                )
            else:
                return (
                    f"10Y yield at {current_value:.2f}% in neutral range. "
                    f"Watch for breakout direction through {end_of_quarter}.",
                    "mixed"
                )
        
        # Yield Curve
        if series_id == "T10Y2Y":
            if current_value < 0:
                return (
                    f"Inverted yield curve ({current_value:.2f}%) historically precedes recession "
                    f"6-18 months out. Defensive positioning may be warranted.",
                    "negative"
                )
            elif current_value > 0 and change and change > 0:
                return (
                    f"Curve steepening ({current_value:.2f}%) suggests improving growth expectations "
                    f"or Fed pivot anticipation. Positive for cyclicals.",
                    "positive"
                )
            else:
                return (
                    f"Flat curve ({current_value:.2f}%) suggests uncertainty about growth trajectory.",
                    "mixed"
                )
        
        # Fed Funds
        if series_id == "FEDFUNDS":
            return (
                f"Fed funds at {current_value:.2f}%. Forward policy path through "
                f"{end_of_quarter} depends on inflation and employment data.",
                "mixed"
            )
        
        # Unemployment
        if series_id == "UNRATE":
            if current_value < 4.0:
                return (
                    f"Unemployment at {current_value:.1f}% indicates tight labor market. "
                    f"May keep wage pressures elevated through {end_of_quarter}.",
                    "mixed"  # Tight labor can be inflationary
                )
            elif current_value > 5.0:
                return (
                    f"Unemployment at {current_value:.1f}% shows labor softening. "
                    f"Fed may pivot dovish, supportive for equities.",
                    "positive"
                )
            else:
                return (
                    f"Unemployment at {current_value:.1f}% in goldilocks range. "
                    f"Supports soft landing narrative through {end_of_quarter}.",
                    "positive"
                )
        
        # Initial Claims
        if series_id == "ICSA":
            if current_value > 250000:
                return (
                    f"Initial claims elevated at {current_value/1000:.0f}k. "
                    f"Leading indicator of labor weakness, watch for trend.",
                    "negative"
                )
            else:
                return (
                    f"Initial claims low at {current_value/1000:.0f}k. "
                    f"Labor market remains healthy looking forward.",
                    "positive"
                )
        
        # CPI Inflation
        if series_id == "CPIAUCSL":
            # CPI is typically reported as YoY %
            if current_value > 3.0:
                return (
                    f"CPI at {current_value:.1f}% YoY above target. "
                    f"Fed likely stays restrictive through {end_of_quarter}. "
                    f"Headwind for rate-sensitive sectors.",
                    "negative"
                )
            elif current_value < 2.5:
                return (
                    f"CPI at {current_value:.1f}% YoY approaching target. "
                    f"Increases odds of Fed pivot through {end_of_quarter}.",
                    "positive"
                )
            else:
                return (
                    f"CPI at {current_value:.1f}% YoY trending toward target. "
                    f"Disinflationary trajectory supports risk assets.",
                    "positive"
                )
        
        # PCE Inflation (Fed's preferred)
        if series_id == "PCEPI":
            if current_value > 2.5:
                return (
                    f"Core PCE at {current_value:.1f}% above 2% target. "
                    f"Fed stays focused on inflation mandate through {end_of_quarter}.",
                    "negative"
                )
            else:
                return (
                    f"Core PCE at {current_value:.1f}% near target. "
                    f"Supports Fed flexibility on rates going forward.",
                    "positive"
                )
        
        # GDP
        if series_id == "GDP":
            if change and change > 0:
                return (
                    f"GDP growth positive. Expansion mode suggests "
                    f"favorable environment for risk assets through {end_of_quarter}.",
                    "positive"
                )
            elif change and change < 0:
                return (
                    f"GDP contraction raises recession concerns. "
                    f"Defensive positioning may be warranted.",
                    "negative"
                )
            else:
                return (
                    f"GDP growth at {current_value:.1f}%. "
                    f"Forward trajectory key for market direction.",
                    "mixed"
                )
        
        # Consumer Sentiment
        if series_id == "UMCSENT":
            if current_value > 80:
                return (
                    f"Consumer sentiment elevated at {current_value:.0f}. "
                    f"Supports consumer spending outlook through {end_of_quarter}.",
                    "positive"
                )
            elif current_value < 60:
                return (
                    f"Consumer sentiment depressed at {current_value:.0f}. "
                    f"May signal spending weakness ahead.",
                    "negative"
                )
            else:
                return (
                    f"Consumer sentiment neutral at {current_value:.0f}. "
                    f"Watch for direction change.",
                    "mixed"
                )
        
        # Default
        return (
            f"{series.display_name} at {current_value}. "
            f"See {series.forward_relevance}.",
            "mixed"
        )
    
    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _fetch_series(
        self,
        series_id: str,
        limit: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch recent observations from a FRED series.
        
        Args:
            series_id: FRED series ID
            limit: Number of recent observations to fetch
            
        Returns:
            Dict with series info and observations
        """
        if not self.is_configured:
            logger.warning("FRED not configured")
            return None
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/series/observations",
                    params=params,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "series_id": series_id,
                        "observations": data.get("observations", []),
                    }
                else:
                    logger.error(f"FRED API error for {series_id}: {response.status_code}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error(f"FRED API timeout for {series_id}")
            return None
        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return None
    
    async def _fetch_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata about a FRED series."""
        if not self.is_configured:
            return None
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/series",
                    params=params,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    series_list = data.get("seriess", [])
                    return series_list[0] if series_list else None
                return None
                
        except Exception as e:
            logger.error(f"FRED series info error: {e}")
            return None
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(
        self, 
        series_id: str, 
        observation_date: str,
    ) -> str:
        """Generate deduplication hash for a FRED observation."""
        normalized = f"fred:{series_id}:{observation_date}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _observation_to_signal(
        self,
        series: FREDSeries,
        observation: Dict[str, Any],
        prior_observation: Optional[Dict[str, Any]] = None,
    ) -> Optional[FREDSignal]:
        """
        Convert a FRED observation to a forward-looking signal.
        
        Args:
            series: FRED series enum
            observation: Current observation data
            prior_observation: Previous observation for change calculation
            
        Returns:
            FREDSignal or None if invalid
        """
        try:
            now = self.temporal_context.now
            
            # Extract values
            obs_date = observation.get("date")
            value_str = observation.get("value", ".")
            
            if value_str == "." or not value_str:
                logger.debug(f"No value for {series.series_id} on {obs_date}")
                return None
            
            current_value = float(value_str)
            
            # Calculate change
            prior_value = None
            change = None
            if prior_observation:
                prior_str = prior_observation.get("value", ".")
                if prior_str != ".":
                    prior_value = float(prior_str)
                    change = current_value - prior_value
            
            # Generate IDs
            signal_id = str(uuid.uuid4())
            dedup_hash = self._generate_dedup_hash(series.series_id, obs_date)
            
            # Check dedup
            if dedup_hash in self._seen_signals:
                return None
            self._seen_signals[dedup_hash] = now
            
            # Get forward-looking context
            forward_implication, directional_bias = self._generate_forward_context(
                series, current_value, prior_value, change
            )
            
            # Build summary with forward context
            change_str = ""
            if change is not None:
                direction = "up" if change > 0 else "down"
                change_str = f" ({direction} {abs(change):.2f} from prior)"
            
            summary = (
                f"{series.display_name}: {current_value}{change_str}. "
                f"{forward_implication}"
            )
            
            # Asset scope
            asset_mapping = SERIES_ASSET_MAPPING.get(series.series_id, {})
            asset_scope = {
                "tickers": asset_mapping.get("tickers", []),
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": asset_mapping.get("asset_classes", ["EQUITY"]),
            }
            
            # Determine unit
            unit_map = {
                "DGS10": "percent",
                "DGS2": "percent",
                "T10Y2Y": "percent",
                "FEDFUNDS": "percent",
                "UNRATE": "percent",
                "CPIAUCSL": "percent",
                "PCEPI": "percent",
                "GDP": "billions_usd",
                "UMCSENT": "index_points",
                "ICSA": "count",
                "PAYEMS": "thousands",
                "RSAFS": "millions_usd",
            }
            unit = unit_map.get(series.series_id, "index_points")
            
            # Raw value object
            raw_value = {
                "type": "rate" if "percent" in unit else "index",
                "value": current_value,
                "unit": unit,
                "prior_value": prior_value,
                "change": change,
                "change_period": f"1{series.frequency[0]}" if series.frequency else None,
            }
            
            # Evidence
            evidence = [{
                "source": f"https://fred.stlouisfed.org/series/{series.series_id}",
                "source_tier": "official",
                "excerpt": f"{series.display_name}: {current_value} as of {obs_date}",
                "timestamp_utc": now.isoformat(),
            }]
            
            # Confidence - official data is high confidence
            confidence_factors = {
                "source_base": 0.95,  # Official government data
                "recency_factor": 0.95,  # Usually recent
                "corroboration_factor": 1.0,
            }
            confidence = min(
                confidence_factors["source_base"] * 
                confidence_factors["recency_factor"] * 
                confidence_factors["corroboration_factor"],
                1.0
            )
            
            # Time horizon based on series frequency
            if series.frequency == "daily":
                time_horizon = "days"
                staleness_seconds = 86400  # 1 day
            elif series.frequency == "weekly":
                time_horizon = "days"
                staleness_seconds = 604800  # 7 days
            elif series.frequency == "monthly":
                time_horizon = "weeks"
                staleness_seconds = 604800  # 7 days
            else:  # quarterly
                time_horizon = "months"
                staleness_seconds = 604800  # 7 days
            
            staleness_policy = {
                "max_age_seconds": staleness_seconds,
                "stale_after_utc": (now + timedelta(seconds=staleness_seconds)).isoformat(),
            }
            
            # Uncertainties
            uncertainties = []
            if series.frequency in ["monthly", "quarterly"]:
                uncertainties.append("Subject to revision in subsequent releases")
            
            return FREDSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="macro",
                source_type="fred",
                asset_scope=asset_scope,
                summary=summary[:300],
                raw_value=raw_value,
                evidence=evidence,
                confidence=confidence,
                confidence_factors=confidence_factors,
                directional_bias=directional_bias,
                time_horizon=time_horizon,
                novelty="new",
                staleness_policy=staleness_policy,
                uncertainties=uncertainties,
                timestamp_utc=now.isoformat(),
                forward_implication=forward_implication,
            )
            
        except Exception as e:
            logger.error(f"Error converting FRED observation to signal: {e}")
            return None
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    async def scan_series(self, series: FREDSeries) -> Optional[FREDSignal]:
        """
        Scan a specific FRED series and generate a forward-looking signal.
        
        Args:
            series: FREDSeries enum to scan
            
        Returns:
            FREDSignal or None if no data
        """
        if not self.is_configured:
            return None
        
        logger.debug(f"Scanning FRED series: {series.series_id}")
        
        data = await self._fetch_series(series.series_id, limit=2)
        
        if not data or not data.get("observations"):
            return None
        
        observations = data["observations"]
        current = observations[0] if observations else None
        prior = observations[1] if len(observations) > 1 else None
        
        if not current:
            return None
        
        return self._observation_to_signal(series, current, prior)
    
    async def scan_all_series(self) -> List[FREDSignal]:
        """
        Scan all configured FRED series.
        
        Returns:
            List of FREDSignal objects with forward-looking context
        """
        if not self.is_configured:
            logger.warning("FRED not configured")
            return []
        
        logger.info("Scanning all FRED series with forward-looking context")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals = []
        
        # Priority series for trading signals
        priority_series = [
            FREDSeries.DGS10,
            FREDSeries.DGS2,
            FREDSeries.T10Y2Y,
            FREDSeries.FEDFUNDS,
            FREDSeries.UNRATE,
            FREDSeries.CPIAUCSL,
            FREDSeries.GDP,
        ]
        
        for series in priority_series:
            try:
                signal = await self.scan_series(series)
                if signal:
                    signals.append(signal)
                    logger.debug(f"Generated signal for {series.series_id}")
            except Exception as e:
                logger.error(f"Error scanning {series.series_id}: {e}")
                continue
        
        logger.info(f"FRED scan complete: {len(signals)} signals generated")
        return signals
    
    async def scan_yields(self) -> List[FREDSignal]:
        """
        Scan treasury yield series specifically.
        
        Returns:
            List of yield-related FREDSignal objects
        """
        if not self.is_configured:
            return []
        
        logger.info("Scanning treasury yields")
        
        signals = []
        yield_series = [FREDSeries.DGS10, FREDSeries.DGS2, FREDSeries.T10Y2Y]
        
        for series in yield_series:
            signal = await self.scan_series(series)
            if signal:
                signals.append(signal)
        
        return signals
    
    async def scan_inflation(self) -> List[FREDSignal]:
        """
        Scan inflation-related series.
        
        Returns:
            List of inflation FREDSignal objects
        """
        if not self.is_configured:
            return []
        
        logger.info("Scanning inflation data")
        
        signals = []
        inflation_series = [FREDSeries.CPIAUCSL, FREDSeries.PCEPI]
        
        for series in inflation_series:
            signal = await self.scan_series(series)
            if signal:
                signals.append(signal)
        
        return signals
    
    async def scan_employment(self) -> List[FREDSignal]:
        """
        Scan employment-related series.
        
        Returns:
            List of employment FREDSignal objects
        """
        if not self.is_configured:
            return []
        
        logger.info("Scanning employment data")
        
        signals = []
        employment_series = [FREDSeries.UNRATE, FREDSeries.ICSA]
        
        for series in employment_series:
            signal = await self.scan_series(series)
            if signal:
                signals.append(signal)
        
        return signals


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def scan_fred() -> List[Dict[str, Any]]:
    """
    Convenience function to run a full FRED scan.
    
    Returns:
        List of signal dictionaries ready for storage
    """
    scanner = FREDScanner()
    signals = await scanner.scan_all_series()
    return [s.to_dict() for s in signals]


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = FREDScanner()
        
        print("\n" + "="*60)
        print("FRED SCANNER TEST (Forward-Looking)")
        print("="*60)
        
        # Show temporal context
        context = scanner.temporal_context.to_dict()
        print(f"\nReference Time: {context['reference_time']}")
        print(f"End of Quarter: {context['key_dates']['end_of_quarter']}")
        
        print("\n" + "-"*60)
        print("Series Tracked:")
        print("-"*60)
        
        for series in FREDSeries:
            print(f"  {series.series_id}: {series.display_name}")
            print(f"    Frequency: {series.frequency}")
            print(f"    Forward Relevance: {series.forward_relevance}")
        
        if scanner.is_configured:
            print("\n" + "-"*60)
            print("Testing API (requires FRED_API_KEY)...")
            print("-"*60)
            
            signals = await scanner.scan_all_series()
            print(f"\nGot {len(signals)} signals")
            
            for sig in signals[:3]:
                print(f"\n  Series: {sig.raw_value.get('unit', 'N/A')}")
                print(f"  Summary: {sig.summary[:100]}...")
                print(f"  Bias: {sig.directional_bias}")
                print(f"  Forward: {sig.forward_implication[:80]}...")
    
    asyncio.run(test())
