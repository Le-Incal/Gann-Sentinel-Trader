"""
Gann Sentinel Trader - FRED Scanner
Retrieves economic data from the Federal Reserve Economic Data API.
"""

import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from config import Config
from models.signals import Signal, SignalType, SignalSource, DirectionalBias, TimeHorizon, AssetScope, RawValue, Evidence

logger = logging.getLogger(__name__)


# FRED series metadata
FRED_SERIES_INFO = {
    "DGS10": {
        "name": "10-Year Treasury Yield",
        "description": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        "frequency": "daily",
        "unit": "percent",
        "impact": "Rising yields typically negative for growth stocks, positive for financials"
    },
    "DGS2": {
        "name": "2-Year Treasury Yield",
        "description": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
        "frequency": "daily",
        "unit": "percent",
        "impact": "Sensitive to Fed policy expectations"
    },
    "T10Y2Y": {
        "name": "10Y-2Y Treasury Spread",
        "description": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
        "frequency": "daily",
        "unit": "percent",
        "impact": "Negative spread (inversion) historically precedes recessions"
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "description": "Civilian Unemployment Rate",
        "frequency": "monthly",
        "unit": "percent",
        "impact": "Rising unemployment is bearish, but may lead to Fed easing"
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index",
        "description": "Consumer Price Index for All Urban Consumers: All Items",
        "frequency": "monthly",
        "unit": "index",
        "impact": "High inflation leads to tighter Fed policy"
    },
    "GDP": {
        "name": "Gross Domestic Product",
        "description": "Gross Domestic Product",
        "frequency": "quarterly",
        "unit": "billions_usd",
        "impact": "Strong GDP growth is generally bullish"
    },
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "description": "Federal Funds Effective Rate",
        "frequency": "daily",
        "unit": "percent",
        "impact": "Higher rates tighten financial conditions"
    }
}


class FREDScanner:
    """
    Retrieves economic data from FRED (Federal Reserve Economic Data).
    """
    
    def __init__(self):
        """Initialize FRED scanner."""
        self.api_key = Config.FRED_API_KEY
        self.base_url = Config.FRED_BASE_URL
        self.series_list = Config.FRED_SERIES
        
        if not self.api_key:
            logger.warning("FRED_API_KEY not configured - FRED scanner disabled")
    
    async def _fetch_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Fetch data for a FRED series."""
        if not self.api_key:
            raise ValueError("FRED_API_KEY not configured")
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit
        }
        
        if observation_start:
            params["observation_start"] = observation_start
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/series/observations",
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    async def scan_all_series(self) -> List[Signal]:
        """
        Scan all configured FRED series.
        
        Returns:
            List of Signal objects with latest data
        """
        if not self.api_key:
            logger.warning("FRED scanner not configured - returning empty signals")
            return []
        
        signals = []
        
        for series_id in self.series_list:
            try:
                signal = await self._scan_series(series_id)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning FRED series {series_id}: {e}")
        
        return signals
    
    async def _scan_series(self, series_id: str) -> Optional[Signal]:
        """Scan a single FRED series."""
        try:
            data = await self._fetch_series(series_id, limit=5)
            observations = data.get("observations", [])
            
            if not observations:
                return None
            
            # Get latest and previous values
            latest = observations[0]
            previous = observations[1] if len(observations) > 1 else None
            
            latest_value = float(latest["value"]) if latest["value"] != "." else None
            previous_value = float(previous["value"]) if previous and previous["value"] != "." else None
            
            if latest_value is None:
                return None
            
            # Calculate change
            change = None
            if previous_value is not None:
                change = latest_value - previous_value
            
            # Get series metadata
            series_info = FRED_SERIES_INFO.get(series_id, {})
            
            # Determine directional bias based on series type
            direction = self._determine_direction(series_id, latest_value, change)
            
            # Determine staleness based on frequency
            frequency = series_info.get("frequency", "daily")
            staleness = self._frequency_to_staleness(frequency)
            
            signal = Signal(
                signal_type=SignalType.MACRO,
                source=SignalSource.FRED,
                asset_scope=AssetScope(
                    macro_regions=["US"],
                    asset_classes=["EQUITY", "FIXED_INCOME"]
                ),
                summary=f"{series_info.get('name', series_id)}: {latest_value:.2f} (change: {change:+.2f})" if change else f"{series_info.get('name', series_id)}: {latest_value:.2f}",
                raw_value=RawValue(
                    type="rate" if "percent" in series_info.get("unit", "") else "index",
                    value=latest_value,
                    unit=series_info.get("unit", ""),
                    prior_value=previous_value,
                    change=change,
                    change_period=self._get_change_period(frequency)
                ),
                evidence=[
                    Evidence(
                        source=f"https://fred.stlouisfed.org/series/{series_id}",
                        source_tier="official",
                        excerpt=f"{series_info.get('description', series_id)}: {latest_value}",
                        timestamp_utc=datetime.fromisoformat(latest["date"].replace("Z", "+00:00")) if "T" in latest["date"] else datetime.strptime(latest["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc),
                        url=f"https://fred.stlouisfed.org/series/{series_id}"
                    )
                ],
                confidence=0.95,  # Official government data
                directional_bias=direction,
                time_horizon=TimeHorizon.WEEKS if frequency == "daily" else TimeHorizon.MONTHS,
                novelty="new",
                timestamp_utc=datetime.now(timezone.utc),
                staleness_seconds=staleness,
                uncertainties=[
                    "Data may be revised in subsequent releases",
                    series_info.get("impact", "Market impact depends on context")
                ]
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return None
    
    async def get_treasury_spread(self) -> Optional[Signal]:
        """Get the 10Y-2Y Treasury spread specifically."""
        return await self._scan_series("T10Y2Y")
    
    async def get_unemployment(self) -> Optional[Signal]:
        """Get the latest unemployment rate."""
        return await self._scan_series("UNRATE")
    
    async def get_fed_funds_rate(self) -> Optional[Signal]:
        """Get the current Fed Funds rate."""
        return await self._scan_series("FEDFUNDS")
    
    async def get_inflation(self) -> Optional[Signal]:
        """Get the latest CPI data."""
        return await self._scan_series("CPIAUCSL")
    
    def _determine_direction(
        self,
        series_id: str,
        value: float,
        change: Optional[float]
    ) -> DirectionalBias:
        """Determine directional bias based on series and change."""
        if change is None:
            return DirectionalBias.UNCLEAR
        
        # Different series have different implications
        if series_id in ["DGS10", "DGS2", "FEDFUNDS"]:
            # Rising rates generally negative for equities
            if change > 0.1:
                return DirectionalBias.NEGATIVE
            elif change < -0.1:
                return DirectionalBias.POSITIVE
            else:
                return DirectionalBias.UNCLEAR
        
        elif series_id == "T10Y2Y":
            # Spread - negative (inverted) is bearish
            if value < 0:
                return DirectionalBias.NEGATIVE
            elif value > 0.5:
                return DirectionalBias.POSITIVE
            else:
                return DirectionalBias.MIXED
        
        elif series_id == "UNRATE":
            # Rising unemployment is bearish
            if change > 0.2:
                return DirectionalBias.NEGATIVE
            elif change < -0.1:
                return DirectionalBias.POSITIVE
            else:
                return DirectionalBias.UNCLEAR
        
        elif series_id == "GDP":
            # Positive GDP growth is bullish
            if change > 0:
                return DirectionalBias.POSITIVE
            elif change < 0:
                return DirectionalBias.NEGATIVE
            else:
                return DirectionalBias.UNCLEAR
        
        elif series_id == "CPIAUCSL":
            # High/rising inflation is generally negative (leads to tighter policy)
            # This is a level, so look at YoY change would be better
            return DirectionalBias.UNCLEAR
        
        return DirectionalBias.UNCLEAR
    
    def _frequency_to_staleness(self, frequency: str) -> int:
        """Convert frequency to staleness in seconds."""
        staleness_map = {
            "daily": 86400,       # 24 hours
            "weekly": 604800,     # 7 days
            "monthly": 2592000,   # 30 days
            "quarterly": 7776000  # 90 days
        }
        return staleness_map.get(frequency, Config.STALENESS_MACRO)
    
    def _get_change_period(self, frequency: str) -> str:
        """Get change period description."""
        period_map = {
            "daily": "1d",
            "weekly": "1w",
            "monthly": "1m",
            "quarterly": "1q"
        }
        return period_map.get(frequency, "unknown")
