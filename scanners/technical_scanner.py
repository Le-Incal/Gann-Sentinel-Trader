"""
Gann Sentinel Trader - Technical Scanner v2.0
Disciplined Chart Trading Analysis via Alpaca Market Data API.

Based on LLM Chart Trading Training principles:
- Market state classification BEFORE indicators
- Scenario-based reasoning (primary + alternate)
- Risk-first with R-multiple calculation
- "No Trade" as a valid, high-quality output
- Liquidity sweep detection

Version: 2.0.0
Last Updated: January 2026
"""

import os
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

# Try to import pandas and technical analysis libraries
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - some features disabled")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available - using basic calculations")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False
    DataFeed = None
    logger.warning("alpaca-py data client not available")


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class MarketState(Enum):
    """
    Primary market state classification.
    MUST be determined before any indicator analysis.
    """
    TRENDING = "trending"          # HH/HL (bull) or LH/LL (bear) structure
    RANGE_BOUND = "range_bound"    # Defined highs/lows, mean reverting
    TRANSITIONAL = "transitional"  # Structure unclear, reduced edge


class TrendBias(Enum):
    """Directional bias within market state."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Verdict(Enum):
    """
    Final trading verdict - determines if trade hypothesis is allowed.
    "No Trade" is a high-quality outcome, not a failure.
    """
    NO_TRADE = "no_trade"              # Insufficient edge, stand aside
    ANALYZE_ONLY = "analyze_only"      # Structure unclear, watch only
    HYPOTHESIS_ALLOWED = "hypothesis"  # Clear setup, trade permitted
    ESCALATE = "escalate"              # Missing data, need human review


class SignalStrength(Enum):
    """Technical signal strength."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    LEAN_BUY = "lean_buy"
    NEUTRAL = "neutral"
    LEAN_SELL = "lean_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Scenario:
    """A trading scenario with confirmation and invalidation triggers."""
    name: str
    thesis: str
    confirm: List[str]      # Conditions that confirm this scenario
    invalidate: List[str]   # Conditions that invalidate this scenario
    probability: float      # 0.0 to 1.0


@dataclass
class TradeHypothesis:
    """
    A potential trade setup with risk parameters.
    Only generated when verdict = HYPOTHESIS_ALLOWED.
    """
    allow_trade: bool
    side: str               # "long" or "short"
    entry_zone: Dict[str, Any]
    invalidation: Dict[str, Any]
    targets: List[Dict[str, Any]]
    expected_r: float       # R-multiple (reward / risk)
    position_sizing: Dict[str, Any]
    risk_notes: List[str]


@dataclass
class LiquiditySweep:
    """
    A liquidity sweep event - price breaks a level, grabs stops, then reverses.
    These are often FALSE breaks and signal the opposite direction.
    """
    timestamp: str
    level: float
    direction: str          # "down" (swept support) or "up" (swept resistance)
    reclaimed: bool         # Did price reclaim the level?
    significance: str       # "major" or "minor"
    note: str


@dataclass
class MarketStateReport:
    """Market state classification with evidence."""
    state: MarketState
    bias: TrendBias
    confidence: str         # "low", "medium", "high"
    evidence: List[str]
    swings: List[Dict[str, Any]]  # Detected swing highs/lows


@dataclass 
class TechnicalSignal:
    """
    Complete technical analysis signal with scenario-based reasoning.
    """
    signal_id: str
    ticker: str
    timestamp_utc: str
    timeframe: str
    lookback_years: float
    
    # Price context
    current_price: float
    price_change_pct: float
    atr: Optional[float]
    atr_pct: Optional[float]
    
    # MARKET STATE (Determined FIRST)
    market_state: MarketStateReport
    
    # Structure
    trend_channel: Optional[Dict[str, Any]]
    support_levels: List[float]
    resistance_levels: List[float]
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    
    # Events
    liquidity_sweeps: List[LiquiditySweep]
    
    # Technical indicators (Secondary to structure)
    indicators: Dict[str, Any]
    
    # Volume
    volume_analysis: Dict[str, Any]
    
    # SCENARIOS (Primary + Alternate)
    primary_scenario: Optional[Scenario]
    alternate_scenario: Optional[Scenario]
    
    # TRADE HYPOTHESIS (Only if verdict allows)
    trade_hypothesis: Optional[TradeHypothesis]
    
    # FINAL VERDICT
    verdict: Verdict
    verdict_reasons: List[str]
    
    # Legacy fields for compatibility
    signal_strength: str
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "signal_type": "technical",
            "category": "technical",
            "ticker": self.ticker,
            "timestamp_utc": self.timestamp_utc,
            "timeframe": self.timeframe,
            "lookback_years": self.lookback_years,
            
            # Price
            "current_price": self.current_price,
            "price_change_pct": self.price_change_pct,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            
            # Market State
            "market_state": {
                "state": self.market_state.state.value,
                "bias": self.market_state.bias.value,
                "confidence": self.market_state.confidence,
                "evidence": self.market_state.evidence,
            },
            
            # Structure
            "trend_channel": self.trend_channel,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "nearest_support": self.nearest_support,
            "nearest_resistance": self.nearest_resistance,
            
            # Events
            "liquidity_sweeps": [
                {
                    "timestamp": ls.timestamp,
                    "level": ls.level,
                    "direction": ls.direction,
                    "reclaimed": ls.reclaimed,
                    "significance": ls.significance,
                    "note": ls.note,
                }
                for ls in self.liquidity_sweeps
            ],
            
            # Indicators
            "indicators": self.indicators,
            "volume_analysis": self.volume_analysis,
            
            # Scenarios
            "primary_scenario": {
                "name": self.primary_scenario.name,
                "thesis": self.primary_scenario.thesis,
                "confirm": self.primary_scenario.confirm,
                "invalidate": self.primary_scenario.invalidate,
                "probability": self.primary_scenario.probability,
            } if self.primary_scenario else None,
            
            "alternate_scenario": {
                "name": self.alternate_scenario.name,
                "thesis": self.alternate_scenario.thesis,
                "confirm": self.alternate_scenario.confirm,
                "invalidate": self.alternate_scenario.invalidate,
                "probability": self.alternate_scenario.probability,
            } if self.alternate_scenario else None,
            
            # Trade Hypothesis
            "trade_hypothesis": {
                "allow_trade": self.trade_hypothesis.allow_trade,
                "side": self.trade_hypothesis.side,
                "entry_zone": self.trade_hypothesis.entry_zone,
                "invalidation": self.trade_hypothesis.invalidation,
                "targets": self.trade_hypothesis.targets,
                "expected_r": self.trade_hypothesis.expected_r,
                "risk_notes": self.trade_hypothesis.risk_notes,
            } if self.trade_hypothesis else None,
            
            # Verdict
            "verdict": self.verdict.value,
            "verdict_reasons": self.verdict_reasons,
            
            # Legacy
            "signal_strength": self.signal_strength,
            "summary": self.summary,
            "directional_bias": self._get_directional_bias(),
        }
    
    def _get_directional_bias(self) -> str:
        """Convert to simple directional bias for compatibility."""
        if self.signal_strength in ["strong_buy", "buy", "lean_buy"]:
            return "positive"
        elif self.signal_strength in ["strong_sell", "sell", "lean_sell"]:
            return "negative"
        return "mixed"


# =============================================================================
# TECHNICAL SCANNER
# =============================================================================

class TechnicalScanner:
    """
    Disciplined Chart Trading Scanner.
    
    Core Principles:
    1. Classify market state BEFORE analyzing indicators
    2. Reason in scenarios and probabilities
    3. "No Trade" is a valid, high-quality output
    4. All trades must have defined invalidation and R-multiple
    """
    
    # Minimum R-multiple required for trade hypothesis
    MIN_R_MULTIPLE = 1.8
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """Initialize the technical scanner."""
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        # Diagnostic logging
        logger.info(f"TechnicalScanner init - ALPACA_DATA_AVAILABLE: {ALPACA_DATA_AVAILABLE}")
        logger.info(f"TechnicalScanner init - API_KEY present: {bool(self.api_key)}")
        logger.info(f"TechnicalScanner init - SECRET_KEY present: {bool(self.secret_key)}")
        
        self.client = None
        if ALPACA_DATA_AVAILABLE and self.api_key and self.secret_key:
            try:
                self.client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                )
                logger.info("Alpaca Market Data client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not ALPACA_DATA_AVAILABLE:
                logger.warning("alpaca-py not installed - Technical Scanner disabled")
            elif not self.api_key:
                logger.warning("ALPACA_API_KEY not set - Technical Scanner disabled")
            elif not self.secret_key:
                logger.warning("ALPACA_SECRET_KEY not set - Technical Scanner disabled")
        
        self.last_error: Optional[str] = None
    
    @property
    def is_configured(self) -> bool:
        """Check if scanner is properly configured."""
        return self.client is not None
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def get_historical_bars(
        self,
        ticker: str,
        timeframe: str = "1D",
        lookback_years: float = 1.0,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Alpaca.
        
        Supported Timeframes:
            - "1M"    : Monthly bars (5-10 year secular trends)
            - "1W"    : Weekly bars (1-5 year major trends) 
            - "1D"    : Daily bars (weeks-months analysis)
            - "4H"    : 4-hour bars
            - "1H"    : Hourly bars
            - "15min" : 15-minute bars
        """
        if not self.is_configured:
            self.last_error = "Alpaca client not configured"
            return None
        
        if not PANDAS_AVAILABLE:
            self.last_error = "pandas not available"
            return None
        
        try:
            tf_map = {
                "1M": TimeFrame(1, TimeFrameUnit.Month),
                "1W": TimeFrame(1, TimeFrameUnit.Week),
                "1D": TimeFrame(1, TimeFrameUnit.Day),
                "4H": TimeFrame(4, TimeFrameUnit.Hour),
                "1H": TimeFrame(1, TimeFrameUnit.Hour),
                "15min": TimeFrame(15, TimeFrameUnit.Minute),
                "5min": TimeFrame(5, TimeFrameUnit.Minute),
                "1min": TimeFrame(1, TimeFrameUnit.Minute),
            }
            
            timeframe_obj = tf_map.get(timeframe)
            if not timeframe_obj:
                self.last_error = f"Invalid timeframe: {timeframe}"
                return None
            
            end = datetime.now(timezone.utc)
            days = int(lookback_years * 365)
            start = end - timedelta(days=days)
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe_obj,
                start=start,
                end=end,
                feed=DataFeed.IEX if DataFeed else None,  # Use free IEX data feed
            )
            
            logger.debug(f"Fetching bars: {ticker}, {timeframe}, start={start}, end={end}, feed=IEX")
            bars = self.client.get_stock_bars(request)
            
            if ticker not in bars.data or not bars.data[ticker]:
                self.last_error = f"No data returned for {ticker}"
                return None
            
            data = []
            for bar in bars.data[ticker]:
                data.append({
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                })
            
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Fetched {len(df)} bars for {ticker} ({timeframe}, {lookback_years}y)")
            return df
            
        except Exception as e:
            self.last_error = f"Error fetching bars for {ticker}: {str(e)}"
            logger.error(self.last_error)
            # Log full traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    # =========================================================================
    # MARKET STATE CLASSIFICATION (DO THIS FIRST)
    # =========================================================================
    
    def classify_market_state(self, df: pd.DataFrame) -> MarketStateReport:
        """
        Classify market state using price structure ONLY.
        
        This MUST be done before any indicator analysis.
        
        States:
        - TRENDING: Clear HH/HL (bull) or LH/LL (bear)
        - RANGE_BOUND: Defined highs and lows, price oscillating
        - TRANSITIONAL: Structure unclear, expanding volatility
        """
        if len(df) < 20:
            return MarketStateReport(
                state=MarketState.TRANSITIONAL,
                bias=TrendBias.NEUTRAL,
                confidence="low",
                evidence=["Insufficient data for classification"],
                swings=[],
            )
        
        # Detect swing highs and lows using ATR-based threshold
        swings = self._detect_swings(df)
        evidence = []
        
        if len(swings) < 4:
            return MarketStateReport(
                state=MarketState.TRANSITIONAL,
                bias=TrendBias.NEUTRAL,
                confidence="low",
                evidence=["Insufficient swing points for classification"],
                swings=swings,
            )
        
        # Analyze swing structure
        swing_highs = [s for s in swings if s["type"] == "high"]
        swing_lows = [s for s in swings if s["type"] == "low"]
        
        # Check for trending structure
        hh_count = 0  # Higher highs
        hl_count = 0  # Higher lows
        lh_count = 0  # Lower highs
        ll_count = 0  # Lower lows
        
        for i in range(1, len(swing_highs)):
            if swing_highs[i]["price"] > swing_highs[i-1]["price"]:
                hh_count += 1
            else:
                lh_count += 1
        
        for i in range(1, len(swing_lows)):
            if swing_lows[i]["price"] > swing_lows[i-1]["price"]:
                hl_count += 1
            else:
                ll_count += 1
        
        total_high_swings = max(len(swing_highs) - 1, 1)
        total_low_swings = max(len(swing_lows) - 1, 1)
        
        hh_pct = hh_count / total_high_swings
        hl_pct = hl_count / total_low_swings
        lh_pct = lh_count / total_high_swings
        ll_pct = ll_count / total_low_swings
        
        # Determine state
        if hh_pct >= 0.6 and hl_pct >= 0.6:
            state = MarketState.TRENDING
            bias = TrendBias.BULLISH
            confidence = "high" if (hh_pct >= 0.75 and hl_pct >= 0.75) else "medium"
            evidence.append(f"Higher highs: {hh_pct:.0%} of swings")
            evidence.append(f"Higher lows: {hl_pct:.0%} of swings")
            evidence.append("Uptrend structure confirmed")
            
        elif lh_pct >= 0.6 and ll_pct >= 0.6:
            state = MarketState.TRENDING
            bias = TrendBias.BEARISH
            confidence = "high" if (lh_pct >= 0.75 and ll_pct >= 0.75) else "medium"
            evidence.append(f"Lower highs: {lh_pct:.0%} of swings")
            evidence.append(f"Lower lows: {ll_pct:.0%} of swings")
            evidence.append("Downtrend structure confirmed")
            
        else:
            # Check for range-bound
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                high_range = max(s["price"] for s in swing_highs) - min(s["price"] for s in swing_highs)
                low_range = max(s["price"] for s in swing_lows) - min(s["price"] for s in swing_lows)
                
                avg_high = statistics.mean(s["price"] for s in swing_highs)
                avg_low = statistics.mean(s["price"] for s in swing_lows)
                
                high_variance = high_range / avg_high if avg_high > 0 else 1
                low_variance = low_range / avg_low if avg_low > 0 else 1
                
                if high_variance < 0.10 and low_variance < 0.10:
                    state = MarketState.RANGE_BOUND
                    bias = TrendBias.NEUTRAL
                    confidence = "medium"
                    evidence.append(f"Swing highs clustered within {high_variance:.1%}")
                    evidence.append(f"Swing lows clustered within {low_variance:.1%}")
                    evidence.append("Range-bound structure - mean reversion likely")
                else:
                    state = MarketState.TRANSITIONAL
                    bias = TrendBias.NEUTRAL
                    confidence = "low"
                    evidence.append("Mixed swing structure")
                    evidence.append("No clear trend or range")
                    evidence.append("Reduced edge - caution advised")
            else:
                state = MarketState.TRANSITIONAL
                bias = TrendBias.NEUTRAL
                confidence = "low"
                evidence.append("Insufficient structure for classification")
        
        return MarketStateReport(
            state=state,
            bias=bias,
            confidence=confidence,
            evidence=evidence,
            swings=swings,
        )
    
    def _detect_swings(
        self,
        df: pd.DataFrame,
        atr_multiplier: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect swing highs and lows using ATR-based threshold.
        
        A swing high requires price to move down by ATR * multiplier
        A swing low requires price to move up by ATR * multiplier
        """
        swings = []
        
        if len(df) < 14:
            return swings
        
        # Calculate ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        threshold = atr * atr_multiplier
        
        # Simple swing detection
        highs = df["high"].values
        lows = df["low"].values
        timestamps = df.index.tolist()
        
        last_swing_type = None
        last_swing_price = None
        last_swing_idx = 0
        
        for i in range(2, len(df) - 2):
            current_threshold = threshold.iloc[i] if not pd.isna(threshold.iloc[i]) else highs[i] * 0.02
            
            # Check for swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                if last_swing_type != "high":
                    if last_swing_price is None or (highs[i] - last_swing_price) > current_threshold:
                        swings.append({
                            "type": "high",
                            "price": highs[i],
                            "timestamp": str(timestamps[i]),
                            "index": i,
                        })
                        last_swing_type = "high"
                        last_swing_price = highs[i]
                        last_swing_idx = i
                elif highs[i] > last_swing_price:
                    # Update last swing high if this is higher
                    if swings and swings[-1]["type"] == "high":
                        swings[-1] = {
                            "type": "high",
                            "price": highs[i],
                            "timestamp": str(timestamps[i]),
                            "index": i,
                        }
                        last_swing_price = highs[i]
            
            # Check for swing low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                if last_swing_type != "low":
                    if last_swing_price is None or (last_swing_price - lows[i]) > current_threshold:
                        swings.append({
                            "type": "low",
                            "price": lows[i],
                            "timestamp": str(timestamps[i]),
                            "index": i,
                        })
                        last_swing_type = "low"
                        last_swing_price = lows[i]
                        last_swing_idx = i
                elif lows[i] < last_swing_price:
                    # Update last swing low if this is lower
                    if swings and swings[-1]["type"] == "low":
                        swings[-1] = {
                            "type": "low",
                            "price": lows[i],
                            "timestamp": str(timestamps[i]),
                            "index": i,
                        }
                        last_swing_price = lows[i]
        
        return swings
    
    # =========================================================================
    # LIQUIDITY SWEEP DETECTION
    # =========================================================================
    
    def detect_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        support_levels: List[float],
        resistance_levels: List[float],
        tolerance_pct: float = 0.005,
    ) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps - price breaks a level then reverses.
        
        A sweep is:
        1. Price breaks below support (or above resistance)
        2. Price then closes back above support (or below resistance)
        3. This is a FALSE BREAK - signals opposite direction
        
        These are critical because they trap traders on the wrong side.
        """
        sweeps = []
        
        if len(df) < 5:
            return sweeps
        
        closes = df["close"].values
        lows = df["low"].values
        highs = df["high"].values
        timestamps = df.index.tolist()
        
        # Check for support sweeps (break below then reclaim)
        for support in support_levels:
            tolerance = support * tolerance_pct
            
            for i in range(2, len(df)):
                # Look for: low breaks support, but close is above
                if lows[i] < (support - tolerance) and closes[i] > support:
                    # Check if previous bar was above support
                    if closes[i-1] > support:
                        # This is a sweep!
                        sweeps.append(LiquiditySweep(
                            timestamp=str(timestamps[i]),
                            level=support,
                            direction="down",
                            reclaimed=True,
                            significance="major" if support == support_levels[0] else "minor",
                            note=f"Swept ${support:.2f} support, closed back above - BULLISH reversal signal",
                        ))
        
        # Check for resistance sweeps (break above then reject)
        for resistance in resistance_levels:
            tolerance = resistance * tolerance_pct
            
            for i in range(2, len(df)):
                # Look for: high breaks resistance, but close is below
                if highs[i] > (resistance + tolerance) and closes[i] < resistance:
                    # Check if previous bar was below resistance
                    if closes[i-1] < resistance:
                        # This is a sweep!
                        sweeps.append(LiquiditySweep(
                            timestamp=str(timestamps[i]),
                            level=resistance,
                            direction="up",
                            reclaimed=False,
                            significance="major" if resistance == resistance_levels[0] else "minor",
                            note=f"Swept ${resistance:.2f} resistance, rejected back below - BEARISH reversal signal",
                        ))
        
        # Sort by timestamp, most recent first
        sweeps.sort(key=lambda x: x.timestamp, reverse=True)
        
        return sweeps[:5]  # Return most recent 5
    
    # =========================================================================
    # TECHNICAL INDICATORS (Secondary to Structure)
    # =========================================================================
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators - these are SECONDARY to price structure."""
        indicators = {}
        
        if len(df) < 20:
            return {"error": "Insufficient data for indicators"}
        
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            
            # Moving Averages
            indicators["sma_20"] = close.rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = close.rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            indicators["sma_200"] = close.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
            
            # Current price vs MAs
            current_price = close.iloc[-1]
            indicators["price_vs_sma20"] = (current_price / indicators["sma_20"] - 1) * 100 if indicators["sma_20"] else None
            indicators["price_vs_sma50"] = (current_price / indicators["sma_50"] - 1) * 100 if indicators["sma_50"] else None
            
            # ATR (Critical for position sizing)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            indicators["atr_14"] = atr.iloc[-1]
            indicators["atr_pct"] = (indicators["atr_14"] / current_price * 100) if indicators["atr_14"] else None
            
            if PANDAS_TA_AVAILABLE:
                # RSI
                rsi = ta.rsi(close, length=14)
                indicators["rsi_14"] = rsi.iloc[-1] if rsi is not None and len(rsi) > 0 else None
                
                # MACD
                macd = ta.macd(close)
                if macd is not None and len(macd) > 0:
                    indicators["macd"] = macd.iloc[-1, 0]
                    indicators["macd_signal"] = macd.iloc[-1, 1]
                    indicators["macd_histogram"] = macd.iloc[-1, 2]
                
                # Bollinger Bands
                bbands = ta.bbands(close, length=20)
                if bbands is not None and len(bbands) > 0:
                    indicators["bb_upper"] = bbands.iloc[-1, 0]
                    indicators["bb_middle"] = bbands.iloc[-1, 1]
                    indicators["bb_lower"] = bbands.iloc[-1, 2]
                    bb_range = indicators["bb_upper"] - indicators["bb_lower"]
                    if bb_range > 0:
                        indicators["bb_position"] = (current_price - indicators["bb_lower"]) / bb_range
            else:
                # Basic RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi_14"] = rsi.iloc[-1]
            
            # Volume analysis
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            indicators["volume_20_avg"] = avg_volume
            indicators["volume_ratio"] = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # 52-week high/low
            lookback = min(252, len(df))
            indicators["high_52w"] = high.tail(lookback).max()
            indicators["low_52w"] = low.tail(lookback).min()
            indicators["pct_from_high"] = (current_price / indicators["high_52w"] - 1) * 100
            indicators["pct_from_low"] = (current_price / indicators["low_52w"] - 1) * 100
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            indicators["error"] = str(e)
        
        return indicators
    
    # =========================================================================
    # TREND CHANNEL DETECTION
    # =========================================================================
    
    def detect_trend_channel(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
    ) -> Optional[Dict[str, Any]]:
        """Detect trend channel using linear regression on highs and lows."""
        if len(df) < lookback:
            lookback = len(df)
        
        if lookback < 20:
            return None
        
        try:
            recent = df.tail(lookback)
            
            x = list(range(len(recent)))
            highs = recent["high"].values
            lows = recent["low"].values
            closes = recent["close"].values
            
            def linear_regression(x_vals, y_vals):
                n = len(x_vals)
                sum_x = sum(x_vals)
                sum_y = sum(y_vals)
                sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
                sum_xx = sum(x * x for x in x_vals)
                
                denom = (n * sum_xx - sum_x * sum_x)
                if denom == 0:
                    return 0, sum_y / n
                    
                slope = (n * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n
                
                return slope, intercept
            
            slope_high, intercept_high = linear_regression(x, highs)
            slope_low, intercept_low = linear_regression(x, lows)
            slope_mid, intercept_mid = linear_regression(x, closes)
            
            last_x = len(recent) - 1
            channel_upper = slope_high * last_x + intercept_high
            channel_lower = slope_low * last_x + intercept_low
            channel_mid = slope_mid * last_x + intercept_mid
            
            future_x = last_x + 10
            projected_upper = slope_high * future_x + intercept_high
            projected_lower = slope_low * future_x + intercept_low
            
            current_price = closes[-1]
            channel_range = channel_upper - channel_lower
            channel_position = (current_price - channel_lower) / channel_range if channel_range > 0 else 0.5
            
            avg_slope = (slope_high + slope_low + slope_mid) / 3
            daily_change_pct = (avg_slope / current_price) * 100 if current_price > 0 else 0
            
            if daily_change_pct > 0.1:
                trend = "uptrend"
            elif daily_change_pct < -0.1:
                trend = "downtrend"
            else:
                trend = "sideways"
            
            return {
                "channel_upper": round(channel_upper, 2),
                "channel_lower": round(channel_lower, 2),
                "channel_mid": round(channel_mid, 2),
                "channel_width_pct": round((channel_range / channel_mid) * 100, 2) if channel_mid > 0 else 0,
                "position_in_channel": round(channel_position, 3),
                "projected_upper_10bars": round(projected_upper, 2),
                "projected_lower_10bars": round(projected_lower, 2),
                "trend": trend,
                "slope_pct_per_bar": round(daily_change_pct, 4),
                "lookback_bars": lookback,
            }
            
        except Exception as e:
            logger.error(f"Error detecting trend channel: {e}")
            return None
    
    # =========================================================================
    # SUPPORT/RESISTANCE
    # =========================================================================
    
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        num_levels: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """Find key support and resistance levels."""
        if len(df) < 20:
            return [], []
        
        try:
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            current_price = closes[-1]
            
            resistance_candidates = []
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_candidates.append(highs[i])
            
            support_candidates = []
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_candidates.append(lows[i])
            
            def cluster_levels(levels: List[float], threshold_pct: float = 0.015) -> List[float]:
                if not levels:
                    return []
                
                sorted_levels = sorted(levels)
                clusters = []
                current_cluster = [sorted_levels[0]]
                
                for level in sorted_levels[1:]:
                    if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                        current_cluster.append(level)
                    else:
                        clusters.append(statistics.mean(current_cluster))
                        current_cluster = [level]
                
                clusters.append(statistics.mean(current_cluster))
                return clusters
            
            resistance_levels = cluster_levels(resistance_candidates)
            support_levels = cluster_levels(support_candidates)
            
            resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:num_levels]
            support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:num_levels]
            
            return (
                [round(s, 2) for s in support_levels],
                [round(r, 2) for r in resistance_levels],
            )
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return [], []
    
    # =========================================================================
    # VOLUME ANALYSIS
    # =========================================================================
    
    def analyze_volume(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if len(df) < lookback:
            return {"trend": "unknown", "ratio": 1.0}
        
        try:
            volume = df["volume"].tail(lookback)
            
            avg_volume = volume.mean()
            recent_volume = volume.tail(5).mean()
            current_volume = volume.iloc[-1]
            
            first_half = volume.head(lookback // 2).mean()
            second_half = volume.tail(lookback // 2).mean()
            
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "current_vs_avg": round(current_volume / avg_volume, 2) if avg_volume > 0 else 1,
                "recent_vs_avg": round(recent_volume / avg_volume, 2) if avg_volume > 0 else 1,
                "avg_volume": int(avg_volume),
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"trend": "unknown", "ratio": 1.0}
    
    # =========================================================================
    # SCENARIO BUILDER
    # =========================================================================
    
    def build_scenarios(
        self,
        market_state: MarketStateReport,
        trend_channel: Optional[Dict[str, Any]],
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float],
        liquidity_sweeps: List[LiquiditySweep],
    ) -> Tuple[Optional[Scenario], Optional[Scenario]]:
        """
        Build primary and alternate scenarios based on structure.
        
        Scenarios include:
        - Confirmation triggers (what price action confirms this)
        - Invalidation triggers (what price action invalidates this)
        - Probability estimate
        """
        primary = None
        alternate = None
        
        nearest_support = support_levels[0] if support_levels else current_price * 0.95
        nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.05
        
        channel_pos = trend_channel.get("position_in_channel", 0.5) if trend_channel else 0.5
        channel_lower = trend_channel.get("channel_lower", nearest_support) if trend_channel else nearest_support
        channel_upper = trend_channel.get("channel_upper", nearest_resistance) if trend_channel else nearest_resistance
        
        # Check for recent sweeps
        recent_bullish_sweep = any(
            s.direction == "down" and s.reclaimed 
            for s in liquidity_sweeps[:2]
        )
        recent_bearish_sweep = any(
            s.direction == "up" and not s.reclaimed 
            for s in liquidity_sweeps[:2]
        )
        
        # Build scenarios based on market state and structure
        if market_state.state == MarketState.TRENDING:
            if market_state.bias == TrendBias.BULLISH:
                if channel_pos < 0.3 or recent_bullish_sweep:
                    # Near support in uptrend - primary is continuation
                    primary = Scenario(
                        name="Trend Continuation from Support",
                        thesis=f"Price at channel support (${channel_lower:.0f}) in established uptrend. Pullback complete.",
                        confirm=[
                            f"Hold above ${channel_lower:.0f} on retest",
                            "Print higher low on daily timeframe",
                            f"Break above ${current_price * 1.02:.0f} with volume",
                        ],
                        invalidate=[
                            f"Daily close below ${channel_lower * 0.98:.0f}",
                            "Lower low forms on daily",
                        ],
                        probability=0.60 if recent_bullish_sweep else 0.55,
                    )
                    alternate = Scenario(
                        name="Trend Breakdown",
                        thesis="Support fails, trend structure breaks.",
                        confirm=[
                            f"Daily close below ${channel_lower:.0f}",
                            "Volume expansion on breakdown",
                        ],
                        invalidate=[
                            f"Price reclaims ${channel_lower * 1.02:.0f}",
                        ],
                        probability=0.25,
                    )
                elif channel_pos > 0.7:
                    # Near resistance in uptrend - caution
                    primary = Scenario(
                        name="Breakout to New Highs",
                        thesis=f"Price testing channel resistance (${channel_upper:.0f}). Breakout potential.",
                        confirm=[
                            f"Daily close above ${channel_upper:.0f}",
                            "Volume expansion on breakout",
                            "No immediate rejection",
                        ],
                        invalidate=[
                            f"Rejection and close below ${channel_upper * 0.97:.0f}",
                            "Bearish engulfing candle at resistance",
                        ],
                        probability=0.40,
                    )
                    alternate = Scenario(
                        name="Rejection at Resistance",
                        thesis="Resistance holds, pullback to support.",
                        confirm=[
                            f"Rejection pattern at ${channel_upper:.0f}",
                            f"Close below ${current_price * 0.98:.0f}",
                        ],
                        invalidate=[
                            f"Acceptance above ${channel_upper:.0f}",
                        ],
                        probability=0.45,
                    )
                else:
                    # Middle of channel
                    primary = Scenario(
                        name="Trend Continuation",
                        thesis="Uptrend intact, price in middle of channel.",
                        confirm=[
                            "Higher low forms",
                            f"Move toward ${channel_upper:.0f}",
                        ],
                        invalidate=[
                            f"Break below ${channel_lower:.0f}",
                        ],
                        probability=0.50,
                    )
                    alternate = Scenario(
                        name="Consolidation/Pullback",
                        thesis="Price consolidates before next move.",
                        confirm=["Range forms between current levels"],
                        invalidate=["Strong directional move either way"],
                        probability=0.35,
                    )
                    
            elif market_state.bias == TrendBias.BEARISH:
                if channel_pos > 0.7 or recent_bearish_sweep:
                    # Near resistance in downtrend - primary is continuation down
                    primary = Scenario(
                        name="Trend Continuation from Resistance",
                        thesis=f"Price at channel resistance (${channel_upper:.0f}) in established downtrend.",
                        confirm=[
                            f"Rejection at ${channel_upper:.0f}",
                            "Lower high forms on daily",
                            f"Break below ${current_price * 0.98:.0f}",
                        ],
                        invalidate=[
                            f"Daily close above ${channel_upper * 1.02:.0f}",
                            "Higher high forms on daily",
                        ],
                        probability=0.60 if recent_bearish_sweep else 0.55,
                    )
                    alternate = Scenario(
                        name="Trend Reversal",
                        thesis="Resistance breaks, trend reverses.",
                        confirm=[
                            f"Daily close above ${channel_upper:.0f}",
                            "Volume expansion on breakout",
                        ],
                        invalidate=[
                            f"Rejection back below ${channel_upper * 0.98:.0f}",
                        ],
                        probability=0.25,
                    )
                else:
                    primary = Scenario(
                        name="Downtrend Continuation",
                        thesis="Downtrend intact, expect lower prices.",
                        confirm=[
                            "Lower high forms",
                            f"Move toward ${channel_lower:.0f}",
                        ],
                        invalidate=[
                            f"Break above ${channel_upper:.0f}",
                        ],
                        probability=0.55,
                    )
                    alternate = Scenario(
                        name="Bounce/Relief Rally",
                        thesis="Oversold bounce before continuation.",
                        confirm=["Short-term higher low"],
                        invalidate=["New low with volume"],
                        probability=0.30,
                    )
                    
        elif market_state.state == MarketState.RANGE_BOUND:
            if channel_pos < 0.25:
                primary = Scenario(
                    name="Range Support Bounce",
                    thesis=f"Price at range low (${channel_lower:.0f}). Mean reversion expected.",
                    confirm=[
                        f"Hold above ${channel_lower:.0f}",
                        "Bullish reversal pattern",
                    ],
                    invalidate=[
                        f"Daily close below ${channel_lower * 0.98:.0f}",
                    ],
                    probability=0.55,
                )
                alternate = Scenario(
                    name="Range Breakdown",
                    thesis="Range support fails, new trend begins.",
                    confirm=[f"Close below ${channel_lower:.0f} with volume"],
                    invalidate=["Quick reclaim of range"],
                    probability=0.30,
                )
            elif channel_pos > 0.75:
                primary = Scenario(
                    name="Range Resistance Rejection",
                    thesis=f"Price at range high (${channel_upper:.0f}). Mean reversion expected.",
                    confirm=[
                        f"Rejection at ${channel_upper:.0f}",
                        "Bearish reversal pattern",
                    ],
                    invalidate=[
                        f"Daily close above ${channel_upper * 1.02:.0f}",
                    ],
                    probability=0.55,
                )
                alternate = Scenario(
                    name="Range Breakout",
                    thesis="Range resistance breaks, new uptrend begins.",
                    confirm=[f"Close above ${channel_upper:.0f} with volume"],
                    invalidate=["Quick rejection back into range"],
                    probability=0.30,
                )
            else:
                primary = Scenario(
                    name="Range Continuation",
                    thesis="Price mid-range, no edge.",
                    confirm=["Move toward range boundary"],
                    invalidate=["Breakout either direction"],
                    probability=0.50,
                )
                alternate = None
                
        else:  # TRANSITIONAL
            primary = Scenario(
                name="Structure Unclear",
                thesis="Market structure is transitional. No high-probability setup.",
                confirm=["Wait for structure to clarify"],
                invalidate=["N/A - observing only"],
                probability=0.50,
            )
            alternate = None
        
        return primary, alternate
    
    # =========================================================================
    # R-MULTIPLE CALCULATION
    # =========================================================================
    
    def calculate_r_multiple(
        self,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> float:
        """
        Calculate R-multiple (reward / risk).
        
        R = (Target - Entry) / (Entry - Stop) for longs
        R = (Entry - Target) / (Stop - Entry) for shorts
        """
        if entry_price == stop_price:
            return 0.0
        
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        
        return round(reward / risk, 2) if risk > 0 else 0.0
    
    # =========================================================================
    # TRADE HYPOTHESIS GENERATOR
    # =========================================================================
    
    def generate_trade_hypothesis(
        self,
        market_state: MarketStateReport,
        primary_scenario: Optional[Scenario],
        current_price: float,
        channel: Optional[Dict[str, Any]],
        support_levels: List[float],
        resistance_levels: List[float],
        atr: Optional[float],
    ) -> Tuple[Optional[TradeHypothesis], Verdict, List[str]]:
        """
        Generate trade hypothesis with risk parameters.
        
        Returns (hypothesis, verdict, reasons)
        
        Trade is only allowed when:
        1. Market state confidence >= medium
        2. Clear structural invalidation level exists
        3. R-multiple >= MIN_R_MULTIPLE (1.8)
        """
        reasons = []
        
        # Gate 1: Market state must be clear
        if market_state.state == MarketState.TRANSITIONAL:
            reasons.append("Market state is transitional - no clear structure")
            return None, Verdict.ANALYZE_ONLY, reasons
        
        if market_state.confidence == "low":
            reasons.append("Market state confidence is low")
            return None, Verdict.ANALYZE_ONLY, reasons
        
        if not primary_scenario:
            reasons.append("No primary scenario identified")
            return None, Verdict.ANALYZE_ONLY, reasons
        
        # Gate 2: Need structural levels
        if not support_levels and not resistance_levels:
            reasons.append("No structural support/resistance levels")
            return None, Verdict.ANALYZE_ONLY, reasons
        
        # Determine trade direction based on channel position and bias
        channel_pos = channel.get("position_in_channel", 0.5) if channel else 0.5
        channel_lower = channel.get("channel_lower") if channel else None
        channel_upper = channel.get("channel_upper") if channel else None
        
        # Calculate ATR-based stop buffer
        atr_buffer = atr * 0.5 if atr else current_price * 0.01
        
        # Generate hypothesis based on position
        hypothesis = None
        
        if market_state.bias == TrendBias.BULLISH or (
            market_state.state == MarketState.RANGE_BOUND and channel_pos < 0.3
        ):
            # LONG setup
            if channel_pos < 0.35:
                # Near support - LONG
                stop_level = (channel_lower - atr_buffer) if channel_lower else (
                    support_levels[0] * 0.98 if support_levels else current_price * 0.95
                )
                target_level = channel_upper if channel_upper else (
                    resistance_levels[0] if resistance_levels else current_price * 1.10
                )
                
                entry_zone_low = current_price * 0.99
                entry_zone_high = current_price * 1.01
                
                r_multiple = self.calculate_r_multiple(current_price, stop_level, target_level)
                
                if r_multiple < self.MIN_R_MULTIPLE:
                    reasons.append(f"R-multiple {r_multiple:.1f} below minimum {self.MIN_R_MULTIPLE}")
                    return None, Verdict.NO_TRADE, reasons
                
                hypothesis = TradeHypothesis(
                    allow_trade=True,
                    side="long",
                    entry_zone={
                        "low": round(entry_zone_low, 2),
                        "high": round(entry_zone_high, 2),
                        "logic": "Near channel support in uptrend/range low",
                    },
                    invalidation={
                        "level": round(stop_level, 2),
                        "logic": "Below channel support - structure invalidated",
                    },
                    targets=[
                        {"level": round(target_level, 2), "logic": "Channel resistance / range high"},
                    ],
                    expected_r=r_multiple,
                    position_sizing={
                        "method": "risk_fraction",
                        "risk_fraction": 0.01,  # 1% account risk
                    },
                    risk_notes=[
                        f"Stop: ${stop_level:.2f} ({((current_price - stop_level) / current_price * 100):.1f}% risk)",
                        f"Target: ${target_level:.2f} ({((target_level - current_price) / current_price * 100):.1f}% potential)",
                        f"R-Multiple: {r_multiple:.1f}",
                    ],
                )
                reasons.append(f"Long setup at channel support, R={r_multiple:.1f}")
                return hypothesis, Verdict.HYPOTHESIS_ALLOWED, reasons
                
            else:
                # Not at support - no edge
                reasons.append("Bullish bias but price not at support - wait for pullback")
                return None, Verdict.NO_TRADE, reasons
                
        elif market_state.bias == TrendBias.BEARISH or (
            market_state.state == MarketState.RANGE_BOUND and channel_pos > 0.7
        ):
            # SHORT setup
            if channel_pos > 0.65:
                # Near resistance - SHORT
                stop_level = (channel_upper + atr_buffer) if channel_upper else (
                    resistance_levels[0] * 1.02 if resistance_levels else current_price * 1.05
                )
                target_level = channel_lower if channel_lower else (
                    support_levels[0] if support_levels else current_price * 0.90
                )
                
                entry_zone_low = current_price * 0.99
                entry_zone_high = current_price * 1.01
                
                r_multiple = self.calculate_r_multiple(current_price, stop_level, target_level)
                
                if r_multiple < self.MIN_R_MULTIPLE:
                    reasons.append(f"R-multiple {r_multiple:.1f} below minimum {self.MIN_R_MULTIPLE}")
                    return None, Verdict.NO_TRADE, reasons
                
                hypothesis = TradeHypothesis(
                    allow_trade=True,
                    side="short",
                    entry_zone={
                        "low": round(entry_zone_low, 2),
                        "high": round(entry_zone_high, 2),
                        "logic": "Near channel resistance in downtrend/range high",
                    },
                    invalidation={
                        "level": round(stop_level, 2),
                        "logic": "Above channel resistance - structure invalidated",
                    },
                    targets=[
                        {"level": round(target_level, 2), "logic": "Channel support / range low"},
                    ],
                    expected_r=r_multiple,
                    position_sizing={
                        "method": "risk_fraction",
                        "risk_fraction": 0.01,
                    },
                    risk_notes=[
                        f"Stop: ${stop_level:.2f} ({((stop_level - current_price) / current_price * 100):.1f}% risk)",
                        f"Target: ${target_level:.2f} ({((current_price - target_level) / current_price * 100):.1f}% potential)",
                        f"R-Multiple: {r_multiple:.1f}",
                    ],
                )
                reasons.append(f"Short setup at channel resistance, R={r_multiple:.1f}")
                return hypothesis, Verdict.HYPOTHESIS_ALLOWED, reasons
                
            else:
                reasons.append("Bearish bias but price not at resistance - wait for rally")
                return None, Verdict.NO_TRADE, reasons
        
        else:
            # Neutral bias
            reasons.append("Neutral bias - no directional edge")
            return None, Verdict.ANALYZE_ONLY, reasons
    
    # =========================================================================
    # SIGNAL STRENGTH (Legacy Compatibility)
    # =========================================================================
    
    def generate_signal_strength(
        self,
        market_state: MarketStateReport,
        channel: Optional[Dict[str, Any]],
        trade_hypothesis: Optional[TradeHypothesis],
        verdict: Verdict,
    ) -> Tuple[str, str]:
        """Generate legacy signal strength and summary."""
        
        if verdict == Verdict.HYPOTHESIS_ALLOWED and trade_hypothesis:
            if trade_hypothesis.side == "long":
                if trade_hypothesis.expected_r >= 2.5:
                    signal = "strong_buy"
                else:
                    signal = "buy"
                summary = f"LONG: {trade_hypothesis.entry_zone.get('logic', '')}; R={trade_hypothesis.expected_r:.1f}"
            else:
                if trade_hypothesis.expected_r >= 2.5:
                    signal = "strong_sell"
                else:
                    signal = "sell"
                summary = f"SHORT: {trade_hypothesis.entry_zone.get('logic', '')}; R={trade_hypothesis.expected_r:.1f}"
                
        elif verdict == Verdict.NO_TRADE:
            signal = "neutral"
            summary = "NO TRADE: " + "; ".join(self._last_verdict_reasons[:2]) if hasattr(self, '_last_verdict_reasons') else "No edge identified"
            
        elif verdict == Verdict.ANALYZE_ONLY:
            # Lean based on market state
            if market_state.bias == TrendBias.BULLISH:
                signal = "lean_buy"
            elif market_state.bias == TrendBias.BEARISH:
                signal = "lean_sell"
            else:
                signal = "neutral"
            summary = f"WATCH: {market_state.state.value} market, {market_state.confidence} confidence"
            
        else:
            signal = "neutral"
            summary = "Insufficient data for analysis"
        
        return signal, summary
    
    # =========================================================================
    # MAIN SCAN METHOD
    # =========================================================================
    
    async def scan_ticker(
        self,
        ticker: str,
        timeframe: str = "1W",
        lookback_years: float = 5.0,
    ) -> Optional[TechnicalSignal]:
        """
        Run disciplined technical analysis on a ticker.
        
        Process:
        1. Fetch data
        2. Classify market state (FIRST)
        3. Map structure (channels, levels)
        4. Detect liquidity sweeps
        5. Calculate indicators (secondary)
        6. Build scenarios
        7. Generate trade hypothesis (or NO_TRADE)
        8. Return verdict
        """
        if not self.is_configured:
            self.last_error = "Technical scanner not configured"
            logger.warning(f"{self.last_error} - client={self.client}, api_key={bool(self.api_key)}, secret={bool(self.secret_key)}")
            return None
        
        logger.info(f"Technical scan starting: {ticker} ({timeframe}, {lookback_years}y)")
        
        # 1. Fetch data
        logger.info(f"Fetching historical bars for {ticker}...")
        df = self.get_historical_bars(ticker, timeframe, lookback_years)
        if df is None:
            logger.warning(f"No data returned for {ticker}. Last error: {self.last_error}")
            return None
        if len(df) < 20:
            logger.warning(f"Insufficient data for {ticker}: only {len(df)} bars (need 20+)")
            return None
        logger.info(f"Got {len(df)} bars for {ticker}")
        
        try:
            current_price = df["close"].iloc[-1]
            price_change = (current_price / df["close"].iloc[0] - 1) * 100
            
            # 2. CLASSIFY MARKET STATE (DO THIS FIRST)
            market_state = self.classify_market_state(df)
            logger.info(f"Market state: {market_state.state.value}, bias: {market_state.bias.value}, confidence: {market_state.confidence}")
            
            # 3. Map structure
            channel_lookback = len(df) if timeframe in ["1W", "1M"] else min(60, len(df))
            trend_channel = self.detect_trend_channel(df, lookback=channel_lookback)
            support_levels, resistance_levels = self.find_support_resistance(df)
            
            # 4. Detect liquidity sweeps
            liquidity_sweeps = self.detect_liquidity_sweeps(
                df, support_levels, resistance_levels
            )
            
            # 5. Calculate indicators (secondary to structure)
            indicators = self.calculate_indicators(df)
            volume_analysis = self.analyze_volume(df)
            
            atr = indicators.get("atr_14")
            atr_pct = indicators.get("atr_pct")
            
            # 6. Build scenarios
            primary_scenario, alternate_scenario = self.build_scenarios(
                market_state=market_state,
                trend_channel=trend_channel,
                current_price=current_price,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                liquidity_sweeps=liquidity_sweeps,
            )
            
            # 7. Generate trade hypothesis
            trade_hypothesis, verdict, verdict_reasons = self.generate_trade_hypothesis(
                market_state=market_state,
                primary_scenario=primary_scenario,
                current_price=current_price,
                channel=trend_channel,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                atr=atr,
            )
            
            self._last_verdict_reasons = verdict_reasons  # For legacy summary
            
            # 8. Generate legacy signal strength
            signal_strength, summary = self.generate_signal_strength(
                market_state=market_state,
                channel=trend_channel,
                trade_hypothesis=trade_hypothesis,
                verdict=verdict,
            )
            
            return TechnicalSignal(
                signal_id=str(uuid.uuid4()),
                ticker=ticker,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                timeframe=timeframe,
                lookback_years=lookback_years,
                current_price=round(current_price, 2),
                price_change_pct=round(price_change, 2),
                atr=round(atr, 2) if atr else None,
                atr_pct=round(atr_pct, 2) if atr_pct else None,
                market_state=market_state,
                trend_channel=trend_channel,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                nearest_support=support_levels[0] if support_levels else None,
                nearest_resistance=resistance_levels[0] if resistance_levels else None,
                liquidity_sweeps=liquidity_sweeps,
                indicators=indicators,
                volume_analysis=volume_analysis,
                primary_scenario=primary_scenario,
                alternate_scenario=alternate_scenario,
                trade_hypothesis=trade_hypothesis,
                verdict=verdict,
                verdict_reasons=verdict_reasons,
                signal_strength=signal_strength,
                summary=summary,
            )
            
        except Exception as e:
            self.last_error = f"Error in technical scan: {str(e)}"
            logger.error(self.last_error)
            import traceback
            traceback.print_exc()
            return None
    
    # =========================================================================
    # PRESET SCAN CONFIGURATIONS
    # =========================================================================
    
    async def scan_weekly_5y(self, ticker: str) -> Optional[TechnicalSignal]:
        """5-year weekly scan - Major trend channel analysis."""
        return await self.scan_ticker(ticker, "1W", 5.0)
    
    async def scan_weekly_2y(self, ticker: str) -> Optional[TechnicalSignal]:
        """2-year weekly scan - Medium-term trend analysis."""
        return await self.scan_ticker(ticker, "1W", 2.0)
    
    async def scan_daily_1y(self, ticker: str) -> Optional[TechnicalSignal]:
        """1-year daily scan - Standard swing trading analysis."""
        return await self.scan_ticker(ticker, "1D", 1.0)
    
    async def scan_daily_6m(self, ticker: str) -> Optional[TechnicalSignal]:
        """6-month daily scan - Short-term trend analysis."""
        return await self.scan_ticker(ticker, "1D", 0.5)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_technicals(
    ticker: str,
    timeframe: str = "1W",
    lookback_years: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Convenience function to run a technical scan."""
    scanner = TechnicalScanner()
    signal = await scanner.scan_ticker(ticker, timeframe, lookback_years)
    return signal.to_dict() if signal else None


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = TechnicalScanner()
        
        print("\n" + "="*70)
        print("TECHNICAL SCANNER v2.0 - DISCIPLINED CHART TRADING")
        print("="*70)
        
        if scanner.is_configured:
            print("\n--- 5-YEAR WEEKLY SCAN: TSLA ---")
            signal = await scanner.scan_weekly_5y("TSLA")
            
            if signal:
                print(f"\nTicker: {signal.ticker}")
                print(f"Price: ${signal.current_price}")
                print(f"ATR: ${signal.atr} ({signal.atr_pct}%)" if signal.atr else "ATR: N/A")
                
                print(f"\n=== MARKET STATE (Determined FIRST) ===")
                print(f"State: {signal.market_state.state.value.upper()}")
                print(f"Bias: {signal.market_state.bias.value}")
                print(f"Confidence: {signal.market_state.confidence}")
                print(f"Evidence:")
                for e in signal.market_state.evidence:
                    print(f"  - {e}")
                
                if signal.trend_channel:
                    print(f"\n=== TREND CHANNEL ===")
                    print(f"Upper: ${signal.trend_channel['channel_upper']}")
                    print(f"Lower: ${signal.trend_channel['channel_lower']}")
                    print(f"Position: {signal.trend_channel['position_in_channel']:.0%}")
                
                if signal.liquidity_sweeps:
                    print(f"\n=== LIQUIDITY SWEEPS ===")
                    for sweep in signal.liquidity_sweeps[:3]:
                        print(f"  - {sweep.note}")
                
                print(f"\n=== PRIMARY SCENARIO ===")
                if signal.primary_scenario:
                    print(f"Name: {signal.primary_scenario.name}")
                    print(f"Thesis: {signal.primary_scenario.thesis}")
                    print(f"Probability: {signal.primary_scenario.probability:.0%}")
                    print(f"Confirm: {signal.primary_scenario.confirm[0]}")
                    print(f"Invalidate: {signal.primary_scenario.invalidate[0]}")
                
                if signal.alternate_scenario:
                    print(f"\n=== ALTERNATE SCENARIO ===")
                    print(f"Name: {signal.alternate_scenario.name}")
                    print(f"Probability: {signal.alternate_scenario.probability:.0%}")
                
                print(f"\n=== VERDICT ===")
                print(f"Verdict: {signal.verdict.value.upper()}")
                print(f"Reasons: {'; '.join(signal.verdict_reasons)}")
                
                if signal.trade_hypothesis:
                    print(f"\n=== TRADE HYPOTHESIS ===")
                    print(f"Side: {signal.trade_hypothesis.side.upper()}")
                    print(f"Entry Zone: ${signal.trade_hypothesis.entry_zone['low']}-${signal.trade_hypothesis.entry_zone['high']}")
                    print(f"Stop: ${signal.trade_hypothesis.invalidation['level']} ({signal.trade_hypothesis.invalidation['logic']})")
                    print(f"Target: ${signal.trade_hypothesis.targets[0]['level']}")
                    print(f"R-Multiple: {signal.trade_hypothesis.expected_r}")
                    for note in signal.trade_hypothesis.risk_notes:
                        print(f"  - {note}")
                
                print(f"\n=== LEGACY SIGNAL ===")
                print(f"Signal: {signal.signal_strength}")
                print(f"Summary: {signal.summary}")
                
            else:
                print(f"Failed: {scanner.last_error}")
        else:
            print("\nAlpaca credentials not configured")
            print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    
    asyncio.run(test())
