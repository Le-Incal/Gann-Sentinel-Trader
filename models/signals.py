"""
Gann Sentinel Trader - Signal Data Models
Defines the structure for signals from various sources.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid
import hashlib
import json


class SignalType(Enum):
    """Types of signals the system can process."""
    SENTIMENT = "sentiment"
    MACRO = "macro"
    POLICY = "policy"
    EVENT = "event"
    PREDICTION_MARKET = "prediction_market"
    NARRATIVE_SHIFT = "narrative_shift"
    PRICE = "price"


class SignalSource(Enum):
    """Sources of signals."""
    GROK_X = "grok_x"           # X/Twitter via Grok
    GROK_WEB = "grok_web"       # Web search via Grok
    FRED = "fred"               # Federal Reserve Economic Data
    POLYMARKET = "polymarket"   # Prediction markets
    ALPACA = "alpaca"           # Price data from broker
    MANUAL = "manual"           # Manually entered signals


class DirectionalBias(Enum):
    """Directional bias of a signal."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class TimeHorizon(Enum):
    """Time horizon for signal relevance."""
    INTRADAY = "intraday"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    UNKNOWN = "unknown"


@dataclass
class Evidence:
    """A piece of evidence supporting a signal."""
    source: str
    source_tier: str  # official, tier1, tier2, social
    excerpt: str
    timestamp_utc: datetime
    url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_tier": self.source_tier,
            "excerpt": self.excerpt,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "url": self.url
        }


@dataclass
class RawValue:
    """Quantitative value associated with a signal."""
    type: Optional[str] = None  # probability, rate, index, count, price
    value: Optional[float] = None
    unit: Optional[str] = None  # percent, bps, usd, index_points
    prior_value: Optional[float] = None
    change: Optional[float] = None
    change_period: Optional[str] = None  # 24h, 1w, etc.
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "prior_value": self.prior_value,
            "change": self.change,
            "change_period": self.change_period
        }


@dataclass
class AssetScope:
    """Defines which assets a signal applies to."""
    tickers: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    macro_regions: List[str] = field(default_factory=list)  # US, EU, ASIA, GLOBAL
    asset_classes: List[str] = field(default_factory=list)  # EQUITY, FIXED_INCOME, etc.
    
    def to_dict(self) -> dict:
        return {
            "tickers": self.tickers,
            "sectors": self.sectors,
            "macro_regions": self.macro_regions,
            "asset_classes": self.asset_classes
        }


@dataclass
class Signal:
    """
    A market signal from any source.
    This is the canonical data structure for all signals in the system.
    """
    
    # Identity
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Classification
    signal_type: SignalType = SignalType.SENTIMENT
    source: SignalSource = SignalSource.GROK_X
    
    # Scope
    asset_scope: AssetScope = field(default_factory=AssetScope)
    
    # Content
    summary: str = ""
    raw_value: RawValue = field(default_factory=RawValue)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Assessment
    confidence: float = 0.5  # 0.0 to 1.0
    directional_bias: DirectionalBias = DirectionalBias.UNCLEAR
    time_horizon: TimeHorizon = TimeHorizon.UNKNOWN
    novelty: str = "new"  # new, developing, recurring
    
    # Staleness
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    staleness_seconds: int = 3600
    
    # Uncertainties
    uncertainties: List[str] = field(default_factory=list)
    
    @property
    def stale_after_utc(self) -> datetime:
        """Calculate when this signal becomes stale."""
        from datetime import timedelta
        return self.timestamp_utc + timedelta(seconds=self.staleness_seconds)
    
    @property
    def is_stale(self) -> bool:
        """Check if signal has exceeded staleness threshold."""
        return datetime.now(timezone.utc) > self.stale_after_utc
    
    @property
    def dedup_hash(self) -> str:
        """Generate deduplication hash based on source, primary ticker, and summary."""
        primary_ticker = self.asset_scope.tickers[0] if self.asset_scope.tickers else "NONE"
        content = f"{self.source.value}|{primary_ticker}|{self.summary.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert signal to dictionary for storage/serialization."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.signal_type.value,
            "source": self.source.value,
            "asset_scope": self.asset_scope.to_dict(),
            "summary": self.summary,
            "raw_value": self.raw_value.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "directional_bias": self.directional_bias.value,
            "time_horizon": self.time_horizon.value,
            "novelty": self.novelty,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "staleness_seconds": self.staleness_seconds,
            "stale_after_utc": self.stale_after_utc.isoformat(),
            "is_stale": self.is_stale,
            "uncertainties": self.uncertainties
        }
    
    def to_json(self) -> str:
        """Convert signal to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Signal":
        """Create Signal from dictionary."""
        return cls(
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            signal_type=SignalType(data.get("signal_type", "sentiment")),
            source=SignalSource(data.get("source", "grok_x")),
            asset_scope=AssetScope(
                tickers=data.get("asset_scope", {}).get("tickers", []),
                sectors=data.get("asset_scope", {}).get("sectors", []),
                macro_regions=data.get("asset_scope", {}).get("macro_regions", []),
                asset_classes=data.get("asset_scope", {}).get("asset_classes", [])
            ),
            summary=data.get("summary", ""),
            raw_value=RawValue(
                type=data.get("raw_value", {}).get("type"),
                value=data.get("raw_value", {}).get("value"),
                unit=data.get("raw_value", {}).get("unit"),
                prior_value=data.get("raw_value", {}).get("prior_value"),
                change=data.get("raw_value", {}).get("change"),
                change_period=data.get("raw_value", {}).get("change_period")
            ),
            confidence=data.get("confidence", 0.5),
            directional_bias=DirectionalBias(data.get("directional_bias", "unclear")),
            time_horizon=TimeHorizon(data.get("time_horizon", "unknown")),
            novelty=data.get("novelty", "new"),
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]) if "timestamp_utc" in data else datetime.now(timezone.utc),
            staleness_seconds=data.get("staleness_seconds", 3600),
            uncertainties=data.get("uncertainties", [])
        )


@dataclass
class SignalBatch:
    """A batch of signals with metadata."""
    signals: List[Signal] = field(default_factory=list)
    query_context: str = ""
    retrieval_time_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_errors: List[Dict[str, Any]] = field(default_factory=list)
    known_blindspots: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "schema_version": "1.1.0",
            "signals": [s.to_dict() for s in self.signals],
            "meta": {
                "query_context": self.query_context,
                "retrieval_time_utc": self.retrieval_time_utc.isoformat(),
                "retrieval_errors": self.retrieval_errors,
                "known_blindspots": self.known_blindspots,
                "signal_disclaimer": "Signals are informational only and may be incorrect or incomplete."
            }
        }
