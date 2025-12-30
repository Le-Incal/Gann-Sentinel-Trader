"""
Gann Sentinel Trader - Analysis Data Models
Defines the structure for Claude's analysis output.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid
import json


class Recommendation(Enum):
    """Trade recommendations."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NONE = "NONE"  # No action recommended


@dataclass
class ExitTrigger:
    """Defines conditions for exiting a position."""
    trigger_type: str  # take_profit, stop_loss, thesis_breaker, time_based
    description: str
    price_target: Optional[float] = None
    percentage: Optional[float] = None
    condition: Optional[str] = None  # For complex conditions
    
    def to_dict(self) -> dict:
        return {
            "trigger_type": self.trigger_type,
            "description": self.description,
            "price_target": self.price_target,
            "percentage": self.percentage,
            "condition": self.condition
        }


@dataclass
class Analysis:
    """
    Claude's analysis of market signals.
    Contains the investment thesis, conviction score, and trade recommendation.
    """
    
    # Identity
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Target
    ticker: Optional[str] = None
    sector: Optional[str] = None
    
    # Recommendation
    recommendation: Recommendation = Recommendation.NONE
    conviction_score: int = 0  # 0-100
    
    # Thesis
    thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    variant_perception: str = ""  # What we believe that others don't
    
    # Trade parameters
    position_size_pct: float = 0.0  # Suggested position size as % of portfolio
    entry_price_target: Optional[float] = None
    entry_strategy: str = ""  # market, limit, scale_in
    
    # Risk management
    stop_loss_pct: float = 0.15
    stop_loss_price: Optional[float] = None
    
    # Exit strategy
    exit_triggers: List[ExitTrigger] = field(default_factory=list)
    thesis_breakers: List[str] = field(default_factory=list)
    time_horizon: str = "weeks"  # intraday, days, weeks, months
    
    # Signals used
    signals_used: List[str] = field(default_factory=list)  # Signal IDs
    
    # Reasoning trace
    reasoning_steps: List[str] = field(default_factory=list)
    key_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence breakdown
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if this analysis results in an actionable trade."""
        from config import Config
        return (
            self.recommendation in [Recommendation.BUY, Recommendation.SELL] and
            self.conviction_score >= Config.MIN_CONVICTION and
            self.ticker is not None
        )
    
    @property
    def conviction_level(self) -> str:
        """Human-readable conviction level."""
        if self.conviction_score >= 90:
            return "Very High"
        elif self.conviction_score >= 80:
            return "High"
        elif self.conviction_score >= 70:
            return "Medium-High"
        elif self.conviction_score >= 60:
            return "Medium"
        elif self.conviction_score >= 50:
            return "Low-Medium"
        else:
            return "Low"
    
    def to_dict(self) -> dict:
        """Convert analysis to dictionary for storage/serialization."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "ticker": self.ticker,
            "sector": self.sector,
            "recommendation": self.recommendation.value,
            "conviction_score": self.conviction_score,
            "conviction_level": self.conviction_level,
            "is_actionable": self.is_actionable,
            "thesis": self.thesis,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "variant_perception": self.variant_perception,
            "position_size_pct": self.position_size_pct,
            "entry_price_target": self.entry_price_target,
            "entry_strategy": self.entry_strategy,
            "stop_loss_pct": self.stop_loss_pct,
            "stop_loss_price": self.stop_loss_price,
            "exit_triggers": [t.to_dict() for t in self.exit_triggers],
            "thesis_breakers": self.thesis_breakers,
            "time_horizon": self.time_horizon,
            "signals_used": self.signals_used,
            "reasoning_steps": self.reasoning_steps,
            "key_factors": self.key_factors,
            "confidence_factors": self.confidence_factors
        }
    
    def to_json(self) -> str:
        """Convert analysis to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_telegram_message(self) -> str:
        """Format analysis as a Telegram message."""
        emoji = {
            Recommendation.BUY: "🟢",
            Recommendation.SELL: "🔴",
            Recommendation.HOLD: "🟡",
            Recommendation.NONE: "⚪"
        }
        
        msg = f"""
{emoji[self.recommendation]} **{self.recommendation.value} {self.ticker or 'N/A'}**

**Conviction:** {self.conviction_score}/100 ({self.conviction_level})

**Thesis:**
{self.thesis[:500]}{'...' if len(self.thesis) > 500 else ''}

**Bull Case:** {self.bull_case[:200]}{'...' if len(self.bull_case) > 200 else ''}

**Bear Case:** {self.bear_case[:200]}{'...' if len(self.bear_case) > 200 else ''}

**Parameters:**
• Position Size: {self.position_size_pct * 100:.1f}%
• Entry Target: ${self.entry_price_target:.2f if self.entry_price_target else 'Market'}
• Stop Loss: {self.stop_loss_pct * 100:.1f}%
• Time Horizon: {self.time_horizon}

**Thesis Breakers:**
{chr(10).join(['• ' + tb for tb in self.thesis_breakers[:3]])}
"""
        return msg.strip()
    
    @classmethod
    def from_dict(cls, data: dict) -> "Analysis":
        """Create Analysis from dictionary."""
        exit_triggers = [
            ExitTrigger(
                trigger_type=t.get("trigger_type", ""),
                description=t.get("description", ""),
                price_target=t.get("price_target"),
                percentage=t.get("percentage"),
                condition=t.get("condition")
            )
            for t in data.get("exit_triggers", [])
        ]
        
        return cls(
            analysis_id=data.get("analysis_id", str(uuid.uuid4())),
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]) if "timestamp_utc" in data else datetime.now(timezone.utc),
            ticker=data.get("ticker"),
            sector=data.get("sector"),
            recommendation=Recommendation(data.get("recommendation", "NONE")),
            conviction_score=data.get("conviction_score", 0),
            thesis=data.get("thesis", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            variant_perception=data.get("variant_perception", ""),
            position_size_pct=data.get("position_size_pct", 0.0),
            entry_price_target=data.get("entry_price_target"),
            entry_strategy=data.get("entry_strategy", ""),
            stop_loss_pct=data.get("stop_loss_pct", 0.15),
            stop_loss_price=data.get("stop_loss_price"),
            exit_triggers=exit_triggers,
            thesis_breakers=data.get("thesis_breakers", []),
            time_horizon=data.get("time_horizon", "weeks"),
            signals_used=data.get("signals_used", []),
            reasoning_steps=data.get("reasoning_steps", []),
            key_factors=data.get("key_factors", []),
            confidence_factors=data.get("confidence_factors", {})
        )
