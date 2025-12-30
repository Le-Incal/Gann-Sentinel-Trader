"""
Gann Sentinel Trader - Trade and Position Data Models
Defines the structure for trades, positions, and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid
import json


class TradeStatus(Enum):
    """Status of a trade through its lifecycle."""
    PENDING_APPROVAL = "pending_approval"  # Awaiting human approval
    APPROVED = "approved"                   # Approved, ready to submit
    SUBMITTED = "submitted"                 # Sent to broker
    PARTIALLY_FILLED = "partially_filled"   # Some shares filled
    FILLED = "filled"                       # Fully executed
    REJECTED = "rejected"                   # Rejected by human or system
    CANCELLED = "cancelled"                 # Cancelled before fill
    FAILED = "failed"                       # Broker error


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Buy or sell."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """
    A trade from recommendation through execution.
    Tracks the full lifecycle of a trading decision.
    """
    
    # Identity
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: Optional[str] = None  # Link to the analysis that generated this
    
    # Order details
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Status tracking
    status: TradeStatus = TradeStatus.PENDING_APPROVAL
    
    # Broker tracking
    alpaca_order_id: Optional[str] = None
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Execution details
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    
    # Context
    thesis: str = ""
    conviction_score: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    
    # Approval tracking
    approved_by: Optional[str] = None  # 'human' or 'auto'
    rejection_reason: Optional[str] = None
    
    @property
    def notional_value(self) -> float:
        """Estimated notional value of the trade."""
        price = self.fill_price or self.limit_price or 0
        return self.quantity * price
    
    @property
    def is_complete(self) -> bool:
        """Check if trade is in a terminal state."""
        return self.status in [
            TradeStatus.FILLED,
            TradeStatus.REJECTED,
            TradeStatus.CANCELLED,
            TradeStatus.FAILED
        ]
    
    def approve(self, by: str = "human") -> None:
        """Mark trade as approved."""
        self.status = TradeStatus.APPROVED
        self.approved_at = datetime.now(timezone.utc)
        self.approved_by = by
        self.updated_at = datetime.now(timezone.utc)
    
    def reject(self, reason: str = "") -> None:
        """Mark trade as rejected."""
        self.status = TradeStatus.REJECTED
        self.rejection_reason = reason
        self.updated_at = datetime.now(timezone.utc)
    
    def submit(self, alpaca_order_id: str) -> None:
        """Mark trade as submitted to broker."""
        self.status = TradeStatus.SUBMITTED
        self.alpaca_order_id = alpaca_order_id
        self.submitted_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def fill(self, price: float, quantity: float) -> None:
        """Mark trade as filled."""
        self.status = TradeStatus.FILLED
        self.fill_price = price
        self.fill_quantity = quantity
        self.filled_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "analysis_id": self.analysis_id,
            "ticker": self.ticker,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "alpaca_order_id": self.alpaca_order_id,
            "idempotency_key": self.idempotency_key,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "commission": self.commission,
            "thesis": self.thesis,
            "conviction_score": self.conviction_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "notional_value": self.notional_value,
            "is_complete": self.is_complete
        }
    
    def to_json(self) -> str:
        """Convert trade to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_telegram_message(self) -> str:
        """Format trade for Telegram notification."""
        emoji = "🟢" if self.side == OrderSide.BUY else "🔴"
        status_emoji = {
            TradeStatus.PENDING_APPROVAL: "⏳",
            TradeStatus.APPROVED: "✅",
            TradeStatus.SUBMITTED: "📤",
            TradeStatus.FILLED: "✅",
            TradeStatus.REJECTED: "❌",
            TradeStatus.CANCELLED: "🚫",
            TradeStatus.FAILED: "⚠️"
        }
        
        msg = f"""
{emoji} **{self.side.value.upper()} {self.ticker}**
{status_emoji.get(self.status, '❓')} Status: {self.status.value}

**Order Details:**
• Quantity: {self.quantity:.2f} shares
• Type: {self.order_type.value}
• Limit: ${self.limit_price:.2f if self.limit_price else 'N/A'}

**Conviction:** {self.conviction_score}/100

**Trade ID:** `{self.trade_id[:8]}...`
"""
        
        if self.status == TradeStatus.PENDING_APPROVAL:
            msg += f"\n\n📱 Reply with:\n`/approve {self.trade_id[:8]}`\n`/reject {self.trade_id[:8]}`"
        
        return msg.strip()


@dataclass
class Position:
    """
    A current position in the portfolio.
    """
    
    # Identity
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str = ""
    
    # Quantities
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    
    # Current state
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    
    # P&L
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    realized_pnl: float = 0.0
    
    # Context
    thesis: str = ""
    analysis_id: Optional[str] = None
    entry_date: Optional[datetime] = None
    
    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    # Tracking
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.quantity * self.avg_entry_price
    
    def update_price(self, current_price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (self.unrealized_pnl / self.cost_basis) if self.cost_basis > 0 else 0
        self.updated_at = datetime.now(timezone.utc)
    
    @property
    def should_stop_loss(self) -> bool:
        """Check if stop loss has been triggered."""
        if self.stop_loss_price and self.current_price:
            return self.current_price <= self.stop_loss_price
        return False
    
    @property
    def should_take_profit(self) -> bool:
        """Check if take profit has been triggered."""
        if self.take_profit_price and self.current_price:
            return self.current_price >= self.take_profit_price
        return False
    
    def to_dict(self) -> dict:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "ticker": self.ticker,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "thesis": self.thesis,
            "analysis_id": self.analysis_id,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "should_stop_loss": self.should_stop_loss,
            "should_take_profit": self.should_take_profit,
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert position to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PortfolioSnapshot:
    """
    A point-in-time snapshot of the portfolio state.
    """
    
    # Identity
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Values
    cash: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    
    # P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    # Positions
    positions: List[Dict[str, Any]] = field(default_factory=list)
    position_count: int = 0
    
    # Risk metrics
    largest_position_pct: float = 0.0
    buying_power: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "positions": self.positions,
            "position_count": self.position_count,
            "largest_position_pct": self.largest_position_pct,
            "buying_power": self.buying_power
        }
    
    def to_json(self) -> str:
        """Convert snapshot to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_telegram_message(self) -> str:
        """Format snapshot for Telegram."""
        pnl_emoji = "📈" if self.daily_pnl >= 0 else "📉"
        
        msg = f"""
📊 **Portfolio Snapshot**

**Value:** ${self.total_value:,.2f}
• Cash: ${self.cash:,.2f}
• Positions: ${self.positions_value:,.2f}

{pnl_emoji} **Today's P&L:** ${self.daily_pnl:+,.2f} ({self.daily_pnl_pct:+.2f}%)
📈 **Total P&L:** ${self.total_pnl:+,.2f} ({self.total_pnl_pct:+.2f}%)

**Positions:** {self.position_count}
**Buying Power:** ${self.buying_power:,.2f}
"""
        return msg.strip()
