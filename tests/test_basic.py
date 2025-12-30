"""
Gann Sentinel Trader - Basic Tests
Run with: pytest tests/test_basic.py -v
"""

import pytest
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signals import Signal, SignalType, SignalSource, AssetScope, RawValue, DirectionalBias, TimeHorizon
from models.analysis import Analysis, Recommendation, ExitTrigger
from models.trades import Trade, Position, PortfolioSnapshot, TradeStatus, OrderType, OrderSide


class TestSignalModel:
    """Tests for Signal model."""
    
    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = Signal(
            signal_type=SignalType.SENTIMENT,
            source=SignalSource.GROK_X,
            summary="Test signal",
            confidence=0.75
        )
        
        assert signal.signal_id is not None
        assert signal.signal_type == SignalType.SENTIMENT
        assert signal.source == SignalSource.GROK_X
        assert signal.confidence == 0.75
    
    def test_signal_dedup_hash(self):
        """Test deduplication hash generation."""
        signal1 = Signal(
            source=SignalSource.GROK_X,
            asset_scope=AssetScope(tickers=["AAPL"]),
            summary="Apple stock is rising"
        )
        
        signal2 = Signal(
            source=SignalSource.GROK_X,
            asset_scope=AssetScope(tickers=["AAPL"]),
            summary="Apple stock is rising"
        )
        
        # Same content should produce same hash
        assert signal1.dedup_hash == signal2.dedup_hash
    
    def test_signal_staleness(self):
        """Test staleness detection."""
        # Create a fresh signal
        fresh_signal = Signal(
            timestamp_utc=datetime.now(timezone.utc),
            staleness_seconds=3600  # 1 hour
        )
        assert not fresh_signal.is_stale
        
        # Create an old signal
        from datetime import timedelta
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        old_signal = Signal(
            timestamp_utc=old_time,
            staleness_seconds=3600
        )
        assert old_signal.is_stale
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal(
            signal_type=SignalType.MACRO,
            source=SignalSource.FRED,
            summary="CPI came in at 3.2%",
            raw_value=RawValue(type="rate", value=3.2, unit="percent")
        )
        
        data = signal.to_dict()
        
        assert data["signal_type"] == "macro"
        assert data["source"] == "fred"
        assert data["summary"] == "CPI came in at 3.2%"
        assert data["raw_value"]["value"] == 3.2


class TestAnalysisModel:
    """Tests for Analysis model."""
    
    def test_analysis_creation(self):
        """Test basic analysis creation."""
        analysis = Analysis(
            ticker="NVDA",
            recommendation=Recommendation.BUY,
            conviction_score=85,
            thesis="Strong AI demand"
        )
        
        assert analysis.ticker == "NVDA"
        assert analysis.recommendation == Recommendation.BUY
        assert analysis.conviction_score == 85
    
    def test_analysis_actionable(self):
        """Test actionable detection."""
        # High conviction BUY should be actionable
        actionable = Analysis(
            ticker="AAPL",
            recommendation=Recommendation.BUY,
            conviction_score=85
        )
        assert actionable.is_actionable
        
        # Low conviction should not be actionable
        low_conviction = Analysis(
            ticker="AAPL",
            recommendation=Recommendation.BUY,
            conviction_score=60
        )
        assert not low_conviction.is_actionable
        
        # NONE recommendation should not be actionable
        no_trade = Analysis(
            ticker="AAPL",
            recommendation=Recommendation.NONE,
            conviction_score=85
        )
        assert not no_trade.is_actionable
    
    def test_conviction_level(self):
        """Test conviction level labels."""
        very_high = Analysis(conviction_score=95)
        assert very_high.conviction_level == "Very High"
        
        high = Analysis(conviction_score=85)
        assert high.conviction_level == "High"
        
        low = Analysis(conviction_score=40)
        assert low.conviction_level == "Low"


class TestTradeModel:
    """Tests for Trade model."""
    
    def test_trade_creation(self):
        """Test basic trade creation."""
        trade = Trade(
            ticker="TSLA",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        assert trade.ticker == "TSLA"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 10
        assert trade.status == TradeStatus.PENDING_APPROVAL
    
    def test_trade_lifecycle(self):
        """Test trade status transitions."""
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=5
        )
        
        # Initial state
        assert trade.status == TradeStatus.PENDING_APPROVAL
        assert not trade.is_complete
        
        # Approve
        trade.approve(by="human")
        assert trade.status == TradeStatus.APPROVED
        assert trade.approved_by == "human"
        assert trade.approved_at is not None
        
        # Submit
        trade.submit("alpaca-order-123")
        assert trade.status == TradeStatus.SUBMITTED
        assert trade.alpaca_order_id == "alpaca-order-123"
        
        # Fill
        trade.fill(price=150.00, quantity=5)
        assert trade.status == TradeStatus.FILLED
        assert trade.fill_price == 150.00
        assert trade.is_complete
    
    def test_trade_rejection(self):
        """Test trade rejection."""
        trade = Trade(ticker="GME", side=OrderSide.BUY, quantity=100)
        
        trade.reject(reason="Too risky")
        
        assert trade.status == TradeStatus.REJECTED
        assert trade.rejection_reason == "Too risky"
        assert trade.is_complete


class TestPositionModel:
    """Tests for Position model."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        position = Position(
            ticker="NVDA",
            quantity=20,
            avg_entry_price=500.00
        )
        
        assert position.ticker == "NVDA"
        assert position.cost_basis == 10000.00
    
    def test_position_pnl(self):
        """Test P&L calculations."""
        position = Position(
            ticker="AAPL",
            quantity=10,
            avg_entry_price=150.00
        )
        
        # Update with higher price
        position.update_price(165.00)
        
        assert position.current_price == 165.00
        assert position.market_value == 1650.00
        assert position.unrealized_pnl == 150.00
        assert position.unrealized_pnl_pct == 0.10  # 10% gain
    
    def test_stop_loss_trigger(self):
        """Test stop loss detection."""
        position = Position(
            ticker="TSLA",
            quantity=5,
            avg_entry_price=200.00,
            stop_loss_price=170.00  # 15% below entry
        )
        
        # Price above stop
        position.update_price(190.00)
        assert not position.should_stop_loss
        
        # Price at stop
        position.update_price(170.00)
        assert position.should_stop_loss
        
        # Price below stop
        position.update_price(160.00)
        assert position.should_stop_loss


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot model."""
    
    def test_snapshot_creation(self):
        """Test portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            cash=5000.00,
            positions_value=15000.00,
            total_value=20000.00,
            daily_pnl=500.00,
            daily_pnl_pct=0.025
        )
        
        assert snapshot.total_value == 20000.00
        assert snapshot.daily_pnl_pct == 0.025


class TestIntegration:
    """Basic integration tests."""
    
    def test_signal_to_analysis_flow(self):
        """Test signal flowing to analysis."""
        # Create signals
        signals = [
            Signal(
                signal_type=SignalType.SENTIMENT,
                source=SignalSource.GROK_X,
                asset_scope=AssetScope(tickers=["NVDA"]),
                summary="Bullish sentiment on NVDA",
                confidence=0.8,
                directional_bias=DirectionalBias.POSITIVE
            ),
            Signal(
                signal_type=SignalType.MACRO,
                source=SignalSource.FRED,
                asset_scope=AssetScope(macro_regions=["US"]),
                summary="Treasury yields falling",
                confidence=0.95,
                directional_bias=DirectionalBias.POSITIVE
            )
        ]
        
        # Create analysis referencing signals
        analysis = Analysis(
            ticker="NVDA",
            recommendation=Recommendation.BUY,
            conviction_score=85,
            thesis="Strong sentiment + falling yields support growth stocks",
            signals_used=[s.signal_id for s in signals]
        )
        
        # Verify linkage
        assert len(analysis.signals_used) == 2
        assert analysis.is_actionable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
