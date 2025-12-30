"""
Gann Sentinel Trader - Demo Script
Test the system with mocked data before full deployment.

Run with: python demo.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from models.signals import Signal, SignalType, SignalSource, AssetScope, RawValue, DirectionalBias, TimeHorizon, Evidence
from models.analysis import Analysis, Recommendation, ExitTrigger
from models.trades import Trade, Position, PortfolioSnapshot, TradeStatus, OrderType, OrderSide
from storage.database import Database
from executors.risk_engine import RiskEngine


def create_mock_signals():
    """Create realistic mock signals for testing."""
    signals = [
        # Grok sentiment signal
        Signal(
            signal_type=SignalType.SENTIMENT,
            source=SignalSource.GROK_X,
            asset_scope=AssetScope(tickers=["RKLB"], sectors=["AEROSPACE"]),
            summary="Rocket Lab seeing surge in retail interest following successful Neutron engine test. Twitter sentiment strongly bullish with 340% increase in mentions.",
            raw_value=RawValue(
                type="index",
                value=0.72,
                unit="sentiment_score",
                prior_value=0.45,
                change=0.27,
                change_period="24h"
            ),
            evidence=[
                Evidence(
                    source="X/Twitter via Grok",
                    source_tier="social",
                    excerpt="Multiple influential space accounts discussing RKLB Neutron progress",
                    timestamp_utc=datetime.now(timezone.utc)
                )
            ],
            confidence=0.65,
            directional_bias=DirectionalBias.POSITIVE,
            time_horizon=TimeHorizon.WEEKS,
            novelty="developing",
            staleness_seconds=3600,
            uncertainties=["Retail sentiment can be fleeting", "No confirmed SpaceX IPO timeline"]
        ),
        
        # FRED macro signal
        Signal(
            signal_type=SignalType.MACRO,
            source=SignalSource.FRED,
            asset_scope=AssetScope(
                tickers=["SPY", "TLT"],
                macro_regions=["US"],
                asset_classes=["EQUITY", "FIXED_INCOME"]
            ),
            summary="10-Year Treasury yield dropped 15bps to 4.25%, suggesting market expects Fed rate cuts.",
            raw_value=RawValue(
                type="rate",
                value=4.25,
                unit="percent",
                prior_value=4.40,
                change=-0.15,
                change_period="1w"
            ),
            evidence=[
                Evidence(
                    source="https://fred.stlouisfed.org/series/DGS10",
                    source_tier="official",
                    excerpt="10-Year Treasury Constant Maturity Rate: 4.25%",
                    timestamp_utc=datetime.now(timezone.utc),
                    url="https://fred.stlouisfed.org/series/DGS10"
                )
            ],
            confidence=0.95,
            directional_bias=DirectionalBias.POSITIVE,
            time_horizon=TimeHorizon.WEEKS,
            novelty="developing",
            staleness_seconds=86400,
            uncertainties=["Fed communication could shift expectations"]
        ),
        
        # Polymarket prediction signal
        Signal(
            signal_type=SignalType.PREDICTION_MARKET,
            source=SignalSource.POLYMARKET,
            asset_scope=AssetScope(
                tickers=["SPY", "IWM"],
                macro_regions=["US"],
                asset_classes=["EQUITY"]
            ),
            summary="Polymarket probability of Fed rate cut in March 2025 dropped from 68% to 52% following strong jobs data.",
            raw_value=RawValue(
                type="probability",
                value=0.52,
                unit="percent",
                prior_value=0.68,
                change=-0.16,
                change_period="48h"
            ),
            evidence=[
                Evidence(
                    source="Polymarket",
                    source_tier="tier2",
                    excerpt="Will the Fed cut rates in March 2025? Current: 52% Yes. Volume: $1.2M",
                    timestamp_utc=datetime.now(timezone.utc)
                )
            ],
            confidence=0.70,
            directional_bias=DirectionalBias.NEGATIVE,
            time_horizon=TimeHorizon.WEEKS,
            novelty="developing",
            staleness_seconds=3600,
            uncertainties=["Prediction markets reflect bettor consensus, not certainty"]
        ),
        
        # News event signal
        Signal(
            signal_type=SignalType.EVENT,
            source=SignalSource.GROK_WEB,
            asset_scope=AssetScope(
                tickers=["NVDA", "AMD", "SMCI"],
                sectors=["SEMICONDUCTORS"],
                macro_regions=["US", "CHINA"]
            ),
            summary="Reuters reports Biden administration considering additional AI chip export restrictions to China. Nvidia H200 potentially targeted.",
            evidence=[
                Evidence(
                    source="Reuters",
                    source_tier="tier1",
                    excerpt="Biden administration considering additional curbs on AI chip exports",
                    timestamp_utc=datetime.now(timezone.utc),
                    url="https://reuters.com/technology/chip-restrictions"
                )
            ],
            confidence=0.75,
            directional_bias=DirectionalBias.NEGATIVE,
            time_horizon=TimeHorizon.DAYS,
            novelty="new",
            staleness_seconds=14400,
            uncertainties=[
                "No official White House confirmation",
                "Timeline unclear",
                "Scope of restrictions not defined"
            ]
        )
    ]
    
    return signals


def create_mock_analysis():
    """Create a mock analysis based on signals."""
    return Analysis(
        ticker="RKLB",
        recommendation=Recommendation.BUY,
        conviction_score=82,
        thesis="Rocket Lab is benefiting from increased space sector attention and successful Neutron development. The falling Treasury yields create a favorable environment for growth stocks. While there are risks from potential SpaceX IPO competition, RKLB has carved a niche in small satellite launches that complements rather than competes with SpaceX.",
        bull_case="SpaceX IPO attention brings sector-wide investment flows. Neutron success positions RKLB as credible second player. Falling rates support growth stock multiples.",
        bear_case="SpaceX IPO could dominate capital allocation. Retail enthusiasm may fade. Export restrictions could affect some defense contracts.",
        variant_perception="Market underestimates Rocket Lab's ability to capture overflow demand from SpaceX capacity constraints.",
        position_size_pct=0.15,
        entry_strategy="market",
        stop_loss_pct=0.15,
        exit_triggers=[
            ExitTrigger(
                trigger_type="take_profit",
                description="30% gain from entry",
                percentage=0.30
            ),
            ExitTrigger(
                trigger_type="thesis_breaker",
                description="SpaceX announces small-sat launch pricing cut"
            )
        ],
        thesis_breakers=[
            "SpaceX announces competitive small-sat pricing",
            "Neutron development major delay (>6 months)",
            "Key government contract loss"
        ],
        time_horizon="weeks"
    )


def create_mock_portfolio():
    """Create a mock portfolio snapshot."""
    return PortfolioSnapshot(
        cash=3500.00,
        positions_value=1500.00,
        total_value=5000.00,
        daily_pnl=75.00,
        daily_pnl_pct=0.015,
        positions=[
            {
                "ticker": "SPY",
                "quantity": 3,
                "avg_entry_price": 480.00,
                "current_price": 495.00,
                "market_value": 1485.00,
                "unrealized_pnl": 45.00,
                "unrealized_pnl_pct": 0.03
            }
        ],
        position_count=1,
        largest_position_pct=0.297,
        buying_power=3500.00
    )


async def run_demo():
    """Run the demo."""
    print("\n" + "=" * 60)
    print("GANN SENTINEL TRADER - DEMO")
    print("=" * 60)
    
    # Initialize database
    print("\n[1] Initializing database...")
    db = Database()
    print("    ✓ Database initialized")
    
    # Create mock signals
    print("\n[2] Creating mock signals...")
    signals = create_mock_signals()
    for signal in signals:
        db.save_signal(signal.to_dict())
        print(f"    ✓ Saved: {signal.signal_type.value} from {signal.source.value}")
    
    # Display signals
    print("\n[3] Signal Summary:")
    print("-" * 60)
    for i, signal in enumerate(signals, 1):
        stale_status = "🔴 STALE" if signal.is_stale else "🟢 Fresh"
        print(f"""
    Signal {i}: {signal.signal_type.value.upper()}
    Source: {signal.source.value}
    Summary: {signal.summary[:80]}...
    Direction: {signal.directional_bias.value}
    Confidence: {signal.confidence:.0%}
    Status: {stale_status}
""")
    
    # Create mock analysis
    print("\n[4] Creating mock analysis...")
    analysis = create_mock_analysis()
    analysis.signals_used = [s.signal_id for s in signals]
    db.save_analysis(analysis.to_dict())
    print(f"    ✓ Analysis created for {analysis.ticker}")
    
    # Display analysis
    print("\n[5] Analysis Summary:")
    print("-" * 60)
    print(f"""
    Ticker: {analysis.ticker}
    Recommendation: {analysis.recommendation.value}
    Conviction: {analysis.conviction_score}/100 ({analysis.conviction_level})
    
    Thesis:
    {analysis.thesis[:200]}...
    
    Position Size: {analysis.position_size_pct:.0%}
    Stop Loss: {analysis.stop_loss_pct:.0%}
    Time Horizon: {analysis.time_horizon}
    
    Actionable: {'✅ YES' if analysis.is_actionable else '❌ NO'}
""")
    
    # Create mock portfolio
    print("\n[6] Creating mock portfolio...")
    portfolio = create_mock_portfolio()
    db.save_snapshot(portfolio.to_dict())
    print(f"    ✓ Portfolio: ${portfolio.total_value:,.2f}")
    
    # Run risk checks
    print("\n[7] Running risk checks...")
    risk_engine = RiskEngine()
    
    # Create mock positions list
    positions = [
        Position(
            ticker="SPY",
            quantity=3,
            avg_entry_price=480.00,
            current_price=495.00
        )
    ]
    
    passed, results = risk_engine.validate_trade(analysis, portfolio, positions)
    
    print(f"\n    Risk Check Results: {'✅ PASSED' if passed else '❌ FAILED'}")
    print("-" * 60)
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"    [{status}] {result.rule}: {result.message}")
    
    # Calculate position size
    print("\n[8] Calculating position size...")
    current_price = 22.50  # Mock RKLB price
    sizing = risk_engine.calculate_position_size(analysis, portfolio, current_price)
    print(f"""
    Current Price: ${current_price:.2f}
    Shares: {sizing['shares']}
    Dollar Amount: ${sizing['dollar_amount']:,.2f}
    Portfolio %: {sizing['percentage']:.1%}
""")
    
    # Create trade
    print("\n[9] Creating trade...")
    trade = Trade(
        analysis_id=analysis.analysis_id,
        ticker=analysis.ticker,
        side=OrderSide.BUY,
        quantity=sizing['shares'],
        order_type=OrderType.MARKET,
        thesis=analysis.thesis,
        conviction_score=analysis.conviction_score
    )
    db.save_trade(trade.to_dict())
    print(f"    ✓ Trade created: {trade.trade_id[:8]}")
    
    # Simulate approval flow
    print("\n[10] Simulating approval flow...")
    print(f"    Status: {trade.status.value}")
    
    trade.approve(by="demo_user")
    db.save_trade(trade.to_dict())
    print(f"    Approved by: {trade.approved_by}")
    print(f"    Status: {trade.status.value}")
    
    # Simulate execution
    print("\n[11] Simulating execution...")
    trade.submit("demo-alpaca-order-123")
    trade.fill(price=22.55, quantity=sizing['shares'])
    db.save_trade(trade.to_dict())
    print(f"    Fill Price: ${trade.fill_price:.2f}")
    print(f"    Status: {trade.status.value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"""
    Database Location: {db.db_path}
    
    Objects Created:
    - {len(signals)} Signals
    - 1 Analysis
    - 1 Portfolio Snapshot
    - 1 Trade (simulated execution)
    
    Next Steps:
    1. Configure your .env file with real API keys
    2. Run: python agent.py
    3. Monitor via Telegram commands
    
    Remember: Start with PAPER trading mode!
""")


if __name__ == "__main__":
    asyncio.run(run_demo())
