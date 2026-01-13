#!/usr/bin/env python3
"""
TDD Tests for Final Phase 2/3 Completion:
1. Automatic Trade Outcome Recording
2. Live SPY Data Fetching
3. Weekly Performance Digest
4. Dynamic Threshold Adjustment
5. Pattern Library with Historical Backreferences

Run with: python test_final_completion.py
"""

import sys
import os
import json
import importlib.util
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import Database
from learning_engine import LearningEngine, add_learning_tables


def create_test_db():
    """Create test database with learning tables."""
    db = Database(db_path=":memory:")
    # Manually add learning tables
    try:
        add_learning_tables(db)
    except Exception as e:
        # Tables might already exist or need different approach
        pass
    return db


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_CLOSED_POSITION = {
    "ticker": "NVDA",
    "side": "BUY",
    "entry_price": 140.00,
    "entry_date": "2026-01-06T14:35:00Z",
    "exit_price": 155.00,
    "exit_date": "2026-01-13T15:30:00Z",
    "quantity": 10,
    "exit_reason": "take_profit",
    "signal_sources": ["grok", "perplexity", "chatgpt"],
    "conviction_at_entry": 85
}

SAMPLE_TRADE_HISTORY = [
    {"ticker": "NVDA", "side": "BUY", "realized_pnl": 150.00, "realized_pnl_pct": 0.107, "alpha": 0.05},
    {"ticker": "AAPL", "side": "BUY", "realized_pnl": -45.00, "realized_pnl_pct": -0.024, "alpha": -0.01},
    {"ticker": "SMCI", "side": "BUY", "realized_pnl": 220.00, "realized_pnl_pct": 0.15, "alpha": 0.08},
    {"ticker": "TSLA", "side": "BUY", "realized_pnl": -80.00, "realized_pnl_pct": -0.04, "alpha": -0.02},
    {"ticker": "AMD", "side": "BUY", "realized_pnl": 95.00, "realized_pnl_pct": 0.063, "alpha": 0.03},
]

SAMPLE_PATTERNS = [
    {
        "pattern_type": "earnings_momentum",
        "ticker": "NVDA",
        "entry_catalyst": "Strong guidance + AI narrative",
        "outcome": "win",
        "return_pct": 10.7,
        "hold_days": 7,
        "market_regime": "normal_vix_up_trend"
    },
    {
        "pattern_type": "sector_rotation",
        "ticker": "SMCI",
        "entry_catalyst": "Server demand narrative",
        "outcome": "win",
        "return_pct": 15.0,
        "hold_days": 5,
        "market_regime": "normal_vix_up_trend"
    }
]


# =============================================================================
# TEST 1: AUTOMATIC TRADE OUTCOME RECORDING
# =============================================================================

def test_record_trade_outcome():
    """Test that we can record a closed trade with full metrics."""
    print("\n[TEST] test_record_trade_outcome")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Record a closed trade
        outcome_id = engine.record_trade_outcome(
            trade_id="trade-001",
            ticker="NVDA",
            side="BUY",
            entry_price=140.00,
            entry_date="2026-01-06T14:35:00Z",
            exit_price=155.00,
            exit_date="2026-01-13T15:30:00Z",
            quantity=10,
            exit_reason="take_profit",
            signal_sources=["grok", "perplexity"],
            conviction_at_entry=85,
            spy_return_same_period=0.025  # 2.5%
        )

        assert outcome_id is not None, "Should return outcome ID"

        # Verify metrics were calculated
        outcome = engine.get_trade_outcome(outcome_id)
        assert outcome is not None, "Should retrieve outcome"
        assert outcome["realized_pnl"] == 150.00, f"Wrong P&L: {outcome.get('realized_pnl')}"
        assert abs(outcome["realized_pnl_pct"] - 0.1071) < 0.01, f"Wrong P&L %: {outcome.get('realized_pnl_pct')}"
        assert outcome["alpha"] is not None, "Should calculate alpha"

        print(f"  PASSED: Outcome recorded (P&L=${outcome['realized_pnl']}, alpha={outcome['alpha']:.2%})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_auto_detect_position_close():
    """Test automatic detection of position closure."""
    print("\n[TEST] test_auto_detect_position_close")

    db = create_test_db()

    # Mock executor with position changes
    mock_executor = MagicMock()

    engine = LearningEngine(db=db, executor=mock_executor)

    try:
        # Simulate previous positions
        previous_positions = {"NVDA": {"quantity": 10, "entry_price": 140.00}}

        # Simulate current positions (NVDA sold)
        current_positions = {}

        # Detect closed positions
        closed = engine.detect_closed_positions(
            previous_positions=previous_positions,
            current_positions=current_positions,
            current_prices={"NVDA": 155.00}
        )

        assert len(closed) == 1, f"Should detect 1 closed position, got {len(closed)}"
        assert closed[0]["ticker"] == "NVDA", "Should detect NVDA closure"
        assert closed[0]["exit_price"] == 155.00, "Should have exit price"

        print(f"  PASSED: Detected {len(closed)} closed position(s)")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


# =============================================================================
# TEST 2: LIVE SPY DATA FETCHING
# =============================================================================

def test_fetch_spy_price():
    """Test fetching current SPY price."""
    print("\n[TEST] test_fetch_spy_price")

    db = create_test_db()

    # Mock executor that returns SPY data
    mock_executor = MagicMock()
    mock_executor.get_current_price = AsyncMock(return_value=485.50)

    engine = LearningEngine(db=db, executor=mock_executor)

    try:
        # This should work with async
        price = asyncio.get_event_loop().run_until_complete(
            engine.fetch_spy_price()
        )

        assert price is not None, "Should return price"
        assert price > 0, "Price should be positive"

        print(f"  PASSED: SPY price fetched (${price})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_calculate_spy_period_return():
    """Test calculating SPY return over a specific period."""
    print("\n[TEST] test_calculate_spy_period_return")

    db = create_test_db()
    mock_executor = MagicMock()
    engine = LearningEngine(db=db, executor=mock_executor)

    try:
        # Calculate return from price data
        spy_return = engine.calculate_spy_period_return(
            start_price=475.00,
            end_price=485.50
        )

        expected = (485.50 - 475.00) / 475.00  # 2.21%
        assert abs(spy_return - expected) < 0.001, f"Wrong return: {spy_return}"

        print(f"  PASSED: SPY period return = {spy_return:.2%}")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_store_daily_spy_benchmark():
    """Test storing daily SPY benchmark."""
    print("\n[TEST] test_store_daily_spy_benchmark")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Store benchmark
        engine.store_spy_benchmark(
            date="2026-01-13",
            open_price=483.00,
            close_price=485.50,
            high=486.20,
            low=482.10
        )

        # Retrieve it
        benchmark = engine.get_spy_benchmark("2026-01-13")
        assert benchmark is not None, "Should retrieve benchmark"
        assert benchmark["close_price"] == 485.50, "Wrong close price"

        print(f"  PASSED: SPY benchmark stored and retrieved")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


# =============================================================================
# TEST 3: WEEKLY PERFORMANCE DIGEST
# =============================================================================

def test_generate_weekly_digest():
    """Test generating weekly performance digest."""
    print("\n[TEST] test_generate_weekly_digest")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Generate digest with sample data
        digest = engine.generate_weekly_digest(
            trades=SAMPLE_TRADE_HISTORY,
            start_date="2026-01-06",
            end_date="2026-01-13"
        )

        assert digest is not None, "Should return digest"
        assert "total_trades" in digest, "Missing total_trades"
        assert "win_rate" in digest, "Missing win_rate"
        assert "total_pnl" in digest, "Missing total_pnl"
        assert "total_alpha" in digest, "Missing total_alpha"
        assert "top_winner" in digest, "Missing top_winner"
        assert "top_loser" in digest, "Missing top_loser"

        # Verify calculations
        assert digest["total_trades"] == 5, f"Wrong trade count: {digest['total_trades']}"
        assert digest["win_rate"] == 0.6, f"Wrong win rate: {digest['win_rate']}"  # 3/5

        print(f"  PASSED: Weekly digest generated (WR={digest['win_rate']:.0%}, P&L=${digest['total_pnl']})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_format_weekly_digest_telegram():
    """Test formatting weekly digest for Telegram."""
    print("\n[TEST] test_format_weekly_digest_telegram")

    from notifications.telegram_bot import TelegramBot

    bot = TelegramBot(token="test", chat_id="123")

    digest = {
        "period": "2026-01-06 to 2026-01-13",
        "total_trades": 5,
        "wins": 3,
        "losses": 2,
        "win_rate": 0.6,
        "total_pnl": 340.00,
        "total_alpha": 0.13,
        "avg_hold_days": 4.2,
        "top_winner": {"ticker": "SMCI", "pnl": 220.00, "return_pct": 0.15},
        "top_loser": {"ticker": "TSLA", "pnl": -80.00, "return_pct": -0.04},
        "best_source": "perplexity",
        "lessons": ["Semiconductor trades outperforming", "Avoid low-conviction entries"]
    }

    try:
        message = bot.format_weekly_digest(digest)

        assert "WEEKLY" in message.upper(), "Missing weekly header"
        assert "60%" in message or "0.6" in message, "Missing win rate"
        assert "340" in message, "Missing total P&L"
        assert "SMCI" in message, "Missing top winner"
        assert len(message) < 4096, "Message too long"

        print(f"  PASSED: Weekly digest formatted ({len(message)} chars)")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


# =============================================================================
# TEST 4: DYNAMIC THRESHOLD ADJUSTMENT
# =============================================================================

def test_calculate_optimal_threshold():
    """Test calculating optimal conviction threshold from history."""
    print("\n[TEST] test_calculate_optimal_threshold")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Historical trades with conviction and outcome
        historical_trades = [
            {"conviction": 90, "won": True},
            {"conviction": 88, "won": True},
            {"conviction": 85, "won": True},
            {"conviction": 85, "won": False},
            {"conviction": 82, "won": True},
            {"conviction": 82, "won": False},
            {"conviction": 80, "won": False},
            {"conviction": 78, "won": False},
            {"conviction": 75, "won": False},
        ]

        optimal = engine.calculate_optimal_threshold(historical_trades)

        assert optimal is not None, "Should return threshold"
        assert 80 <= optimal <= 90, f"Threshold should be 80-90, got {optimal}"

        print(f"  PASSED: Optimal threshold = {optimal}")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_regime_based_threshold_adjustment():
    """Test adjusting thresholds based on market regime."""
    print("\n[TEST] test_regime_based_threshold_adjustment")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Base threshold
        base_threshold = 80

        # Test different regimes
        normal_threshold = engine.adjust_threshold_for_regime(
            base_threshold=base_threshold,
            vix_regime="normal",
            spy_trend="up"
        )

        elevated_threshold = engine.adjust_threshold_for_regime(
            base_threshold=base_threshold,
            vix_regime="elevated",
            spy_trend="down"
        )

        # In elevated VIX + down trend, should require higher conviction
        assert elevated_threshold > normal_threshold, "Elevated regime should have higher threshold"
        assert elevated_threshold >= 85, "Should be at least 85 in risky regime"

        print(f"  PASSED: Normal={normal_threshold}, Elevated={elevated_threshold}")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_get_dynamic_threshold():
    """Test getting current dynamic threshold."""
    print("\n[TEST] test_get_dynamic_threshold")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        threshold = engine.get_dynamic_threshold()

        assert threshold is not None, "Should return threshold"
        assert 70 <= threshold <= 95, f"Threshold should be reasonable: {threshold}"

        print(f"  PASSED: Dynamic threshold = {threshold}")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


# =============================================================================
# TEST 5: PATTERN LIBRARY WITH HISTORICAL BACKREFERENCES
# =============================================================================

def test_save_winning_pattern():
    """Test saving a successful trade pattern."""
    print("\n[TEST] test_save_winning_pattern")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        pattern_id = engine.save_pattern(
            pattern_type="earnings_momentum",
            ticker="NVDA",
            entry_catalyst="Strong Q4 guidance + AI narrative surge",
            entry_signals={"grok": "bullish", "perplexity": "bullish", "chatgpt": "bullish"},
            outcome="win",
            return_pct=10.7,
            hold_days=7,
            market_regime="normal_vix_up_trend",
            conviction_at_entry=85,
            notes="3/3 AI agreement, high conviction"
        )

        assert pattern_id is not None, "Should return pattern ID"

        # Retrieve pattern
        pattern = engine.get_pattern(pattern_id)
        assert pattern is not None, "Should retrieve pattern"
        assert pattern["ticker"] == "NVDA", "Wrong ticker"

        print(f"  PASSED: Pattern saved (ID={pattern_id[:8]})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_find_similar_patterns():
    """Test finding historically similar patterns."""
    print("\n[TEST] test_find_similar_patterns")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # First, save some patterns
        engine.save_pattern(
            pattern_type="earnings_momentum",
            ticker="NVDA",
            entry_catalyst="Strong guidance",
            outcome="win",
            return_pct=10.7,
            market_regime="normal_vix_up_trend"
        )

        engine.save_pattern(
            pattern_type="earnings_momentum",
            ticker="AMD",
            entry_catalyst="Strong guidance",
            outcome="win",
            return_pct=8.5,
            market_regime="normal_vix_up_trend"
        )

        engine.save_pattern(
            pattern_type="sector_rotation",
            ticker="XLE",
            entry_catalyst="Oil price surge",
            outcome="loss",
            return_pct=-3.2,
            market_regime="elevated_vix_down_trend"
        )

        # Find similar to current situation
        similar = engine.find_similar_patterns(
            pattern_type="earnings_momentum",
            market_regime="normal_vix_up_trend"
        )

        assert len(similar) >= 2, f"Should find at least 2 similar patterns, got {len(similar)}"

        # Calculate historical win rate for this pattern
        win_rate = sum(1 for p in similar if p["outcome"] == "win") / len(similar)
        assert win_rate == 1.0, f"All matching patterns were wins, got {win_rate}"

        print(f"  PASSED: Found {len(similar)} similar patterns (WR={win_rate:.0%})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_generate_pattern_context():
    """Test generating pattern context for Claude."""
    print("\n[TEST] test_generate_pattern_context")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        # Save some patterns first
        engine.save_pattern(
            pattern_type="earnings_momentum",
            ticker="NVDA",
            entry_catalyst="Strong guidance",
            outcome="win",
            return_pct=10.7,
            market_regime="normal_vix_up_trend"
        )

        # Generate context
        context = engine.generate_pattern_context(
            ticker="AMD",
            pattern_type="earnings_momentum",
            current_regime="normal_vix_up_trend"
        )

        assert context is not None, "Should return context"
        assert "similar_patterns" in context, "Missing similar_patterns"
        assert "historical_win_rate" in context, "Missing historical_win_rate"
        assert "avg_return" in context, "Missing avg_return"

        print(f"  PASSED: Pattern context generated")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_pattern_library_stats():
    """Test getting pattern library statistics."""
    print("\n[TEST] test_pattern_library_stats")

    db = create_test_db()
    engine = LearningEngine(db=db)

    try:
        stats = engine.get_pattern_library_stats()

        assert stats is not None, "Should return stats"
        assert "total_patterns" in stats, "Missing total_patterns"
        assert "by_type" in stats, "Missing by_type breakdown"
        assert "overall_win_rate" in stats, "Missing overall_win_rate"

        print(f"  PASSED: Pattern library stats retrieved")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("FINAL PHASE 2/3 COMPLETION TESTS - TDD")
    print("=" * 60)

    tests = [
        # 1. Trade Outcome Recording
        test_record_trade_outcome,
        test_auto_detect_position_close,
        # 2. Live SPY Data
        test_fetch_spy_price,
        test_calculate_spy_period_return,
        test_store_daily_spy_benchmark,
        # 3. Weekly Digest
        test_generate_weekly_digest,
        test_format_weekly_digest_telegram,
        # 4. Dynamic Thresholds
        test_calculate_optimal_threshold,
        test_regime_based_threshold_adjustment,
        test_get_dynamic_threshold,
        # 5. Pattern Library
        test_save_winning_pattern,
        test_find_similar_patterns,
        test_generate_pattern_context,
        test_pattern_library_stats,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[TEST] {test.__name__}")
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nTests failing as expected (TDD Step 2)")
        print("-> Now implementing...")
    else:
        print("\nAll tests passing!")

    return failed == 0


if __name__ == "__main__":
    run_all_tests()
