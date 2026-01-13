#!/usr/bin/env python3
"""
TDD Tests for Learning Engine
Tests for trade outcome tracking, signal accuracy, SPY benchmarking, and context generation.

Run with: python test_learning_engine.py
"""

import sys
import os
import importlib.util
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_TRADE_ENTRY = {
    "trade_id": "trade-001",
    "ticker": "NVDA",
    "side": "BUY",
    "entry_price": 140.00,
    "entry_date": "2026-01-06T14:35:00Z",
    "entry_conviction": 85,
    "entry_thesis": "AI infrastructure demand strong",
    "quantity": 10,
    "primary_signal_source": "chatgpt",
    "signal_sources_agreed": 2
}

SAMPLE_TRADE_EXIT = {
    "exit_price": 152.00,
    "exit_date": "2026-01-10T16:00:00Z",
    "exit_reason": "take_profit"
}

SAMPLE_SPY_DATA = {
    "2026-01-06": {"close": 480.00},
    "2026-01-10": {"close": 485.00}  # +1.04% over same period
}

SAMPLE_SIGNALS = [
    {
        "id": "sig-001",
        "source": "grok",
        "ticker": "NVDA",
        "predicted_direction": "bullish",
        "conviction": 78,
        "signal_date": "2026-01-05T14:00:00Z",
        "price_at_signal": 138.00
    },
    {
        "id": "sig-002",
        "source": "perplexity",
        "ticker": "SMCI",
        "predicted_direction": "bullish",
        "conviction": 82,
        "signal_date": "2026-01-05T14:00:00Z",
        "price_at_signal": 35.00
    }
]

SAMPLE_REGIME = {
    "vix_level": 18.5,
    "vix_regime": "normal",
    "spy_trend": "up",
    "spy_20d_return": 0.032
}


# =============================================================================
# MOCK DATABASE
# =============================================================================

class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.trade_outcomes = []
        self.signal_performance = []
        self.source_reliability = {}
        self.learning_stats = {}
        self.spy_benchmarks = {}
        self.market_regimes = []

    def save_trade_outcome(self, outcome):
        self.trade_outcomes.append(outcome)

    def get_trade_outcomes(self, days_back=30):
        return self.trade_outcomes

    def save_signal_performance(self, perf):
        self.signal_performance.append(perf)

    def update_source_reliability(self, source, period, stats):
        self.source_reliability[(source, period)] = stats

    def get_source_reliability(self, source, period):
        return self.source_reliability.get((source, period))

    def save_learning_stats(self, stat_type, key, period, stats):
        self.learning_stats[(stat_type, key, period)] = stats

    def get_learning_stats(self, stat_type, key, period):
        return self.learning_stats.get((stat_type, key, period))

    def save_spy_benchmark(self, date, data):
        self.spy_benchmarks[date] = data

    def get_spy_benchmark(self, date):
        return self.spy_benchmarks.get(date)

    def save_market_regime(self, regime):
        self.market_regimes.append(regime)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_calculate_trade_metrics():
    """Test trade performance calculation."""
    print("\n[TEST] test_calculate_trade_metrics")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        metrics = engine.calculate_trade_metrics(
            entry=SAMPLE_TRADE_ENTRY,
            exit=SAMPLE_TRADE_EXIT,
            spy_return=0.0104  # SPY did +1.04% same period
        )

        # Check P&L calculation
        # Bought at 140, sold at 152 = +8.57%
        assert metrics["realized_pnl"] == 120.00, f"Expected 120, got {metrics['realized_pnl']}"
        assert abs(metrics["realized_pnl_pct"] - 0.0857) < 0.001, f"Expected ~8.57%, got {metrics['realized_pnl_pct']}"

        # Check alpha (our return - SPY return)
        # 8.57% - 1.04% = 7.53%
        assert abs(metrics["alpha"] - 0.0753) < 0.001, f"Expected ~7.53% alpha, got {metrics['alpha']}"

        # Check hold time
        # Jan 6 14:35 to Jan 10 16:00 = ~4 days
        assert metrics["hold_time_hours"] > 90, f"Expected >90 hours, got {metrics['hold_time_hours']}"

        print(f"  PASSED: P&L=${metrics['realized_pnl']}, Alpha={metrics['alpha']:.2%}")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_calculate_win_rate():
    """Test win rate calculation."""
    print("\n[TEST] test_calculate_win_rate")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Add some trade outcomes
        db.trade_outcomes = [
            {"realized_pnl": 100, "realized_pnl_pct": 0.05},   # Win
            {"realized_pnl": -50, "realized_pnl_pct": -0.03},  # Loss
            {"realized_pnl": 80, "realized_pnl_pct": 0.04},    # Win
            {"realized_pnl": 120, "realized_pnl_pct": 0.06},   # Win
            {"realized_pnl": -30, "realized_pnl_pct": -0.02},  # Loss
        ]

        stats = engine.calculate_performance_stats(db.trade_outcomes)

        # 3 wins, 2 losses = 60% win rate
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert abs(stats["win_rate"] - 0.60) < 0.01

        # Avg return = (5 - 3 + 4 + 6 - 2) / 5 = 2%
        assert abs(stats["avg_return"] - 0.02) < 0.01

        print(f"  PASSED: Win rate={stats['win_rate']:.0%}, Avg return={stats['avg_return']:.2%}")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_source_reliability_scoring():
    """Test source accuracy tracking."""
    print("\n[TEST] test_source_reliability_scoring")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Simulate signal performance data
        signal_results = [
            {"source": "grok", "accurate_5d": True},
            {"source": "grok", "accurate_5d": False},
            {"source": "grok", "accurate_5d": True},
            {"source": "perplexity", "accurate_5d": True},
            {"source": "perplexity", "accurate_5d": True},
            {"source": "perplexity", "accurate_5d": True},
            {"source": "chatgpt", "accurate_5d": True},
            {"source": "chatgpt", "accurate_5d": False},
        ]

        reliability = engine.calculate_source_reliability(signal_results)

        # Grok: 2/3 = 66.7%
        assert abs(reliability["grok"]["accuracy_rate"] - 0.667) < 0.01

        # Perplexity: 3/3 = 100%
        assert reliability["perplexity"]["accuracy_rate"] == 1.0

        # ChatGPT: 1/2 = 50%
        assert reliability["chatgpt"]["accuracy_rate"] == 0.5

        print(f"  PASSED: Grok={reliability['grok']['accuracy_rate']:.0%}, Perplexity={reliability['perplexity']['accuracy_rate']:.0%}")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_conviction_calibration():
    """Test conviction accuracy calibration."""
    print("\n[TEST] test_conviction_calibration")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Trades with different conviction levels
        trades = [
            {"entry_conviction": 90, "realized_pnl_pct": 0.05},   # High conv, win
            {"entry_conviction": 88, "realized_pnl_pct": 0.03},   # High conv, win
            {"entry_conviction": 86, "realized_pnl_pct": -0.02},  # High conv, loss
            {"entry_conviction": 82, "realized_pnl_pct": 0.02},   # Med conv, win
            {"entry_conviction": 81, "realized_pnl_pct": -0.04},  # Med conv, loss
            {"entry_conviction": 80, "realized_pnl_pct": -0.03},  # Med conv, loss
        ]

        calibration = engine.calculate_conviction_calibration(trades)

        # High conviction (85+): 2/3 = 66.7%
        assert "85-100" in calibration
        assert abs(calibration["85-100"]["actual_win_rate"] - 0.667) < 0.01

        # Medium conviction (80-84): 1/3 = 33.3%
        assert "80-84" in calibration
        assert abs(calibration["80-84"]["actual_win_rate"] - 0.333) < 0.01

        print(f"  PASSED: High conv WR={calibration['85-100']['actual_win_rate']:.0%}, Med conv WR={calibration['80-84']['actual_win_rate']:.0%}")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_generate_claude_context():
    """Test context generation for Claude."""
    print("\n[TEST] test_generate_claude_context")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Set up some data
        db.learning_stats[("overall", None, "30d")] = {
            "total_trades": 20,
            "win_rate": 0.65,
            "avg_return": 0.028,
            "alpha": 0.015
        }

        db.source_reliability[("grok", "30d")] = {"accuracy_rate": 0.58}
        db.source_reliability[("perplexity", "30d")] = {"accuracy_rate": 0.72}

        context = engine.generate_claude_context()

        # Should have all required sections
        assert "performance_summary" in context
        assert "source_reliability" in context
        assert "current_regime" in context or context.get("regime_not_available")

        # Performance should reflect our data
        perf = context["performance_summary"]
        assert perf["total_trades"] == 20
        assert perf["win_rate"] == 0.65

        print(f"  PASSED: Context has {len(context)} sections")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_market_regime_detection():
    """Test market regime classification."""
    print("\n[TEST] test_market_regime_detection")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Test different VIX levels
        assert engine.classify_vix_regime(12) == "low"
        assert engine.classify_vix_regime(20) == "normal"
        assert engine.classify_vix_regime(30) == "elevated"
        assert engine.classify_vix_regime(45) == "extreme"

        # Test SPY trend classification
        assert engine.classify_spy_trend(0.08) == "strong_up"
        assert engine.classify_spy_trend(0.02) == "up"
        assert engine.classify_spy_trend(0.005) == "sideways"
        assert engine.classify_spy_trend(-0.03) == "down"
        assert engine.classify_spy_trend(-0.10) == "strong_down"

        print("  PASSED: Regime classification working")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_smart_scheduling():
    """Test smart scan scheduling (2x daily, no weekends)."""
    print("\n[TEST] test_smart_scheduling")

    try:
        from learning_engine import SmartScheduler

        scheduler = SmartScheduler()

        # Monday 9:35 AM ET = 14:35 UTC - should scan (morning)
        monday_morning = datetime(2026, 1, 12, 14, 35, tzinfo=timezone.utc)
        assert scheduler.should_scan(monday_morning) == True

        # Monday 12:30 PM ET = 17:30 UTC - should scan (midday)
        monday_midday = datetime(2026, 1, 12, 17, 30, tzinfo=timezone.utc)
        assert scheduler.should_scan(monday_midday) == True

        # Monday 3:00 PM ET = 20:00 UTC - should NOT scan (not a scan time)
        monday_afternoon = datetime(2026, 1, 12, 20, 0, tzinfo=timezone.utc)
        assert scheduler.should_scan(monday_afternoon) == False

        # Saturday - should NOT scan
        saturday = datetime(2026, 1, 17, 14, 35, tzinfo=timezone.utc)
        assert scheduler.should_scan(saturday) == False

        # Sunday - should NOT scan
        sunday = datetime(2026, 1, 18, 14, 35, tzinfo=timezone.utc)
        assert scheduler.should_scan(sunday) == False

        print("  PASSED: Smart scheduling working")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_spy_alpha_calculation():
    """Test SPY benchmark comparison."""
    print("\n[TEST] test_spy_alpha_calculation")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        # Our trades returned 12% total
        # SPY returned 8% same period
        our_return = 0.12
        spy_return = 0.08

        alpha = engine.calculate_alpha_simple(our_return, spy_return)

        # Alpha = 12% - 8% = 4%
        assert abs(alpha - 0.04) < 0.001

        # Test negative alpha
        alpha_negative = engine.calculate_alpha_simple(0.05, 0.10)
        assert abs(alpha_negative - (-0.05)) < 0.001

        print(f"  PASSED: Alpha calculation working (+{alpha:.1%} outperformance)")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_format_performance_summary():
    """Test human-readable performance summary."""
    print("\n[TEST] test_format_performance_summary")

    try:
        from learning_engine import LearningEngine

        db = MockDatabase()
        engine = LearningEngine(db=db, executor=None)

        stats = {
            "total_trades": 25,
            "win_rate": 0.64,
            "avg_return": 0.032,
            "total_return": 0.18,
            "alpha": 0.045,
            "sharpe_ratio": 1.3
        }

        summary = engine.format_performance_summary(stats)

        # Should be a readable string
        assert isinstance(summary, str)
        assert "25" in summary  # total trades
        assert "64" in summary  # win rate
        assert "SPY" in summary.upper() or "ALPHA" in summary.upper()

        print(f"  PASSED: Summary generated ({len(summary)} chars)")
        return True

    except ImportError as e:
        print(f"  FAILED (import): {e}")
        return False
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("LEARNING ENGINE TESTS - TDD PHASE")
    print("=" * 60)

    tests = [
        test_calculate_trade_metrics,
        test_calculate_win_rate,
        test_source_reliability_scoring,
        test_conviction_calibration,
        test_generate_claude_context,
        test_market_regime_detection,
        test_smart_scheduling,
        test_spy_alpha_calculation,
        test_format_performance_summary,
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
        print("\n✓ Tests failing as expected (TDD Step 2 complete)")
        print("→ Now implementing the Learning Engine...")
    else:
        print("\n✓ All tests passing!")

    return failed == 0


if __name__ == "__main__":
    run_all_tests()
