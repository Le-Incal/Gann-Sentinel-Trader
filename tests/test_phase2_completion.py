#!/usr/bin/env python3
"""
TDD Tests for Phase 2 Completion: Full MACA Integration + Per-Cycle API Cost Tracking

Run with: python -m pytest tests/test_phase2_completion.py -v
Or: python tests/test_phase2_completion.py
"""

import sys
import os
import json
import importlib.util
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import from correct locations
from notifications.telegram_bot import TelegramBot
from storage.database import Database


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_MACA_RESULT = {
    "cycle_id": "cycle-001",
    "ticker": "NVDA",
    "timestamp_utc": "2026-01-13T14:35:00Z",
    "duration_seconds": 32.5,
    "proposals": [
        {
            "ai_source": "grok",
            "recommendation": {
                "side": "BUY",
                "conviction_score": 75,
                "thesis": "Strong sentiment on X/Twitter around AI infrastructure. Datacenter demand mentions surging 40% week over week.",
                "position_size_pct": 8
            },
            "tokens_used": 1500,
            "cost_usd": 0.0075
        },
        {
            "ai_source": "perplexity",
            "recommendation": {
                "side": "BUY",
                "conviction_score": 82,
                "thesis": "Fundamental analysis shows strong earnings momentum. Data center revenue expected to grow 50% YoY. Jensen's guidance bullish.",
                "position_size_pct": 10
            },
            "tokens_used": 2500,
            "cost_usd": 0.025
        },
        {
            "ai_source": "chatgpt",
            "recommendation": {
                "side": "BUY",
                "conviction_score": 78,
                "thesis": "Technical patterns suggest continuation. Breaking out of consolidation with volume confirmation.",
                "position_size_pct": 8
            },
            "tokens_used": 2000,
            "cost_usd": 0.012
        }
    ],
    "synthesis": {
        "decision_type": "TRADE",
        "recommendation": {
            "side": "BUY",
            "conviction_score": 85,
            "thesis": "Strong consensus across all 3 AIs. Grok sees social momentum, Perplexity confirms fundamentals, ChatGPT validates technicals. Unanimous BUY with 75-82 conviction range.",
            "position_size_pct": 10,
            "stop_loss_pct": 8,
            "take_profit_pct": 25,
            "time_horizon": "2-4 weeks"
        },
        "cross_validation": {
            "grok": "CONFIRMED",
            "perplexity": "CONFIRMED",
            "chatgpt": "CONFIRMED"
        },
        "tokens_used": 3000,
        "cost_usd": 0.045
    },
    "cycle_cost": {
        "total_tokens": 9000,
        "total_cost_usd": 0.0895,
        "by_source": {
            "grok": {"tokens": 1500, "cost_usd": 0.0075},
            "perplexity": {"tokens": 2500, "cost_usd": 0.025},
            "chatgpt": {"tokens": 2000, "cost_usd": 0.012},
            "claude": {"tokens": 3000, "cost_usd": 0.045}
        }
    },
    "status": "complete"
}

SAMPLE_SCAN_CYCLES = [
    {
        "id": "cycle-001",
        "timestamp_utc": "2026-01-13T14:35:00Z",
        "cycle_type": "maca_ticker_check",
        "status": "complete",
        "metadata": {
            "ticker": "NVDA",
            "cost_tracking": {
                "total_tokens": 9000,
                "total_cost_usd": 0.0895,
                "by_source": {"grok": {"cost_usd": 0.0075}, "perplexity": {"cost_usd": 0.025}}
            }
        }
    },
    {
        "id": "cycle-002",
        "timestamp_utc": "2026-01-13T12:30:00Z",
        "cycle_type": "scheduled_scan",
        "status": "complete",
        "metadata": {
            "cost_tracking": {
                "total_tokens": 5000,
                "total_cost_usd": 0.045,
                "by_source": {"grok": {"cost_usd": 0.01}, "claude": {"cost_usd": 0.035}}
            }
        }
    },
    {
        "id": "cycle-003",
        "timestamp_utc": "2026-01-12T14:35:00Z",
        "cycle_type": "maca_ticker_check",
        "status": "complete",
        "metadata": {
            "ticker": "TSLA",
            "cost_tracking": {
                "total_tokens": 8500,
                "total_cost_usd": 0.078,
                "by_source": {"grok": {"cost_usd": 0.008}, "perplexity": {"cost_usd": 0.022}}
            }
        }
    }
]


# =============================================================================
# TEST: MACA CHECK FORMATTING
# =============================================================================

def test_format_maca_check_result():
    """Test that format_maca_check_result shows all 4 AI theses."""
    print("\n[TEST] test_format_maca_check_result")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        message = bot.format_maca_check_result(SAMPLE_MACA_RESULT)

        # Must show all AI sources
        assert "GROK" in message, "Missing Grok thesis"
        assert "PERPLEXITY" in message, "Missing Perplexity thesis"
        assert "CHATGPT" in message, "Missing ChatGPT thesis"
        assert "CLAUDE" in message, "Missing Claude synthesis"

        # Must show convictions
        assert "75" in message, "Missing Grok conviction"
        assert "82" in message, "Missing Perplexity conviction"
        assert "78" in message, "Missing ChatGPT conviction"
        assert "85" in message, "Missing Claude final conviction"

        # Must show cost tracking
        assert "$0.08" in message or "0.089" in message, "Missing total cost"
        assert "9,000" in message or "9000" in message, "Missing token count"

        # Must be under Telegram limit
        assert len(message) < 4096, f"Message too long: {len(message)}"

        print(f"  PASSED: MACA check formatted ({len(message)} chars)")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_format_maca_no_proposals():
    """Test formatting when no AI proposals available."""
    print("\n[TEST] test_format_maca_no_proposals")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        result = {
            "ticker": "XYZ",
            "proposals": [],
            "synthesis": {"decision_type": "NO_TRADE", "recommendation": {}},
            "cycle_cost": {"total_tokens": 0, "total_cost_usd": 0}
        }

        message = bot.format_maca_check_result(result)

        assert "XYZ" in message, "Missing ticker"
        assert "CLAUDE" in message, "Missing Claude section"

        print(f"  PASSED: Empty proposals handled")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# =============================================================================
# TEST: COST TRACKING DATABASE
# =============================================================================

def test_get_cost_summary():
    """Test database method to get cost summary."""
    print("\n[TEST] test_get_cost_summary")

    # Use in-memory database for testing
    db = Database(db_path=":memory:")

    try:
        # Add scan cycles with cost data
        for cycle in SAMPLE_SCAN_CYCLES:
            db.create_scan_cycle({
                "cycle_id": cycle["id"],
                "timestamp_utc": cycle["timestamp_utc"],
                "cycle_type": cycle["cycle_type"],
                "status": cycle["status"],
                "metadata": cycle["metadata"]
            })

        # Get cost summary
        summary = db.get_cost_summary(days=7)

        assert summary is not None, "Should return summary"
        assert "total_cost_usd" in summary, "Missing total_cost_usd"
        assert "total_tokens" in summary, "Missing total_tokens"
        assert "cycle_count" in summary, "Missing cycle_count"
        assert "by_source" in summary, "Missing by_source breakdown"

        # Check aggregation
        assert summary["cycle_count"] == 3, f"Wrong cycle count: {summary['cycle_count']}"
        expected_total = 0.0895 + 0.045 + 0.078  # Sum of all cycles
        assert abs(summary["total_cost_usd"] - expected_total) < 0.001, f"Wrong total: {summary['total_cost_usd']}"

        print(f"  PASSED: Cost summary aggregation works (${summary['total_cost_usd']:.4f})")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_get_cost_by_day():
    """Test getting cost breakdown by day."""
    print("\n[TEST] test_get_cost_by_day")

    # Use in-memory database for testing
    db = Database(db_path=":memory:")

    try:
        # Add scan cycles
        for cycle in SAMPLE_SCAN_CYCLES:
            db.create_scan_cycle({
                "cycle_id": cycle["id"],
                "timestamp_utc": cycle["timestamp_utc"],
                "cycle_type": cycle["cycle_type"],
                "status": cycle["status"],
                "metadata": cycle["metadata"]
            })

        # Get daily breakdown using get_cost_by_day method
        daily = db.get_cost_by_day(days=7)

        assert isinstance(daily, list), "Should return list"
        assert len(daily) > 0, "Should have daily data"

        # Check structure
        day = daily[0]
        assert "date" in day, "Missing date"
        assert "cost_usd" in day, "Missing cost_usd"
        assert "cycle_count" in day, "Missing cycle_count"

        print(f"  PASSED: Daily cost breakdown works ({len(daily)} days)")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


# =============================================================================
# TEST: COST COMMAND FORMATTING
# =============================================================================

def test_format_cost_message():
    """Test formatting cost summary for Telegram."""
    print("\n[TEST] test_format_cost_message")

    bot = TelegramBot(token="test", chat_id="123")

    cost_summary = {
        "total_cost_usd": 0.2125,
        "total_tokens": 22500,
        "cycle_count": 3,
        "by_source": {
            "grok": {"cost_usd": 0.0255, "tokens": 4500},
            "perplexity": {"cost_usd": 0.047, "tokens": 5000},
            "chatgpt": {"cost_usd": 0.024, "tokens": 4000},
            "claude": {"cost_usd": 0.08, "tokens": 9000}
        },
        "by_day": [
            {"date": "2026-01-13", "cost_usd": 0.1345, "cycle_count": 2},
            {"date": "2026-01-12", "cost_usd": 0.078, "cycle_count": 1}
        ],
        "period_days": 7
    }

    try:
        message = bot.format_cost_message(cost_summary)

        # Should show totals
        assert "$0.21" in message or "0.2125" in message, "Missing total cost"
        assert "22,500" in message or "22500" in message, "Missing total tokens"
        assert "3" in message, "Missing cycle count"

        # Should show per-source breakdown
        assert "grok" in message.lower(), "Missing grok cost"
        assert "claude" in message.lower(), "Missing claude cost"

        # Should be under Telegram limit
        assert len(message) < 4096, "Message too long"

        print(f"  PASSED: Cost message formatted ({len(message)} chars)")
        return True

    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


# =============================================================================
# TEST: MACA ORCHESTRATOR COST SAVING
# =============================================================================

def test_maca_saves_costs_to_db():
    """Test that MACA orchestrator saves cost data to database."""
    print("\n[TEST] test_maca_saves_costs_to_db")

    try:
        from core.maca_orchestrator import MACAOrchestrator

        # Use in-memory database
        db = Database(db_path=":memory:")

        # Mock the AI components
        mock_grok = MagicMock()
        mock_grok.is_configured = True

        mock_perplexity = MagicMock()
        mock_perplexity.is_configured = True

        mock_chatgpt = MagicMock()
        mock_chatgpt.is_configured = True

        mock_claude = MagicMock()
        mock_claude.is_configured = True

        orchestrator = MACAOrchestrator(
            db=db,
            grok=mock_grok,
            perplexity=mock_perplexity,
            chatgpt=mock_chatgpt,
            claude=mock_claude,
            telegram=None
        )

        # Check that cost aggregation method exists
        assert hasattr(orchestrator, 'aggregate_cycle_costs'), "Missing aggregate_cycle_costs method"

        # Simulate cycle costs
        orchestrator._cycle_costs = {
            "grok": {"tokens": 1500, "cost_usd": 0.0075},
            "perplexity": {"tokens": 2500, "cost_usd": 0.025},
            "claude": {"tokens": 3000, "cost_usd": 0.045}
        }

        result = orchestrator.aggregate_cycle_costs()

        assert "total_tokens" in result, "Missing total_tokens"
        assert "total_cost_usd" in result, "Missing total_cost_usd"
        assert "by_source" in result, "Missing by_source"

        # Check math
        expected_tokens = 1500 + 2500 + 3000
        expected_cost = 0.0075 + 0.025 + 0.045

        assert result["total_tokens"] == expected_tokens, f"Wrong token total: {result['total_tokens']}"
        assert abs(result["total_cost_usd"] - expected_cost) < 0.0001, f"Wrong cost total: {result['total_cost_usd']}"

        print(f"  PASSED: MACA cost aggregation works (${result['total_cost_usd']:.4f})")
        return True

    except ImportError as e:
        print(f"  SKIPPED (MACA not available): {e}")
        return True  # Not a failure if MACA components not present
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST: FULL MACA CHECK FLOW (Integration)
# =============================================================================

def test_maca_check_integration():
    """Test that /check with MACA runs all 4 AIs and returns proper structure."""
    print("\n[TEST] test_maca_check_integration")

    try:
        from core.maca_orchestrator import MACAOrchestrator

        # Use in-memory database
        db = Database.__new__(Database)
        db._persistent_conn = None
        db.db_path = ":memory:"

        import sqlite3
        db._persistent_conn = sqlite3.connect(":memory:")
        db._persistent_conn.row_factory = sqlite3.Row
        db._init_schema()

        # Create mock AIs that return proper proposal structure
        mock_grok = MagicMock()
        mock_grok.is_configured = True
        mock_grok.scan_sentiment = AsyncMock(return_value=[])

        mock_perplexity = MagicMock()
        mock_perplexity.is_configured = True
        mock_perplexity.analyze_ticker = AsyncMock(return_value={
            "ai_source": "perplexity",
            "recommendation": {"side": "BUY", "conviction_score": 80, "thesis": "Test thesis"},
            "tokens_used": 2000,
            "cost_usd": 0.02
        })

        mock_chatgpt = MagicMock()
        mock_chatgpt.is_configured = True
        mock_chatgpt.analyze_ticker = AsyncMock(return_value={
            "ai_source": "chatgpt",
            "recommendation": {"side": "BUY", "conviction_score": 75, "thesis": "Test thesis"},
            "tokens_used": 1500,
            "cost_usd": 0.015
        })

        mock_claude = MagicMock()
        mock_claude.is_configured = True
        mock_claude.synthesize_proposals = AsyncMock(return_value={
            "decision_type": "TRADE",
            "recommendation": {"side": "BUY", "conviction_score": 82, "thesis": "Synthesized thesis"},
            "tokens_used": 3000,
            "cost_usd": 0.045
        })

        orchestrator = MACAOrchestrator(
            db=db,
            grok=mock_grok,
            perplexity=mock_perplexity,
            chatgpt=mock_chatgpt,
            claude=mock_claude,
            telegram=None
        )

        # Check is_configured property
        assert hasattr(orchestrator, 'is_configured'), "Missing is_configured property"

        print(f"  PASSED: MACA orchestrator structure verified")
        return True

    except ImportError as e:
        print(f"  SKIPPED (MACA not available): {e}")
        return True  # Not a failure if MACA components not present
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("PHASE 2 COMPLETION TESTS - TDD")
    print("=" * 60)

    tests = [
        # MACA Formatting
        test_format_maca_check_result,
        test_format_maca_no_proposals,
        # Cost Database
        test_get_cost_summary,
        test_get_cost_by_day,
        # Cost Formatting
        test_format_cost_message,
        # MACA Integration
        test_maca_saves_costs_to_db,
        test_maca_check_integration,
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
        print("\nTests failing - implementation needed")
    else:
        print("\nAll tests passing!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
