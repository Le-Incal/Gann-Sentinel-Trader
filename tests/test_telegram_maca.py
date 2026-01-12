#!/usr/bin/env python3
"""
TDD Tests for MACA Telegram Integration
Tests for inline keyboard buttons, AI proposal display, and callback handling.

This test file imports telegram_bot directly without going through __init__.py

Run with: python tests/test_telegram_maca.py
"""

import sys
import os

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from notifications module
from notifications.telegram_bot import TelegramBot

# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_GROK_PROPOSAL = {
    "ai_source": "grok",
    "proposal_type": "NEW_BUY",
    "recommendation": {
        "ticker": "NVDA",
        "side": "BUY",
        "conviction_score": 78,
        "thesis": "Strong AI infrastructure narrative on X/Twitter. Datacenter buildout tweets surging.",
        "time_horizon": "days",
        "catalyst": "CES 2026 keynote Jan 15"
    },
    "supporting_evidence": {
        "bull_case": "AI capex cycle accelerating",
        "bear_case": "Valuation stretched at 35x forward"
    }
}

SAMPLE_PERPLEXITY_PROPOSAL = {
    "ai_source": "perplexity",
    "proposal_type": "NEW_BUY",
    "recommendation": {
        "ticker": "SMCI",
        "side": "BUY",
        "conviction_score": 82,
        "thesis": "Server demand accelerating per channel checks. AI infrastructure spend +40% YoY.",
        "time_horizon": "weeks",
        "catalyst": "Q4 earnings Feb 4"
    },
    "supporting_evidence": {
        "bull_case": "Direct NVDA partnership",
        "bear_case": "Accounting scrutiny history"
    }
}

SAMPLE_CHATGPT_PROPOSAL = {
    "ai_source": "chatgpt",
    "proposal_type": "NEW_BUY",
    "recommendation": {
        "ticker": "NVDA",
        "side": "BUY",
        "conviction_score": 85,
        "thesis": "Technical pattern suggests consolidation breakout imminent. 2.3:1 R/R ratio.",
        "time_horizon": "weeks",
        "catalyst": "Sector rotation into AI"
    },
    "supporting_evidence": {
        "bull_case": "Pattern recognition shows high-probability setup",
        "bear_case": "Crowded trade, potential for sharp reversal"
    }
}

SAMPLE_SYNTHESIS = {
    "decision_type": "TRADE",
    "selected_proposal": {
        "ai_source": "chatgpt",
        "modifications": "Adjusted position size from 15% to 12%"
    },
    "recommendation": {
        "ticker": "NVDA",
        "side": "BUY",
        "conviction_score": 83,
        "thesis": "Two analysts converged on NVDA with complementary theses.",
        "position_size_pct": 12,
        "stop_loss_pct": 8,
        "time_horizon": "weeks"
    },
    "cross_validation": {
        "fred_alignment": "neutral",
        "polymarket_alignment": "supports",
        "technical_alignment": "supports"
    },
    "rationale": "Convergence on NVDA with technical confirmation."
}

SAMPLE_TECHNICAL = [
    {
        "ticker": "NVDA",
        "current_price": 142.50,
        "market_state": {"state": "trending", "bias": "bullish", "confidence": "high"},
        "trend_channel": {"position_in_channel": 0.65},
        "verdict": "hypothesis"
    }
]

SAMPLE_PORTFOLIO = {
    "equity": 100000.00,
    "cash": 85000.00,
    "position_count": 1
}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_build_approval_keyboard():
    """Test building approve/reject inline keyboard buttons."""
    print("\n[TEST] test_build_approval_keyboard")

    bot = TelegramBot(token="test", chat_id="123")
    trade_id = "abc12345"

    try:
        keyboard = bot.build_approval_keyboard(trade_id)

        assert "inline_keyboard" in keyboard
        assert len(keyboard["inline_keyboard"]) >= 1

        # Check approve button exists
        row1 = keyboard["inline_keyboard"][0]
        assert any("APPROVE" in btn.get("text", "").upper() for btn in row1)
        assert any(f"approve_{trade_id}" in btn.get("callback_data", "") for btn in row1)

        print("  PASSED: build_approval_keyboard works correctly")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False
    except AssertionError as e:
        print(f"  FAILED (assertion): {e}")
        return False


def test_build_command_keyboard():
    """Test building quick command keyboard."""
    print("\n[TEST] test_build_command_keyboard")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        keyboard = bot.build_command_keyboard()

        assert "inline_keyboard" in keyboard

        # Check command buttons exist
        has_cmd_callback = False
        for row in keyboard["inline_keyboard"]:
            for btn in row:
                if "cmd_" in btn.get("callback_data", ""):
                    has_cmd_callback = True

        assert has_cmd_callback, "No cmd_ callbacks found"

        print("  PASSED: build_command_keyboard works correctly")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_format_ai_proposal():
    """Test formatting individual AI proposal."""
    print("\n[TEST] test_format_ai_proposal")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        text = bot.format_ai_proposal(SAMPLE_GROK_PROPOSAL)

        assert "GROK" in text.upper(), "Missing GROK identifier"
        assert "NVDA" in text, "Missing ticker"
        assert "78" in text, "Missing conviction score"
        assert len(text) < 1000, "Text too long"

        print(f"  PASSED: format_ai_proposal output:\n{text[:200]}...")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_format_ai_council_message():
    """Test formatting full AI council message (Message 1)."""
    print("\n[TEST] test_format_ai_council_message")

    bot = TelegramBot(token="test", chat_id="123")
    proposals = [SAMPLE_GROK_PROPOSAL, SAMPLE_PERPLEXITY_PROPOSAL, SAMPLE_CHATGPT_PROPOSAL]

    try:
        text = bot.format_ai_council_message(proposals)

        assert "GROK" in text.upper(), "Missing GROK"
        assert "PERPLEXITY" in text.upper(), "Missing PERPLEXITY"
        assert "CHATGPT" in text.upper(), "Missing CHATGPT"
        assert len(text) < 4096, "Exceeds Telegram limit"

        print(f"  PASSED: format_ai_council_message ({len(text)} chars)")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_format_decision_message():
    """Test formatting Claude decision message (Message 2)."""
    print("\n[TEST] test_format_decision_message")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        text = bot.format_decision_message(
            synthesis=SAMPLE_SYNTHESIS,
            technical_signals=SAMPLE_TECHNICAL,
            portfolio=SAMPLE_PORTFOLIO,
            trade_id="abc12345"
        )

        assert "CLAUDE" in text.upper(), "Missing CLAUDE"
        assert "83" in text, "Missing conviction"
        assert "NVDA" in text, "Missing ticker"
        assert len(text) < 4096, "Exceeds Telegram limit"

        print(f"  PASSED: format_decision_message ({len(text)} chars)")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_parse_callback_data():
    """Test parsing callback data from inline buttons."""
    print("\n[TEST] test_parse_callback_data")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        action, trade_id = bot.parse_callback_data("approve_abc12345")
        assert action == "approve"
        assert trade_id == "abc12345"

        action, trade_id = bot.parse_callback_data("reject_xyz99999")
        assert action == "reject"
        assert trade_id == "xyz99999"

        action, param = bot.parse_callback_data("cmd_status")
        assert action == "cmd"
        assert param == "status"

        print("  PASSED: parse_callback_data works correctly")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_parse_update_callback():
    """Test parsing Telegram update with callback_query."""
    print("\n[TEST] test_parse_update_callback")

    bot = TelegramBot(token="test", chat_id="123")

    mock_update = {
        "update_id": 123,
        "callback_query": {
            "id": "query123",
            "data": "approve_abc12345",
            "message": {"chat": {"id": "123"}}
        }
    }

    try:
        result = bot.parse_update(mock_update)

        assert result is not None, "parse_update returned None"
        assert result.get("type") == "callback_query"
        assert result.get("data") == "approve_abc12345"

        print("  PASSED: parse_update handles callback_query")
        return True
    except AttributeError as e:
        print(f"  FAILED (method not found): {e}")
        return False


def test_conviction_bar():
    """Test existing conviction bar method."""
    print("\n[TEST] test_conviction_bar (existing method)")

    bot = TelegramBot(token="test", chat_id="123")

    try:
        bar = bot._build_conviction_bar(85)
        assert len(bar) > 0, "Empty conviction bar"
        print(f"  PASSED: conviction bar = {bar}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MACA TELEGRAM TESTS - TDD PHASE")
    print("=" * 60)

    tests = [
        test_conviction_bar,  # Should pass (existing method)
        test_build_approval_keyboard,
        test_build_command_keyboard,
        test_format_ai_proposal,
        test_format_ai_council_message,
        test_format_decision_message,
        test_parse_callback_data,
        test_parse_update_callback,
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
        print("\nSome tests failed - check implementation")
    else:
        print("\nAll tests passing!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
