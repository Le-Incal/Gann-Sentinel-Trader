#!/usr/bin/env python3
"""
Gann Sentinel Trader - Event Scanner Tests (Standalone)
Tests the LevelFields-style event-driven signal scanner.

Run with: python3 tests/test_event_scanner_standalone.py
"""

import sys
import os
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanners.event_scanner import (
    EventScanner,
    EventSignal,
    EVENT_TYPES,
    EVENT_HISTORICAL_PATTERNS,
)


def test_all_27_event_types_defined():
    """Verify all 27 event types exist in the scanner."""
    print("\n[TEST] test_all_27_event_types_defined")

    expected_events = [
        "CEO_EXIT", "CEO_APPOINTMENT", "EXECUTIVE_DEPARTURE", "INSIDER_BUYING", "INSIDER_SELLING",
        "STOCK_BUYBACK", "DIVIDEND_INCREASE", "DIVIDEND_CUT", "RETURN_OF_CAPITAL",
        "FDA_BREAKTHROUGH", "FDA_APPROVAL", "FDA_REJECTION", "DOJ_INVESTIGATION", "CLASS_ACTION_LAWSUIT",
        "SP500_ADDITION", "SP500_REMOVAL", "INDEX_REBALANCING",
        "ACTIVIST_INVESTOR", "SHORT_SELLER_REPORT", "PROXY_FIGHT",
        "GOVERNMENT_CONTRACT", "MAJOR_PARTNERSHIP", "CONTRACT_LOSS",
        "MA_ANNOUNCEMENT", "SPINOFF", "BANKRUPTCY_FILING", "DEBT_RESTRUCTURING",
    ]

    assert len(EVENT_TYPES) == 27, f"Expected 27 event types, got {len(EVENT_TYPES)}"

    for event in expected_events:
        assert event in EVENT_TYPES, f"Missing event type: {event}"

    print(f"  PASSED: All 27 event types defined")
    return True


def test_event_historical_patterns_complete():
    """Verify each event type has historical pattern data."""
    print("\n[TEST] test_event_historical_patterns_complete")

    for event_type in EVENT_TYPES:
        assert event_type in EVENT_HISTORICAL_PATTERNS, f"Missing historical pattern for: {event_type}"

        pattern = EVENT_HISTORICAL_PATTERNS[event_type]
        assert "bias" in pattern, f"Missing bias for {event_type}"
        assert "avg_move_pct" in pattern, f"Missing avg_move_pct for {event_type}"
        assert "hold_days" in pattern, f"Missing hold_days for {event_type}"
        assert "win_rate" in pattern, f"Missing win_rate for {event_type}"

        assert pattern["bias"] in ["bullish", "bearish", "mixed"], f"Invalid bias for {event_type}"
        assert 0.0 <= pattern["win_rate"] <= 1.0, f"Invalid win_rate for {event_type}"
        assert pattern["hold_days"] > 0, f"Invalid hold_days for {event_type}"

    print(f"  PASSED: All 27 patterns complete with valid data")
    return True


def test_event_signal_creation():
    """Test creating an EventSignal with required fields."""
    print("\n[TEST] test_event_signal_creation")

    signal = EventSignal(
        signal_id="test-123",
        dedup_hash="abc123",
        category="corporate_event",
        source_type="event_scanner",
        event_type="CEO_EXIT",
        asset_scope={"tickers": ["AAPL"], "sectors": ["TECH"], "macro_regions": ["US"], "asset_classes": ["EQUITY"]},
        summary="AAPL: CEO departure announced",
        raw_value={"event_type": "CEO_EXIT", "event_date": "2026-01-13"},
        evidence=[{"source": "SEC", "excerpt": "Apple CEO announces retirement"}],
        confidence=0.75,
        confidence_factors={"source_quality": 0.8, "recency": 0.9},
        directional_bias="mixed",
        time_horizon="weeks",
        novelty="new",
        staleness_policy={"max_age_seconds": 86400},
        uncertainties=["CEO successor not yet named"],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        event_metadata={"actionable": True},
    )

    assert signal.signal_id == "test-123"
    assert signal.event_type == "CEO_EXIT"
    assert signal.category == "corporate_event"

    print(f"  PASSED: EventSignal created successfully")
    return True


def test_event_signal_to_dict():
    """Test EventSignal serialization."""
    print("\n[TEST] test_event_signal_to_dict")

    signal = EventSignal(
        signal_id="test-456",
        dedup_hash="def456",
        category="corporate_event",
        source_type="event_scanner",
        event_type="FDA_BREAKTHROUGH",
        asset_scope={"tickers": ["MRNA"], "sectors": ["BIOTECH"], "macro_regions": ["US"], "asset_classes": ["EQUITY"]},
        summary="MRNA: FDA Breakthrough Therapy designation",
        raw_value={"event_type": "FDA_BREAKTHROUGH"},
        evidence=[],
        confidence=0.85,
        confidence_factors={},
        directional_bias="bullish",
        time_horizon="days",
        novelty="new",
        staleness_policy={"max_age_seconds": 86400},
        uncertainties=[],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        event_metadata={"actionable": True},
    )

    result = signal.to_dict()

    assert isinstance(result, dict)
    assert result["signal_id"] == "test-456"
    assert result["event_type"] == "FDA_BREAKTHROUGH"
    assert result["directional_bias"] == "bullish"

    print(f"  PASSED: EventSignal serializes to dict")
    return True


def test_scanner_init_with_api_key():
    """Test scanner initializes with API key."""
    print("\n[TEST] test_scanner_init_with_api_key")

    scanner = EventScanner(api_key='test-key-123')

    assert scanner.is_configured is True
    assert scanner.api_key == 'test-key-123'

    print(f"  PASSED: Scanner configured with API key")
    return True


def test_scanner_init_without_api_key():
    """Test scanner handles missing API key gracefully."""
    print("\n[TEST] test_scanner_init_without_api_key")

    original = os.environ.pop('XAI_API_KEY', None)
    try:
        scanner = EventScanner(api_key=None)
        assert scanner.is_configured is False
    finally:
        if original:
            os.environ['XAI_API_KEY'] = original

    print(f"  PASSED: Scanner handles missing API key")
    return True


def test_build_market_wide_prompt():
    """Test market-wide event scan prompt includes key elements."""
    print("\n[TEST] test_build_market_wide_prompt")

    scanner = EventScanner(api_key='test-key')
    prompt = scanner._build_market_wide_prompt()

    assert "json" in prompt.lower(), "Prompt should request JSON"
    assert "ticker" in prompt.lower(), "Prompt should mention tickers"

    key_events = ["CEO", "FDA", "buyback", "activist", "S&P 500"]
    found_events = sum(1 for e in key_events if e.lower() in prompt.lower())
    assert found_events >= 3, f"Prompt should reference key event types (found {found_events})"

    print(f"  PASSED: Prompt built with key elements")
    return True


def test_parse_single_event():
    """Test parsing a single event from Grok response."""
    print("\n[TEST] test_parse_single_event")

    scanner = EventScanner(api_key='test-key')

    mock_response = {
        "events": [{
            "ticker": "NVDA",
            "event_type": "STOCK_BUYBACK",
            "headline": "NVIDIA announces $25B stock buyback program",
            "event_date": "2026-01-13",
            "source": "Company press release",
            "details": "Board authorized additional repurchase program"
        }]
    }

    signals = scanner._parse_events_response(mock_response)

    assert len(signals) == 1
    assert signals[0].event_type == "STOCK_BUYBACK"
    assert signals[0].asset_scope["tickers"] == ["NVDA"]
    assert signals[0].directional_bias == "bullish"

    print(f"  PASSED: Single event parsed correctly")
    return True


def test_parse_multiple_events():
    """Test parsing multiple events from Grok response."""
    print("\n[TEST] test_parse_multiple_events")

    scanner = EventScanner(api_key='test-key')

    mock_response = {
        "events": [
            {"ticker": "AAPL", "event_type": "CEO_EXIT", "headline": "Apple CEO retires", "event_date": "2026-01-13", "source": "SEC", "details": ""},
            {"ticker": "TSLA", "event_type": "ACTIVIST_INVESTOR", "headline": "Elliott stakes Tesla", "event_date": "2026-01-12", "source": "13D", "details": ""},
            {"ticker": "MRNA", "event_type": "FDA_BREAKTHROUGH", "headline": "Moderna BTD", "event_date": "2026-01-13", "source": "FDA", "details": ""},
        ]
    }

    signals = scanner._parse_events_response(mock_response)

    assert len(signals) == 3
    tickers = [s.asset_scope["tickers"][0] for s in signals]
    assert "AAPL" in tickers and "TSLA" in tickers and "MRNA" in tickers

    print(f"  PASSED: Multiple events parsed correctly")
    return True


def test_parse_empty_response():
    """Test handling of no events found."""
    print("\n[TEST] test_parse_empty_response")

    scanner = EventScanner(api_key='test-key')
    signals = scanner._parse_events_response({"events": []})

    assert len(signals) == 0

    print(f"  PASSED: Empty response handled")
    return True


def test_bullish_events():
    """Test events that should have bullish bias."""
    print("\n[TEST] test_bullish_events")

    bullish_events = ["STOCK_BUYBACK", "DIVIDEND_INCREASE", "FDA_BREAKTHROUGH", "FDA_APPROVAL",
                      "SP500_ADDITION", "GOVERNMENT_CONTRACT", "MAJOR_PARTNERSHIP", "INSIDER_BUYING"]

    for event_type in bullish_events:
        pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
        assert pattern.get("bias") == "bullish", f"{event_type} should be bullish"

    print(f"  PASSED: All bullish events verified")
    return True


def test_bearish_events():
    """Test events that should have bearish bias."""
    print("\n[TEST] test_bearish_events")

    bearish_events = ["FDA_REJECTION", "SP500_REMOVAL", "SHORT_SELLER_REPORT", "DIVIDEND_CUT",
                      "BANKRUPTCY_FILING", "CONTRACT_LOSS", "INSIDER_SELLING"]

    for event_type in bearish_events:
        pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
        assert pattern.get("bias") == "bearish", f"{event_type} should be bearish"

    print(f"  PASSED: All bearish events verified")
    return True


def test_mixed_events():
    """Test events that have mixed bias."""
    print("\n[TEST] test_mixed_events")

    mixed_events = ["CEO_EXIT", "CEO_APPOINTMENT", "MA_ANNOUNCEMENT", "SPINOFF", "DEBT_RESTRUCTURING"]

    for event_type in mixed_events:
        pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
        assert pattern.get("bias") == "mixed", f"{event_type} should be mixed"

    print(f"  PASSED: All mixed events verified")
    return True


def test_dedup_hash_generation():
    """Test deduplication hash is consistent."""
    print("\n[TEST] test_dedup_hash_generation")

    scanner = EventScanner(api_key='test-key')

    hash1 = scanner._generate_dedup_hash("STOCK_BUYBACK", "AAPL", "Apple announces buyback")
    hash2 = scanner._generate_dedup_hash("STOCK_BUYBACK", "AAPL", "Apple announces buyback")
    hash3 = scanner._generate_dedup_hash("STOCK_BUYBACK", "MSFT", "Apple announces buyback")

    assert hash1 == hash2, "Same inputs should produce same hash"
    assert hash1 != hash3, "Different ticker should produce different hash"

    print(f"  PASSED: Dedup hash generation works")
    return True


def test_duplicate_events_filtered():
    """Test that duplicate events are filtered."""
    print("\n[TEST] test_duplicate_events_filtered")

    scanner = EventScanner(api_key='test-key')

    mock_response = {
        "events": [
            {"ticker": "AAPL", "event_type": "STOCK_BUYBACK", "headline": "Apple $50B buyback", "event_date": "2026-01-13", "source": "PR", "details": ""},
            {"ticker": "AAPL", "event_type": "STOCK_BUYBACK", "headline": "Apple $50B buyback", "event_date": "2026-01-13", "source": "News", "details": ""},
        ]
    }

    signals = scanner._parse_events_response(mock_response)

    aapl_buybacks = [s for s in signals if s.asset_scope["tickers"][0] == "AAPL" and s.event_type == "STOCK_BUYBACK"]
    assert len(aapl_buybacks) == 1

    print(f"  PASSED: Duplicates filtered")
    return True


def test_high_confidence_recent_event():
    """Test that recent events from quality sources get high confidence."""
    print("\n[TEST] test_high_confidence_recent_event")

    scanner = EventScanner(api_key='test-key')

    confidence = scanner._calculate_confidence(
        event_type="FDA_APPROVAL",
        source="FDA.gov",
        hours_since_event=1,
        has_corroboration=True
    )

    assert confidence >= 0.80, f"Confidence should be >= 0.80, got {confidence}"

    print(f"  PASSED: High confidence for recent quality source ({confidence:.2f})")
    return True


def test_lower_confidence_older_event():
    """Test that older events get lower confidence."""
    print("\n[TEST] test_lower_confidence_older_event")

    scanner = EventScanner(api_key='test-key')

    confidence = scanner._calculate_confidence(
        event_type="FDA_APPROVAL",
        source="FDA.gov",
        hours_since_event=20,
        has_corroboration=True
    )

    assert confidence < 0.85, f"Confidence should be < 0.85 for older event, got {confidence}"

    print(f"  PASSED: Lower confidence for older event ({confidence:.2f})")
    return True


async def test_scan_market_wide_success():
    """Test successful market-wide scan."""
    print("\n[TEST] test_scan_market_wide_success")

    scanner = EventScanner(api_key='test-key')

    mock_response = {
        "events": [{
            "ticker": "NVDA",
            "event_type": "GOVERNMENT_CONTRACT",
            "headline": "NVIDIA wins $500M DoD AI contract",
            "event_date": "2026-01-13",
            "source": "Defense.gov",
            "details": "AI infrastructure modernization"
        }]
    }

    with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_response
        signals = await scanner.scan_market_wide()

        assert len(signals) >= 1
        assert any(s.asset_scope["tickers"][0] == "NVDA" for s in signals)

    print(f"  PASSED: Market-wide scan successful")
    return True


async def test_scan_market_wide_api_failure():
    """Test graceful handling of API failure."""
    print("\n[TEST] test_scan_market_wide_api_failure")

    scanner = EventScanner(api_key='test-key')

    with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = None
        signals = await scanner.scan_market_wide()

        assert signals == []

    print(f"  PASSED: API failure handled gracefully")
    return True


async def test_full_scan_cycle():
    """Test complete scan cycle produces valid signals."""
    print("\n[TEST] test_full_scan_cycle")

    scanner = EventScanner(api_key='test-key')

    mock_response = {
        "events": [
            {"ticker": "LMT", "event_type": "GOVERNMENT_CONTRACT", "headline": "Lockheed wins $2.3B contract",
             "event_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "source": "Defense.gov", "details": ""},
            {"ticker": "BIIB", "event_type": "FDA_APPROVAL", "headline": "Biogen FDA approval",
             "event_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "source": "FDA.gov", "details": ""},
        ]
    }

    with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_response
        signals = await scanner.scan_market_wide()

        for signal in signals:
            assert signal.signal_id is not None
            assert signal.dedup_hash is not None
            assert signal.category == "corporate_event"
            assert signal.source_type == "event_scanner"
            assert len(signal.asset_scope["tickers"]) > 0
            assert signal.confidence > 0
            assert signal.directional_bias in ["bullish", "bearish", "mixed"]

            d = signal.to_dict()
            assert isinstance(d, dict)

    print(f"  PASSED: Full scan cycle validated ({len(signals)} signals)")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("EVENT SCANNER TESTS")
    print("=" * 60)

    sync_tests = [
        test_all_27_event_types_defined,
        test_event_historical_patterns_complete,
        test_event_signal_creation,
        test_event_signal_to_dict,
        test_scanner_init_with_api_key,
        test_scanner_init_without_api_key,
        test_build_market_wide_prompt,
        test_parse_single_event,
        test_parse_multiple_events,
        test_parse_empty_response,
        test_bullish_events,
        test_bearish_events,
        test_mixed_events,
        test_dedup_hash_generation,
        test_duplicate_events_filtered,
        test_high_confidence_recent_event,
        test_lower_confidence_older_event,
    ]

    async_tests = [
        test_scan_market_wide_success,
        test_scan_market_wide_api_failure,
        test_full_scan_cycle,
    ]

    passed = 0
    failed = 0

    # Run sync tests
    for test in sync_tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[TEST] {test.__name__}")
            print(f"  FAILED: {e}")
            failed += 1

    # Run async tests
    for test in async_tests:
        try:
            if asyncio.get_event_loop().run_until_complete(test()):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[TEST] {test.__name__}")
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests passing!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
