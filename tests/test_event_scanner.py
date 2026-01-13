"""
Gann Sentinel Trader - Event Scanner Tests
TDD: Tests written BEFORE implementation.

Tests the LevelFields-style event-driven signal scanner.
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanners.event_scanner import (
    EventScanner,
    EventSignal,
    EVENT_TYPES,
    EVENT_HISTORICAL_PATTERNS,
)


class TestEventTypes:
    """Test that all 27 event types are properly defined."""

    def test_all_27_event_types_defined(self):
        """Verify all 27 event types exist in the scanner."""
        expected_events = [
            # Leadership (5)
            "CEO_EXIT",
            "CEO_APPOINTMENT",
            "EXECUTIVE_DEPARTURE",
            "INSIDER_BUYING",
            "INSIDER_SELLING",
            # Capital Allocation (4)
            "STOCK_BUYBACK",
            "DIVIDEND_INCREASE",
            "DIVIDEND_CUT",
            "RETURN_OF_CAPITAL",
            # Regulatory (5)
            "FDA_BREAKTHROUGH",
            "FDA_APPROVAL",
            "FDA_REJECTION",
            "DOJ_INVESTIGATION",
            "CLASS_ACTION_LAWSUIT",
            # Index Changes (3)
            "SP500_ADDITION",
            "SP500_REMOVAL",
            "INDEX_REBALANCING",
            # External Pressure (3)
            "ACTIVIST_INVESTOR",
            "SHORT_SELLER_REPORT",
            "PROXY_FIGHT",
            # Contracts (3)
            "GOVERNMENT_CONTRACT",
            "MAJOR_PARTNERSHIP",
            "CONTRACT_LOSS",
            # Corporate Actions (4)
            "MA_ANNOUNCEMENT",
            "SPINOFF",
            "BANKRUPTCY_FILING",
            "DEBT_RESTRUCTURING",
        ]

        assert len(EVENT_TYPES) == 27, f"Expected 27 event types, got {len(EVENT_TYPES)}"

        for event in expected_events:
            assert event in EVENT_TYPES, f"Missing event type: {event}"

    def test_event_historical_patterns_complete(self):
        """Verify each event type has historical pattern data."""
        for event_type in EVENT_TYPES:
            assert event_type in EVENT_HISTORICAL_PATTERNS, \
                f"Missing historical pattern for: {event_type}"

            pattern = EVENT_HISTORICAL_PATTERNS[event_type]
            assert "bias" in pattern, f"Missing bias for {event_type}"
            assert "avg_move_pct" in pattern, f"Missing avg_move_pct for {event_type}"
            assert "hold_days" in pattern, f"Missing hold_days for {event_type}"
            assert "win_rate" in pattern, f"Missing win_rate for {event_type}"

            # Validate ranges
            assert pattern["bias"] in ["bullish", "bearish", "mixed"], \
                f"Invalid bias for {event_type}: {pattern['bias']}"
            assert 0.0 <= pattern["win_rate"] <= 1.0, \
                f"Invalid win_rate for {event_type}: {pattern['win_rate']}"
            assert pattern["hold_days"] > 0, \
                f"Invalid hold_days for {event_type}: {pattern['hold_days']}"


class TestEventSignalSchema:
    """Test EventSignal dataclass structure."""

    def test_event_signal_creation(self):
        """Test creating an EventSignal with required fields."""
        signal = EventSignal(
            signal_id="test-123",
            dedup_hash="abc123",
            category="corporate_event",
            source_type="event_scanner",
            event_type="CEO_EXIT",
            asset_scope={
                "tickers": ["AAPL"],
                "sectors": ["TECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            summary="AAPL: CEO departure announced",
            raw_value={
                "event_type": "CEO_EXIT",
                "event_date": "2026-01-13",
                "historical_win_rate": 0.50,
                "historical_avg_move": 0.0,
                "time_since_event": "2 hours ago",
            },
            evidence=[{
                "source": "grok_event_search",
                "excerpt": "Apple CEO announces retirement",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }],
            confidence=0.75,
            confidence_factors={
                "source_quality": 0.8,
                "recency": 0.9,
                "corroboration": 0.6,
            },
            directional_bias="mixed",
            time_horizon="weeks",
            novelty="new",
            staleness_policy={
                "max_age_seconds": 86400,
                "stale_after_utc": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            },
            uncertainties=["CEO successor not yet named"],
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            event_metadata={
                "catalyst_window": "1-30 days typical impact",
                "actionable": True,
                "hours_since_event": 2,
            },
        )

        assert signal.signal_id == "test-123"
        assert signal.event_type == "CEO_EXIT"
        assert signal.category == "corporate_event"

    def test_event_signal_to_dict(self):
        """Test EventSignal serialization."""
        signal = EventSignal(
            signal_id="test-456",
            dedup_hash="def456",
            category="corporate_event",
            source_type="event_scanner",
            event_type="FDA_BREAKTHROUGH",
            asset_scope={
                "tickers": ["MRNA"],
                "sectors": ["BIOTECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
            summary="MRNA: FDA Breakthrough Therapy designation. Historically +12% avg move.",
            raw_value={
                "event_type": "FDA_BREAKTHROUGH",
                "event_date": "2026-01-13",
                "historical_win_rate": 0.75,
                "historical_avg_move": 0.12,
                "time_since_event": "4 hours ago",
            },
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
        assert "event_metadata" in result


class TestEventScannerInit:
    """Test EventScanner initialization."""

    def test_scanner_init_with_api_key(self):
        """Test scanner initializes with API key."""
        scanner = EventScanner(api_key='test-key-123')

        assert scanner.is_configured is True
        assert scanner.api_key == 'test-key-123'

    def test_scanner_init_without_api_key(self):
        """Test scanner handles missing API key gracefully."""
        # Temporarily remove XAI_API_KEY
        original = os.environ.pop('XAI_API_KEY', None)
        try:
            scanner = EventScanner(api_key=None)
            assert scanner.is_configured is False
        finally:
            if original:
                os.environ['XAI_API_KEY'] = original


class TestEventScannerPromptBuilding:
    """Test the prompt construction for Grok API."""

    def test_build_market_wide_prompt(self):
        """Test market-wide event scan prompt includes all event types."""
        scanner = EventScanner(api_key='test-key')
        prompt = scanner._build_market_wide_prompt()

        # Should mention scanning for corporate events
        assert "corporate event" in prompt.lower() or "market event" in prompt.lower() or "event" in prompt.lower()

        # Should mention 24-hour window
        assert "24" in prompt or "today" in prompt.lower() or "recent" in prompt.lower()

        # Should request JSON output
        assert "json" in prompt.lower()

        # Should include at least some key event types
        key_events = ["CEO", "FDA", "buyback", "activist", "S&P 500"]
        found_events = sum(1 for e in key_events if e.lower() in prompt.lower())
        assert found_events >= 3, "Prompt should reference key event types"

    def test_build_market_wide_prompt_structure(self):
        """Test prompt requests properly structured output."""
        scanner = EventScanner(api_key='test-key')
        prompt = scanner._build_market_wide_prompt()

        # Should ask for ticker identification
        assert "ticker" in prompt.lower()

        # Should ask for event type
        assert "event" in prompt.lower()


class TestEventScannerParsing:
    """Test parsing Grok responses into EventSignals."""

    def test_parse_single_event(self):
        """Test parsing a single event from Grok response."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {
            "events": [
                {
                    "ticker": "NVDA",
                    "event_type": "STOCK_BUYBACK",
                    "headline": "NVIDIA announces $25B stock buyback program",
                    "event_date": "2026-01-13",
                    "source": "Company press release",
                    "details": "Board authorized additional repurchase program"
                }
            ]
        }

        signals = scanner._parse_events_response(mock_response)

        assert len(signals) == 1
        assert signals[0].event_type == "STOCK_BUYBACK"
        assert signals[0].asset_scope["tickers"] == ["NVDA"]
        assert signals[0].directional_bias == "bullish"  # Buybacks are bullish

    def test_parse_multiple_events(self):
        """Test parsing multiple events from Grok response."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {
            "events": [
                {
                    "ticker": "AAPL",
                    "event_type": "CEO_EXIT",
                    "headline": "Apple CEO Tim Cook announces retirement",
                    "event_date": "2026-01-13",
                    "source": "SEC Filing",
                    "details": "Effective Q2 2026"
                },
                {
                    "ticker": "TSLA",
                    "event_type": "ACTIVIST_INVESTOR",
                    "headline": "Elliott Management takes stake in Tesla",
                    "event_date": "2026-01-12",
                    "source": "13D Filing",
                    "details": "5.2% stake acquired"
                },
                {
                    "ticker": "MRNA",
                    "event_type": "FDA_BREAKTHROUGH",
                    "headline": "Moderna receives BTD for cancer vaccine",
                    "event_date": "2026-01-13",
                    "source": "FDA announcement",
                    "details": "mRNA-4157 combination therapy"
                }
            ]
        }

        signals = scanner._parse_events_response(mock_response)

        assert len(signals) == 3

        tickers = [s.asset_scope["tickers"][0] for s in signals]
        assert "AAPL" in tickers
        assert "TSLA" in tickers
        assert "MRNA" in tickers

    def test_parse_empty_response(self):
        """Test handling of no events found."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {"events": []}

        signals = scanner._parse_events_response(mock_response)

        assert len(signals) == 0

    def test_parse_filters_old_events(self):
        """Test that events older than 24 hours are marked not actionable."""
        scanner = EventScanner(api_key='test-key')

        old_date = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%d")

        mock_response = {
            "events": [
                {
                    "ticker": "OLD",
                    "event_type": "STOCK_BUYBACK",
                    "headline": "Old buyback news",
                    "event_date": old_date,
                    "source": "News",
                    "details": "Should be marked not actionable"
                }
            ]
        }

        signals = scanner._parse_events_response(mock_response)

        # Old events should be marked as not actionable
        if len(signals) > 0:
            assert signals[0].event_metadata.get("actionable") is False

    def test_parse_unknown_event_type(self):
        """Test handling of unrecognized event types."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {
            "events": [
                {
                    "ticker": "XYZ",
                    "event_type": "UNKNOWN_EVENT_TYPE",
                    "headline": "Something happened",
                    "event_date": "2026-01-13",
                    "source": "News",
                    "details": "Details"
                }
            ]
        }

        signals = scanner._parse_events_response(mock_response)

        # Should skip unknown event types
        assert len(signals) == 0


class TestEventScannerDirectionalBias:
    """Test directional bias assignment based on event type."""

    def test_bullish_events(self):
        """Test events that should have bullish bias."""
        bullish_events = [
            "STOCK_BUYBACK",
            "DIVIDEND_INCREASE",
            "FDA_BREAKTHROUGH",
            "FDA_APPROVAL",
            "SP500_ADDITION",
            "GOVERNMENT_CONTRACT",
            "MAJOR_PARTNERSHIP",
            "INSIDER_BUYING",
        ]

        for event_type in bullish_events:
            pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
            assert pattern.get("bias") == "bullish", \
                f"{event_type} should be bullish, got {pattern.get('bias')}"

    def test_bearish_events(self):
        """Test events that should have bearish bias."""
        bearish_events = [
            "FDA_REJECTION",
            "SP500_REMOVAL",
            "SHORT_SELLER_REPORT",
            "DIVIDEND_CUT",
            "BANKRUPTCY_FILING",
            "CONTRACT_LOSS",
            "INSIDER_SELLING",
        ]

        for event_type in bearish_events:
            pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
            assert pattern.get("bias") == "bearish", \
                f"{event_type} should be bearish, got {pattern.get('bias')}"

    def test_mixed_events(self):
        """Test events that have mixed/context-dependent bias."""
        mixed_events = [
            "CEO_EXIT",
            "CEO_APPOINTMENT",
            "MA_ANNOUNCEMENT",
            "SPINOFF",
            "DEBT_RESTRUCTURING",
        ]

        for event_type in mixed_events:
            pattern = EVENT_HISTORICAL_PATTERNS.get(event_type, {})
            assert pattern.get("bias") == "mixed", \
                f"{event_type} should be mixed, got {pattern.get('bias')}"


class TestEventScannerScan:
    """Test the main scan functionality."""

    @pytest.mark.asyncio
    async def test_scan_market_wide_success(self):
        """Test successful market-wide scan."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {
            "events": [
                {
                    "ticker": "NVDA",
                    "event_type": "GOVERNMENT_CONTRACT",
                    "headline": "NVIDIA wins $500M DoD AI contract",
                    "event_date": "2026-01-13",
                    "source": "Defense.gov",
                    "details": "AI infrastructure modernization"
                }
            ]
        }

        with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            signals = await scanner.scan_market_wide()

            assert len(signals) >= 1
            assert any(s.asset_scope["tickers"][0] == "NVDA" for s in signals)

    @pytest.mark.asyncio
    async def test_scan_market_wide_api_failure(self):
        """Test graceful handling of API failure."""
        scanner = EventScanner(api_key='test-key')

        with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None  # API failure

            signals = await scanner.scan_market_wide()

            assert signals == []  # Should return empty list, not crash

    @pytest.mark.asyncio
    async def test_scan_not_configured(self):
        """Test scan returns empty when not configured."""
        # Temporarily remove XAI_API_KEY
        original = os.environ.pop('XAI_API_KEY', None)
        try:
            scanner = EventScanner(api_key=None)

            signals = await scanner.scan_market_wide()

            assert signals == []
        finally:
            if original:
                os.environ['XAI_API_KEY'] = original


class TestEventScannerDeduplication:
    """Test signal deduplication."""

    def test_dedup_hash_generation(self):
        """Test deduplication hash is consistent."""
        scanner = EventScanner(api_key='test-key')

        hash1 = scanner._generate_dedup_hash("STOCK_BUYBACK", "AAPL", "Apple announces buyback")
        hash2 = scanner._generate_dedup_hash("STOCK_BUYBACK", "AAPL", "Apple announces buyback")
        hash3 = scanner._generate_dedup_hash("STOCK_BUYBACK", "MSFT", "Apple announces buyback")

        assert hash1 == hash2  # Same inputs = same hash
        assert hash1 != hash3  # Different ticker = different hash

    def test_duplicate_events_filtered(self):
        """Test that duplicate events are filtered."""
        scanner = EventScanner(api_key='test-key')

        mock_response = {
            "events": [
                {
                    "ticker": "AAPL",
                    "event_type": "STOCK_BUYBACK",
                    "headline": "Apple $50B buyback",
                    "event_date": "2026-01-13",
                    "source": "PR",
                    "details": "Details"
                },
                {
                    "ticker": "AAPL",
                    "event_type": "STOCK_BUYBACK",
                    "headline": "Apple $50B buyback",  # Duplicate
                    "event_date": "2026-01-13",
                    "source": "News",
                    "details": "Same event"
                }
            ]
        }

        signals = scanner._parse_events_response(mock_response)

        # Should deduplicate
        aapl_buybacks = [s for s in signals
                       if s.asset_scope["tickers"][0] == "AAPL"
                       and s.event_type == "STOCK_BUYBACK"]
        assert len(aapl_buybacks) == 1


class TestEventScannerConfidenceScoring:
    """Test confidence score calculation."""

    def test_high_confidence_recent_event(self):
        """Test that recent events from quality sources get high confidence."""
        scanner = EventScanner(api_key='test-key')

        # FDA approval from official source, very recent
        confidence = scanner._calculate_confidence(
            event_type="FDA_APPROVAL",
            source="FDA.gov",
            hours_since_event=1,
            has_corroboration=True
        )

        assert confidence >= 0.80

    def test_lower_confidence_older_event(self):
        """Test that older events get lower confidence."""
        scanner = EventScanner(api_key='test-key')

        # Same event but 20 hours old
        confidence = scanner._calculate_confidence(
            event_type="FDA_APPROVAL",
            source="FDA.gov",
            hours_since_event=20,
            has_corroboration=True
        )

        # Should be lower than very recent
        assert confidence < 0.85


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestEventScannerIntegration:
    """Integration tests for full scan cycle."""

    @pytest.mark.asyncio
    async def test_full_scan_cycle(self):
        """Test complete scan cycle produces valid signals."""
        scanner = EventScanner(api_key='test-key')

        # Mock a realistic Grok response
        mock_response = {
            "events": [
                {
                    "ticker": "LMT",
                    "event_type": "GOVERNMENT_CONTRACT",
                    "headline": "Lockheed wins $2.3B F-35 sustainment contract",
                    "event_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "source": "Defense.gov press release",
                    "details": "Multi-year sustainment and logistics support"
                },
                {
                    "ticker": "BIIB",
                    "event_type": "FDA_APPROVAL",
                    "headline": "Biogen receives FDA approval for Alzheimer's drug",
                    "event_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "source": "FDA.gov",
                    "details": "Full approval granted after accelerated pathway"
                }
            ]
        }

        with patch.object(scanner, '_call_grok_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            signals = await scanner.scan_market_wide()

            # Validate signal structure
            for signal in signals:
                assert signal.signal_id is not None
                assert signal.dedup_hash is not None
                assert signal.category == "corporate_event"
                assert signal.source_type == "event_scanner"
                assert signal.event_type in [
                    "GOVERNMENT_CONTRACT", "FDA_APPROVAL"
                ]
                assert len(signal.asset_scope["tickers"]) > 0
                assert signal.confidence > 0
                assert signal.directional_bias in ["bullish", "bearish", "mixed"]

                # Verify to_dict works
                d = signal.to_dict()
                assert isinstance(d, dict)
                assert d["event_type"] == signal.event_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
