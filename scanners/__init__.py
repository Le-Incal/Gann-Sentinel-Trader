"""
Gann Sentinel Trader - Scanners Module

This module contains all signal scanners with system-wide temporal awareness.

Core Principle: All scanners look FORWARD from today, never backward.

Scanners:
- GrokScanner: Sentiment and news from X/Twitter and web
- FREDScanner: Macroeconomic data with forward implications
- PolymarketScanner: Prediction market probabilities

All scanners use the shared TemporalContext from temporal.py to ensure
consistent date handling across the entire system.

Version: 2.0.0
"""

from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    SignalRelevance,
    get_temporal_context,
    get_forward_window,
    format_horizon_for_display,
    verify_temporal_logic,
)

from scanners.grok_scanner import GrokScanner
from scanners.fred_scanner import FREDScanner
from scanners.polymarket_scanner import PolymarketScanner

__all__ = [
    # Temporal framework
    "TemporalContext",
    "TemporalQueryBuilder", 
    "TimeHorizon",
    "SignalRelevance",
    "get_temporal_context",
    "get_forward_window",
    "format_horizon_for_display",
    "verify_temporal_logic",
    # Scanners
    "GrokScanner",
    "FREDScanner",
    "PolymarketScanner",
]
