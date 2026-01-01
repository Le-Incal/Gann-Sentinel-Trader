"""
Gann Sentinel Trader - Temporal Awareness Framework

This module provides system-wide temporal utilities ensuring all components
maintain a consistent forward-looking perspective.

CORE PRINCIPLE: We always look FORWARD from today. Never backward.
The system is date-agnostic - it doesn't matter what year we're in,
only what date TODAY is.

Version: 1.0.0
Last Updated: January 2026
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TIME HORIZON DEFINITIONS
# =============================================================================

class TimeHorizon(Enum):
    """
    Forward-looking time windows from current date.
    
    These are the canonical time horizons used across the entire system:
    - Grok uses these to frame sentiment queries
    - FRED uses these to contextualize macro implications
    - Polymarket uses these to filter market resolution dates
    - Claude uses these to assess trade time horizons
    """
    IMMEDIATE = 7        # Next 7 days - intraday/very short term
    SHORT_TERM = 30      # Next 30 days - days/weeks
    MEDIUM_TERM = 90     # Next 90 days - weeks/months
    LONG_TERM = 180      # Next 180 days - months
    EXTENDED = 365       # Next 365 days - quarters/year
    
    @property
    def label(self) -> str:
        """Human-readable label for this horizon."""
        labels = {
            TimeHorizon.IMMEDIATE: "immediate (1 week)",
            TimeHorizon.SHORT_TERM: "short-term (1 month)",
            TimeHorizon.MEDIUM_TERM: "medium-term (3 months)",
            TimeHorizon.LONG_TERM: "long-term (6 months)",
            TimeHorizon.EXTENDED: "extended (12 months)",
        }
        return labels.get(self, "unknown")
    
    @property
    def signal_horizon_label(self) -> str:
        """Label for signal schema time_horizon field."""
        labels = {
            TimeHorizon.IMMEDIATE: "intraday",
            TimeHorizon.SHORT_TERM: "days",
            TimeHorizon.MEDIUM_TERM: "weeks",
            TimeHorizon.LONG_TERM: "months",
            TimeHorizon.EXTENDED: "months",
        }
        return labels.get(self, "unknown")


class SignalRelevance(Enum):
    """How relevant a signal is based on its temporal characteristics."""
    HIGH = "high"        # Directly actionable within our horizons
    MEDIUM = "medium"    # Relevant context, may become actionable
    LOW = "low"          # Background information
    STALE = "stale"      # Past its relevance window


# =============================================================================
# TEMPORAL CONTEXT
# =============================================================================

@dataclass
class TemporalContext:
    """
    Encapsulates the current temporal context for the system.
    
    All scanners should use this to ensure consistent date handling.
    """
    # The reference point - always "now"
    now: datetime
    
    # Calculated windows
    windows: Dict[TimeHorizon, Tuple[datetime, datetime]]
    
    # Key dates for context
    today: datetime
    end_of_week: datetime
    end_of_month: datetime
    end_of_quarter: datetime
    end_of_year: datetime
    
    @classmethod
    def create(cls, reference_time: Optional[datetime] = None) -> "TemporalContext":
        """
        Create a new temporal context.
        
        Args:
            reference_time: Override for testing. If None, uses current UTC time.
            
        Returns:
            TemporalContext with all windows calculated
        """
        now = reference_time or datetime.now(timezone.utc)
        
        # Calculate forward-looking windows
        windows = {}
        for horizon in TimeHorizon:
            windows[horizon] = (now, now + timedelta(days=horizon.value))
        
        # Calculate key dates
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # End of week (Sunday)
        days_until_sunday = 6 - now.weekday()
        end_of_week = today + timedelta(days=days_until_sunday)
        
        # End of month
        if now.month == 12:
            end_of_month = now.replace(year=now.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_of_month = now.replace(month=now.month + 1, day=1) - timedelta(days=1)
        
        # End of quarter
        quarter_end_months = {1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6, 
                             7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12}
        quarter_end_month = quarter_end_months[now.month]
        if quarter_end_month == 12:
            end_of_quarter = now.replace(month=12, day=31)
        else:
            end_of_quarter = now.replace(month=quarter_end_month + 1, day=1) - timedelta(days=1)
        
        # End of year
        end_of_year = now.replace(month=12, day=31)
        
        return cls(
            now=now,
            windows=windows,
            today=today,
            end_of_week=end_of_week,
            end_of_month=end_of_month,
            end_of_quarter=end_of_quarter,
            end_of_year=end_of_year,
        )
    
    def get_window(self, horizon: TimeHorizon) -> Tuple[datetime, datetime]:
        """Get the date range for a specific horizon."""
        return self.windows[horizon]
    
    def format_date(self, dt: datetime, format: str = "%Y-%m-%d") -> str:
        """Format a datetime for display or API calls."""
        return dt.strftime(format)
    
    def days_until(self, target: datetime) -> int:
        """Calculate days from now until a target date."""
        return (target - self.now).days
    
    def classify_future_date(self, target: datetime) -> Optional[TimeHorizon]:
        """
        Classify a future date into the appropriate time horizon.
        
        Args:
            target: A future datetime to classify
            
        Returns:
            TimeHorizon or None if the date is in the past
        """
        if target < self.now:
            return None
        
        days = self.days_until(target)
        
        if days <= TimeHorizon.IMMEDIATE.value:
            return TimeHorizon.IMMEDIATE
        elif days <= TimeHorizon.SHORT_TERM.value:
            return TimeHorizon.SHORT_TERM
        elif days <= TimeHorizon.MEDIUM_TERM.value:
            return TimeHorizon.MEDIUM_TERM
        elif days <= TimeHorizon.LONG_TERM.value:
            return TimeHorizon.LONG_TERM
        else:
            return TimeHorizon.EXTENDED
    
    def is_within_horizon(self, target: datetime, horizon: TimeHorizon) -> bool:
        """Check if a target date falls within a specific horizon."""
        start, end = self.windows[horizon]
        return start <= target <= end
    
    def get_horizon_label(self, days_out: int) -> str:
        """Get the signal schema time_horizon label for a number of days."""
        if days_out <= 7:
            return "intraday"
        elif days_out <= 30:
            return "days"
        elif days_out <= 90:
            return "weeks"
        else:
            return "months"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "reference_time": self.now.isoformat(),
            "today": self.format_date(self.today),
            "windows": {
                horizon.name: {
                    "start": self.format_date(start),
                    "end": self.format_date(end),
                    "days": horizon.value,
                }
                for horizon, (start, end) in self.windows.items()
            },
            "key_dates": {
                "end_of_week": self.format_date(self.end_of_week),
                "end_of_month": self.format_date(self.end_of_month),
                "end_of_quarter": self.format_date(self.end_of_quarter),
                "end_of_year": self.format_date(self.end_of_year),
            }
        }
    
    def log_context(self) -> None:
        """Log the current temporal context."""
        logger.info(f"Temporal Context initialized: {self.format_date(self.now, '%Y-%m-%d %H:%M UTC')}")
        for horizon in TimeHorizon:
            start, end = self.windows[horizon]
            logger.debug(f"  {horizon.name}: {self.format_date(start)} → {self.format_date(end)}")


# =============================================================================
# QUERY BUILDERS
# =============================================================================

class TemporalQueryBuilder:
    """
    Builds forward-looking queries for various data sources.
    
    Ensures all queries across the system maintain temporal consistency.
    """
    
    def __init__(self, context: Optional[TemporalContext] = None):
        """
        Initialize with a temporal context.
        
        Args:
            context: If None, creates a new context using current time.
        """
        self.context = context or TemporalContext.create()
    
    # =========================================================================
    # GROK / SENTIMENT QUERIES
    # =========================================================================
    
    def build_sentiment_query(
        self, 
        topic: str, 
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM
    ) -> str:
        """
        Build a forward-looking sentiment query for Grok.
        
        Instead of: "NVDA stock news"
        Produces:   "NVDA outlook next 30 days forecast expectations"
        
        Args:
            topic: The base topic (ticker, sector, theme)
            horizon: Which time horizon to focus on
            
        Returns:
            Forward-looking query string
        """
        horizon_terms = {
            TimeHorizon.IMMEDIATE: ["this week", "next few days", "imminent"],
            TimeHorizon.SHORT_TERM: ["next month", "near-term", "upcoming", "outlook"],
            TimeHorizon.MEDIUM_TERM: ["Q1", "Q2", "next quarter", "3 month outlook"],
            TimeHorizon.LONG_TERM: ["H1", "H2", "first half", "second half", "6 month"],
            TimeHorizon.EXTENDED: ["2026", "next year", "annual outlook", "yearly forecast"],
        }
        
        terms = horizon_terms.get(horizon, ["outlook", "forecast"])
        
        # Build query with forward-looking terms
        query = f"{topic} {' OR '.join(terms)} forecast expectations"
        
        return query
    
    def build_catalyst_query(self, ticker: str) -> str:
        """
        Build a query focused on upcoming catalysts for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Query string for finding upcoming catalysts
        """
        # Get key upcoming dates
        eow = self.context.format_date(self.context.end_of_week)
        eom = self.context.format_date(self.context.end_of_month)
        
        return (
            f"{ticker} upcoming catalyst earnings release "
            f"investor day conference guidance FDA approval "
            f"product launch {self.context.now.year}"
        )
    
    def build_market_outlook_query(self) -> str:
        """Build a query for general market forward outlook."""
        year = self.context.now.year
        month = self.context.now.strftime("%B")
        
        return (
            f"stock market outlook {month} {year} "
            f"S&P 500 forecast expectations "
            f"equity market next quarter prediction"
        )
    
    # =========================================================================
    # FRED / MACRO QUERIES
    # =========================================================================
    
    def get_fred_context_prompt(self, series_name: str, current_value: float) -> str:
        """
        Generate forward-looking context for a FRED data point.
        
        Instead of just reporting "CPI is 3.2%", we add:
        "What does this imply for the next 3-6 months?"
        
        Args:
            series_name: Name of the FRED series
            current_value: Current value of the series
            
        Returns:
            Context prompt for forward-looking analysis
        """
        end_q = self.context.format_date(self.context.end_of_quarter)
        end_h = self.context.format_date(
            self.context.now + timedelta(days=180)
        )
        
        return (
            f"Current {series_name}: {current_value}. "
            f"Forward implications through {end_q} (end of quarter) "
            f"and {end_h} (6-month horizon). "
            f"What does this reading suggest for Fed policy and market direction?"
        )
    
    def get_upcoming_releases_window(self) -> Tuple[datetime, datetime]:
        """
        Get the window for upcoming economic data releases.
        
        Focuses on SHORT_TERM horizon for release schedules.
        
        Returns:
            (start, end) tuple for the release window
        """
        return self.context.get_window(TimeHorizon.SHORT_TERM)
    
    # =========================================================================
    # POLYMARKET QUERIES
    # =========================================================================
    
    def get_polymarket_date_filters(
        self, 
        horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    ) -> Dict[str, str]:
        """
        Get date filter parameters for Polymarket API.
        
        Args:
            horizon: Which time horizon to query
            
        Returns:
            Dict with end_date_min and end_date_max for API
        """
        start, end = self.context.get_window(horizon)
        
        return {
            "end_date_min": self.context.format_date(start),
            "end_date_max": self.context.format_date(end),
        }
    
    # =========================================================================
    # SIGNAL CLASSIFICATION
    # =========================================================================
    
    def classify_signal_relevance(
        self,
        signal_date: Optional[datetime],
        signal_type: str,
    ) -> Tuple[SignalRelevance, str]:
        """
        Classify how relevant a signal is based on its temporal characteristics.
        
        Args:
            signal_date: When the signal becomes relevant/resolves (if known)
            signal_type: Type of signal (sentiment, macro, prediction, etc.)
            
        Returns:
            Tuple of (SignalRelevance, reason_string)
        """
        # If no date, classify based on signal type freshness
        if signal_date is None:
            if signal_type in ["sentiment", "news"]:
                return SignalRelevance.HIGH, "Real-time signal, immediately relevant"
            else:
                return SignalRelevance.MEDIUM, "Undated signal, context-dependent relevance"
        
        # Check if date is in the past
        if signal_date < self.context.now:
            return SignalRelevance.STALE, f"Event occurred {self.context.days_until(signal_date)} days ago"
        
        # Classify by horizon
        horizon = self.context.classify_future_date(signal_date)
        
        if horizon == TimeHorizon.IMMEDIATE:
            return SignalRelevance.HIGH, "Resolves within 1 week - immediate relevance"
        elif horizon == TimeHorizon.SHORT_TERM:
            return SignalRelevance.HIGH, "Resolves within 1 month - high relevance"
        elif horizon == TimeHorizon.MEDIUM_TERM:
            return SignalRelevance.MEDIUM, "Resolves within 3 months - medium relevance"
        else:
            return SignalRelevance.LOW, "Resolves beyond 3 months - background context"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_temporal_context() -> TemporalContext:
    """Get a fresh temporal context for the current moment."""
    return TemporalContext.create()


def get_forward_window(horizon: TimeHorizon) -> Tuple[datetime, datetime]:
    """Quick access to a specific forward-looking window."""
    context = TemporalContext.create()
    return context.get_window(horizon)


def format_horizon_for_display(days: int) -> str:
    """Convert days to human-readable horizon string."""
    if days <= 7:
        return "this week"
    elif days <= 14:
        return "next two weeks"
    elif days <= 30:
        return "this month"
    elif days <= 60:
        return "next two months"
    elif days <= 90:
        return "this quarter"
    elif days <= 180:
        return "next six months"
    elif days <= 365:
        return "this year"
    else:
        return f"next {days // 365} year(s)"


# =============================================================================
# TESTING / VERIFICATION
# =============================================================================

def verify_temporal_logic() -> Dict[str, Any]:
    """
    Verify the temporal logic is working correctly.
    
    Returns:
        Dict with verification results
    """
    context = TemporalContext.create()
    
    results = {
        "current_time": context.now.isoformat(),
        "verification_passed": True,
        "checks": [],
    }
    
    # Check 1: All windows start from now
    for horizon in TimeHorizon:
        start, end = context.windows[horizon]
        check = {
            "name": f"{horizon.name} window",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "starts_from_now": start == context.now,
            "end_is_future": end > context.now,
            "duration_days": (end - start).days,
            "expected_days": horizon.value,
        }
        check["passed"] = (
            check["starts_from_now"] and 
            check["end_is_future"] and 
            check["duration_days"] == check["expected_days"]
        )
        
        if not check["passed"]:
            results["verification_passed"] = False
        
        results["checks"].append(check)
    
    # Check 2: Key dates are in the future
    key_date_checks = [
        ("end_of_week", context.end_of_week),
        ("end_of_month", context.end_of_month),
        ("end_of_quarter", context.end_of_quarter),
        ("end_of_year", context.end_of_year),
    ]
    
    for name, date in key_date_checks:
        check = {
            "name": f"key_date_{name}",
            "date": date.isoformat(),
            "is_future_or_today": date >= context.today,
            "passed": date >= context.today,
        }
        
        if not check["passed"]:
            results["verification_passed"] = False
        
        results["checks"].append(check)
    
    return results


if __name__ == "__main__":
    # Run verification
    import json
    
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("TEMPORAL AWARENESS FRAMEWORK VERIFICATION")
    print("=" * 60)
    
    results = verify_temporal_logic()
    
    print(f"\nCurrent Time: {results['current_time']}")
    print(f"Overall Status: {'✓ PASSED' if results['verification_passed'] else '✗ FAILED'}")
    
    print("\nWindow Checks:")
    for check in results["checks"]:
        if "duration_days" in check:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['name']}: {check['duration_days']} days")
        else:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['name']}: {check['date']}")
    
    print("\n" + "=" * 60)
    
    # Demo query builder
    print("\nQuery Builder Demo:")
    builder = TemporalQueryBuilder()
    
    print(f"\nSentiment Query (NVDA, SHORT_TERM):")
    print(f"  {builder.build_sentiment_query('NVDA', TimeHorizon.SHORT_TERM)}")
    
    print(f"\nCatalyst Query (AAPL):")
    print(f"  {builder.build_catalyst_query('AAPL')}")
    
    print(f"\nMarket Outlook Query:")
    print(f"  {builder.build_market_outlook_query()}")
    
    print(f"\nPolymarket Date Filters (MEDIUM_TERM):")
    print(f"  {builder.get_polymarket_date_filters(TimeHorizon.MEDIUM_TERM)}")
