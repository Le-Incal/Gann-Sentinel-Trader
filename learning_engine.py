"""
Learning Engine for Gann Sentinel Trader
Tracks performance, learns from outcomes, and provides context for Claude's decisions.

Version: 1.0.0
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class VIXRegime(Enum):
    LOW = "low"              # VIX < 15
    NORMAL = "normal"        # 15 <= VIX < 25
    ELEVATED = "elevated"    # 25 <= VIX < 35
    EXTREME = "extreme"      # VIX >= 35


class SPYTrend(Enum):
    STRONG_UP = "strong_up"    # > 5% in 20d
    UP = "up"                  # 1-5%
    SIDEWAYS = "sideways"      # -1% to 1%
    DOWN = "down"              # -5% to -1%
    STRONG_DOWN = "strong_down"  # < -5%


# Scan times in UTC (ET + 5 hours standard, +4 daylight)
MORNING_SCAN_HOUR_UTC = 14   # 9:30 AM ET = 14:30 UTC (winter)
MORNING_SCAN_MINUTE = 35     # 5 min after market open
MIDDAY_SCAN_HOUR_UTC = 17    # 12:30 PM ET = 17:30 UTC
MIDDAY_SCAN_MINUTE = 30


# =============================================================================
# SMART SCHEDULER
# =============================================================================

class SmartScheduler:
    """
    Controls when automatic scans run.

    Rules:
    - Morning scan at 9:35 AM ET (14:35 UTC)
    - Midday scan at 12:30 PM ET (17:30 UTC)
    - No weekends
    - No after hours
    - Manual /check and /scan always work
    """

    def __init__(self):
        self.last_morning_scan: Optional[datetime] = None
        self.last_midday_scan: Optional[datetime] = None

    def should_scan(self, now: datetime) -> bool:
        """
        Check if an automatic scan should run.

        Returns True only at designated scan times on weekdays.
        """
        # Check weekend (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return False

        hour = now.hour
        minute = now.minute

        # Morning scan window: 14:35-14:45 UTC
        if hour == MORNING_SCAN_HOUR_UTC and MORNING_SCAN_MINUTE <= minute < MORNING_SCAN_MINUTE + 10:
            if self._can_run_morning_scan(now):
                return True

        # Midday scan window: 17:30-17:40 UTC
        if hour == MIDDAY_SCAN_HOUR_UTC and MIDDAY_SCAN_MINUTE <= minute < MIDDAY_SCAN_MINUTE + 10:
            if self._can_run_midday_scan(now):
                return True

        return False

    def _can_run_morning_scan(self, now: datetime) -> bool:
        """Check if morning scan hasn't run today."""
        if self.last_morning_scan is None:
            return True
        return self.last_morning_scan.date() < now.date()

    def _can_run_midday_scan(self, now: datetime) -> bool:
        """Check if midday scan hasn't run today."""
        if self.last_midday_scan is None:
            return True
        return self.last_midday_scan.date() < now.date()

    def record_scan(self, now: datetime, scan_type: str) -> None:
        """Record that a scan was run."""
        if scan_type == "morning":
            self.last_morning_scan = now
        elif scan_type == "midday":
            self.last_midday_scan = now

    def get_scan_type(self, now: datetime) -> str:
        """Determine which scan type based on time."""
        if now.hour == MORNING_SCAN_HOUR_UTC:
            return "morning"
        elif now.hour == MIDDAY_SCAN_HOUR_UTC:
            return "midday"
        return "manual"

    def get_next_scan_time(self, now: datetime) -> Optional[datetime]:
        """Get the next scheduled scan time."""
        # If before morning scan today
        if now.hour < MORNING_SCAN_HOUR_UTC:
            return now.replace(hour=MORNING_SCAN_HOUR_UTC, minute=MORNING_SCAN_MINUTE, second=0, microsecond=0)

        # If before midday scan today
        if now.hour < MIDDAY_SCAN_HOUR_UTC:
            return now.replace(hour=MIDDAY_SCAN_HOUR_UTC, minute=MIDDAY_SCAN_MINUTE, second=0, microsecond=0)

        # Next day morning scan
        tomorrow = now + timedelta(days=1)
        # Skip to Monday if weekend
        while tomorrow.weekday() >= 5:
            tomorrow += timedelta(days=1)

        return tomorrow.replace(hour=MORNING_SCAN_HOUR_UTC, minute=MORNING_SCAN_MINUTE, second=0, microsecond=0)


# =============================================================================
# LEARNING ENGINE
# =============================================================================

class LearningEngine:
    """
    Analyzes historical performance and generates insights for Claude.

    Core capabilities:
    1. Trade outcome tracking with P&L and alpha calculation
    2. Signal accuracy scoring by source
    3. Conviction calibration (actual vs stated win rates)
    4. Market regime detection
    5. Context generation for Claude's decision-making
    6. SPY benchmark comparison
    """

    def __init__(self, db, executor=None):
        """
        Initialize the Learning Engine.

        Args:
            db: Database instance for persistence
            executor: AlpacaExecutor for fetching SPY data (optional)
        """
        self.db = db
        self.executor = executor
        self.scheduler = SmartScheduler()

        logger.info("Learning Engine initialized")

    # =========================================================================
    # TRADE METRICS
    # =========================================================================

    def calculate_trade_metrics(
        self,
        entry: Dict[str, Any],
        exit: Dict[str, Any],
        spy_return: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a completed trade.

        Args:
            entry: Trade entry data (price, date, conviction, etc.)
            exit: Trade exit data (price, date, reason)
            spy_return: SPY return over the same period

        Returns:
            Dict with realized P&L, hold time, alpha, etc.
        """
        entry_price = entry.get("entry_price", 0)
        exit_price = exit.get("exit_price", 0)
        quantity = entry.get("quantity", 1)

        # Calculate P&L
        if entry.get("side", "BUY").upper() == "BUY":
            realized_pnl = (exit_price - entry_price) * quantity
            realized_pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        else:  # SHORT
            realized_pnl = (entry_price - exit_price) * quantity
            realized_pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0

        # Calculate hold time
        entry_date = self._parse_datetime(entry.get("entry_date"))
        exit_date = self._parse_datetime(exit.get("exit_date"))

        if entry_date and exit_date:
            hold_time = exit_date - entry_date
            hold_time_hours = hold_time.total_seconds() / 3600
        else:
            hold_time_hours = 0

        # Calculate alpha (our return - SPY return)
        alpha = realized_pnl_pct - spy_return

        return {
            "trade_id": entry.get("trade_id"),
            "ticker": entry.get("ticker"),
            "side": entry.get("side"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(realized_pnl_pct, 4),
            "hold_time_hours": round(hold_time_hours, 1),
            "spy_return_same_period": spy_return,
            "alpha": round(alpha, 4),
            "entry_conviction": entry.get("entry_conviction"),
            "exit_reason": exit.get("exit_reason"),
            "primary_signal_source": entry.get("primary_signal_source"),
            "signal_sources_agreed": entry.get("signal_sources_agreed", 1)
        }

    def calculate_performance_stats(self, outcomes: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate performance statistics.

        Args:
            outcomes: List of trade outcome dicts

        Returns:
            Dict with win rate, avg return, Sharpe, etc.
        """
        if not outcomes:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "total_return": 0.0,
                "avg_alpha": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_hold_time_hours": 0.0
            }

        total_trades = len(outcomes)
        winning_trades = sum(1 for t in outcomes if t.get("realized_pnl", 0) > 0)
        losing_trades = total_trades - winning_trades

        returns = [t.get("realized_pnl_pct", 0) for t in outcomes]
        alphas = [t.get("alpha", 0) for t in outcomes if t.get("alpha") is not None]
        hold_times = [t.get("hold_time_hours", 0) for t in outcomes]

        avg_return = sum(returns) / len(returns) if returns else 0
        total_return = sum(returns)
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        # Simple Sharpe approximation (assuming risk-free = 0)
        if len(returns) > 1:
            import statistics
            try:
                std_dev = statistics.stdev(returns)
                sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
            except:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown from cumulative returns
        max_drawdown = self._calculate_max_drawdown(returns)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "avg_return": avg_return,
            "total_return": total_return,
            "avg_alpha": avg_alpha,
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": max_drawdown,
            "avg_hold_time_hours": round(avg_hold_time, 1)
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from a series of returns."""
        if not returns:
            return 0.0

        cumulative = []
        total = 1.0
        for r in returns:
            total *= (1 + r)
            cumulative.append(total)

        peak = cumulative[0]
        max_dd = 0.0

        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return round(max_dd, 4)

    # =========================================================================
    # SOURCE RELIABILITY
    # =========================================================================

    def calculate_source_reliability(self, signal_results: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate accuracy metrics by signal source.

        Args:
            signal_results: List of signal performance records

        Returns:
            Dict mapping source -> accuracy metrics
        """
        by_source = {}

        for result in signal_results:
            source = result.get("source", "unknown")
            if source not in by_source:
                by_source[source] = {"total": 0, "accurate": 0}

            by_source[source]["total"] += 1
            if result.get("accurate_5d"):
                by_source[source]["accurate"] += 1

        reliability = {}
        for source, counts in by_source.items():
            total = counts["total"]
            accurate = counts["accurate"]

            reliability[source] = {
                "total_signals": total,
                "accurate_signals": accurate,
                "accuracy_rate": accurate / total if total > 0 else 0
            }

        return reliability

    # =========================================================================
    # CONVICTION CALIBRATION
    # =========================================================================

    def calculate_conviction_calibration(self, trades: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate actual win rates by conviction bucket.

        Helps identify if we're overconfident or underconfident.

        Args:
            trades: List of trade outcomes with conviction scores

        Returns:
            Dict mapping conviction bucket -> actual performance
        """
        buckets = {
            "85-100": {"trades": [], "label": "High Conviction"},
            "80-84": {"trades": [], "label": "Medium Conviction"},
            "below_80": {"trades": [], "label": "Low Conviction (rejected)"}
        }

        for trade in trades:
            conviction = trade.get("entry_conviction", 0)
            pnl_pct = trade.get("realized_pnl_pct", 0)

            if conviction >= 85:
                buckets["85-100"]["trades"].append(pnl_pct)
            elif conviction >= 80:
                buckets["80-84"]["trades"].append(pnl_pct)
            else:
                buckets["below_80"]["trades"].append(pnl_pct)

        calibration = {}
        for bucket, data in buckets.items():
            trades_list = data["trades"]
            if trades_list:
                wins = sum(1 for t in trades_list if t > 0)
                actual_wr = wins / len(trades_list)
                avg_return = sum(trades_list) / len(trades_list)
            else:
                actual_wr = 0
                avg_return = 0

            calibration[bucket] = {
                "total_trades": len(trades_list),
                "actual_win_rate": round(actual_wr, 3),
                "avg_return": round(avg_return, 4),
                "label": data["label"]
            }

        return calibration

    # =========================================================================
    # MARKET REGIME
    # =========================================================================

    def classify_vix_regime(self, vix: float) -> str:
        """Classify VIX level into regime."""
        if vix < 15:
            return "low"
        elif vix < 25:
            return "normal"
        elif vix < 35:
            return "elevated"
        else:
            return "extreme"

    def classify_spy_trend(self, return_20d: float) -> str:
        """Classify SPY 20-day return into trend."""
        if return_20d > 0.05:
            return "strong_up"
        elif return_20d > 0.01:
            return "up"
        elif return_20d > -0.01:
            return "sideways"
        elif return_20d > -0.05:
            return "down"
        else:
            return "strong_down"

    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get current market regime.

        In production, this would fetch live VIX and SPY data.
        """
        # Placeholder - in production, fetch from executor
        return {
            "vix_level": 18.0,
            "vix_regime": "normal",
            "spy_trend": "up",
            "spy_20d_return": 0.02,
            "detected_at": datetime.now(timezone.utc).isoformat()
        }

    # =========================================================================
    # ALPHA CALCULATION
    # =========================================================================

    def calculate_alpha_simple(self, our_return: float, spy_return: float) -> float:
        """
        Calculate simple alpha (our return - benchmark return).

        Args:
            our_return: Our portfolio return as decimal (e.g., 0.12 for 12%)
            spy_return: SPY return over same period

        Returns:
            Alpha as decimal
        """
        return our_return - spy_return

    # =========================================================================
    # CONTEXT GENERATION
    # =========================================================================

    def generate_claude_context(self) -> Dict[str, Any]:
        """
        Generate the learning context to inject into Claude's prompts.

        This is the core output of the Learning Engine.
        """
        context = {}

        # Performance summary
        overall_stats = None
        if hasattr(self.db, 'get_learning_stats'):
            overall_stats = self.db.get_learning_stats("overall", None, "30d")

        if overall_stats:
            context["performance_summary"] = {
                "total_trades": overall_stats.get("total_trades", 0),
                "win_rate": overall_stats.get("win_rate", 0),
                "avg_return": overall_stats.get("avg_return", 0),
                "alpha": overall_stats.get("alpha", 0),
                "sharpe_ratio": overall_stats.get("sharpe_ratio", 0),
                "period": "30 days"
            }
        else:
            context["performance_summary"] = {
                "total_trades": 0,
                "win_rate": 0,
                "note": "No historical data yet"
            }

        # Source reliability
        source_reliability = {}
        for source in ["grok", "perplexity", "chatgpt", "technical"]:
            if hasattr(self.db, 'get_source_reliability'):
                rel = self.db.get_source_reliability(source, "30d")
                if rel:
                    source_reliability[source] = {
                        "accuracy": rel.get("accuracy_rate", 0),
                        "total_signals": rel.get("total_signals", 0)
                    }

        context["source_reliability"] = source_reliability if source_reliability else {
            "note": "No signal accuracy data yet"
        }

        # Current regime
        context["current_regime"] = self.get_current_regime()

        # Recent lessons (placeholder - in production, generate from data)
        context["recent_lessons"] = [
            "System is learning - more data needed for insights"
        ]

        return context

    def format_performance_summary(self, stats: Dict[str, Any]) -> str:
        """
        Format performance stats into a human-readable summary.

        Used for Telegram digests and logs.
        """
        total = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0) * 100
        avg_return = stats.get("avg_return", 0) * 100
        alpha = stats.get("alpha", 0) * 100
        sharpe = stats.get("sharpe_ratio", 0)

        lines = [
            "PERFORMANCE SUMMARY",
            "=" * 30,
            f"Total Trades: {total}",
            f"Win Rate: {win_rate:.1f}%",
            f"Avg Return: {avg_return:.2f}%",
            f"Alpha vs SPY: {alpha:+.2f}%",
            f"Sharpe Ratio: {sharpe:.2f}",
        ]

        # Add color commentary
        if alpha > 0:
            lines.append(f"\n✓ Outperforming SPY by {alpha:.1f}%")
        elif alpha < 0:
            lines.append(f"\n✗ Underperforming SPY by {abs(alpha):.1f}%")
        else:
            lines.append("\n→ Matching SPY performance")

        return "\n".join(lines)

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format learning context as text for Claude's system prompt.
        """
        perf = context.get("performance_summary", {})
        sources = context.get("source_reliability", {})
        regime = context.get("current_regime", {})
        lessons = context.get("recent_lessons", [])

        lines = [
            "=== LEARNING ENGINE CONTEXT ===",
            "",
            "HISTORICAL PERFORMANCE (30d):",
            f"  Trades: {perf.get('total_trades', 0)}",
            f"  Win Rate: {perf.get('win_rate', 0):.0%}",
            f"  Alpha vs SPY: {perf.get('alpha', 0):+.1%}",
            "",
            "SOURCE RELIABILITY:",
        ]

        for source, data in sources.items():
            if isinstance(data, dict) and "accuracy" in data:
                lines.append(f"  {source}: {data['accuracy']:.0%} accurate")

        lines.extend([
            "",
            "CURRENT MARKET REGIME:",
            f"  VIX: {regime.get('vix_level', 'N/A')} ({regime.get('vix_regime', 'unknown')})",
            f"  SPY Trend: {regime.get('spy_trend', 'unknown')}",
            "",
            "LESSONS LEARNED:",
        ])

        for lesson in lessons[:3]:
            lines.append(f"  - {lesson}")

        lines.append("=" * 35)

        return "\n".join(lines)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            # Handle various ISO formats
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except:
            return None


# =============================================================================
# DATABASE EXTENSION METHODS
# =============================================================================

def add_learning_tables(db):
    """
    Add learning engine tables to existing database.

    Call this during database initialization.
    """
    cursor = db.conn.cursor()

    # Trade outcomes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id TEXT PRIMARY KEY,
            trade_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            entry_conviction INTEGER,
            entry_thesis TEXT,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            hold_time_hours REAL,
            max_drawdown_pct REAL,
            max_gain_pct REAL,
            spy_return_same_period REAL,
            alpha REAL,
            primary_signal_source TEXT,
            signal_sources_agreed INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        )
    """)

    # Signal performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_performance (
            id TEXT PRIMARY KEY,
            signal_id TEXT,
            source TEXT NOT NULL,
            signal_type TEXT,
            ticker TEXT,
            signal_date TEXT NOT NULL,
            predicted_direction TEXT,
            conviction_at_signal INTEGER,
            actual_direction TEXT,
            price_at_signal REAL,
            price_after_1d REAL,
            price_after_5d REAL,
            price_after_20d REAL,
            accurate_1d BOOLEAN,
            accurate_5d BOOLEAN,
            accurate_20d BOOLEAN,
            evaluated_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Source reliability table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_reliability (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            period TEXT NOT NULL,
            total_signals INTEGER,
            accurate_signals INTEGER,
            accuracy_rate REAL,
            avg_conviction_when_right REAL,
            avg_conviction_when_wrong REAL,
            accuracy_by_sector TEXT,
            accuracy_trend TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source, period)
        )
    """)

    # Market regimes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL UNIQUE,
            vix_level REAL,
            vix_regime TEXT,
            spy_20d_return REAL,
            spy_trend TEXT,
            advance_decline_ratio REAL,
            pct_above_50dma REAL,
            leading_sectors TEXT,
            lagging_sectors TEXT,
            our_win_rate_this_regime REAL,
            trades_in_regime INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Learning stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_stats (
            id TEXT PRIMARY KEY,
            stat_type TEXT NOT NULL,
            stat_key TEXT,
            period TEXT NOT NULL,
            total_trades INTEGER,
            winning_trades INTEGER,
            win_rate REAL,
            avg_return REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            avg_hold_time_hours REAL,
            spy_return_same_period REAL,
            alpha REAL,
            beta REAL,
            insights TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stat_type, stat_key, period)
        )
    """)

    # SPY benchmarks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spy_benchmarks (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL UNIQUE,
            open_price REAL,
            close_price REAL,
            daily_return REAL,
            cumulative_return_30d REAL,
            cumulative_return_90d REAL,
            cumulative_return_ytd REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.conn.commit()
    logger.info("Learning Engine tables created")
