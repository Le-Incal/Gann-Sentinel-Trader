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
            lines.append(f"\n\u2713 Outperforming SPY by {alpha:.1f}%")
        elif alpha < 0:
            lines.append(f"\n\u2717 Underperforming SPY by {abs(alpha):.1f}%")
        else:
            lines.append("\n\u2192 Matching SPY performance")

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
    # TRADE OUTCOME RECORDING
    # =========================================================================

    def record_trade_outcome(
        self,
        trade_id: str,
        ticker: str,
        side: str,
        entry_price: float,
        entry_date: str,
        exit_price: float,
        exit_date: str,
        quantity: int,
        exit_reason: str = None,
        signal_sources: List[str] = None,
        conviction_at_entry: int = None,
        spy_return_same_period: float = None
    ) -> str:
        """
        Record a closed trade with full metrics.

        Returns:
            outcome_id
        """
        outcome_id = str(uuid.uuid4())

        # Calculate metrics
        realized_pnl = (exit_price - entry_price) * quantity
        realized_pnl_pct = (exit_price - entry_price) / entry_price

        # Calculate hold time
        entry_dt = self._parse_datetime(entry_date)
        exit_dt = self._parse_datetime(exit_date)
        hold_time_hours = None
        if entry_dt and exit_dt:
            hold_time_hours = (exit_dt - entry_dt).total_seconds() / 3600

        # Calculate alpha
        alpha = None
        if spy_return_same_period is not None:
            alpha = realized_pnl_pct - spy_return_same_period

        # Store in database
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_outcomes
                (id, trade_id, ticker, side, entry_price, entry_date, exit_price, exit_date,
                 exit_reason, realized_pnl, realized_pnl_pct, hold_time_hours,
                 spy_return_same_period, alpha, entry_conviction, primary_signal_source,
                 signal_sources_agreed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome_id, trade_id, ticker, side, entry_price, entry_date,
                exit_price, exit_date, exit_reason, realized_pnl, realized_pnl_pct,
                hold_time_hours, spy_return_same_period, alpha, conviction_at_entry,
                signal_sources[0] if signal_sources else None,
                len(signal_sources) if signal_sources else 0,
                datetime.now(timezone.utc).isoformat()
            ))

        logger.info(f"Recorded trade outcome: {ticker} P&L=${realized_pnl:.2f} alpha={alpha:.2%}" if alpha else f"Recorded trade outcome: {ticker} P&L=${realized_pnl:.2f}")
        return outcome_id

    def get_trade_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a trade outcome by ID."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_outcomes WHERE id = ?", (outcome_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def detect_closed_positions(
        self,
        previous_positions: Dict[str, Dict],
        current_positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Detect positions that have been closed.

        Args:
            previous_positions: Dict of ticker -> position data (from last check)
            current_positions: Dict of ticker -> position data (current)
            current_prices: Dict of ticker -> current price

        Returns:
            List of closed position dicts
        """
        closed = []

        for ticker, prev_pos in previous_positions.items():
            if ticker not in current_positions:
                # Position was closed
                closed.append({
                    "ticker": ticker,
                    "quantity": prev_pos.get("quantity", 0),
                    "entry_price": prev_pos.get("entry_price", prev_pos.get("avg_entry_price", 0)),
                    "exit_price": current_prices.get(ticker, prev_pos.get("entry_price", 0)),
                    "entry_date": prev_pos.get("entry_date"),
                    "exit_date": datetime.now(timezone.utc).isoformat()
                })

        return closed

    # =========================================================================
    # SPY DATA FETCHING
    # =========================================================================

    async def fetch_spy_price(self) -> Optional[float]:
        """Fetch current SPY price from executor."""
        if not self.executor:
            logger.warning("No executor configured for SPY fetch")
            return None

        try:
            price = await self.executor.get_current_price("SPY")
            return price
        except Exception as e:
            logger.error(f"Error fetching SPY price: {e}")
            return None

    def calculate_spy_period_return(
        self,
        start_price: float,
        end_price: float
    ) -> float:
        """Calculate SPY return over a period."""
        if start_price <= 0:
            return 0.0
        return (end_price - start_price) / start_price

    def store_spy_benchmark(
        self,
        date: str,
        open_price: float = None,
        close_price: float = None,
        high: float = None,
        low: float = None
    ) -> None:
        """Store daily SPY benchmark data."""
        benchmark_id = str(uuid.uuid4())

        daily_return = None
        if open_price and close_price:
            daily_return = (close_price - open_price) / open_price

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO spy_benchmarks
                (id, date, open_price, close_price, high, low, daily_return, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark_id, date, open_price, close_price, high, low,
                daily_return, datetime.now(timezone.utc).isoformat()
            ))

    def get_spy_benchmark(self, date: str) -> Optional[Dict[str, Any]]:
        """Retrieve SPY benchmark for a specific date."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM spy_benchmarks WHERE date = ?", (date,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # WEEKLY PERFORMANCE DIGEST
    # =========================================================================

    def generate_weekly_digest(
        self,
        trades: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Generate weekly performance digest.

        Args:
            trades: List of trade outcome dicts
            start_date: Week start date
            end_date: Week end date

        Returns:
            Digest dict with all metrics
        """
        if not trades:
            return {
                "period": f"{start_date} to {end_date}",
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_alpha": 0,
                "top_winner": None,
                "top_loser": None
            }

        # Calculate metrics
        wins = [t for t in trades if t.get("realized_pnl", 0) > 0]
        losses = [t for t in trades if t.get("realized_pnl", 0) <= 0]

        total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
        total_alpha = sum(t.get("alpha", 0) or 0 for t in trades)

        # Find top winner and loser
        sorted_by_pnl = sorted(trades, key=lambda x: x.get("realized_pnl", 0), reverse=True)
        top_winner = sorted_by_pnl[0] if sorted_by_pnl and sorted_by_pnl[0].get("realized_pnl", 0) > 0 else None
        top_loser = sorted_by_pnl[-1] if sorted_by_pnl and sorted_by_pnl[-1].get("realized_pnl", 0) < 0 else None

        # Calculate hold times
        hold_times = [t.get("hold_time_hours", 0) or 0 for t in trades if t.get("hold_time_hours")]
        avg_hold_hours = sum(hold_times) / len(hold_times) if hold_times else 0
        avg_hold_days = avg_hold_hours / 24

        return {
            "period": f"{start_date} to {end_date}",
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "total_alpha": total_alpha,
            "avg_hold_days": avg_hold_days,
            "top_winner": {
                "ticker": top_winner.get("ticker"),
                "pnl": top_winner.get("realized_pnl"),
                "return_pct": top_winner.get("realized_pnl_pct")
            } if top_winner else None,
            "top_loser": {
                "ticker": top_loser.get("ticker"),
                "pnl": top_loser.get("realized_pnl"),
                "return_pct": top_loser.get("realized_pnl_pct")
            } if top_loser else None
        }

    # =========================================================================
    # DYNAMIC THRESHOLD ADJUSTMENT
    # =========================================================================

    def calculate_optimal_threshold(
        self,
        historical_trades: List[Dict[str, Any]]
    ) -> int:
        """
        Calculate optimal conviction threshold from historical data.

        Finds the conviction level that maximizes win rate while
        maintaining reasonable trade volume.

        Returns:
            Optimal conviction threshold (70-95)
        """
        if not historical_trades:
            return 80  # Default

        # Group by conviction buckets
        buckets = {}
        for trade in historical_trades:
            conv = trade.get("conviction", 0)
            bucket = (conv // 5) * 5  # Round to nearest 5

            if bucket not in buckets:
                buckets[bucket] = {"wins": 0, "total": 0}

            buckets[bucket]["total"] += 1
            if trade.get("won"):
                buckets[bucket]["wins"] += 1

        # Find threshold with best risk-adjusted performance
        best_threshold = 80
        best_score = 0

        for threshold in range(95, 69, -5):
            trades_above = sum(b["total"] for c, b in buckets.items() if c >= threshold)
            wins_above = sum(b["wins"] for c, b in buckets.items() if c >= threshold)

            if trades_above < 3:  # Need minimum sample
                continue

            win_rate = wins_above / trades_above

            # Score = win_rate * sqrt(trade_count) to balance accuracy and volume
            score = win_rate * (trades_above ** 0.5)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def adjust_threshold_for_regime(
        self,
        base_threshold: int,
        vix_regime: str,
        spy_trend: str
    ) -> int:
        """
        Adjust conviction threshold based on market regime.

        In risky regimes, require higher conviction.
        In favorable regimes, can be slightly more aggressive.

        Returns:
            Adjusted threshold
        """
        adjustment = 0

        # VIX adjustments
        vix_adjustments = {
            "low": -3,      # Can be more aggressive
            "normal": 0,    # No change
            "elevated": 5,  # More cautious
            "extreme": 10   # Very cautious
        }
        adjustment += vix_adjustments.get(vix_regime, 0)

        # SPY trend adjustments
        trend_adjustments = {
            "strong_up": -2,
            "up": 0,
            "sideways": 2,
            "down": 5,
            "strong_down": 8
        }
        adjustment += trend_adjustments.get(spy_trend, 0)

        # Clamp result
        adjusted = base_threshold + adjustment
        return max(70, min(95, adjusted))

    def get_dynamic_threshold(self) -> int:
        """
        Get current dynamic conviction threshold.

        Combines historical optimization with regime adjustment.
        """
        # Get historical trades for optimization
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_conviction as conviction,
                       CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END as won
                FROM trade_outcomes
                WHERE entry_conviction IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 50
            """)
            rows = cursor.fetchall()
            historical = [dict(r) for r in rows]

        # Calculate base threshold
        if historical:
            base_threshold = self.calculate_optimal_threshold(historical)
        else:
            base_threshold = 80

        # Get current regime
        regime = self.get_current_regime()
        vix_regime = regime.get("vix_regime", "normal")
        spy_trend = regime.get("spy_trend", "sideways")

        # Adjust for regime
        return self.adjust_threshold_for_regime(base_threshold, vix_regime, spy_trend)

    # =========================================================================
    # PATTERN LIBRARY
    # =========================================================================

    def save_pattern(
        self,
        pattern_type: str,
        outcome: str,
        ticker: str = None,
        entry_catalyst: str = None,
        entry_signals: Dict[str, str] = None,
        return_pct: float = None,
        hold_days: int = None,
        market_regime: str = None,
        conviction_at_entry: int = None,
        notes: str = None,
        trade_id: str = None
    ) -> str:
        """
        Save a trading pattern to the library.

        Args:
            pattern_type: e.g., "earnings_momentum", "sector_rotation", "technical_breakout"
            outcome: "win" or "loss"
            ...

        Returns:
            pattern_id
        """
        pattern_id = str(uuid.uuid4())

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patterns
                (id, pattern_type, ticker, entry_catalyst, entry_signals, outcome,
                 return_pct, hold_days, market_regime, conviction_at_entry, notes,
                 trade_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id, pattern_type, ticker, entry_catalyst,
                json.dumps(entry_signals) if entry_signals else None,
                outcome, return_pct, hold_days, market_regime,
                conviction_at_entry, notes, trade_id,
                datetime.now(timezone.utc).isoformat()
            ))

        logger.info(f"Saved pattern: {pattern_type} -> {outcome}")
        return pattern_id

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pattern by ID."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get("entry_signals"):
                    d["entry_signals"] = json.loads(d["entry_signals"])
                return d
            return None

    def find_similar_patterns(
        self,
        pattern_type: str = None,
        market_regime: str = None,
        ticker: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find historically similar patterns.

        Args:
            pattern_type: Filter by pattern type
            market_regime: Filter by market regime
            ticker: Filter by specific ticker
            limit: Max results

        Returns:
            List of matching patterns
        """
        conditions = []
        params = []

        if pattern_type:
            conditions.append("pattern_type = ?")
            params.append(pattern_type)

        if market_regime:
            conditions.append("market_regime = ?")
            params.append(market_regime)

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM patterns
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                d = dict(row)
                if d.get("entry_signals"):
                    d["entry_signals"] = json.loads(d["entry_signals"])
                results.append(d)

            return results

    def generate_pattern_context(
        self,
        ticker: str,
        pattern_type: str = None,
        current_regime: str = None
    ) -> Dict[str, Any]:
        """
        Generate pattern context for Claude's decision making.

        Returns historical performance for similar situations.
        """
        # Find similar patterns
        similar = self.find_similar_patterns(
            pattern_type=pattern_type,
            market_regime=current_regime,
            limit=20
        )

        if not similar:
            return {
                "similar_patterns": 0,
                "historical_win_rate": None,
                "avg_return": None,
                "message": "No historical patterns found for this setup"
            }

        wins = [p for p in similar if p.get("outcome") == "win"]
        win_rate = len(wins) / len(similar)

        returns = [p.get("return_pct", 0) for p in similar if p.get("return_pct") is not None]
        avg_return = sum(returns) / len(returns) if returns else 0

        return {
            "similar_patterns": len(similar),
            "historical_win_rate": win_rate,
            "avg_return": avg_return,
            "avg_hold_days": sum(p.get("hold_days", 0) or 0 for p in similar) / len(similar),
            "winning_examples": [
                {"ticker": p.get("ticker"), "return": p.get("return_pct")}
                for p in wins[:3]
            ],
            "message": f"Found {len(similar)} similar patterns with {win_rate:.0%} win rate"
        }

    def get_pattern_library_stats(self) -> Dict[str, Any]:
        """Get statistics about the pattern library."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            # Total patterns
            cursor.execute("SELECT COUNT(*) as count FROM patterns")
            total = cursor.fetchone()["count"]

            # By type
            cursor.execute("""
                SELECT pattern_type, COUNT(*) as count,
                       SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
                FROM patterns
                GROUP BY pattern_type
            """)
            by_type = {}
            for row in cursor.fetchall():
                by_type[row["pattern_type"]] = {
                    "count": row["count"],
                    "win_rate": row["wins"] / row["count"] if row["count"] > 0 else 0
                }

            # Overall win rate
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
                FROM patterns
            """)
            overall = cursor.fetchone()
            overall_wr = overall["wins"] / overall["total"] if overall["total"] > 0 else 0

            return {
                "total_patterns": total,
                "by_type": by_type,
                "overall_win_rate": overall_wr
            }

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
    with db._get_connection() as conn:
        cursor = conn.cursor()

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
                high REAL,
                low REAL,
                daily_return REAL,
                cumulative_return_30d REAL,
                cumulative_return_90d REAL,
                cumulative_return_ytd REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pattern library table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                ticker TEXT,
                entry_catalyst TEXT,
                entry_signals TEXT,
                outcome TEXT NOT NULL,
                return_pct REAL,
                hold_days INTEGER,
                market_regime TEXT,
                conviction_at_entry INTEGER,
                notes TEXT,
                trade_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for pattern queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_regime ON patterns(market_regime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_outcome ON patterns(outcome)")

    logger.info("Learning Engine tables created")
