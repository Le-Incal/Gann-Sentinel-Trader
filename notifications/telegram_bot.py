"""
Telegram Bot for Gann Sentinel Trader
Handles notifications, commands, and daily digests.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import httpx

logger = logging.getLogger(__name__)


@dataclass
class SourceQuery:
    """Track individual source queries."""
    source: str
    query: str
    timestamp_utc: datetime
    signals_returned: int
    error: Optional[str] = None


@dataclass
class DigestData:
    """Accumulates data throughout the day for the daily digest."""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    run_count: int = 0
    
    # Source tracking
    source_queries: List[SourceQuery] = field(default_factory=list)
    
    # Signal tracking
    signals_by_source: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signals_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signal_themes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Staleness tracking
    stale_signals_excluded: List[Dict[str, Any]] = field(default_factory=list)
    
    # Decision tracking
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error tracking
    retrieval_errors: List[Dict[str, Any]] = field(default_factory=list)
    system_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Known blind spots (static + dynamic)
    blind_spots: List[str] = field(default_factory=list)
    
    def reset(self):
        """Reset for new day."""
        self.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_count = 0
        self.source_queries = []
        self.signals_by_source = defaultdict(int)
        self.signals_by_category = defaultdict(int)
        self.signal_themes = defaultdict(int)
        self.stale_signals_excluded = []
        self.decisions = []
        self.retrieval_errors = []
        self.system_errors = []
        self.blind_spots = []


class TelegramBot:
    """Telegram bot for notifications and commands."""
    
    # Static blind spots we always report
    STATIC_BLIND_SPOTS = [
        "Chinese language sources not checked",
        "Options flow data not available",
        "Earnings calendar not integrated",
        "After-hours price moves not captured",
        "Dark pool activity not tracked",
    ]
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.digest_data = DigestData()
        self._last_digest_date: Optional[str] = None
        
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    # ══════════════════════════════════════════════════════════════════
    # DIGEST DATA COLLECTION METHODS
    # ══════════════════════════════════════════════════════════════════
    
    def record_scan_start(self):
        """Call at the start of each scan cycle."""
        # Check if we need to reset for new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.digest_data.date != today:
            self.digest_data.reset()
        
        self.digest_data.run_count += 1
    
    def record_source_query(
        self, 
        source: str, 
        query: str, 
        signals_returned: int,
        error: Optional[str] = None
    ):
        """Record a source query for the digest."""
        self.digest_data.source_queries.append(SourceQuery(
            source=source,
            query=query,
            timestamp_utc=datetime.now(timezone.utc),
            signals_returned=signals_returned,
            error=error
        ))
        
        if error:
            self.digest_data.retrieval_errors.append({
                "source": source,
                "query": query,
                "error": error,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            })
    
    def record_signal(self, signal: Dict[str, Any]):
        """Record a signal for the digest."""
        source = signal.get("source_type", signal.get("source", "unknown"))
        category = signal.get("category", "unknown")
        
        self.digest_data.signals_by_source[source] += 1
        self.digest_data.signals_by_category[category] += 1
        
        # Extract themes from summary
        summary = signal.get("summary", "").lower()
        theme_keywords = {
            "fed": "Fed rate expectations",
            "rate cut": "Fed rate expectations",
            "rate hike": "Fed rate expectations",
            "inflation": "Inflation concerns",
            "cpi": "Inflation concerns",
            "tariff": "Trade/tariff policy",
            "china": "China trade policy",
            "ai": "AI/semiconductor demand",
            "nvidia": "AI/semiconductor demand",
            "chip": "AI/semiconductor demand",
            "semiconductor": "AI/semiconductor demand",
            "earnings": "Earnings season",
            "recession": "Recession risk",
            "unemployment": "Labor market",
            "jobs": "Labor market",
        }
        
        for keyword, theme in theme_keywords.items():
            if keyword in summary:
                self.digest_data.signal_themes[theme] += 1
                break
    
    def record_stale_signal(self, signal: Dict[str, Any], reason: str):
        """Record a signal that was excluded due to staleness."""
        self.digest_data.stale_signals_excluded.append({
            "signal_id": signal.get("signal_id"),
            "source": signal.get("source_type", signal.get("source")),
            "category": signal.get("category"),
            "reason": reason,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        })
    
    def record_decision(self, decision: Dict[str, Any]):
        """Record a decision (trade or no-trade) for the digest."""
        self.digest_data.decisions.append({
            "decision_type": decision.get("decision_type"),
            "ticker": decision.get("trade_details", {}).get("ticker"),
            "side": decision.get("trade_details", {}).get("side"),
            "conviction": decision.get("trade_details", {}).get("conviction_score"),
            "rationale": decision.get("reasoning", {}).get("rationale", "")[:100],
            "status": decision.get("status", "logged"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        })
    
    def record_system_error(self, component: str, error: str, critical: bool = False):
        """Record a system error for the digest."""
        self.digest_data.system_errors.append({
            "component": component,
            "error": error,
            "critical": critical,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        })
    
    def add_blind_spot(self, blind_spot: str):
        """Add a dynamic blind spot discovered during scanning."""
        if blind_spot not in self.digest_data.blind_spots:
            self.digest_data.blind_spots.append(blind_spot)
    
    # ══════════════════════════════════════════════════════════════════
    # DIGEST GENERATION
    # ══════════════════════════════════════════════════════════════════
    
    def _format_sources_section(self) -> str:
        """Format the sources reviewed section."""
        lines = ["═══ SOURCES REVIEWED ═══"]
        
        # Aggregate by source
        source_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "queries": 0,
            "signals": 0,
            "errors": 0,
            "query_list": []
        })
        
        for sq in self.digest_data.source_queries:
            stats = source_stats[sq.source]
            stats["queries"] += 1
            stats["signals"] += sq.signals_returned
            if sq.error:
                stats["errors"] += 1
            if sq.query and len(stats["query_list"]) < 5:
                stats["query_list"].append(sq.query)
        
        # Format each source
        for source, stats in sorted(source_stats.items()):
            if stats["errors"] > 0:
                status = "⚠️"
                error_note = f" | {stats['errors']} errors"
            else:
                status = "✅"
                error_note = ""
            
            lines.append(
                f"{status} {source:<18} | {stats['queries']} queries | "
                f"{stats['signals']} signals{error_note}"
            )
        
        # Source details
        lines.append("")
        lines.append("Source Details:")
        
        for source, stats in sorted(source_stats.items()):
            if stats["query_list"]:
                queries_str = ", ".join(f'"{q}"' for q in stats["query_list"][:3])
                if len(stats["query_list"]) > 3:
                    queries_str += "..."
                lines.append(f"• {source}: {queries_str}")
        
        return "\n".join(lines)
    
    def _format_errors_section(self) -> str:
        """Format the retrieval errors section."""
        lines = ["═══ RETRIEVAL ERRORS ═══"]
        
        if not self.digest_data.retrieval_errors:
            lines.append("None")
        else:
            # Group by source
            by_source: Dict[str, List[Dict]] = defaultdict(list)
            for err in self.digest_data.retrieval_errors:
                by_source[err["source"]].append(err)
            
            for source, errors in by_source.items():
                error_types = defaultdict(int)
                times = []
                for err in errors:
                    error_types[err["error"]] += 1
                    times.append(err["timestamp_utc"].split("T")[1][:5])
                
                for err_type, count in error_types.items():
                    times_str = ", ".join(times[:3])
                    if len(times) > 3:
                        times_str += "..."
                    lines.append(f"• {source}: {err_type} ({count}x) - {times_str} UTC")
        
        return "\n".join(lines)
    
    def _format_blind_spots_section(self) -> str:
        """Format the known blind spots section."""
        lines = ["═══ KNOWN BLIND SPOTS ═══"]
        
        all_blind_spots = self.STATIC_BLIND_SPOTS + self.digest_data.blind_spots
        for spot in all_blind_spots:
            lines.append(f"• {spot}")
        
        return "\n".join(lines)
    
    def _format_signals_section(self) -> str:
        """Format the signals received section."""
        total = sum(self.digest_data.signals_by_source.values())
        
        lines = [
            "═══ SIGNALS RECEIVED ═══",
            f"Total: {total}",
            "",
            "By Source:"
        ]
        
        for source, count in sorted(
            self.digest_data.signals_by_source.items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"• {source}: {count}")
        
        lines.append("")
        lines.append("By Category:")
        
        for category, count in sorted(
            self.digest_data.signals_by_category.items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"• {category}: {count}")
        
        # Top themes
        if self.digest_data.signal_themes:
            lines.append("")
            lines.append("Top Themes:")
            sorted_themes = sorted(
                self.digest_data.signal_themes.items(),
                key=lambda x: -x[1]
            )[:5]
            for theme, count in sorted_themes:
                lines.append(f"• {theme} ({count} signals)")
        
        return "\n".join(lines)
    
    def _format_stale_section(self) -> str:
        """Format the stale signals section."""
        lines = ["═══ STALE SIGNALS EXCLUDED ═══"]
        
        if not self.digest_data.stale_signals_excluded:
            lines.append("None")
        else:
            # Group by source
            by_source: Dict[str, int] = defaultdict(int)
            for stale in self.digest_data.stale_signals_excluded:
                by_source[stale.get("source", "unknown")] += 1
            
            for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
                lines.append(f"• {count} {source} signals")
        
        return "\n".join(lines)
    
    def _format_decisions_section(self) -> str:
        """Format the decisions section."""
        lines = ["═══ DECISIONS ═══"]
        
        trades = [d for d in self.digest_data.decisions if d["decision_type"] == "TRADE"]
        no_trades = [d for d in self.digest_data.decisions if d["decision_type"] == "NO_TRADE"]
        watches = [d for d in self.digest_data.decisions if d["decision_type"] == "WATCH"]
        
        lines.append(f"TRADE: {len(trades)}")
        for t in trades:
            lines.append(
                f"• {t['ticker']} | {t['side']} | "
                f"Conviction: {t['conviction']} | Status: {t['status']}"
            )
        
        lines.append(f"\nNO_TRADE: {len(no_trades)}")
        # Summarize reasons
        reasons: Dict[str, int] = defaultdict(int)
        for nt in no_trades:
            rationale = nt.get("rationale", "")
            if "conviction" in rationale.lower():
                reasons["Insufficient conviction"] += 1
            elif "stale" in rationale.lower():
                reasons["Stale data"] += 1
            elif "risk" in rationale.lower():
                reasons["Risk limit"] += 1
            else:
                reasons["Other"] += 1
        
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"• {reason} ({count})")
        
        if watches:
            lines.append(f"\nWATCH: {len(watches)}")
            for w in watches:
                lines.append(f"• {w['ticker']} - {w.get('rationale', '')[:50]}")
        
        return "\n".join(lines)
    
    def _format_system_errors_section(self) -> str:
        """Format the system errors section."""
        lines = ["═══ SYSTEM ERRORS ═══"]
        
        if not self.digest_data.system_errors:
            lines.append("None")
        else:
            by_component: Dict[str, List[str]] = defaultdict(list)
            for err in self.digest_data.system_errors:
                by_component[err["component"]].append(err["error"])
            
            for component, errors in by_component.items():
                error_counts = defaultdict(int)
                for e in errors:
                    error_counts[e] += 1
                
                for error, count in error_counts.items():
                    critical_marker = "🚨 " if any(
                        e.get("critical") for e in self.digest_data.system_errors
                        if e["component"] == component and e["error"] == error
                    ) else ""
                    lines.append(f"• {critical_marker}{component}: {error} ({count}x)")
        
        return "\n".join(lines)
    
    async def send_daily_digest(
        self, 
        positions: List[Dict[str, Any]] = None,
        portfolio: Dict[str, Any] = None,
        pending_approvals: List[Dict[str, Any]] = None
    ) -> bool:
        """Generate and send the daily digest."""
        
        now = datetime.now(timezone.utc)
        
        # Header
        digest_parts = [
            "📊 GANN SENTINEL DAILY DIGEST",
            f"Date: {self.digest_data.date}",
            f"Run Count: {self.digest_data.run_count}",
            f"Generated: {now.strftime('%H:%M')} UTC",
            "",
        ]
        
        # Sources reviewed
        digest_parts.append(self._format_sources_section())
        digest_parts.append("")
        
        # Retrieval errors
        digest_parts.append(self._format_errors_section())
        digest_parts.append("")
        
        # Known blind spots
        digest_parts.append(self._format_blind_spots_section())
        digest_parts.append("")
        
        # Signals received
        digest_parts.append(self._format_signals_section())
        digest_parts.append("")
        
        # Stale signals
        digest_parts.append(self._format_stale_section())
        digest_parts.append("")
        
        # Decisions
        digest_parts.append(self._format_decisions_section())
        digest_parts.append("")
        
        # Positions
        digest_parts.append("═══ POSITIONS ═══")
        if not positions:
            digest_parts.append("None")
        else:
            for pos in positions:
                pnl_pct = pos.get("unrealized_pnl_pct", 0) * 100
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                digest_parts.append(
                    f"{pnl_emoji} {pos['ticker']} | {pos['quantity']} shares | "
                    f"${pos.get('market_value', 0):,.2f} | {pnl_pct:+.1f}%"
                )
        digest_parts.append("")
        
        # Portfolio
        digest_parts.append("═══ PORTFOLIO ═══")
        if portfolio:
            digest_parts.append(f"Cash: ${portfolio.get('cash', 0):,.2f}")
            digest_parts.append(f"Positions Value: ${portfolio.get('positions_value', 0):,.2f}")
            digest_parts.append(f"Total: ${portfolio.get('total_value', 0):,.2f}")
            daily_pnl = portfolio.get('daily_pnl', 0)
            daily_pnl_pct = portfolio.get('daily_pnl_pct', 0) * 100
            pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
            digest_parts.append(f"Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        else:
            digest_parts.append("Portfolio data unavailable")
        digest_parts.append("")
        
        # System errors
        digest_parts.append(self._format_system_errors_section())
        digest_parts.append("")
        
        # Pending approvals
        digest_parts.append("═══ PENDING APPROVAL ═══")
        if not pending_approvals:
            digest_parts.append("None")
        else:
            for pa in pending_approvals:
                digest_parts.append(f"ID: {pa['id']}")
                digest_parts.append(
                    f"Ticker: {pa['ticker']} | {pa['side']} | "
                    f"{pa.get('position_size_pct', 0)*100:.0f}% position"
                )
                digest_parts.append(f"Thesis: {pa.get('thesis', '')[:80]}...")
                digest_parts.append(f"/approve {pa['id']}")
                digest_parts.append("")
        
        # Combine and send
        full_digest = "\n".join(digest_parts)
        
        # Telegram has a 4096 char limit, split if needed
        if len(full_digest) > 4000:
            # Send in parts
            parts = self._split_message(full_digest, 4000)
            for i, part in enumerate(parts):
                if i > 0:
                    part = f"... (continued {i+1}/{len(parts)})\n\n" + part
                await self.send_message(part, parse_mode=None)
            return True
        else:
            return await self.send_message(full_digest, parse_mode=None)
    
    def _split_message(self, text: str, max_len: int) -> List[str]:
        """Split a message at section boundaries."""
        parts = []
        current = ""
        
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > max_len:
                parts.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        
        if current:
            parts.append(current)
        
        return parts
    
    # ══════════════════════════════════════════════════════════════════
    # EXISTING NOTIFICATION METHODS
    # ══════════════════════════════════════════════════════════════════
    
    async def send_startup_message(self):
        """Send startup notification."""
        await self.send_message("🚀 Gann Sentinel Trader Started")
    
    async def send_trade_recommendation(
        self,
        trade_id: str,
        ticker: str,
        side: str,
        conviction: int,
        position_size_pct: float,
        thesis: str,
        bull_case: str,
        bear_case: str,
        time_horizon: str,
        stop_loss_pct: float,
        entry_price: float,
        signals_count: int
    ):
        """Send a trade recommendation for approval."""
        stop_loss_price = entry_price * (1 - stop_loss_pct) if side == "BUY" else entry_price * (1 + stop_loss_pct)
        
        message = f"""
🔔 TRADE RECOMMENDATION

Ticker: {ticker}
Action: {side}
Conviction: {conviction}/100
Position Size: {position_size_pct*100:.0f}%

📈 THESIS
{thesis}

✅ BULL CASE
{bull_case}

⚠️ BEAR CASE
{bear_case}

⏰ Time Horizon: {time_horizon}
🛑 Stop Loss: {stop_loss_pct*100:.0f}% (${stop_loss_price:.2f})

Signals Used: {signals_count}

/approve {trade_id}
/reject {trade_id}
"""
        return await self.send_message(message.strip(), parse_mode=None)
    
    async def send_trade_executed(
        self,
        ticker: str,
        side: str,
        quantity: float,
        fill_price: float,
        thesis: str
    ):
        """Send trade execution confirmation."""
        message = f"""
✅ TRADE EXECUTED

{side} {quantity} {ticker} @ ${fill_price:.2f}

Thesis: {thesis[:100]}...
"""
        return await self.send_message(message.strip(), parse_mode=None)
    
    async def send_stop_loss_triggered(
        self,
        ticker: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ):
        """Send stop loss notification."""
        message = f"""
🛑 STOP LOSS TRIGGERED

{ticker}: Sold {quantity} shares
Entry: ${entry_price:.2f}
Exit: ${exit_price:.2f}
P&L: ${pnl:.2f} ({pnl_pct*100:+.1f}%)
"""
        return await self.send_message(message.strip(), parse_mode=None)
    
    async def send_error_alert(self, component: str, error: str, critical: bool = False):
        """Send error alert."""
        emoji = "🚨" if critical else "⚠️"
        message = f"{emoji} ERROR: {component}\n{error}"
        
        # Also record for digest
        self.record_system_error(component, error, critical)
        
        return await self.send_message(message)
    
    async def send_circuit_breaker_alert(self, breaker: str, reset_time: str):
        """Send circuit breaker notification."""
        message = f"""
🔴 CIRCUIT BREAKER TRIGGERED

{breaker}

Trading halted until: {reset_time}
Use /resume to manually restart.
"""
        return await self.send_message(message.strip(), parse_mode=None)
    
    async def send_status(
        self,
        positions: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        system_status: str
    ):
        """Send current status."""
        lines = [
            "📊 STATUS",
            f"System: {system_status}",
            "",
            "Portfolio:",
            f"  Cash: ${portfolio.get('cash', 0):,.2f}",
            f"  Positions: ${portfolio.get('positions_value', 0):,.2f}",
            f"  Total: ${portfolio.get('total_value', 0):,.2f}",
            "",
            "Positions:"
        ]
        
        if not positions:
            lines.append("  None")
        else:
            for pos in positions:
                pnl_pct = pos.get('unrealized_pnl_pct', 0) * 100
                lines.append(
                    f"  {pos['ticker']}: {pos['quantity']} @ ${pos.get('avg_entry_price', 0):.2f} "
                    f"({pnl_pct:+.1f}%)"
                )
        
        return await self.send_message("\n".join(lines), parse_mode=None)
    
    # ══════════════════════════════════════════════════════════════════
    # COMMAND PROCESSING
    # ══════════════════════════════════════════════════════════════════
    
    async def get_updates(self) -> List[Dict]:
        """Get new messages/commands from Telegram."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self.last_update_id + 1,
                        "timeout": 1
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        updates = data.get("result", [])
                        if updates:
                            self.last_update_id = updates[-1]["update_id"]
                        return updates
                return []
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return []
    
    def parse_command(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse a command from message text."""
        if not text or not text.startswith("/"):
            return None, None
        
        parts = text.split(maxsplit=1)
        command = parts[0][1:].lower()  # Remove leading /
        arg = parts[1] if len(parts) > 1 else None
        
        return command, arg


# Factory function
def create_telegram_bot() -> TelegramBot:
    """Create a TelegramBot instance from environment variables."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    
    return TelegramBot(token, chat_id)
