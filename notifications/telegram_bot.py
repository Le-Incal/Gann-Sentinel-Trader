"""
Telegram Bot for Gann Sentinel Trader
Handles notifications and command processing for trade approvals and system control.

Version: 2.1.0 - Added Activity Logging for Full Observability
- All outgoing messages logged to database
- All incoming commands logged to database
- New /logs command to view activity
- New /export_logs command for formatted export
- Integration with Database class for persistent storage

Uses Unicode escape sequences for all emojis and special characters.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import httpx

# Avoid circular import
if TYPE_CHECKING:
    from storage.database import Database

logger = logging.getLogger(__name__)


# =============================================================================
# EMOJI & CHARACTER CONSTANTS
# =============================================================================
EMOJI_ROCKET = "\U0001F680"      # 🚀
EMOJI_STOP = "\U0001F6D1"        # 🛑
EMOJI_CHART = "\U0001F4CA"       # 📊
EMOJI_MONEY = "\U0001F4B0"       # 💰
EMOJI_CHART_UP = "\U0001F4C8"    # 📈
EMOJI_CHART_DOWN = "\U0001F4C9"  # 📉
EMOJI_WARNING = "\U000026A0"     # ⚠
EMOJI_CHECK = "\U00002705"       # ✅
EMOJI_CROSS = "\U0000274C"       # ❌
EMOJI_BELL = "\U0001F514"        # 🔔
EMOJI_BRAIN = "\U0001F9E0"       # 🧠
EMOJI_SEARCH = "\U0001F50D"      # 🔍
EMOJI_TARGET = "\U0001F3AF"      # 🎯
EMOJI_HOURGLASS = "\U000023F3"   # ⏳
EMOJI_GREEN_CIRCLE = "\U0001F7E2"  # 🟢
EMOJI_RED_CIRCLE = "\U0001F534"    # 🔴
EMOJI_YELLOW_CIRCLE = "\U0001F7E1" # 🟡
EMOJI_WHITE_CIRCLE = "\U000026AA"  # ⚪
EMOJI_BIRD = "\U0001F426"        # 🐦
EMOJI_ANTENNA = "\U0001F4E1"     # 📡
EMOJI_MEMO = "\U0001F4CB"        # 📋
EMOJI_BULLET = "\U00002022"      # •
EMOJI_KEYBOARD = "\U00002328"    # ⌨
EMOJI_BEAR = "\U0001F43B"        # 🐻
EMOJI_BULL = "\U0001F402"        # 🐂
EMOJI_ZZZ = "\U0001F4A4"         # 💤
EMOJI_CANDLE = "\U0001F56F"      # 🕯️
EMOJI_RULER = "\U0001F4CF"       # 📏
EMOJI_SCROLL = "\U0001F4DC"      # 📜 (NEW - for logs)
EMOJI_EYES = "\U0001F440"        # 👀 (NEW - for activity)

# Progress bar characters (Unicode block elements)
BAR_FILLED = "\U00002588"        # Full block
BAR_EMPTY = "\U00002591"         # Light shade


class TelegramBot:
    """
    Telegram bot for Gann Sentinel Trader notifications and commands.

    Version 2.1.0 adds full activity logging:
    - Every message sent is logged to telegram_messages table
    - Every command received is logged
    - New /logs and /export_logs commands for observability
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        db: Optional["Database"] = None
    ):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.db = db  # Database instance for logging

        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN not set - notifications disabled")
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set - notifications disabled")

        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.last_update_id = 0

        # Digest tracking state
        self._scan_start_time: Optional[datetime] = None
        self._source_queries: List[Dict[str, Any]] = []
        self._signals: List[Dict[str, Any]] = []
        self._decisions: List[Dict[str, Any]] = []
        self._system_errors: List[Dict[str, Any]] = []
        self._pending_approvals: List[str] = []
        self._risk_rejections: List[Dict[str, Any]] = []
        self._trade_blockers: List[Dict[str, Any]] = []
        self._technical_signals: List[Dict[str, Any]] = []

    def set_database(self, db: "Database") -> None:
        """Set the database instance for logging (allows deferred initialization)."""
        self.db = db

    @property
    def is_configured(self) -> bool:
        """Check if bot is properly configured."""
        return bool(self.token and self.chat_id)

    # =========================================================================
    # ACTIVITY LOGGING
    # =========================================================================

    def _log_outgoing(
        self,
        content: str,
        message_type: str,
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Log an outgoing message to the database."""
        if not self.db:
            return

        try:
            self.db.log_telegram_message(
                direction="outgoing",
                message_type=message_type,
                content=content,
                related_entity_id=related_entity_id,
                related_entity_type=related_entity_type,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to log outgoing message: {e}")

    def _log_incoming(
        self,
        content: str,
        command: str,
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Log an incoming command to the database."""
        if not self.db:
            return

        try:
            self.db.log_telegram_message(
                direction="incoming",
                message_type="command",
                content=content,
                command=command,
                related_entity_id=related_entity_id,
                related_entity_type=related_entity_type,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to log incoming command: {e}")

    # =========================================================================
    # CORE MESSAGING
    # =========================================================================

    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = "Markdown",
        disable_notification: bool = False,
        message_type: str = "notification",
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None
    ) -> bool:
        """Send a message to Telegram with fallback for parse errors."""
        if not self.is_configured:
            logger.warning("Telegram not configured - message not sent")
            return False

        target_chat = chat_id or self.chat_id

        # Log outgoing message
        self._log_outgoing(
            content=text,
            message_type=message_type,
            related_entity_id=related_entity_id,
            related_entity_type=related_entity_type
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "chat_id": target_chat,
                    "text": text,
                    "disable_notification": disable_notification
                }

                if parse_mode:
                    payload["parse_mode"] = parse_mode

                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                )

                if response.status_code == 200:
                    return True

                # If Markdown failed, retry without parse mode
                if parse_mode and response.status_code == 400:
                    logger.warning("Markdown parse failed, retrying without formatting")
                    payload.pop("parse_mode", None)
                    response = await client.post(
                        f"{self.base_url}/sendMessage",
                        json=payload
                    )
                    return response.status_code == 200

                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    async def get_commands(self) -> List[Dict[str, Any]]:
        """Poll for new commands."""
        if not self.is_configured:
            return []

        commands = []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self.last_update_id + 1,
                        "timeout": 5
                    }
                )

                if response.status_code != 200:
                    return []

                data = response.json()

                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]

                    message = update.get("message", {})
                    text = message.get("text", "")

                    if text.startswith("/"):
                        parts = text.split(maxsplit=2)
                        command = parts[0][1:].lower()

                        cmd_data = {"command": command}
                        related_entity_id = None
                        related_entity_type = None

                        if len(parts) > 1:
                            if command in ["approve", "reject"]:
                                cmd_data["trade_id"] = parts[1]
                                related_entity_id = parts[1]
                                related_entity_type = "trade"
                            elif command == "check":
                                cmd_data["ticker"] = parts[1]
                                related_entity_id = parts[1]
                                related_entity_type = "ticker"
                            elif command == "catalyst":
                                cmd_data["description"] = " ".join(parts[1:])
                            elif command == "whatif":
                                cmd_data["scenario"] = " ".join(parts[1:])
                            elif command == "logs":
                                # /logs [N] - get last N messages
                                try:
                                    cmd_data["count"] = int(parts[1])
                                except (ValueError, IndexError):
                                    cmd_data["count"] = 20
                            elif command == "export_logs":
                                try:
                                    cmd_data["count"] = int(parts[1])
                                except (ValueError, IndexError):
                                    cmd_data["count"] = 50

                        # Log incoming command
                        self._log_incoming(
                            content=text,
                            command=command,
                            related_entity_id=related_entity_id,
                            related_entity_type=related_entity_type,
                            metadata=cmd_data
                        )

                        commands.append(cmd_data)

        except Exception as e:
            logger.error(f"Error getting Telegram commands: {e}")

        return commands

    # =========================================================================
    # NEW: LOGS COMMAND HANDLERS
    # =========================================================================

    async def send_logs_summary(self, count: int = 20) -> bool:
        """Send a summary of recent telegram activity."""
        if not self.db:
            return await self.send_message(
                f"{EMOJI_WARNING} Logging not configured - database not connected",
                parse_mode=None,
                message_type="response"
            )

        messages = self.db.get_telegram_messages(limit=count)

        if not messages:
            return await self.send_message(
                f"{EMOJI_SCROLL} No activity logs found",
                parse_mode=None,
                message_type="response"
            )

        lines = [
            f"{EMOJI_EYES} RECENT ACTIVITY ({len(messages)} messages)",
            "=" * 40,
            ""
        ]

        for msg in messages[:15]:  # Show max 15 to avoid message length limits
            ts = msg.get("timestamp_utc", "")[:16].replace("T", " ")
            direction = msg.get("direction", "?")
            msg_type = msg.get("message_type", "?")
            preview = msg.get("content_preview", "")[:60]

            arrow = "\U00002192" if direction == "outgoing" else "\U00002190"  # → or ←

            lines.append(f"{arrow} {ts}")
            lines.append(f"   [{msg_type}] {preview}")
            lines.append("")

        if len(messages) > 15:
            lines.append(f"... and {len(messages) - 15} more")
            lines.append(f"Use /export_logs {count} for full export")

        # Activity summary
        summary = self.db.get_telegram_activity_summary(hours=24)
        lines.extend([
            "",
            "-" * 40,
            f"{EMOJI_CHART} 24-HOUR SUMMARY",
            f"  Incoming: {summary.get('total_incoming', 0)}",
            f"  Outgoing: {summary.get('total_outgoing', 0)}",
        ])

        message = "\n".join(lines)
        return await self.send_message(message, parse_mode=None, message_type="response")

    async def send_logs_export(self, count: int = 50) -> bool:
        """Send a detailed export of recent telegram activity."""
        if not self.db:
            return await self.send_message(
                f"{EMOJI_WARNING} Logging not configured",
                parse_mode=None,
                message_type="response"
            )

        messages = self.db.get_telegram_messages(limit=count)

        if not messages:
            return await self.send_message(
                f"{EMOJI_SCROLL} No activity logs found",
                parse_mode=None,
                message_type="response"
            )

        # Send in chunks to avoid message length limits
        chunk_size = 10
        total_chunks = (len(messages) + chunk_size - 1) // chunk_size

        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i + chunk_size]
            chunk_num = i // chunk_size + 1

            lines = [f"{EMOJI_SCROLL} LOGS ({chunk_num}/{total_chunks})", ""]

            for msg in chunk:
                ts = msg.get("timestamp_utc", "")[:19].replace("T", " ")
                direction = msg.get("direction", "?")[0].upper()
                msg_type = msg.get("message_type", "?")
                command = msg.get("command", "")
                content = msg.get("content", "")[:200]

                lines.append(f"[{ts}] {direction}:{msg_type}")
                if command:
                    lines.append(f"  CMD: /{command}")
                lines.append(f"  {content}")
                lines.append("-" * 30)

            await self.send_message("\n".join(lines), parse_mode=None, message_type="response")

        return True

    # =========================================================================
    # SCAN TRACKING METHODS
    # =========================================================================

    def record_scan_start(self) -> None:
        """Record the start of a scan cycle."""
        self._scan_start_time = datetime.now(timezone.utc)
        self._source_queries = []
        self._signals = []
        self._decisions = []
        self._system_errors = []
        self._risk_rejections = []
        self._trade_blockers = []
        self._technical_signals = []

    def record_source_query(
        self,
        source: str,
        query: str,
        signals_returned: int,
        error: Optional[str] = None
    ) -> None:
        """Record a data source query."""
        self._source_queries.append({
            "source": source,
            "query": query,
            "signals_returned": signals_returned,
            "error": error
        })

    def record_signal(self, signal: Dict[str, Any]) -> None:
        """Record a signal for the summary."""
        self._signals.append(signal)

    def record_technical_signal(self, signal: Dict[str, Any]) -> None:
        """Record a technical analysis signal for chart section."""
        self._technical_signals.append(signal)

    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record a trading decision."""
        self._decisions.append(decision)

    def record_risk_rejection(self, rejection: Dict[str, Any]) -> None:
        """Record a risk engine rejection."""
        self._risk_rejections.append(rejection)

    def record_trade_blocker(self, blocker: Dict[str, Any]) -> None:
        """Record why a trade wasn't created."""
        self._trade_blockers.append(blocker)

    def record_system_error(self, component: str, error: str) -> None:
        """Record a system error."""
        self._system_errors.append({
            "component": component,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _build_conviction_bar(self, conviction: int) -> str:
        """Build a visual conviction bar."""
        filled = int(conviction / 10)
        empty = 10 - filled
        return f"[{BAR_FILLED * filled}{BAR_EMPTY * empty}]"

    def _format_verdict(self, verdict: str) -> str:
        """Format verdict with emoji."""
        verdict_map = {
            "hypothesis": f"{EMOJI_GREEN_CIRCLE} TRADE ALLOWED",
            "hypothesis_allowed": f"{EMOJI_GREEN_CIRCLE} TRADE ALLOWED",
            "no_trade": f"{EMOJI_RED_CIRCLE} NO TRADE",
            "analyze_only": f"{EMOJI_YELLOW_CIRCLE} WATCH ONLY",
            "escalate": f"{EMOJI_WARNING} NEEDS REVIEW",
        }
        return verdict_map.get(verdict.lower(), verdict.upper())

    def _format_market_state(self, state: str, bias: str, confidence: str) -> str:
        """Format market state for display."""
        state_emoji = {
            "trending": EMOJI_CHART_UP if bias == "bullish" else EMOJI_CHART_DOWN,
            "range_bound": EMOJI_RULER,
            "transitional": EMOJI_WARNING,
        }
        emoji = state_emoji.get(state.lower(), EMOJI_CHART)
        return f"{emoji} {state.upper()} ({bias}, {confidence} conf)"

    # =========================================================================
    # SCAN SUMMARY (Updated with logging)
    # =========================================================================

    async def send_scan_summary(
        self,
        signals: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        portfolio: Optional[Dict[str, Any]] = None,
        pending_trade_id: Optional[str] = None,
        technical_signals: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Send a comprehensive scan summary after each scan cycle."""
        now = datetime.now(timezone.utc)
        scan_duration = None
        if self._scan_start_time:
            scan_duration = (now - self._scan_start_time).total_seconds()

        lines = []

        # Header
        lines.append("=" * 40)
        lines.append(f"{EMOJI_SEARCH} SCAN CYCLE COMPLETE")
        lines.append(f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if scan_duration:
            lines.append(f"Duration: {scan_duration:.1f}s")
        lines.append("=" * 40)

        # =====================================================================
        # SOURCES & SIGNAL COUNTS
        # =====================================================================
        lines.append(f"\n{EMOJI_ANTENNA} DATA SOURCES")

        total_signals = 0
        errors = []

        for query in self._source_queries:
            source = query.get("source", "Unknown")
            count = query.get("signals_returned", 0)
            error = query.get("error")
            total_signals += count

            if error:
                lines.append(f"  {EMOJI_CROSS} {source}: ERROR ({error})")
                errors.append(source)
            else:
                lines.append(f"  {EMOJI_CHECK} {source}: {count} signals")

        lines.append(f"\nTotal: {total_signals} signals")

        # =====================================================================
        # TECHNICAL ANALYSIS SECTION
        # =====================================================================
        tech_signals = technical_signals or self._technical_signals
        if tech_signals:
            lines.append("\n" + "-" * 40)
            lines.append(f"{EMOJI_CANDLE} CHART ANALYSIS")

            for tech in tech_signals[:3]:
                ticker = tech.get("ticker", "???")
                ms = tech.get("market_state", {})
                state = ms.get("state", "unknown")
                bias = ms.get("bias", "neutral")
                confidence = ms.get("confidence", "low")
                verdict = tech.get("verdict", "unknown")
                channel = tech.get("trend_channel", {})
                channel_pos = channel.get("position_in_channel", 0.5) if channel else 0.5
                channel_pos_pct = f"{channel_pos:.0%}"
                price = tech.get("current_price", 0)

                state_text = self._format_market_state(state, bias, confidence)
                verdict_text = self._format_verdict(verdict)

                lines.append(f"\n{EMOJI_BULLET} {ticker} @ ${price:.2f}")
                lines.append(f"  State: {state_text}")
                lines.append(f"  Channel: {channel_pos_pct} from bottom")
                lines.append(f"  Verdict: {verdict_text}")

                hypo = tech.get("trade_hypothesis")
                if hypo and hypo.get("allow_trade"):
                    side = hypo.get("side", "").upper()
                    r_mult = hypo.get("expected_r", 0)
                    lines.append(f"  Setup: {side} (R={r_mult:.1f})")

                primary = tech.get("primary_scenario")
                if primary:
                    lines.append(f"  Scenario: {primary.get('name', '')[:40]}")

        # =====================================================================
        # KEY SIGNALS BY CATEGORY
        # =====================================================================
        if signals:
            lines.append("\n" + "-" * 40)
            lines.append(f"{EMOJI_CHART} KEY SIGNALS")

            sentiment_signals = []
            macro_signals = []
            prediction_signals = []

            for sig in signals:
                sig_type = sig.get("signal_type") or sig.get("category") or ""
                source = sig.get("source") or sig.get("source_type") or ""

                if "technical" in sig_type.lower():
                    continue

                if "sentiment" in sig_type.lower() or "grok" in source.lower():
                    sentiment_signals.append(sig)
                elif "macro" in sig_type.lower() or "fred" in source.lower():
                    macro_signals.append(sig)
                elif "prediction" in sig_type.lower() or "polymarket" in source.lower():
                    prediction_signals.append(sig)

            if sentiment_signals:
                lines.append(f"\n{EMOJI_BIRD} Sentiment: {len(sentiment_signals)} signals")
                for sig in sentiment_signals[:2]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")

            if macro_signals:
                lines.append(f"\n{EMOJI_CHART_UP} Macro: {len(macro_signals)} signals")
                for sig in macro_signals[:3]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")

            if prediction_signals:
                lines.append(f"\n{EMOJI_TARGET} Predictions: {len(prediction_signals)} signals")
                sorted_preds = sorted(
                    prediction_signals,
                    key=lambda x: abs(x.get("raw_value", {}).get("change") or 0),
                    reverse=True
                )
                for sig in sorted_preds[:3]:
                    summary = sig.get("summary", "")[:60]
                    lines.append(f"  {EMOJI_BULLET} {summary}")

        # =====================================================================
        # CLAUDE'S ANALYSIS
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_BRAIN} CLAUDE'S ANALYSIS")

        if analysis:
            ticker = analysis.get("ticker")
            recommendation = analysis.get("recommendation", "NONE")
            conviction = analysis.get("conviction_score", 0)
            thesis = analysis.get("thesis", "")
            bull_case = analysis.get("bull_case", "")
            bear_case = analysis.get("bear_case", "")
            time_horizon = analysis.get("time_horizon", "unknown")

            bar = self._build_conviction_bar(conviction)
            is_actionable = conviction >= 80 and recommendation in ["BUY", "SELL"]

            lines.append(f"\nConviction: {conviction}/100")
            if is_actionable:
                lines.append(f"{bar} {EMOJI_GREEN_CIRCLE} ACTIONABLE")
            else:
                lines.append(f"{bar} {EMOJI_WHITE_CIRCLE} Threshold: 80")

            if ticker and recommendation in ["BUY", "SELL"]:
                lines.append(f"\nRecommendation: {recommendation} {ticker}")
                lines.append(f"Time Horizon: {time_horizon}")

                if thesis:
                    lines.append(f"\nThesis: {thesis[:250]}...")

                if bull_case:
                    lines.append(f"\n{EMOJI_BULL} Bull: {bull_case[:150]}...")

                if bear_case:
                    lines.append(f"\n{EMOJI_BEAR} Bear: {bear_case[:150]}...")

                position_size = analysis.get("position_size_pct", 0)
                stop_loss = analysis.get("stop_loss_pct", 0)

                if position_size > 1:
                    position_display = f"{position_size:.0f}%"
                else:
                    position_display = f"{position_size * 100:.0f}%"

                if stop_loss > 1:
                    stop_display = f"{stop_loss:.0f}%"
                else:
                    stop_display = f"{stop_loss * 100:.0f}%"

                lines.append(f"\nTrade Parameters:")
                lines.append(f"  Stop Loss: {stop_display}")
                lines.append(f"  Position Size: {position_display}")
            else:
                lines.append(f"\n{EMOJI_ZZZ} No actionable opportunity")
                if thesis:
                    lines.append(f"\nThesis: {thesis[:200]}...")
                if bull_case:
                    lines.append(f"\n{EMOJI_BULL} Bull: {bull_case[:120]}...")
                if bear_case:
                    lines.append(f"\n{EMOJI_BEAR} Bear: {bear_case[:120]}...")
        else:
            lines.append(f"\n{EMOJI_CROSS} Analysis not generated")

        # =====================================================================
        # TRADE STATUS
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_WARNING} TRADE STATUS")

        if self._risk_rejections:
            for rejection in self._risk_rejections:
                lines.append(f"\n{EMOJI_RED_CIRCLE} REJECTED BY RISK ENGINE")
                lines.append(f"Ticker: {rejection.get('ticker', 'N/A')}")
                lines.append(f"Reason: {rejection.get('reason', rejection.get('message', 'Unknown'))}")
            lines.append(f"\n{EMOJI_BULLET} Trade blocked - no approval needed")

        elif self._trade_blockers:
            lines.append(f"\n{EMOJI_RED_CIRCLE} TRADE NOT CREATED")
            for blocker in self._trade_blockers:
                lines.append(f"{EMOJI_BULLET} {blocker['type']}: {blocker['details']}")

        elif pending_trade_id:
            lines.append(f"\n{EMOJI_BELL} TRADE PENDING APPROVAL")
            lines.append(f"Trade ID: {pending_trade_id}")
            lines.append(f"\n{EMOJI_CHECK} /approve {pending_trade_id}")
            lines.append(f"{EMOJI_CROSS} /reject {pending_trade_id}")

        elif analysis and analysis.get("conviction_score", 0) >= 80:
            lines.append(f"\n{EMOJI_WHITE_CIRCLE} Conviction met, no trade created")
            lines.append("(Unknown issue - check logs)")

        else:
            lines.append(f"\n{EMOJI_WHITE_CIRCLE} No trade - conviction below 80")

        # =====================================================================
        # PORTFOLIO SNAPSHOT
        # =====================================================================
        if portfolio:
            lines.append("\n" + "-" * 40)
            lines.append(f"{EMOJI_MONEY} PORTFOLIO")

            equity = portfolio.get("equity") or portfolio.get("total_value", 0)
            cash = portfolio.get("cash", 0)
            position_count = portfolio.get("position_count", 0)

            lines.append(f"  Equity: ${equity:,.2f}")
            lines.append(f"  Cash: ${cash:,.2f}")
            lines.append(f"  Positions: {position_count}")

        # =====================================================================
        # COMMANDS
        # =====================================================================
        lines.append("\n" + "-" * 40)
        lines.append(f"{EMOJI_KEYBOARD} COMMANDS")
        lines.append("/status - System status")
        lines.append("/pending - Pending trades")
        lines.append("/check [TICKER] - Analyze any stock")
        lines.append("/logs - View activity logs")
        lines.append("/approve [id] - Approve trade")
        lines.append("/reject [id] - Reject trade")

        # Footer
        lines.append("\n" + "=" * 40)
        lines.append("Next scan: ~60 minutes")

        message = "\n".join(lines)

        if len(message) > 4000:
            message = message[:3950] + "\n\n[Truncated]"

        return await self.send_message(
            message,
            parse_mode=None,
            message_type="scan_summary",
            related_entity_id=pending_trade_id,
            related_entity_type="trade" if pending_trade_id else None
        )

    # =========================================================================
    # CHECK RESULT (with Technical Analysis)
    # =========================================================================

    async def send_check_result(self, result: Dict[str, Any]) -> bool:
        """Send on-demand /check analysis result with technical context."""
        ticker = result.get("ticker", "???")
        conviction = result.get("conviction", 0)
        is_tradeable = result.get("is_tradeable", False)
        current_price = result.get("current_price")
        recommendation = result.get("recommendation", "NONE")
        thesis = result.get("thesis", "No analysis available")
        pending_trade_id = result.get("pending_trade_id")
        risk_rejection = result.get("risk_rejection")
        historical_context = result.get("historical_context")
        bull_case = result.get("bull_case")
        bear_case = result.get("bear_case")
        technical = result.get("technical_analysis")

        bar = self._build_conviction_bar(conviction)

        if is_tradeable and current_price:
            status = f"Tradeable @ ${current_price:.2f}"
        elif is_tradeable:
            status = "Tradeable (price unavailable)"
        else:
            status = f"Not Tradeable (pre-IPO or unlisted)"

        if recommendation == "BUY":
            rec_emoji = EMOJI_BULL
            rec_text = "BUY"
        elif recommendation == "SELL":
            rec_emoji = EMOJI_BEAR
            rec_text = "SELL"
        else:
            rec_emoji = EMOJI_WHITE_CIRCLE
            rec_text = "HOLD/WATCH"

        lines = [
            f"{EMOJI_TARGET} ANALYSIS: {ticker}",
            "=" * 35,
            "",
            f"Status: {status}",
            f"Signals: {result.get('signals_count', 0)} gathered",
            "",
            f"{EMOJI_CHART} CONVICTION: {conviction}/100",
            bar,
            "",
            f"{rec_emoji} RECOMMENDATION: {rec_text}",
        ]

        # Technical Analysis Section
        if technical:
            lines.extend(["", "-" * 35, f"{EMOJI_CANDLE} CHART STRUCTURE"])

            ms = technical.get("market_state", {})
            state = ms.get("state", "unknown")
            bias = ms.get("bias", "neutral")
            confidence = ms.get("confidence", "low")
            lines.append(f"Market State: {state.upper()} ({bias})")
            lines.append(f"Confidence: {confidence}")

            evidence = ms.get("evidence", [])
            if evidence:
                lines.append("Evidence:")
                for e in evidence[:3]:
                    lines.append(f"  {EMOJI_BULLET} {e[:50]}")

            channel = technical.get("trend_channel")
            if channel:
                upper = channel.get("channel_upper", 0)
                lower = channel.get("channel_lower", 0)
                pos = channel.get("position_in_channel", 0.5)

                lines.append("")
                lines.append(f"{EMOJI_RULER} TREND CHANNEL")
                lines.append(f"  Upper: ${upper:.2f}")
                lines.append(f"  Lower: ${lower:.2f}")
                lines.append(f"  Position: {pos:.0%} from bottom")

                if pos < 0.3:
                    pos_text = f"{EMOJI_GREEN_CIRCLE} Near SUPPORT (buy zone)"
                elif pos > 0.7:
                    pos_text = f"{EMOJI_RED_CIRCLE} Near RESISTANCE (sell zone)"
                else:
                    pos_text = f"{EMOJI_YELLOW_CIRCLE} Middle of channel"
                lines.append(f"  {pos_text}")

            support = technical.get("support_levels", [])
            resistance = technical.get("resistance_levels", [])
            if support or resistance:
                lines.append("")
                if support:
                    lines.append(f"Support: ${support[0]:.2f}" + (f", ${support[1]:.2f}" if len(support) > 1 else ""))
                if resistance:
                    lines.append(f"Resistance: ${resistance[0]:.2f}" + (f", ${resistance[1]:.2f}" if len(resistance) > 1 else ""))

            verdict = technical.get("verdict", "unknown")
            lines.append("")
            lines.append(f"Verdict: {self._format_verdict(verdict)}")

            hypo = technical.get("trade_hypothesis")
            if hypo and hypo.get("allow_trade"):
                side = hypo.get("side", "").upper()
                entry = hypo.get("entry_zone", {})
                stop = hypo.get("invalidation", {})
                targets = hypo.get("targets", [])
                r_mult = hypo.get("expected_r", 0)

                lines.append("")
                lines.append(f"{EMOJI_TARGET} TRADE SETUP: {side}")
                lines.append(f"  Entry: ${entry.get('low', 0):.2f} - ${entry.get('high', 0):.2f}")
                lines.append(f"  Stop: ${stop.get('level', 0):.2f}")
                if targets:
                    lines.append(f"  Target: ${targets[0].get('level', 0):.2f}")
                lines.append(f"  R-Multiple: {r_mult:.1f}")

            primary = technical.get("primary_scenario")
            if primary:
                lines.append("")
                lines.append(f"Primary: {primary.get('name', '')}")
                lines.append(f"  Prob: {primary.get('probability', 0):.0%}")
                confirm = primary.get("confirm", [])
                if confirm:
                    lines.append(f"  Confirm: {confirm[0][:40]}")

        # Thesis
        lines.extend([
            "",
            "-" * 35,
            f"{EMOJI_BRAIN} THESIS:",
            thesis[:400],
        ])

        if bull_case:
            lines.extend(["", f"{EMOJI_BULL} Bull: {bull_case[:150]}"])
        if bear_case:
            lines.extend(["", f"{EMOJI_BEAR} Bear: {bear_case[:150]}"])

        if historical_context:
            lines.extend([
                "",
                f"{EMOJI_SEARCH} HISTORICAL PATTERN:",
                str(historical_context)[:200],
            ])

        # Trade Action
        lines.append("")
        lines.append("=" * 35)

        if pending_trade_id:
            lines.extend([
                f"{EMOJI_BELL} Trade Created - Pending Approval",
                "",
                f"/approve {pending_trade_id}",
                f"/reject {pending_trade_id}",
            ])
        elif risk_rejection:
            lines.extend([
                f"{EMOJI_WARNING} Risk Check Failed:",
                risk_rejection[:100],
            ])
        elif not is_tradeable:
            lines.extend([
                f"{EMOJI_SEARCH} WATCH LIST",
                "Cannot trade (pre-IPO or unlisted)",
                "Monitor for when it becomes available",
            ])
        elif conviction >= 80 and recommendation == "BUY":
            lines.extend([
                f"{EMOJI_WHITE_CIRCLE} Trade criteria met but not created",
                "(Check logs for details)",
            ])
        else:
            lines.extend([
                f"{EMOJI_ZZZ} No trade action",
                f"Conviction {conviction} below 80 threshold" if conviction < 80 else "Recommendation is not BUY",
            ])

        message = "\n".join(lines)

        if len(message) > 4000:
            message = message[:3950] + "\n\n[Truncated]"

        return await self.send_message(
            message,
            parse_mode=None,
            message_type="response",
            related_entity_id=ticker,
            related_entity_type="ticker"
        )

    # =========================================================================
    # OTHER NOTIFICATION METHODS (Updated with logging)
    # =========================================================================

    async def send_startup_message(self) -> bool:
        """Send notification when agent starts."""
        now = datetime.now(timezone.utc)
        message = (
            f"{EMOJI_ROCKET} GANN SENTINEL STARTED\n\n"
            f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Mode: PAPER\n"
            f"Version: 2.1.0 (MACA Preview)\n"
            f"Approval Gate: ON\n\n"
            f"/status - Check status\n"
            f"/logs - View activity\n"
            f"/help - Show commands"
        )
        return await self.send_message(message, parse_mode=None, message_type="notification")

    async def send_error_alert(self, component: str, error: str) -> bool:
        """Send error notification."""
        message = f"{EMOJI_WARNING} ERROR: {component}\n\n{error[:500]}"
        return await self.send_message(message, parse_mode=None, message_type="error")

    async def send_trade_approval_request(
        self,
        trade: Dict[str, Any],
        analysis: Optional[Dict[str, Any]],
        current_price: Optional[float] = None
    ) -> bool:
        """Send trade approval request."""
        trade_id = trade.get("id", "")[:8]
        ticker = trade.get("ticker", "???")
        side = trade.get("side", "BUY").upper()
        quantity = trade.get("quantity", 0)
        conviction = trade.get("conviction_score", 0)
        thesis = trade.get("thesis", "")

        if trade_id not in self._pending_approvals:
            self._pending_approvals.append(trade_id)

        bar = self._build_conviction_bar(conviction)

        lines = [
            f"{EMOJI_BELL} TRADE PENDING APPROVAL",
            "",
            f"Ticker: {ticker}",
            f"Action: {side}",
            f"Quantity: {quantity} shares",
        ]

        if current_price:
            lines.append(f"Price: ${current_price:.2f}")

        lines.extend([
            f"Conviction: {conviction}/100",
            bar,
            "",
            f"Thesis: {thesis[:300]}...",
            "",
            f"/approve {trade_id}",
            f"/reject {trade_id}",
        ])

        message = "\n".join(lines)
        return await self.send_message(
            message,
            parse_mode=None,
            message_type="approval_request",
            related_entity_id=trade_id,
            related_entity_type="trade"
        )

    async def send_status_message(
        self,
        portfolio: Dict[str, Any],
        positions: List[Dict[str, Any]],
        pending_approvals: List[Dict[str, Any]],
        agent_running: bool
    ) -> bool:
        """Send status message."""
        status = "RUNNING" if agent_running else "STOPPED"

        lines = [
            f"{EMOJI_CHART} SYSTEM STATUS",
            "=" * 30,
            "",
            f"Status: {status}",
            f"Mode: PAPER",
            f"Version: 2.1.0",
            f"Approval Gate: ON",
        ]

        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)

        lines.extend([
            "",
            f"{EMOJI_MONEY} PORTFOLIO",
            f"  Equity: ${equity:,.2f}",
            f"  Cash: ${cash:,.2f}",
            f"  Daily P&L: ${daily_pnl:,.2f}",
        ])

        if positions:
            lines.append(f"\n{EMOJI_CHART_UP} POSITIONS ({len(positions)})")
            for pos in positions[:5]:
                ticker = pos.get("ticker", "???")
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                emoji = EMOJI_GREEN_CIRCLE if pnl >= 0 else EMOJI_RED_CIRCLE
                lines.append(f"  {emoji} {ticker}: ${pnl:,.2f} ({pnl_pct:+.1f}%)")

        if pending_approvals:
            lines.append(f"\n{EMOJI_HOURGLASS} PENDING ({len(pending_approvals)})")
            for trade in pending_approvals[:3]:
                tid = trade.get("id", "")[:8]
                ticker = trade.get("ticker", "???")
                lines.append(f"  {EMOJI_BULLET} {ticker} - /approve {tid}")

        # Activity summary if db is connected
        if self.db:
            summary = self.db.get_telegram_activity_summary(hours=24)
            lines.extend([
                "",
                f"{EMOJI_EYES} 24H ACTIVITY",
                f"  Commands: {summary.get('total_incoming', 0)}",
                f"  Messages: {summary.get('total_outgoing', 0)}",
            ])

        message = "\n".join(lines)
        return await self.send_message(message, parse_mode=None, message_type="status")

    async def send_daily_digest(
        self,
        positions: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        pending_approvals: List[Dict[str, Any]]
    ) -> bool:
        """Send the daily digest summary."""
        now = datetime.now(timezone.utc)

        lines = [
            "=" * 40,
            f"{EMOJI_CHART} DAILY DIGEST",
            f"{now.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 40
        ]

        lines.append(f"\n{EMOJI_MONEY} PORTFOLIO")
        total_value = portfolio.get("total_value") or portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pnl = portfolio.get("daily_pnl", 0)
        daily_pnl_pct = portfolio.get("daily_pnl_pct", 0)

        pnl_emoji = EMOJI_GREEN_CIRCLE if daily_pnl >= 0 else EMOJI_RED_CIRCLE
        lines.append(f"Total: ${total_value:,.2f}")
        lines.append(f"Cash: ${cash:,.2f}")
        lines.append(f"P&L: {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")

        if positions:
            lines.append(f"\n{EMOJI_CHART_UP} POSITIONS ({len(positions)})")
            for pos in positions[:5]:
                ticker = pos.get("ticker", "N/A")
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                pos_emoji = EMOJI_GREEN_CIRCLE if pnl >= 0 else EMOJI_RED_CIRCLE
                lines.append(f"  {EMOJI_BULLET} {ticker}: {pos_emoji} ${pnl:,.2f} ({pnl_pct:+.1f}%)")
        else:
            lines.append(f"\n{EMOJI_CHART_UP} POSITIONS: None")

        if pending_approvals:
            lines.append(f"\n{EMOJI_HOURGLASS} PENDING ({len(pending_approvals)})")
            for trade in pending_approvals[:3]:
                ticker = trade.get("ticker", "N/A")
                side = trade.get("side", "N/A").upper()
                trade_id = trade.get("id", "")[:8]
                lines.append(f"  {EMOJI_BULLET} {side} {ticker} - /approve {trade_id}")

        lines.append(f"\n{EMOJI_KEYBOARD} /status /pending /check [TICKER] /logs /help")
        lines.append("=" * 40)

        message = "\n".join(lines)

        # Reset tracking state
        self._source_queries = []
        self._signals = []
        self._decisions = []
        self._system_errors = []
        self._technical_signals = []

        return await self.send_message(message, parse_mode=None, message_type="daily_digest")


async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
