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
EMOJI_ROCKET = "\U0001F680"
EMOJI_STOP = "\U0001F6D1"
EMOJI_CHART = "\U0001F4CA"
EMOJI_MONEY = "\U0001F4B0"
EMOJI_CHART_UP = "\U0001F4C8"
EMOJI_CHART_DOWN = "\U0001F4C9"
EMOJI_WARNING = "\U000026A0"
EMOJI_CHECK = "\U00002705"
EMOJI_CROSS = "\U0000274C"
EMOJI_BELL = "\U0001F514"
EMOJI_BRAIN = "\U0001F9E0"
EMOJI_SEARCH = "\U0001F50D"
EMOJI_TARGET = "\U0001F3AF"
EMOJI_HOURGLASS = "\U000023F3"
EMOJI_GREEN_CIRCLE = "\U0001F7E2"
EMOJI_RED_CIRCLE = "\U0001F534"
EMOJI_YELLOW_CIRCLE = "\U0001F7E1"
EMOJI_WHITE_CIRCLE = "\U000026AA"
EMOJI_BIRD = "\U0001F426"
EMOJI_ANTENNA = "\U0001F4E1"
EMOJI_MEMO = "\U0001F4CB"
EMOJI_BULLET = "\U00002022"
EMOJI_KEYBOARD = "\U00002328"
EMOJI_BEAR = "\U0001F43B"
EMOJI_BULL = "\U0001F402"
EMOJI_ZZZ = "\U0001F4A4"
EMOJI_CANDLE = "\U0001F56F"
EMOJI_RULER = "\U0001F4CF"
EMOJI_SCROLL = "\U0001F4DC"
EMOJI_EYES = "\U0001F440"

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
        """Poll for new commands and callback queries."""
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

                    # Handle callback_query (inline button press)
                    if "callback_query" in update:
                        cb = update["callback_query"]
                        callback_data = cb.get("data", "")
                        callback_id = cb.get("id")

                        action, param = self.parse_callback_data(callback_data)

                        cmd_data = {
                            "type": "callback_query",
                            "callback_id": callback_id
                        }

                        if action == "approve":
                            cmd_data["command"] = "approve"
                            cmd_data["trade_id"] = param
                        elif action == "reject":
                            cmd_data["command"] = "reject"
                            cmd_data["trade_id"] = param
                        elif action == "cmd":
                            cmd_data["command"] = param  # status, pending, scan, help
                        else:
                            cmd_data["command"] = action

                        # Log incoming callback
                        self._log_incoming(
                            content=f"Button: {callback_data}",
                            command=cmd_data.get("command", "unknown"),
                            related_entity_id=param,
                            metadata=cmd_data
                        )

                        commands.append(cmd_data)
                        continue

                    # Handle regular message
                    message = update.get("message", {})
                    text = message.get("text", "")

                    if text.startswith("/"):
                        parts = text.split(maxsplit=2)
                        command = parts[0][1:].lower()

                        cmd_data = {"command": command, "type": "message"}
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
                            elif command == "history":
                                # /history [N] - get last N trades
                                try:
                                    cmd_data["count"] = int(parts[1])
                                except (ValueError, IndexError):
                                    cmd_data["count"] = 10
                            elif command == "export":
                                # /export [csv/parquet]
                                cmd_data["format"] = parts[1].lower()
                            elif command == "cost":
                                # /cost [days] - default 7
                                try:
                                    cmd_data["days"] = int(parts[1])
                                except (ValueError, IndexError):
                                    cmd_data["days"] = 7
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

            arrow = "\U00002192" if direction == "outgoing" else "\U00002190"

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
            f"Version: 2.2.0\n"
            f"Approval Gate: ON\n\n"
            f"{EMOJI_BRAIN} Learning Engine: ON\n"
            f"{EMOJI_CHART} Smart Schedule:\n"
            f"  Morning: 9:35 AM ET\n"
            f"  Midday: 12:30 PM ET\n"
            f"  Weekends: OFF\n\n"
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

    # =========================================================================
    # MACA TELEGRAM METHODS - Inline Buttons & AI Council Display
    # =========================================================================

    def build_approval_keyboard(self, trade_id: str) -> Dict[str, Any]:
        """
        Build inline keyboard with Approve/Reject buttons.

        Returns Telegram InlineKeyboardMarkup structure.
        """
        return {
            "inline_keyboard": [
                [
                    {"text": f"{EMOJI_CHECK} APPROVE", "callback_data": f"approve_{trade_id}"},
                    {"text": f"{EMOJI_CROSS} REJECT", "callback_data": f"reject_{trade_id}"}
                ],
                [
                    {"text": f"{EMOJI_CHART} Status", "callback_data": "cmd_status"},
                    {"text": f"{EMOJI_HOURGLASS} Pending", "callback_data": "cmd_pending"},
                    {"text": f"{EMOJI_SEARCH} Scan", "callback_data": "cmd_scan"},
                    {"text": f"{EMOJI_MEMO} Help", "callback_data": "cmd_help"}
                ]
            ]
        }

    def build_command_keyboard(self) -> Dict[str, Any]:
        """Build inline keyboard with quick command buttons."""
        return {
            "inline_keyboard": [
                [
                    {"text": f"{EMOJI_CHART} Status", "callback_data": "cmd_status"},
                    {"text": f"{EMOJI_HOURGLASS} Pending", "callback_data": "cmd_pending"},
                    {"text": f"{EMOJI_SEARCH} Scan", "callback_data": "cmd_scan"}
                ],
                [
                    {"text": f"{EMOJI_SCROLL} Logs", "callback_data": "cmd_logs"},
                    {"text": f"{EMOJI_MEMO} Help", "callback_data": "cmd_help"}
                ]
            ]
        }

    def parse_callback_data(self, callback_data: str) -> tuple:
        """
        Parse callback data from inline button press.

        Returns:
            Tuple of (action, parameter)
            e.g., "approve_abc123" -> ("approve", "abc123")
                  "cmd_status" -> ("cmd", "status")
        """
        if "_" in callback_data:
            parts = callback_data.split("_", 1)
            return parts[0], parts[1]
        return callback_data, None

    def parse_update(self, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a Telegram update, handling both messages and callback queries.

        Returns:
            Dict with 'type' and relevant data, or None if not parseable
        """
        # Handle callback_query (inline button press)
        if "callback_query" in update:
            cb = update["callback_query"]
            return {
                "type": "callback_query",
                "id": cb.get("id"),
                "data": cb.get("data"),
                "chat_id": cb.get("message", {}).get("chat", {}).get("id"),
                "message_id": cb.get("message", {}).get("message_id")
            }

        # Handle regular message
        if "message" in update:
            msg = update["message"]
            text = msg.get("text", "")
            return {
                "type": "message",
                "text": text,
                "chat_id": msg.get("chat", {}).get("id"),
                "message_id": msg.get("message_id")
            }

        return None

    def format_ai_proposal(self, proposal: Dict[str, Any]) -> str:
        """
        Format a single AI proposal section for Telegram.

        Args:
            proposal: Thesis proposal from Grok/Perplexity/ChatGPT

        Returns:
            Formatted string for display
        """
        source = proposal.get("ai_source", "unknown").upper()
        proposal_type = proposal.get("proposal_type", "")
        rec = proposal.get("recommendation", {})
        evidence = proposal.get("supporting_evidence", {})

        ticker = rec.get("ticker")
        side = rec.get("side", "")
        conviction = rec.get("conviction_score", 0)
        thesis = rec.get("thesis", "")
        thesis_desc = rec.get("thesis_description", "")
        catalyst = rec.get("catalyst", "")
        time_horizon = rec.get("time_horizon", "")

        # AI source emoji mapping
        emoji_map = {
            "GROK": EMOJI_BIRD,           # Social sentiment
            "PERPLEXITY": EMOJI_TARGET,   # Fundamental research
            "CHATGPT": EMOJI_BRAIN,       # Pattern recognition
        }
        emoji = emoji_map.get(source, EMOJI_CHART)

        # Signal inventory (preferred)
        sig_inv = proposal.get("signal_inventory", {})
        sig_total = sig_inv.get("total_signals")
        if sig_total is None:
            # Backwards compat: some proposals store signals_count under supporting_evidence
            sig_total = proposal.get("supporting_evidence", {}).get("signals_count")

        # Build conviction bar
        bar = self._build_conviction_bar(conviction)
        is_actionable = conviction >= 80

        lines = []
        lines.append(f"{emoji} {source}")
        lines.append("-" * 30)

        key_signals = proposal.get("supporting_evidence", {}).get("key_signals", [])
        if proposal_type == "NO_OPPORTUNITY" or not ticker:
            lines.append("Recommendation: HOLD")
            if sig_total is not None:
                lines.append(f"Signals received: {sig_total}")
            lines.append(f"Conviction: {conviction}/100")
            lines.append(bar)
            considered = proposal.get("signals_considered", []) or []
            if considered:
                lines.append("Key signals:")
                for sc in considered[:2]:
                    src = sc.get("source") or "unknown"
                    summary = sc.get("summary") or ""
                    lines.append(f"  - [{src}] {summary[:120]}")
            elif key_signals:
                lines.append("Key signals:")
                for ks in key_signals[:2]:
                    s = ks.get("summary") or ""
                    src = ks.get("source") or ks.get("signal_type") or ""
                    lines.append(f"  - [{src}] {s[:120]}")
            else:
                lines.append("Key signals: none provided")
            # Show thesis - use thesis_desc if longer and different, otherwise thesis
            if thesis_desc and thesis and thesis_desc.strip() != thesis.strip() and len(thesis_desc) > len(thesis):
                lines.append(f"\nThesis: {thesis_desc[:300]}")
            elif thesis:
                lines.append(f"\nThesis: {thesis[:300]}")
        else:
            lines.append(f"Recommendation: {side}")
            lines.append(f"Ticker: {ticker}")
            if sig_total is not None:
                lines.append(f"Signals received: {sig_total}")
            lines.append(f"Conviction: {conviction}/100")
            if is_actionable:
                lines.append(f"{bar} {EMOJI_GREEN_CIRCLE}")
            else:
                lines.append(bar)

            # Key signals (top 3) - show first for context
            if key_signals:
                lines.append("Key signals:")
                for ks in key_signals[:3]:
                    s = ks.get("summary") or ""
                    src = ks.get("source") or ks.get("signal_type") or ""
                    lines.append(f"  - [{src}] {s[:120]}")
            else:
                # Try signals_considered as fallback
                considered = proposal.get("signals_considered", []) or []
                if considered:
                    lines.append("Key signals:")
                    for sc in considered[:3]:
                        src = sc.get("source") or "unknown"
                        summary = sc.get("summary") or ""
                        lines.append(f"  - [{src}] {summary[:120]}")
                else:
                    lines.append("Key signals: none provided")

            # Show thesis - use thesis_desc if longer and different, otherwise thesis
            if thesis_desc and thesis and thesis_desc.strip() != thesis.strip() and len(thesis_desc) > len(thesis):
                lines.append(f"\nThesis: {thesis_desc[:300]}")
            elif thesis:
                lines.append(f"\nThesis: {thesis[:300]}")

            if catalyst:
                lines.append(f"Catalyst: {catalyst}")

            if time_horizon:
                lines.append(f"Horizon: {time_horizon}")

        return "\n".join(lines)

    def format_ai_council_message(
        self,
        proposals: List[Dict[str, Any]],
        signal_inventory: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format Message 1: AI Council Views.

        Shows signal inventory and thesis proposals from all AI analysts.

        Args:
            proposals: List of analyst proposals
            signal_inventory: Dict with by_source counts (FRED, Polymarket, Events, Technical)
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        lines = []
        lines.append("=" * 40)
        lines.append(f"{EMOJI_SEARCH} MACA SCAN - AI COUNCIL")
        lines.append(f"{now.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 40)

        # Signal Inventory header with actual signals
        if signal_inventory:
            by_source = signal_inventory.get("by_source", {})
            total = signal_inventory.get("total", 0)
            fred_count = by_source.get("FRED", 0)
            poly_count = by_source.get("Polymarket", 0)
            event_count = by_source.get("Events", 0)
            tech_count = by_source.get("Technical", 0)

            lines.append("")
            lines.append(f"{EMOJI_CHART} Signals Collected: {total}")

            # FRED Signals
            lines.append(f"\n📈 FRED ({fred_count}):")
            fred_sigs = signal_inventory.get("fred_signals", [])
            if fred_sigs:
                for fs in fred_sigs[:3]:
                    summary = fs.get("summary", "")[:100]
                    lines.append(f"  • {summary}")
            else:
                lines.append("  • None collected")

            # Polymarket Signals
            lines.append(f"\n🎲 Polymarket ({poly_count}):")
            poly_sigs = signal_inventory.get("polymarket_signals", [])
            if poly_sigs:
                for ps in poly_sigs[:3]:
                    summary = ps.get("summary", "")[:100]
                    lines.append(f"  • {summary}")
            else:
                lines.append("  • None collected")

            # Event Signals
            if event_count > 0:
                lines.append(f"\n📅 Events ({event_count}):")
                event_sigs = signal_inventory.get("event_signals", [])
                if event_sigs:
                    for es in event_sigs[:2]:
                        summary = es.get("summary", "")[:100]
                        lines.append(f"  • {summary}")

            # Technical
            if tech_count > 0:
                lines.append(f"\n📊 Technical: {tech_count} chart(s) analyzed")

            lines.append("")
            lines.append("-" * 40)

        # Sort proposals by source for consistent ordering
        source_order = {"grok": 0, "perplexity": 1, "chatgpt": 2}
        sorted_proposals = sorted(
            proposals,
            key=lambda p: source_order.get(p.get("ai_source", "").lower(), 99)
        )

        for proposal in sorted_proposals:
            lines.append("")
            lines.append(self.format_ai_proposal(proposal))

        lines.append("")
        lines.append("=" * 40)
        lines.append(f"{EMOJI_BRAIN} Claude's synthesis follows...")

        return "\n".join(lines)

    def format_decision_message(
        self,
        synthesis: Dict[str, Any],
        technical_signals: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        trade_id: Optional[str] = None
    ) -> str:
        """
        Format Message 2: Claude's Decision + Technical Analysis.

        Shows chart analysis, Claude's synthesis, and trade status.
        """
        lines = []

        # =====================================================================
        # TECHNICAL ANALYSIS SECTION
        # =====================================================================
        if technical_signals:
            lines.append("=" * 40)
            lines.append(f"{EMOJI_CANDLE} CHART ANALYSIS")
            lines.append("-" * 40)

            for tech in technical_signals[:3]:
                ticker = tech.get("ticker", "???")
                price = tech.get("current_price", 0)
                ms = tech.get("market_state", {})
                state = ms.get("state", "unknown")
                bias = ms.get("bias", "neutral")
                confidence = ms.get("confidence", "low")
                verdict = tech.get("verdict", "unknown")
                channel = tech.get("trend_channel", {})
                channel_pos = channel.get("position_in_channel", 0.5) if channel else 0.5

                # Historical context if available
                hist = tech.get("historical_context", {})
                pct_from_ath = hist.get("pct_from_ath")
                pct_52w = hist.get("pct_52w_change")

                state_text = self._format_market_state(state, bias, confidence)
                verdict_text = self._format_verdict(verdict)

                lines.append(f"\n{EMOJI_BULLET} {ticker} @ ${price:,.2f}")
                lines.append(f"  State: {state_text}")
                lines.append(f"  Channel: {channel_pos:.0%} from bottom")
                lines.append(f"  Verdict: {verdict_text}")

                # Add historical context if available
                if pct_from_ath is not None or pct_52w is not None:
                    context_parts = []
                    if pct_from_ath is not None:
                        context_parts.append(f"{pct_from_ath:+.1f}% from ATH")
                    if pct_52w is not None:
                        context_parts.append(f"{pct_52w:+.1f}% 52wk")
                    lines.append(f"  5yr: {', '.join(context_parts)}")

                # Trade setup if allowed
                hypo = tech.get("trade_hypothesis", {})
                if hypo.get("allow_trade"):
                    side = hypo.get("side", "").upper()
                    r_mult = hypo.get("expected_r", 0)
                    lines.append(f"  Setup: {side} (R={r_mult:.1f})")

        # =====================================================================
        # CLAUDE'S SYNTHESIS
        # =====================================================================
        lines.append("")
        lines.append("=" * 40)
        lines.append(f"{EMOJI_BRAIN} CLAUDE'S SYNTHESIS (Senior Trader)")
        lines.append("=" * 40)

        decision_type = synthesis.get("decision_type", "NO_TRADE")
        rec = synthesis.get("recommendation", {})
        selected = synthesis.get("selected_proposal", {})
        cross_val = synthesis.get("cross_validation", {})
        rationale = synthesis.get("rationale", "")

        ticker = rec.get("ticker")
        side = rec.get("side", "")
        conviction = rec.get("conviction_score", 0)
        thesis = rec.get("thesis", "")
        final_thesis = synthesis.get("final_thesis", {}) or {}
        ctx_explain = synthesis.get("context_explainers", {}) or {}
        position_size = rec.get("position_size_pct", 0)
        stop_loss = rec.get("stop_loss_pct", 0)
        time_horizon = rec.get("time_horizon", "")

        bar = self._build_conviction_bar(conviction)
        is_actionable = decision_type == "TRADE" and conviction >= 80

        lines.append(f"\nDecision: {decision_type}")

        if selected.get("ai_source"):
            lines.append(f"Selected: {selected['ai_source'].upper()} proposal")

        if ticker and side:
            lines.append(f"\nRecommendation: {side} {ticker}")

        lines.append(f"Conviction: {conviction}/100")
        if is_actionable:
            lines.append(f"{bar} {EMOJI_GREEN_CIRCLE} ACTIONABLE")
        else:
            lines.append(f"{bar} {EMOJI_WHITE_CIRCLE}")

        # Final thesis (preferred)
        if final_thesis.get("summary"):
            lines.append(f"\nFinal thesis: {final_thesis.get('summary')[:260]}")
            desc = final_thesis.get("description") or ""
            if desc:
                lines.append(f"\nWhy now (detail): {desc[:600]}...")
            inv = final_thesis.get("invalidation")
            if inv:
                lines.append(f"\nInvalidation: {inv[:220]}")
        elif thesis:
            lines.append(f"\nThesis: {thesis[:250]}...")

        # Explain the cross-reference sources in plain English
        fred_ex = ctx_explain.get("fred")
        poly_ex = ctx_explain.get("polymarket")
        if fred_ex or poly_ex:
            lines.append("\nContext: why these sources matter")
            if fred_ex:
                lines.append(f"  FRED: {fred_ex[:260]}")
            if poly_ex:
                lines.append(f"  Polymarket: {poly_ex[:260]}")

        # Cross-validation
        if cross_val:
            lines.append(f"\nCross-Validation:")
            fred = cross_val.get("fred_alignment", "N/A")
            poly = cross_val.get("polymarket_alignment", "N/A")
            tech = cross_val.get("technical_alignment", "N/A")
            lines.append(f"  FRED: {fred}")
            lines.append(f"  Polymarket: {poly}")
            lines.append(f"  Technical: {tech}")

        # Trade parameters
        if is_actionable and (position_size or stop_loss):
            lines.append(f"\nTrade Parameters:")
            if stop_loss:
                lines.append(f"  Stop Loss: {stop_loss}%")
            if position_size:
                lines.append(f"  Position Size: {position_size}%")
            if time_horizon:
                lines.append(f"  Horizon: {time_horizon}")

        # =====================================================================
        # TRADE STATUS
        # =====================================================================
        lines.append("")
        lines.append("-" * 40)

        if trade_id and is_actionable:
            lines.append(f"{EMOJI_BELL} TRADE PENDING APPROVAL")
            lines.append(f"Trade ID: {trade_id}")
            if not self._risk_rejections:
                lines.append(f"{EMOJI_CHECK} Risk Engine: PASS")
            lines.append("")
            lines.append("Use buttons below to approve or reject")
        elif self._risk_rejections:
            # Show risk rejections if any
            lines.append(f"{EMOJI_RED_CIRCLE} BLOCKED BY RISK ENGINE")
            for rejection in self._risk_rejections[:2]:
                check = rejection.get("check_name", "Unknown")
                msg = rejection.get("message", rejection.get("reason", ""))
                lines.append(f"  {check}: {msg[:80]}")
        elif self._trade_blockers:
            # Show trade blockers if any
            lines.append(f"{EMOJI_RED_CIRCLE} TRADE NOT CREATED")
            for blocker in self._trade_blockers[:3]:
                lines.append(f"  {blocker.get('type', 'ERROR')}: {blocker.get('details', '')[:80]}")
        elif is_actionable:
            # High conviction but no trade created - unknown reason
            lines.append(f"{EMOJI_WARNING} ACTIONABLE BUT NO TRADE")
            lines.append("Check logs - possible quote/risk issue")
        elif decision_type == "NO_TRADE":
            lines.append(f"{EMOJI_ZZZ} NO TRADE")
            if rationale:
                lines.append(f"Reason: {rationale[:150]}")
        else:
            lines.append(f"{EMOJI_WHITE_CIRCLE} Watching - conviction below threshold")

        # =====================================================================
        # PORTFOLIO SNAPSHOT
        # =====================================================================
        if portfolio:
            lines.append("")
            lines.append("-" * 40)
            lines.append(f"{EMOJI_MONEY} PORTFOLIO")

            equity = portfolio.get("equity", 0)
            cash = portfolio.get("cash", 0)
            position_count = portfolio.get("position_count", 0)

            lines.append(f"  Equity: ${equity:,.2f}")
            lines.append(f"  Cash: ${cash:,.2f}")
            lines.append(f"  Positions: {position_count}")

        return "\n".join(lines)

    async def send_message_with_buttons(
        self,
        text: str,
        reply_markup: Dict[str, Any],
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        message_type: str = "notification",
        related_entity_id: Optional[str] = None
    ) -> bool:
        """
        Send a message with inline keyboard buttons.

        Args:
            text: Message text
            reply_markup: InlineKeyboardMarkup dict
            chat_id: Target chat (defaults to self.chat_id)
            parse_mode: Optional parse mode
            message_type: For logging
            related_entity_id: For logging
        """
        if not self.is_configured:
            logger.warning("Telegram not configured - message not sent")
            return False

        target_chat = chat_id or self.chat_id

        # Log outgoing message
        self._log_outgoing(
            content=text,
            message_type=message_type,
            related_entity_id=related_entity_id,
            metadata={"has_buttons": True}
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "chat_id": target_chat,
                    "text": text,
                    "reply_markup": reply_markup
                }

                if parse_mode:
                    payload["parse_mode"] = parse_mode

                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                )

                if response.status_code == 200:
                    return True

                # If failed, try without parse_mode
                if parse_mode and response.status_code == 400:
                    logger.warning("Parse mode failed, retrying without formatting")
                    payload.pop("parse_mode", None)
                    response = await client.post(
                        f"{self.base_url}/sendMessage",
                        json=payload
                    )
                    return response.status_code == 200

                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram message with buttons: {e}")
            return False

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False
    ) -> bool:
        """
        Answer a callback query (required by Telegram API after button press).

        Args:
            callback_query_id: The callback query ID from the update
            text: Optional text to show (toast notification)
            show_alert: If True, shows a modal alert instead of toast
        """
        if not self.is_configured:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {"callback_query_id": callback_query_id}

                if text:
                    payload["text"] = text
                    payload["show_alert"] = show_alert

                response = await client.post(
                    f"{self.base_url}/answerCallbackQuery",
                    json=payload
                )

                return response.status_code == 200

        except Exception as e:
            logger.error(f"Error answering callback query: {e}")
            return False

    async def send_maca_scan_summary(
        self,
        proposals: List[Dict[str, Any]],
        synthesis: Dict[str, Any],
        technical_signals: List[Dict[str, Any]],
        portfolio: Dict[str, Any],
        trade_id: Optional[str] = None,
        debate: Optional[Dict[str, Any]] = None,
        vote_summary: Optional[Dict[str, Any]] = None,
        cycle_id: Optional[str] = None,
        signal_inventory: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send full MACA scan summary as three messages.

        Message 1: AI Council views (Grok, Perplexity, ChatGPT) + signal inventory
        Message 2: Debate transcript (who changed mind and why)
        Message 3: Chart analysis + Claude's decision + trade status
        """
        # Message 1: AI Council with signal inventory
        msg1 = self.format_ai_council_message(proposals, signal_inventory=signal_inventory)
        await self.send_message(msg1, parse_mode=None, message_type="maca_ai_council")

        # Small delay between messages
        import asyncio
        await asyncio.sleep(0.5)

        # Message 2: Debate transcript (if available) or unanimous agreement note
        if debate and (debate.get("rounds") or []):
            await self.send_maca_debate_summary(
                cycle_id=cycle_id or "",
                debate=debate,
                vote_summary=vote_summary or {},
            )
            await asyncio.sleep(0.5)
        else:
            vs = vote_summary or {}
            reason = vs.get("reason") or ""
            hold = vs.get("hold") is True
            if hold and reason:
                lines = [
                    "\U0001F5E3\uFE0F MACA Debate (IC Minutes)",
                    f"Cycle: {cycle_id or 'N/A'}",
                    "",
                    "\u2705 Unanimous Agreement: HOLD",
                    f"Reason: {reason[:220]}",
                ]
                await self.send_message(
                    "\n".join(lines),
                    parse_mode=None,
                    message_type="maca_debate",
                )
                await asyncio.sleep(0.5)

        # Message 3: Decision with buttons
        msg2 = self.format_decision_message(
            synthesis=synthesis,
            technical_signals=technical_signals,
            portfolio=portfolio,
            trade_id=trade_id
        )

        if trade_id and synthesis.get("decision_type") == "TRADE":
            # Send with approval buttons
            keyboard = self.build_approval_keyboard(trade_id)
            return await self.send_message_with_buttons(
                text=msg2,
                reply_markup=keyboard,
                message_type="maca_decision",
                related_entity_id=trade_id
            )
        else:
            # Send with command buttons only
            keyboard = self.build_command_keyboard()
            return await self.send_message_with_buttons(
                text=msg2,
                reply_markup=keyboard,
                message_type="maca_decision"
            )

    async def send_maca_debate_summary(
        self,
        *,
        cycle_id: str,
        debate: Dict[str, Any],
        vote_summary: Dict[str, Any],
    ) -> bool:
        """Send committee debate transcript as a standalone Telegram message."""

        try:
            rounds = debate.get("rounds") or []
            if not rounds:
                return False

            lines: List[str] = []
            lines.append("🗣️ MACA Debate (IC Minutes)")
            lines.append(f"Cycle: {cycle_id}")

            top = (vote_summary or {}).get("top") or {}
            reason = (vote_summary or {}).get("reason") or ""
            avg_conf = (vote_summary or {}).get("avg_confidence")
            early_exit = (debate or {}).get("early_exit_reason")

            if top:
                lines.append(f"Majority: {top.get('action')} {top.get('ticker') or ''} ({top.get('count')})")
            if isinstance(avg_conf, (int, float)):
                lines.append(f"Avg confidence: {avg_conf:.2f}")
            if reason:
                lines.append(f"Blocker: {reason}")
            if early_exit:
                lines.append(f"Rounds: {len(rounds)} (note: {early_exit})")

            # Vote table snapshot (last known votes)
            vs_votes = (vote_summary or {}).get("votes") or []
            if vs_votes:
                lines.append("")
                lines.append("Votes:")
                for v in vs_votes:
                    sp = v.get("speaker")
                    act = v.get("action")
                    tk = v.get("ticker") or ""
                    cf = v.get("confidence")
                    cf_s = f" ({float(cf):.2f})" if isinstance(cf, (int, float)) else ""
                    lines.append(f"• {sp}: {act} {tk}{cf_s}")

            lines.append("")

            # Round-by-round deltas (one-liners)
            for r in rounds:
                lines.append(f"--- Round {r.get('round')} ---")
                for t in (r.get("turns") or []):
                    sp = (t.get("speaker") or "unknown")
                    status = (t.get("status") or "ok")
                    v = (t.get("vote") or {})
                    act = (v.get("action") or "HOLD")
                    tk = v.get("ticker") or ""
                    cf = v.get("confidence")
                    cf_s = f" ({float(cf):.2f})" if isinstance(cf, (int, float)) else ""

                    if status == "error":
                        err = (t.get("message") or "error").strip().replace("\n", " ")
                        err = err[:180] + ("…" if len(err) > 180 else "")
                        lines.append(f"• {sp}: ERROR — {err}")
                        continue

                    claim = (t.get("claim") or t.get("message") or "").strip().replace("\n", " ")
                    change = (t.get("change_my_mind") or "").strip().replace("\n", " ")
                    changed = bool(t.get("changed_mind"))
                    claim = claim[:140] + ("…" if len(claim) > 140 else "")
                    change = change[:90] + ("…" if len(change) > 90 else "")

                    if change:
                        change_tag = "changed" if changed else "change"
                        lines.append(f"• {sp}: {act} {tk}{cf_s} — {claim} | {change_tag}: {change}")
                    else:
                        changed_tag = " (changed mind)" if changed else ""
                        lines.append(f"• {sp}: {act} {tk}{cf_s} — {claim}{changed_tag}")
                lines.append("")

            text = "\n".join(lines).strip()
            return await self.send_message(text, parse_mode=None, message_type="maca_debate")
        except Exception as e:
            logger.error(f"Failed to send debate summary: {e}")
            return False

    # =========================================================================
    # POSITIONS & HISTORY FORMATTING
    # =========================================================================

    def format_positions_message(self, positions: List[Dict[str, Any]]) -> str:
        """
        Format current positions for Telegram display.

        Args:
            positions: List of position dicts from database/executor

        Returns:
            Formatted string for Telegram
        """
        if not positions:
            return (
                f"{EMOJI_CHART} CURRENT POSITIONS\n"
                f"{'=' * 35}\n\n"
                f"No open positions.\n\n"
                f"Use /check [TICKER] to analyze stocks."
            )

        lines = [
            f"{EMOJI_CHART} CURRENT POSITIONS",
            "=" * 35,
            ""
        ]

        total_value = 0
        total_pnl = 0

        for pos in positions:
            ticker = pos.get("ticker", "???")
            qty = pos.get("quantity", 0)
            entry = pos.get("avg_entry_price", 0)
            current = pos.get("current_price", entry)
            value = pos.get("market_value", qty * current)
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            entry_date = pos.get("entry_date", "")

            total_value += value
            total_pnl += pnl

            # P&L indicator
            if pnl > 0:
                pnl_emoji = EMOJI_GREEN_CIRCLE
                pnl_str = f"+${pnl:,.2f} (+{pnl_pct*100:.1f}%)"
            elif pnl < 0:
                pnl_emoji = EMOJI_RED_CIRCLE
                pnl_str = f"-${abs(pnl):,.2f} ({pnl_pct*100:.1f}%)"
            else:
                pnl_emoji = EMOJI_WHITE_CIRCLE
                pnl_str = "$0.00 (0.0%)"

            # Format entry date - FIXED: catch specific exceptions
            if entry_date:
                try:
                    dt = datetime.fromisoformat(entry_date.replace("Z", "+00:00"))
                    date_str = dt.strftime("%m/%d")
                except (ValueError, AttributeError):
                    date_str = ""
            else:
                date_str = ""

            lines.append(f"{pnl_emoji} {ticker}")
            lines.append(f"   Qty: {qty} @ ${entry:,.2f}")
            lines.append(f"   Now: ${current:,.2f} = ${value:,.2f}")
            lines.append(f"   P&L: {pnl_str}")
            if date_str:
                lines.append(f"   Entry: {date_str}")
            lines.append("")

        # Summary
        lines.append("-" * 35)
        if total_pnl >= 0:
            lines.append(f"Total Value: ${total_value:,.2f}")
            lines.append(f"Total P&L: +${total_pnl:,.2f} {EMOJI_GREEN_CIRCLE}")
        else:
            lines.append(f"Total Value: ${total_value:,.2f}")
            lines.append(f"Total P&L: -${abs(total_pnl):,.2f} {EMOJI_RED_CIRCLE}")

        return "\n".join(lines)

    def format_history_message(
        self,
        trades: List[Dict[str, Any]],
        limit: int = 10
    ) -> str:
        """
        Format trade history for Telegram display.

        Args:
            trades: List of trade dicts from database
            limit: Maximum trades to show

        Returns:
            Formatted string for Telegram
        """
        if not trades:
            return (
                f"{EMOJI_SCROLL} TRADE HISTORY\n"
                f"{'=' * 35}\n\n"
                f"No trade history yet.\n\n"
                f"Trades will appear here after execution."
            )

        lines = [
            f"{EMOJI_SCROLL} TRADE HISTORY (Last {min(len(trades), limit)})",
            "=" * 35,
            ""
        ]

        # Status emoji mapping
        status_emoji = {
            "filled": EMOJI_CHECK,
            "submitted": EMOJI_HOURGLASS,
            "pending_approval": EMOJI_BELL,
            "rejected": EMOJI_CROSS,
            "cancelled": EMOJI_CROSS,
            "failed": EMOJI_WARNING
        }

        for trade in trades[:limit]:
            trade_id = trade.get("id", "???")[:8]
            ticker = trade.get("ticker", "???")
            side = trade.get("side", "???").upper()
            qty = trade.get("quantity", 0)
            status = trade.get("status", "unknown")
            conviction = trade.get("conviction_score", 0)
            fill_price = trade.get("fill_price")
            created = trade.get("created_at", "")

            emoji = status_emoji.get(status, EMOJI_WHITE_CIRCLE)

            # Format date - FIXED: catch specific exceptions
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    date_str = dt.strftime("%m/%d %H:%M")
                except (ValueError, AttributeError):
                    date_str = created[:10]
            else:
                date_str = ""

            lines.append(f"{emoji} {side} {ticker}")
            lines.append(f"   Qty: {qty} | Conv: {conviction}")

            if fill_price:
                lines.append(f"   Fill: ${fill_price:,.2f}")

            lines.append(f"   Status: {status}")

            if date_str:
                lines.append(f"   Date: {date_str}")

            # Show rejection reason if applicable
            if status == "rejected" and trade.get("rejection_reason"):
                reason = trade.get("rejection_reason", "")[:40]
                lines.append(f"   Reason: {reason}")

            lines.append(f"   ID: {trade_id}")
            lines.append("")

        # Summary
        filled = sum(1 for t in trades if t.get("status") == "filled")
        rejected = sum(1 for t in trades if t.get("status") == "rejected")
        pending = sum(1 for t in trades if t.get("status") == "pending_approval")

        lines.append("-" * 35)
        lines.append(f"Filled: {filled} | Rejected: {rejected} | Pending: {pending}")

        return "\n".join(lines)

    def format_maca_check_result(self, result: Dict[str, Any]) -> str:
        """
        Format MACA ticker check result for Telegram.

        Shows all 4 AI theses and Claude's final synthesis.

        Args:
            result: Output from MACAOrchestrator.run_ticker_check()

        Returns:
            Formatted string for Telegram
        """
        ticker = result.get("ticker", "???")
        proposals = result.get("proposals", [])
        synthesis = result.get("synthesis", {})
        cycle_cost = result.get("cycle_cost", {})

        lines = [
            "=" * 40,
            f"{EMOJI_BRAIN} MACA CHECK: {ticker}",
            "=" * 40,
            ""
        ]

        # AI Council Theses
        lines.append(f"{EMOJI_TARGET} AI COUNCIL ANALYSIS")
        lines.append("-" * 35)

        # Source emoji mapping
        source_emoji = {
            "grok": EMOJI_BIRD,
            "perplexity": EMOJI_TARGET,
            "chatgpt": EMOJI_BRAIN
        }

        for proposal in proposals:
            source = proposal.get("ai_source", "unknown")
            emoji = source_emoji.get(source, EMOJI_CHART)
            rec = proposal.get("recommendation", {})

            side = rec.get("side", "N/A")
            conviction = rec.get("conviction_score", 0)
            thesis = rec.get("thesis", "No thesis")[:100]

            # Conviction bar
            bar = self._build_conviction_bar(conviction)

            # Side indicator
            if side == "BUY":
                side_emoji = EMOJI_GREEN_CIRCLE
            elif side == "SELL":
                side_emoji = EMOJI_RED_CIRCLE
            else:
                side_emoji = EMOJI_WHITE_CIRCLE

            lines.append(f"\n{emoji} {source.upper()}")
            lines.append(f"   {side_emoji} {side} | Conv: {conviction}/100")
            lines.append(f"   {bar}")
            lines.append(f"   {thesis}...")

        # Claude Synthesis
        lines.append("")
        lines.append("=" * 35)
        lines.append(f"{EMOJI_BRAIN} CLAUDE'S SYNTHESIS")
        lines.append("-" * 35)

        synth_rec = synthesis.get("recommendation", {})
        decision_type = synthesis.get("decision_type", "NO_TRADE")
        final_conviction = synth_rec.get("conviction_score", 0)
        final_thesis = synth_rec.get("thesis", "No synthesis available")[:150]
        final_side = synth_rec.get("side", "HOLD")

        # Decision indicator
        if decision_type == "TRADE" and final_conviction >= 80:
            decision_emoji = EMOJI_CHECK
            decision_text = "ACTIONABLE"
        else:
            decision_emoji = EMOJI_WHITE_CIRCLE
            decision_text = "WATCH ONLY"

        lines.append(f"Decision: {decision_type}")
        lines.append(f"Side: {final_side}")
        lines.append(f"Conviction: {final_conviction}/100 {decision_emoji} {decision_text}")
        lines.append(f"{self._build_conviction_bar(final_conviction)}")
        lines.append(f"\n{final_thesis}")

        # Cross-validation
        cross_val = synthesis.get("cross_validation", {})
        if cross_val:
            lines.append("")
            lines.append("Cross-Validation:")
            for source, status in cross_val.items():
                lines.append(f"  {source}: {status}")

        # Cost tracking
        lines.append("")
        lines.append("-" * 35)
        total_cost = cycle_cost.get("total_cost_usd", 0)
        total_tokens = cycle_cost.get("total_tokens", 0)
        lines.append(f"API Cost: ${total_cost:.4f} ({total_tokens:,} tokens)")

        # By source
        by_source = cycle_cost.get("by_source", {})
        if by_source:
            cost_parts = []
            for src, data in by_source.items():
                cost_parts.append(f"{src}=${data.get('cost_usd', 0):.3f}")
            lines.append(f"  ({', '.join(cost_parts)})")

        lines.append("=" * 40)

        return "\n".join(lines)

    def format_cost_message(self, cost_summary: Dict[str, Any]) -> str:
        """
        Format API cost summary for Telegram display.

        Args:
            cost_summary: Output from Database.get_cost_summary()

        Returns:
            Formatted string for Telegram
        """
        total_cost = cost_summary.get("total_cost_usd", 0)
        total_tokens = cost_summary.get("total_tokens", 0)
        cycle_count = cost_summary.get("cycle_count", 0)
        period_days = cost_summary.get("period_days", 7)
        by_source = cost_summary.get("by_source", {})
        by_day = cost_summary.get("by_day", [])

        lines = [
            f"{EMOJI_CHART} API COST SUMMARY",
            "=" * 35,
            f"Period: Last {period_days} days",
            "",
            f"Total Cost: ${total_cost:.4f}",
            f"Total Tokens: {total_tokens:,}",
            f"Scan Cycles: {cycle_count}",
        ]

        if cycle_count > 0:
            avg_cost = total_cost / cycle_count
            lines.append(f"Avg Cost/Cycle: ${avg_cost:.4f}")

        # By source breakdown
        if by_source:
            lines.append("")
            lines.append("-" * 35)
            lines.append("BY AI SOURCE:")

            # Sort by cost descending
            sorted_sources = sorted(by_source.items(), key=lambda x: x[1].get("cost_usd", 0), reverse=True)

            for source, data in sorted_sources:
                cost = data.get("cost_usd", 0)
                tokens = data.get("tokens", 0)
                pct = (cost / total_cost * 100) if total_cost > 0 else 0
                lines.append(f"  {source}: ${cost:.4f} ({pct:.0f}%)")

        # Daily breakdown (last 3 days)
        if by_day:
            lines.append("")
            lines.append("-" * 35)
            lines.append("RECENT DAYS:")

            for day in by_day[:3]:
                date = day.get("date", "")
                cost = day.get("cost_usd", 0)
                count = day.get("cycle_count", 0)
                lines.append(f"  {date}: ${cost:.4f} ({count} cycles)")

        # Projection
        if cycle_count > 0 and period_days > 0:
            daily_avg = total_cost / period_days
            monthly_proj = daily_avg * 30
            lines.append("")
            lines.append("-" * 35)
            lines.append(f"Monthly Projection: ~${monthly_proj:.2f}")

        lines.append("=" * 35)

        return "\n".join(lines)

    def format_weekly_digest(self, digest: Dict[str, Any]) -> str:
        """
        Format weekly performance digest for Telegram.

        Args:
            digest: Output from LearningEngine.generate_weekly_digest()

        Returns:
            Formatted string for Telegram
        """
        period = digest.get("period", "Unknown period")
        total_trades = digest.get("total_trades", 0)
        wins = digest.get("wins", 0)
        losses = digest.get("losses", 0)
        win_rate = digest.get("win_rate", 0)
        total_pnl = digest.get("total_pnl", 0)
        total_alpha = digest.get("total_alpha", 0)
        avg_hold_days = digest.get("avg_hold_days", 0)
        top_winner = digest.get("top_winner")
        top_loser = digest.get("top_loser")
        best_source = digest.get("best_source")
        lessons = digest.get("lessons", [])

        # Header
        lines = [
            f"{EMOJI_CHART} WEEKLY PERFORMANCE DIGEST",
            "=" * 35,
            f"Period: {period}",
            ""
        ]

        # Key Metrics
        pnl_emoji = EMOJI_GREEN_CIRCLE if total_pnl >= 0 else EMOJI_RED_CIRCLE
        alpha_emoji = EMOJI_GREEN_CIRCLE if total_alpha >= 0 else EMOJI_RED_CIRCLE

        lines.append(f"{EMOJI_TARGET} KEY METRICS")
        lines.append("-" * 30)
        lines.append(f"Total Trades: {total_trades}")
        lines.append(f"Win Rate: {win_rate:.0%} ({wins}W / {losses}L)")
        lines.append(f"Total P&L: {pnl_emoji} ${total_pnl:,.2f}")
        lines.append(f"Total Alpha: {alpha_emoji} {total_alpha:.1%}")
        if avg_hold_days:
            lines.append(f"Avg Hold: {avg_hold_days:.1f} days")

        # Top Winner
        if top_winner:
            lines.append("")
            lines.append(f"{EMOJI_ROCKET} TOP WINNER")
            lines.append(f"  {top_winner['ticker']}: +${top_winner['pnl']:,.2f} ({top_winner['return_pct']:.1%})")

        # Top Loser
        if top_loser:
            lines.append("")
            lines.append(f"{EMOJI_WARNING} TOP LOSER")
            lines.append(f"  {top_loser['ticker']}: ${top_loser['pnl']:,.2f} ({top_loser['return_pct']:.1%})")

        # Best Source
        if best_source:
            lines.append("")
            lines.append(f"{EMOJI_BRAIN} BEST AI SOURCE: {best_source}")

        # Lessons
        if lessons:
            lines.append("")
            lines.append("-" * 30)
            lines.append(f"{EMOJI_SCROLL} LESSONS LEARNED:")
            for lesson in lessons[:3]:
                lines.append(f"  {lesson}")

        lines.append("")
        lines.append("=" * 35)

        return "\n".join(lines)

    def format_event_signal(self, signal: Dict[str, Any]) -> str:
        """
        Format event signal for Telegram display.

        Args:
            signal: EventSignal.to_dict() output

        Returns:
            Formatted single-line string for Telegram
        """
        event_type = signal.get("event_type", "UNKNOWN")
        ticker = signal.get("asset_scope", {}).get("tickers", ["???"])[0]
        bias = signal.get("directional_bias", "mixed")
        confidence = signal.get("confidence", 0)
        summary = signal.get("summary", "")

        # Bias emoji
        if bias == "bullish":
            bias_emoji = EMOJI_GREEN_CIRCLE
        elif bias == "bearish":
            bias_emoji = EMOJI_RED_CIRCLE
        else:
            bias_emoji = EMOJI_YELLOW_CIRCLE

        # Format event type nicely
        event_display = event_type.replace("_", " ").title()

        return f"{bias_emoji} {ticker}: {event_display} (conf: {confidence:.0%})"


async def send_telegram_message(text: str) -> bool:
    """Quick helper to send a Telegram message."""
    bot = TelegramBot()
    return await bot.send_message(text)
