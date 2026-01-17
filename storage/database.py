"""
Gann Sentinel Trader - Database
SQLite storage for signals, analyses, trades, portfolio state, and telegram activity logs.

Version: 2.3.0 - SQL Injection Prevention + Cost Tracking
- Added column whitelisting for dynamic UPDATE queries
- In-memory mode for testing
- Per-cycle cost tracking methods
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import Config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for Gann Sentinel Trader."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        self._persistent_conn = None  # For in-memory mode

        if db_path == ":memory:" or str(db_path) == ":memory:":
            # In-memory mode for testing - needs persistent connection
            self.db_path = ":memory:"
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            self.db_path = db_path or Config.DATABASE_PATH
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        if self._persistent_conn:
            # Use persistent connection for in-memory mode
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ticker TEXT,
                    data JSON NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    staleness_seconds INTEGER,
                    dedup_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Analyses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    ticker TEXT,
                    recommendation TEXT,
                    conviction_score INTEGER,
                    thesis TEXT,
                    full_analysis JSON NOT NULL,
                    signals_used JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    alpaca_order_id TEXT,
                    fill_price REAL,
                    fill_quantity REAL,
                    filled_at TEXT,
                    thesis TEXT,
                    conviction_score INTEGER,
                    rejection_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL UNIQUE,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_pct REAL,
                    thesis TEXT,
                    analysis_id TEXT,
                    entry_date TEXT,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    total_value REAL NOT NULL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,
                    positions JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Errors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    context JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # =========================================================================
            # NEW: Telegram Messages Table for Full Activity Logging
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    command TEXT,
                    content TEXT NOT NULL,
                    content_preview TEXT,
                    related_entity_id TEXT,
                    related_entity_type TEXT,
                    metadata JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # =========================================================================
            # NEW: AI Thesis Proposals Table for Multi-Agent Architecture
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_proposals (
                    id TEXT PRIMARY KEY,
                    scan_cycle_id TEXT NOT NULL,
                    ai_source TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    proposal_type TEXT NOT NULL,
                    ticker TEXT,
                    side TEXT,
                    conviction_score INTEGER,
                    thesis TEXT,
                    raw_data JSON,
                    time_sensitive BOOLEAN DEFAULT FALSE,
                    catalyst_deadline TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # =========================================================================
            # NEW: AI Reviews Table for Peer Review Phase
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_reviews (
                    id TEXT PRIMARY KEY,
                    scan_cycle_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    reviewer_ai TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    concerns TEXT,
                    confidence_adjustment INTEGER DEFAULT 0,
                    raw_response JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (proposal_id) REFERENCES ai_proposals(id)
                )
            """)

            # =========================================================================
            # NEW: Scan Cycles Table for Tracking Multi-Agent Rounds
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_cycles (
                    id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    cycle_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proposals_count INTEGER DEFAULT 0,
                    selected_proposal_id TEXT,
                    final_conviction INTEGER,
                    final_decision TEXT,
                    restart_count INTEGER DEFAULT 0,
                    duration_seconds REAL,
                    metadata JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (selected_proposal_id) REFERENCES ai_proposals(id)
                )
            """)

            # =========================================================================
            # NEW: Debate Sessions Table for Investment Committee Deliberation
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debate_sessions (
                    id TEXT PRIMARY KEY,
                    scan_cycle_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # =========================================================================
            # NEW: Debate Turns Table for Recording Each Speaker's Arguments
            # =========================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debate_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    scan_cycle_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    message TEXT,
                    agreements JSON,
                    disagreements JSON,
                    changed_mind INTEGER,
                    vote_action TEXT,
                    vote_ticker TEXT,
                    vote_side TEXT,
                    vote_confidence REAL,
                    raw JSON,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES debate_sessions(id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_dedup ON signals(dedup_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(ticker)")

            # New indices for telegram and multi-agent tables
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telegram_timestamp ON telegram_messages(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telegram_direction ON telegram_messages(direction)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telegram_type ON telegram_messages(message_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_proposals_cycle ON ai_proposals(scan_cycle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_proposals_source ON ai_proposals(ai_source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_reviews_cycle ON ai_reviews(scan_cycle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_cycles_status ON scan_cycles(status)")

            # Debate indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_sessions_cycle ON debate_sessions(scan_cycle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_turns_session ON debate_turns(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_turns_cycle ON debate_turns(scan_cycle_id)")

            logger.info(f"Database initialized at {self.db_path}")

    # =========================================================================
    # TELEGRAM MESSAGE LOGGING
    # =========================================================================

    def log_telegram_message(
        self,
        direction: str,
        message_type: str,
        content: str,
        command: Optional[str] = None,
        related_entity_id: Optional[str] = None,
        related_entity_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Log a Telegram message (incoming or outgoing).

        Args:
            direction: 'incoming' or 'outgoing'
            message_type: 'command', 'notification', 'approval_request',
                         'scan_summary', 'error', 'response', 'status', 'daily_digest'
            content: Full message text
            command: If incoming command, the command name (e.g., 'status', 'approve')
            related_entity_id: Associated trade_id, decision_id, scan_id
            related_entity_type: 'trade', 'decision', 'scan', 'position'
            metadata: Additional context as dict

        Returns:
            The message ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            timestamp = datetime.now(timezone.utc).isoformat()

            # Create preview (first 100 chars, single line)
            preview = content.replace('\n', ' ')[:100]
            if len(content) > 100:
                preview += "..."

            cursor.execute("""
                INSERT INTO telegram_messages
                (timestamp_utc, direction, message_type, command, content, content_preview,
                 related_entity_id, related_entity_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                direction,
                message_type,
                command,
                content,
                preview,
                related_entity_id,
                related_entity_type,
                json.dumps(metadata) if metadata else None
            ))

            message_id = cursor.lastrowid
            logger.debug(f"Logged telegram message: {direction} {message_type} (id={message_id})")
            return message_id

    def get_telegram_messages(
        self,
        limit: int = 50,
        direction: Optional[str] = None,
        message_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[dict]:
        """
        Get telegram messages with optional filters.

        Args:
            limit: Max messages to return
            direction: Filter by 'incoming' or 'outgoing'
            message_type: Filter by message type
            since: Only messages after this timestamp

        Returns:
            List of message dicts, newest first
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM telegram_messages WHERE 1=1"
            params = []

            if direction:
                query += " AND direction = ?"
                params.append(direction)

            if message_type:
                query += " AND message_type = ?"
                params.append(message_type)

            if since:
                query += " AND timestamp_utc > ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp_utc DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_telegram_activity_summary(self, hours: int = 24) -> dict:
        """
        Get summary of telegram activity for the last N hours.

        Returns:
            Dict with counts by direction and type
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            since = datetime.now(timezone.utc) - timedelta(hours=hours)

            cursor.execute("""
                SELECT direction, message_type, COUNT(*) as count
                FROM telegram_messages
                WHERE timestamp_utc > ?
                GROUP BY direction, message_type
                ORDER BY count DESC
            """, (since.isoformat(),))

            rows = cursor.fetchall()

            summary = {
                "period_hours": hours,
                "incoming": {},
                "outgoing": {},
                "total_incoming": 0,
                "total_outgoing": 0
            }

            for row in rows:
                direction = row["direction"]
                msg_type = row["message_type"]
                count = row["count"]

                summary[direction][msg_type] = count
                summary[f"total_{direction}"] += count

            return summary

    # =========================================================================
    # AI PROPOSALS (Multi-Agent Architecture)
    # =========================================================================

    def save_ai_proposal(self, proposal_data: dict) -> str:
        """Save an AI thesis proposal."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            proposal_id = proposal_data.get("proposal_id")

            cursor.execute("""
                INSERT INTO ai_proposals
                (id, scan_cycle_id, ai_source, timestamp_utc, proposal_type,
                 ticker, side, conviction_score, thesis, raw_data,
                 time_sensitive, catalyst_deadline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proposal_id,
                proposal_data.get("scan_cycle_id"),
                proposal_data.get("ai_source"),
                proposal_data.get("timestamp_utc"),
                proposal_data.get("proposal_type"),
                proposal_data.get("ticker"),
                proposal_data.get("side"),
                proposal_data.get("conviction_score"),
                proposal_data.get("thesis"),
                json.dumps(proposal_data.get("raw_data")) if proposal_data.get("raw_data") else None,
                proposal_data.get("time_sensitive", False),
                proposal_data.get("catalyst_deadline")
            ))

            logger.info(f"Saved AI proposal: {proposal_id} from {proposal_data.get('ai_source')}")
            return proposal_id

    def get_proposals_for_cycle(self, scan_cycle_id: str) -> List[dict]:
        """Get all proposals for a scan cycle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM ai_proposals WHERE scan_cycle_id = ? ORDER BY conviction_score DESC",
                (scan_cycle_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("raw_data"):
                    d["raw_data"] = json.loads(d["raw_data"])
                results.append(d)
            return results

    # =========================================================================
    # AI REVIEWS (Peer Review Phase)
    # =========================================================================

    def save_ai_review(self, review_data: dict) -> str:
        """Save an AI peer review."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            review_id = review_data.get("review_id")

            cursor.execute("""
                INSERT INTO ai_reviews
                (id, scan_cycle_id, proposal_id, reviewer_ai, timestamp_utc,
                 verdict, concerns, confidence_adjustment, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review_id,
                review_data.get("scan_cycle_id"),
                review_data.get("proposal_id"),
                review_data.get("reviewer_ai"),
                review_data.get("timestamp_utc"),
                review_data.get("verdict"),
                review_data.get("concerns"),
                review_data.get("confidence_adjustment", 0),
                json.dumps(review_data.get("raw_response")) if review_data.get("raw_response") else None
            ))

            logger.info(f"Saved AI review: {review_id} from {review_data.get('reviewer_ai')}")
            return review_id

    def get_reviews_for_proposal(self, proposal_id: str) -> List[dict]:
        """Get all reviews for a proposal."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM ai_reviews WHERE proposal_id = ? ORDER BY timestamp_utc",
                (proposal_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("raw_response"):
                    d["raw_response"] = json.loads(d["raw_response"])
                results.append(d)
            return results

    # =========================================================================
    # SCAN CYCLES (Multi-Agent Tracking)
    # =========================================================================

    def create_scan_cycle(self, cycle_data: dict) -> str:
        """Create a new scan cycle record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cycle_id = cycle_data.get("cycle_id")

            cursor.execute("""
                INSERT INTO scan_cycles
                (id, timestamp_utc, cycle_type, status, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cycle_id,
                cycle_data.get("timestamp_utc"),
                cycle_data.get("cycle_type", "scheduled"),
                cycle_data.get("status", "started"),
                json.dumps(cycle_data.get("metadata")) if cycle_data.get("metadata") else None
            ))

            logger.info(f"Created scan cycle: {cycle_id}")
            return cycle_id

    # Allowed columns for dynamic updates (SQL injection prevention)
    SCAN_CYCLE_COLUMNS = frozenset({
        "status", "proposals_count", "selected_proposal_id", "final_conviction",
        "final_decision", "restart_count", "duration_seconds", "metadata",
        "total_cost_usd", "error"
    })

    def update_scan_cycle(self, cycle_id: str, **kwargs) -> bool:
        """Update scan cycle with results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            for key, value in kwargs.items():
                # SQL injection prevention: only allow whitelisted columns
                if key not in self.SCAN_CYCLE_COLUMNS:
                    logger.warning(f"Ignoring invalid column in update_scan_cycle: {key}")
                    continue

                if key == "metadata" and value:
                    value = json.dumps(value)
                updates.append(f"{key} = ?")
                params.append(value)

            if not updates:
                return False

            params.append(cycle_id)

            cursor.execute(f"""
                UPDATE scan_cycles SET {', '.join(updates)}
                WHERE id = ?
            """, params)

            return cursor.rowcount > 0

    def update_cycle_costs(
        self,
        cycle_id: str,
        total_tokens: int,
        total_cost_usd: float,
        cost_by_source: Dict[str, Any]
    ) -> bool:
        """
        Update scan cycle with cost tracking data.

        Args:
            cycle_id: Scan cycle ID
            total_tokens: Total tokens used across all AIs
            total_cost_usd: Total cost in USD
            cost_by_source: Dict mapping source -> {tokens, cost_usd}

        Returns:
            True if updated successfully
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get existing metadata
            cursor.execute("SELECT metadata FROM scan_cycles WHERE id = ?", (cycle_id,))
            row = cursor.fetchone()

            if row:
                existing_metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            else:
                existing_metadata = {}

            # Add cost data to metadata
            existing_metadata["cost_tracking"] = {
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "by_source": cost_by_source,
                "tracked_at": datetime.now(timezone.utc).isoformat()
            }

            cursor.execute("""
                UPDATE scan_cycles SET metadata = ?
                WHERE id = ?
            """, (json.dumps(existing_metadata), cycle_id))

            logger.info(f"Updated cycle {cycle_id[:8]} costs: ${total_cost_usd:.4f}")

            return cursor.rowcount > 0

    def get_recent_scan_cycles(self, limit: int = 10) -> List[dict]:
        """Get recent scan cycles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM scan_cycles ORDER BY timestamp_utc DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("metadata"):
                    d["metadata"] = json.loads(d["metadata"])
                results.append(d)
            return results

    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get aggregated cost summary for scan cycles.

        Args:
            days: Number of days to look back

        Returns:
            Dict with total_cost_usd, total_tokens, cycle_count, by_source
        """
        from datetime import datetime, timezone, timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM scan_cycles WHERE timestamp_utc > ? AND status = 'complete'",
                (cutoff,)
            )
            rows = cursor.fetchall()

            total_cost = 0.0
            total_tokens = 0
            by_source: Dict[str, Dict[str, float]] = {}

            for row in rows:
                d = dict(row)
                metadata = json.loads(d.get("metadata", "{}")) if d.get("metadata") else {}
                cost_tracking = metadata.get("cost_tracking", {})

                total_cost += cost_tracking.get("total_cost_usd", 0)
                total_tokens += cost_tracking.get("total_tokens", 0)

                # Aggregate by source
                for source, data in cost_tracking.get("by_source", {}).items():
                    if source not in by_source:
                        by_source[source] = {"cost_usd": 0, "tokens": 0}
                    by_source[source]["cost_usd"] += data.get("cost_usd", 0)
                    by_source[source]["tokens"] += data.get("tokens", 0)

            return {
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "cycle_count": len(rows),
                "by_source": by_source,
                "period_days": days
            }

    def get_cost_by_day(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get cost breakdown by day.

        Args:
            days: Number of days to look back

        Returns:
            List of dicts with date, cost_usd, cycle_count
        """
        from datetime import datetime, timezone, timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM scan_cycles WHERE timestamp_utc > ? AND status = 'complete' ORDER BY timestamp_utc DESC",
                (cutoff,)
            )
            rows = cursor.fetchall()

            # Group by date
            daily: Dict[str, Dict[str, Any]] = {}

            for row in rows:
                d = dict(row)
                # Extract date from timestamp
                ts = d.get("timestamp_utc", "")
                date = ts[:10] if ts else "unknown"

                if date not in daily:
                    daily[date] = {"date": date, "cost_usd": 0, "cycle_count": 0, "tokens": 0}

                metadata = json.loads(d.get("metadata", "{}")) if d.get("metadata") else {}
                cost_tracking = metadata.get("cost_tracking", {})

                daily[date]["cost_usd"] += cost_tracking.get("total_cost_usd", 0)
                daily[date]["tokens"] += cost_tracking.get("total_tokens", 0)
                daily[date]["cycle_count"] += 1

            # Sort by date descending
            result = sorted(daily.values(), key=lambda x: x["date"], reverse=True)
            return result

    # =========================================================================
    # DEBATE SESSIONS AND TURNS
    # =========================================================================

    def create_debate_session(self, scan_cycle_id: str) -> str:
        """Create a new debate session for a scan cycle."""
        import uuid
        session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO debate_sessions (id, scan_cycle_id) VALUES (?, ?)",
                (session_id, scan_cycle_id),
            )
            conn.commit()
        return session_id

    def save_debate_turn(self, session_id: str, scan_cycle_id: str, turn: dict) -> str:
        """Persist a single debate turn."""
        import uuid
        turn_id = str(uuid.uuid4())
        vote = turn.get("vote") or {}
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO debate_turns (
                    id, session_id, scan_cycle_id, round, speaker, message,
                    agreements, disagreements, changed_mind,
                    vote_action, vote_ticker, vote_side, vote_confidence,
                    raw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    scan_cycle_id,
                    turn.get("round", 0),
                    turn.get("speaker", "unknown"),
                    turn.get("message"),
                    json.dumps(turn.get("agreements")) if turn.get("agreements") else None,
                    json.dumps(turn.get("disagreements")) if turn.get("disagreements") else None,
                    1 if turn.get("changed_mind") else 0,
                    vote.get("action"),
                    vote.get("ticker"),
                    vote.get("side"),
                    vote.get("confidence"),
                    json.dumps(turn),
                ),
            )
            conn.commit()
        return turn_id

    def get_debate_turns(self, session_id: str) -> List[dict]:
        """Get all debate turns for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM debate_turns WHERE session_id = ? ORDER BY round, created_at",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # SIGNALS (existing)
    # =========================================================================

    def save_signal(self, signal_data: dict) -> str:
        """Save a signal to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            signal_id = signal_data.get("signal_id")
            dedup_hash = signal_data.get("dedup_hash")

            # Check for duplicate
            if dedup_hash:
                cursor.execute("SELECT id FROM signals WHERE dedup_hash = ?", (dedup_hash,))
                existing = cursor.fetchone()
                if existing:
                    logger.debug(f"Duplicate signal detected: {dedup_hash}")
                    return existing["id"]

            # Safely extract ticker from asset_scope
            tickers = signal_data.get("asset_scope", {}).get("tickers", [])
            ticker = tickers[0] if tickers else None

            cursor.execute("""
                INSERT INTO signals (id, signal_type, source, ticker, data, timestamp_utc, staleness_seconds, dedup_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                signal_data.get("signal_type") or signal_data.get("category"),
                signal_data.get("source") or signal_data.get("source_type"),
                ticker,
                json.dumps(signal_data),
                signal_data.get("timestamp_utc"),
                signal_data.get("staleness_seconds"),
                dedup_hash
            ))

            logger.info(f"Saved signal: {signal_id}")
            return signal_id

    def get_signals_since(self, since: datetime, source: Optional[str] = None) -> List[dict]:
        """Get signals since a given timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if source:
                cursor.execute(
                    "SELECT * FROM signals WHERE timestamp_utc > ? AND source = ? ORDER BY timestamp_utc DESC",
                    (since.isoformat(), source)
                )
            else:
                cursor.execute(
                    "SELECT * FROM signals WHERE timestamp_utc > ? ORDER BY timestamp_utc DESC",
                    (since.isoformat(),)
                )

            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["data"] = json.loads(d["data"])
                results.append(d)
            return results

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        """Get recent signals."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM signals ORDER BY timestamp_utc DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["data"] = json.loads(d["data"])
                results.append(d)
            return results

    # =========================================================================
    # ANALYSES (existing)
    # =========================================================================

    def save_analysis(self, analysis_data: dict) -> str:
        """Save an analysis."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            analysis_id = analysis_data.get("analysis_id")

            cursor.execute("""
                INSERT INTO analyses (id, timestamp_utc, ticker, recommendation, conviction_score, thesis, full_analysis, signals_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                analysis_data.get("timestamp_utc"),
                analysis_data.get("ticker"),
                analysis_data.get("recommendation"),
                analysis_data.get("conviction_score"),
                analysis_data.get("thesis"),
                json.dumps(analysis_data.get("full_analysis", {})),
                json.dumps(analysis_data.get("signals_used", []))
            ))

            logger.info(f"Saved analysis: {analysis_id}")
            return analysis_id

    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        """Get a specific analysis."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["full_analysis"] = json.loads(d.get("full_analysis", "{}"))
                d["signals_used"] = json.loads(d.get("signals_used", "[]"))
                return d
            return None

    def get_recent_analyses(self, limit: int = 20) -> List[dict]:
        """Get recent analyses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM analyses ORDER BY timestamp_utc DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["full_analysis"] = json.loads(d.get("full_analysis", "{}"))
                d["signals_used"] = json.loads(d.get("signals_used", "[]"))
                results.append(d)
            return results

    # =========================================================================
    # TRADES (existing)
    # =========================================================================

    def save_trade(self, trade_data: dict) -> str:
        """Save a trade."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            trade_id = trade_data.get("id") or trade_data.get("trade_id")

            # Check if trade exists
            cursor.execute("SELECT id FROM trades WHERE id = ?", (trade_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing trade
                cursor.execute("""
                    UPDATE trades SET
                        status = ?,
                        alpaca_order_id = ?,
                        fill_price = ?,
                        fill_quantity = ?,
                        filled_at = ?,
                        rejection_reason = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    trade_data.get("status"),
                    trade_data.get("alpaca_order_id"),
                    trade_data.get("fill_price"),
                    trade_data.get("fill_quantity"),
                    trade_data.get("filled_at"),
                    trade_data.get("rejection_reason"),
                    datetime.now(timezone.utc).isoformat(),
                    trade_id
                ))
            else:
                # Insert new trade
                cursor.execute("""
                    INSERT INTO trades
                    (id, analysis_id, ticker, side, quantity, order_type, limit_price,
                     status, thesis, conviction_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    trade_data.get("analysis_id"),
                    trade_data.get("ticker"),
                    trade_data.get("side"),
                    trade_data.get("quantity"),
                    trade_data.get("order_type"),
                    trade_data.get("limit_price"),
                    trade_data.get("status"),
                    trade_data.get("thesis"),
                    trade_data.get("conviction_score")
                ))

            logger.info(f"Saved trade: {trade_id}")
            return trade_id

    def get_trade(self, trade_id: str) -> Optional[dict]:
        """Get a specific trade by ID (supports partial ID match)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE id = ? OR id LIKE ?", (trade_id, f"{trade_id}%"))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_pending_trades(self) -> List[dict]:
        """Get all trades pending approval."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'pending_approval' ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_trades(self, limit: int = 10) -> List[dict]:
        """Get recent trades ordered by creation date."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # Allowed columns for trade updates (SQL injection prevention)
    TRADE_UPDATE_COLUMNS = frozenset({
        "fill_price", "fill_quantity", "filled_at", "alpaca_order_id",
        "rejection_reason", "error", "metadata"
    })

    def update_trade_status(self, trade_id: str, status: str, **kwargs) -> bool:
        """Update trade status and optional fields."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            updates = ["status = ?", "updated_at = ?"]
            params = [status, datetime.now(timezone.utc).isoformat()]

            for key, value in kwargs.items():
                # SQL injection prevention: only allow whitelisted columns
                if key not in self.TRADE_UPDATE_COLUMNS:
                    logger.warning(f"Ignoring invalid column in update_trade_status: {key}")
                    continue
                updates.append(f"{key} = ?")
                params.append(value)

            params.append(trade_id)

            cursor.execute(f"""
                UPDATE trades SET {', '.join(updates)}
                WHERE id = ? OR id LIKE ?
            """, params + [f"{trade_id}%"])

            return cursor.rowcount > 0

    # =========================================================================
    # POSITIONS (existing)
    # =========================================================================

    def save_position(self, position_data: dict) -> str:
        """Save or update a position."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            ticker = position_data.get("ticker")

            cursor.execute("""
                INSERT OR REPLACE INTO positions
                (id, ticker, quantity, avg_entry_price, current_price, market_value,
                 unrealized_pnl, unrealized_pnl_pct, thesis, analysis_id, entry_date,
                 stop_loss_price, take_profit_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data.get("position_id"),
                ticker,
                position_data.get("quantity"),
                position_data.get("avg_entry_price"),
                position_data.get("current_price"),
                position_data.get("market_value"),
                position_data.get("unrealized_pnl"),
                position_data.get("unrealized_pnl_pct"),
                position_data.get("thesis"),
                position_data.get("analysis_id"),
                position_data.get("entry_date"),
                position_data.get("stop_loss_price"),
                position_data.get("take_profit_price"),
                datetime.now(timezone.utc).isoformat()
            ))

            logger.info(f"Saved position: {ticker}")
            return ticker

    def get_positions(self) -> List[dict]:
        """Get all current positions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE quantity > 0")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get a specific position."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_position(self, ticker: str) -> bool:
        """Delete a position (when fully closed)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
            return cursor.rowcount > 0

    # =========================================================================
    # PORTFOLIO SNAPSHOTS (existing)
    # =========================================================================

    def save_snapshot(self, snapshot_data: dict) -> str:
        """Save a portfolio snapshot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            snapshot_id = snapshot_data.get("snapshot_id")

            cursor.execute("""
                INSERT INTO portfolio_snapshots (id, timestamp_utc, cash, positions_value, total_value, daily_pnl, daily_pnl_pct, positions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                snapshot_data.get("timestamp_utc"),
                snapshot_data.get("cash"),
                snapshot_data.get("positions_value"),
                snapshot_data.get("total_value"),
                snapshot_data.get("daily_pnl"),
                snapshot_data.get("daily_pnl_pct"),
                json.dumps(snapshot_data.get("positions", []))
            ))

            logger.info(f"Saved portfolio snapshot: {snapshot_id}")
            return snapshot_id

    def get_latest_snapshot(self) -> Optional[dict]:
        """Get the most recent portfolio snapshot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp_utc DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["positions"] = json.loads(result.get("positions", "[]"))
                return result
            return None

    # =========================================================================
    # ERRORS (existing)
    # =========================================================================

    def log_error(
        self,
        error_type: str,
        component: str,
        message: str,
        stack_trace: Optional[str] = None,
        context: Optional[dict] = None
    ) -> None:
        """Log an error to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO errors (error_type, component, message, stack_trace, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                error_type,
                component,
                message,
                stack_trace,
                json.dumps(context) if context else None
            ))
            logger.error(f"[{component}] {error_type}: {message}")

    def get_recent_errors(self, limit: int = 50) -> List[dict]:
        """Get recent errors."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM errors ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# Import timedelta for the activity summary
from datetime import timedelta
