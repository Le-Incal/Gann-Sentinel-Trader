"""
Gann Sentinel Trader - Database
SQLite storage for signals, analyses, trades, and portfolio state.
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
        self.db_path = db_path or Config.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
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
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_dedup ON signals(dedup_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(ticker)")
            
            logger.info(f"Database initialized at {self.db_path}")
    
    # =========================================================================
    # SIGNALS
    # =========================================================================
    
    def save_signal(self, signal_data: dict) -> str:
        """Save a signal to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            signal_id = signal_data.get("signal_id")
            dedup_hash = signal_data.get("dedup_hash")
            
            # Check for duplicate
            cursor.execute("SELECT id FROM signals WHERE dedup_hash = ?", (dedup_hash,))
            existing = cursor.fetchone()
            if existing:
                logger.debug(f"Duplicate signal detected: {dedup_hash}")
                return existing["id"]
            
            cursor.execute("""
                INSERT INTO signals (id, signal_type, source, ticker, data, timestamp_utc, staleness_seconds, dedup_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                signal_data.get("signal_type"),
                signal_data.get("source"),
                signal_data.get("asset_scope", {}).get("tickers", [None])[0],
                json.dumps(signal_data),
                signal_data.get("timestamp_utc"),
                signal_data.get("staleness_seconds"),
                dedup_hash
            ))
            
            logger.info(f"Saved signal: {signal_id}")
            return signal_id
    
    def get_signals(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[dict]:
        """Retrieve signals with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT data FROM signals WHERE 1=1"
            params = []
            
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)
            
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)
            
            if since:
                query += " AND timestamp_utc >= ?"
                params.append(since.isoformat())
            
            query += " ORDER BY timestamp_utc DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [json.loads(row["data"]) for row in rows]
    
    # =========================================================================
    # ANALYSES
    # =========================================================================
    
    def save_analysis(self, analysis_data: dict) -> str:
        """Save an analysis to the database."""
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
                json.dumps(analysis_data),
                json.dumps(analysis_data.get("signals_used", []))
            ))
            
            logger.info(f"Saved analysis: {analysis_id}")
            return analysis_id
    
    def get_analyses(
        self,
        ticker: Optional[str] = None,
        min_conviction: Optional[int] = None,
        limit: int = 50
    ) -> List[dict]:
        """Retrieve analyses with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT full_analysis FROM analyses WHERE 1=1"
            params = []
            
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)
            
            if min_conviction:
                query += " AND conviction_score >= ?"
                params.append(min_conviction)
            
            query += " ORDER BY timestamp_utc DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [json.loads(row["full_analysis"]) for row in rows]
    
    # =========================================================================
    # TRADES
    # =========================================================================
    
    def save_trade(self, trade_data: dict) -> str:
        """Save a trade to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            trade_id = trade_data.get("trade_id")
            
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
                        updated_at = ?
                    WHERE id = ?
                """, (
                    trade_data.get("status"),
                    trade_data.get("alpaca_order_id"),
                    trade_data.get("fill_price"),
                    trade_data.get("fill_quantity"),
                    trade_data.get("filled_at"),
                    datetime.now(timezone.utc).isoformat(),
                    trade_id
                ))
            else:
                # Insert new trade
                cursor.execute("""
                    INSERT INTO trades (id, analysis_id, ticker, side, quantity, order_type, limit_price, status, thesis, conviction_score)
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
        """Get a specific trade by ID."""
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
    
    def update_trade_status(self, trade_id: str, status: str, **kwargs) -> bool:
        """Update trade status and optional fields."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?", "updated_at = ?"]
            params = [status, datetime.now(timezone.utc).isoformat()]
            
            for key, value in kwargs.items():
                updates.append(f"{key} = ?")
                params.append(value)
            
            params.append(trade_id)
            
            cursor.execute(f"""
                UPDATE trades SET {', '.join(updates)}
                WHERE id = ? OR id LIKE ?
            """, params + [f"{trade_id}%"])
            
            return cursor.rowcount > 0
    
    # =========================================================================
    # POSITIONS
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
    # PORTFOLIO SNAPSHOTS
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
    # ERRORS
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
