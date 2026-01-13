"""
Data Exporter for Gann Sentinel Trader
Exports trades, signals, positions, and performance data to CSV and Parquet formats.

Version: 1.0.0
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CSV EXPORTS
# =============================================================================

def export_trades_csv(trades: List[Dict[str, Any]]) -> str:
    """
    Export trades to CSV format.

    Args:
        trades: List of trade dicts from database

    Returns:
        CSV content as string
    """
    if not trades:
        return "id,ticker,side,quantity,status,fill_price,conviction_score,created_at\n"

    output = io.StringIO()

    # Define columns
    columns = [
        "id", "ticker", "side", "quantity", "order_type", "status",
        "fill_price", "fill_quantity", "conviction_score", "thesis",
        "rejection_reason", "created_at", "filled_at", "alpaca_order_id"
    ]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for trade in trades:
        # Flatten and clean data
        row = {col: trade.get(col, "") for col in columns}
        writer.writerow(row)

    return output.getvalue()


def export_signals_csv(signals: List[Dict[str, Any]]) -> str:
    """
    Export signals to CSV format.

    Args:
        signals: List of signal dicts from database

    Returns:
        CSV content as string
    """
    if not signals:
        return "id,source,ticker,signal_type,timestamp_utc,data\n"

    output = io.StringIO()

    columns = [
        "id", "source", "ticker", "signal_type", "timestamp_utc",
        "staleness_seconds", "data"
    ]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for signal in signals:
        row = {col: signal.get(col, "") for col in columns}

        # Serialize nested data to JSON string
        if isinstance(row.get("data"), dict):
            row["data"] = json.dumps(row["data"])

        writer.writerow(row)

    return output.getvalue()


def export_positions_csv(positions: List[Dict[str, Any]]) -> str:
    """
    Export positions to CSV format.

    Args:
        positions: List of position dicts

    Returns:
        CSV content as string
    """
    if not positions:
        return "ticker,quantity,avg_entry_price,current_price,market_value,unrealized_pnl,unrealized_pnl_pct,entry_date\n"

    output = io.StringIO()

    columns = [
        "ticker", "quantity", "avg_entry_price", "current_price",
        "market_value", "unrealized_pnl", "unrealized_pnl_pct",
        "entry_date", "thesis", "stop_loss_price", "take_profit_price"
    ]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for position in positions:
        row = {col: position.get(col, "") for col in columns}
        writer.writerow(row)

    return output.getvalue()


def export_analyses_csv(analyses: List[Dict[str, Any]]) -> str:
    """
    Export analyses to CSV format.

    Args:
        analyses: List of analysis dicts from database

    Returns:
        CSV content as string
    """
    if not analyses:
        return "id,ticker,recommendation,conviction_score,thesis,created_at\n"

    output = io.StringIO()

    columns = [
        "id", "ticker", "recommendation", "conviction_score", "position_size_pct",
        "stop_loss_pct", "time_horizon", "thesis", "bull_case", "bear_case",
        "catalyst", "catalyst_date", "is_actionable", "created_at"
    ]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for analysis in analyses:
        row = {col: analysis.get(col, "") for col in columns}
        writer.writerow(row)

    return output.getvalue()


def export_trade_outcomes_csv(outcomes: List[Dict[str, Any]]) -> str:
    """
    Export trade outcomes to CSV format (for Learning Engine data).

    Args:
        outcomes: List of trade outcome dicts

    Returns:
        CSV content as string
    """
    if not outcomes:
        return "trade_id,ticker,side,entry_price,exit_price,realized_pnl,realized_pnl_pct,alpha,hold_time_hours,exit_reason\n"

    output = io.StringIO()

    columns = [
        "trade_id", "ticker", "side", "entry_price", "entry_date",
        "exit_price", "exit_date", "exit_reason", "realized_pnl",
        "realized_pnl_pct", "hold_time_hours", "spy_return_same_period",
        "alpha", "primary_signal_source", "signal_sources_agreed"
    ]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()

    for outcome in outcomes:
        row = {col: outcome.get(col, "") for col in columns}
        writer.writerow(row)

    return output.getvalue()


# =============================================================================
# PARQUET EXPORTS
# =============================================================================

def export_trades_parquet(trades: List[Dict[str, Any]]) -> bytes:
    """
    Export trades to Parquet format.

    Args:
        trades: List of trade dicts from database

    Returns:
        Parquet file content as bytes
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")

    if not trades:
        # Return empty parquet with schema
        schema = pa.schema([
            ("id", pa.string()),
            ("ticker", pa.string()),
            ("side", pa.string()),
            ("quantity", pa.float64()),
            ("status", pa.string()),
            ("fill_price", pa.float64()),
            ("conviction_score", pa.int64()),
            ("created_at", pa.string())
        ])
        table = pa.Table.from_pydict({col: [] for col in schema.names}, schema=schema)
    else:
        # Build columns
        columns = {
            "id": [t.get("id", "") for t in trades],
            "ticker": [t.get("ticker", "") for t in trades],
            "side": [t.get("side", "") for t in trades],
            "quantity": [float(t.get("quantity", 0) or 0) for t in trades],
            "order_type": [t.get("order_type", "") for t in trades],
            "status": [t.get("status", "") for t in trades],
            "fill_price": [float(t.get("fill_price", 0) or 0) for t in trades],
            "fill_quantity": [float(t.get("fill_quantity", 0) or 0) for t in trades],
            "conviction_score": [int(t.get("conviction_score", 0) or 0) for t in trades],
            "thesis": [t.get("thesis", "") or "" for t in trades],
            "rejection_reason": [t.get("rejection_reason", "") or "" for t in trades],
            "created_at": [t.get("created_at", "") for t in trades],
            "filled_at": [t.get("filled_at", "") or "" for t in trades],
            "alpaca_order_id": [t.get("alpaca_order_id", "") or "" for t in trades]
        }

        table = pa.Table.from_pydict(columns)

    # Write to bytes buffer
    output = io.BytesIO()
    pq.write_table(table, output)
    return output.getvalue()


def export_signals_parquet(signals: List[Dict[str, Any]]) -> bytes:
    """
    Export signals to Parquet format.

    Args:
        signals: List of signal dicts from database

    Returns:
        Parquet file content as bytes
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")

    if not signals:
        schema = pa.schema([
            ("id", pa.string()),
            ("source", pa.string()),
            ("ticker", pa.string()),
            ("signal_type", pa.string()),
            ("timestamp_utc", pa.string()),
            ("data_json", pa.string())
        ])
        table = pa.Table.from_pydict({col: [] for col in schema.names}, schema=schema)
    else:
        columns = {
            "id": [s.get("id", "") for s in signals],
            "source": [s.get("source", "") for s in signals],
            "ticker": [s.get("ticker", "") or "" for s in signals],
            "signal_type": [s.get("signal_type", "") for s in signals],
            "timestamp_utc": [s.get("timestamp_utc", "") for s in signals],
            "staleness_seconds": [int(s.get("staleness_seconds", 0) or 0) for s in signals],
            "data_json": [json.dumps(s.get("data", {})) if isinstance(s.get("data"), dict) else str(s.get("data", "")) for s in signals]
        }

        table = pa.Table.from_pydict(columns)

    output = io.BytesIO()
    pq.write_table(table, output)
    return output.getvalue()


def export_positions_parquet(positions: List[Dict[str, Any]]) -> bytes:
    """
    Export positions to Parquet format.

    Args:
        positions: List of position dicts

    Returns:
        Parquet file content as bytes
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")

    if not positions:
        schema = pa.schema([
            ("ticker", pa.string()),
            ("quantity", pa.float64()),
            ("avg_entry_price", pa.float64()),
            ("current_price", pa.float64()),
            ("market_value", pa.float64()),
            ("unrealized_pnl", pa.float64())
        ])
        table = pa.Table.from_pydict({col: [] for col in schema.names}, schema=schema)
    else:
        columns = {
            "ticker": [p.get("ticker", "") for p in positions],
            "quantity": [float(p.get("quantity", 0) or 0) for p in positions],
            "avg_entry_price": [float(p.get("avg_entry_price", 0) or 0) for p in positions],
            "current_price": [float(p.get("current_price", 0) or 0) for p in positions],
            "market_value": [float(p.get("market_value", 0) or 0) for p in positions],
            "unrealized_pnl": [float(p.get("unrealized_pnl", 0) or 0) for p in positions],
            "unrealized_pnl_pct": [float(p.get("unrealized_pnl_pct", 0) or 0) for p in positions],
            "entry_date": [p.get("entry_date", "") or "" for p in positions],
            "thesis": [p.get("thesis", "") or "" for p in positions]
        }

        table = pa.Table.from_pydict(columns)

    output = io.BytesIO()
    pq.write_table(table, output)
    return output.getvalue()


# =============================================================================
# COMBINED EXPORTS
# =============================================================================

def export_full_dataset(
    db,
    format: str = "csv",
    include_trades: bool = True,
    include_signals: bool = True,
    include_positions: bool = True,
    include_analyses: bool = True
) -> Dict[str, Any]:
    """
    Export full dataset from database.

    Args:
        db: Database instance
        format: "csv" or "parquet"
        include_*: What to include

    Returns:
        Dict mapping filename -> content
    """
    exports = {}

    if include_trades:
        trades = db.get_recent_trades(limit=1000)
        if format == "csv":
            exports["trades.csv"] = export_trades_csv(trades)
        else:
            exports["trades.parquet"] = export_trades_parquet(trades)

    if include_signals:
        signals = db.get_recent_signals(limit=1000)
        if format == "csv":
            exports["signals.csv"] = export_signals_csv(signals)
        else:
            exports["signals.parquet"] = export_signals_parquet(signals)

    if include_positions:
        positions = db.get_positions()
        if format == "csv":
            exports["positions.csv"] = export_positions_csv(positions)
        else:
            exports["positions.parquet"] = export_positions_parquet(positions)

    if include_analyses:
        analyses = db.get_recent_analyses(limit=1000) if hasattr(db, 'get_recent_analyses') else []
        if format == "csv":
            exports["analyses.csv"] = export_analyses_csv(analyses)

    return exports


def generate_export_filename(data_type: str, format: str) -> str:
    """
    Generate timestamped filename for export.

    Args:
        data_type: Type of data (trades, signals, etc.)
        format: File format (csv, parquet)

    Returns:
        Filename string
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"gst_{data_type}_{timestamp}.{format}"
