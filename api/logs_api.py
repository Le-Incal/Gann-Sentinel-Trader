"""
Gann Sentinel Trader - Logs API
Lightweight HTTP endpoint for remote log access.

Version: 1.0.0

Usage:
    GET /api/logs?token=xxx&limit=50
    GET /api/status?token=xxx
    GET /api/errors?token=xxx&limit=20
    GET /health  (no auth required)
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# Import will be done after app initialization to avoid circular imports
# from storage.database import Database

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GST Logs API",
    description="Remote access to Gann Sentinel Trader logs",
    version="1.0.0"
)

# Token for authentication
API_TOKEN = os.getenv("LOGS_API_TOKEN")


def get_db():
    """Get database instance."""
    from storage.database import Database
    return Database()


def verify_token(token: str) -> bool:
    """Verify API token."""
    if not API_TOKEN:
        logger.warning("LOGS_API_TOKEN not set - API disabled")
        return False
    return token == API_TOKEN


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "gst-logs-api"
    }


@app.get("/api/logs")
async def get_logs(
    token: str = Query(..., description="API token"),
    limit: int = Query(50, ge=1, le=200, description="Number of logs to return"),
    direction: Optional[str] = Query(None, description="Filter: incoming or outgoing"),
    message_type: Optional[str] = Query(None, description="Filter by message type")
):
    """
    Get recent Telegram activity logs.

    Returns messages in reverse chronological order (newest first).
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        db = get_db()
        messages = db.get_telegram_messages(
            limit=limit,
            direction=direction,
            message_type=message_type
        )

        # Get summary stats
        summary = db.get_telegram_activity_summary(hours=24)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(messages),
            "summary_24h": summary,
            "logs": messages
        }

    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status(
    token: str = Query(..., description="API token")
):
    """
    Get system status overview.

    Includes portfolio, positions, pending trades, and recent activity.
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        db = get_db()

        # Get various status info
        positions = db.get_positions()
        pending_trades = db.get_pending_trades()
        recent_trades = db.get_recent_trades(limit=5)
        latest_snapshot = db.get_latest_snapshot()
        telegram_summary = db.get_telegram_activity_summary(hours=24)
        recent_errors = db.get_recent_errors(limit=5)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": latest_snapshot,
            "positions": positions,
            "pending_trades": pending_trades,
            "recent_trades": recent_trades,
            "telegram_activity_24h": telegram_summary,
            "recent_errors": recent_errors
        }

    except Exception as e:
        logger.error(f"Error fetching status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/errors")
async def get_errors(
    token: str = Query(..., description="API token"),
    limit: int = Query(20, ge=1, le=100, description="Number of errors to return")
):
    """Get recent system errors."""
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        db = get_db()
        errors = db.get_recent_errors(limit=limit)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(errors),
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Error fetching errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_signals(
    token: str = Query(..., description="API token"),
    limit: int = Query(50, ge=1, le=200, description="Number of signals to return")
):
    """Get recent signals."""
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        db = get_db()
        signals = db.get_recent_signals(limit=limit)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(signals),
            "signals": signals
        }

    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scan_cycles")
async def get_scan_cycles(
    token: str = Query(..., description="API token"),
    limit: int = Query(10, ge=1, le=50, description="Number of cycles to return")
):
    """Get recent MACA scan cycles."""
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        db = get_db()
        cycles = db.get_recent_scan_cycles(limit=limit)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(cycles),
            "scan_cycles": cycles
        }

    except Exception as e:
        logger.error(f"Error fetching scan cycles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STANDALONE RUNNER (for testing)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Set a default token for testing
    if not os.getenv("LOGS_API_TOKEN"):
        os.environ["LOGS_API_TOKEN"] = "test-token-123"
        print("WARNING: Using default test token")

    uvicorn.run(app, host="0.0.0.0", port=8080)
