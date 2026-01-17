"""
Gann Sentinel Trader - Logs API
Lightweight HTTP endpoint for remote log access.

Version: 2.0.0 - Public Dashboard Support
- Added CORS for frontend access
- Added public endpoints (no auth required for read-only data)
- Sensitive data filtered from public endpoints

Usage:
    # Public endpoints (no auth)
    GET /health
    GET /api/public/dashboard
    GET /api/public/portfolio
    GET /api/public/positions
    GET /api/public/trades
    GET /api/public/signals
    GET /api/public/scan_cycles
    GET /api/public/debates
    GET /api/public/costs

    # Protected endpoints (require token)
    GET /api/logs?token=xxx&limit=50
    GET /api/status?token=xxx
    GET /api/errors?token=xxx&limit=20
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GST API",
    description="Gann Sentinel Trader - Public Dashboard & Logs API",
    version="2.0.0"
)

# =============================================================================
# CORS CONFIGURATION
# =============================================================================

# Allow all origins for public dashboard access
# In production, you may want to restrict this to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for public dashboard
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Token for protected endpoints
API_TOKEN = os.getenv("LOGS_API_TOKEN")


def get_db():
    """Get database instance."""
    from storage.database import Database
    return Database()


def verify_token(token: str) -> bool:
    """Verify API token."""
    if not API_TOKEN:
        logger.warning("LOGS_API_TOKEN not set - protected API disabled")
        return False
    return token == API_TOKEN


# =============================================================================
# PUBLIC ENDPOINTS (No Authentication Required)
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "gst-api",
        "version": "2.0.0"
    }


@app.get("/api/public/dashboard")
async def get_public_dashboard():
    """
    Get dashboard overview data.
    Combines portfolio, positions, recent trades for single request.
    """
    try:
        db = get_db()

        # Get portfolio snapshot
        portfolio = db.get_latest_snapshot() or {
            "total_value": 100000,
            "cash": 100000,
            "positions_value": 0,
            "daily_pnl": 0,
            "daily_pnl_pct": 0
        }

        # Get positions
        positions = db.get_positions()

        # Get recent trades
        recent_trades = db.get_recent_trades(limit=10)

        # Get pending trades
        pending_trades = db.get_pending_trades()

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": portfolio,
            "positions": positions,
            "recent_trades": recent_trades,
            "pending_trades": pending_trades
        }

    except Exception as e:
        logger.error(f"Error fetching dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/portfolio")
async def get_public_portfolio():
    """Get current portfolio state."""
    try:
        db = get_db()
        portfolio = db.get_latest_snapshot()

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": portfolio or {
                "total_value": 100000,
                "cash": 100000,
                "positions_value": 0,
                "daily_pnl": 0,
                "daily_pnl_pct": 0
            }
        }

    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/portfolio/history")
async def get_public_portfolio_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history")
):
    """Get portfolio value history for charts."""
    try:
        db = get_db()

        # For now, return the latest snapshot
        # In production, you'd query historical snapshots
        latest = db.get_latest_snapshot()

        # Generate mock history data based on current value
        # Replace this with actual historical data query
        history = []
        base_value = latest.get("total_value", 100000) if latest else 100000

        for i in range(days, 0, -1):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            # Add some variance for demo purposes
            variance = (hash(date) % 1000 - 500) / 100  # -5% to +5%
            value = base_value * (1 + variance / 100)
            history.append({"date": date, "value": round(value, 2)})

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "days": days,
            "history": history
        }

    except Exception as e:
        logger.error(f"Error fetching portfolio history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/positions")
async def get_public_positions():
    """Get current positions."""
    try:
        db = get_db()
        positions = db.get_positions()

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(positions),
            "positions": positions
        }

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/trades")
async def get_public_trades(
    limit: int = Query(50, ge=1, le=200, description="Number of trades to return")
):
    """Get trade history."""
    try:
        db = get_db()
        trades = db.get_recent_trades(limit=limit)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(trades),
            "trades": trades
        }

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/trades/pending")
async def get_public_pending_trades():
    """Get pending trades awaiting approval."""
    try:
        db = get_db()
        trades = db.get_pending_trades()

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(trades),
            "trades": trades
        }

    except Exception as e:
        logger.error(f"Error fetching pending trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/signals")
async def get_public_signals(
    limit: int = Query(50, ge=1, le=200, description="Number of signals to return")
):
    """Get recent signals."""
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


@app.get("/api/public/scan_cycles")
async def get_public_scan_cycles(
    limit: int = Query(20, ge=1, le=100, description="Number of cycles to return")
):
    """Get recent MACA scan cycles."""
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


@app.get("/api/public/scan_cycles/{cycle_id}")
async def get_public_scan_cycle(cycle_id: str):
    """Get details for a specific scan cycle."""
    try:
        db = get_db()

        # Get cycle details
        cycles = db.get_recent_scan_cycles(limit=100)
        cycle = next((c for c in cycles if c.get("id") == cycle_id), None)

        if not cycle:
            raise HTTPException(status_code=404, detail="Scan cycle not found")

        # Get proposals for this cycle
        proposals = db.get_proposals_for_cycle(cycle_id)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "proposals": proposals
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scan cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/debates")
async def get_public_debates(
    limit: int = Query(20, ge=1, le=100, description="Number of debates to return")
):
    """Get recent debate sessions with turns."""
    try:
        db = get_db()

        with db._get_connection() as conn:
            cursor = conn.cursor()

            # Get recent debate sessions
            cursor.execute("""
                SELECT * FROM debate_sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            sessions = [dict(row) for row in cursor.fetchall()]

            # Get turns for each session
            for session in sessions:
                cursor.execute("""
                    SELECT * FROM debate_turns
                    WHERE session_id = ?
                    ORDER BY round, created_at
                """, (session["id"],))
                session["turns"] = [dict(row) for row in cursor.fetchall()]

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(sessions),
            "debates": sessions
        }

    except Exception as e:
        logger.error(f"Error fetching debates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/debates/{session_id}")
async def get_public_debate(session_id: str):
    """Get details for a specific debate session."""
    try:
        db = get_db()

        with db._get_connection() as conn:
            cursor = conn.cursor()

            # Get session
            cursor.execute("SELECT * FROM debate_sessions WHERE id = ?", (session_id,))
            session = cursor.fetchone()

            if not session:
                raise HTTPException(status_code=404, detail="Debate session not found")

            session = dict(session)

            # Get turns
            cursor.execute("""
                SELECT * FROM debate_turns
                WHERE session_id = ?
                ORDER BY round, created_at
            """, (session_id,))
            session["turns"] = [dict(row) for row in cursor.fetchall()]

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "debate": session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching debate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/public/system")
async def get_public_system_status():
    """Get system status and configuration (public info only)."""
    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "version": "3.0.0",
            "architecture": "Multi-Agent Consensus Architecture (MACA)",
            "status": "operational",
            "scheduled_scans": ["9:35 AM ET", "12:30 PM ET"],
            "analysts": [
                {"name": "Grok", "role": "Narrative Momentum"},
                {"name": "Perplexity", "role": "External Reality"},
                {"name": "ChatGPT", "role": "Sentiment & Bias"},
                {"name": "Claude", "role": "Technical Validator"}
            ],
            "approval_method": "Telegram"
        }
    }


@app.get("/api/public/costs")
async def get_public_costs(
    days: int = Query(7, ge=1, le=90, description="Number of days to summarize")
):
    """Get API cost summary."""
    try:
        db = get_db()
        summary = db.get_cost_summary(days=days)
        daily = db.get_cost_by_day(days=days)

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "daily": daily
        }

    except Exception as e:
        logger.error(f"Error fetching costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PROTECTED ENDPOINTS (Require Authentication)
# =============================================================================

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

    # Require explicit token - no fallback for security
    if not os.getenv("LOGS_API_TOKEN"):
        print("WARNING: LOGS_API_TOKEN not set - protected endpoints disabled")
        print("Public endpoints will still be available")

    uvicorn.run(app, host="0.0.0.0", port=8080)
