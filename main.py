"""
Gann Sentinel Trader - Main Entry Point with Logs API
Runs the trading agent and logs API server together.

Version: 2.1.0

This replaces the main entry point to include the logs API.
"""

import asyncio
import logging
import os
import sys
import threading
from typing import Optional

import uvicorn

from agent import GannSentinelAgent
from api.logs_api import app as logs_api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_logs_api(host: str = "0.0.0.0", port: int = 8080):
    """Run the logs API server in a separate thread."""
    config = uvicorn.Config(
        logs_api_app,
        host=host,
        port=port,
        log_level="warning",  # Reduce noise
        access_log=False
    )
    server = uvicorn.Server(config)
    server.run()


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("GANN SENTINEL TRADER v2.1.0")
    logger.info("Multi-Agent Consensus Architecture (MACA) Preview")
    logger.info("=" * 60)

    # Check for logs API token
    api_token = os.getenv("LOGS_API_TOKEN")
    if api_token:
        # Start logs API in background thread
        api_port = int(os.getenv("LOGS_API_PORT", "8080"))
        logger.info(f"Starting Logs API on port {api_port}...")

        api_thread = threading.Thread(
            target=run_logs_api,
            kwargs={"port": api_port},
            daemon=True
        )
        api_thread.start()
        logger.info(f"Logs API available at http://0.0.0.0:{api_port}/api/logs")
    else:
        logger.warning("LOGS_API_TOKEN not set - Logs API disabled")

    # Start the trading agent
    agent = GannSentinelAgent()

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        await agent.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await agent.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
