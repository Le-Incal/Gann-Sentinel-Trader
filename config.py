"""
Gann Sentinel Trader - Configuration
Loads environment variables and provides system-wide configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration for Gann Sentinel Trader."""
    
    # ==========================================================================
    # API CREDENTIALS
    # ==========================================================================
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # xAI (Grok)
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-3-fast-beta"  # Matches scanners/grok_scanner.py

    # Anthropic (Claude)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = "claude-3-5-sonnet-latest"  # Matches analyzers/claude_analyst.py
    
    # Alpaca
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # FRED
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    FRED_BASE_URL: str = "https://api.stlouisfed.org/fred"
    
    # ==========================================================================
    # SYSTEM SETTINGS
    # ==========================================================================
    
    # Mode: PAPER or LIVE
    MODE: str = os.getenv("MODE", "PAPER").upper()
    
    # Approval gate: ON or OFF
    APPROVAL_GATE: bool = os.getenv("APPROVAL_GATE", "ON").upper() == "ON"
    
    # Logging level
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # ==========================================================================
    # TRADING PARAMETERS
    # ==========================================================================
    
    # Position limits
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.25"))
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "5"))
    
    # Conviction threshold (0-100)
    MIN_CONVICTION: int = int(os.getenv("MIN_CONVICTION", "80"))
    
    # Risk limits
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.15"))
    DAILY_LOSS_LIMIT_PCT: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.05"))
    MIN_MARKET_CAP: int = int(os.getenv("MIN_MARKET_CAP", "500000000"))
    
    # ==========================================================================
    # STALENESS POLICIES (seconds)
    # ==========================================================================
    
    STALENESS_SENTIMENT: int = 3600      # 1 hour
    STALENESS_NEWS: int = 14400          # 4 hours
    STALENESS_MACRO: int = 86400         # 24 hours
    STALENESS_PREDICTION: int = 3600     # 1 hour
    
    # ==========================================================================
    # SCHEDULING
    # ==========================================================================
    
    SCAN_INTERVAL_MINUTES: int = 60      # Full scan every hour
    POSITION_CHECK_MINUTES: int = 15     # Check positions every 15 min
    
    # ==========================================================================
    # PATHS
    # ==========================================================================
    
    BASE_DIR: Path = Path(__file__).parent
    DATABASE_PATH: Path = BASE_DIR / "data" / "sentinel.db"
    LOG_PATH: Path = BASE_DIR / "logs"
    
    # ==========================================================================
    # FRED SERIES TO TRACK
    # ==========================================================================
    
    FRED_SERIES: list = [
        "DGS10",      # 10-Year Treasury Yield
        "DGS2",       # 2-Year Treasury Yield
        "T10Y2Y",     # 10Y-2Y Spread
        "UNRATE",     # Unemployment Rate
        "CPIAUCSL",   # CPI (Inflation)
        "GDP",        # Gross Domestic Product
        "FEDFUNDS",   # Federal Funds Rate
    ]
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    
    @classmethod
    def validate(cls) -> dict:
        """Validate that all required configuration is present."""
        issues = []
        
        # Check required API keys
        if not cls.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN is not set")
        if not cls.TELEGRAM_CHAT_ID:
            issues.append("TELEGRAM_CHAT_ID is not set")
        if not cls.XAI_API_KEY:
            issues.append("XAI_API_KEY is not set")
        if not cls.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY is not set")
        if not cls.ALPACA_API_KEY:
            issues.append("ALPACA_API_KEY is not set")
        if not cls.ALPACA_SECRET_KEY:
            issues.append("ALPACA_SECRET_KEY is not set")
        if not cls.FRED_API_KEY:
            issues.append("FRED_API_KEY is not set")
        
        # Check mode
        if cls.MODE not in ["PAPER", "LIVE"]:
            issues.append(f"Invalid MODE: {cls.MODE}. Must be PAPER or LIVE")
        
        # Warn about LIVE mode
        if cls.MODE == "LIVE" and cls.APPROVAL_GATE:
            pass  # This is fine - LIVE with approval gate
        elif cls.MODE == "LIVE" and not cls.APPROVAL_GATE:
            issues.append("WARNING: LIVE mode with APPROVAL_GATE=OFF is dangerous!")
        
        return {
            "valid": len([i for i in issues if not i.startswith("WARNING")]) == 0,
            "issues": issues
        }
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (hiding sensitive values)."""
        print("\n" + "=" * 60)
        print("GANN SENTINEL TRADER - CONFIGURATION")
        print("=" * 60)
        print(f"Mode:            {cls.MODE}")
        print(f"Approval Gate:   {'ON' if cls.APPROVAL_GATE else 'OFF'}")
        print(f"Log Level:       {cls.LOG_LEVEL}")
        print("-" * 60)
        print(f"Max Position:    {cls.MAX_POSITION_PCT * 100}%")
        print(f"Max Positions:   {cls.MAX_POSITIONS}")
        print(f"Min Conviction:  {cls.MIN_CONVICTION}")
        print(f"Stop Loss:       {cls.STOP_LOSS_PCT * 100}%")
        print(f"Daily Loss Lim:  {cls.DAILY_LOSS_LIMIT_PCT * 100}%")
        print("-" * 60)
        print(f"Telegram:        {'✓ Configured' if cls.TELEGRAM_BOT_TOKEN else '✗ Missing'}")
        print(f"Grok (xAI):      {'✓ Configured' if cls.XAI_API_KEY else '✗ Missing'}")
        print(f"Claude:          {'✓ Configured' if cls.ANTHROPIC_API_KEY else '✗ Missing'}")
        print(f"Alpaca:          {'✓ Configured' if cls.ALPACA_API_KEY else '✗ Missing'}")
        print(f"FRED:            {'✓ Configured' if cls.FRED_API_KEY else '✗ Missing'}")
        print("=" * 60 + "\n")


# Create directories if they don't exist
Config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.LOG_PATH.mkdir(parents=True, exist_ok=True)
