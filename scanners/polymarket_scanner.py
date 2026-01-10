"""
Gann Sentinel Trader - Polymarket Scanner
Forward-looking prediction market signal extraction.

This scanner uses the shared temporal awareness framework to ensure
we always look projectively (1mo, 3mo, 6mo, 12mo forward).

Version: 2.0.0 (Temporal Awareness Update)
Last Updated: January 2026
"""

import os
import uuid
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import httpx

# Import shared temporal framework
from scanners.temporal import (
    TemporalContext,
    TemporalQueryBuilder,
    TimeHorizon,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


class MarketCategory(Enum):
    """
    Investment-focused categories for stock trading signals.

    Each category maps to specific tradeable assets and represents
    a distinct market-moving theme.
    """
    # Core monetary/macro
    FEDERAL_RESERVE = "federal_reserve"
    MACRO_ECONOMIC = "macro_economic"
    FISCAL_TREASURY = "fiscal_treasury"

    # Trade & geopolitics
    TRADE_POLICY = "trade_policy"
    GEOPOLITICAL = "geopolitical"
    CHINA_RISK = "china_risk"

    # Technology themes
    AI_SECTOR = "ai_sector"
    SEMICONDUCTOR = "semiconductor"
    TECH_GIANTS = "tech_giants"

    # Other sectors
    SPACE_INDUSTRY = "space_industry"
    ENERGY_COMMODITIES = "energy_commodities"
    HEALTHCARE_BIOTECH = "healthcare_biotech"

    # Corporate & regulatory
    CEO_EXECUTIVE = "ceo_executive"
    REGULATORY_LEGAL = "regulatory_legal"
    IPO_CAPITAL_MARKETS = "ipo_capital_markets"

    # Policy affecting equities
    LABOR_IMMIGRATION = "labor_immigration"
    CRYPTO_POLICY = "crypto_policy"

    # Catch-all (filtered out by default)
    OTHER = "other"


class SignalPurpose(Enum):
    """Purpose classification for how signals inform trading decisions."""
    HYPOTHESIS_GENERATOR = "hypothesis_generator"  # Novel info that could spark a trade idea
    SENTIMENT_VALIDATOR = "sentiment_validator"    # Crowd odds to check against existing thesis
    CATALYST_TIMING = "catalyst_timing"            # Tells us WHEN something resolves


# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================

CATEGORY_KEYWORDS = {
    # -------------------------------------------------------------------------
    # FEDERAL RESERVE - Rate decisions, Fed policy
    # -------------------------------------------------------------------------
    MarketCategory.FEDERAL_RESERVE: [
        "fed", "federal reserve", "fomc", "rate cut", "rate hike",
        "interest rate", "powell", "jerome powell", "monetary policy",
        "basis points", "quantitative tightening", "quantitative easing",
        "qt", "qe", "fed funds", "federal funds rate", "fomc meeting",
        "dot plot", "fed pivot", "hawkish", "dovish", "terminal rate"
    ],

    # -------------------------------------------------------------------------
    # MACRO ECONOMIC - GDP, recession, inflation
    # -------------------------------------------------------------------------
    MarketCategory.MACRO_ECONOMIC: [
        "gdp", "inflation", "cpi", "pce", "unemployment", "jobs report",
        "nonfarm payroll", "payroll", "recession", "soft landing",
        "hard landing", "economic growth", "retail sales", "consumer spending",
        "manufacturing", "pmi", "ism", "consumer confidence", "stagflation",
        "deflation", "disinflation", "yield curve", "inverted yield"
    ],

    # -------------------------------------------------------------------------
    # FISCAL & TREASURY - Government spending, debt
    # -------------------------------------------------------------------------
    MarketCategory.FISCAL_TREASURY: [
        "debt ceiling", "government shutdown", "treasury", "treasury auction",
        "deficit", "fiscal policy", "spending bill", "budget", "appropriations",
        "continuing resolution", "national debt", "bond auction", "t-bill",
        "treasury yield", "sovereign debt", "fiscal stimulus", "debt limit"
    ],

    # -------------------------------------------------------------------------
    # TRADE POLICY - Tariffs, sanctions, trade deals
    # -------------------------------------------------------------------------
    MarketCategory.TRADE_POLICY: [
        "tariff", "trump tariff", "trade war", "trade deal", "sanctions",
        "import duty", "export ban", "trade agreement", "nafta", "usmca",
        "wto", "trade deficit", "protectionism", "dumping", "countervailing",
        "section 301", "section 232", "most favored nation", "trade representative"
    ],

    # -------------------------------------------------------------------------
    # GEOPOLITICAL - Wars, conflicts, international relations
    # -------------------------------------------------------------------------
    MarketCategory.GEOPOLITICAL: [
        "war", "conflict", "invasion", "military", "nato", "eu", "european union",
        "brexit", "russia", "ukraine", "middle east", "iran", "israel",
        "north korea", "nuclear", "missile", "defense", "pentagon",
        "state department", "diplomacy", "treaty", "alliance", "sanctions"
    ],

    # -------------------------------------------------------------------------
    # CHINA RISK - Taiwan, ADRs, chip bans, stimulus
    # -------------------------------------------------------------------------
    MarketCategory.CHINA_RISK: [
        "taiwan", "taiwan invasion", "china invasion", "pla", "ccp",
        "adr delisting", "china adr", "hkex", "chinese stocks",
        "chip ban", "export control", "entity list", "huawei",
        "china stimulus", "pboc", "xi jinping", "chinese communist party",
        "south china sea", "one china", "tsmc taiwan", "baba", "alibaba",
        "pdd", "pinduoduo", "jd", "nio", "byd", "tencent"
    ],

    # -------------------------------------------------------------------------
    # AI SECTOR - AI companies, models, regulation
    # -------------------------------------------------------------------------
    MarketCategory.AI_SECTOR: [
        "openai", "anthropic", "claude", "chatgpt", "gpt-5", "gpt5",
        "xai", "x.ai", "grok", "google ai", "gemini", "deepmind", "bard",
        "microsoft copilot", "copilot", "meta ai", "llama", "mistral",
        "ai regulation", "artificial intelligence", "machine learning",
        "large language model", "llm", "generative ai", "gen ai",
        "ai safety", "agi", "artificial general intelligence",
        "sam altman", "dario amodei", "demis hassabis", "ai chips",
        "ai infrastructure", "ai training", "inference", "transformer"
    ],

    # -------------------------------------------------------------------------
    # SEMICONDUCTOR - Chips, fabs, data centers
    # -------------------------------------------------------------------------
    MarketCategory.SEMICONDUCTOR: [
        "semiconductor", "chip", "nvidia", "nvda", "amd", "intel", "intc",
        "tsmc", "taiwan semiconductor", "asml", "lam research", "lrcx",
        "applied materials", "amat", "micron", "mu", "qualcomm", "qcom",
        "broadcom", "avgo", "data center", "gpu", "cpu", "hbm",
        "high bandwidth memory", "fab", "foundry", "chips act",
        "blackwell", "hopper", "h100", "h200", "b100", "b200",
        "grace", "cuda", "tensor core", "ai accelerator", "tpu",
        "chip shortage", "wafer", "lithography", "euv", "packaging",
        "cowos", "jensen huang", "pat gelsinger", "lisa su"
    ],

    # -------------------------------------------------------------------------
    # TECH GIANTS - FAANG+, enterprise tech
    # -------------------------------------------------------------------------
    MarketCategory.TECH_GIANTS: [
        "google", "alphabet", "googl", "goog", "microsoft", "msft",
        "amazon", "amzn", "aws", "meta", "facebook", "apple", "aapl",
        "netflix", "nflx", "tesla", "tsla", "oracle", "orcl",
        "salesforce", "crm", "adobe", "adbe", "ibm", "cisco", "csco",
        "cloud computing", "azure", "google cloud", "gcp",
        "sundar pichai", "satya nadella", "tim cook", "andy jassy",
        "mark zuckerberg", "zuck", "elon musk tesla"
    ],

    # -------------------------------------------------------------------------
    # SPACE INDUSTRY - SpaceX, rockets, satellites
    # -------------------------------------------------------------------------
    MarketCategory.SPACE_INDUSTRY: [
        "spacex", "rocket lab", "rklb", "blue origin", "nasa",
        "starlink", "starship", "falcon", "falcon 9", "falcon heavy",
        "neutron rocket", "electron rocket", "satellite", "orbit",
        "launch", "rocket", "space station", "iss", "artemis",
        "lunar", "moon", "mars", "gwynne shotwell", "peter beck",
        "jeff bezos space", "ula", "united launch alliance",
        "northrop grumman space", "lockheed martin space", "boeing space",
        "space force", "gps satellite", "leo", "geo", "constellation"
    ],

    # -------------------------------------------------------------------------
    # ENERGY & COMMODITIES - Oil, gas, renewables
    # -------------------------------------------------------------------------
    MarketCategory.ENERGY_COMMODITIES: [
        "opec", "oil price", "crude oil", "wti", "brent", "natural gas",
        "lng", "petroleum", "exxon", "xom", "chevron", "cvx",
        "occidental", "oxy", "conocophillips", "cop", "bp", "shell",
        "ev mandate", "electric vehicle", "renewable", "solar", "wind",
        "energy policy", "drilling", "fracking", "pipeline", "refinery",
        "gasoline", "diesel", "energy transition", "clean energy",
        "enphase", "enph", "solaredge", "sedg", "first solar", "fslr",
        "nextera", "nee", "plug power", "plug", "hydrogen"
    ],

    # -------------------------------------------------------------------------
    # HEALTHCARE & BIOTECH - FDA, drugs, healthcare policy
    # -------------------------------------------------------------------------
    MarketCategory.HEALTHCARE_BIOTECH: [
        "fda", "fda approval", "drug approval", "clinical trial",
        "phase 3", "phase 2", "pdufa", "nda", "bla", "medicare",
        "medicaid", "healthcare reform", "aca", "obamacare", "vaccine",
        "pandemic", "epidemic", "outbreak", "pfizer", "pfe", "moderna",
        "mrna", "merck", "mrk", "eli lilly", "lly", "johnson & johnson",
        "jnj", "abbvie", "abbv", "bristol myers", "bmy", "amgen", "amgn",
        "gilead", "gild", "regeneron", "regn", "biogen", "biib",
        "biotech", "pharmaceutical", "drug pricing", "cms", "unh",
        "unitedhealth", "managed care", "health insurance"
    ],

    # -------------------------------------------------------------------------
    # CEO & EXECUTIVE - Major CEO actions and decisions
    # -------------------------------------------------------------------------
    MarketCategory.CEO_EXECUTIVE: [
        "elon musk", "musk", "jensen huang", "satya nadella", "sundar pichai",
        "mark zuckerberg", "zuckerberg", "tim cook", "andy jassy",
        "sam altman", "dario amodei", "lisa su", "pat gelsinger",
        "jamie dimon", "warren buffett", "brian moynihan", "david solomon",
        "ceo steps down", "ceo resigns", "ceo fired", "ceo appointed",
        "executive departure", "succession", "founder", "board of directors",
        "activist investor", "proxy fight", "shareholder vote"
    ],

    # -------------------------------------------------------------------------
    # REGULATORY & LEGAL - Antitrust, SEC, lawsuits
    # -------------------------------------------------------------------------
    MarketCategory.REGULATORY_LEGAL: [
        "antitrust", "ftc", "sec", "doj", "department of justice",
        "lawsuit", "ruling", "court", "verdict", "settlement",
        "injunction", "monopoly", "anticompetitive", "merger blocked",
        "acquisition blocked", "consent decree", "fine", "penalty",
        "investigation", "subpoena", "enforcement", "compliance",
        "lina khan", "gary gensler", "breakup", "divestiture",
        "class action", "securities fraud", "insider trading"
    ],

    # -------------------------------------------------------------------------
    # IPO & CAPITAL MARKETS - IPOs, SPACs, offerings
    # -------------------------------------------------------------------------
    MarketCategory.IPO_CAPITAL_MARKETS: [
        "ipo", "initial public offering", "direct listing", "spac",
        "secondary offering", "follow-on offering", "valuation",
        "public offering", "going public", "private placement",
        "pre-ipo", "ipo pricing", "ipo pop", "lock-up expiration",
        "spacex ipo", "stripe ipo", "openai ipo", "reddit ipo",
        "instacart ipo", "arm ipo", "unicorn", "decacorn"
    ],

    # -------------------------------------------------------------------------
    # LABOR & IMMIGRATION - Strikes, H1B, workforce
    # -------------------------------------------------------------------------
    MarketCategory.LABOR_IMMIGRATION: [
        "strike", "union", "uaw", "united auto workers", "teamsters",
        "labor union", "collective bargaining", "walkout", "picket",
        "h1b", "h-1b", "visa", "immigration", "work visa", "green card",
        "minimum wage", "wage increase", "labor shortage", "hiring freeze",
        "layoffs", "rif", "workforce reduction", "remote work",
        "return to office", "rto", "work from home", "wfh"
    ],

    # -------------------------------------------------------------------------
    # CRYPTO POLICY - Bitcoin ETF, stablecoin, regulation (equity impact)
    # -------------------------------------------------------------------------
    MarketCategory.CRYPTO_POLICY: [
        "bitcoin etf", "spot bitcoin", "ethereum etf", "crypto etf",
        "stablecoin", "stablecoin regulation", "crypto regulation",
        "sec crypto", "cftc crypto", "coinbase", "coin",
        "microstrategy", "mstr", "bitcoin price", "btc price",
        "crypto exchange", "defi regulation", "crypto custody",
        "crypto mining", "bitcoin mining", "mara", "riot", "clsk",
        "digital assets", "cbdc", "central bank digital currency"
    ],
}


# =============================================================================
# EXCLUSION FILTERS - Markets to ignore
# =============================================================================

EXCLUDED_KEYWORDS = [
    # Sports
    "nfl", "nba", "mlb", "nhl", "mls", "soccer", "football game",
    "basketball game", "baseball game", "hockey", "super bowl",
    "world series", "stanley cup", "nba finals", "march madness",
    "world cup", "olympics", "olympic", "championship game",
    "playoff", "playoffs", "mvp", "touchdown", "home run",
    "three pointer", "goalkeeper", "quarterback", "pitcher",
    "boxing", "ufc", "mma", "wrestling", "tennis", "golf tournament",
    "grand slam", "wimbledon", "us open tennis", "pga", "lpga",
    "formula 1", "f1 race", "nascar", "indycar", "le mans",

    # Entertainment
    "grammy", "oscar", "emmy", "tony award", "golden globe",
    "academy award", "billboard", "mtv", "vma", "bet awards",
    "reality tv", "bachelor", "bachelorette", "survivor",
    "american idol", "the voice", "dancing with the stars",
    "celebrity", "kardashian", "influencer", "tiktoker", "youtuber",
    "movie box office", "film premiere", "album release", "concert",
    "tour dates", "netflix show", "hbo show", "disney plus",

    # Non-market personal
    "dating", "marriage", "divorce", "wedding", "baby", "pregnant",
    "relationship", "affair", "scandal gossip",

    # Pure sports betting language
    "point spread", "over under", "moneyline", "parlay", "odds maker",
    "vegas odds", "betting line", "sportsbook",

    # Weather (unless market-relevant)
    "hurricane name", "tropical storm name", "weather forecast",
]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PolymarketSignal:
    """Signal extracted from Polymarket conforming to Grok Spec v1.1.0."""
    signal_id: str
    dedup_hash: str
    category: str  # prediction_market (per spec)
    source_type: str  # polymarket

    # Asset scope
    asset_scope: Dict[str, List[str]]

    # Signal content
    summary: str
    raw_value: Dict[str, Any]
    evidence: List[Dict[str, Any]]

    # Scoring and metadata
    confidence: float
    confidence_factors: Dict[str, float]
    directional_bias: str
    time_horizon: str
    novelty: str

    # Staleness
    staleness_policy: Dict[str, Any]

    # Uncertainties
    uncertainties: List[str]

    # Timestamps
    timestamp_utc: str

    # Our additions for internal tracking
    market_category: str  # Our internal categorization (one of 17 categories)
    resolution_date: Optional[str] = None

    # Signal purpose classification
    signal_purpose: str = "hypothesis_generator"  # hypothesis_generator, sentiment_validator, catalyst_timing

    # Probability momentum tracking
    probability_24h_prior: Optional[float] = None
    probability_change_24h: Optional[float] = None
    momentum_flag: Optional[str] = None  # "rapid_rise", "rapid_fall", "stable", None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "dedup_hash": self.dedup_hash,
            "signal_type": self.category,  # For database compatibility
            "category": self.category,
            "source_type": self.source_type,
            "source": self.source_type,  # For database compatibility
            "asset_scope": self.asset_scope,
            "summary": self.summary,
            "raw_value": self.raw_value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "directional_bias": self.directional_bias,
            "time_horizon": self.time_horizon,
            "novelty": self.novelty,
            "staleness_policy": self.staleness_policy,
            "staleness_seconds": self.staleness_policy.get("max_age_seconds", 3600),
            "uncertainties": self.uncertainties,
            "timestamp_utc": self.timestamp_utc,
            "market_category": self.market_category,
            "resolution_date": self.resolution_date,
            # New fields for focused trading signals
            "signal_purpose": self.signal_purpose,
            "probability_24h_prior": self.probability_24h_prior,
            "probability_change_24h": self.probability_change_24h,
            "momentum_flag": self.momentum_flag,
        }


# =============================================================================
# POLYMARKET SCANNER
# =============================================================================

class PolymarketScanner:
    """
    Scanner for Polymarket prediction markets.
    
    Uses the shared temporal framework to ensure consistent forward-looking
    date windows across all scanners.
    
    Time Windows (from temporal.py):
    - IMMEDIATE: 7 days
    - SHORT_TERM: 30 days
    - MEDIUM_TERM: 90 days  
    - LONG_TERM: 180 days
    - EXTENDED: 365 days
    """
    
    def __init__(self):
        """Initialize the Polymarket scanner."""
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.markets_endpoint = f"{self.gamma_url}/markets"
        self.events_endpoint = f"{self.gamma_url}/events"
        
        # HTTP client settings
        self.timeout = 30.0
        self.max_retries = 3
        
        # Initialize shared temporal framework
        self.temporal_context = get_temporal_context()
        self.query_builder = TemporalQueryBuilder(self.temporal_context)
        
        # Cache for seen markets (deduplication)
        self._seen_markets: Dict[str, datetime] = {}
        
        # Log temporal context
        self.temporal_context.log_context()
        logger.info("PolymarketScanner initialized with shared temporal framework")
    
    # =========================================================================
    # DATE WINDOW CALCULATION (delegates to temporal framework)
    # =========================================================================
    
    def _get_current_date(self) -> datetime:
        """Get current UTC datetime from temporal context."""
        return self.temporal_context.now
    
    def _get_date_window(self, horizon: TimeHorizon) -> Tuple[datetime, datetime]:
        """Get date window for a specific horizon."""
        return self.temporal_context.get_window(horizon)
    
    def _format_date_for_api(self, dt: datetime) -> str:
        """Format datetime for Polymarket API (YYYY-MM-DD)."""
        return self.temporal_context.format_date(dt)
    
    # =========================================================================
    # MARKET CATEGORIZATION
    # =========================================================================
    
    def _is_excluded_market(self, combined_text: str) -> bool:
        """
        Check if a market should be excluded (sports, entertainment, etc.).

        Args:
            combined_text: Combined question + description + title text

        Returns:
            True if market should be excluded
        """
        for excluded_keyword in EXCLUDED_KEYWORDS:
            if excluded_keyword in combined_text:
                logger.debug(f"Excluding market - matched '{excluded_keyword}'")
                return True
        return False

    def _categorize_market(self, market: Dict[str, Any]) -> MarketCategory:
        """
        Categorize a market based on its title and description.

        Uses multi-pass matching:
        1. First check exclusion list (sports, entertainment)
        2. Then match against specific categories (most specific first)
        3. Fall back to OTHER if no match

        Args:
            market: Market data from Polymarket API

        Returns:
            MarketCategory enum value
        """
        # Get text to analyze
        question = (market.get("question") or "").lower()
        description = (market.get("description") or "").lower()
        title = (market.get("title") or "").lower()

        combined_text = f"{question} {description} {title}"

        # PASS 1: Check exclusion list first
        if self._is_excluded_market(combined_text):
            return MarketCategory.OTHER

        # PASS 2: Match against categories in priority order
        # More specific categories should be checked before broader ones
        category_priority = [
            # Most specific first
            MarketCategory.CHINA_RISK,       # China-specific before general geopolitical
            MarketCategory.TRADE_POLICY,     # Tariffs before general geopolitical
            MarketCategory.CEO_EXECUTIVE,    # CEO-specific before company categories
            MarketCategory.AI_SECTOR,        # AI before general tech
            MarketCategory.SEMICONDUCTOR,    # Chips before general tech
            MarketCategory.SPACE_INDUSTRY,   # Space before general defense
            MarketCategory.IPO_CAPITAL_MARKETS,
            MarketCategory.REGULATORY_LEGAL,
            MarketCategory.HEALTHCARE_BIOTECH,
            MarketCategory.ENERGY_COMMODITIES,
            MarketCategory.LABOR_IMMIGRATION,
            MarketCategory.CRYPTO_POLICY,
            MarketCategory.FEDERAL_RESERVE,
            MarketCategory.FISCAL_TREASURY,
            MarketCategory.MACRO_ECONOMIC,
            MarketCategory.TECH_GIANTS,      # Broader tech last
            MarketCategory.GEOPOLITICAL,     # Broader geopolitical last
        ]

        # Score each category by number of keyword matches
        category_scores: Dict[MarketCategory, int] = {}

        for category in category_priority:
            keywords = CATEGORY_KEYWORDS.get(category, [])
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    score += 1
            if score > 0:
                category_scores[category] = score

        # Return highest scoring category, respecting priority order for ties
        if category_scores:
            # Sort by score descending, then by priority order
            best_category = max(
                category_scores.keys(),
                key=lambda c: (category_scores[c], -category_priority.index(c))
            )
            return best_category

        return MarketCategory.OTHER
    
    def _map_category_to_time_horizon(
        self,
        category: MarketCategory,
        resolution_date: Optional[datetime]
    ) -> str:
        """Map category and resolution date to time horizon string."""
        if resolution_date:
            horizon = self.temporal_context.classify_future_date(resolution_date)
            if horizon:
                return horizon.signal_horizon_label
            return "unknown"  # Date is in the past

        # Default based on category when no resolution date
        short_term_categories = [
            MarketCategory.FEDERAL_RESERVE,
            MarketCategory.MACRO_ECONOMIC,
            MarketCategory.FISCAL_TREASURY,
            MarketCategory.LABOR_IMMIGRATION,
        ]

        medium_term_categories = [
            MarketCategory.TRADE_POLICY,
            MarketCategory.AI_SECTOR,
            MarketCategory.SEMICONDUCTOR,
            MarketCategory.TECH_GIANTS,
            MarketCategory.ENERGY_COMMODITIES,
            MarketCategory.HEALTHCARE_BIOTECH,
            MarketCategory.REGULATORY_LEGAL,
            MarketCategory.IPO_CAPITAL_MARKETS,
            MarketCategory.CRYPTO_POLICY,
            MarketCategory.CEO_EXECUTIVE,
        ]

        long_term_categories = [
            MarketCategory.GEOPOLITICAL,
            MarketCategory.CHINA_RISK,
            MarketCategory.SPACE_INDUSTRY,
        ]

        if category in short_term_categories:
            return "weeks"
        elif category in medium_term_categories:
            return "months"
        elif category in long_term_categories:
            return "quarters"
        else:
            return "unknown"
    
    def _category_to_asset_scope(self, category: MarketCategory) -> Dict[str, List[str]]:
        """
        Map market category to relevant asset scope with specific tradeable tickers.

        Each category maps to:
        - Primary tickers: Direct plays on the theme
        - Secondary/ETF tickers: Sector or thematic exposure
        """
        scope_map = {
            # -----------------------------------------------------------------
            # FEDERAL RESERVE - Rate-sensitive assets
            # -----------------------------------------------------------------
            MarketCategory.FEDERAL_RESERVE: {
                "tickers": ["SPY", "TLT", "IEF", "XLF", "KRE"],
                "sectors": ["FINANCIALS", "REAL_ESTATE"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY", "FIXED_INCOME"],
            },

            # -----------------------------------------------------------------
            # MACRO ECONOMIC - Broad market, cyclicals
            # -----------------------------------------------------------------
            MarketCategory.MACRO_ECONOMIC: {
                "tickers": ["SPY", "IWM", "TLT", "XLY", "XLP"],
                "sectors": ["CONSUMER_DISCRETIONARY", "CONSUMER_STAPLES"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY", "FIXED_INCOME"],
            },

            # -----------------------------------------------------------------
            # FISCAL & TREASURY - Bonds, financials
            # -----------------------------------------------------------------
            MarketCategory.FISCAL_TREASURY: {
                "tickers": ["TLT", "IEF", "SHY", "XLF", "JPM", "BAC"],
                "sectors": ["FINANCIALS"],
                "macro_regions": ["US"],
                "asset_classes": ["FIXED_INCOME", "EQUITY"],
            },

            # -----------------------------------------------------------------
            # TRADE POLICY - Varies by target country
            # -----------------------------------------------------------------
            MarketCategory.TRADE_POLICY: {
                "tickers": ["EEM", "FXI", "EWJ", "SPY", "CAT", "DE"],
                "sectors": ["INDUSTRIALS", "MATERIALS"],
                "macro_regions": ["US", "ASIA", "GLOBAL"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # GEOPOLITICAL - Defense, commodities, safe havens
            # -----------------------------------------------------------------
            MarketCategory.GEOPOLITICAL: {
                "tickers": ["LMT", "RTX", "NOC", "GD", "GLD", "USO", "VIX"],
                "sectors": ["DEFENSE", "ENERGY"],
                "macro_regions": ["GLOBAL"],
                "asset_classes": ["EQUITY", "COMMODITY"],
            },

            # -----------------------------------------------------------------
            # CHINA RISK - ADRs, chip equipment, Taiwan
            # -----------------------------------------------------------------
            MarketCategory.CHINA_RISK: {
                "tickers": ["TSM", "BABA", "PDD", "JD", "NIO", "ASML", "LRCX", "AMAT", "FXI", "KWEB"],
                "sectors": ["TECH", "CONSUMER"],
                "macro_regions": ["ASIA", "CHINA"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # AI SECTOR - AI infrastructure and beneficiaries
            # -----------------------------------------------------------------
            MarketCategory.AI_SECTOR: {
                "tickers": ["NVDA", "MSFT", "GOOGL", "META", "AMD", "SMCI", "ARM", "PLTR"],
                "secondary_tickers": ["SMH", "BOTZ", "AIQ", "ROBO"],
                "sectors": ["TECH", "SEMICONDUCTORS"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # SEMICONDUCTOR - Chip makers and equipment
            # -----------------------------------------------------------------
            MarketCategory.SEMICONDUCTOR: {
                "tickers": ["NVDA", "AMD", "INTC", "TSM", "ASML", "LRCX", "AMAT", "MU", "QCOM", "AVGO"],
                "secondary_tickers": ["SMH", "SOXX"],
                "sectors": ["SEMICONDUCTORS"],
                "macro_regions": ["US", "ASIA"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # TECH GIANTS - FAANG+ and enterprise
            # -----------------------------------------------------------------
            MarketCategory.TECH_GIANTS: {
                "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NFLX", "TSLA", "ORCL", "CRM", "ADBE"],
                "secondary_tickers": ["QQQ", "XLK", "VGT"],
                "sectors": ["TECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # SPACE INDUSTRY - Launch providers and defense
            # -----------------------------------------------------------------
            MarketCategory.SPACE_INDUSTRY: {
                "tickers": ["RKLB", "LMT", "NOC", "BA", "GSAT", "IRDM"],
                "secondary_tickers": ["UFO", "ARKX"],
                "sectors": ["AEROSPACE", "DEFENSE"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # ENERGY & COMMODITIES - Oil, gas, renewables
            # -----------------------------------------------------------------
            MarketCategory.ENERGY_COMMODITIES: {
                "tickers": ["XOM", "CVX", "OXY", "COP", "SLB", "ENPH", "SEDG", "FSLR", "NEE"],
                "secondary_tickers": ["XLE", "USO", "TAN", "ICLN"],
                "sectors": ["ENERGY", "UTILITIES"],
                "macro_regions": ["US", "GLOBAL"],
                "asset_classes": ["EQUITY", "COMMODITY"],
            },

            # -----------------------------------------------------------------
            # HEALTHCARE & BIOTECH - Pharma and biotech
            # -----------------------------------------------------------------
            MarketCategory.HEALTHCARE_BIOTECH: {
                "tickers": ["PFE", "MRNA", "MRK", "LLY", "JNJ", "ABBV", "BMY", "AMGN", "GILD", "UNH"],
                "secondary_tickers": ["XLV", "IBB", "XBI"],
                "sectors": ["HEALTHCARE", "BIOTECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # CEO & EXECUTIVE - Company-specific based on CEO
            # -----------------------------------------------------------------
            MarketCategory.CEO_EXECUTIVE: {
                "tickers": ["TSLA", "NVDA", "MSFT", "GOOGL", "META", "AAPL", "AMZN", "JPM", "BRK.B"],
                "sectors": ["TECH", "FINANCIALS"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # REGULATORY & LEGAL - Varies by case
            # -----------------------------------------------------------------
            MarketCategory.REGULATORY_LEGAL: {
                "tickers": ["GOOGL", "META", "AAPL", "AMZN", "MSFT"],
                "sectors": ["TECH"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # IPO & CAPITAL MARKETS - Adjacent sector plays
            # -----------------------------------------------------------------
            MarketCategory.IPO_CAPITAL_MARKETS: {
                "tickers": ["RKLB", "COIN", "HOOD", "SQ", "PYPL"],  # Adjacencies to big IPOs
                "secondary_tickers": ["IPO", "XLF"],
                "sectors": ["TECH", "FINANCIALS"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # LABOR & IMMIGRATION - Affected companies
            # -----------------------------------------------------------------
            MarketCategory.LABOR_IMMIGRATION: {
                "tickers": ["F", "GM", "STLA", "UPS", "FDX", "WMT", "TGT", "MCD"],
                "sectors": ["INDUSTRIALS", "CONSUMER"],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # CRYPTO POLICY - Crypto-adjacent equities
            # -----------------------------------------------------------------
            MarketCategory.CRYPTO_POLICY: {
                "tickers": ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "SQ", "PYPL"],
                "sectors": ["CRYPTO", "FINTECH"],
                "macro_regions": ["US", "GLOBAL"],
                "asset_classes": ["EQUITY"],
            },

            # -----------------------------------------------------------------
            # OTHER - Generic fallback
            # -----------------------------------------------------------------
            MarketCategory.OTHER: {
                "tickers": [],
                "sectors": [],
                "macro_regions": ["US"],
                "asset_classes": ["EQUITY"],
            },
        }

        return scope_map.get(category, scope_map[MarketCategory.OTHER])
    
    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================
    
    def _calculate_confidence(
        self,
        market: Dict[str, Any],
        category: MarketCategory
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate confidence score per Grok Spec v1.1.0 formula.
        
        confidence = source_base × recency_factor × corroboration_factor
        
        Returns:
            Tuple of (final_confidence, confidence_factors_dict)
        """
        # Source base: Polymarket is a tier-2 prediction market
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)
        
        # Higher volume = more reliable signal
        if volume >= 100000:  # >$100k volume
            source_base = 0.75
        elif volume >= 50000:  # $50k-$100k
            source_base = 0.60
        elif volume >= 10000:  # $10k-$50k
            source_base = 0.45
        else:
            source_base = 0.30
        
        # Recency factor based on last trade
        # Since we're querying active markets, assume recent
        recency_factor = 0.95
        
        # Corroboration: Single source (Polymarket)
        corroboration_factor = 1.0
        
        # Calculate final (capped at 1.0)
        confidence = min(source_base * recency_factor * corroboration_factor, 1.0)
        
        factors = {
            "source_base": source_base,
            "recency_factor": recency_factor,
            "corroboration_factor": corroboration_factor,
        }
        
        return confidence, factors
    
    # =========================================================================
    # DIRECTIONAL BIAS DETERMINATION
    # =========================================================================
    
    def _determine_directional_bias(
        self,
        market: Dict[str, Any],
        category: MarketCategory
    ) -> str:
        """
        Determine directional bias based on probability and category.
        
        For Fed rate cuts: Higher probability of cut = positive for equities
        For recession: Higher probability = negative for equities
        etc.
        """
        # Get current probability
        prob = self._extract_probability(market)
        if prob is None:
            return "unclear"
        
        question = (market.get("question") or "").lower()
        
        # Fed rate cuts - cuts are generally positive for equities
        if "rate cut" in question:
            return "positive" if prob >= 0.5 else "negative"
        
        # Fed rate hikes - hikes are generally negative for equities
        if "rate hike" in question or "rate increase" in question:
            return "negative" if prob >= 0.5 else "positive"
        
        # Recession - clearly negative
        if "recession" in question:
            return "negative" if prob >= 0.5 else "positive"
        
        # Tariffs/trade war - generally negative
        if "tariff" in question or "trade war" in question:
            return "negative" if prob >= 0.5 else "mixed"
        
        # Default: significant moves in either direction = mixed
        if prob >= 0.7:
            return "positive"
        elif prob <= 0.3:
            return "negative"
        else:
            return "mixed"
    
    def _extract_probability(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract current probability from market data."""
        # Try various fields where probability might be
        for field in ["outcomePrices", "lastTradePrice", "bestBid", "bestAsk"]:
            if field in market and market[field]:
                try:
                    if isinstance(market[field], list):
                        # Usually [YES_price, NO_price]
                        return float(market[field][0])
                    elif isinstance(market[field], (int, float)):
                        return float(market[field])
                    elif isinstance(market[field], str):
                        return float(market[field])
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_prior_probability(self, market: Dict[str, Any]) -> Optional[float]:
        """Extract 24h prior probability if available."""
        # Try to get historical price data from market
        # Polymarket API may include this in some responses
        for field in ["previousDayPrice", "price24hAgo", "historicalPrices"]:
            if field in market and market[field]:
                try:
                    if isinstance(market[field], list) and len(market[field]) > 0:
                        return float(market[field][0])
                    elif isinstance(market[field], (int, float)):
                        return float(market[field])
                except (ValueError, IndexError, TypeError):
                    continue
        return None

    # =========================================================================
    # PROBABILITY MOMENTUM & SIGNAL PURPOSE
    # =========================================================================

    def _calculate_probability_momentum(
        self,
        current_prob: Optional[float],
        prior_prob: Optional[float]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate probability momentum and flag rapid movements.

        Rapid movements (±10% in 24h) often precede news or indicate
        informed money moving the market.

        Args:
            current_prob: Current probability (0-1)
            prior_prob: Probability 24h ago (0-1)

        Returns:
            Tuple of (change_24h, momentum_flag)
            momentum_flag: "rapid_rise", "rapid_fall", "stable", or None
        """
        if current_prob is None or prior_prob is None:
            return None, None

        change = current_prob - prior_prob

        # Flag rapid movements (±10% in 24h)
        if change >= 0.10:
            momentum_flag = "rapid_rise"
        elif change <= -0.10:
            momentum_flag = "rapid_fall"
        elif abs(change) < 0.02:
            momentum_flag = "stable"
        else:
            momentum_flag = None  # Normal movement

        return change, momentum_flag

    def _determine_signal_purpose(
        self,
        market: Dict[str, Any],
        category: MarketCategory,
        resolution_date: Optional[datetime],
        momentum_flag: Optional[str]
    ) -> str:
        """
        Determine the purpose of this signal for trading decisions.

        Signal purposes:
        - hypothesis_generator: Novel information that could spark a trade idea
        - sentiment_validator: Crowd odds useful to check against existing thesis
        - catalyst_timing: Tells us WHEN something resolves (entry timing)

        Args:
            market: Raw market data
            category: Categorized market category
            resolution_date: When the market resolves
            momentum_flag: Whether odds are moving rapidly

        Returns:
            SignalPurpose value as string
        """
        question = (market.get("question") or "").lower()
        volume = float(market.get("volume", 0) or 0)

        # Catalyst timing: Markets with specific resolution dates
        # These help with entry/exit timing
        if resolution_date:
            now = self._get_current_date()
            days_to_resolution = (resolution_date - now).days

            # Near-term catalysts (within 30 days) are timing signals
            if 0 < days_to_resolution <= 30:
                return SignalPurpose.CATALYST_TIMING.value

        # Hypothesis generators: Novel, high-volume markets that could
        # inform a new trade thesis
        hypothesis_categories = [
            MarketCategory.IPO_CAPITAL_MARKETS,  # IPOs create new opportunities
            MarketCategory.CEO_EXECUTIVE,        # CEO changes create volatility
            MarketCategory.REGULATORY_LEGAL,     # Legal outcomes create binary events
            MarketCategory.TRADE_POLICY,         # Policy changes create sector moves
            MarketCategory.CHINA_RISK,           # Geopolitical shifts
        ]

        if category in hypothesis_categories and volume >= 50000:
            return SignalPurpose.HYPOTHESIS_GENERATOR.value

        # Rapid momentum also suggests hypothesis-worthy signals
        if momentum_flag in ["rapid_rise", "rapid_fall"] and volume >= 25000:
            return SignalPurpose.HYPOTHESIS_GENERATOR.value

        # Default: sentiment validators - useful for checking consensus
        return SignalPurpose.SENTIMENT_VALIDATOR.value

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_dedup_hash(
        self, 
        source_type: str, 
        question: str,
        primary_ticker_or_region: str
    ) -> str:
        """Generate deduplication hash per Grok Spec v1.1.0."""
        normalized = f"{source_type}:{primary_ticker_or_region}:{question.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _market_to_signal(
        self, 
        market: Dict[str, Any],
        window_name: str
    ) -> Optional[PolymarketSignal]:
        """
        Convert a Polymarket market to a standardized signal.
        
        Args:
            market: Raw market data from API
            window_name: Which time window this came from
            
        Returns:
            PolymarketSignal or None if invalid
        """
        try:
            # Extract basic info
            question = market.get("question") or market.get("title") or ""
            if not question:
                logger.debug("Skipping market with no question/title")
                return None
            
            market_id = market.get("id") or market.get("conditionId") or str(uuid.uuid4())
            
            # Categorize
            category = self._categorize_market(market)
            
            # Skip OTHER category if not interesting
            if category == MarketCategory.OTHER:
                logger.debug(f"Skipping OTHER category market: {question[:50]}")
                return None
            
            # Asset scope
            asset_scope = self._category_to_asset_scope(category)
            primary_identifier = (
                asset_scope["tickers"][0] if asset_scope["tickers"]
                else asset_scope["macro_regions"][0] if asset_scope["macro_regions"]
                else "US"
            )
            
            # Generate IDs
            signal_id = str(uuid.uuid4())
            dedup_hash = self._generate_dedup_hash("polymarket", question, primary_identifier)
            
            # Check dedup cache
            if dedup_hash in self._seen_markets:
                last_seen = self._seen_markets[dedup_hash]
                if (self._get_current_date() - last_seen).total_seconds() < 3600:
                    logger.debug(f"Skipping duplicate: {question[:50]}")
                    return None
            
            self._seen_markets[dedup_hash] = self._get_current_date()
            
            # Extract probability
            current_prob = self._extract_probability(market)
            prior_prob = self._extract_prior_probability(market)

            # Calculate probability momentum
            prob_change_24h, momentum_flag = self._calculate_probability_momentum(
                current_prob, prior_prob
            )

            raw_value = {
                "type": "probability",
                "value": current_prob,
                "unit": "percent",
                "prior_value": prior_prob,
                "change": prob_change_24h,
                "change_period": "24h" if prior_prob else None,
            }

            # Parse resolution date
            resolution_date = None
            end_date_str = market.get("endDate") or market.get("endDateIso") or market.get("closedTime")
            if end_date_str:
                try:
                    resolution_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Time horizon
            time_horizon = self._map_category_to_time_horizon(category, resolution_date)

            # Confidence
            confidence, confidence_factors = self._calculate_confidence(market, category)

            # Directional bias
            directional_bias = self._determine_directional_bias(market, category)

            # Determine signal purpose
            signal_purpose = self._determine_signal_purpose(
                market, category, resolution_date, momentum_flag
            )

            # Build summary with momentum indicator
            prob_str = f"{current_prob*100:.0f}%" if current_prob else "unknown"
            momentum_indicator = ""
            if momentum_flag == "rapid_rise":
                momentum_indicator = " [RISING FAST]"
            elif momentum_flag == "rapid_fall":
                momentum_indicator = " [FALLING FAST]"
            summary = f"Polymarket: {question} Currently at {prob_str}.{momentum_indicator}"
            
            # Evidence
            evidence = [{
                "source": f"https://polymarket.com/market/{market_id}",
                "source_tier": "tier2",
                "excerpt": question,
                "timestamp_utc": self._get_current_date().isoformat(),
            }]
            
            # Volume context for uncertainties
            volume = float(market.get("volume", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)
            
            uncertainties = []
            if volume < 50000:
                uncertainties.append(f"Low volume market (${volume:,.0f})")
            if liquidity < 10000:
                uncertainties.append(f"Low liquidity (${liquidity:,.0f})")
            if not resolution_date:
                uncertainties.append("Resolution date unclear")
            
            # Staleness policy - prediction markets are fast-moving
            now = self._get_current_date()
            staleness_policy = {
                "max_age_seconds": 3600,  # 1 hour
                "stale_after_utc": (now + timedelta(hours=1)).isoformat(),
            }
            
            # Novelty
            novelty = "developing" if dedup_hash in self._seen_markets else "new"
            
            return PolymarketSignal(
                signal_id=signal_id,
                dedup_hash=dedup_hash,
                category="prediction_market",
                source_type="polymarket",
                asset_scope=asset_scope,
                summary=summary,
                raw_value=raw_value,
                evidence=evidence,
                confidence=confidence,
                confidence_factors=confidence_factors,
                directional_bias=directional_bias,
                time_horizon=time_horizon,
                novelty=novelty,
                staleness_policy=staleness_policy,
                uncertainties=uncertainties,
                timestamp_utc=now.isoformat(),
                market_category=category.value,
                resolution_date=resolution_date.isoformat() if resolution_date else None,
                # New focused trading fields
                signal_purpose=signal_purpose,
                probability_24h_prior=prior_prob,
                probability_change_24h=prob_change_24h,
                momentum_flag=momentum_flag,
            )
            
        except Exception as e:
            logger.error(f"Error converting market to signal: {e}")
            return None
    
    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _fetch_markets(
        self,
        end_date_min: str,
        end_date_max: str,
        active: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Polymarket Gamma API with date filtering.
        
        Args:
            end_date_min: Minimum resolution date (YYYY-MM-DD)
            end_date_max: Maximum resolution date (YYYY-MM-DD)
            active: Only fetch active markets
            limit: Max results
            
        Returns:
            List of market dicts
        """
        params = {
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
            "active": str(active).lower(),
            "closed": "false",
            "limit": limit,
            "order": "volume",  # Most active first
            "ascending": "false",
        }
        
        logger.debug(f"Fetching markets: {end_date_min} to {end_date_max}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.markets_endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Fetched {len(data)} markets from Polymarket")
                    return data if isinstance(data, list) else []
                else:
                    logger.error(f"Polymarket API error: {response.status_code}")
                    return []
                    
        except httpx.TimeoutException:
            logger.error("Polymarket API timeout")
            return []
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
            return []
    
    async def _fetch_events_by_tag(
        self,
        tag: str,
        end_date_min: str,
        end_date_max: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fetch events filtered by tag (e.g., 'fed', 'economics').
        
        Args:
            tag: Tag to filter by
            end_date_min: Minimum resolution date
            end_date_max: Maximum resolution date
            limit: Max results
            
        Returns:
            List of event dicts with nested markets
        """
        params = {
            "tag": tag,
            "end_date_min": end_date_min,
            "end_date_max": end_date_max,
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.events_endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"Fetched {len(data)} events for tag '{tag}'")
                    return data if isinstance(data, list) else []
                else:
                    logger.warning(f"Events API error for tag '{tag}': {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Events API error: {e}")
            return []
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    async def scan_all_markets(self) -> List[PolymarketSignal]:
        """
        Scan all relevant Polymarket markets across time windows.
        
        This is the main entry point for the agent.
        
        Returns:
            List of PolymarketSignal objects
        """
        logger.info("Starting Polymarket scan with shared temporal framework...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Use MEDIUM_TERM (3 months) as the main window
        start, end = self._get_date_window(TimeHorizon.MEDIUM_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        logger.info(f"Scanning markets resolving between {start_str} and {end_str}")
        
        # Fetch markets
        markets = await self._fetch_markets(
            end_date_min=start_str,
            end_date_max=end_str,
            limit=100,
        )
        
        # Convert to signals
        for market in markets:
            signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
            if signal:
                signals.append(signal)
        
        # Fetch by investment-focused tags organized by category
        # These tags are specifically chosen to find stock-relevant markets
        relevant_tags = [
            # Federal Reserve & Macro
            "fed", "fomc", "interest-rates", "inflation", "recession",
            "economics", "gdp", "unemployment",

            # Trade & Geopolitics
            "tariffs", "trade", "china", "sanctions", "taiwan",

            # Technology & AI
            "ai", "artificial-intelligence", "tech", "nvidia",
            "semiconductors", "chips",

            # Space Industry
            "spacex", "space", "rockets",

            # Energy & Commodities
            "oil", "energy", "opec", "renewable",

            # Healthcare & Biotech
            "fda", "healthcare", "pharma", "biotech",

            # Corporate & Regulatory
            "antitrust", "sec", "ipo", "crypto",

            # Policy & Government
            "trump", "policy", "regulation", "debt-ceiling",
        ]

        for tag in relevant_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=20,
                )
                
                for event in events:
                    # Events have nested markets
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
                        if signal:
                            # Dedupe by hash
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)
                                
            except Exception as e:
                logger.warning(f"Error fetching tag '{tag}': {e}")
                continue
        
        logger.info(f"Polymarket scan complete: {len(signals)} signals generated")
        return signals
    
    async def scan_fed_markets(self) -> List[PolymarketSignal]:
        """
        Specifically scan Fed/monetary policy markets.
        
        Returns:
            List of Fed-related signals
        """
        logger.info("Scanning Fed/monetary policy markets...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Look 6 months out for Fed decisions
        start, end = self._get_date_window(TimeHorizon.LONG_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        # Try Fed-specific tags
        fed_tags = ["fed", "fomc", "federal-reserve", "interest-rates"]
        
        for tag in fed_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=30,
                )
                
                for event in events:
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.LONG_TERM.name)
                        if signal and signal.market_category == MarketCategory.FEDERAL_RESERVE.value:
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)

            except Exception as e:
                logger.warning(f"Error fetching Fed tag '{tag}': {e}")
                continue

        logger.info(f"Fed market scan complete: {len(signals)} signals")
        return signals
    
    async def scan_economic_events(self) -> List[PolymarketSignal]:
        """
        Scan for economic event markets (GDP, CPI, jobs, etc.).
        
        Returns:
            List of economic event signals
        """
        logger.info("Scanning economic event markets...")
        
        # Refresh temporal context
        self.temporal_context = get_temporal_context()
        
        signals: List[PolymarketSignal] = []
        
        # Economic data tends to be shorter term
        start, end = self._get_date_window(TimeHorizon.MEDIUM_TERM)
        
        start_str = self._format_date_for_api(start)
        end_str = self._format_date_for_api(end)
        
        econ_tags = ["economics", "gdp", "inflation", "unemployment", "recession"]
        
        for tag in econ_tags:
            try:
                events = await self._fetch_events_by_tag(
                    tag=tag,
                    end_date_min=start_str,
                    end_date_max=end_str,
                    limit=20,
                )
                
                for event in events:
                    event_markets = event.get("markets", [])
                    for market in event_markets:
                        signal = self._market_to_signal(market, TimeHorizon.MEDIUM_TERM.name)
                        if signal and signal.market_category == MarketCategory.MACRO_ECONOMIC.value:
                            if not any(s.dedup_hash == signal.dedup_hash for s in signals):
                                signals.append(signal)

            except Exception as e:
                logger.warning(f"Error fetching econ tag '{tag}': {e}")
                continue
        
        logger.info(f"Economic events scan complete: {len(signals)} signals")
        return signals


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_polymarket() -> List[Dict[str, Any]]:
    """
    Convenience function to run a full Polymarket scan.
    
    Returns:
        List of signal dictionaries ready for storage
    """
    scanner = PolymarketScanner()
    signals = await scanner.scan_all_markets()
    return [s.to_dict() for s in signals]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def test():
        scanner = PolymarketScanner()
        
        print("\n" + "="*60)
        print("POLYMARKET SCANNER TEST")
        print("="*60)
        
        # Show date windows
        windows = scanner._calculate_date_windows()
        print("\nForward-Looking Date Windows:")
        for name, (start, end) in windows.items():
            print(f"  {name}: {start.date()} to {end.date()}")
        
        print("\n" + "-"*60)
        print("Scanning markets...")
        print("-"*60)
        
        signals = await scanner.scan_all_markets()
        
        print(f"\nFound {len(signals)} signals:")
        for signal in signals[:5]:
            print(f"\n  Category: {signal.market_category}")
            print(f"  Summary: {signal.summary[:80]}...")
            print(f"  Probability: {signal.raw_value.get('value')}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Bias: {signal.directional_bias}")
            print(f"  Horizon: {signal.time_horizon}")
        
        if len(signals) > 5:
            print(f"\n  ... and {len(signals) - 5} more")
    
    asyncio.run(test())
