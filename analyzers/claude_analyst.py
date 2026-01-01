"""
Gann Sentinel Trader - Claude Analyst
Forward-Predictive Reasoning Engine with Historical Context

This is the strategic brain of the system. Claude anchors analysis in 
HISTORICAL PATTERNS while orienting toward FUTURE CATALYSTS.

CORE PRINCIPLE: Anchor in history, orient toward the future
- "When X happened before, Y followed"
- "Given historical patterns + what's coming, how do we position NOW?"

The analyst combines:
1. Historical pattern recognition (analogous events, cycles, technicals)
2. Forward-looking signals (catalysts, sentiment, predictions)
3. Second-order thinking (non-obvious beneficiaries)

Version: 2.1.0 (Historical Context Update)
Last Updated: January 2026
"""

import os
import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import httpx

# Import temporal framework
from scanners.temporal import (
    TemporalContext,
    TimeHorizon,
    get_temporal_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Recommendation(Enum):
    """Trade recommendations."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NONE = "NONE"


class AnalysisType(Enum):
    """Types of analysis."""
    TRADE = "TRADE"           # Actionable trade opportunity
    NO_TRADE = "NO_TRADE"     # Analyzed but no opportunity
    WATCH = "WATCH"           # Interesting but not yet actionable


# =============================================================================
# HISTORICAL PATTERN KNOWLEDGE BASE
# =============================================================================

# Historical analogues: "When X happened, Y followed"
HISTORICAL_PATTERNS = {
    # Fed Rate Cycles
    "fed_rate_cut_cycle": {
        "description": "Historical Fed rate cutting cycles",
        "analogues": [
            {"period": "2019", "event": "Fed cuts 3x", "outcome": "Small caps +25% in 6 months, growth outperformed value"},
            {"period": "2001-2003", "event": "Fed cuts from 6.5% to 1%", "outcome": "Bottom came 6 months after first cut, tech led recovery"},
            {"period": "2007-2008", "event": "Fed cuts into recession", "outcome": "Defensive sectors outperformed until bottom"},
        ],
        "typical_beneficiaries": ["IWM", "XLF", "XHB", "TLT"],
        "typical_duration": "6-12 months",
    },
    
    "fed_rate_hike_pause": {
        "description": "Fed pauses after hiking cycle",
        "analogues": [
            {"period": "2006", "event": "Fed pauses at 5.25%", "outcome": "Market rallied 15% over next 12 months before recession"},
            {"period": "2018-2019", "event": "Fed pauses then pivots", "outcome": "Growth stocks led 30%+ rally"},
        ],
        "typical_beneficiaries": ["QQQ", "SPY", "XLK"],
        "typical_duration": "6-18 months until recession or reacceleration",
    },
    
    # Inflation Cycles
    "inflation_cooling": {
        "description": "CPI trending down toward target",
        "analogues": [
            {"period": "2023", "event": "CPI fell from 9% to 3%", "outcome": "Growth stocks outperformed, duration assets rallied"},
            {"period": "1982-1983", "event": "Volcker breaks inflation", "outcome": "Massive bull market began"},
        ],
        "typical_beneficiaries": ["TLT", "QQQ", "XLK", "ARKK"],
        "typical_duration": "Trend persists 12-24 months",
    },
    
    # Sector IPO Spillover
    "major_ipo_sector_attention": {
        "description": "Major IPO brings attention to entire sector",
        "analogues": [
            {"period": "2021", "event": "Rivian IPO", "outcome": "EV sector rallied 20%+ in anticipation, LCID, NIO benefited"},
            {"period": "2020", "event": "Snowflake IPO", "outcome": "Cloud/SaaS sector attention, comparable companies rallied"},
            {"period": "2019", "event": "Beyond Meat IPO", "outcome": "Plant-based sector attention, Tyson and others moved"},
        ],
        "typical_beneficiaries": ["Comparable public companies in same sector"],
        "typical_duration": "2-4 weeks pre-IPO, 1-2 weeks post",
    },
    
    # Chip Restrictions
    "china_chip_restrictions": {
        "description": "US restricts chip exports to China",
        "analogues": [
            {"period": "2022", "event": "Biden chip restrictions", "outcome": "NVDA/AMD sold off 10-15%, recovered in 60 days"},
            {"period": "2019", "event": "Huawei ban", "outcome": "Semis volatile, domestic alternatives rallied"},
        ],
        "typical_beneficiaries": ["INTC", "TXN", "ON", "domestic fabs"],
        "pattern": "Initial selloff in exposed names, recovery within 60 days, domestic alternatives benefit",
    },
    
    # Earnings Patterns
    "nvidia_earnings_reaction": {
        "description": "NVIDIA earnings reaction patterns",
        "analogues": [
            {"period": "2023-2024", "event": "Beat and raise", "outcome": "Sector-wide rally, AMD/SMCI/AVGO followed within 48 hours"},
            {"period": "2022", "event": "Guidance cut", "outcome": "Sector-wide selloff, 2-week recovery"},
        ],
        "typical_beneficiaries": ["AMD", "SMCI", "AVGO", "MRVL", "TSM"],
        "pattern": "AMD moves same direction as NVDA within 24-48 hours post-earnings",
    },
    
    # Yield Curve
    "yield_curve_inversion": {
        "description": "10Y-2Y yield curve inversion/uninversion",
        "analogues": [
            {"period": "2019", "event": "Curve inverts", "outcome": "Recession 12-18 months later"},
            {"period": "2006", "event": "Curve inverts", "outcome": "Recession 18 months later"},
            {"period": "Historical", "event": "Curve uninverts", "outcome": "Recession typically follows 6-12 months AFTER uninversion"},
        ],
        "typical_implications": "Uninversion often signals recession approaching, not avoiding it",
        "watch_for": "Steepening from inversion is a warning, not all-clear",
    },
    
    # Seasonal Patterns
    "january_effect": {
        "description": "Small caps tend to outperform in January",
        "typical_beneficiaries": ["IWM", "IJR", "VB"],
        "pattern": "Historically strongest in first 2 weeks of January",
    },
    
    "sell_in_may": {
        "description": "May-October historically weaker than Nov-April",
        "pattern": "Not reliable every year, but worth noting seasonal headwind",
    },
    
    "santa_claus_rally": {
        "description": "Last 5 trading days of year + first 2 of new year",
        "pattern": "Historically positive, failure is bearish signal for Q1",
    },
    
    # Crisis Patterns
    "crisis_playbook": {
        "description": "Historical crisis market behavior",
        "analogues": [
            {"period": "2020 COVID", "event": "Exogenous shock", "outcome": "V-bottom, Fed/fiscal response drove recovery, growth led"},
            {"period": "2008 GFC", "event": "Financial crisis", "outcome": "Extended bottom, financials led down, quality/defensive first up"},
            {"period": "2022", "event": "Inflation/rate shock", "outcome": "Duration selloff, value beat growth, energy led"},
        ],
        "pattern": "Crisis type determines leadership: exogenous=growth, financial=quality, inflation=value",
    },
    
    # Technical Patterns
    "breakout_from_range": {
        "description": "Stock breaks out of multi-month trading range",
        "pattern": "Breakouts with volume tend to continuation, measure move = range height",
        "watch_for": "Retest of breakout level is healthy, failure = false breakout",
    },
    
    "support_bounce": {
        "description": "Stock bounces off established support level",
        "pattern": "Multiple touches of support = stronger level, entry near support reduces risk",
    },
}


# Second-order effect mappings
# When X happens, Y benefits
SECOND_ORDER_EFFECTS = {
    # Space sector
    "spacex_ipo": ["RKLB", "LMT", "RTX", "BA"],
    "spacex_starship": ["RKLB", "LMT"],
    "nasa_budget": ["RKLB", "LMT", "NOC"],
    
    # AI/Chips
    "nvidia_earnings": ["AMD", "SMCI", "AVGO", "MRVL"],
    "ai_regulation": ["GOOGL", "MSFT", "META"],
    "chip_shortage": ["TSM", "ASML", "AMAT", "LRCX"],
    "china_chip_ban": ["INTC", "TXN"],  # Domestic alternatives
    
    # Fed/Rates
    "fed_cut": ["TLT", "IWM", "XLF", "XHB"],
    "fed_hike": ["SHY", "XLU"],
    "fed_pause": ["SPY", "QQQ"],
    
    # Crypto
    "bitcoin_etf": ["COIN", "MSTR", "SQ"],
    "crypto_regulation": ["COIN"],
    
    # EV/Energy
    "tesla_earnings": ["RIVN", "LCID", "NIO"],
    "ev_subsidies": ["TSLA", "RIVN", "F", "GM"],
    "oil_spike": ["XOM", "CVX", "OXY", "XLE"],
    
    # Retail/Consumer
    "holiday_sales": ["AMZN", "WMT", "TGT", "XRT"],
    "consumer_sentiment": ["XLY", "XRT"],
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExitTrigger:
    """Defines when to exit a position."""
    trigger_type: str  # price_target, stop_loss, time_based, thesis_breaker
    value: Any
    description: str


@dataclass
class HistoricalContext:
    """Historical patterns informing the analysis."""
    analogous_event: str          # What historical event this resembles
    historical_period: str        # When it happened (e.g., "2019", "2008 GFC")
    historical_outcome: str       # What happened after
    pattern_confidence: str       # high, medium, low
    key_differences: List[str]    # How current situation differs
    rhymes_with: str              # Brief description of the pattern


@dataclass
class Analysis:
    """
    Claude's analysis output with historical context and forward-predictive reasoning.
    
    This represents Claude's strategic thinking that:
    1. ANCHORS in historical patterns
    2. ORIENTS toward future catalysts
    3. Applies second-order thinking
    """
    analysis_id: str
    timestamp_utc: datetime
    
    # Core recommendation
    ticker: Optional[str]
    recommendation: Recommendation
    conviction_score: int  # 0-100
    
    # Historical context (NEW)
    historical_context: Optional[HistoricalContext]
    technical_context: str        # Where is price vs historical range
    macro_cycle_position: str     # Where are we in the business cycle
    seasonal_factors: str         # Any seasonal patterns at play
    
    # Forward-predictive reasoning
    thesis: str
    catalyst: str                    # What event drives this trade
    catalyst_date: Optional[str]     # When the catalyst occurs
    catalyst_horizon: str            # immediate/days/weeks/months
    
    # Second-order thinking
    primary_beneficiary: bool        # Is this the obvious play?
    second_order_rationale: str      # Why this benefits from the catalyst
    
    # Bull/Bear cases
    bull_case: str
    bear_case: str
    variant_perception: str          # What we see that others don't
    
    # Trade parameters
    position_size_pct: float
    entry_strategy: str              # market, limit, scale_in
    entry_price_target: Optional[float]
    stop_loss_pct: float
    
    # Exit strategy
    exit_triggers: List[ExitTrigger]
    thesis_breakers: List[str]       # What would invalidate the thesis
    time_horizon: str
    
    # Context
    signals_used: List[str]
    market_context: str
    
    @property
    def is_actionable(self) -> bool:
        """Check if this analysis produces an actionable trade."""
        return (
            self.recommendation in [Recommendation.BUY, Recommendation.SELL] and
            self.conviction_score >= 80 and
            self.ticker is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        historical_dict = None
        if self.historical_context:
            historical_dict = {
                "analogous_event": self.historical_context.analogous_event,
                "historical_period": self.historical_context.historical_period,
                "historical_outcome": self.historical_context.historical_outcome,
                "pattern_confidence": self.historical_context.pattern_confidence,
                "key_differences": self.historical_context.key_differences,
                "rhymes_with": self.historical_context.rhymes_with,
            }
        
        return {
            "analysis_id": self.analysis_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "ticker": self.ticker,
            "recommendation": self.recommendation.value,
            "conviction_score": self.conviction_score,
            # Historical context
            "historical_context": historical_dict,
            "technical_context": self.technical_context,
            "macro_cycle_position": self.macro_cycle_position,
            "seasonal_factors": self.seasonal_factors,
            # Forward reasoning
            "thesis": self.thesis,
            "catalyst": self.catalyst,
            "catalyst_date": self.catalyst_date,
            "catalyst_horizon": self.catalyst_horizon,
            "primary_beneficiary": self.primary_beneficiary,
            "second_order_rationale": self.second_order_rationale,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "variant_perception": self.variant_perception,
            "position_size_pct": self.position_size_pct,
            "entry_strategy": self.entry_strategy,
            "entry_price_target": self.entry_price_target,
            "stop_loss_pct": self.stop_loss_pct,
            "exit_triggers": [
                {"type": t.trigger_type, "value": t.value, "description": t.description}
                for t in self.exit_triggers
            ],
            "thesis_breakers": self.thesis_breakers,
            "time_horizon": self.time_horizon,
            "signals_used": self.signals_used,
            "market_context": self.market_context,
        }


# =============================================================================
# CLAUDE ANALYST
# =============================================================================

class ClaudeAnalyst:
    """
    Forward-Predictive Reasoning Engine with Historical Context.
    
    The analyst:
    1. ANCHORS in historical patterns (what happened before)
    2. ORIENTS toward future catalysts (what's coming)
    3. Applies second-order thinking (non-obvious plays)
    4. Synthesizes into high-conviction trade recommendations
    
    Like a seasoned trader: "This reminds me of Q4 2018 when... 
    and back then, the play was..."
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude Analyst."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - analysis disabled")
        
        self.model = "claude-sonnet-4-20250514"
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # Load temporal context
        self.temporal_context = get_temporal_context()
        
        # Load historical knowledge
        self.historical_patterns = HISTORICAL_PATTERNS
        self.second_order_effects = SECOND_ORDER_EFFECTS
        
        logger.info(f"Claude Analyst initialized with model: {self.model}")
        logger.info(f"Loaded {len(self.historical_patterns)} historical patterns")
    
    def refresh_temporal_context(self) -> None:
        """Refresh temporal context (call at start of each analysis)."""
        self.temporal_context = get_temporal_context()
    
    # =========================================================================
    # PROMPT CONSTRUCTION
    # =========================================================================
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for Claude.
        
        This prompt establishes:
        - Historical pattern recognition
        - Forward-predictive mindset
        - Second-order thinking
        - Catalyst-driven analysis
        """
        now = self.temporal_context.now
        end_of_month = self.temporal_context.format_date(self.temporal_context.end_of_month)
        end_of_quarter = self.temporal_context.format_date(self.temporal_context.end_of_quarter)
        
        # Get current month for seasonal context
        current_month = now.strftime("%B")
        current_quarter = f"Q{(now.month - 1) // 3 + 1}"
        
        return f"""You are a senior trading strategist for an autonomous trading system, combining 
historical market wisdom with forward-predictive analysis.

CURRENT DATE: {self.temporal_context.format_date(now, '%B %d, %Y')}
KEY DATES:
- End of Month: {end_of_month}
- End of Quarter: {end_of_quarter}
- Current Month: {current_month}
- Current Quarter: {current_quarter}

=== CORE METHODOLOGY ===

ANCHOR in HISTORY, ORIENT toward the FUTURE.

Like a seasoned trader, you think: "This reminds me of [historical period] when 
[similar conditions existed], and back then [outcome]. Given that pattern plus 
[current catalyst], the play is [recommendation]."

1. HISTORICAL PATTERN RECOGNITION
   - Identify analogous historical situations
   - What happened AFTER similar setups?
   - How does current situation rhyme with history?
   - Key question: "When has this happened before, and what followed?"
   
   Examples:
   - "This Fed pivot setup looks like 2019, when small caps outperformed 25% in 6 months"
   - "NVDA earnings reactions historically lift AMD within 48 hours"
   - "Major sector IPOs bring attention to comparable public companies"

2. TECHNICAL CONTEXT
   - Where is price relative to historical ranges?
   - Key support/resistance levels
   - Is the stock at the top, middle, or bottom of its range?
   - Pattern setups (breakout, support bounce, etc.)

3. MACRO CYCLE POSITION
   - Where are we in the business cycle? (early, mid, late, recession)
   - How have similar cycle positions played out?
   - What sectors/factors typically lead at this stage?

4. SEASONAL PATTERNS
   - January effect, sell in May, Santa Claus rally, etc.
   - Earnings seasonality
   - Sector rotation patterns

5. FORWARD-PREDICTIVE REASONING
   - What catalysts are coming?
   - Position AHEAD of events, not after
   - Every trade needs a catalyst with a timeline

6. SECOND-ORDER THINKING
   - The obvious trade is often crowded or impossible
   - Who benefits that isn't obvious?
   - Where will capital flow AFTER the primary event?

=== HISTORICAL KNOWLEDGE BASE ===

You have access to these historical patterns:

FED CYCLES:
- Rate cuts: Small caps typically outperform 20-25% in 6-12 months post first cut
- 2019 parallel: Fed cut 3x, growth outperformed, IWM rallied
- Pause after hikes: Usually bullish for 6-18 months until recession

EARNINGS REACTIONS:
- NVDA beat: AMD, SMCI, AVGO typically follow within 48 hours
- Sector leaders set tone for group

IPO SPILLOVER:
- Major IPO brings sector attention
- Comparable public companies benefit 2-4 weeks before/after
- Rivian IPO 2021: EV sector rallied 20%+ in anticipation

CHIP RESTRICTIONS:
- Initial selloff 10-15% in exposed names
- Recovery within 60 days typical
- Domestic alternatives benefit (INTC, TXN)

YIELD CURVE:
- Inversion: Recession 12-18 months later (not immediate)
- Uninversion: Often signals recession approaching, not avoiding

SEASONAL:
- January: Small caps historically strong first 2 weeks
- May-October: Historically weaker than Nov-April
- Earnings seasons: Volatility clusters around reporting dates

=== CONVICTION SCORING ===

Score trades 0-100 based on:

80-100: HIGH CONVICTION - Trade
- Historical pattern matches clearly
- Clear catalyst with known timeline
- Strong second-order logic
- Favorable risk/reward at current price
- Technical setup supports entry timing

60-79: MEDIUM CONVICTION - Watch
- Pattern similarity is partial
- Catalyst exists but timing uncertain
- Logic is sound but may be crowded
- Wait for better entry or confirmation

0-59: LOW CONVICTION - Pass
- No clear historical parallel
- No catalyst or very speculative
- Risk/reward unfavorable
- Thesis relies on hope, not pattern

=== OUTPUT REQUIREMENTS ===

For every analysis, you MUST provide:

HISTORICAL CONTEXT:
1. ANALOGOUS EVENT: What historical period/event does this resemble?
2. HISTORICAL OUTCOME: What happened after that analogous situation?
3. KEY DIFFERENCES: How does current setup differ from history?
4. PATTERN CONFIDENCE: High/Medium/Low

FORWARD CATALYST:
5. CATALYST: The specific future event driving the opportunity
6. TIMELINE: When the catalyst occurs

TRADE RECOMMENDATION:
7. TICKER: The recommended trade
8. SECOND-ORDER RATIONALE: Why THIS ticker benefits
9. TECHNICAL CONTEXT: Where is price vs historical range
10. MACRO POSITION: Where are we in the cycle
11. SEASONAL FACTORS: Any seasonal tailwinds/headwinds

THESIS:
12. THESIS: Investment thesis combining history + forward view
13. BULL CASE: Best-case scenario
14. BEAR CASE: What could go wrong
15. VARIANT PERCEPTION: What we see that others don't

TRADE PARAMETERS:
16. CONVICTION SCORE: 0-100 with justification
17. Entry strategy, stop loss, time horizon

=== RESPONSE FORMAT ===

Respond in JSON format:
{{
    "analysis_type": "TRADE|NO_TRADE|WATCH",
    
    "historical_context": {{
        "analogous_event": "description of similar historical event",
        "historical_period": "e.g., 2019, Q4 2018, 2008 GFC",
        "historical_outcome": "what happened after",
        "pattern_confidence": "high|medium|low",
        "key_differences": ["how current differs from history"],
        "rhymes_with": "brief pattern description"
    }},
    
    "technical_context": "where is price vs historical range",
    "macro_cycle_position": "early|mid|late cycle or recession",
    "seasonal_factors": "any seasonal patterns at play",
    
    "catalyst": "specific future event",
    "catalyst_date": "YYYY-MM-DD or estimated timeframe",
    "catalyst_horizon": "immediate|days|weeks|months",
    
    "ticker": "SYMBOL or null",
    "recommendation": "BUY|SELL|HOLD|NONE",
    "conviction_score": 0-100,
    
    "primary_beneficiary": true/false,
    "second_order_rationale": "why this ticker benefits",
    
    "thesis": "2-3 sentence thesis combining history + forward view",
    "bull_case": "best case scenario",
    "bear_case": "what could go wrong",
    "variant_perception": "what we see that others don't",
    
    "position_size_pct": 0-25,
    "entry_strategy": "market|limit|scale_in",
    "entry_price_target": number or null,
    "stop_loss_pct": 0.05-0.15,
    "time_horizon": "days|weeks|months",
    
    "exit_triggers": [
        {{"type": "price_target|thesis_breaker|time", "value": "...", "description": "..."}}
    ],
    "thesis_breakers": ["condition that invalidates thesis"],
    "market_context": "brief market environment assessment",
    "reasoning_chain": "step-by-step: historical parallel -> current setup -> catalyst -> trade"
}}
"""
    
    def _build_analysis_prompt(
        self,
        signals: List[Dict[str, Any]],
        portfolio_context: Dict[str, Any],
        watchlist: List[str],
    ) -> str:
        """
        Build the user prompt with signals and context.
        
        Organizes signals by type and includes historical framing.
        """
        now = self.temporal_context.now
        
        # Group signals by type
        sentiment_signals = []
        macro_signals = []
        prediction_signals = []
        event_signals = []
        
        for sig in signals:
            sig_type = sig.get("signal_type") or sig.get("category", "")
            if "sentiment" in sig_type.lower():
                sentiment_signals.append(sig)
            elif "macro" in sig_type.lower():
                macro_signals.append(sig)
            elif "prediction" in sig_type.lower():
                prediction_signals.append(sig)
            else:
                event_signals.append(sig)
        
        # Build signal summaries
        signal_text = "=== CURRENT SIGNALS ===\n\n"
        
        if prediction_signals:
            signal_text += "PREDICTION MARKETS (Forward-Looking Probabilities):\n"
            for sig in prediction_signals[:5]:
                prob = sig.get("raw_value", {}).get("value")
                change = sig.get("raw_value", {}).get("change")
                summary = sig.get("summary", "")[:150]
                bias = sig.get("directional_bias", "unclear")
                
                prob_str = f"{prob:.0%}" if prob else "N/A"
                change_str = f" (Δ {change:+.1%})" if change else ""
                
                signal_text += f"  - [{prob_str}{change_str}] {summary}\n"
                signal_text += f"    Bias: {bias}\n"
            signal_text += "\n"
        
        if sentiment_signals:
            signal_text += "SENTIMENT (Market Expectations):\n"
            for sig in sentiment_signals[:5]:
                summary = sig.get("summary", "")[:150]
                bias = sig.get("directional_bias", "unclear")
                horizon = sig.get("forward_horizon") or sig.get("time_horizon", "")
                
                signal_text += f"  - {summary}\n"
                signal_text += f"    Bias: {bias}, Horizon: {horizon}\n"
            signal_text += "\n"
        
        if macro_signals:
            signal_text += "MACRO DATA (Economic Context):\n"
            for sig in macro_signals[:5]:
                summary = sig.get("summary", "")[:150]
                forward = sig.get("forward_implication", "")[:100]
                bias = sig.get("directional_bias", "unclear")
                
                signal_text += f"  - {summary}\n"
                if forward:
                    signal_text += f"    Forward Implication: {forward}\n"
                signal_text += f"    Bias: {bias}\n"
            signal_text += "\n"
        
        if event_signals:
            signal_text += "EVENTS & CATALYSTS:\n"
            for sig in event_signals[:5]:
                summary = sig.get("summary", "")[:150]
                catalyst_date = sig.get("catalyst_date", "")
                
                signal_text += f"  - {summary}\n"
                if catalyst_date:
                    signal_text += f"    Date: {catalyst_date}\n"
            signal_text += "\n"
        
        # Portfolio context
        portfolio_text = "=== PORTFOLIO CONTEXT ===\n"
        equity = portfolio_context.get("equity", 0)
        cash = portfolio_context.get("cash", 0)
        buying_power = portfolio_context.get("buying_power", 0)
        positions = portfolio_context.get("positions", [])
        
        portfolio_text += f"Equity: ${equity:,.2f}\n"
        portfolio_text += f"Cash: ${cash:,.2f}\n"
        portfolio_text += f"Buying Power: ${buying_power:,.2f}\n"
        
        if positions:
            portfolio_text += f"Current Positions ({len(positions)}):\n"
            for pos in positions[:5]:
                ticker = pos.get("ticker", "N/A")
                pnl_pct = pos.get("unrealized_pnl_pct", 0)
                portfolio_text += f"  - {ticker}: {pnl_pct:+.1f}%\n"
        else:
            portfolio_text += "No current positions\n"
        
        # Watchlist
        watchlist_text = f"\n=== WATCHLIST ===\n{', '.join(watchlist)}\n"
        
        # Final prompt
        return f"""Analyze these signals for trade opportunities using both historical pattern recognition 
and forward-looking analysis.

{signal_text}
{portfolio_text}
{watchlist_text}

=== YOUR TASK ===

Think like a seasoned trader:

1. ANCHOR IN HISTORY
   - Does this setup remind you of a historical period?
   - What happened after similar conditions?
   - "This looks like [period] when [outcome]..."

2. ASSESS CURRENT POSITION
   - Where are we in the macro cycle?
   - Any seasonal factors at play?
   - Technical context for potential trades?

3. IDENTIFY FORWARD CATALYST
   - What specific event will drive the next move?
   - When does it happen?

4. APPLY SECOND-ORDER THINKING
   - Who benefits that isn't obvious?
   - Where does capital flow after the primary event?

5. SYNTHESIZE INTO RECOMMENDATION
   - Combine historical pattern + current setup + catalyst
   - Score conviction based on pattern match strength

Remember: History doesn't repeat, but it rhymes. Find the rhyme.

If no compelling opportunity exists (weak historical parallel, no clear catalyst, 
poor risk/reward), return analysis_type: "NO_TRADE" with explanation.

Respond with your analysis in JSON format.
"""
    
    def _build_catalyst_prompt(
        self,
        catalyst_description: str,
        catalyst_date: Optional[str],
        portfolio_context: Dict[str, Any],
    ) -> str:
        """
        Build a prompt for analyzing a specific catalyst.
        
        Used for manual /catalyst queries.
        """
        portfolio_text = ""
        if portfolio_context:
            equity = portfolio_context.get("equity", 0)
            cash = portfolio_context.get("cash", 0)
            portfolio_text = f"""
=== PORTFOLIO CONTEXT ===
Equity: ${equity:,.2f}
Cash: ${cash:,.2f}
"""
        
        date_text = f"\nExpected Date: {catalyst_date}" if catalyst_date else ""
        
        return f"""Analyze this specific catalyst for trade opportunities using historical pattern 
recognition and second-order thinking.

=== CATALYST ===
{catalyst_description}{date_text}
{portfolio_text}

=== YOUR TASK ===

1. FIND HISTORICAL PARALLEL
   - When has something similar happened before?
   - What were the market implications?
   - What was the best trade to capture it?

2. IDENTIFY BENEFICIARIES
   - Primary beneficiary (often obvious or untradeable)
   - Second-order beneficiaries (the actual trade)

3. ASSESS TIMING
   - When should we position?
   - What's the expected duration?

4. TECHNICAL CONTEXT
   - For recommended ticker, where is price vs range?
   - Is entry timing favorable?

5. BUILD THE TRADE
   - Specific ticker recommendation
   - Entry strategy and risk parameters
   - Thesis that combines history + catalyst

Think: "This reminds me of [historical event]. Back then, [outcome]. 
Given that pattern, the play here is [recommendation] because [rationale]."

Respond with your analysis in JSON format.
"""
    
    # =========================================================================
    # API CALLS
    # =========================================================================
    
    async def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude API and parse JSON response.
        """
        if not self.api_key:
            logger.error("No API key - cannot call Claude")
            return None
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 4096,
                        "system": system_prompt,
                        "messages": [
                            {"role": "user", "content": user_prompt}
                        ],
                    },
                )
                
                if response.status_code != 200:
                    logger.error(f"Claude API error: {response.status_code} - {response.text}")
                    return None
                
                result = response.json()
                content = result.get("content", [{}])[0].get("text", "")
                
                # Parse JSON from response
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                return json.loads(content.strip())
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    async def analyze_signals(
        self,
        signals: List[Any],
        portfolio_context: Dict[str, Any],
        watchlist: List[str],
    ) -> Analysis:
        """
        Analyze signals for trade opportunities with historical context.
        
        This is the main entry point for the agent's scan cycle.
        """
        self.refresh_temporal_context()
        
        # Convert signals to dicts if needed
        signals_dicts = []
        for sig in signals:
            if hasattr(sig, 'to_dict'):
                signals_dicts.append(sig.to_dict())
            elif isinstance(sig, dict):
                signals_dicts.append(sig)
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_analysis_prompt(
            signals=signals_dicts,
            portfolio_context=portfolio_context,
            watchlist=watchlist,
        )
        
        # Call Claude
        response = await self._call_claude(system_prompt, user_prompt)
        
        # Parse response
        return self._parse_response_to_analysis(
            response=response,
            signals_used=[s.get("signal_id", "unknown") for s in signals_dicts],
        )
    
    async def analyze_specific_catalyst(
        self,
        catalyst_description: str,
        catalyst_date: Optional[str] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> Analysis:
        """
        Analyze a specific catalyst query with historical pattern matching.
        
        Used for manual /catalyst commands.
        """
        self.refresh_temporal_context()
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_catalyst_prompt(
            catalyst_description=catalyst_description,
            catalyst_date=catalyst_date,
            portfolio_context=portfolio_context or {},
        )
        
        # Call Claude
        response = await self._call_claude(system_prompt, user_prompt)
        
        # Parse response
        return self._parse_response_to_analysis(
            response=response,
            signals_used=[f"catalyst_query:{catalyst_description[:50]}"],
        )
    
    def _parse_response_to_analysis(
        self,
        response: Optional[Dict[str, Any]],
        signals_used: List[str],
    ) -> Analysis:
        """
        Parse Claude's JSON response into an Analysis object.
        """
        now = datetime.now(timezone.utc)
        analysis_id = str(uuid.uuid4())
        
        if not response:
            # Return empty analysis on error
            return Analysis(
                analysis_id=analysis_id,
                timestamp_utc=now,
                ticker=None,
                recommendation=Recommendation.NONE,
                conviction_score=0,
                historical_context=None,
                technical_context="Analysis failed",
                macro_cycle_position="unknown",
                seasonal_factors="none",
                thesis="Failed to generate analysis",
                catalyst="none",
                catalyst_date=None,
                catalyst_horizon="unknown",
                primary_beneficiary=True,
                second_order_rationale="",
                bull_case="",
                bear_case="Analysis error",
                variant_perception="",
                position_size_pct=0,
                entry_strategy="none",
                entry_price_target=None,
                stop_loss_pct=0.15,
                exit_triggers=[],
                thesis_breakers=["Analysis failed"],
                time_horizon="unknown",
                signals_used=signals_used,
                market_context="Error in analysis",
            )
        
        # Parse historical context
        hist_ctx = response.get("historical_context")
        historical_context = None
        if hist_ctx and isinstance(hist_ctx, dict):
            historical_context = HistoricalContext(
                analogous_event=hist_ctx.get("analogous_event", ""),
                historical_period=hist_ctx.get("historical_period", ""),
                historical_outcome=hist_ctx.get("historical_outcome", ""),
                pattern_confidence=hist_ctx.get("pattern_confidence", "low"),
                key_differences=hist_ctx.get("key_differences", []),
                rhymes_with=hist_ctx.get("rhymes_with", ""),
            )
        
        # Parse exit triggers
        exit_triggers = []
        for trigger in response.get("exit_triggers", []):
            if isinstance(trigger, dict):
                exit_triggers.append(ExitTrigger(
                    trigger_type=trigger.get("type", "unknown"),
                    value=trigger.get("value"),
                    description=trigger.get("description", ""),
                ))
        
        # Parse recommendation
        rec_str = response.get("recommendation", "NONE").upper()
        try:
            recommendation = Recommendation(rec_str)
        except ValueError:
            recommendation = Recommendation.NONE
        
        return Analysis(
            analysis_id=analysis_id,
            timestamp_utc=now,
            ticker=response.get("ticker"),
            recommendation=recommendation,
            conviction_score=response.get("conviction_score", 0),
            # Historical context
            historical_context=historical_context,
            technical_context=response.get("technical_context", ""),
            macro_cycle_position=response.get("macro_cycle_position", "unknown"),
            seasonal_factors=response.get("seasonal_factors", "none"),
            # Forward reasoning
            thesis=response.get("thesis", ""),
            catalyst=response.get("catalyst", ""),
            catalyst_date=response.get("catalyst_date"),
            catalyst_horizon=response.get("catalyst_horizon", "unknown"),
            primary_beneficiary=response.get("primary_beneficiary", True),
            second_order_rationale=response.get("second_order_rationale", ""),
            bull_case=response.get("bull_case", ""),
            bear_case=response.get("bear_case", ""),
            variant_perception=response.get("variant_perception", ""),
            position_size_pct=response.get("position_size_pct", 0),
            entry_strategy=response.get("entry_strategy", "market"),
            entry_price_target=response.get("entry_price_target"),
            stop_loss_pct=response.get("stop_loss_pct", 0.15),
            exit_triggers=exit_triggers,
            thesis_breakers=response.get("thesis_breakers", []),
            time_horizon=response.get("time_horizon", "unknown"),
            signals_used=signals_used,
            market_context=response.get("market_context", ""),
        )
