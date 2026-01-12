"""
Gann Sentinel Trader - Claude Analyst
Forward-Predictive Reasoning Engine with Historical Context + Business Model Audit + Technical Structure

Version: 4.0.0 (Technical Integration Update)
Last Updated: January 2026

Change Log:
- 4.0.0: Added Technical Scanner integration
         Technical signals now feed into conviction scoring
         Chart structure (market state, channel position, verdict) included in analysis
         Updated conviction scoring to weight technical setup quality
- 3.0.0: Added Business Model Audit - analyze full business before pattern matching
         Added Platform Transformation historical patterns (AMZN, NVDA, MSFT)
         Added "Challenge Priced In" framework
         Added business_intel signal handling
- 2.1.0: Historical Context Update
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
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    WATCH = "WATCH"


# =============================================================================
# HISTORICAL PATTERN KNOWLEDGE BASE
# =============================================================================

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
    
    # PLATFORM TRANSFORMATION PATTERNS (NEW)
    "platform_transformation_aws": {
        "description": "Amazon AWS transformation pattern - hidden business line emergence",
        "analogues": [
            {"period": "2014-2020", "event": "AWS revealed as major profit driver", 
             "outcome": "Stock 10x as market re-rated from 'e-commerce' to 'cloud platform'"},
        ],
        "pattern": "Market systematically undervalues emerging business lines in platform companies",
        "key_signals": [
            "New business growing 40%+ while 'core' business mature",
            "Analysts still categorizing by legacy business",
            "Sum-of-parts missing synergies",
        ],
        "typical_upside": "50-80% over 3-5 years when consensus breaks",
    },
    
    "platform_transformation_nvidia": {
        "description": "NVIDIA AI transformation - TAM expansion pattern",
        "analogues": [
            {"period": "2016-2024", "event": "Gaming GPU company becomes AI infrastructure monopoly",
             "outcome": "Stock 20x+ as AI TAM dwarfed gaming TAM"},
        ],
        "pattern": "Developer ecosystem lock-in (CUDA) created moat that market underestimated",
        "key_signals": [
            "Developer mindshare/ecosystem ignored by financial analysts",
            "New use case (AI training) requires existing product",
            "Competitors years behind on software stack",
        ],
        "typical_upside": "10-20x over 5-8 years for true platform shifts",
    },
    
    "platform_transformation_microsoft": {
        "description": "Microsoft cloud transformation pattern",
        "analogues": [
            {"period": "2014-2020", "event": "Legacy software company becomes cloud leader",
             "outcome": "Stock 5x as Azure + Office 365 re-rated the business"},
        ],
        "pattern": "Enterprise relationships + cloud transition underestimated",
        "key_signals": [
            "'Legacy' label despite growing cloud business",
            "Enterprise stickiness undervalued",
            "Multiple product lines benefiting from single platform shift",
        ],
    },
    
    "platform_optionality_tesla": {
        "description": "Tesla platform optionality pattern",
        "analogues": [
            {"period": "2020-present", "event": "EV company expands to Energy, AI, Robotics",
             "outcome": "Multiple business lines emerging beyond automotive"},
        ],
        "pattern": "Market prices only automotive TAM, ignores Energy/FSD/Robotics optionality",
        "key_signals": [
            "FSD subscription/licensing = software business (not priced)",
            "Optimus = entirely new labor TAM (priced at zero)",
            "Energy storage growing faster than auto",
            "Manufacturing cost advantages across all lines",
        ],
        "what_needs_to_be_true_for_bear": [
            "FSD never achieves full autonomy",
            "Optimus fails commercially",
            "Energy business hits ceiling",
            "Competition catches up on manufacturing",
        ],
    },
    
    # Sector IPO Spillover
    "major_ipo_sector_attention": {
        "description": "Major IPO brings attention to entire sector",
        "analogues": [
            {"period": "2021", "event": "Rivian IPO", "outcome": "EV sector rallied 20%+ in anticipation"},
            {"period": "2020", "event": "Snowflake IPO", "outcome": "Cloud/SaaS sector attention"},
        ],
        "typical_beneficiaries": ["Comparable public companies in same sector"],
        "typical_duration": "2-4 weeks pre-IPO, 1-2 weeks post",
    },
    
    # Chip Restrictions
    "china_chip_restrictions": {
        "description": "US restricts chip exports to China",
        "analogues": [
            {"period": "2022", "event": "Biden chip restrictions", "outcome": "NVDA/AMD sold off 10-15%, recovered in 60 days"},
        ],
        "typical_beneficiaries": ["INTC", "TXN", "ON"],
        "pattern": "Initial selloff, recovery within 60 days, domestic alternatives benefit",
    },
    
    # Earnings Patterns
    "nvidia_earnings_reaction": {
        "description": "NVIDIA earnings reaction patterns",
        "analogues": [
            {"period": "2023-2024", "event": "Beat and raise", "outcome": "Sector-wide rally, AMD/SMCI/AVGO followed within 48 hours"},
        ],
        "typical_beneficiaries": ["AMD", "SMCI", "AVGO", "MRVL", "TSM"],
    },
    
    # Yield Curve
    "yield_curve_inversion": {
        "description": "10Y-2Y yield curve inversion/uninversion",
        "analogues": [
            {"period": "2019", "event": "Curve inverts", "outcome": "Recession 12-18 months later"},
        ],
        "typical_implications": "Uninversion often signals recession approaching, not avoiding it",
    },
    
    # Seasonal Patterns
    "january_effect": {
        "description": "Small caps tend to outperform in January",
        "typical_beneficiaries": ["IWM", "IJR", "VB"],
    },
    
    # Crisis Patterns
    "crisis_playbook": {
        "description": "Historical crisis market behavior",
        "analogues": [
            {"period": "2020 COVID", "event": "Exogenous shock", "outcome": "V-bottom, growth led"},
            {"period": "2008 GFC", "event": "Financial crisis", "outcome": "Extended bottom, quality/defensive first up"},
        ],
        "pattern": "Crisis type determines leadership: exogenous=growth, financial=quality, inflation=value",
    },
}


# Second-order effect mappings
SECOND_ORDER_EFFECTS = {
    # Space sector
    "spacex_ipo": ["RKLB", "LMT", "RTX", "BA"],
    
    # AI/Chips
    "nvidia_earnings": ["AMD", "SMCI", "AVGO", "MRVL"],
    "china_chip_ban": ["INTC", "TXN"],
    
    # Fed/Rates
    "fed_cut": ["TLT", "IWM", "XLF", "XHB"],
    
    # Platform companies - cross-business effects
    "tesla_fsd_approval": ["TSLA", "mobility sector"],
    "tesla_optimus_demo": ["TSLA", "robotics sector"],
    "tesla_energy_growth": ["TSLA", "grid storage sector"],
    
    # Crypto
    "bitcoin_etf": ["COIN", "MSTR", "SQ"],
    
    # EV/Energy
    "tesla_earnings": ["RIVN", "LCID", "NIO"],
    "ev_subsidies": ["TSLA", "RIVN", "F", "GM"],
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExitTrigger:
    """Defines when to exit a position."""
    trigger_type: str
    value: Any
    description: str


@dataclass
class HistoricalContext:
    """Historical patterns informing the analysis."""
    analogous_event: str
    historical_period: str
    historical_outcome: str
    pattern_confidence: str
    key_differences: List[str]
    rhymes_with: str


@dataclass
class BusinessLineAnalysis:
    """Analysis of a single business line within a company."""
    segment: str
    tam_size: str
    current_valuation: str  # priced_in, undervalued, zero
    growth_trajectory: str
    catalyst_timeline: Optional[str]


@dataclass
class Analysis:
    """Claude's analysis output with business model audit and historical context."""
    analysis_id: str
    timestamp_utc: datetime
    
    # Core recommendation
    ticker: Optional[str]
    recommendation: Recommendation
    conviction_score: int
    
    # Business Model Audit (NEW)
    company_type: str  # platform, single_product, conglomerate
    business_lines_analyzed: List[Dict[str, Any]]
    segments_priced_at_zero: List[str]
    optionality_value: str
    
    # Historical context
    historical_context: Optional[HistoricalContext]
    technical_context: str
    macro_cycle_position: str
    seasonal_factors: str
    
    # Forward-predictive reasoning
    thesis: str
    catalyst: str
    catalyst_date: Optional[str]
    catalyst_horizon: str
    
    # Second-order thinking
    primary_beneficiary: bool
    second_order_rationale: str
    
    # Bull/Bear cases
    bull_case: str
    bear_case: str
    variant_perception: str
    what_needs_to_be_true_for_bear: List[str]
    
    # Trade parameters
    position_size_pct: float
    entry_strategy: str
    entry_price_target: Optional[float]
    stop_loss_pct: float
    
    # Exit strategy
    exit_triggers: List[ExitTrigger]
    thesis_breakers: List[str]
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
            # Business Model Audit
            "company_type": self.company_type,
            "business_lines_analyzed": self.business_lines_analyzed,
            "segments_priced_at_zero": self.segments_priced_at_zero,
            "optionality_value": self.optionality_value,
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
            "what_needs_to_be_true_for_bear": self.what_needs_to_be_true_for_bear,
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
    Forward-Predictive Reasoning Engine with Business Model Audit.
    
    The analyst:
    1. AUDITS the business model first (map all TAMs, find what's priced at zero)
    2. ANCHORS in historical patterns (what happened before)
    3. ORIENTS toward future catalysts (what's coming)
    4. CHALLENGES "priced in" narratives (what is consensus missing?)
    5. Synthesizes into high-conviction trade recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude Analyst."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set - analysis disabled")
        
        # Use Claude 3 Sonnet - stable production model
        self.model = "claude-3-sonnet-20240229"
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        self.temporal_context = get_temporal_context()
        self.historical_patterns = HISTORICAL_PATTERNS
        self.second_order_effects = SECOND_ORDER_EFFECTS
        
        logger.info(f"Claude Analyst v4.0.0 initialized - Business Intelligence Mode")
    
    def refresh_temporal_context(self) -> None:
        """Refresh temporal context."""
        self.temporal_context = get_temporal_context()
    
    # =========================================================================
    # PROMPT CONSTRUCTION
    # =========================================================================
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for Claude.
        
        CRITICAL ADDITION: Business Model Audit BEFORE pattern matching.
        """
        now = self.temporal_context.now
        end_of_month = self.temporal_context.format_date(self.temporal_context.end_of_month)
        end_of_quarter = self.temporal_context.format_date(self.temporal_context.end_of_quarter)
        current_month = now.strftime("%B")
        current_quarter = f"Q{(now.month - 1) // 3 + 1}"
        
        return f"""You are a senior trading strategist for an autonomous trading system.

CURRENT DATE: {self.temporal_context.format_date(now, '%B %d, %Y')}
KEY DATES: End of Month: {end_of_month}, End of Quarter: {end_of_quarter}
Current: {current_month}, {current_quarter}

=== CRITICAL: BUSINESS MODEL AUDIT (DO THIS FIRST) ===

Before ANY pattern matching or sentiment analysis, you MUST audit the business:

1. MAP ALL BUSINESS LINES
   - What are ALL revenue streams and business lines?
   - Don't accept surface-level categorization ("auto company", "search company")
   - Platform companies often have 5-8+ distinct TAMs

2. TAM ANALYSIS PER SEGMENT
   - What's the total addressable market for EACH segment?
   - Which TAMs are being priced in by the market?
   - Which TAMs are being valued at ZERO?

3. CHALLENGE "PRICED IN" NARRATIVES
   - When signals say "upside priced in," ask: PRICED IN FOR WHICH BUSINESS LINE?
   - What optionality is NOT being priced?
   - What would the transformative bull case look like that most dismiss?

4. WHAT NEEDS TO BE TRUE (FOR BEAR CASE)
   - For the bear case to be correct, what must happen?
   - How probable are those conditions?
   - Are bears making assumptions about business lines they don't understand?

5. ASYMMETRY CHECK
   - If the market is wrong about platform optionality, what's the magnitude?
   - Platform companies have historically delivered 50-80% upside when consensus breaks

=== PLATFORM COMPANY PATTERNS (Historical Reference) ===

AMAZON PATTERN (2014-2020):
- Consensus: "Just an e-commerce company, AWS is tiny"
- Reality: Platform optionality massively undervalued
- Outcome: Stock 10x as AWS revealed, advertising emerged
- Key Signal: New business line growing 40%+ while core "mature"

NVIDIA PATTERN (2016-2024):
- Consensus: "Gaming GPU company, crypto exposure"  
- Reality: AI training infrastructure monopoly emerging
- Outcome: Stock 20x as AI adoption accelerated
- Key Signal: Developer ecosystem (CUDA) lock-in ignored by analysts

MICROSOFT PATTERN (2014-2020):
- Consensus: "Legacy software, Windows declining"
- Reality: Cloud transformation + enterprise moat
- Outcome: Stock 5x as Azure + Office 365 ramped
- Key Signal: Enterprise stickiness undervalued

COMMON THREAD: Markets systematically undervalue optionality in platform companies.
Sum-of-the-parts analysis misses synergies. "Priced in" calls often wrong by 50-80%.

=== CORE METHODOLOGY ===

AUDIT first, then ANCHOR in HISTORY, ORIENT toward the FUTURE.

1. BUSINESS MODEL AUDIT (see above - do this FIRST)

2. HISTORICAL PATTERN RECOGNITION
   - Identify analogous situations from the patterns above
   - "When has this happened before, and what followed?"
   - For platform companies, use platform transformation patterns

3. TECHNICAL CONTEXT
   - Where is price relative to historical ranges?
   - Is entry timing favorable?

4. MACRO CYCLE POSITION
   - Where are we in the business cycle?
   - What typically leads at this stage?

5. FORWARD-PREDICTIVE REASONING
   - What catalysts are coming for EACH business line?
   - Position AHEAD of events

6. SECOND-ORDER THINKING
   - Who benefits that isn't obvious?
   - For platform companies: which segment drives the re-rating?

=== ADDITIONAL HISTORICAL PATTERNS ===

FED CYCLES:
- Rate cuts: Small caps +20-25% in 6-12 months post first cut
- Pause after hikes: Usually bullish 6-18 months

EARNINGS REACTIONS:
- NVDA beat: AMD, SMCI, AVGO follow within 48 hours
- Sector leaders set tone for group

IPO SPILLOVER:
- Major IPO brings sector attention
- Comparable public companies benefit 2-4 weeks before/after

CHIP RESTRICTIONS:
- Initial selloff 10-15%, recovery within 60 days
- Domestic alternatives benefit

=== CONVICTION SCORING ===

80-100: HIGH CONVICTION - Trade
- Business model audit reveals undervalued optionality
- Historical pattern matches clearly (including platform transformation)
- Clear catalyst with timeline
- Technical setup supports entry (CRITICAL):
  * Technical verdict is "hypothesis_allowed"
  * Price at channel support (for longs) or resistance (for shorts)
  * Market state is TRENDING or RANGE_BOUND (not transitional)
  * R-multiple >= 1.8
  * If technical verdict is "no_trade", reduce conviction by 15-20 points

60-79: MEDIUM - Watch
- Some optionality unclear
- Pattern partial
- Wait for better entry
- Technical verdict is "analyze_only" or position is mid-channel

0-59: LOW - Pass
- No clear undervalued segments
- Bear case assumptions are reasonable
- Risk/reward unfavorable
- Technical verdict is "no_trade" with no clear setup
- Market state is "transitional" with low confidence

TECHNICAL STRUCTURE INTEGRATION:
When technical signals are present:
1. Check market state FIRST - transitional states reduce conviction by 10
2. Check channel position - mid-channel (30-70%) has no edge, reduce by 5
3. Check verdict - "no_trade" means wait, reduce by 15
4. Check R-multiple - if < 1.8, trade is not favorable
5. If technical says SELL but fundamentals say BUY, resolve conflict:
   - If channel position > 85%, wait for pullback
   - If channel position < 15%, fundamental BUY is confirmed by technical

=== OUTPUT REQUIREMENTS ===

For EVERY analysis, you MUST provide:

BUSINESS MODEL AUDIT:
1. COMPANY TYPE: platform/single_product/conglomerate
2. BUSINESS LINES: List ALL segments with TAM and current valuation treatment
3. SEGMENTS AT ZERO: Which business lines are being valued at $0?
4. OPTIONALITY VALUE: What's not being priced?
5. WHAT NEEDS TO BE TRUE FOR BEAR: Specific conditions

HISTORICAL CONTEXT:
6. ANALOGOUS EVENT: What historical pattern applies?
7. HISTORICAL OUTCOME: What happened after?
8. KEY DIFFERENCES: How does current differ?

FORWARD CATALYST:
9. CATALYST: Specific future event (BY SEGMENT if platform company)
10. TIMELINE: When

TRADE RECOMMENDATION:
11. TICKER, CONVICTION SCORE, ENTRY STRATEGY
12. THESIS combining business audit + history + catalyst

=== RESPONSE FORMAT ===

Respond in JSON:
{{
    "analysis_type": "TRADE|NO_TRADE|WATCH",
    
    "business_model_audit": {{
        "company_type": "platform|single_product|conglomerate",
        "business_lines": [
            {{
                "segment": "name",
                "tam_size": "estimate",
                "current_valuation": "priced_in|undervalued|zero",
                "growth_rate": "estimate",
                "catalyst_timeline": "when next catalyst"
            }}
        ],
        "segments_priced_at_zero": ["list of overlooked segments"],
        "optionality_value": "description of unpriced optionality",
        "what_needs_to_be_true_for_bear": ["specific conditions"]
    }},
    
    "historical_context": {{
        "analogous_event": "description",
        "historical_period": "when",
        "historical_outcome": "what happened",
        "pattern_confidence": "high|medium|low",
        "key_differences": ["list"],
        "rhymes_with": "brief pattern description"
    }},
    
    "technical_context": "price vs range",
    "technical_structure": {{
        "market_state": "trending|range_bound|transitional",
        "market_state_bias": "bullish|bearish|neutral",
        "channel_position": "0-100% (0=bottom, 100=top)",
        "technical_verdict": "hypothesis_allowed|no_trade|analyze_only",
        "entry_timing": "favorable|wait|unfavorable",
        "technical_conviction_adjustment": "-20 to +10 (adjustment to base conviction)"
    }},
    "macro_cycle_position": "cycle position",
    "seasonal_factors": "any seasonal patterns",
    
    "catalyst": "specific event",
    "catalyst_date": "YYYY-MM-DD or timeframe",
    "catalyst_horizon": "immediate|days|weeks|months",
    
    "ticker": "SYMBOL or null",
    "recommendation": "BUY|SELL|HOLD|NONE",
    "conviction_score": 0-100,
    
    "primary_beneficiary": true/false,
    "second_order_rationale": "why this ticker",
    
    "thesis": "2-3 sentences combining audit + history + catalyst",
    "bull_case": "best case",
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
    "thesis_breakers": ["conditions that invalidate thesis"],
    "market_context": "market environment",
    "reasoning_chain": "audit -> history -> catalyst -> trade"
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
        
        Updated to handle business_intel signals from deep scans.
        """
        now = self.temporal_context.now
        
        # Group signals by type
        sentiment_signals = []
        macro_signals = []
        prediction_signals = []
        event_signals = []
        business_intel_signals = []  # NEW
        technical_signals = []  # NEW - Chart analysis
        
        for sig in signals:
            sig_type = sig.get("signal_type") or sig.get("category", "")
            if "technical" in sig_type.lower():
                technical_signals.append(sig)
            elif "business_intel" in sig_type.lower():
                business_intel_signals.append(sig)
            elif "sentiment" in sig_type.lower():
                sentiment_signals.append(sig)
            elif "macro" in sig_type.lower():
                macro_signals.append(sig)
            elif "prediction" in sig_type.lower():
                prediction_signals.append(sig)
            else:
                event_signals.append(sig)
        
        # Build signal summaries
        signal_text = "=== CURRENT SIGNALS ===\n\n"
        
        # TECHNICAL ANALYSIS (Chart Structure - Process FIRST)
        if technical_signals:
            signal_text += "CHART STRUCTURE (Technical Analysis):\n"
            for sig in technical_signals[:5]:
                ticker = sig.get("ticker", "Unknown")
                price = sig.get("current_price", 0)
                
                # Market state
                ms = sig.get("market_state", {})
                state = ms.get("state", "unknown")
                bias = ms.get("bias", "neutral")
                confidence = ms.get("confidence", "low")
                evidence = ms.get("evidence", [])
                
                # Verdict
                verdict = sig.get("verdict", "unknown")
                verdict_reasons = sig.get("verdict_reasons", [])
                
                # Channel
                channel = sig.get("trend_channel", {})
                channel_pos = channel.get("position_in_channel", 0.5) if channel else 0.5
                channel_upper = channel.get("channel_upper", 0) if channel else 0
                channel_lower = channel.get("channel_lower", 0) if channel else 0
                
                # Support/Resistance
                support = sig.get("support_levels", [])
                resistance = sig.get("resistance_levels", [])
                
                # Trade hypothesis
                hypo = sig.get("trade_hypothesis")
                
                # Primary scenario
                primary = sig.get("primary_scenario", {})
                
                signal_text += f"\n  {ticker} @ ${price:.2f}:\n"
                signal_text += f"    Market State: {state.upper()} ({bias}, {confidence} confidence)\n"
                if evidence:
                    signal_text += f"    Evidence: {evidence[0][:60]}\n"
                
                if channel_upper and channel_lower:
                    signal_text += f"    Channel: ${channel_lower:.2f} - ${channel_upper:.2f} (position: {channel_pos:.0%} from bottom)\n"
                
                if support:
                    signal_text += f"    Support: ${support[0]:.2f}\n"
                if resistance:
                    signal_text += f"    Resistance: ${resistance[0]:.2f}\n"
                
                signal_text += f"    VERDICT: {verdict.upper()}\n"
                if verdict_reasons:
                    signal_text += f"    Reason: {verdict_reasons[0][:60]}\n"
                
                if hypo and hypo.get("allow_trade"):
                    side = hypo.get("side", "").upper()
                    r_mult = hypo.get("expected_r", 0)
                    entry = hypo.get("entry_zone", {})
                    stop = hypo.get("invalidation", {})
                    signal_text += f"    SETUP: {side} at ${entry.get('low', 0):.2f}-${entry.get('high', 0):.2f}, stop ${stop.get('level', 0):.2f}, R={r_mult:.1f}\n"
                
                if primary:
                    signal_text += f"    Primary Scenario: {primary.get('name', '')} ({primary.get('probability', 0):.0%})\n"
            
            signal_text += "\n"
        
        # Business Intel signals (prioritize these - NEW)
        if business_intel_signals:
            signal_text += "BUSINESS INTELLIGENCE (Deep Analysis):\n"
            for sig in business_intel_signals[:5]:
                summary = sig.get("summary", "")[:200]
                business_lines = sig.get("business_lines", [])
                contrarian = sig.get("contrarian_signals", {})
                
                signal_text += f"  - {summary}\n"
                
                if business_lines:
                    signal_text += "    Business Lines:\n"
                    for bl in business_lines[:5]:
                        seg = bl.get("segment", "Unknown")
                        val = bl.get("valuation", bl.get("sentiment", "N/A"))
                        signal_text += f"      * {seg}: {val}\n"
                
                if contrarian:
                    undervalued = contrarian.get("undervalued_segment") or contrarian.get("underappreciated_catalyst")
                    if undervalued:
                        signal_text += f"    Contrarian: {undervalued[:80]}\n"
            signal_text += "\n"
        
        if prediction_signals:
            signal_text += "PREDICTION MARKETS:\n"
            for sig in prediction_signals[:5]:
                prob = sig.get("raw_value", {}).get("value")
                summary = sig.get("summary", "")[:150]
                prob_str = f"{prob:.0%}" if prob else "N/A"
                signal_text += f"  - [{prob_str}] {summary}\n"
            signal_text += "\n"
        
        if sentiment_signals:
            signal_text += "SENTIMENT:\n"
            for sig in sentiment_signals[:5]:
                summary = sig.get("summary", "")[:150]
                bias = sig.get("directional_bias", "unclear")
                
                # Include business line breakdown if available (from deep scans)
                business_lines = sig.get("business_lines", [])
                
                signal_text += f"  - {summary}\n"
                signal_text += f"    Bias: {bias}\n"
                
                if business_lines:
                    overlooked = [bl.get("segment") for bl in business_lines if bl.get("overlooked")]
                    if overlooked:
                        signal_text += f"    Overlooked segments: {', '.join(overlooked)}\n"
            signal_text += "\n"
        
        if macro_signals:
            signal_text += "MACRO DATA:\n"
            for sig in macro_signals[:5]:
                summary = sig.get("summary", "")[:150]
                signal_text += f"  - {summary}\n"
            signal_text += "\n"
        
        if event_signals:
            signal_text += "EVENTS & CATALYSTS:\n"
            for sig in event_signals[:5]:
                summary = sig.get("summary", "")[:150]
                signal_text += f"  - {summary}\n"
            signal_text += "\n"
        
        # Portfolio context
        portfolio_text = "=== PORTFOLIO CONTEXT ===\n"
        equity = portfolio_context.get("equity", 0)
        cash = portfolio_context.get("cash", 0)
        positions = portfolio_context.get("positions", [])
        
        portfolio_text += f"Equity: ${equity:,.2f}, Cash: ${cash:,.2f}\n"
        
        if positions:
            portfolio_text += f"Positions: {', '.join(p.get('ticker', 'N/A') for p in positions[:5])}\n"
        
        # Watchlist
        watchlist_text = f"\n=== WATCHLIST ===\n{', '.join(watchlist)}\n"
        
        return f"""Analyze these signals for trade opportunities.

{signal_text}
{portfolio_text}
{watchlist_text}

=== YOUR TASK ===

1. BUSINESS MODEL AUDIT FIRST
   - For each ticker with business intel, map ALL business lines
   - Identify what's being valued at ZERO
   - Challenge any "priced in" sentiment - priced in for WHICH segments?

2. TECHNICAL STRUCTURE CHECK
   - Review chart analysis signals if present
   - What is the market state? (trending/range/transitional)
   - Where is price in the channel? (buy zone/sell zone/no edge)
   - What is the technical verdict? (hypothesis_allowed/no_trade/analyze_only)
   - Does chart structure support entry timing?

3. ANCHOR IN HISTORY
   - Does this setup match a platform transformation pattern?
   - What happened after similar conditions?

4. IDENTIFY FORWARD CATALYSTS
   - What events are coming for EACH business line?
   - Where should we position ahead?

5. APPLY SECOND-ORDER THINKING
   - For platform companies: which SEGMENT drives re-rating?
   - What optionality is market missing?

6. SYNTHESIZE
   - Combine business audit + technical structure + historical pattern + catalyst
   - Score conviction based on: unpriced optionality + pattern match + TECHNICAL SETUP
   - If technical verdict is "no_trade", reduce conviction accordingly
   - If technical verdict is "hypothesis_allowed", boost conviction if fundamentals align

CRITICAL: Do not accept surface-level "priced in" takes. 
Audit the business first. Check the chart structure. What is actually being priced?

Respond with your analysis in JSON format.
"""
    
    def _build_catalyst_prompt(
        self,
        catalyst_description: str,
        catalyst_date: Optional[str],
        portfolio_context: Dict[str, Any],
    ) -> str:
        """Build a prompt for analyzing a specific catalyst."""
        portfolio_text = ""
        if portfolio_context:
            equity = portfolio_context.get("equity", 0)
            portfolio_text = f"\n=== PORTFOLIO ===\nEquity: ${equity:,.2f}\n"
        
        date_text = f"\nExpected Date: {catalyst_date}" if catalyst_date else ""
        
        return f"""Analyze this catalyst for trade opportunities.

=== CATALYST ===
{catalyst_description}{date_text}
{portfolio_text}

=== YOUR TASK ===

1. BUSINESS MODEL AUDIT
   - For affected companies, what are ALL business lines?
   - Which segments does this catalyst impact?
   - Is the market pricing in all affected segments?

2. FIND HISTORICAL PARALLEL
   - When has something similar happened?
   - What were the market implications?
   - Does a platform transformation pattern apply?

3. IDENTIFY BENEFICIARIES
   - Primary beneficiary (often obvious)
   - Second-order beneficiaries (the actual trade)
   - Which segment drives the re-rating?

4. BUILD THE TRADE
   - Specific ticker recommendation
   - Thesis combining business audit + history + catalyst
   - What makes this NOT priced in?

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

        Returns dict on success, or dict with 'error' key on failure.
        """
        if not self.api_key:
            logger.error("No API key - cannot call Claude")
            return {"error": "ANTHROPIC_API_KEY not configured"}

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
                    error_body = response.text[:200]  # First 200 chars of error
                    logger.error(f"Claude API error {response.status_code}: {error_body}")
                    return {"error": f"API {response.status_code}: {error_body[:100]}"}

                result = response.json()
                content = result.get("content", [{}])[0].get("text", "")

                if not content:
                    logger.error("Claude returned empty response")
                    return {"error": "Claude returned empty response"}

                # Parse JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            return {"error": f"JSON parse error: {str(e)[:80]}"}
        except httpx.TimeoutException:
            logger.error("Claude API timeout after 120 seconds")
            return {"error": "API timeout (120s)"}
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return {"error": f"API error: {str(e)[:80]}"}
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    async def analyze_signals(
        self,
        signals: List[Any],
        portfolio_context: Dict[str, Any],
        watchlist: List[str],
    ) -> 'Analysis':
        """Analyze signals for trade opportunities with business model audit."""
        self.refresh_temporal_context()
        
        # Convert signals to dicts
        signals_dicts = []
        for sig in signals:
            if hasattr(sig, 'to_dict'):
                signals_dicts.append(sig.to_dict())
            elif isinstance(sig, dict):
                signals_dicts.append(sig)
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_analysis_prompt(
            signals=signals_dicts,
            portfolio_context=portfolio_context,
            watchlist=watchlist,
        )
        
        response = await self._call_claude(system_prompt, user_prompt)
        
        return self._parse_response_to_analysis(
            response=response,
            signals_used=[s.get("signal_id", "unknown") for s in signals_dicts],
        )
    
    async def analyze_specific_catalyst(
        self,
        catalyst_description: str,
        catalyst_date: Optional[str] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> 'Analysis':
        """Analyze a specific catalyst with business model audit."""
        self.refresh_temporal_context()
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_catalyst_prompt(
            catalyst_description=catalyst_description,
            catalyst_date=catalyst_date,
            portfolio_context=portfolio_context or {},
        )
        
        response = await self._call_claude(system_prompt, user_prompt)
        
        return self._parse_response_to_analysis(
            response=response,
            signals_used=[f"catalyst_query:{catalyst_description[:50]}"],
        )
    
    def _parse_response_to_analysis(
        self,
        response: Optional[Dict[str, Any]],
        signals_used: List[str],
    ) -> Analysis:
        """Parse Claude's JSON response into an Analysis object."""
        now = datetime.now(timezone.utc)

        if not response:
            return self._create_error_analysis(signals_used, "No response from Claude")

        # Check if response contains an error
        if "error" in response:
            return self._create_error_analysis(signals_used, response["error"])
        
        try:
            # Parse business model audit
            audit = response.get("business_model_audit", {})
            business_lines = audit.get("business_lines", [])
            
            # Parse historical context
            hist = response.get("historical_context", {})
            historical_context = None
            if hist:
                historical_context = HistoricalContext(
                    analogous_event=hist.get("analogous_event", ""),
                    historical_period=hist.get("historical_period", ""),
                    historical_outcome=hist.get("historical_outcome", ""),
                    pattern_confidence=hist.get("pattern_confidence", "medium"),
                    key_differences=hist.get("key_differences", []),
                    rhymes_with=hist.get("rhymes_with", ""),
                )
            
            # Parse exit triggers
            exit_triggers = []
            for trigger in response.get("exit_triggers", []):
                exit_triggers.append(ExitTrigger(
                    trigger_type=trigger.get("type", "unknown"),
                    value=trigger.get("value"),
                    description=trigger.get("description", ""),
                ))
            
            # Get recommendation
            rec_str = response.get("recommendation", "NONE").upper()
            try:
                recommendation = Recommendation[rec_str]
            except KeyError:
                recommendation = Recommendation.NONE
            
            return Analysis(
                analysis_id=str(uuid.uuid4()),
                timestamp_utc=now,
                ticker=response.get("ticker"),
                recommendation=recommendation,
                conviction_score=response.get("conviction_score", 0),
                # Business Model Audit
                company_type=audit.get("company_type", "single_product"),
                business_lines_analyzed=business_lines,
                segments_priced_at_zero=audit.get("segments_priced_at_zero", []),
                optionality_value=audit.get("optionality_value", ""),
                # Historical
                historical_context=historical_context,
                technical_context=response.get("technical_context", ""),
                macro_cycle_position=response.get("macro_cycle_position", ""),
                seasonal_factors=response.get("seasonal_factors", ""),
                # Forward
                thesis=response.get("thesis", ""),
                catalyst=response.get("catalyst", ""),
                catalyst_date=response.get("catalyst_date"),
                catalyst_horizon=response.get("catalyst_horizon", ""),
                primary_beneficiary=response.get("primary_beneficiary", True),
                second_order_rationale=response.get("second_order_rationale", ""),
                # Cases
                bull_case=response.get("bull_case", ""),
                bear_case=response.get("bear_case", ""),
                variant_perception=response.get("variant_perception", ""),
                what_needs_to_be_true_for_bear=audit.get("what_needs_to_be_true_for_bear", []),
                # Trade params
                position_size_pct=response.get("position_size_pct", 0),
                entry_strategy=response.get("entry_strategy", "market"),
                entry_price_target=response.get("entry_price_target"),
                stop_loss_pct=response.get("stop_loss_pct", 0.10),
                exit_triggers=exit_triggers,
                thesis_breakers=response.get("thesis_breakers", []),
                time_horizon=response.get("time_horizon", "weeks"),
                signals_used=signals_used,
                market_context=response.get("market_context", ""),
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return self._create_error_analysis(signals_used, str(e))
    
    def _create_error_analysis(
        self,
        signals_used: List[str],
        error: str,
    ) -> Analysis:
        """Create an error/no-trade analysis."""
        return Analysis(
            analysis_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc),
            ticker=None,
            recommendation=Recommendation.NONE,
            conviction_score=0,
            company_type="unknown",
            business_lines_analyzed=[],
            segments_priced_at_zero=[],
            optionality_value="",
            historical_context=None,
            technical_context="",
            macro_cycle_position="",
            seasonal_factors="",
            thesis=f"Analysis error: {error}",
            catalyst="",
            catalyst_date=None,
            catalyst_horizon="",
            primary_beneficiary=False,
            second_order_rationale="",
            bull_case="",
            bear_case="",
            variant_perception="",
            what_needs_to_be_true_for_bear=[],
            position_size_pct=0,
            entry_strategy="none",
            entry_price_target=None,
            stop_loss_pct=0,
            exit_triggers=[],
            thesis_breakers=[],
            time_horizon="",
            signals_used=signals_used,
            market_context="Error during analysis",
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def analyze_signals(
    signals: List[Any],
    portfolio_context: Dict[str, Any],
    watchlist: List[str],
) -> Dict[str, Any]:
    """Convenience function to analyze signals."""
    analyst = ClaudeAnalyst()
    analysis = await analyst.analyze_signals(signals, portfolio_context, watchlist)
    return analysis.to_dict()


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        analyst = ClaudeAnalyst()
        print(f"Claude Analyst v4.0.0 initialized")
        print(f"API Key configured: {bool(analyst.api_key)}")
        print(f"Historical patterns: {len(analyst.historical_patterns)}")
        print(f"Second-order effects: {len(analyst.second_order_effects)}")
    
    asyncio.run(test())
