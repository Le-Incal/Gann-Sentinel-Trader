"""
Gann Sentinel Trader - Claude Analyst
Uses Claude API for strategic reasoning and trade decisions.
"""

import logging
import json
import anthropic
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from config import Config
from models.signals import Signal
from models.analysis import Analysis, Recommendation, ExitTrigger

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior quantitative analyst at a hedge fund. Your role is to analyze market signals and generate high-conviction trade recommendations.

## Your Philosophy
1. **Sentiment precedes price.** Crowd psychology often signals moves before price action.
2. **Second-order effects create opportunity.** The best trades are adjacent plays, not obvious ones.
3. **High conviction, low frequency.** Only recommend trades with 80%+ conviction.
4. **Thesis-driven trading.** Every trade needs a clear, falsifiable thesis.

## Your Task
Analyze the provided signals and either:
- Generate a trade recommendation (if conviction >= 80)
- Explain why no trade is warranted (if conviction < 80)

## Your Output Format
Always respond with valid JSON matching this schema:
{
  "ticker": "SYMBOL or null",
  "recommendation": "BUY | SELL | HOLD | NONE",
  "conviction_score": 0-100,
  "thesis": "Clear statement of the investment thesis",
  "bull_case": "Best case scenario",
  "bear_case": "What could go wrong",
  "variant_perception": "What we believe that the market doesn't",
  "position_size_pct": 0.0-0.25,
  "entry_strategy": "market | limit | scale_in",
  "entry_price_target": null or number,
  "stop_loss_pct": 0.10-0.20,
  "time_horizon": "intraday | days | weeks | months",
  "exit_triggers": [
    {"trigger_type": "take_profit | thesis_breaker", "description": "...", "price_target": null}
  ],
  "thesis_breakers": ["What would invalidate this thesis"],
  "key_factors": [{"factor": "...", "weight": 0.0-1.0, "direction": "bullish | bearish"}],
  "reasoning_steps": ["Step by step reasoning"]
}

## Rules
1. NEVER recommend a trade with conviction below 80
2. Be specific about entry and exit criteria
3. Always consider second-order effects
4. Cite specific signals in your reasoning
5. Be honest about uncertainties
6. Position size should reflect conviction (higher conviction = larger size, max 25%)"""


class ClaudeAnalyst:
    """
    Uses Claude for strategic analysis and trade recommendations.
    """
    
    def __init__(self):
        """Initialize Claude analyst."""
        self.api_key = Config.ANTHROPIC_API_KEY
        self.model = Config.CLAUDE_MODEL
        self.client = None
        
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            logger.warning("ANTHROPIC_API_KEY not configured - Claude analyst disabled")
    
    async def analyze_signals(
        self,
        signals: List[Signal],
        portfolio_context: Optional[Dict[str, Any]] = None,
        watchlist: Optional[List[str]] = None
    ) -> Analysis:
        """
        Analyze signals and generate trade recommendation.
        
        Args:
            signals: List of Signal objects to analyze
            portfolio_context: Current portfolio state
            watchlist: Tickers we're specifically watching
            
        Returns:
            Analysis object with recommendation
        """
        if not self.client:
            logger.warning("Claude analyst not configured - returning empty analysis")
            return Analysis(recommendation=Recommendation.NONE)
        
        # Prepare signals for Claude
        signals_text = self._format_signals(signals)
        portfolio_text = self._format_portfolio(portfolio_context)
        watchlist_text = f"Watchlist: {', '.join(watchlist)}" if watchlist else ""
        
        user_prompt = f"""## Current Market Signals

{signals_text}

## Portfolio Context
{portfolio_text}

{watchlist_text}

## Task
Analyze these signals and determine if there's a high-conviction trade opportunity.

Remember:
- Only recommend if conviction >= 80
- Focus on second-order effects (e.g., if SpaceX news is positive, consider Rocket Lab)
- Be specific about thesis and exit criteria
- Position size max 25% of portfolio

Respond with JSON only, no additional text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            analysis_data = self._parse_response(response_text)
            
            # Convert to Analysis object
            analysis = self._to_analysis(analysis_data, signals)
            
            logger.info(f"Analysis complete: {analysis.ticker} - {analysis.recommendation.value} ({analysis.conviction_score})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}")
            return Analysis(
                recommendation=Recommendation.NONE,
                thesis=f"Analysis failed: {str(e)}"
            )
    
    async def analyze_specific_ticker(
        self,
        ticker: str,
        signals: List[Signal],
        portfolio_context: Optional[Dict[str, Any]] = None
    ) -> Analysis:
        """
        Analyze signals specifically for a given ticker.
        
        Args:
            ticker: Stock ticker to analyze
            signals: Relevant signals
            portfolio_context: Current portfolio state
            
        Returns:
            Analysis object
        """
        if not self.client:
            return Analysis(recommendation=Recommendation.NONE)
        
        # Filter signals for this ticker
        relevant_signals = [
            s for s in signals
            if ticker in s.asset_scope.tickers or not s.asset_scope.tickers
        ]
        
        signals_text = self._format_signals(relevant_signals)
        portfolio_text = self._format_portfolio(portfolio_context)
        
        user_prompt = f"""## Analysis Request: {ticker}

## Relevant Signals
{signals_text}

## Portfolio Context
{portfolio_text}

## Task
Provide a focused analysis of {ticker} based on these signals.

Consider:
1. Direct impact of signals on {ticker}
2. Indirect/second-order effects
3. Current valuation and sentiment
4. Risk/reward profile

Respond with JSON only."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text
            analysis_data = self._parse_response(response_text)
            analysis_data["ticker"] = ticker  # Ensure ticker is set
            
            return self._to_analysis(analysis_data, relevant_signals)
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return Analysis(
                ticker=ticker,
                recommendation=Recommendation.NONE,
                thesis=f"Analysis failed: {str(e)}"
            )
    
    async def evaluate_exit(
        self,
        ticker: str,
        entry_thesis: str,
        current_signals: List[Signal],
        current_pnl_pct: float
    ) -> Dict[str, Any]:
        """
        Evaluate whether to exit an existing position.
        
        Args:
            ticker: Position ticker
            entry_thesis: Original thesis for the trade
            current_signals: Current market signals
            current_pnl_pct: Current P&L percentage
            
        Returns:
            Dict with exit recommendation
        """
        if not self.client:
            return {"should_exit": False, "reason": "Analyst not configured"}
        
        signals_text = self._format_signals(current_signals)
        
        user_prompt = f"""## Exit Evaluation: {ticker}

## Original Entry Thesis
{entry_thesis}

## Current P&L
{current_pnl_pct:+.2f}%

## Current Market Signals
{signals_text}

## Task
Evaluate whether to exit this position.

Consider:
1. Is the original thesis still valid?
2. Have any thesis breakers occurred?
3. Is the risk/reward still favorable?
4. Are there better opportunities elsewhere?

Respond with JSON:
{{
  "should_exit": true/false,
  "reason": "explanation",
  "urgency": "immediate | soon | monitor",
  "exit_type": "take_profit | stop_loss | thesis_broken | rebalance"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are a portfolio manager evaluating position exits. Be decisive but thoughtful.",
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text
            return self._parse_response(response_text)
            
        except Exception as e:
            logger.error(f"Error evaluating exit for {ticker}: {e}")
            return {"should_exit": False, "reason": f"Evaluation failed: {str(e)}"}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _format_signals(self, signals: List[Signal]) -> str:
        """Format signals for Claude prompt."""
        if not signals:
            return "No signals available."
        
        formatted = []
        for i, signal in enumerate(signals, 1):
            formatted.append(f"""
### Signal {i}: {signal.signal_type.value.upper()}
- **Source:** {signal.source.value}
- **Summary:** {signal.summary}
- **Direction:** {signal.directional_bias.value}
- **Confidence:** {signal.confidence:.0%}
- **Time Horizon:** {signal.time_horizon.value}
- **Assets:** {', '.join(signal.asset_scope.tickers) or 'Broad market'}
- **Raw Value:** {signal.raw_value.value} {signal.raw_value.unit or ''}
- **Uncertainties:** {'; '.join(signal.uncertainties) if signal.uncertainties else 'None noted'}
- **Age:** {self._signal_age(signal)}
""")
        
        return "\n".join(formatted)
    
    def _format_portfolio(self, portfolio: Optional[Dict[str, Any]]) -> str:
        """Format portfolio context for Claude prompt."""
        if not portfolio:
            return "No portfolio context available."
        
        return f"""
- **Total Value:** ${portfolio.get('total_value', 0):,.2f}
- **Cash:** ${portfolio.get('cash', 0):,.2f}
- **Positions Value:** ${portfolio.get('positions_value', 0):,.2f}
- **Current Positions:** {portfolio.get('position_count', 0)}
- **Daily P&L:** {portfolio.get('daily_pnl_pct', 0):+.2f}%
"""
    
    def _signal_age(self, signal: Signal) -> str:
        """Calculate signal age as human-readable string."""
        age = datetime.now(timezone.utc) - signal.timestamp_utc
        
        if age.total_seconds() < 3600:
            return f"{int(age.total_seconds() / 60)} minutes ago"
        elif age.total_seconds() < 86400:
            return f"{int(age.total_seconds() / 3600)} hours ago"
        else:
            return f"{int(age.total_seconds() / 86400)} days ago"
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from Claude's response."""
        # Try to extract JSON from response
        text = response_text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {text[:500]}")
            return {}
    
    def _to_analysis(
        self,
        data: Dict[str, Any],
        signals: List[Signal]
    ) -> Analysis:
        """Convert parsed response to Analysis object."""
        # Build exit triggers
        exit_triggers = []
        for trigger in data.get("exit_triggers", []):
            exit_triggers.append(ExitTrigger(
                trigger_type=trigger.get("trigger_type", ""),
                description=trigger.get("description", ""),
                price_target=trigger.get("price_target"),
                percentage=trigger.get("percentage")
            ))
        
        return Analysis(
            ticker=data.get("ticker"),
            recommendation=Recommendation(data.get("recommendation", "NONE")),
            conviction_score=data.get("conviction_score", 0),
            thesis=data.get("thesis", ""),
            bull_case=data.get("bull_case", ""),
            bear_case=data.get("bear_case", ""),
            variant_perception=data.get("variant_perception", ""),
            position_size_pct=data.get("position_size_pct", 0.0),
            entry_price_target=data.get("entry_price_target"),
            entry_strategy=data.get("entry_strategy", "market"),
            stop_loss_pct=data.get("stop_loss_pct", 0.15),
            exit_triggers=exit_triggers,
            thesis_breakers=data.get("thesis_breakers", []),
            time_horizon=data.get("time_horizon", "weeks"),
            signals_used=[s.signal_id for s in signals],
            reasoning_steps=data.get("reasoning_steps", []),
            key_factors=data.get("key_factors", [])
        )
