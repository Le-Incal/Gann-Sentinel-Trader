"""
Gann Sentinel Trader - Risk Engine
Validates trades against risk rules before execution.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from config import Config
from models.analysis import Analysis, Recommendation
from models.trades import Trade, Position, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    passed: bool
    rule: str
    message: str
    severity: str = "error"  # error, warning, info


class RiskEngine:
    """
    Validates trades against risk rules.
    All trades must pass the risk engine before execution.
    """
    
    def __init__(self):
        """Initialize risk engine with config."""
        self.max_position_pct = Config.MAX_POSITION_PCT
        self.max_positions = Config.MAX_POSITIONS
        self.stop_loss_pct = Config.STOP_LOSS_PCT
        self.daily_loss_limit_pct = Config.DAILY_LOSS_LIMIT_PCT
        self.min_market_cap = Config.MIN_MARKET_CAP
        self.min_conviction = Config.MIN_CONVICTION
        
        # Track daily losses
        self.daily_pnl = 0.0
        self.daily_pnl_reset_date = datetime.now(timezone.utc).date()
        
        # Track trading halts
        self.trading_halted = False
        self.halt_reason = ""
    
    def validate_trade(
        self,
        analysis: Analysis,
        portfolio: PortfolioSnapshot,
        current_positions: List[Position]
    ) -> Tuple[bool, List[RiskCheckResult]]:
        """
        Validate a trade against all risk rules.
        
        Args:
            analysis: The analysis generating the trade
            portfolio: Current portfolio state
            current_positions: List of current positions
            
        Returns:
            Tuple of (passed, list of check results)
        """
        results = []
        
        # Run all checks
        results.append(self._check_trading_halted())
        results.append(self._check_conviction(analysis))
        results.append(self._check_position_size(analysis, portfolio))
        results.append(self._check_position_count(current_positions))
        results.append(self._check_daily_loss_limit(portfolio))
        results.append(self._check_existing_position(analysis.ticker, current_positions))
        results.append(self._check_stop_loss_defined(analysis))
        
        # Additional checks can be added here
        # results.append(self._check_market_cap(analysis.ticker))
        # results.append(self._check_correlation(analysis.ticker, current_positions))
        
        # Determine if all critical checks passed
        critical_failures = [r for r in results if not r.passed and r.severity == "error"]
        passed = len(critical_failures) == 0
        
        # Log results
        for result in results:
            if not result.passed:
                if result.severity == "error":
                    logger.error(f"Risk check FAILED: {result.rule} - {result.message}")
                else:
                    logger.warning(f"Risk check WARNING: {result.rule} - {result.message}")
            else:
                logger.debug(f"Risk check passed: {result.rule}")
        
        return passed, results
    
    def calculate_position_size(
        self,
        analysis: Analysis,
        portfolio: PortfolioSnapshot,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on analysis and risk rules.
        
        Args:
            analysis: The trade analysis
            portfolio: Current portfolio state
            current_price: Current price of the asset
            
        Returns:
            Dict with shares, dollar_amount, and percentage
        """
        # Start with suggested size from analysis
        suggested_pct = min(analysis.position_size_pct, self.max_position_pct)
        
        # Scale by conviction (higher conviction = closer to max)
        conviction_factor = analysis.conviction_score / 100
        adjusted_pct = suggested_pct * conviction_factor
        
        # Ensure we don't exceed max
        final_pct = min(adjusted_pct, self.max_position_pct)
        
        # Calculate dollar amount
        available_cash = portfolio.cash
        portfolio_value = portfolio.total_value
        
        dollar_amount = portfolio_value * final_pct
        
        # Don't exceed available cash
        dollar_amount = min(dollar_amount, available_cash * 0.95)  # Keep 5% buffer
        
        # Calculate shares
        shares = int(dollar_amount / current_price)
        actual_dollar_amount = shares * current_price
        actual_pct = actual_dollar_amount / portfolio_value
        
        return {
            "shares": shares,
            "dollar_amount": actual_dollar_amount,
            "percentage": actual_pct,
            "current_price": current_price,
            "conviction_factor": conviction_factor
        }
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        analysis: Analysis
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            analysis: Trade analysis
            
        Returns:
            Stop loss price
        """
        stop_loss_pct = analysis.stop_loss_pct or self.stop_loss_pct
        return entry_price * (1 - stop_loss_pct)
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking."""
        today = datetime.now(timezone.utc).date()
        
        if today != self.daily_pnl_reset_date:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_date = today
            
            # Reset halt if it was due to daily loss
            if self.trading_halted and "daily loss" in self.halt_reason.lower():
                self.trading_halted = False
                self.halt_reason = ""
        
        self.daily_pnl += pnl
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading."""
        self.trading_halted = True
        self.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading after halt."""
        self.trading_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")
    
    # =========================================================================
    # INDIVIDUAL RISK CHECKS
    # =========================================================================
    
    def _check_trading_halted(self) -> RiskCheckResult:
        """Check if trading is halted."""
        if self.trading_halted:
            return RiskCheckResult(
                passed=False,
                rule="trading_halt",
                message=f"Trading is halted: {self.halt_reason}",
                severity="error"
            )
        return RiskCheckResult(
            passed=True,
            rule="trading_halt",
            message="Trading is active"
        )
    
    def _check_conviction(self, analysis: Analysis) -> RiskCheckResult:
        """Check if conviction meets minimum threshold."""
        if analysis.conviction_score < self.min_conviction:
            return RiskCheckResult(
                passed=False,
                rule="min_conviction",
                message=f"Conviction {analysis.conviction_score} < minimum {self.min_conviction}",
                severity="error"
            )
        return RiskCheckResult(
            passed=True,
            rule="min_conviction",
            message=f"Conviction {analysis.conviction_score} >= {self.min_conviction}"
        )
    
    def _check_position_size(
        self,
        analysis: Analysis,
        portfolio: PortfolioSnapshot
    ) -> RiskCheckResult:
        """Check if position size is within limits."""
        if analysis.position_size_pct > self.max_position_pct:
            return RiskCheckResult(
                passed=False,
                rule="max_position_size",
                message=f"Position size {analysis.position_size_pct:.1%} > max {self.max_position_pct:.1%}",
                severity="error"
            )
        return RiskCheckResult(
            passed=True,
            rule="max_position_size",
            message=f"Position size {analysis.position_size_pct:.1%} within limit"
        )
    
    def _check_position_count(
        self,
        current_positions: List[Position]
    ) -> RiskCheckResult:
        """Check if we're at max positions."""
        position_count = len([p for p in current_positions if p.quantity > 0])
        
        if position_count >= self.max_positions:
            return RiskCheckResult(
                passed=False,
                rule="max_positions",
                message=f"At max positions ({position_count}/{self.max_positions})",
                severity="error"
            )
        return RiskCheckResult(
            passed=True,
            rule="max_positions",
            message=f"Position count {position_count}/{self.max_positions}"
        )
    
    def _check_daily_loss_limit(
        self,
        portfolio: PortfolioSnapshot
    ) -> RiskCheckResult:
        """Check if daily loss limit has been hit."""
        if portfolio.total_value > 0:
            daily_loss_pct = abs(min(0, portfolio.daily_pnl)) / portfolio.total_value
            
            if daily_loss_pct >= self.daily_loss_limit_pct:
                self.halt_trading(f"Daily loss limit hit: {daily_loss_pct:.1%}")
                return RiskCheckResult(
                    passed=False,
                    rule="daily_loss_limit",
                    message=f"Daily loss {daily_loss_pct:.1%} >= limit {self.daily_loss_limit_pct:.1%}",
                    severity="error"
                )
        
        return RiskCheckResult(
            passed=True,
            rule="daily_loss_limit",
            message="Daily loss within limit"
        )
    
    def _check_existing_position(
        self,
        ticker: str,
        current_positions: List[Position]
    ) -> RiskCheckResult:
        """Check if we already have a position in this ticker."""
        for position in current_positions:
            if position.ticker == ticker and position.quantity > 0:
                return RiskCheckResult(
                    passed=True,  # Not a failure, just a warning
                    rule="existing_position",
                    message=f"Already have position in {ticker}: {position.quantity} shares",
                    severity="warning"
                )
        
        return RiskCheckResult(
            passed=True,
            rule="existing_position",
            message=f"No existing position in {ticker}"
        )
    
    def _check_stop_loss_defined(self, analysis: Analysis) -> RiskCheckResult:
        """Check if stop loss is defined."""
        if analysis.stop_loss_pct is None or analysis.stop_loss_pct <= 0:
            return RiskCheckResult(
                passed=False,
                rule="stop_loss_required",
                message="Stop loss must be defined",
                severity="error"
            )
        
        if analysis.stop_loss_pct > self.stop_loss_pct:
            return RiskCheckResult(
                passed=True,
                rule="stop_loss_required",
                message=f"Stop loss {analysis.stop_loss_pct:.1%} > recommended {self.stop_loss_pct:.1%}",
                severity="warning"
            )
        
        return RiskCheckResult(
            passed=True,
            rule="stop_loss_required",
            message=f"Stop loss defined at {analysis.stop_loss_pct:.1%}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_reset_date": self.daily_pnl_reset_date.isoformat(),
            "config": {
                "max_position_pct": self.max_position_pct,
                "max_positions": self.max_positions,
                "stop_loss_pct": self.stop_loss_pct,
                "daily_loss_limit_pct": self.daily_loss_limit_pct,
                "min_conviction": self.min_conviction
            }
        }
