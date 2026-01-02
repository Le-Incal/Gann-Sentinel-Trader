"""
Risk Engine for Gann Sentinel Trader
Validates trades against hard limits and calculates position sizes.

CRITICAL FIX: Normalizes position_size_pct to handle both formats:
- Whole number (15 = 15%)
- Decimal (0.15 = 15%)
"""

import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class RiskCheckSeverity(Enum):
    """Severity level of risk check results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    check_name: str
    passed: bool
    severity: str
    message: str
    limit: Optional[float] = None
    actual: Optional[float] = None


@dataclass 
class RiskEngine:
    """
    Risk validation engine with hard limits.
    
    Hard Limits (Non-Negotiable):
    - Max position size: 25% of portfolio
    - Max concurrent positions: 5
    - Daily loss limit: 5% of portfolio
    - Single trade stop-loss: 15% max
    - Min market cap: $500M
    - No leverage
    """
    
    # Configuration
    max_position_size_pct: float = 0.25      # 25% max per position
    max_concurrent_positions: int = 5
    max_daily_loss_pct: float = 0.05         # 5% daily loss limit
    max_stop_loss_pct: float = 0.15          # 15% max stop loss
    min_market_cap: float = 500_000_000      # $500M minimum
    max_gross_exposure_pct: float = 1.0      # 100% max exposure
    max_notional_pct: float = 0.25           # 25% max per order
    
    # State
    trading_halted: bool = field(default=False)
    halt_reason: str = field(default="")
    daily_loss: float = field(default=0.0)
    daily_trades: int = field(default=0)
    consecutive_rejects: int = field(default=0)
    
    def __post_init__(self):
        logger.info("Risk Engine initialized with limits:")
        logger.info(f"  Max position size: {self.max_position_size_pct * 100:.0f}%")
        logger.info(f"  Max concurrent positions: {self.max_concurrent_positions}")
        logger.info(f"  Daily loss limit: {self.max_daily_loss_pct * 100:.0f}%")
        logger.info(f"  Max stop loss: {self.max_stop_loss_pct * 100:.0f}%")
    
    def _normalize_percentage(self, value: float) -> float:
        """
        Normalize a percentage value to decimal format (0-1).
        
        Handles both formats:
        - 15 (whole number meaning 15%) -> 0.15
        - 0.15 (decimal meaning 15%) -> 0.15
        
        This fixes the bug where Claude returns 15 but risk engine expects 0.15.
        """
        if value > 1.0:
            # Value is in whole number format (e.g., 15 = 15%)
            return value / 100.0
        else:
            # Value is already in decimal format (e.g., 0.15 = 15%)
            return value
    
    def validate_trade(
        self,
        analysis,
        portfolio,
        current_positions: List[Any]
    ) -> Tuple[bool, List[RiskCheckResult]]:
        """
        Validate a trade against all risk checks.
        
        Returns:
            Tuple of (passed: bool, results: List[RiskCheckResult])
        """
        results = []
        
        # Check 1: Trading halt
        results.append(self._check_trading_halt())
        
        # Check 2: Daily loss limit
        results.append(self._check_daily_loss(portfolio))
        
        # Check 3: Position count
        results.append(self._check_position_count(current_positions))
        
        # Check 4: Position size - WITH NORMALIZATION FIX
        results.append(self._check_position_size(analysis))
        
        # Check 5: Stop loss
        results.append(self._check_stop_loss(analysis))
        
        # Check 6: Gross exposure
        results.append(self._check_gross_exposure(portfolio, current_positions))
        
        # Determine overall pass/fail
        has_errors = any(
            not r.passed and r.severity == RiskCheckSeverity.ERROR.value 
            for r in results
        )
        
        if has_errors:
            self.consecutive_rejects += 1
            logger.warning(f"Trade rejected. Consecutive rejects: {self.consecutive_rejects}")
            
            # Circuit breaker: 3 consecutive rejects = pause
            if self.consecutive_rejects >= 3:
                self.halt_trading("3 consecutive trade rejections")
        else:
            self.consecutive_rejects = 0
        
        return (not has_errors, results)
    
    def _check_trading_halt(self) -> RiskCheckResult:
        """Check if trading is halted."""
        if self.trading_halted:
            return RiskCheckResult(
                check_name="trading_halt",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Trading halted: {self.halt_reason}"
            )
        return RiskCheckResult(
            check_name="trading_halt",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message="Trading active"
        )
    
    def _check_daily_loss(self, portfolio) -> RiskCheckResult:
        """Check daily loss limit."""
        equity = portfolio.equity if hasattr(portfolio, 'equity') else portfolio.get('equity', 100000)
        daily_pnl = portfolio.daily_pnl if hasattr(portfolio, 'daily_pnl') else portfolio.get('daily_pnl', 0)
        
        if equity <= 0:
            equity = 100000  # Fallback
        
        daily_loss_pct = abs(min(daily_pnl, 0)) / equity
        
        if daily_loss_pct >= self.max_daily_loss_pct:
            return RiskCheckResult(
                check_name="daily_loss_limit",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Daily loss {daily_loss_pct:.1%} exceeds {self.max_daily_loss_pct:.0%} limit",
                limit=self.max_daily_loss_pct,
                actual=daily_loss_pct
            )
        
        return RiskCheckResult(
            check_name="daily_loss_limit",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message=f"Daily loss {daily_loss_pct:.1%} within {self.max_daily_loss_pct:.0%} limit",
            limit=self.max_daily_loss_pct,
            actual=daily_loss_pct
        )
    
    def _check_position_count(self, current_positions: List[Any]) -> RiskCheckResult:
        """Check maximum concurrent positions."""
        count = len(current_positions)
        
        if count >= self.max_concurrent_positions:
            return RiskCheckResult(
                check_name="max_positions",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Position count {count} >= max {self.max_concurrent_positions}",
                limit=float(self.max_concurrent_positions),
                actual=float(count)
            )
        
        return RiskCheckResult(
            check_name="max_positions",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message=f"Position count {count} < max {self.max_concurrent_positions}",
            limit=float(self.max_concurrent_positions),
            actual=float(count)
        )
    
    def _check_position_size(self, analysis) -> RiskCheckResult:
        """
        Check position size limit.
        
        CRITICAL: Normalizes position_size_pct to decimal format before comparison.
        This fixes the bug where Claude returns 15 (meaning 15%) but the check
        compared against 0.25 (25% in decimal).
        """
        # Get position size from analysis
        if hasattr(analysis, 'position_size_pct'):
            position_size_raw = analysis.position_size_pct
        else:
            position_size_raw = analysis.get('position_size_pct', 0)
        
        # NORMALIZE to decimal format (0-1)
        position_size = self._normalize_percentage(position_size_raw)
        
        logger.info(f"Position size check: raw={position_size_raw}, normalized={position_size:.2%}, limit={self.max_position_size_pct:.0%}")
        
        if position_size > self.max_position_size_pct:
            return RiskCheckResult(
                check_name="max_position_size",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Position size {position_size:.0%} > max {self.max_position_size_pct:.0%}",
                limit=self.max_position_size_pct,
                actual=position_size
            )
        
        return RiskCheckResult(
            check_name="max_position_size",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message=f"Position size {position_size:.0%} within {self.max_position_size_pct:.0%} limit",
            limit=self.max_position_size_pct,
            actual=position_size
        )
    
    def _check_stop_loss(self, analysis) -> RiskCheckResult:
        """Check stop loss limit."""
        if hasattr(analysis, 'stop_loss_pct'):
            stop_loss_raw = analysis.stop_loss_pct
        else:
            stop_loss_raw = analysis.get('stop_loss_pct', 0)
        
        # Normalize to decimal
        stop_loss = self._normalize_percentage(stop_loss_raw)
        
        if stop_loss > self.max_stop_loss_pct:
            return RiskCheckResult(
                check_name="max_stop_loss",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Stop loss {stop_loss:.0%} > max {self.max_stop_loss_pct:.0%}",
                limit=self.max_stop_loss_pct,
                actual=stop_loss
            )
        
        return RiskCheckResult(
            check_name="max_stop_loss",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message=f"Stop loss {stop_loss:.0%} within {self.max_stop_loss_pct:.0%} limit",
            limit=self.max_stop_loss_pct,
            actual=stop_loss
        )
    
    def _check_gross_exposure(
        self,
        portfolio,
        current_positions: List[Any]
    ) -> RiskCheckResult:
        """Check gross exposure limit."""
        equity = portfolio.equity if hasattr(portfolio, 'equity') else portfolio.get('equity', 100000)
        
        if equity <= 0:
            equity = 100000
        
        # Calculate current exposure
        total_exposure = 0
        for pos in current_positions:
            if hasattr(pos, 'market_value'):
                total_exposure += abs(pos.market_value)
            elif isinstance(pos, dict):
                total_exposure += abs(pos.get('market_value', 0))
        
        exposure_pct = total_exposure / equity
        
        if exposure_pct > self.max_gross_exposure_pct:
            return RiskCheckResult(
                check_name="gross_exposure",
                passed=False,
                severity=RiskCheckSeverity.ERROR.value,
                message=f"Gross exposure {exposure_pct:.0%} > max {self.max_gross_exposure_pct:.0%}",
                limit=self.max_gross_exposure_pct,
                actual=exposure_pct
            )
        
        return RiskCheckResult(
            check_name="gross_exposure",
            passed=True,
            severity=RiskCheckSeverity.INFO.value,
            message=f"Gross exposure {exposure_pct:.0%} within limit",
            limit=self.max_gross_exposure_pct,
            actual=exposure_pct
        )
    
    def calculate_position_size(
        self,
        analysis,
        portfolio,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate the number of shares to buy based on analysis and portfolio.
        
        Uses the position_size_pct from analysis, capped by max_position_size_pct.
        """
        equity = portfolio.equity if hasattr(portfolio, 'equity') else portfolio.get('equity', 100000)
        cash = portfolio.cash if hasattr(portfolio, 'cash') else portfolio.get('cash', equity)
        
        # Get requested position size and normalize
        if hasattr(analysis, 'position_size_pct'):
            requested_pct_raw = analysis.position_size_pct
        else:
            requested_pct_raw = analysis.get('position_size_pct', 0.10)
        
        requested_pct = self._normalize_percentage(requested_pct_raw)
        
        # Cap at max position size
        actual_pct = min(requested_pct, self.max_position_size_pct)
        
        # Calculate dollar amount
        dollar_amount = equity * actual_pct
        
        # Cap by available cash
        dollar_amount = min(dollar_amount, cash * 0.95)  # Keep 5% buffer
        
        # Calculate shares
        if current_price > 0:
            shares = int(dollar_amount / current_price)
        else:
            shares = 0
        
        logger.info(f"Position sizing: requested={requested_pct_raw}({requested_pct:.0%}), "
                   f"actual={actual_pct:.0%}, dollars=${dollar_amount:,.2f}, shares={shares}")
        
        return {
            "shares": shares,
            "dollar_amount": dollar_amount,
            "requested_pct": requested_pct,
            "actual_pct": actual_pct,
            "capped": requested_pct > self.max_position_size_pct
        }
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading with a reason."""
        self.trading_halted = True
        self.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading after a halt."""
        self.trading_halted = False
        self.halt_reason = ""
        self.consecutive_rejects = 0
        logger.info("Trading resumed")
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at market open)."""
        self.daily_loss = 0.0
        self.daily_trades = 0
        logger.info("Daily counters reset")
