"""
Gann Sentinel Trader - Alpaca Executor
Executes trades via Alpaca API.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import Config
from models.trades import Trade, Position, PortfolioSnapshot, TradeStatus, OrderType, OrderSide as ModelOrderSide

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    """
    Executes trades via Alpaca API.
    Handles both paper and live trading.
    """
    
    def __init__(self):
        """Initialize Alpaca client."""
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        self.base_url = Config.ALPACA_BASE_URL
        self.is_paper = "paper" in self.base_url.lower()
        
        self.trading_client = None
        self.data_client = None
        
        if self.api_key and self.secret_key:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.is_paper
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            logger.info(f"Alpaca executor initialized ({'PAPER' if self.is_paper else 'LIVE'} mode)")
        else:
            logger.warning("Alpaca credentials not configured - executor disabled")
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.trading_client:
            return {"error": "Alpaca not configured"}
        
        try:
            account = self.trading_client.get_account()
            return {
                "id": account.id,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "daytrade_count": account.daytrade_count,
                "status": account.status,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {"error": str(e)}
    
    async def get_positions(self) -> List[Position]:
        """Get all current positions."""
        if not self.trading_client:
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            
            for pos in positions:
                position = Position(
                    ticker=pos.symbol,
                    quantity=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc)
                )
                result.append(position)
            
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        account = await self.get_account()
        positions = await self.get_positions()
        
        if "error" in account:
            return PortfolioSnapshot()
        
        positions_value = sum(p.market_value or 0 for p in positions)
        total_value = float(account.get("portfolio_value", 0))
        cash = float(account.get("cash", 0))
        last_equity = float(account.get("last_equity", total_value))
        
        daily_pnl = total_value - last_equity
        daily_pnl_pct = (daily_pnl / last_equity) if last_equity > 0 else 0
        
        # Calculate largest position
        largest_position_pct = 0
        if total_value > 0 and positions:
            largest_position_pct = max(
                (p.market_value or 0) / total_value for p in positions
            )
        
        return PortfolioSnapshot(
            cash=cash,
            positions_value=positions_value,
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            positions=[p.to_dict() for p in positions],
            position_count=len(positions),
            largest_position_pct=largest_position_pct,
            buying_power=float(account.get("buying_power", 0))
        )
    
    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """Get latest quote for a ticker."""
        if not self.data_client:
            return {"error": "Data client not configured"}
        
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if ticker in quotes:
                quote = quotes[ticker]
                return {
                    "ticker": ticker,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "mid": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "timestamp": quote.timestamp.isoformat()
                }
            return {"error": f"No quote found for {ticker}"}
        except Exception as e:
            logger.error(f"Error getting quote for {ticker}: {e}")
            return {"error": str(e)}
    
    async def submit_order(self, trade: Trade) -> Trade:
        """
        Submit an order to Alpaca.
        
        Args:
            trade: Trade object to execute
            
        Returns:
            Updated Trade object with order details
        """
        if not self.trading_client:
            trade.status = TradeStatus.FAILED
            return trade
        
        try:
            # Determine order side
            side = OrderSide.BUY if trade.side == ModelOrderSide.BUY else OrderSide.SELL
            
            # Create order request based on type
            if trade.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            elif trade.order_type == OrderType.LIMIT:
                order_request = LimitOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=trade.limit_price
                )
            elif trade.order_type == OrderType.STOP:
                order_request = StopOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=trade.stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {trade.order_type}")
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Update trade with order details
            trade.alpaca_order_id = order.id
            trade.status = self._map_order_status(order.status)
            trade.submitted_at = datetime.now(timezone.utc)
            
            # If filled immediately (market orders often are)
            if order.status == OrderStatus.FILLED:
                trade.fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
                trade.fill_quantity = float(order.filled_qty) if order.filled_qty else None
                trade.filled_at = order.filled_at
            
            logger.info(f"Order submitted: {trade.ticker} {trade.side.value} {trade.quantity} - Status: {trade.status.value}")
            return trade
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            trade.status = TradeStatus.FAILED
            trade.rejection_reason = str(e)
            return trade
    
    async def get_order_status(self, alpaca_order_id: str) -> Dict[str, Any]:
        """Get status of an order."""
        if not self.trading_client:
            return {"error": "Alpaca not configured"}
        
        try:
            order = self.trading_client.get_order_by_id(alpaca_order_id)
            return {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "status": order.status.value,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None
            }
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"error": str(e)}
    
    async def cancel_order(self, alpaca_order_id: str) -> bool:
        """Cancel an order."""
        if not self.trading_client:
            return False
        
        try:
            self.trading_client.cancel_order_by_id(alpaca_order_id)
            logger.info(f"Order cancelled: {alpaca_order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if not self.trading_client:
            return 0
        
        try:
            cancelled = self.trading_client.cancel_orders()
            count = len(cancelled)
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def close_position(self, ticker: str) -> bool:
        """Close a position entirely."""
        if not self.trading_client:
            return False
        
        try:
            self.trading_client.close_position(ticker)
            logger.info(f"Position closed: {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {ticker}: {e}")
            return False
    
    async def close_all_positions(self) -> int:
        """Close all positions (emergency flatten)."""
        if not self.trading_client:
            return 0
        
        try:
            closed = self.trading_client.close_all_positions(cancel_orders=True)
            count = len(closed)
            logger.warning(f"EMERGENCY: Closed {count} positions")
            return count
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0
    
    def _map_order_status(self, alpaca_status: OrderStatus) -> TradeStatus:
        """Map Alpaca order status to our TradeStatus."""
        mapping = {
            OrderStatus.NEW: TradeStatus.SUBMITTED,
            OrderStatus.ACCEPTED: TradeStatus.SUBMITTED,
            OrderStatus.PENDING_NEW: TradeStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED: TradeStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED: TradeStatus.FILLED,
            OrderStatus.CANCELED: TradeStatus.CANCELLED,
            OrderStatus.REJECTED: TradeStatus.REJECTED,
            OrderStatus.EXPIRED: TradeStatus.CANCELLED,
        }
        return mapping.get(alpaca_status, TradeStatus.SUBMITTED)
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        if not self.trading_client:
            return False
        
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours for today."""
        if not self.trading_client:
            return {"error": "Alpaca not configured"}
        
        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat() if clock.next_open else None,
                "next_close": clock.next_close.isoformat() if clock.next_close else None
            }
        except Exception as e:
            logger.error(f"Error getting market hours: {e}")
            return {"error": str(e)}
