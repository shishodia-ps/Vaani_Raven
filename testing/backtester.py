"""
Backtesting Engine - Comprehensive Strategy Testing
Includes slippage, fees, walk-forward validation, and out-of-sample testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import yaml

from orchestrator.runner import OrchestatorRunner
from data_pipeline.data_collector import DataCollector

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    commission_per_trade: float = 7.0
    spread_cost: float = 0.0001
    slippage_factor: float = 0.5
    max_slippage: float = 0.0005
    walk_forward_window: int = 252
    out_of_sample_ratio: float = 0.2
    rebalance_frequency: str = 'monthly'

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    signal: str
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    commission: float
    slippage: float
    profit: Optional[float]
    return_pct: Optional[float]
    confidence: float
    agent_signals: List[Dict[str, Any]]

class BacktestEngine:
    """Comprehensive backtesting engine for VAANI-RAVEN X"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.logger = self._setup_logger()
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        
        self.orchestrator: Optional[OrchestatorRunner] = None
        self.data_collector: Optional[DataCollector] = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup backtesting logger"""
        logger = logging.getLogger("vaani_raven_x.backtester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - BACKTESTER - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_system(self, config_path: str = "config/system_config.yaml") -> bool:
        """Initialize trading system for backtesting"""
        try:
            self.orchestrator = OrchestatorRunner(config_path)
            
            config = self.orchestrator.config
            self.data_collector = DataCollector(config)
            
            self.logger.info("Backtesting system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def run_backtest(self, data: pd.DataFrame, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        if not self.orchestrator:
            raise ValueError("System not initialized")
        
        self.logger.info("Starting comprehensive backtest...")
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        self.logger.info(f"Backtesting on {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        
        self._reset_backtest_state()
        
        open_positions: List[Trade] = []
        
        for i in range(100, len(data)):
            current_time = data.index[i]
            current_data = data.iloc[i-100:i+1]
            
            market_data = self._prepare_market_data(current_data)
            
            result = await self.orchestrator.process_market_data(market_data)
            
            signal = result.get('final_signal', 'HOLD')
            confidence = result.get('confidence', 0.0)
            agent_signals = result.get('agent_signals', [])
            
            current_price = float(current_data.iloc[-1]['close'])
            
            self._close_positions_if_needed(open_positions, current_time, current_price)
            
            if signal in ['BUY', 'SELL'] and confidence > 0.3:
                trade = self._open_position(
                    signal, current_time, current_price, 
                    confidence, agent_signals
                )
                if trade:
                    open_positions.append(trade)
            
            self._update_equity_curve(current_time, current_price, open_positions)
        
        self._close_all_positions(open_positions, data.index[-1], float(data.iloc[-1]['close']))
        
        results = self._calculate_performance_metrics()
        
        self.logger.info("Backtest completed")
        return results
    
    def run_walk_forward_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        
        self.logger.info("Starting walk-forward analysis...")
        
        window_size = self.config.walk_forward_window
        out_of_sample_size = int(window_size * self.config.out_of_sample_ratio)
        
        results = []
        
        for start_idx in range(0, len(data) - window_size, out_of_sample_size):
            end_idx = start_idx + window_size
            
            if end_idx >= len(data):
                break
            
            train_data = data.iloc[start_idx:end_idx - out_of_sample_size]
            test_data = data.iloc[end_idx - out_of_sample_size:end_idx]
            
            self.logger.info(f"Walk-forward window: {train_data.index[0]} to {test_data.index[-1]}")
            
            window_result = asyncio.run(self.run_backtest(
                test_data, 
                test_data.index[0], 
                test_data.index[-1]
            ))
            
            window_result['period_start'] = test_data.index[0]
            window_result['period_end'] = test_data.index[-1]
            
            results.append(window_result)
        
        combined_results = self._combine_walk_forward_results(results)
        
        self.logger.info("Walk-forward analysis completed")
        return combined_results
    
    def _reset_backtest_state(self) -> None:
        """Reset backtest state"""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
    
    def _prepare_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare market data for orchestrator"""
        
        ohlcv_data = []
        for _, row in data.tail(20).iterrows():
            ohlcv_data.append({
                'timestamp': row.name,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row.get('volume', 100000))
            })
        
        current_price = float(data.iloc[-1]['close'])
        
        returns = data['close'].pct_change().dropna()
        volatility = float(returns.tail(20).std()) if len(returns) >= 20 else 0.015
        
        atr = self._calculate_atr(data.tail(14)) if len(data) >= 14 else current_price * 0.001
        
        return {
            'current_price': current_price,
            'bid': current_price - self.config.spread_cost / 2,
            'ask': current_price + self.config.spread_cost / 2,
            'spread': self.config.spread_cost,
            'volume': 100000,
            'volatility': volatility,
            'atr': atr,
            'ohlcv': ohlcv_data,
            'timestamp': data.index[-1],
            'source': 'backtest'
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        return float(np.mean(tr_list)) if tr_list else 0.001
    
    def _open_position(self, signal: str, entry_time: datetime, 
                      entry_price: float, confidence: float,
                      agent_signals: List[Dict[str, Any]]) -> Optional[Trade]:
        """Open new trading position"""
        
        position_size = self._calculate_position_size(confidence)
        
        if position_size <= 0:
            return None
        
        slippage = self._calculate_slippage(position_size)
        commission = self.config.commission_per_trade
        
        if signal == 'BUY':
            actual_entry_price = entry_price + slippage
        else:
            actual_entry_price = entry_price - slippage
        
        stop_loss, take_profit = self._calculate_sl_tp(signal, actual_entry_price, confidence)
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=None,
            signal=signal,
            entry_price=actual_entry_price,
            exit_price=None,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
            slippage=slippage,
            profit=None,
            return_pct=None,
            confidence=confidence,
            agent_signals=agent_signals
        )
        
        self.current_capital -= commission
        
        self.logger.debug(f"Opened {signal} position: {position_size:.2f} lots at {actual_entry_price:.5f}")
        
        return trade
    
    def _close_positions_if_needed(self, open_positions: List[Trade], 
                                  current_time: datetime, current_price: float) -> None:
        """Close positions based on stop loss, take profit, or time"""
        
        positions_to_close = []
        
        for trade in open_positions:
            should_close = False
            exit_reason = ""
            
            if trade.signal == 'BUY':
                if trade.stop_loss and current_price <= trade.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif trade.take_profit and current_price >= trade.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            else:
                if trade.stop_loss and current_price >= trade.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif trade.take_profit and current_price <= trade.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            
            time_diff = (current_time - trade.entry_time).total_seconds() / 3600
            if time_diff > 24:
                should_close = True
                exit_reason = "Time Exit"
            
            if should_close:
                self._close_position(trade, current_time, current_price, exit_reason)
                positions_to_close.append(trade)
        
        for trade in positions_to_close:
            open_positions.remove(trade)
    
    def _close_position(self, trade: Trade, exit_time: datetime, 
                       exit_price: float, exit_reason: str) -> None:
        """Close trading position"""
        
        slippage = self._calculate_slippage(trade.position_size)
        commission = self.config.commission_per_trade
        
        if trade.signal == 'BUY':
            actual_exit_price = exit_price - slippage
            profit = (actual_exit_price - trade.entry_price) * trade.position_size
        else:
            actual_exit_price = exit_price + slippage
            profit = (trade.entry_price - actual_exit_price) * trade.position_size
        
        profit -= commission
        
        trade.exit_time = exit_time
        trade.exit_price = actual_exit_price
        trade.profit = profit
        trade.return_pct = profit / (trade.entry_price * trade.position_size)
        
        self.current_capital += profit
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        self.trades.append(trade)
        
        self.logger.debug(f"Closed {trade.signal} position: {profit:.2f} profit ({exit_reason})")
    
    def _close_all_positions(self, open_positions: List[Trade], 
                           exit_time: datetime, exit_price: float) -> None:
        """Close all remaining open positions"""
        
        for trade in open_positions:
            self._close_position(trade, exit_time, exit_price, "End of Backtest")
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk management"""
        
        base_risk = 0.02
        adjusted_risk = base_risk * confidence
        
        risk_amount = self.current_capital * adjusted_risk
        
        position_size = risk_amount / (self.config.spread_cost * 100000)
        
        min_size = 0.01
        max_size = self.current_capital * 0.1 / 100000
        
        return max(min_size, min(position_size, max_size))
    
    def _calculate_slippage(self, position_size: float) -> float:
        """Calculate realistic slippage based on position size"""
        
        base_slippage = self.config.spread_cost * self.config.slippage_factor
        
        size_impact = position_size * 0.00001
        
        total_slippage = base_slippage + size_impact
        
        return min(total_slippage, self.config.max_slippage)
    
    def _calculate_sl_tp(self, signal: str, entry_price: float, 
                        confidence: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        
        atr_multiplier = 2.0
        tp_multiplier = 3.0
        
        atr_estimate = entry_price * 0.001
        
        confidence_adjustment = 0.5 + confidence * 0.5
        sl_distance = atr_estimate * atr_multiplier * confidence_adjustment
        tp_distance = atr_estimate * tp_multiplier * confidence_adjustment
        
        if signal == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def _update_equity_curve(self, timestamp: datetime, current_price: float,
                           open_positions: List[Trade]) -> None:
        """Update equity curve with current portfolio value"""
        
        unrealized_pnl = 0.0
        
        for trade in open_positions:
            if trade.signal == 'BUY':
                unrealized_pnl += (current_price - trade.entry_price) * trade.position_size
            else:
                unrealized_pnl += (trade.entry_price - current_price) * trade.position_size
        
        total_equity = self.current_capital + unrealized_pnl
        drawdown = (self.peak_capital - total_equity) / self.peak_capital if self.peak_capital > 0 else 0
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'realized_pnl': self.current_capital - self.config.initial_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown': drawdown,
            'open_positions': len(open_positions)
        })
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {'error': 'No trades executed'}
        
        total_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if trade.profit and trade.profit > 0)
        
        total_profit = sum(trade.profit for trade in self.trades if trade.profit)
        total_return = total_profit / self.config.initial_capital
        
        win_rate = profitable_trades / total_trades
        
        profits = [trade.profit for trade in self.trades if trade.profit and trade.profit > 0]
        losses = [trade.profit for trade in self.trades if trade.profit and trade.profit < 0]
        
        avg_win = float(np.mean(profits)) if profits else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
        
        returns = [trade.return_pct for trade in self.trades if trade.return_pct is not None]
        
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio()
        
        max_drawdown = max(point['drawdown'] for point in self.equity_curve) if self.equity_curve else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'final_equity': self.current_capital,
            'equity_curve': self.equity_curve,
            'trades': [self._trade_to_dict(trade) for trade in self.trades]
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / std_return * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        if not self.equity_curve:
            return 0.0
        
        max_drawdown = max(point['drawdown'] for point in self.equity_curve)
        
        if max_drawdown == 0:
            return float('inf') if self.current_capital > self.config.initial_capital else 0.0
        
        annual_return = (self.current_capital / self.config.initial_capital - 1) * 252 / len(self.equity_curve)
        
        return annual_return / max_drawdown
    
    def _combine_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine walk-forward analysis results"""
        
        combined_metrics = {
            'total_periods': len(results),
            'avg_win_rate': np.mean([r['win_rate'] for r in results]),
            'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
            'avg_sortino_ratio': np.mean([r['sortino_ratio'] for r in results]),
            'avg_calmar_ratio': np.mean([r['calmar_ratio'] for r in results]),
            'consistency_score': np.std([r['total_return'] for r in results]),
            'period_results': results
        }
        
        return combined_metrics
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'signal': trade.signal,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'profit': trade.profit,
            'return_pct': trade.return_pct,
            'confidence': trade.confidence,
            'commission': trade.commission,
            'slippage': trade.slippage
        }
    
    def save_results(self, results: Dict[str, Any], filename: str) -> bool:
        """Save backtest results to file"""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            filepath = results_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            
            self.logger.info(f"Results saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False

import asyncio

async def main():
    """Main function for testing backtester"""
    
    backtester = BacktestEngine()
    
    if not backtester.initialize_system():
        print("Failed to initialize system")
        return
    
    if backtester.data_collector:
        historical_data = backtester.data_collector.get_historical_data(1)
        
        if not historical_data.empty:
            results = await backtester.run_backtest(historical_data)
            
            print("Backtest Results:")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"Total Return: {results.get('total_return', 0):.1%}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.1%}")
            
            backtester.save_results(results, "backtest_results.yaml")

if __name__ == "__main__":
    asyncio.run(main())
