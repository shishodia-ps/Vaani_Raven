"""
Risk Agent - Layer 4
Risk Management using Kelly Criterion and Dynamic Position Sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class RiskAgent(BaseAgent):
    """Layer 4: Risk Management Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "risk_agent")
        
        self.kelly_criterion = config.get('kelly_criterion', True)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.equity_cutoff = config.get('equity_cutoff', 0.8)
        self.dynamic_sizing = config.get('dynamic_sizing', True)
        
        self.current_equity = 10000.0
        self.initial_equity = 10000.0
        self.peak_equity = 10000.0
        self.open_positions = []
        self.trade_history = []
        
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process risk management and position sizing"""
        
        if not self.enabled:
            return self._create_neutral_signal("Agent disabled")
        
        try:
            sentiment_signal = None
            for signal in previous_signals:
                if signal.get('agent') == 'sentiment_agent':
                    sentiment_signal = signal
                    break
            
            if not sentiment_signal or sentiment_signal.get('signal') in ['HOLD', 'HALT']:
                return self._create_neutral_signal("No sentiment signal to process")
            
            current_price = market_data.get('current_price', 1.0)
            
            risk_assessment = self._assess_risk_conditions()
            
            if risk_assessment['halt_trading']:
                return self._create_halt_signal(risk_assessment['reason'])
            
            position_size = self._calculate_position_size(
                sentiment_signal, market_data, risk_assessment
            )
            
            stop_loss, take_profit = self._calculate_sl_tp(
                sentiment_signal, current_price, market_data
            )
            
            signal = {
                'agent': self.name,
                'signal': sentiment_signal['signal'],
                'confidence': sentiment_signal['confidence'] * risk_assessment['confidence_multiplier'],
                'timestamp': datetime.now(),
                'metadata': {
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_assessment': risk_assessment,
                    'current_equity': self.current_equity,
                    'drawdown': self._calculate_current_drawdown(),
                    'risk_per_trade': self._calculate_risk_per_trade(position_size, current_price, stop_loss),
                    'sentiment_signal_confidence': sentiment_signal.get('confidence', 0)
                }
            }
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {e}")
            return self._create_neutral_signal(f"Risk management error: {str(e)}")
    
    def _assess_risk_conditions(self) -> Dict[str, Any]:
        """Assess current risk conditions"""
        
        current_drawdown = self._calculate_current_drawdown()
        equity_ratio = self.current_equity / self.initial_equity
        
        halt_trading = False
        reason = ""
        confidence_multiplier = 1.0
        
        if current_drawdown > self.max_drawdown:
            halt_trading = True
            reason = f"Maximum drawdown exceeded: {current_drawdown:.2%}"
        
        elif equity_ratio < self.equity_cutoff:
            halt_trading = True
            reason = f"Equity cutoff reached: {equity_ratio:.2%}"
        
        elif len(self.open_positions) >= 3:
            halt_trading = True
            reason = "Maximum number of open positions reached"
        
        if current_drawdown > self.max_drawdown * 0.7:
            confidence_multiplier *= 0.5
        elif current_drawdown > self.max_drawdown * 0.5:
            confidence_multiplier *= 0.7
        
        volatility_risk = self._assess_volatility_risk()
        if volatility_risk > 0.8:
            confidence_multiplier *= 0.6
        
        return {
            'halt_trading': halt_trading,
            'reason': reason,
            'confidence_multiplier': confidence_multiplier,
            'current_drawdown': current_drawdown,
            'equity_ratio': equity_ratio,
            'volatility_risk': volatility_risk
        }
    
    def _calculate_position_size(self, sentiment_signal: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               risk_assessment: Dict[str, Any]) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        
        base_risk = self.max_risk_per_trade
        
        if self.dynamic_sizing:
            confidence = sentiment_signal.get('confidence', 0.5)
            base_risk *= confidence
        
        if self.kelly_criterion:
            kelly_fraction = self._calculate_kelly_fraction()
            base_risk = min(base_risk, kelly_fraction)
        
        equity_adjustment = min(1.0, self.current_equity / self.initial_equity)
        adjusted_risk = base_risk * equity_adjustment
        
        volatility_adjustment = 1.0 - risk_assessment.get('volatility_risk', 0) * 0.5
        final_risk = adjusted_risk * volatility_adjustment
        
        current_price = market_data.get('current_price', 1.0)
        atr = market_data.get('atr', current_price * 0.01)
        
        risk_amount = self.current_equity * final_risk
        position_size = risk_amount / (atr * 2)
        
        min_size = 0.01
        max_size = self.current_equity * 0.1 / current_price
        
        return max(min_size, min(position_size, max_size))
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction based on historical performance"""
        
        if len(self.trade_history) < 10:
            return self.max_risk_per_trade
        
        recent_trades = self.trade_history[-50:]
        
        wins = [t for t in recent_trades if t.get('profit', 0) > 0]
        losses = [t for t in recent_trades if t.get('profit', 0) < 0]
        
        if not wins or not losses:
            return self.max_risk_per_trade * 0.5
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(t['profit'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
        
        if avg_loss == 0:
            return self.max_risk_per_trade * 0.5
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        return max(0, min(kelly_fraction * 0.25, self.max_risk_per_trade))
    
    def _calculate_sl_tp(self, sentiment_signal: Dict[str, Any], 
                        current_price: float, market_data: Dict[str, Any]) -> tuple:
        """Calculate stop loss and take profit levels"""
        
        signal_type = sentiment_signal.get('signal', 'HOLD')
        confidence = sentiment_signal.get('confidence', 0.5)
        
        atr = market_data.get('atr', current_price * 0.01)
        volatility_regime = market_data.get('volatility_regime', {})
        
        base_sl_multiplier = 2.0
        base_tp_multiplier = 3.0
        
        if volatility_regime.get('volatility') == 'high':
            base_sl_multiplier *= 1.5
            base_tp_multiplier *= 1.2
        elif volatility_regime.get('volatility') == 'low':
            base_sl_multiplier *= 0.8
            base_tp_multiplier *= 1.5
        
        confidence_adjustment = 0.5 + confidence * 0.5
        sl_distance = atr * base_sl_multiplier * confidence_adjustment
        tp_distance = atr * base_tp_multiplier * confidence_adjustment
        
        if signal_type == 'BUY':
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        elif signal_type == 'SELL':
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        else:
            stop_loss = current_price
            take_profit = current_price
        
        return stop_loss, take_profit
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity"""
        if self.peak_equity == 0:
            return 0.0
        
        return max(0, (self.peak_equity - self.current_equity) / self.peak_equity)
    
    def _calculate_risk_per_trade(self, position_size: float, 
                                 current_price: float, stop_loss: float) -> float:
        """Calculate actual risk per trade as percentage of equity"""
        
        if self.current_equity == 0:
            return 0.0
        
        risk_amount = abs(current_price - stop_loss) * position_size
        return risk_amount / self.current_equity
    
    def _assess_volatility_risk(self) -> float:
        """Assess volatility-based risk level"""
        
        if len(self.trade_history) < 5:
            return 0.5
        
        recent_trades = self.trade_history[-20:]
        returns = [t.get('return', 0) for t in recent_trades]
        
        if not returns:
            return 0.5
        
        volatility = np.std(returns)
        
        if volatility > 0.05:
            return 1.0
        elif volatility > 0.03:
            return 0.8
        elif volatility > 0.02:
            return 0.6
        elif volatility > 0.01:
            return 0.4
        else:
            return 0.2
    
    def update_equity(self, new_equity: float) -> None:
        """Update current equity and peak equity"""
        self.current_equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
    
    def add_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Add trade result to history"""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'profit': trade_result.get('profit', 0),
            'return': trade_result.get('return', 0),
            'duration': trade_result.get('duration', 0)
        })
        
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def add_position(self, position: Dict[str, Any]) -> None:
        """Add open position"""
        self.open_positions.append(position)
    
    def remove_position(self, position_id: str) -> None:
        """Remove closed position"""
        self.open_positions = [p for p in self.open_positions if p.get('id') != position_id]
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'position_size': 0.0,
                'current_equity': self.current_equity
            }
        }
    
    def _create_halt_signal(self, reason: str) -> Dict[str, Any]:
        """Create a halt signal due to risk conditions"""
        return {
            'agent': self.name,
            'signal': 'HALT',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'halt_triggered': True,
                'current_equity': self.current_equity,
                'drawdown': self._calculate_current_drawdown()
            }
        }
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and content"""
        required_fields = ['agent', 'signal', 'confidence', 'timestamp']
        
        if not all(field in signal for field in required_fields):
            return False
        
        if signal['signal'] not in ['BUY', 'SELL', 'HOLD', 'HALT']:
            return False
        
        if not 0 <= signal['confidence'] <= 1:
            return False
        
        return True
