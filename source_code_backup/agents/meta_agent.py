"""
Meta Agent - Layer 6
Meta-Learning Agent for Performance Evaluation and Strategy Selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class MetaAgent(BaseAgent):
    """Layer 6: Meta-Learning Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "meta_agent")
        
        self.evaluation_metrics = config.get('evaluation_metrics', ['sharpe', 'sortino', 'calmar'])
        self.retraining_frequency = config.get('retraining_frequency', 'monthly')
        self.fallback_strategies = config.get('fallback_strategies', ['grid', 'martingale'])
        self.performance_threshold = config.get('performance_threshold', 0.5)
        
        self.agent_performance = {}
        self.strategy_performance = {}
        self.retraining_schedule = {}
        self.fallback_active = False
        
        self._initialize_performance_tracking()
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking for all agents"""
        agent_names = ['pattern_agent', 'quant_agent', 'sentiment_agent', 'risk_agent', 'execution_agent']
        
        for agent_name in agent_names:
            self.agent_performance[agent_name] = {
                'trades': [],
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'last_evaluation': datetime.now(),
                'needs_retraining': False
            }
    
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process meta-learning evaluation and strategy selection"""
        
        if not self.enabled:
            return self._create_neutral_signal("Agent disabled")
        
        try:
            execution_signal = None
            for signal in previous_signals:
                if signal.get('agent') == 'execution_agent':
                    execution_signal = signal
                    break
            
            if not execution_signal:
                return self._create_neutral_signal("No execution signal to evaluate")
            
            self._update_agent_performance(previous_signals)
            
            performance_evaluation = self._evaluate_system_performance()
            
            strategy_recommendation = self._recommend_strategy(
                performance_evaluation, market_data
            )
            
            retraining_decisions = self._check_retraining_needs()
            
            fallback_decision = self._evaluate_fallback_strategies(performance_evaluation)
            
            final_signal = self._make_final_decision(
                execution_signal, strategy_recommendation, fallback_decision
            )
            
            signal = {
                'agent': self.name,
                'signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'timestamp': datetime.now(),
                'metadata': {
                    'performance_evaluation': performance_evaluation,
                    'strategy_recommendation': strategy_recommendation,
                    'retraining_decisions': retraining_decisions,
                    'fallback_decision': fallback_decision,
                    'agent_scores': self._get_agent_scores(),
                    'execution_signal_confidence': execution_signal.get('confidence', 0)
                }
            }
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning evaluation: {e}")
            return self._create_neutral_signal(f"Meta evaluation error: {str(e)}")
    
    def _update_agent_performance(self, previous_signals: List[Dict[str, Any]]) -> None:
        """Update performance metrics for each agent"""
        
        for signal in previous_signals:
            agent_name = signal.get('agent')
            if agent_name in self.agent_performance:
                
                confidence = signal.get('confidence', 0)
                timestamp = signal.get('timestamp', datetime.now())
                
                self.agent_performance[agent_name]['trades'].append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'signal': signal.get('signal', 'HOLD')
                })
                
                if len(self.agent_performance[agent_name]['trades']) > 1000:
                    self.agent_performance[agent_name]['trades'] = \
                        self.agent_performance[agent_name]['trades'][-1000:]
    
    def _evaluate_system_performance(self) -> Dict[str, Any]:
        """Evaluate overall system performance using multiple metrics"""
        
        overall_performance = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'consistency_score': 0.0
        }
        
        all_trades = []
        for agent_name, performance in self.agent_performance.items():
            all_trades.extend(performance['trades'])
        
        if len(all_trades) < 10:
            return overall_performance
        
        recent_trades = sorted(all_trades, key=lambda x: x['timestamp'])[-100:]
        
        confidences = [trade['confidence'] for trade in recent_trades]
        overall_performance['avg_confidence'] = float(np.mean(confidences))
        
        returns = self._simulate_returns(recent_trades)
        
        if len(returns) > 0:
            overall_performance['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            overall_performance['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            overall_performance['calmar_ratio'] = self._calculate_calmar_ratio(returns)
            overall_performance['consistency_score'] = self._calculate_consistency_score(returns)
        
        return overall_performance
    
    def _simulate_returns(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Simulate returns based on trade confidence and signals"""
        
        returns = []
        
        for i, trade in enumerate(trades):
            confidence = trade['confidence']
            signal = trade['signal']
            
            if signal in ['BUY', 'SELL']:
                base_return = np.random.normal(0, 0.01)
                
                confidence_multiplier = 1 + (confidence - 0.5) * 2
                simulated_return = base_return * confidence_multiplier
                
                if signal == 'SELL':
                    simulated_return *= -1
                
                returns.append(simulated_return)
        
        return returns
    
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
    
    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
        if max_drawdown == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        annual_return = np.mean(returns) * 252
        
        return float(annual_return / max_drawdown)
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score based on return stability"""
        
        if len(returns) < 5:
            return 0.0
        
        rolling_means = []
        window_size = min(10, len(returns) // 2)
        
        for i in range(window_size, len(returns)):
            window_mean = np.mean(returns[i-window_size:i])
            rolling_means.append(window_mean)
        
        if len(rolling_means) < 2:
            return 0.0
        
        consistency = 1.0 - (np.std(rolling_means) / (abs(np.mean(rolling_means)) + 1e-8))
        
        return float(max(0.0, min(1.0, float(consistency))))
    
    def _recommend_strategy(self, performance_evaluation: Dict[str, Any], 
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal strategy based on performance and market conditions"""
        
        sharpe_ratio = performance_evaluation.get('sharpe_ratio', 0)
        consistency = performance_evaluation.get('consistency_score', 0)
        
        volatility = market_data.get('volatility', 0.01)
        
        if sharpe_ratio > 1.5 and consistency > 0.7:
            strategy = 'aggressive'
            confidence_multiplier = 1.2
        elif sharpe_ratio > 1.0 and consistency > 0.5:
            strategy = 'moderate'
            confidence_multiplier = 1.0
        elif sharpe_ratio > 0.5:
            strategy = 'conservative'
            confidence_multiplier = 0.8
        else:
            strategy = 'defensive'
            confidence_multiplier = 0.6
        
        if volatility > 0.03:
            confidence_multiplier *= 0.8
        elif volatility < 0.01:
            confidence_multiplier *= 1.1
        
        return {
            'strategy': strategy,
            'confidence_multiplier': confidence_multiplier,
            'reasoning': f"Sharpe: {sharpe_ratio:.2f}, Consistency: {consistency:.2f}, Vol: {volatility:.3f}"
        }
    
    def _check_retraining_needs(self) -> Dict[str, Any]:
        """Check which agents need retraining"""
        
        retraining_decisions = {}
        current_time = datetime.now()
        
        for agent_name, performance in self.agent_performance.items():
            last_evaluation = performance.get('last_evaluation', current_time)
            time_since_evaluation = (current_time - last_evaluation).days
            
            needs_retraining = False
            reason = ""
            
            if self.retraining_frequency == 'monthly' and time_since_evaluation > 30:
                needs_retraining = True
                reason = "Monthly retraining schedule"
            
            elif performance.get('sharpe_ratio', 0) < 0.5:
                needs_retraining = True
                reason = "Poor Sharpe ratio performance"
            
            elif len(performance.get('trades', [])) > 100:
                recent_confidences = [t['confidence'] for t in performance['trades'][-50:]]
                if np.mean(recent_confidences) < 0.3:
                    needs_retraining = True
                    reason = "Low confidence scores"
            
            retraining_decisions[agent_name] = {
                'needs_retraining': needs_retraining,
                'reason': reason,
                'last_evaluation': last_evaluation,
                'days_since_evaluation': time_since_evaluation
            }
            
            if needs_retraining:
                performance['needs_retraining'] = True
                performance['last_evaluation'] = current_time
        
        return retraining_decisions
    
    def _evaluate_fallback_strategies(self, performance_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether to activate fallback strategies"""
        
        sharpe_ratio = performance_evaluation.get('sharpe_ratio', 0)
        consistency = performance_evaluation.get('consistency_score', 0)
        
        activate_fallback = False
        fallback_strategy = None
        reason = ""
        
        if sharpe_ratio < 0.0 and consistency < 0.3:
            activate_fallback = True
            fallback_strategy = 'grid'
            reason = "Poor performance across all metrics"
        
        elif sharpe_ratio < 0.2 and len(self.agent_performance.get('execution_agent', {}).get('trades', [])) > 50:
            activate_fallback = True
            fallback_strategy = 'martingale'
            reason = "Consistent underperformance"
        
        return {
            'activate_fallback': activate_fallback,
            'strategy': fallback_strategy,
            'reason': reason,
            'current_fallback_active': self.fallback_active
        }
    
    def _make_final_decision(self, execution_signal: Dict[str, Any],
                           strategy_recommendation: Dict[str, Any],
                           fallback_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Make final trading decision based on all evaluations"""
        
        base_signal = execution_signal.get('signal', 'HOLD')
        base_confidence = execution_signal.get('confidence', 0)
        
        if fallback_decision.get('activate_fallback', False):
            self.fallback_active = True
            return {
                'signal': 'HOLD',
                'confidence': 0.1,
                'reason': f"Fallback strategy activated: {fallback_decision.get('strategy')}"
            }
        
        confidence_multiplier = strategy_recommendation.get('confidence_multiplier', 1.0)
        final_confidence = min(1.0, base_confidence * confidence_multiplier)
        
        if final_confidence < self.performance_threshold:
            final_signal = 'HOLD'
            final_confidence = 0.0
        else:
            final_signal = base_signal
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reason': f"Strategy: {strategy_recommendation.get('strategy')}, Multiplier: {confidence_multiplier:.2f}"
        }
    
    def _get_agent_scores(self) -> Dict[str, float]:
        """Get performance scores for all agents"""
        
        scores = {}
        
        for agent_name, performance in self.agent_performance.items():
            trades = performance.get('trades', [])
            
            if len(trades) > 0:
                avg_confidence = np.mean([t['confidence'] for t in trades[-20:]])
                sharpe = performance.get('sharpe_ratio', 0)
                
                score = (avg_confidence + max(0, sharpe)) / 2
                scores[agent_name] = score
            else:
                scores[agent_name] = 0.0
        
        return scores
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Update performance tracking with actual trade results"""
        
        for agent_name in self.agent_performance:
            if len(self.agent_performance[agent_name]['trades']) > 0:
                
                profit = trade_result.get('profit', 0)
                return_pct = trade_result.get('return', 0)
                
                self.agent_performance[agent_name]['trades'][-1]['actual_profit'] = profit
                self.agent_performance[agent_name]['trades'][-1]['actual_return'] = return_pct
    
    def trigger_retraining(self, agent_name: str) -> bool:
        """Trigger retraining for specific agent"""
        
        if agent_name in self.agent_performance:
            self.agent_performance[agent_name]['needs_retraining'] = True
            self.agent_performance[agent_name]['last_evaluation'] = datetime.now()
            
            self.logger.info(f"Retraining triggered for {agent_name}")
            return True
        
        return False
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'performance_evaluation': {},
                'agent_scores': self._get_agent_scores()
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
