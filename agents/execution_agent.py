"""
Execution Agent - Layer 5
Optimal Entry/Exit using PPO Reinforcement Learning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent

class PPOExecutionModel(nn.Module):
    """PPO model for optimal trade execution"""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.slippage_predictor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        slippage_pred = self.slippage_predictor(state)
        
        return action_probs, value, slippage_pred

class ExecutionAgent(BaseAgent):
    """Layer 5: Execution Agent with PPO Reinforcement Learning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "execution_agent")
        
        self.model_type = config.get('model', 'ppo')
        self.order_types = config.get('order_types', ['market', 'limit', 'stop'])
        self.slippage_awareness = config.get('slippage_awareness', True)
        self.latency_logging = config.get('latency_logging', True)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.execution_history = []
        self.slippage_history = []
        self.latency_history = []
        
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize PPO execution model"""
        try:
            self.model = PPOExecutionModel().to(self.device)
            self.model.eval()
            self.logger.info("PPO execution model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize execution model: {e}")
    
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process execution optimization and timing"""
        
        if not self.enabled:
            return self._create_neutral_signal("Agent disabled")
        
        try:
            risk_signal = None
            for signal in previous_signals:
                if signal.get('agent') == 'risk_agent':
                    risk_signal = signal
                    break
            
            if not risk_signal or risk_signal.get('signal') in ['HOLD', 'HALT']:
                return self._create_neutral_signal("No risk signal to execute")
            
            execution_state = self._prepare_execution_state(market_data, risk_signal)
            
            optimal_execution = self._determine_optimal_execution(
                execution_state, risk_signal, market_data
            )
            
            slippage_prediction = self._predict_slippage(execution_state)
            latency_estimate = self._estimate_execution_latency(optimal_execution)
            
            signal = {
                'agent': self.name,
                'signal': risk_signal['signal'],
                'confidence': risk_signal['confidence'] * optimal_execution['confidence_adjustment'],
                'timestamp': datetime.now(),
                'metadata': {
                    'execution_plan': optimal_execution,
                    'slippage_prediction': slippage_prediction,
                    'latency_estimate': latency_estimate,
                    'order_type': optimal_execution['order_type'],
                    'entry_price': optimal_execution['entry_price'],
                    'position_size': risk_signal['metadata'].get('position_size', 0),
                    'stop_loss': risk_signal['metadata'].get('stop_loss', 0),
                    'take_profit': risk_signal['metadata'].get('take_profit', 0),
                    'risk_signal_confidence': risk_signal.get('confidence', 0)
                }
            }
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in execution planning: {e}")
            return self._create_neutral_signal(f"Execution error: {str(e)}")
    
    def _prepare_execution_state(self, market_data: Dict[str, Any], 
                               risk_signal: Dict[str, Any]) -> torch.Tensor:
        """Prepare state vector for PPO model"""
        
        current_price = market_data.get('current_price', 1.0)
        spread = market_data.get('spread', 0.0001)
        volume = market_data.get('volume', 1000)
        volatility = market_data.get('volatility', 0.01)
        
        ohlcv_data = market_data.get('ohlcv', [])
        if len(ohlcv_data) >= 5:
            recent_prices = [candle.get('close', current_price) for candle in ohlcv_data[-5:]]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            price_volatility = np.std(recent_prices) / np.mean(recent_prices)
        else:
            price_momentum = 0.0
            price_volatility = volatility
        
        position_size = risk_signal['metadata'].get('position_size', 0)
        confidence = risk_signal.get('confidence', 0)
        
        time_features = self._get_time_features()
        
        avg_slippage = np.mean(self.slippage_history[-10:]) if self.slippage_history else 0.0001
        avg_latency = np.mean(self.latency_history[-10:]) if self.latency_history else 50.0
        
        state_vector = [
            current_price / 1.1,
            spread * 10000,
            np.log(volume + 1) / 10,
            volatility * 100,
            price_momentum * 100,
            price_volatility * 100,
            position_size,
            confidence,
            time_features['hour'] / 24,
            time_features['day_of_week'] / 7,
            time_features['is_session_open'],
            avg_slippage * 10000,
            avg_latency / 1000,
            len(self.execution_history) / 1000,
            self._calculate_market_impact_factor(position_size, volume),
            self._calculate_urgency_factor(market_data),
            1.0 if risk_signal['signal'] == 'BUY' else 0.0,
            1.0 if risk_signal['signal'] == 'SELL' else 0.0,
            market_data.get('bid', current_price) / current_price,
            market_data.get('ask', current_price) / current_price
        ]
        
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        return state_tensor
    
    def _determine_optimal_execution(self, execution_state: torch.Tensor,
                                   risk_signal: Dict[str, Any],
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal execution strategy using PPO"""
        
        try:
            if self.model is None:
                return self._get_default_execution(risk_signal, market_data)
            
            with torch.no_grad():
                action_probs, value, slippage_pred = self.model(execution_state)
                
                action_idx = torch.argmax(action_probs, dim=1).item()
                confidence_adjustment = value.item()
                
                action_map = {0: 'market', 1: 'limit', 2: 'stop'}
                order_type = action_map.get(action_idx, 'market')
                
                current_price = market_data.get('current_price', 1.0)
                spread = market_data.get('spread', 0.0001)
                
                if order_type == 'market':
                    entry_price = current_price
                    confidence_adjustment = max(0.8, confidence_adjustment)
                elif order_type == 'limit':
                    if risk_signal['signal'] == 'BUY':
                        entry_price = current_price - spread * 0.5
                    else:
                        entry_price = current_price + spread * 0.5
                    confidence_adjustment = max(0.9, confidence_adjustment)
                else:
                    entry_price = current_price
                    confidence_adjustment = max(0.7, confidence_adjustment)
                
                return {
                    'order_type': order_type,
                    'entry_price': entry_price,
                    'confidence_adjustment': min(1.0, confidence_adjustment),
                    'execution_timing': 'immediate' if order_type == 'market' else 'patient',
                    'expected_slippage': slippage_pred.item()
                }
                
        except Exception as e:
            self.logger.error(f"Error in PPO execution: {e}")
            return self._get_default_execution(risk_signal, market_data)
    
    def _get_default_execution(self, risk_signal: Dict[str, Any],
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get default execution strategy when PPO is unavailable"""
        
        current_price = market_data.get('current_price', 1.0)
        volatility = market_data.get('volatility', 0.01)
        
        if volatility > 0.02:
            order_type = 'limit'
            confidence_adjustment = 0.9
        else:
            order_type = 'market'
            confidence_adjustment = 0.95
        
        return {
            'order_type': order_type,
            'entry_price': current_price,
            'confidence_adjustment': confidence_adjustment,
            'execution_timing': 'immediate',
            'expected_slippage': 0.0001
        }
    
    def _predict_slippage(self, execution_state: torch.Tensor) -> float:
        """Predict expected slippage for execution"""
        
        try:
            if self.model is None:
                return float(np.mean(self.slippage_history[-10:])) if self.slippage_history else 0.0001
            
            with torch.no_grad():
                _, _, slippage_pred = self.model(execution_state)
                return slippage_pred.item() * 0.001
                
        except Exception as e:
            self.logger.error(f"Error predicting slippage: {e}")
            return 0.0001
    
    def _estimate_execution_latency(self, execution_plan: Dict[str, Any]) -> float:
        """Estimate execution latency in milliseconds"""
        
        base_latency = 50.0
        
        if execution_plan['order_type'] == 'market':
            latency = base_latency
        elif execution_plan['order_type'] == 'limit':
            latency = base_latency * 2.0
        else:
            latency = base_latency * 1.5
        
        if self.latency_history:
            avg_historical_latency = np.mean(self.latency_history[-20:])
            latency = (latency + avg_historical_latency) / 2
        
        return float(latency)
    
    def _get_time_features(self) -> Dict[str, float]:
        """Get time-based features for execution timing"""
        
        now = datetime.now()
        
        return {
            'hour': float(now.hour),
            'day_of_week': float(now.weekday()),
            'is_session_open': 1.0 if 8 <= now.hour <= 17 else 0.0
        }
    
    def _calculate_market_impact_factor(self, position_size: float, volume: float) -> float:
        """Calculate market impact factor based on position size and volume"""
        
        if volume == 0:
            return 1.0
        
        impact_ratio = position_size / volume
        
        if impact_ratio > 0.1:
            return 1.0
        elif impact_ratio > 0.05:
            return 0.8
        elif impact_ratio > 0.01:
            return 0.6
        else:
            return 0.4
    
    def _calculate_urgency_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate urgency factor based on market conditions"""
        
        volatility = market_data.get('volatility', 0.01)
        spread = market_data.get('spread', 0.0001)
        
        if volatility > 0.03 or spread > 0.0005:
            return 1.0
        elif volatility > 0.02 or spread > 0.0003:
            return 0.7
        else:
            return 0.4
    
    def record_execution_result(self, execution_result: Dict[str, Any]) -> None:
        """Record execution result for learning"""
        
        self.execution_history.append({
            'timestamp': datetime.now(),
            'order_type': execution_result.get('order_type', 'market'),
            'expected_price': execution_result.get('expected_price', 0),
            'actual_price': execution_result.get('actual_price', 0),
            'slippage': execution_result.get('slippage', 0),
            'latency': execution_result.get('latency', 0),
            'success': execution_result.get('success', True)
        })
        
        if execution_result.get('slippage') is not None:
            self.slippage_history.append(execution_result['slippage'])
        
        if execution_result.get('latency') is not None:
            self.latency_history.append(execution_result['latency'])
        
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        if len(self.slippage_history) > 100:
            self.slippage_history = self.slippage_history[-100:]
        
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'execution_plan': None
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
