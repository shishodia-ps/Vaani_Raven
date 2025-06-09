"""
Quant Agent - Layer 2
Quantitative Signal Validation using Technical Indicators and GARCH-LSTM
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent

class GARCHLSTMModel(nn.Module):
    """GARCH + LSTM model for volatility regime classification"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        self.volatility_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        
        volatility = self.volatility_classifier(final_hidden)
        regime = self.regime_classifier(final_hidden)
        
        return volatility, regime

class QuantAgent(BaseAgent):
    """Layer 2: Quantitative Signal Validation Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "quant_agent")
        
        self.indicators_config = config.get('indicators', {})
        self.volatility_model_type = config.get('volatility_model', 'garch_lstm')
        
        self.volatility_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.enabled:
            self._initialize_volatility_model()
    
    def _initialize_volatility_model(self):
        """Initialize GARCH-LSTM volatility model"""
        try:
            self.volatility_model = GARCHLSTMModel().to(self.device)
            self.volatility_model.eval()
            self.logger.info("GARCH-LSTM volatility model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize volatility model: {e}")
    
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate signals using technical indicators and volatility analysis"""
        
        if not self.enabled:
            return self._create_neutral_signal("Agent disabled")
        
        try:
            ohlcv_data = market_data.get('ohlcv', [])
            
            if len(ohlcv_data) < 50:
                return self._create_neutral_signal("Insufficient data for technical analysis")
            
            df = pd.DataFrame(ohlcv_data)
            
            indicators = self._calculate_indicators(df)
            volatility_regime = self._analyze_volatility_regime(df)
            
            pattern_signal = None
            for signal in previous_signals:
                if signal.get('agent') == 'pattern_agent':
                    pattern_signal = signal
                    break
            
            if not pattern_signal or pattern_signal.get('signal') == 'HOLD':
                return self._create_neutral_signal("No pattern signal to validate")
            
            validation_result = self._validate_with_indicators(
                pattern_signal, indicators, volatility_regime
            )
            
            signal = {
                'agent': self.name,
                'signal': validation_result['signal'],
                'confidence': validation_result['confidence'],
                'timestamp': datetime.now(),
                'metadata': {
                    'indicators': indicators,
                    'volatility_regime': volatility_regime,
                    'validation_score': validation_result['validation_score'],
                    'pattern_signal_confidence': pattern_signal.get('confidence', 0),
                    'turbulence_level': self._calculate_turbulence(df)
                }
            }
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in quant validation: {e}")
            return self._create_neutral_signal(f"Validation error: {str(e)}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        if self.indicators_config.get('rsi', {}).get('enabled', True):
            indicators['rsi'] = self._calculate_rsi(df, 
                self.indicators_config.get('rsi', {}).get('period', 14))
        
        if self.indicators_config.get('bollinger_bands', {}).get('enabled', True):
            bb_config = self.indicators_config.get('bollinger_bands', {})
            indicators['bollinger'] = self._calculate_bollinger_bands(
                df, bb_config.get('period', 20), bb_config.get('deviation', 2.0))
        
        if self.indicators_config.get('macd', {}).get('enabled', True):
            macd_config = self.indicators_config.get('macd', {})
            indicators['macd'] = self._calculate_macd(
                df, 
                macd_config.get('fast_ema', 12),
                macd_config.get('slow_ema', 26),
                macd_config.get('signal', 9)
            )
        
        if self.indicators_config.get('atr', {}).get('enabled', True):
            indicators['atr'] = self._calculate_atr(df,
                self.indicators_config.get('atr', {}).get('period', 14))
        
        indicators['z_score'] = self._calculate_z_score(df)
        
        return indicators
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calculate RSI indicator"""
        close = df['close'].values
        delta = np.diff(close.astype(float))
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = float(np.mean(gain[-period:])) if len(gain) >= period else 0.0
        avg_loss = float(np.mean(loss[-period:])) if len(loss) >= period else 0.0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1 + rs))
        
        return {
            'value': float(rsi),
            'overbought': float(rsi > self.indicators_config.get('rsi', {}).get('overbought', 70)),
            'oversold': float(rsi < self.indicators_config.get('rsi', {}).get('oversold', 30))
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                 deviation: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        close = df['close'].values[-period:]
        
        if len(close) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5}
        
        middle = np.mean(close)
        std = np.std(close)
        
        upper = middle + (deviation * std)
        lower = middle - (deviation * std)
        
        current_price = close[-1]
        position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        return {
            'upper': float(upper),
            'middle': float(middle),
            'lower': float(lower),
            'position': float(position),
            'squeeze': float((upper - lower) / middle < 0.1)
        }
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        close = df['close'].values
        
        if len(close) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'bullish': False}
        
        ema_fast = self._calculate_ema(close.astype(float), fast)
        ema_slow = self._calculate_ema(close.astype(float), slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(np.array([macd_line]), signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'bullish': macd_line > signal_line
        }
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0.0
        
        alpha = 2 / (period + 1)
        ema = float(data[0])
        
        for price in data[1:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        
        return float(ema)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0
        
        high = df['high'].values[-period:]
        low = df['low'].values[-period:]
        close = df['close'].values[-period:]
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        return float(np.mean(tr_list)) if tr_list else 0.0
    
    def _calculate_z_score(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Z-score for mean reversion"""
        close = df['close'].values[-period:]
        
        if len(close) < period:
            return 0
        
        mean_price = np.mean(close[:-1])
        std_price = np.std(close[:-1])
        
        if std_price == 0:
            return 0
        
        return (close[-1] - mean_price) / std_price
    
    def _analyze_volatility_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility regime using GARCH-LSTM"""
        try:
            if self.volatility_model is None:
                return {'regime': 'unknown', 'volatility': 'medium', 'confidence': 0.0}
            
            features = self._prepare_volatility_features(df)
            
            with torch.no_grad():
                volatility_logits, regime_logits = self.volatility_model(features)
                
                volatility_probs = torch.softmax(volatility_logits, dim=1)
                regime_probs = torch.softmax(regime_logits, dim=1)
                
                volatility_map = {0: 'low', 1: 'medium', 2: 'high', 3: 'crisis'}
                regime_map = {0: 'trending', 1: 'ranging', 2: 'transitional'}
                
                volatility_class = torch.argmax(volatility_probs, dim=1).item()
                regime_class = torch.argmax(regime_probs, dim=1).item()
                
                return {
                    'volatility': volatility_map[volatility_class],
                    'regime': regime_map[regime_class],
                    'volatility_confidence': volatility_probs[0][volatility_class].item(),
                    'regime_confidence': regime_probs[0][regime_class].item()
                }
                
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {e}")
            return {'regime': 'unknown', 'volatility': 'medium', 'confidence': 0.0}
    
    def _prepare_volatility_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare features for volatility model"""
        if len(df) < 20:
            return torch.zeros(1, 20, 10).to(self.device)
        
        features = []
        for i in range(len(df) - 20, len(df)):
            if i < 1:
                continue
                
            returns = (df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            volume_change = (df.iloc[i]['volume'] - df.iloc[i-1]['volume']) / (df.iloc[i-1]['volume'] + 1e-8)
            
            feature_vector = [
                returns,
                abs(returns),
                returns ** 2,
                volume_change,
                (df.iloc[i]['high'] - df.iloc[i]['low']) / df.iloc[i]['close'],
                df.iloc[i]['close'] / df.iloc[i]['open'] - 1,
                np.log(df.iloc[i]['volume'] + 1),
                df.iloc[i]['high'] / df.iloc[i]['close'] - 1,
                df.iloc[i]['low'] / df.iloc[i]['close'] - 1,
                i / len(df)
            ]
            features.append(feature_vector)
        
        if len(features) < 20:
            features.extend([[0] * 10] * (20 - len(features)))
        
        features_array = np.array(features[-20:], dtype=np.float32)
        return torch.tensor(features_array).unsqueeze(0).to(self.device)
    
    def _calculate_turbulence(self, df: pd.DataFrame) -> float:
        """Calculate market turbulence using ATR"""
        atr = self._calculate_atr(df)
        avg_price = df['close'].iloc[-20:].mean() if len(df) >= 20 else df['close'].mean()
        
        return atr / avg_price if avg_price > 0 else 0
    
    def _validate_with_indicators(self, pattern_signal: Dict[str, Any], 
                                indicators: Dict[str, Any], 
                                volatility_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pattern signal with technical indicators"""
        
        signal_type = pattern_signal.get('signal', 'HOLD')
        pattern_confidence = pattern_signal.get('confidence', 0)
        
        validation_score = 0
        total_weight = 0
        
        if 'rsi' in indicators:
            rsi_data = indicators['rsi']
            if signal_type == 'BUY' and rsi_data['oversold']:
                validation_score += 0.3
            elif signal_type == 'SELL' and rsi_data['overbought']:
                validation_score += 0.3
            elif signal_type == 'BUY' and rsi_data['overbought']:
                validation_score -= 0.2
            elif signal_type == 'SELL' and rsi_data['oversold']:
                validation_score -= 0.2
            total_weight += 0.3
        
        if 'bollinger' in indicators:
            bb_data = indicators['bollinger']
            if signal_type == 'BUY' and bb_data['position'] < 0.2:
                validation_score += 0.25
            elif signal_type == 'SELL' and bb_data['position'] > 0.8:
                validation_score += 0.25
            total_weight += 0.25
        
        if 'macd' in indicators:
            macd_data = indicators['macd']
            if signal_type == 'BUY' and macd_data['bullish']:
                validation_score += 0.25
            elif signal_type == 'SELL' and not macd_data['bullish']:
                validation_score += 0.25
            total_weight += 0.25
        
        if 'z_score' in indicators:
            z_score = indicators['z_score']
            if signal_type == 'BUY' and z_score < -1.5:
                validation_score += 0.2
            elif signal_type == 'SELL' and z_score > 1.5:
                validation_score += 0.2
            total_weight += 0.2
        
        normalized_score = validation_score / total_weight if total_weight > 0 else 0
        
        volatility_penalty = 0
        if volatility_regime.get('volatility') == 'crisis':
            volatility_penalty = 0.5
        elif volatility_regime.get('volatility') == 'high':
            volatility_penalty = 0.2
        
        final_confidence = max(0, min(1, pattern_confidence * (1 + normalized_score) - volatility_penalty))
        
        if final_confidence < 0.3:
            final_signal = 'HOLD'
        else:
            final_signal = signal_type
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'validation_score': normalized_score
        }
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'validation_score': 0.0
            }
        }
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and content"""
        required_fields = ['agent', 'signal', 'confidence', 'timestamp']
        
        if not all(field in signal for field in required_fields):
            return False
        
        if signal['signal'] not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        if not 0 <= signal['confidence'] <= 1:
            return False
        
        return True
