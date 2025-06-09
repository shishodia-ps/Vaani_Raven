"""
Pattern Agent - Layer 1
Advanced Pattern Recognition using Transformer + CNN-LSTM
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from .base_agent import BaseAgent

class TransformerCNNLSTM(nn.Module):
    """Transformer + CNN-LSTM model for price pattern recognition"""
    
    def __init__(self, input_dim: int = 5, seq_length: int = 100, 
                 d_model: int = 64, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        transformer_out = self.transformer(x)
        
        cnn_input = transformer_out.transpose(1, 2)
        cnn_out = self.cnn(cnn_input)
        cnn_out = cnn_out.transpose(1, 2)
        
        lstm_out, (hidden, _) = self.lstm(cnn_out)
        
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        
        signal = self.classifier(final_hidden)
        confidence = self.confidence_head(final_hidden)
        
        return signal, confidence

class PatternAgent(BaseAgent):
    """Layer 1: Advanced Pattern Recognition Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "pattern_agent")
        
        self.sequence_length = config.get('sequence_length', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.use_cnn_lstm = config.get('use_cnn_lstm', True)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Transformer + CNN-LSTM model"""
        try:
            self.model = TransformerCNNLSTM(
                seq_length=self.sequence_length
            ).to(self.device)
            
            self.model.eval()
            self.logger.info("Pattern recognition model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            self.enabled = False
    
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process market data to generate pattern-based signals"""
        
        if not self.enabled or self.model is None:
            return self._create_neutral_signal("Agent disabled or model not loaded")
        
        try:
            ohlcv_data = market_data.get('ohlcv', [])
            
            if len(ohlcv_data) < self.sequence_length:
                return self._create_neutral_signal("Insufficient data for pattern analysis")
            
            features = self._prepare_features(ohlcv_data[-self.sequence_length:])
            
            with torch.no_grad():
                signal_logits, confidence = self.model(features)
                signal_probs = torch.softmax(signal_logits, dim=1)
                
                predicted_class = torch.argmax(signal_probs, dim=1).item()
                confidence_score = confidence.item()
                
                signal_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
                signal_type = signal_map[predicted_class]
                
                if confidence_score < self.confidence_threshold:
                    signal_type = 'HOLD'
                
                signal = {
                    'agent': self.name,
                    'signal': signal_type,
                    'confidence': confidence_score,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'pattern_type': self._identify_pattern_type(ohlcv_data),
                        'signal_probabilities': {
                            'buy': signal_probs[0][0].item(),
                            'sell': signal_probs[0][1].item(),
                            'hold': signal_probs[0][2].item()
                        },
                        'sequence_length': len(ohlcv_data),
                        'model_type': 'transformer_cnn_lstm'
                    }
                }
                
                self.log_signal(signal)
                return signal
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return self._create_neutral_signal(f"Processing error: {str(e)}")
    
    def _prepare_features(self, ohlcv_data: List[Dict]) -> torch.Tensor:
        """Prepare OHLCV data for model input"""
        features = []
        
        for candle in ohlcv_data:
            feature_vector = [
                candle.get('open', 0),
                candle.get('high', 0),
                candle.get('low', 0),
                candle.get('close', 0),
                candle.get('volume', 0)
            ]
            features.append(feature_vector)
        
        features_array = np.array(features, dtype=np.float32)
        
        if len(features_array) > 0:
            features_array = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
        
        features_tensor = torch.tensor(features_array).unsqueeze(0).to(self.device)
        return features_tensor
    
    def _identify_pattern_type(self, ohlcv_data: List[Dict]) -> str:
        """Identify basic pattern types for metadata"""
        if len(ohlcv_data) < 10:
            return "insufficient_data"
        
        recent_closes = [candle.get('close', 0) for candle in ohlcv_data[-10:]]
        
        if all(recent_closes[i] <= recent_closes[i+1] for i in range(len(recent_closes)-1)):
            return "uptrend"
        elif all(recent_closes[i] >= recent_closes[i+1] for i in range(len(recent_closes)-1)):
            return "downtrend"
        else:
            return "sideways"
    
    def _create_neutral_signal(self, reason: str) -> Dict[str, Any]:
        """Create a neutral signal with reason"""
        return {
            'agent': self.name,
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'metadata': {
                'reason': reason,
                'pattern_type': 'none',
                'model_type': 'transformer_cnn_lstm'
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
