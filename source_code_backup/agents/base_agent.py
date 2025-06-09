"""
Base Agent Class for VAANI-RAVEN X
Provides common interface and functionality for all agent layers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import yaml

class BaseAgent(ABC):
    """Abstract base class for all VAANI-RAVEN X agents"""
    
    def __init__(self, config: Dict[str, Any], name: str):
        self.config = config
        self.name = name
        self.enabled = config.get('enabled', True)
        self.logger = self._setup_logger()
        self.last_signal = None
        self.signal_history = []
        self.performance_metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger"""
        logger = logging.getLogger(f"vaani_raven_x.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def process_signal(self, market_data: Dict[str, Any], 
                      previous_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process market data and previous signals to generate agent output
        
        Args:
            market_data: Current market data (OHLCV, indicators, etc.)
            previous_signals: Signals from previous agent layers
            
        Returns:
            Dict containing agent signal, confidence, and metadata
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and content"""
        pass
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update agent performance metrics based on trade results"""
        if not hasattr(self, 'trade_results'):
            self.trade_results = []
        
        self.trade_results.append({
            'timestamp': datetime.now(),
            'signal_confidence': self.last_signal.get('confidence', 0) if self.last_signal else 0,
            'trade_result': trade_result
        })
        
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate agent-specific performance metrics"""
        if not hasattr(self, 'trade_results') or not self.trade_results:
            return
            
        total_trades = len(self.trade_results)
        profitable_trades = sum(1 for result in self.trade_results 
                              if result['trade_result'].get('profit', 0) > 0)
        
        self.performance_metrics = {
            'total_signals': total_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'avg_confidence': sum(r['signal_confidence'] for r in self.trade_results) / total_trades if total_trades > 0 else 0,
            'last_updated': datetime.now()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'last_signal_time': self.last_signal.get('timestamp') if self.last_signal else None,
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
    
    def log_signal(self, signal: Dict[str, Any]) -> None:
        """Log signal for debugging and analysis"""
        self.logger.info(f"Generated signal: {signal}")
        self.last_signal = signal
        self.signal_history.append(signal)
        
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
