"""
Orchestrator Runner - Central Pipeline Coordinator
Manages signal passing between all agent layers
"""

import yaml
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from pathlib import Path

from agents.pattern_agent import PatternAgent
from agents.quant_agent import QuantAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.meta_agent import MetaAgent

AGENT_LAYERS = [
    'pattern_agent',
    'quant_agent', 
    'sentiment_agent',
    'risk_agent',
    'execution_agent',
    'meta_agent'
]

class OrchestatorRunner:
    """Central orchestrator for VAANI-RAVEN X agent pipeline"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.agents = {}
        self.pipeline_history = []
        self.logger = self._setup_logger()
        
        self._initialize_agents()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available"""
        return {
            'system': {'name': 'VAANI-RAVEN X', 'debug_mode': True},
            'agents': {
                'pattern_agent': {'enabled': True},
                'quant_agent': {'enabled': True},
                'sentiment_agent': {'enabled': True},
                'risk_agent': {'enabled': True},
                'execution_agent': {'enabled': True},
                'meta_agent': {'enabled': True}
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup orchestrator logger"""
        logger = logging.getLogger("vaani_raven_x.orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_agents(self) -> None:
        """Initialize all agents based on configuration"""
        
        agent_classes = {
            'pattern_agent': PatternAgent,
            'quant_agent': QuantAgent,
            'sentiment_agent': SentimentAgent,
            'risk_agent': RiskAgent,
            'execution_agent': ExecutionAgent,
            'meta_agent': MetaAgent
        }
        
        agents_config = self.config.get('agents', {})
        
        for agent_name in AGENT_LAYERS:
            if agent_name in agent_classes:
                agent_config = agents_config.get(agent_name, {})
                
                try:
                    self.agents[agent_name] = agent_classes[agent_name](agent_config)
                    self.logger.info(f"Initialized {agent_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {agent_name}: {e}")
                    self.agents[agent_name] = None
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through the agent pipeline"""
        
        pipeline_start = datetime.now()
        signals = []
        
        try:
            self.logger.info("Starting agent pipeline processing")
            
            for agent_name in AGENT_LAYERS:
                agent = self.agents.get(agent_name)
                
                if agent is None:
                    self.logger.warning(f"Agent {agent_name} not available, skipping")
                    continue
                
                if not agent.enabled:
                    self.logger.info(f"Agent {agent_name} disabled, skipping")
                    continue
                
                try:
                    agent_start = datetime.now()
                    
                    signal = agent.process_signal(market_data, signals.copy())
                    
                    agent_duration = (datetime.now() - agent_start).total_seconds() * 1000
                    
                    if agent.validate_signal(signal):
                        signals.append(signal)
                        self.logger.info(
                            f"{agent_name} processed in {agent_duration:.1f}ms: "
                            f"{signal.get('signal', 'UNKNOWN')} "
                            f"(confidence: {signal.get('confidence', 0):.2f})"
                        )
                    else:
                        self.logger.error(f"Invalid signal from {agent_name}: {signal}")
                        
                except Exception as e:
                    self.logger.error(f"Error in {agent_name}: {e}")
                    continue
            
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds() * 1000
            
            final_result = self._compile_final_result(signals, pipeline_duration)
            
            self._log_pipeline_execution(market_data, signals, final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return self._create_error_result(str(e))
    
    def _compile_final_result(self, signals: List[Dict[str, Any]], 
                            pipeline_duration: float) -> Dict[str, Any]:
        """Compile final trading decision from all agent signals"""
        
        if not signals:
            return {
                'final_signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No valid signals generated',
                'pipeline_duration_ms': pipeline_duration,
                'agent_signals': [],
                'timestamp': datetime.now()
            }
        
        meta_signal = None
        for signal in reversed(signals):
            if signal.get('agent') == 'meta_agent':
                meta_signal = signal
                break
        
        if meta_signal:
            final_signal = meta_signal.get('signal', 'HOLD')
            final_confidence = meta_signal.get('confidence', 0.0)
            reason = "Meta-agent decision"
        else:
            execution_signal = None
            for signal in reversed(signals):
                if signal.get('agent') == 'execution_agent':
                    execution_signal = signal
                    break
            
            if execution_signal:
                final_signal = execution_signal.get('signal', 'HOLD')
                final_confidence = execution_signal.get('confidence', 0.0)
                reason = "Execution agent decision (meta-agent unavailable)"
            else:
                final_signal = 'HOLD'
                final_confidence = 0.0
                reason = "No execution or meta-agent signals available"
        
        return {
            'final_signal': final_signal,
            'confidence': final_confidence,
            'reason': reason,
            'pipeline_duration_ms': pipeline_duration,
            'agent_signals': signals,
            'timestamp': datetime.now(),
            'agent_count': len(signals),
            'pipeline_success': True
        }
    
    def _log_pipeline_execution(self, market_data: Dict[str, Any], 
                              signals: List[Dict[str, Any]], 
                              final_result: Dict[str, Any]) -> None:
        """Log pipeline execution for analysis"""
        
        execution_log = {
            'timestamp': datetime.now(),
            'market_data_keys': list(market_data.keys()),
            'signals_generated': len(signals),
            'final_signal': final_result.get('final_signal'),
            'final_confidence': final_result.get('confidence'),
            'pipeline_duration_ms': final_result.get('pipeline_duration_ms'),
            'agent_performance': {}
        }
        
        for signal in signals:
            agent_name = signal.get('agent')
            if agent_name:
                execution_log['agent_performance'][agent_name] = {
                    'signal': signal.get('signal'),
                    'confidence': signal.get('confidence'),
                    'timestamp': signal.get('timestamp')
                }
        
        self.pipeline_history.append(execution_log)
        
        if len(self.pipeline_history) > 1000:
            self.pipeline_history = self.pipeline_history[-1000:]
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result when pipeline fails"""
        return {
            'final_signal': 'HALT',
            'confidence': 0.0,
            'reason': f'Pipeline error: {error_message}',
            'pipeline_duration_ms': 0.0,
            'agent_signals': [],
            'timestamp': datetime.now(),
            'agent_count': 0,
            'pipeline_success': False,
            'error': error_message
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        status = {
            'orchestrator': {
                'active': True,
                'config_loaded': bool(self.config),
                'agents_initialized': len(self.agents),
                'pipeline_executions': len(self.pipeline_history)
            },
            'agents': {}
        }
        
        for agent_name, agent in self.agents.items():
            if agent:
                status['agents'][agent_name] = agent.get_status()
            else:
                status['agents'][agent_name] = {
                    'name': agent_name,
                    'enabled': False,
                    'error': 'Failed to initialize'
                }
        
        return status
    
    def update_agent_config(self, agent_name: str, new_config: Dict[str, Any]) -> bool:
        """Update configuration for specific agent"""
        
        if agent_name not in self.agents:
            return False
        
        try:
            self.config['agents'][agent_name].update(new_config)
            
            if self.agents[agent_name]:
                self.agents[agent_name].config.update(new_config)
                self.logger.info(f"Updated config for {agent_name}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to update config for {agent_name}: {e}")
        
        return False
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        
        if not self.pipeline_history:
            return {'error': 'No pipeline history available'}
        
        recent_executions = self.pipeline_history[-100:]
        
        avg_duration = sum(e.get('pipeline_duration_ms', 0) for e in recent_executions) / len(recent_executions)
        
        signal_distribution = {}
        for execution in recent_executions:
            signal = execution.get('final_signal', 'UNKNOWN')
            signal_distribution[signal] = signal_distribution.get(signal, 0) + 1
        
        avg_confidence = sum(e.get('final_confidence', 0) for e in recent_executions) / len(recent_executions)
        
        success_rate = sum(1 for e in recent_executions if e.get('pipeline_success', False)) / len(recent_executions)
        
        return {
            'total_executions': len(self.pipeline_history),
            'recent_executions': len(recent_executions),
            'avg_pipeline_duration_ms': avg_duration,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'success_rate': success_rate,
            'last_execution': recent_executions[-1]['timestamp'] if recent_executions else None
        }
    
    def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources"""
        
        self.logger.info("Shutting down orchestrator")
        
        for agent_name, agent in self.agents.items():
            if agent:
                try:
                    if hasattr(agent, 'shutdown'):
                        agent.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down {agent_name}: {e}")
        
        self.logger.info("Orchestrator shutdown complete")

async def main():
    """Main function for testing orchestrator"""
    
    orchestrator = OrchestatorRunner()
    
    sample_market_data = {
        'current_price': 1.0850,
        'spread': 0.0001,
        'volume': 1000000,
        'volatility': 0.015,
        'atr': 0.0012,
        'ohlcv': [
            {'open': 1.0840, 'high': 1.0860, 'low': 1.0835, 'close': 1.0850, 'volume': 50000},
            {'open': 1.0850, 'high': 1.0865, 'low': 1.0845, 'close': 1.0855, 'volume': 55000},
        ]
    }
    
    result = await orchestrator.process_market_data(sample_market_data)
    
    print("Pipeline Result:")
    print(f"Signal: {result.get('final_signal')}")
    print(f"Confidence: {result.get('confidence'):.2f}")
    print(f"Duration: {result.get('pipeline_duration_ms'):.1f}ms")
    print(f"Agents: {result.get('agent_count')}")
    
    status = orchestrator.get_agent_status()
    print(f"\nAgent Status: {len(status['agents'])} agents initialized")
    
    metrics = orchestrator.get_pipeline_metrics()
    print(f"Pipeline Metrics: {metrics.get('success_rate', 0):.1%} success rate")

if __name__ == "__main__":
    asyncio.run(main())
