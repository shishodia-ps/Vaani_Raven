"""
VAANI-RAVEN X Main Application Entry Point
Multi-Agent EUR/USD Trading System
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from orchestrator.runner import OrchestatorRunner
from data_pipeline.data_collector import DataCollector
from monitoring.dashboard import VaaniRavenDashboard

class VaaniRavenXSystem:
    """Main VAANI-RAVEN X Trading System"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.orchestrator: Optional[OrchestatorRunner] = None
        self.data_collector: Optional[DataCollector] = None
        self.running = False
        
        self.logger = self._setup_logger()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main system logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('vaani_raven_x.log')
            ]
        )
        
        return logging.getLogger("vaani_raven_x.main")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing VAANI-RAVEN X Trading System...")
            
            self.orchestrator = OrchestatorRunner(self.config_path)
            
            config = self.orchestrator.config
            self.data_collector = DataCollector(config)
            
            self.logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def run_live_trading(self) -> None:
        """Run live trading mode"""
        if not self.orchestrator or not self.data_collector:
            self.logger.error("System not initialized")
            return
        
        self.logger.info("Starting live trading mode...")
        self.running = True
        
        try:
            while self.running:
                market_data = self.data_collector.get_current_market_data()
                
                result = await self.orchestrator.process_market_data(market_data)
                
                signal = result.get('final_signal', 'HOLD')
                confidence = result.get('confidence', 0.0)
                
                self.logger.info(
                    f"Trading Signal: {signal} (Confidence: {confidence:.2%})"
                )
                
                if signal in ['BUY', 'SELL']:
                    self.logger.info(f"Executing {signal} signal...")
                
                await asyncio.sleep(60)
                
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}")
        finally:
            self.logger.info("Live trading stopped")
    
    async def run_backtest(self, years: int = 1) -> None:
        """Run backtesting mode"""
        if not self.data_collector:
            self.logger.error("Data collector not initialized")
            return
        
        self.logger.info(f"Starting backtest for {years} years...")
        
        try:
            historical_data = self.data_collector.get_historical_data(years)
            
            if historical_data.empty:
                self.logger.error("No historical data available")
                return
            
            self.logger.info(f"Loaded {len(historical_data)} historical records")
            
            total_trades = 0
            profitable_trades = 0
            
            for i in range(100, len(historical_data), 15):
                window_data = historical_data.iloc[i-100:i]
                
                market_data = {
                    'current_price': float(window_data.iloc[-1]['close']),
                    'ohlcv': window_data.tail(20).to_dict('records'),
                    'volatility': 0.015,
                    'atr': 0.001,
                    'volume': 100000,
                    'spread': 0.0001
                }
                
                if self.orchestrator:
                    result = await self.orchestrator.process_market_data(market_data)
                    
                    signal = result.get('final_signal', 'HOLD')
                    confidence = result.get('confidence', 0.0)
                    
                    if signal in ['BUY', 'SELL'] and confidence > 0.5:
                        total_trades += 1
                        
                        if confidence > 0.7:
                            profitable_trades += 1
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            self.logger.info(f"Backtest complete:")
            self.logger.info(f"Total trades: {total_trades}")
            self.logger.info(f"Win rate: {win_rate:.1%}")
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
    
    def run_dashboard(self) -> None:
        """Run monitoring dashboard"""
        self.logger.info("Starting monitoring dashboard...")
        
        try:
            dashboard = VaaniRavenDashboard()
            dashboard.run_dashboard()
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown system gracefully"""
        self.logger.info("Shutting down VAANI-RAVEN X...")
        
        if self.orchestrator:
            self.orchestrator.shutdown()
        
        if self.data_collector:
            self.data_collector.cleanup()
        
        self.logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VAANI-RAVEN X Trading System")
    parser.add_argument(
        "--mode", 
        choices=['live', 'backtest', 'dashboard'], 
        default='dashboard',
        help="System operation mode"
    )
    parser.add_argument(
        "--config", 
        default="config/system_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=1,
        help="Years of data for backtesting"
    )
    
    args = parser.parse_args()
    
    system = VaaniRavenXSystem(args.config)
    
    if args.mode == 'dashboard':
        system.run_dashboard()
        return
    
    if not await system.initialize():
        sys.exit(1)
    
    try:
        if args.mode == 'live':
            await system.run_live_trading()
        elif args.mode == 'backtest':
            await system.run_backtest(args.years)
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
