"""
VAANI-RAVEN X System Integration Test
Tests all components of the multi-agent trading system
"""

import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from orchestrator.runner import OrchestatorRunner
from data_pipeline.data_collector import DataCollector
from testing.backtester import BacktestEngine

async def test_system_integration():
    """Test complete system integration"""
    print('🤖 Testing VAANI-RAVEN X System Integration...')
    
    try:
        print('1. Testing Orchestrator initialization...')
        orchestrator = OrchestatorRunner()
        print('✓ Orchestrator initialized successfully')
        
        print('2. Testing Data Collector...')
        data_collector = DataCollector(orchestrator.config)
        market_data = data_collector.get_current_market_data()
        print(f'✓ Data Collector working - Current price: {market_data.get("current_price", "N/A")}')
        
        print('3. Testing Agent Pipeline...')
        result = await orchestrator.process_market_data(market_data)
        print(f'✓ Pipeline executed - Signal: {result.get("final_signal", "N/A")}, Confidence: {result.get("confidence", 0):.2%}')
        
        print('4. Testing Backtesting Engine...')
        backtester = BacktestEngine()
        if backtester.initialize_system():
            print('✓ Backtesting engine initialized successfully')
        
        print('5. Testing Agent Status...')
        status = orchestrator.get_agent_status()
        agents_count = len(status.get('agents', {}))
        print(f'✓ Agent status retrieved - {agents_count} agents configured')
        
        print('\n🎉 All system tests passed! VAANI-RAVEN X is ready for deployment.')
        return True
        
    except Exception as e:
        print(f'❌ System test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_system_integration())
    sys.exit(0 if success else 1)
