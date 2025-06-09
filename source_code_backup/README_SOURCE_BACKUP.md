# VAANI-RAVEN X Source Code Backup

This folder contains readable copies of all source code files for the VAANI-RAVEN X trading system. These files are provided as a backup reference since some files may become unreadable after PyInstaller bundling.

## Purpose

When creating a standalone .exe file with PyInstaller, some source files may be compiled or compressed in a way that makes them difficult to read or modify. This backup ensures you always have access to the original, readable source code for:

- Making modifications and improvements
- Debugging and troubleshooting
- Understanding the system architecture
- Training and documentation purposes

## File Structure

### Backend Files
- `backend_main.py` - FastAPI backend main entry point with WebSocket support
- `orchestrator_runner.py` - Trading orchestrator coordinating all 6 agent layers
- `data_collector.py` - MT5 integration and market data collection
- `pattern_agent.py` - Layer 1: Transformer + CNN-LSTM pattern recognition
- `quant_agent.py` - Layer 2: Technical indicators and volatility analysis
- `sentiment_agent.py` - Layer 3: FinBERT news sentiment analysis
- `risk_agent.py` - Layer 4: Kelly Criterion risk management
- `execution_agent.py` - Layer 5: PPO reinforcement learning execution
- `meta_agent.py` - Layer 6: Performance evaluation and strategy selection

### Frontend Files
- `frontend_App.tsx` - React main application component
- `MarketChart.tsx` - Real-time EUR/USD chart component
- `AgentMonitor.tsx` - Multi-agent status monitoring dashboard
- `TradeManager.tsx` - Trade placement and position management
- `PerformanceMetrics.tsx` - Performance analytics and metrics
- `useWebSocket.ts` - WebSocket hook for real-time data
- `useApi.ts` - API integration hook

### Configuration Files
- `system_config.yaml` - Main system configuration
- `requirements.txt` - Python dependencies
- `package.json` - Node.js dependencies
- `.env` - Environment variables

### Utility Files
- `installer.spec` - PyInstaller configuration
- `test_system_integration.py` - System integration tests
- `performance_log.sql` - Database schema for metrics

## Usage Instructions

1. **For Modifications**: Use these files as reference when making changes to the system
2. **For Debugging**: Compare with bundled versions to identify issues
3. **For Deployment**: Use these files to rebuild the system if needed
4. **For Learning**: Study the architecture and implementation details

## System Architecture

The VAANI-RAVEN X system follows a 6-layer agent architecture:

1. **Pattern Agent** - Analyzes price patterns using Transformer + CNN-LSTM
2. **Quant Agent** - Validates signals with technical indicators
3. **Sentiment Agent** - Filters based on news sentiment
4. **Risk Agent** - Applies position sizing and risk management
5. **Execution Agent** - Optimizes entry/exit timing with RL
6. **Meta Agent** - Evaluates performance and triggers retraining

## Key Features

- **Real-time Trading**: Live EUR/USD data with WebSocket connections
- **Professional UI**: React-based dashboard with TradingView-style charts
- **Multi-Agent Coordination**: Sophisticated signal processing pipeline
- **Risk Management**: Advanced capital protection and drawdown monitoring
- **Self-Learning**: Automated model retraining and strategy adaptation
- **MT5 Integration**: Direct connection to MetaTrader 5 for live trading

## Important Notes

- Always backup this folder before making system changes
- Keep this folder synchronized with any modifications to the main system
- Use version control to track changes to these backup files
- Refer to the main README.md for setup and deployment instructions

## Contact

For questions about the source code or system architecture, refer to the main project documentation or contact the development team.
