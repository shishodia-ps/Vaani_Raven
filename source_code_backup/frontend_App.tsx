/*
VAANI-RAVEN X React Frontend Main Application Component
This file contains the complete React application with real-time trading dashboard,
agent monitoring, and trade management capabilities.

Original location: /home/ubuntu/EA/vaani-raven-x-frontend/src/App.tsx
*/

import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Activity, TrendingUp, Brain, Zap, BarChart3 } from 'lucide-react';
import MarketChart from './components/MarketChart';
import AgentMonitor from './components/AgentMonitor';
import TradeManager from './components/TradeManager';
import PerformanceMetrics from './components/PerformanceMetrics';
import { useWebSocket } from './hooks/useWebSocket';
import { useApi } from './hooks/useApi';

interface MarketData {
  current_price: number;
  bid: number;
  ask: number;
  spread: number;
  volume: number;
  volatility: number;
  atr: number;
  ohlcv: Array<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
  source: string;
}

interface TradingSignal {
  action: string;
  confidence: number;
  reasoning: string;
  timestamp: string;
}

interface AgentStatus {
  orchestrator: {
    status: string;
    agents_initialized: number;
    pipeline_executions: number;
    config_loaded: boolean;
  };
  agents: {
    [key: string]: {
      name: string;
      description: string;
      status: string;
      last_signal: string;
      confidence: number;
    };
  };
}

function App() {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [tradingSignal, setTradingSignal] = useState<TradingSignal | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  const { data: wsData, connectionState } = useWebSocket('ws://localhost:8000/ws/market-data');
  const { data: healthData } = useApi('/api/health', 5000);

  useEffect(() => {
    setConnectionStatus(connectionState);
  }, [connectionState]);

  useEffect(() => {
    if (wsData) {
      const { market_data, signal, agent_status } = wsData.data;
      
      if (market_data) {
        setMarketData(market_data);
      }
      
      if (signal) {
        setTradingSignal(signal);
      }
      
      if (agent_status) {
        setAgentStatus(agent_status);
      }
    }
  }, [wsData]);

  const getSignalColor = (action: string) => {
    switch (action?.toUpperCase()) {
      case 'BUY': return 'bg-green-600';
      case 'SELL': return 'bg-red-600';
      default: return 'bg-gray-600';
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'bg-green-500';
      case 'connecting': return 'bg-yellow-500';
      default: return 'bg-red-500';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold">VAANI-RAVEN X</h1>
              <p className="text-sm text-gray-400">Multi-Agent EUR/USD Trading System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor()}`}></div>
              <span className="text-sm capitalize">{connectionStatus}</span>
            </div>
            
            {marketData && (
              <div className="text-right">
                <div className="text-lg font-bold">{marketData.current_price.toFixed(5)}</div>
                <div className="text-xs text-gray-400">Spread: {marketData.spread.toFixed(5)}</div>
              </div>
            )}
            
            {tradingSignal && (
              <Badge className={getSignalColor(tradingSignal.action)}>
                {tradingSignal.action} ({(tradingSignal.confidence * 100).toFixed(1)}%)
              </Badge>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        <Tabs defaultValue="live-trading" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-gray-800">
            <TabsTrigger value="live-trading" className="flex items-center space-x-2">
              <Activity className="w-4 h-4" />
              <span>Live Trading</span>
            </TabsTrigger>
            <TabsTrigger value="agent-monitor" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>Agent Monitor</span>
            </TabsTrigger>
            <TabsTrigger value="trade-manager" className="flex items-center space-x-2">
              <Zap className="w-4 h-4" />
              <span>Trade Manager</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4" />
              <span>Performance</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="live-trading" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Chart */}
              <div className="lg:col-span-2">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <TrendingUp className="w-5 h-5" />
                      <span>EUR/USD Live Chart</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <MarketChart data={marketData} />
                  </CardContent>
                </Card>
              </div>

              {/* Side Panel */}
              <div className="space-y-6">
                {/* Latest Signal */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Latest Signal</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {tradingSignal ? (
                      <div className="space-y-2">
                        <Badge className={getSignalColor(tradingSignal.action)}>
                          {tradingSignal.action}
                        </Badge>
                        <div className="text-sm text-gray-400">
                          Confidence: {(tradingSignal.confidence * 100).toFixed(1)}%
                        </div>
                        {tradingSignal.reasoning && (
                          <div className="text-xs text-gray-500">
                            {tradingSignal.reasoning}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-gray-400">No signal generated yet</div>
                    )}
                  </CardContent>
                </Card>

                {/* Market Info */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Market Info</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {marketData ? (
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Bid:</span>
                          <span>{marketData.bid.toFixed(5)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Ask:</span>
                          <span>{marketData.ask.toFixed(5)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Volume:</span>
                          <span>{marketData.volume.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Volatility:</span>
                          <span className="text-yellow-400">{(marketData.volatility * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">ATR:</span>
                          <span>{marketData.atr.toFixed(5)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Source:</span>
                          <Badge variant="outline">{marketData.source}</Badge>
                        </div>
                      </div>
                    ) : (
                      <div className="text-gray-400">Loading market data...</div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="agent-monitor">
            <AgentMonitor agentStatus={agentStatus} />
          </TabsContent>

          <TabsContent value="trade-manager">
            <TradeManager />
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceMetrics />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
