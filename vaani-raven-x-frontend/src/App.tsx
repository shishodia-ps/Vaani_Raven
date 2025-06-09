import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Activity, TrendingUp, Brain, Zap, BarChart3, Settings } from 'lucide-react';
import MarketChart from './components/MarketChart';
import AgentMonitor from './components/AgentMonitor';
import TradeManager from './components/TradeManager';
import PerformanceMetrics from './components/PerformanceMetrics';
import MT5Config from './components/MT5Config';
import { useWebSocket } from './hooks/useWebSocket';


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
  final_signal: string;
  confidence: number;
  reason: string;
  pipeline_duration_ms: number;
  agent_signals: any[];
  timestamp: string;
}

interface AgentStatus {
  orchestrator: {
    active: boolean;
    config_loaded: boolean;
    agents_initialized: number;
    pipeline_executions: number;
  };
  agents: Record<string, any>;
}

function App() {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [tradingSignal, setTradingSignal] = useState<TradingSignal | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [mt5ConnectionStatus, setMt5ConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');

  const { data: wsData, connectionState } = useWebSocket('ws://localhost:8000/ws/market-data');

  useEffect(() => {
    setConnectionStatus(connectionState);
  }, [connectionState]);

  useEffect(() => {
    if (wsData) {
      const { market_data, signal, agent_status, mt5_status } = wsData.data;
      
      if (market_data) {
        setMarketData(market_data);
      }
      
      if (signal) {
        setTradingSignal(signal);
      }
      
      if (agent_status) {
        setAgentStatus(agent_status);
      }

      if (mt5_status) {
        setMt5ConnectionStatus(mt5_status.connected ? 'connected' : 'disconnected');
      }
    }
  }, [wsData]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-500';
      case 'SELL': return 'bg-red-500';
      case 'HALT': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getConnectionStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500';
      case 'connecting': return 'bg-yellow-500';
      default: return 'bg-red-500';
    }
  };

  const handleMT5CredentialsUpdate = async (credentials: any) => {
    console.log('MT5 credentials updated:', credentials);
    setMt5ConnectionStatus('connecting');
    
    try {
      const response = await fetch('http://localhost:8000/api/mt5/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });
      
      const result = await response.json();
      if (result.status === 'success') {
        setMt5ConnectionStatus('connected');
      } else {
        setMt5ConnectionStatus('disconnected');
      }
    } catch (error) {
      console.error('Error updating MT5 credentials:', error);
      setMt5ConnectionStatus('disconnected');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Brain className="h-8 w-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold">VAANI-RAVEN X</h1>
                <p className="text-sm text-gray-400">Multi-Agent EUR/USD Trading System</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor(connectionStatus)}`}></div>
                  <span className="text-sm">
                    {connectionStatus === 'connected' ? 'Connected' : 
                     connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    mt5ConnectionStatus === 'connected' ? 'bg-green-500' :
                    mt5ConnectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-sm">
                    {mt5ConnectionStatus === 'connected' ? 'MT5 Live' : 
                     mt5ConnectionStatus === 'connecting' ? 'MT5 Connecting...' : 'Mock Data'}
                  </span>
                </div>
              </div>
              
              {/* Current Price */}
              {marketData && (
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-4 w-4 text-blue-500" />
                      <div>
                        <div className="text-lg font-bold">
                          {marketData.current_price.toFixed(5)}
                        </div>
                        <div className="text-xs text-gray-400">
                          Spread: {marketData.spread.toFixed(5)}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
              
              {/* Current Signal */}
              {tradingSignal && (
                <Badge className={`${getSignalColor(tradingSignal.final_signal)} text-white`}>
                  {tradingSignal.final_signal} ({(tradingSignal.confidence * 100).toFixed(1)}%)
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs defaultValue="trading" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-gray-800">
            <TabsTrigger value="trading" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Live Trading</span>
            </TabsTrigger>
            <TabsTrigger value="agents" className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span>Agent Monitor</span>
            </TabsTrigger>
            <TabsTrigger value="trades" className="flex items-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Trade Manager</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>Performance</span>
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center space-x-2">
              <Settings className="h-4 w-4" />
              <span>Settings</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="trading" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Market Chart */}
              <div className="lg:col-span-2">
                <MarketChart data={marketData} />
              </div>
              
              {/* Trading Signal Panel */}
              <div className="space-y-4">
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle>Latest Signal</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {tradingSignal ? (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span>Signal:</span>
                          <Badge className={`${getSignalColor(tradingSignal.final_signal)} text-white`}>
                            {tradingSignal.final_signal}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Confidence:</span>
                          <span className="font-bold">{(tradingSignal.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Duration:</span>
                          <span>{tradingSignal.pipeline_duration_ms.toFixed(1)}ms</span>
                        </div>
                        <div className="text-sm text-gray-400">
                          <strong>Reason:</strong> {tradingSignal.reason}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-400">
                        No signal generated yet
                      </div>
                    )}
                  </CardContent>
                </Card>
                
                {/* Market Info */}
                {marketData && (
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle>Market Info</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between">
                        <span>Bid:</span>
                        <span>{marketData.bid.toFixed(5)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Ask:</span>
                        <span>{marketData.ask.toFixed(5)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Volume:</span>
                        <span>{marketData.volume.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Volatility:</span>
                        <span>{(marketData.volatility * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ATR:</span>
                        <span>{marketData.atr.toFixed(5)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Source:</span>
                        <Badge variant="outline">{marketData.source.toUpperCase()}</Badge>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="agents">
            <AgentMonitor agentStatus={agentStatus} />
          </TabsContent>

          <TabsContent value="trades">
            <TradeManager />
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceMetrics />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <MT5Config 
                onCredentialsUpdate={handleMT5CredentialsUpdate}
                connectionStatus={mt5ConnectionStatus}
              />
              
              {/* Additional settings components can be added here */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle>System Settings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center text-gray-400 py-8">
                    Additional system configuration options will be available here
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
