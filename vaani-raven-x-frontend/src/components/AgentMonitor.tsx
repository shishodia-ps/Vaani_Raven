import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Brain, Activity, Shield, TrendingUp, Zap, Target } from 'lucide-react';

interface AgentStatus {
  orchestrator: {
    active: boolean;
    config_loaded: boolean;
    agents_initialized: number;
    pipeline_executions: number;
  };
  agents: Record<string, any>;
}

interface AgentMonitorProps {
  agentStatus: AgentStatus | null;
}

const AgentMonitor: React.FC<AgentMonitorProps> = ({ agentStatus }) => {
  if (!agentStatus) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        <div className="text-center">
          <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <div className="text-lg mb-2">Agent Status Loading...</div>
          <div className="text-sm">Connecting to orchestrator...</div>
        </div>
      </div>
    );
  }

  const agentLayers = [
    {
      name: 'Pattern Agent',
      layer: 1,
      icon: Brain,
      description: 'Transformer + CNN-LSTM trend detection',
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      status: agentStatus.agents.pattern_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    },
    {
      name: 'Quant Agent',
      layer: 2,
      icon: TrendingUp,
      description: 'Technical indicators & volatility analysis',
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      status: agentStatus.agents.quant_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    },
    {
      name: 'Sentiment Agent',
      layer: 3,
      icon: Activity,
      description: 'FinBERT news analysis & event filtering',
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
      status: agentStatus.agents.sentiment_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    },
    {
      name: 'Risk Agent',
      layer: 4,
      icon: Shield,
      description: 'Kelly Criterion & dynamic risk management',
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      status: agentStatus.agents.risk_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    },
    {
      name: 'Execution Agent',
      layer: 5,
      icon: Zap,
      description: 'PPO reinforcement learning optimization',
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      status: agentStatus.agents.execution_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    },
    {
      name: 'Meta Agent',
      layer: 6,
      icon: Target,
      description: 'Performance evaluation & strategy selection',
      color: 'text-orange-500',
      bgColor: 'bg-orange-500/10',
      status: agentStatus.agents.meta_agent || { active: false, last_signal: 'N/A', confidence: 0 }
    }
  ];

  const getStatusBadge = (active: boolean) => {
    return active ? (
      <Badge className="bg-green-600 text-white">Active</Badge>
    ) : (
      <Badge variant="outline" className="border-red-500 text-red-500">Inactive</Badge>
    );
  };

  return (
    <div className="space-y-6">
      {/* Orchestrator Status */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-blue-500" />
            <span>Orchestrator Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">
                {agentStatus.orchestrator.active ? '✓' : '✗'}
              </div>
              <div className="text-sm text-gray-400">System Active</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">
                {agentStatus.orchestrator.agents_initialized}
              </div>
              <div className="text-sm text-gray-400">Agents Initialized</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-500">
                {agentStatus.orchestrator.pipeline_executions}
              </div>
              <div className="text-sm text-gray-400">Pipeline Executions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-500">
                {agentStatus.orchestrator.config_loaded ? '✓' : '✗'}
              </div>
              <div className="text-sm text-gray-400">Config Loaded</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Agent Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agentLayers.map((agent) => {
          const IconComponent = agent.icon;
          return (
            <Card key={agent.layer} className={`bg-gray-800 border-gray-700 ${agent.bgColor}`}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <IconComponent className={`h-5 w-5 ${agent.color}`} />
                    <div>
                      <CardTitle className="text-sm">Layer {agent.layer}</CardTitle>
                      <CardDescription className="text-xs">{agent.name}</CardDescription>
                    </div>
                  </div>
                  {getStatusBadge(agent.status.active)}
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  <div className="text-xs text-gray-400 mb-2">
                    {agent.description}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="text-gray-400">Last Signal</div>
                      <div className="font-semibold">
                        {agent.status.last_signal || 'HOLD'}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Confidence</div>
                      <div className="font-semibold">
                        {((agent.status.confidence || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {agent.status.processing_time && (
                    <div className="text-xs">
                      <div className="text-gray-400">Processing Time</div>
                      <div className="font-semibold">
                        {agent.status.processing_time.toFixed(1)}ms
                      </div>
                    </div>
                  )}

                  {agent.status.last_update && (
                    <div className="text-xs">
                      <div className="text-gray-400">Last Update</div>
                      <div className="font-semibold">
                        {new Date(agent.status.last_update).toLocaleTimeString()}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Pipeline Flow Visualization */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle>Signal Pipeline Flow</CardTitle>
          <CardDescription>Real-time signal processing through agent layers</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between space-x-2 overflow-x-auto">
            {agentLayers.map((agent, index) => {
              const IconComponent = agent.icon;
              return (
                <React.Fragment key={agent.layer}>
                  <div className="flex flex-col items-center space-y-2 min-w-0 flex-shrink-0">
                    <div className={`p-3 rounded-full ${agent.bgColor} ${agent.status.active ? 'ring-2 ring-blue-500' : ''}`}>
                      <IconComponent className={`h-6 w-6 ${agent.color}`} />
                    </div>
                    <div className="text-xs text-center">
                      <div className="font-semibold">L{agent.layer}</div>
                      <div className="text-gray-400">{agent.name.split(' ')[0]}</div>
                    </div>
                    {agent.status.active && (
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    )}
                  </div>
                  {index < agentLayers.length - 1 && (
                    <div className="flex-1 h-px bg-gray-600 min-w-4"></div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AgentMonitor;
