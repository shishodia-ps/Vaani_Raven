import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, DollarSign, Target, Shield } from 'lucide-react';

const PerformanceMetrics: React.FC = () => {
  const equityData = [
    { date: '2024-01-01', equity: 10000, drawdown: 0 },
    { date: '2024-01-02', equity: 10150, drawdown: -50 },
    { date: '2024-01-03', equity: 10300, drawdown: 0 },
    { date: '2024-01-04', equity: 10250, drawdown: -50 },
    { date: '2024-01-05', equity: 10400, drawdown: 0 },
    { date: '2024-01-06', equity: 10550, drawdown: 0 },
    { date: '2024-01-07', equity: 10480, drawdown: -70 },
    { date: '2024-01-08', equity: 10620, drawdown: 0 },
    { date: '2024-01-09', equity: 10750, drawdown: 0 },
    { date: '2024-01-10', equity: 10890, drawdown: 0 },
  ];

  const agentPerformance = [
    { agent: 'Pattern', accuracy: 72.5, signals: 145 },
    { agent: 'Quant', accuracy: 68.3, signals: 132 },
    { agent: 'Sentiment', accuracy: 75.1, signals: 89 },
    { agent: 'Risk', accuracy: 82.4, signals: 156 },
    { agent: 'Execution', accuracy: 79.2, signals: 134 },
    { agent: 'Meta', accuracy: 85.6, signals: 98 },
  ];

  const signalDistribution = [
    { name: 'BUY', value: 45, color: '#10B981' },
    { name: 'SELL', value: 38, color: '#EF4444' },
    { name: 'HOLD', value: 17, color: '#6B7280' },
  ];

  const monthlyReturns = [
    { month: 'Jan', return: 8.9 },
    { month: 'Feb', return: -2.1 },
    { month: 'Mar', return: 12.4 },
    { month: 'Apr', return: 5.7 },
    { month: 'May', return: -1.8 },
    { month: 'Jun', return: 9.3 },
  ];

  return (
    <div className="space-y-6">
      {/* Key Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <div>
                <div className="text-sm text-gray-400">Total Return</div>
                <div className="text-2xl font-bold text-green-500">+8.9%</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-sm text-gray-400">Sharpe Ratio</div>
                <div className="text-2xl font-bold text-blue-500">1.42</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Shield className="h-4 w-4 text-red-500" />
              <div>
                <div className="text-sm text-gray-400">Max Drawdown</div>
                <div className="text-2xl font-bold text-red-500">-2.8%</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-yellow-500" />
              <div>
                <div className="text-sm text-gray-400">Win Rate</div>
                <div className="text-2xl font-bold text-yellow-500">68.5%</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Equity Curve</CardTitle>
            <CardDescription>Portfolio value over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="equity" 
                    stroke="#10B981" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Agent Performance */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Agent Performance</CardTitle>
            <CardDescription>Accuracy by agent layer</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={agentPerformance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="agent" 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Bar dataKey="accuracy" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Signal Distribution */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Signal Distribution</CardTitle>
            <CardDescription>Breakdown of trading signals</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={signalDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {signalDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      color: '#F9FAFB'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Monthly Returns */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Monthly Returns</CardTitle>
            <CardDescription>Performance by month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="month" 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      color: '#F9FAFB'
                    }}
                    formatter={(value: number) => [`${value}%`, 'Return']}
                  />
                  <Bar 
                    dataKey="return" 
                    fill="#3B82F6"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Statistics */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle>Detailed Statistics</CardTitle>
          <CardDescription>Comprehensive performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Total Trades</div>
              <div className="text-xl font-bold">247</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Winning Trades</div>
              <div className="text-xl font-bold text-green-500">169</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Losing Trades</div>
              <div className="text-xl font-bold text-red-500">78</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Avg Trade Duration</div>
              <div className="text-xl font-bold">2.4h</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Profit Factor</div>
              <div className="text-xl font-bold text-blue-500">1.68</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Sortino Ratio</div>
              <div className="text-xl font-bold text-purple-500">1.89</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Calmar Ratio</div>
              <div className="text-xl font-bold text-orange-500">3.18</div>
            </div>
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Recovery Factor</div>
              <div className="text-xl font-bold text-cyan-500">3.21</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceMetrics;
