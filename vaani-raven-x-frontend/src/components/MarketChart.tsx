import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

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

interface MarketChartProps {
  data: MarketData | null;
}

const MarketChart: React.FC<MarketChartProps> = ({ data }) => {
  if (!data || !data.ohlcv || data.ohlcv.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="text-lg mb-2">No market data available</div>
          <div className="text-sm">Waiting for EUR/USD price feed...</div>
        </div>
      </div>
    );
  }

  const chartData = data.ohlcv.slice(-100).map((candle) => ({
    time: new Date(candle.timestamp).toLocaleTimeString(),
    price: candle.close,
    volume: candle.volume,
    high: candle.high,
    low: candle.low,
    open: candle.open,
  }));

  const formatPrice = (value: number) => value.toFixed(5);

  return (
    <div className="space-y-4">
      {/* Price Chart */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9CA3AF"
              fontSize={12}
              interval="preserveStartEnd"
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={formatPrice}
              domain={['dataMin - 0.0001', 'dataMax + 0.0001']}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '6px',
                color: '#F9FAFB'
              }}
              formatter={(value: number) => [formatPrice(value), 'Price']}
              labelStyle={{ color: '#9CA3AF' }}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#3B82F6" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: '#3B82F6', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Current Price Info */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-700 p-3 rounded">
          <div className="text-gray-400">Current Price</div>
          <div className="text-lg font-bold text-blue-400">
            {formatPrice(data.current_price)}
          </div>
        </div>
        <div className="bg-gray-700 p-3 rounded">
          <div className="text-gray-400">Spread</div>
          <div className="text-lg font-bold">
            {formatPrice(data.spread)}
          </div>
        </div>
        <div className="bg-gray-700 p-3 rounded">
          <div className="text-gray-400">Volatility</div>
          <div className="text-lg font-bold text-yellow-400">
            {(data.volatility * 100).toFixed(2)}%
          </div>
        </div>
        <div className="bg-gray-700 p-3 rounded">
          <div className="text-gray-400">ATR</div>
          <div className="text-lg font-bold">
            {formatPrice(data.atr)}
          </div>
        </div>
      </div>

      {/* Volume Chart */}
      <div className="h-32">
        <div className="text-sm text-gray-400 mb-2">Volume</div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="time" 
              stroke="#9CA3AF"
              fontSize={10}
              interval="preserveStartEnd"
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={10}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '6px',
                color: '#F9FAFB'
              }}
              formatter={(value: number) => [value.toLocaleString(), 'Volume']}
            />
            <Line 
              type="monotone" 
              dataKey="volume" 
              stroke="#10B981" 
              strokeWidth={1}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MarketChart;
