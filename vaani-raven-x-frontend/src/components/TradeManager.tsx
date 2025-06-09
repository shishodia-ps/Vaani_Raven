import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Zap, TrendingUp, TrendingDown, Square, Clock, DollarSign } from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  volume: number;
  openPrice: number;
  currentPrice: number;
  profit: number;
  openTime: string;
  status: 'OPEN' | 'CLOSED' | 'PENDING';
}

const TradeManager: React.FC = () => {
  const [trades] = useState<Trade[]>([
    {
      id: 'T001',
      symbol: 'EURUSD',
      type: 'BUY',
      volume: 0.1,
      openPrice: 1.08450,
      currentPrice: 1.08687,
      profit: 23.70,
      openTime: '2024-01-15 14:30:00',
      status: 'OPEN'
    },
    {
      id: 'T002',
      symbol: 'EURUSD',
      type: 'SELL',
      volume: 0.05,
      openPrice: 1.08920,
      currentPrice: 1.08687,
      profit: 11.65,
      openTime: '2024-01-15 15:45:00',
      status: 'OPEN'
    }
  ]);

  const [orderForm, setOrderForm] = useState({
    type: 'BUY',
    volume: '0.1',
    stopLoss: '',
    takeProfit: ''
  });

  const totalProfit = trades.reduce((sum, trade) => sum + trade.profit, 0);
  const openTrades = trades.filter(trade => trade.status === 'OPEN').length;

  const handlePlaceOrder = () => {
    console.log('Placing order:', orderForm);
    alert('Order placement functionality will be integrated with MT5');
  };

  const handleClosePosition = (tradeId: string) => {
    console.log('Closing position:', tradeId);
    alert('Position closing functionality will be integrated with MT5');
  };

  return (
    <div className="space-y-6">
      {/* Trading Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Zap className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-sm text-gray-400">Open Trades</div>
                <div className="text-2xl font-bold">{openTrades}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-green-500" />
              <div>
                <div className="text-sm text-gray-400">Total P&L</div>
                <div className={`text-2xl font-bold ${totalProfit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {'$' + totalProfit.toFixed(2)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-sm text-gray-400">Win Rate</div>
                <div className="text-2xl font-bold">68.5%</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-yellow-500" />
              <div>
                <div className="text-sm text-gray-400">Avg Duration</div>
                <div className="text-2xl font-bold">2.4h</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Order Placement */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Place New Order</CardTitle>
            <CardDescription>Manual order placement for EUR/USD</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant={orderForm.type === 'BUY' ? 'default' : 'outline'}
                onClick={() => setOrderForm({...orderForm, type: 'BUY'})}
                className="bg-green-600 hover:bg-green-700"
              >
                <TrendingUp className="h-4 w-4 mr-2" />
                BUY
              </Button>
              <Button
                variant={orderForm.type === 'SELL' ? 'default' : 'outline'}
                onClick={() => setOrderForm({...orderForm, type: 'SELL'})}
                className="bg-red-600 hover:bg-red-700"
              >
                <TrendingDown className="h-4 w-4 mr-2" />
                SELL
              </Button>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Volume</label>
              <input
                type="number"
                step="0.01"
                value={orderForm.volume}
                onChange={(e) => setOrderForm({...orderForm, volume: e.target.value})}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                placeholder="0.1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Stop Loss</label>
              <input
                type="number"
                step="0.00001"
                value={orderForm.stopLoss}
                onChange={(e) => setOrderForm({...orderForm, stopLoss: e.target.value})}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                placeholder="1.08000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Take Profit</label>
              <input
                type="number"
                step="0.00001"
                value={orderForm.takeProfit}
                onChange={(e) => setOrderForm({...orderForm, takeProfit: e.target.value})}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                placeholder="1.09000"
              />
            </div>

            <Button onClick={handlePlaceOrder} className="w-full">
              Place Order
            </Button>
          </CardContent>
        </Card>

        {/* Open Positions */}
        <div className="lg:col-span-2">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle>Open Positions</CardTitle>
              <CardDescription>Currently active trades</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {trades.filter(trade => trade.status === 'OPEN').map((trade) => (
                  <div key={trade.id} className="p-4 bg-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Badge className={trade.type === 'BUY' ? 'bg-green-600' : 'bg-red-600'}>
                          {trade.type}
                        </Badge>
                        <span className="font-semibold">{trade.symbol}</span>
                        <span className="text-sm text-gray-400">Vol: {trade.volume}</span>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleClosePosition(trade.id)}
                        className="border-red-500 text-red-500 hover:bg-red-500 hover:text-white"
                      >
                        <Square className="h-3 w-3 mr-1" />
                        Close
                      </Button>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-gray-400">Open Price</div>
                        <div className="font-semibold">{trade.openPrice.toFixed(5)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400">Current Price</div>
                        <div className="font-semibold">{trade.currentPrice.toFixed(5)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400">P&L</div>
                        <div className={`font-semibold ${trade.profit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {'$' + trade.profit.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Open Time</div>
                        <div className="font-semibold">{new Date(trade.openTime).toLocaleTimeString()}</div>
                      </div>
                    </div>
                  </div>
                ))}
                
                {trades.filter(trade => trade.status === 'OPEN').length === 0 && (
                  <div className="text-center text-gray-400 py-8">
                    No open positions
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Trade History */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle>Recent Trade History</CardTitle>
          <CardDescription>Last 10 completed trades</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400 py-8">
            Trade history will be populated from MT5 integration
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TradeManager;
