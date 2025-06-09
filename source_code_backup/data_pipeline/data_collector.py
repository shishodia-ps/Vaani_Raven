"""
Data Collector - MT5 Integration and Historical Data Management
Handles real-time and historical EUR/USD data collection
"""

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path

class DataCollector:
    """Data collection and management for VAANI-RAVEN X"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mt5_config = config.get('mt5', {})
        self.data_config = config.get('data', {})
        
        self.symbol = config.get('system', {}).get('symbol', 'EURUSD')
        self.timeframe = config.get('system', {}).get('timeframe', 'M15')
        self.historical_years = self.data_config.get('historical_years', 5)
        
        self.logger = self._setup_logger()
        self.mt5_connected = False
        
        self._initialize_mt5()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup data collector logger"""
        logger = logging.getLogger("vaani_raven_x.data_collector")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - DATA_COLLECTOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_mt5(self) -> None:
        """Initialize MT5 connection"""
        try:
            if not MT5_AVAILABLE or mt5 is None:
                self.logger.warning("MT5 not available on this platform, using mock data")
                self.mt5_connected = False
                return
            
            if not mt5.initialize():
                self.logger.warning("MT5 initialization failed, using mock data")
                self.mt5_connected = False
                return
            
            login = self.mt5_config.get('login')
            password = self.mt5_config.get('password')
            server = self.mt5_config.get('server', 'MetaQuotes-Demo')
            
            if login and password:
                if mt5.login(login, password, server):
                    self.logger.info(f"Connected to MT5: {server}")
                    self.mt5_connected = True
                else:
                    self.logger.error("Failed to login to MT5")
                    self.mt5_connected = False
            else:
                self.logger.info("MT5 credentials not provided, using demo mode")
                self.mt5_connected = True
                
        except Exception as e:
            self.logger.error(f"MT5 initialization failed: {e}")
            self.mt5_connected = False
    
    def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for EUR/USD"""
        
        if self.mt5_connected:
            return self._get_mt5_current_data()
        else:
            return self._get_mock_current_data()
    
    def _get_mt5_current_data(self) -> Dict[str, Any]:
        """Get current data from MT5"""
        try:
            if mt5 is None:
                raise Exception("MT5 not available")
                
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                raise Exception(f"Symbol {self.symbol} not found")
            
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                raise Exception(f"Failed to get tick for {self.symbol}")
            
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 100)
            if rates is None or len(rates) == 0:
                raise Exception("Failed to get historical rates")
            
            rates_df = pd.DataFrame(rates)
            rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
            
            ohlcv_data = []
            for _, row in rates_df.tail(20).iterrows():
                ohlcv_data.append({
                    'timestamp': row['time'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['tick_volume'])
                })
            
            current_price = float(tick.bid + tick.ask) / 2
            spread = float(tick.ask - tick.bid)
            
            volatility = self._calculate_volatility(rates_df['close'].values.astype(float))
            atr = self._calculate_atr(rates_df)
            
            return {
                'current_price': current_price,
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'spread': spread,
                'volume': int(tick.volume),
                'volatility': volatility,
                'atr': atr,
                'ohlcv': ohlcv_data,
                'timestamp': datetime.now(),
                'source': 'mt5'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting MT5 data: {e}")
            return self._get_mock_current_data()
    
    def _get_mock_current_data(self) -> Dict[str, Any]:
        """Get mock market data for testing"""
        
        base_price = 1.0850
        volatility = 0.015
        
        current_price = base_price + np.random.normal(0, volatility * 0.1)
        spread = 0.0001 + np.random.uniform(0, 0.0002)
        
        ohlcv_data = []
        for i in range(20):
            timestamp = datetime.now() - timedelta(minutes=15 * (19 - i))
            
            price_change = np.random.normal(0, volatility * 0.05)
            open_price = base_price + price_change
            
            high_low_range = np.random.uniform(0.0005, 0.002)
            high_price = open_price + high_low_range * np.random.uniform(0.3, 1.0)
            low_price = open_price - high_low_range * np.random.uniform(0.3, 1.0)
            
            close_change = np.random.normal(0, volatility * 0.03)
            close_price = open_price + close_change
            
            volume = int(np.random.uniform(10000, 100000))
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        return {
            'current_price': round(current_price, 5),
            'bid': round(current_price - spread/2, 5),
            'ask': round(current_price + spread/2, 5),
            'spread': round(spread, 5),
            'volume': int(np.random.uniform(50000, 200000)),
            'volatility': volatility,
            'atr': round(np.random.uniform(0.0008, 0.0015), 5),
            'ohlcv': ohlcv_data,
            'timestamp': datetime.now(),
            'source': 'mock'
        }
    
    def get_historical_data(self, years: Optional[int] = None) -> pd.DataFrame:
        """Get historical EUR/USD data"""
        
        years_to_use = years or self.historical_years or 2
        
        if self.mt5_connected:
            return self._get_mt5_historical_data(years_to_use)
        else:
            return self._get_mock_historical_data(years_to_use)
    
    def _get_mt5_historical_data(self, years: int) -> pd.DataFrame:
        """Get historical data from MT5"""
        try:
            if mt5 is None:
                raise Exception("MT5 not available")
                
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            rates = mt5.copy_rates_range(
                self.symbol, 
                mt5.TIMEFRAME_M15, 
                start_date, 
                end_date
            )
            
            if rates is None or len(rates) == 0:
                raise Exception("Failed to get historical rates from MT5")
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df.rename(columns={
                'tick_volume': 'volume'
            }, inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} historical records from MT5")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting MT5 historical data: {e}")
            return self._get_mock_historical_data(years)
    
    def _get_mock_historical_data(self, years: int) -> pd.DataFrame:
        """Generate mock historical data"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        base_price = 1.0850
        volatility = 0.015
        trend = 0.0001
        
        prices = []
        current_price = base_price
        
        for i, timestamp in enumerate(date_range):
            price_change = np.random.normal(trend, volatility)
            current_price += price_change
            
            high_low_range = np.random.uniform(0.0005, 0.002)
            open_price = current_price + np.random.normal(0, volatility * 0.5)
            high_price = open_price + high_low_range * np.random.uniform(0.3, 1.0)
            low_price = open_price - high_low_range * np.random.uniform(0.3, 1.0)
            close_price = open_price + np.random.normal(0, volatility * 0.3)
            
            volume = int(np.random.uniform(10000, 100000))
            
            prices.append({
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        df = pd.DataFrame(prices, index=date_range)
        
        self.logger.info(f"Generated {len(df)} mock historical records")
        return df
    
    def _calculate_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < window:
            return 0.015
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-window:])
        
        return float(volatility)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.001
        
        high = df['high'].values[-period:]
        low = df['low'].values[-period:]
        close = df['close'].values[-period:]
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        return float(np.mean(tr_list)) if tr_list else 0.001
    
    def save_data(self, df: pd.DataFrame, filename: str) -> bool:
        """Save data to file"""
        try:
            data_dir = Path(self.data_config.get('storage_path', './data'))
            data_dir.mkdir(exist_ok=True)
            
            filepath = data_dir / filename
            df.to_csv(filepath)
            
            self.logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file"""
        try:
            data_dir = Path(self.data_config.get('storage_path', './data'))
            filepath = data_dir / filename
            
            if not filepath.exists():
                self.logger.warning(f"Data file not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            self.logger.info(f"Data loaded from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def cleanup(self) -> None:
        """Cleanup MT5 connection"""
        if self.mt5_connected and mt5 is not None:
            try:
                mt5.shutdown()
                self.logger.info("MT5 connection closed")
            except Exception as e:
                self.logger.error(f"Error closing MT5: {e}")
        
        self.mt5_connected = False
