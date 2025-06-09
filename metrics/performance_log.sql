
CREATE TABLE IF NOT EXISTS agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    signal_type TEXT,
    confidence REAL,
    market_price REAL,
    volatility REAL,
    success BOOLEAN,
    profit REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_ms REAL,
    final_signal TEXT,
    final_confidence REAL,
    agents_processed INTEGER,
    success BOOLEAN,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS trading_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_start DATETIME,
    session_end DATETIME,
    total_trades INTEGER,
    profitable_trades INTEGER,
    total_profit REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    win_rate REAL
);

CREATE TABLE IF NOT EXISTS market_data_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT DEFAULT 'EURUSD',
    price REAL,
    spread REAL,
    volume INTEGER,
    volatility REAL,
    atr REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS retraining_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    agent_name TEXT NOT NULL,
    trigger_reason TEXT,
    previous_performance REAL,
    training_duration_minutes INTEGER,
    new_performance REAL,
    success BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_agent_performance_timestamp ON agent_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_performance_agent ON agent_performance(agent_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_timestamp ON pipeline_executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_retraining_timestamp ON retraining_log(timestamp);
