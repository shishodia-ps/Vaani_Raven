"""
VAANI-RAVEN X FastAPI Backend Main Entry Point
This file contains the complete FastAPI backend implementation with WebSocket support
for real-time trading data and agent monitoring.

Original location: /home/ubuntu/EA/backend/main.py
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.runner import TradingOrchestrator
from data_pipeline.data_collector import DataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaani_raven_x.api")

app = FastAPI(title="VAANI-RAVEN X API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = None
data_collector = None
connected_websockets: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the trading system on startup"""
    global orchestrator, data_collector
    
    logger.info("Initializing VAANI-RAVEN X API...")
    
    try:
        orchestrator = TradingOrchestrator()
        
        data_collector = DataCollector()
        
        logger.info("API initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "VAANI-RAVEN X Trading System API", "status": "active"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_initialized": orchestrator is not None,
        "data_collector_active": data_collector is not None
    }

@app.get("/api/market-data")
async def get_market_data():
    """Get current market data"""
    if not data_collector:
        return JSONResponse(
            status_code=503,
            content={"error": "Data collector not initialized"}
        )
    
    try:
        market_data = data_collector.get_current_data()
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "Orchestrator not initialized"}
        )
    
    try:
        status = orchestrator.get_agent_status()
        return status
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/trading/signal")
async def get_trading_signal():
    """Get latest trading signal from orchestrator"""
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "Orchestrator not initialized"}
        )
    
    try:
        market_data = data_collector.get_current_data() if data_collector else None
        
        signal = orchestrator.process_signal(market_data)
        
        return {
            "signal": signal,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data
        }
    except Exception as e:
        logger.error(f"Error getting trading signal: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            if data_collector:
                market_data = data_collector.get_current_data()
                
                agent_status = orchestrator.get_agent_status() if orchestrator else {}
                
                signal = orchestrator.process_signal(market_data) if orchestrator else {"action": "HOLD", "confidence": 0.0}
                
                await websocket.send_text(json.dumps({
                    "type": "market_update",
                    "data": {
                        "market_data": market_data,
                        "agent_status": agent_status,
                        "signal": signal,
                        "timestamp": datetime.now().isoformat()
                    }
                }))
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

@app.post("/api/trading/place-order")
async def place_order(order_data: Dict[str, Any]):
    """Place a trading order"""
    logger.info(f"Order placement request: {order_data}")
    
    return {
        "status": "success",
        "message": "Order placement functionality will be integrated with MT5",
        "order_id": f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/trading/close-position")
async def close_position(position_data: Dict[str, Any]):
    """Close a trading position"""
    logger.info(f"Position close request: {position_data}")
    
    return {
        "status": "success",
        "message": "Position closing functionality will be integrated with MT5",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
