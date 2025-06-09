"""
FastAPI Backend for VAANI-RAVEN X Trading Dashboard
Real-time WebSocket support for EUR/USD data and agent monitoring
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.runner import OrchestatorRunner
from data_pipeline.data_collector import DataCollector

app = FastAPI(title="VAANI-RAVEN X Trading API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator: Optional[OrchestatorRunner] = None
data_collector: Optional[DataCollector] = None
websocket_connections: List[WebSocket] = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaani_raven_x.api")

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global orchestrator, data_collector
    
    try:
        logger.info("Initializing VAANI-RAVEN X API...")
        
        orchestrator = OrchestatorRunner()
        config = orchestrator.config
        data_collector = DataCollector(config)
        
        logger.info("API initialization complete")
        
        asyncio.create_task(stream_market_data())
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator, data_collector
    
    if orchestrator:
        orchestrator.shutdown()
    
    if data_collector:
        data_collector.cleanup()
    
    logger.info("API shutdown complete")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "orchestrator": orchestrator is not None,
            "data_collector": data_collector is not None
        }
    }

@app.get("/api/market-data")
async def get_market_data():
    """Get current market data"""
    if not data_collector:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    try:
        market_data = data_collector.get_current_market_data()
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        status = orchestrator.get_agent_status()
        return status
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline/metrics")
async def get_pipeline_metrics():
    """Get pipeline performance metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        metrics = orchestrator.get_pipeline_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/signal")
async def generate_trading_signal():
    """Generate new trading signal"""
    if not orchestrator or not data_collector:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        market_data = data_collector.get_current_market_data()
        result = await orchestrator.process_market_data(market_data)
        return result
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical-data")
async def get_historical_data(years: int = 1):
    """Get historical EUR/USD data"""
    if not data_collector:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    try:
        historical_data = data_collector.get_historical_data(years)
        
        data_dict = historical_data.tail(1000).to_dict('records')
        
        return {
            "data": data_dict,
            "total_records": len(historical_data),
            "years": years
        }
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mt5/configure")
async def configure_mt5_credentials(credentials: Dict[str, Any]):
    """Configure MT5 credentials and establish connection"""
    global data_collector
    
    try:
        if not data_collector:
            raise HTTPException(status_code=503, detail="Data collector not initialized")
        
        required_fields = ['server', 'login', 'password']
        for field in required_fields:
            if not credentials.get(field):
                return {
                    "status": "error",
                    "message": f"Missing required field: {field}",
                    "connected": False,
                    "timestamp": datetime.now().isoformat()
                }
        
        success = data_collector.update_mt5_credentials(credentials)
        
        if success:
            return {
                "status": "success",
                "message": "MT5 credentials configured successfully",
                "connected": data_collector.mt5_connected,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to configure MT5 credentials - check login details",
                "connected": False,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error configuring MT5 credentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mt5/test-connection")
async def test_mt5_connection(credentials: Dict[str, Any]):
    """Test MT5 connection with provided credentials"""
    try:
        if not data_collector:
            raise HTTPException(status_code=503, detail="Data collector not initialized")
        
        success = data_collector.test_mt5_connection(credentials)
        
        return {
            "success": success,
            "message": "Connection test successful" if success else "Connection test failed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing MT5 connection: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/mt5/status")
async def get_mt5_status():
    """Get current MT5 connection status"""
    if not data_collector:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    return {
        "connected": data_collector.mt5_connected,
        "source": "mt5" if data_collector.mt5_connected else "mock",
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data streaming"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                await websocket.send_text(f"Echo: {data}")
            except asyncio.TimeoutError:
                pass
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

async def stream_market_data():
    """Background task to stream market data to WebSocket clients"""
    while True:
        try:
            if websocket_connections and data_collector and orchestrator:
                market_data = data_collector.get_current_market_data()
                
                signal_result = await orchestrator.process_market_data(market_data)
                
                agent_status = orchestrator.get_agent_status()
                
                stream_data = {
                    "type": "market_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "market_data": market_data,
                        "signal": signal_result,
                        "agent_status": agent_status,
                        "mt5_status": {
                            "connected": data_collector.mt5_connected,
                            "source": "mt5" if data_collector.mt5_connected else "mock"
                        }
                    }
                }
                
                disconnected_clients = []
                for websocket in websocket_connections:
                    try:
                        await websocket.send_text(json.dumps(stream_data, default=str))
                    except Exception as e:
                        logger.error(f"Error sending to WebSocket client: {e}")
                        disconnected_clients.append(websocket)
                
                for client in disconnected_clients:
                    websocket_connections.remove(client)
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in market data streaming: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
