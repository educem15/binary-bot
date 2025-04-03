from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import logging
import json
import os

from app.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Binary Options Trading Signals API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

# Initialize signal generator
signal_generator = SignalGenerator()

class TradingResult(BaseModel):
    pair: str
    timeframe: str
    timestamp: str
    direction: str
    entry_price: float
    exit_price: float
    success: bool
    profit_loss: float

class SignalRequest(BaseModel):
    timeframes: Optional[List[str]] = None
    pairs: Optional[List[str]] = None
    min_strength: Optional[float] = 3.0

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse('app/frontend/index.html')

@app.post("/signals")
async def get_signals(request: SignalRequest):
    """Get trading signals based on specified criteria"""
    try:
        signals = signal_generator.generate_signals(
            timeframes=request.timeframes,
            pairs=request.pairs
        )
        
        # Filter by minimum strength
        if request.min_strength:
            signals = [s for s in signals if s['signal_strength'] >= request.min_strength]
            
        return {
            "timestamp": datetime.now().isoformat(),
            "signals": signals
        }
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best_signals")
async def get_best_signals(min_strength: float = 3.0):
    """Get the strongest current trading signals"""
    try:
        signals = signal_generator.get_best_signals(min_strength)
        return {
            "timestamp": datetime.now().isoformat(),
            "signals": signals
        }
    except Exception as e:
        logger.error(f"Error getting best signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_results")
async def update_results(results: List[TradingResult], background_tasks: BackgroundTasks):
    """Update models with actual trading results"""
    try:
        # Update models in the background
        background_tasks.add_task(signal_generator.update_models, [r.dict() for r in results])
        return {"status": "success", "message": "Model update scheduled"}
    except Exception as e:
        logger.error(f"Error updating results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get signal generation statistics"""
    try:
        stats = signal_generator.get_signal_statistics()
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_models")
async def save_models():
    """Save current model state"""
    try:
        signal_generator.save_models()
        return {"status": "success", "message": "Models saved successfully"}
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_models")
async def load_models():
    """Load saved model state"""
    try:
        signal_generator.load_models()
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
