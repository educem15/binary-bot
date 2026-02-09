from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import logging
import json
import os

from app.options.screener import OptionsScreener
from app.options.analyzer import OptionsAnalyzer
from app.options.report_generator import DailyOptionsReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to the project root (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(BASE_DIR, "app", "frontend")

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
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Lazy-load SignalGenerator (requires heavy ML dependencies)
signal_generator = None

def _get_signal_generator():
    global signal_generator
    if signal_generator is None:
        from app.signal_generator import SignalGenerator
        signal_generator = SignalGenerator()
    return signal_generator

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
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/options")
async def options_dashboard():
    """Serve the options research dashboard"""
    return FileResponse(os.path.join(FRONTEND_DIR, "options.html"))

@app.post("/signals")
async def get_signals(request: SignalRequest):
    """Get trading signals based on specified criteria"""
    try:
        signals = _get_signal_generator().generate_signals(
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
        signals = _get_signal_generator().get_best_signals(min_strength)
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
        background_tasks.add_task(_get_signal_generator().update_models, [r.dict() for r in results])
        return {"status": "success", "message": "Model update scheduled"}
    except Exception as e:
        logger.error(f"Error updating results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get signal generation statistics"""
    try:
        stats = _get_signal_generator().get_signal_statistics()
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
        _get_signal_generator().save_models()
        return {"status": "success", "message": "Models saved successfully"}
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_models")
async def load_models():
    """Load saved model state"""
    try:
        _get_signal_generator().load_models()
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Options Research Endpoints ─────────────────────────────────────

class OptionsScreenRequest(BaseModel):
    budget: Optional[float] = 500.0
    watchlist: Optional[List[str]] = None
    top_n: Optional[int] = 10

class OptionsAnalyzeRequest(BaseModel):
    ticker: str
    budget: Optional[float] = 500.0
    expiration: Optional[str] = None

class DailyReportRequest(BaseModel):
    budget: Optional[float] = 500.0
    watchlist: Optional[List[str]] = None
    top_n: Optional[int] = 10
    deep_analysis_n: Optional[int] = 5
    expiration: Optional[str] = None

@app.post("/options/scan")
async def scan_options(request: OptionsScreenRequest):
    """
    Screen the market for high-potential options trades.
    Returns ranked tickers with composite scores, volatility, momentum, and IV data.
    """
    try:
        screener = OptionsScreener(
            budget=request.budget,
            watchlist=request.watchlist,
        )
        results = screener.get_top_candidates(n=request.top_n)
        return {
            "timestamp": datetime.now().isoformat(),
            "budget": request.budget,
            "candidates": results,
        }
    except Exception as e:
        logger.error(f"Error scanning options: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/options/analyze")
async def analyze_options(request: OptionsAnalyzeRequest):
    """
    Deep-analyze a specific ticker's options chain.
    Returns Greeks, probability of profit, and strategy recommendations.
    """
    try:
        analyzer = OptionsAnalyzer(budget=request.budget)
        analysis = analyzer.analyze_ticker(
            ticker=request.ticker,
            expiration=request.expiration,
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
        }
    except Exception as e:
        logger.error(f"Error analyzing {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/options/daily-report")
async def daily_report(request: DailyReportRequest):
    """
    Generate the full daily options research report (JSON).
    Screens the watchlist, deep-analyzes top picks, and returns ranked strategies.
    """
    try:
        report_gen = DailyOptionsReport(
            budget=request.budget,
            watchlist=request.watchlist,
        )
        report = report_gen.generate_report(
            top_n=request.top_n,
            deep_analysis_n=request.deep_analysis_n,
            expiration=request.expiration,
        )
        return report
    except Exception as e:
        logger.error(f"Error generating daily report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/options/daily-report/text")
async def daily_report_text(request: DailyReportRequest):
    """
    Generate the full daily options research report (human-readable text).
    Perfect for reading in a terminal or copying to notes.
    """
    try:
        report_gen = DailyOptionsReport(
            budget=request.budget,
            watchlist=request.watchlist,
        )
        text = report_gen.generate_text_report(
            top_n=request.top_n,
            deep_analysis_n=request.deep_analysis_n,
        )
        return PlainTextResponse(content=text)
    except Exception as e:
        logger.error(f"Error generating text report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options/quick-scan/{ticker}")
async def quick_scan_ticker(ticker: str, budget: float = 500.0):
    """
    Quick scan a single ticker for options potential.
    Use this for ad-hoc checks on specific stocks.
    """
    try:
        screener = OptionsScreener(budget=budget)
        result = screener.scan_ticker(ticker.upper())
        if result is None:
            raise HTTPException(status_code=404, detail=f"No qualifying options found for {ticker}")
        return {
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
