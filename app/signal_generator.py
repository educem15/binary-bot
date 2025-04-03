import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
import os

from app.models.ml_model import TradingModel
from app.strategies.technical_indicators import TechnicalIndicators
from app.utils.data_fetcher import DataFetcher

class SignalGenerator:
    def __init__(self, config_path: str = 'config/.env'):
        """Initialize the signal generator with configuration"""
        self.config = self._load_config(config_path)
        self.data_fetcher = DataFetcher()
        self.trading_model = TradingModel(self.config)
        self.signals_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from environment file"""
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=')
                    try:
                        config[key] = json.loads(value)
                    except json.JSONDecodeError:
                        config[key] = value
        return config
        
    def generate_signals(self, 
                        timeframes: Optional[List[str]] = None,
                        pairs: Optional[List[str]] = None) -> List[Dict]:
        """Generate trading signals for specified timeframes and pairs"""
        if timeframes is None:
            timeframes = self.config['DEFAULT_TIMEFRAMES']
        if pairs is None:
            pairs = self.config['CURRENCY_PAIRS']
            
        all_signals = []
        
        for pair in pairs:
            for timeframe in timeframes:
                try:
                    # Get historical data
                    df = self.data_fetcher.get_combined_data(pair, timeframe)
                    
                    if df is None or df.empty:
                        logging.warning(f"No data available for {pair} on {timeframe}")
                        continue
                        
                    # Calculate technical indicators
                    tech_indicators = TechnicalIndicators(df)
                    tech_signals = tech_indicators.generate_signals()
                    
                    # Get ML predictions
                    ml_predictions = self.trading_model.predict(df)
                    
                    # Calculate entry point
                    entry_price, direction = tech_indicators.get_entry_point()
                    
                    # Combine signals
                    signal_strength = self._combine_signals(
                        tech_signals['overall_strength'],
                        ml_predictions['signal_strength']
                    )
                    
                    # Create signal object
                    signal = {
                        'pair': pair,
                        'timeframe': timeframe,
                        'timestamp': datetime.now().isoformat(),
                        'direction': direction,
                        'entry_price': entry_price,
                        'signal_strength': signal_strength,
                        'technical_signals': tech_signals,
                        'ml_predictions': ml_predictions,
                        'current_price': df['close'].iloc[-1]
                    }
                    
                    all_signals.append(signal)
                    self.signals_history.append(signal)
                    
                except Exception as e:
                    logging.error(f"Error generating signal for {pair} {timeframe}: {str(e)}")
                    continue
                    
        # Sort signals by strength
        all_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        return all_signals
        
    def _combine_signals(self, tech_strength: float, ml_strength: float) -> float:
        """Combine technical and ML signal strengths"""
        # Weight technical analysis more heavily (60/40 split)
        combined_strength = (0.6 * tech_strength) + (0.4 * ml_strength)
        return round(combined_strength, 2)
        
    def update_models(self, results: List[Dict]):
        """Update models with actual trading results"""
        if not results:
            return
            
        # Prepare results DataFrame
        results_df = pd.DataFrame(results)
        
        # Get historical data for the period
        start_time = pd.to_datetime(results_df['timestamp'].min())
        end_time = pd.to_datetime(results_df['timestamp'].max())
        
        for pair in results_df['pair'].unique():
            try:
                # Get data for the period
                df = self.data_fetcher.get_combined_data(pair)
                df = df[start_time:end_time]
                
                if df is None or df.empty:
                    continue
                    
                # Update ML model
                pair_results = results_df[results_df['pair'] == pair]
                self.trading_model.update_model(df, pair_results)
                
            except Exception as e:
                logging.error(f"Error updating models for {pair}: {str(e)}")
                continue
                
    def get_best_signals(self, min_strength: float = 3.0) -> List[Dict]:
        """Get the strongest current signals"""
        signals = self.generate_signals()
        return [s for s in signals if s['signal_strength'] >= min_strength]
        
    def save_models(self, path: str = 'models/'):
        """Save ML models to disk"""
        os.makedirs(path, exist_ok=True)
        self.trading_model.save_models(path)
        
    def load_models(self, path: str = 'models/'):
        """Load ML models from disk"""
        self.trading_model.load_models(path)
        
    def get_signal_statistics(self) -> Dict:
        """Get statistics about signal performance"""
        if not self.signals_history:
            return {}
            
        df = pd.DataFrame(self.signals_history)
        
        stats = {
            'total_signals': len(df),
            'average_strength': df['signal_strength'].mean(),
            'signals_by_pair': df['pair'].value_counts().to_dict(),
            'signals_by_timeframe': df['timeframe'].value_counts().to_dict(),
            'average_strength_by_pair': df.groupby('pair')['signal_strength'].mean().to_dict(),
            'signals_last_24h': len(df[df['timestamp'] >= (datetime.now() - timedelta(days=1)).isoformat()])
        }
        
        return stats
