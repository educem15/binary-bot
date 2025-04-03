import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data
        data should have columns: ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators and return enhanced DataFrame"""
        df = self.data.copy()
        
        # Trend Indicators
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Bollinger Bands
        bollinger = ta.bbands(df['close'], length=20)
        df = pd.concat([df, bollinger], axis=1)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df = pd.concat([df, stoch], axis=1)
        
        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'])
        df = pd.concat([df, adx], axis=1)
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        df = pd.concat([df, ichimoku], axis=1)
        
        # Volume Indicators
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Momentum Indicators
        df['mom'] = ta.mom(df['close'])
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        return df

    def generate_signals(self) -> Dict[str, float]:
        """Generate trading signals and their strength (1-5 scale)"""
        df = self.calculate_all_indicators()
        signals = {}
        
        # Trend Following Signals
        signals['trend_strength'] = self._calculate_trend_strength(df)
        
        # Momentum Signals
        signals['momentum_strength'] = self._calculate_momentum_strength(df)
        
        # Volatility Signals
        signals['volatility_strength'] = self._calculate_volatility_strength(df)
        
        # Volume Signals
        signals['volume_strength'] = self._calculate_volume_strength(df)
        
        # Combined Signal
        signals['overall_strength'] = self._calculate_overall_strength(signals)
        
        return signals

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength on a scale of 1-5"""
        strength = 1.0
        
        # SMA Crossover
        if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            strength += 1
            
        # MACD Signal
        if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1]:
            strength += 1
            
        # ADX Strength
        adx_value = df['ADX_14'].iloc[-1]
        if adx_value > 25:
            strength += 1
        if adx_value > 40:
            strength += 1
            
        return min(strength, 5.0)

    def _calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """Calculate momentum strength on a scale of 1-5"""
        strength = 1.0
        
        # RSI
        rsi = df['RSI_14'].iloc[-1]
        if 30 <= rsi <= 70:
            strength += 1
        
        # Stochastic
        if df['STOCHk_14_3_3'].iloc[-1] > df['STOCHd_14_3_3'].iloc[-1]:
            strength += 1
            
        # CCI
        cci = df['CCI_14_0.015'].iloc[-1]
        if -100 <= cci <= 100:
            strength += 1
            
        # Momentum
        if df['MOM_10'].iloc[-1] > 0:
            strength += 1
            
        return min(strength, 5.0)

    def _calculate_volatility_strength(self, df: pd.DataFrame) -> float:
        """Calculate volatility-based strength on a scale of 1-5"""
        strength = 1.0
        
        # Bollinger Bands
        bb_width = (df['BBU_20_2.0'].iloc[-1] - df['BBL_20_2.0'].iloc[-1]) / df['BBM_20_2.0'].iloc[-1]
        
        if bb_width < 0.1:
            strength += 2  # Low volatility
        elif bb_width < 0.2:
            strength += 1
            
        # Price position relative to Bollinger Bands
        if df['BBL_20_2.0'].iloc[-1] <= df['close'].iloc[-1] <= df['BBU_20_2.0'].iloc[-1]:
            strength += 1
            
        return min(strength, 5.0)

    def _calculate_volume_strength(self, df: pd.DataFrame) -> float:
        """Calculate volume-based strength on a scale of 1-5"""
        strength = 1.0
        
        # Volume trend
        vol_sma = df['volume'].rolling(20).mean()
        if df['volume'].iloc[-1] > vol_sma.iloc[-1]:
            strength += 1
            
        # OBV trend
        if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]:
            strength += 1
            
        # Volume consistency
        vol_std = df['volume'].rolling(20).std()
        if vol_std.iloc[-1] < vol_std.rolling(5).mean().iloc[-1]:
            strength += 1
            
        return min(strength, 5.0)

    def _calculate_overall_strength(self, signals: Dict[str, float]) -> float:
        """Calculate overall signal strength on a scale of 1-5"""
        weights = {
            'trend_strength': 0.35,
            'momentum_strength': 0.25,
            'volatility_strength': 0.20,
            'volume_strength': 0.20
        }
        
        overall = sum(signals[k] * weights[k] for k in weights.keys())
        return min(max(overall, 1.0), 5.0)

    def get_entry_point(self) -> Tuple[float, str]:
        """
        Calculate the optimal entry point based on current indicators
        Returns: (price_level, signal_type ['CALL', 'PUT'])
        """
        df = self.calculate_all_indicators()
        current_price = df['close'].iloc[-1]
        
        # Analyze multiple factors for entry
        trend = 'CALL' if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1] else 'PUT'
        
        # Calculate support/resistance levels
        support = df['BBL_20_2.0'].iloc[-1]
        resistance = df['BBU_20_2.0'].iloc[-1]
        
        # Determine entry price based on trend and S/R levels
        if trend == 'CALL':
            entry_price = support + (resistance - support) * 0.3
        else:
            entry_price = resistance - (resistance - support) * 0.3
            
        return entry_price, trend
