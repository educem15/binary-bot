import requests
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.fixer_api_key = os.getenv('FIXER_API_KEY')
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
    def get_alpha_vantage_data(self, symbol: str, interval: str = '1min') -> pd.DataFrame:
        """Fetch data from Alpha Vantage API"""
        try:
            base_url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': symbol.split('/')[0],
                'to_symbol': symbol.split('/')[1],
                'interval': interval,
                'apikey': self.alpha_vantage_api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if 'Time Series FX' not in data:
                raise ValueError(f"Invalid response from Alpha Vantage: {data}")
                
            df = pd.DataFrame.from_dict(data['Time Series FX (1min)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching Alpha Vantage data: {str(e)}")
            return None
            
    def get_finnhub_data(self, symbol: str, resolution: str = '1') -> pd.DataFrame:
        """Fetch data from Finnhub API"""
        try:
            base_url = 'https://finnhub.io/api/v1/forex/candle'
            
            # Calculate time range (last 24 hours)
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=1)).timestamp())
            
            params = {
                'symbol': symbol.replace('/', ''),
                'resolution': resolution,
                'from': start_time,
                'to': end_time,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if data['s'] != 'ok':
                raise ValueError(f"Invalid response from Finnhub: {data}")
                
            df = pd.DataFrame({
                'timestamp': data['t'],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching Finnhub data: {str(e)}")
            return None
            
    def get_fixer_data(self, symbol: str) -> float:
        """Fetch current exchange rate from Fixer API"""
        try:
            base_url = 'http://data.fixer.io/api/latest'
            base_currency = symbol.split('/')[0]
            target_currency = symbol.split('/')[1]
            
            params = {
                'access_key': self.fixer_api_key,
                'base': base_currency,
                'symbols': target_currency
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data['success']:
                raise ValueError(f"Invalid response from Fixer: {data}")
                
            return data['rates'][target_currency]
            
        except Exception as e:
            logging.error(f"Error fetching Fixer data: {str(e)}")
            return None
            
    def get_combined_data(self, symbol: str, interval: str = '1min') -> pd.DataFrame:
        """Get data from multiple sources and combine them"""
        dfs = []
        
        # Get Alpha Vantage data
        av_data = self.get_alpha_vantage_data(symbol, interval)
        if av_data is not None:
            av_data['source'] = 'alpha_vantage'
            dfs.append(av_data)
            
        # Get Finnhub data
        fh_data = self.get_finnhub_data(symbol, '1' if interval == '1min' else '5')
        if fh_data is not None:
            fh_data['source'] = 'finnhub'
            dfs.append(fh_data)
            
        if not dfs:
            raise ValueError("No data available from any source")
            
        # Combine and clean data
        combined_df = pd.concat(dfs)
        combined_df = combined_df.sort_index()
        
        # Remove duplicates and handle conflicts
        combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
        
        # Resample to ensure consistent intervals
        interval_map = {'1min': '1T', '5min': '5T'}
        combined_df = combined_df.resample(interval_map[interval]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).fillna(method='ffill')
        
        return combined_df
        
    def get_latest_price(self, symbol: str) -> Dict[str, float]:
        """Get latest price from all available sources"""
        prices = {}
        
        # Try Fixer
        fixer_price = self.get_fixer_data(symbol)
        if fixer_price is not None:
            prices['fixer'] = fixer_price
            
        # Try Alpha Vantage
        av_data = self.get_alpha_vantage_data(symbol)
        if av_data is not None and not av_data.empty:
            prices['alpha_vantage'] = av_data['close'].iloc[-1]
            
        # Try Finnhub
        fh_data = self.get_finnhub_data(symbol)
        if fh_data is not None and not fh_data.empty:
            prices['finnhub'] = fh_data['close'].iloc[-1]
            
        if not prices:
            raise ValueError("No price data available from any source")
            
        # Calculate consensus price (weighted average)
        weights = {'fixer': 0.3, 'alpha_vantage': 0.4, 'finnhub': 0.3}
        available_weights = {k: weights[k] for k in prices.keys()}
        weight_sum = sum(available_weights.values())
        
        consensus_price = sum(prices[k] * available_weights[k] / weight_sum 
                            for k in prices.keys())
                            
        prices['consensus'] = consensus_price
        return prices
