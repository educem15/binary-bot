import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from typing import Dict, List, Tuple
import logging

class TradingModel:
    def __init__(self, config: Dict):
        self.config = config
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = self._build_lstm_model()
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def _build_lstm_model(self) -> Sequential:
        """Build and compile LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 30)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model training"""
        feature_cols = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'MACD_12_26_9', 'MACDs_12_26_9', 'rsi',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
            'STOCHk_14_3_3', 'STOCHd_14_3_3',
            'ADX_14', 'obv', 'mom', 'cci'
        ]
        
        self.feature_columns = feature_cols
        features = df[feature_cols].fillna(method='ffill')
        return self.scaler.fit_transform(features)
        
    def prepare_targets(self, df: pd.DataFrame, timeframe: int) -> np.ndarray:
        """Prepare target variables based on future price movement"""
        future_price = df['close'].shift(-timeframe)
        targets = (future_price > df['close']).astype(int)
        return targets[:-timeframe]  # Remove last rows where we don't have future data
        
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
        
    def train(self, df: pd.DataFrame, timeframe: int):
        """Train both RF and LSTM models"""
        # Prepare data
        features = self.prepare_features(df)
        targets = self.prepare_targets(df, timeframe)
        
        # Remove any remaining NaN values
        valid_idx = ~np.isnan(targets)
        features = features[valid_idx]
        targets = targets[valid_idx]
        
        # Train Random Forest
        self.rf_model.fit(features, targets)
        
        # Prepare sequences for LSTM
        seq_length = 60
        X_lstm, y_lstm = self.create_sequences(features, seq_length)
        
        # Train LSTM
        self.lstm_model.fit(
            X_lstm, y_lstm,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions using both models"""
        features = self.prepare_features(df)
        
        # RF prediction
        rf_prob = self.rf_model.predict_proba(features[-1:])[:, 1][0]
        
        # LSTM prediction
        seq_length = 60
        lstm_input = features[-seq_length:].reshape(1, seq_length, -1)
        lstm_prob = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
        
        # Ensemble prediction (weighted average)
        ensemble_prob = 0.6 * rf_prob + 0.4 * lstm_prob
        
        return {
            'rf_probability': float(rf_prob),
            'lstm_probability': float(lstm_prob),
            'ensemble_probability': float(ensemble_prob),
            'signal_strength': self._calculate_signal_strength(ensemble_prob)
        }
        
    def _calculate_signal_strength(self, probability: float) -> float:
        """Convert probability to signal strength (1-5 scale)"""
        # Center around 0.5
        deviation = abs(probability - 0.5)
        
        # Scale to 1-5
        strength = 1 + (deviation * 8)  # 0.5 deviation = max strength
        return min(max(strength, 1.0), 5.0)
        
    def update_model(self, df: pd.DataFrame, results: pd.DataFrame):
        """Update model with new results"""
        features = self.prepare_features(df)
        
        # Prepare actual results
        targets = results['success'].astype(int).values
        
        # Update Random Forest (online learning)
        self.rf_model.fit(features, targets)
        
        # Update LSTM
        seq_length = 60
        X_lstm, y_lstm = self.create_sequences(features, seq_length)
        
        self.lstm_model.fit(
            X_lstm, y_lstm,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
    def save_models(self, path: str):
        """Save models to disk"""
        # Save Random Forest
        joblib.dump(self.rf_model, f"{path}/rf_model.joblib")
        
        # Save LSTM
        self.lstm_model.save(f"{path}/lstm_model")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        
    def load_models(self, path: str):
        """Load models from disk"""
        try:
            self.rf_model = joblib.load(f"{path}/rf_model.joblib")
            self.lstm_model = tf.keras.models.load_model(f"{path}/lstm_model")
            self.scaler = joblib.load(f"{path}/scaler.joblib")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise
