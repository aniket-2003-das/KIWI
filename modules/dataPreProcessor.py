# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler



# =============================================================================
# 3. PREPROCESSING MODULE
# =============================================================================

class MultiCoinPreprocessor:
    """Handles preprocessing for multiple cryptocurrencies"""
    
    def __init__(self, prediction_days: int = 180):
        self.prediction_days = prediction_days
        self.scalers = {}
        self.coin_encoders = {}
        self.coins = []
        
    def prepare_coin_data(self, combined_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Prepare training and test data for each coin
        
        Returns:
            Dict with coin-specific train/test splits and scalers
        """
        self.coins = combined_data['Coin'].unique()
        coin_data = {}
        
        for coin in self.coins:
            coin_df = combined_data[combined_data['Coin'] == coin].copy()
            coin_df = coin_df.sort_values('Date')
            
            # Extract close prices
            prices = coin_df['Close'].values.reshape(-1, 1)
            
            # Split into train and test
            train_size = len(prices) - self.prediction_days
            if train_size <= 0:
                print(f"Warning: Not enough data for {coin}. Skipping...")
                continue
                
            train_data = prices[:train_size]
            test_data = prices[train_size:]
            
            # Create and fit scaler for this coin
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train = scaler.fit_transform(train_data)
            scaled_test = scaler.transform(test_data)
            
            # Store scaler for later use
            self.scalers[coin] = scaler
            
            coin_data[coin] = {
                'train_data': train_data,
                'test_data': test_data,
                'scaled_train': scaled_train,
                'scaled_test': scaled_test,
                'dates': coin_df['Date'].values
            }
            
            print(f"{coin}: Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        return coin_data
    
    def create_coin_encoding(self, coins: List[str]) -> Dict[str, np.ndarray]:
        """Create one-hot encodings for coins"""
        self.coins = coins
        coin_encodings = {}
        
        for i, coin in enumerate(coins):
            encoding = np.zeros(len(coins))
            encoding[i] = 1
            coin_encodings[coin] = encoding
            
        self.coin_encoders = coin_encodings
        return coin_encodings
    
    def save_preprocessor(self, filepath: str):
        """Save scalers and encoders"""
        joblib.dump({
            'scalers': self.scalers,
            'coin_encoders': self.coin_encoders,
            'coins': self.coins,
            'prediction_days': self.prediction_days
        }, filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load scalers and encoders"""
        data = joblib.load(filepath)
        self.scalers = data['scalers']
        self.coin_encoders = data['coin_encoders']
        self.coins = data['coins']
        self.prediction_days = data['prediction_days']

