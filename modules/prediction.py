# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from modules.dataPreProcessor import MultiCoinPreprocessor
from modules.sequenceGenerator import SequenceGenerator
from modules.modelBuilding import MultiCoinLSTM

# =============================================================================
# 7. PREDICTOR MODULE
# =============================================================================

class MultiCoinPredictor:
    """Handles predictions for multiple coins"""
    
    def __init__(self, model: MultiCoinLSTM, preprocessor: MultiCoinPreprocessor, 
                 sequence_generator: SequenceGenerator):
        self.model = model
        self.preprocessor = preprocessor
        self.sequence_generator = sequence_generator
        
    def predict_single_coin(self, coin_name: str, coin_data: Dict, 
                           num_future_days: int = 7) -> Dict:
        """Make predictions for a single coin"""
        
        if coin_name not in self.preprocessor.scalers:
            raise ValueError(f"No scaler found for {coin_name}")
        
        scaler = self.preprocessor.scalers[coin_name]
        coin_encoding = self.preprocessor.coin_encoders[coin_name]
        
        # Prepare test sequences
        test_X, test_y = self.sequence_generator.generate_sequences_single_coin(
            coin_data['scaled_test'], coin_encoding
        )
        
        # Make predictions on test data
        if len(test_X) > 0:
            test_predictions = self.model.model.predict(test_X)
            test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
            test_actual = scaler.inverse_transform(test_y.reshape(-1, 1))
        else:
            test_predictions = np.array([])
            test_actual = np.array([])
        
        # Future predictions
        future_predictions = []
        if num_future_days > 0:
            # Use last available data for future predictions
            last_sequence = coin_data['scaled_test'][-self.sequence_generator.lookback:]
            
            for _ in range(num_future_days):
                # Prepare sequence with coin encoding
                sequence_with_coin = []
                for j in range(self.sequence_generator.lookback):
                    timestep = np.concatenate([
                        [last_sequence[j, 0]],
                        coin_encoding
                    ])
                    sequence_with_coin.append(timestep)
                
                # Predict next value
                next_pred = self.model.model.predict(
                    np.array([sequence_with_coin]), verbose=0
                )
                
                # Inverse transform
                next_pred_price = scaler.inverse_transform(next_pred.reshape(-1, 1))
                future_predictions.append(next_pred_price[0, 0])
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1, 0] = next_pred[0, 0]
        
        return {
            'coin': coin_name,
            'test_predictions': test_predictions.flatten(),
            'test_actual': test_actual.flatten(),
            'future_predictions': np.array(future_predictions),
            'scaler': scaler
        }
    
    def predict_all_coins(self, coin_data: Dict, num_future_days: int = 7) -> Dict:
        """Make predictions for all coins"""
        all_predictions = {}
        
        for coin_name in coin_data.keys():
            if coin_name in self.preprocessor.scalers:
                try:
                    predictions = self.predict_single_coin(
                        coin_name, coin_data[coin_name], num_future_days
                    )
                    all_predictions[coin_name] = predictions
                    print(f"Predictions completed for {coin_name}")
                except Exception as e:
                    print(f"Error predicting {coin_name}: {e}")
            else:
                print(f"No scaler available for {coin_name}")
        
        return all_predictions
    