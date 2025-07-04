# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 4. SEQUENCE GENERATOR MODULE
# =============================================================================

class SequenceGenerator:
    """Generates training sequences for LSTM with coin identity encoding"""
    
    def __init__(self, lookback: int = 7):
        self.lookback = lookback
        
    def generate_sequences_single_coin(self, scaled_data: np.ndarray, coin_encoding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sequences for a single coin"""
        X, y = [], []
        
        for i in range(len(scaled_data) - self.lookback):
            # Price sequence
            price_sequence = scaled_data[i:(i + self.lookback), 0]
            
            # Add coin encoding to each timestep
            sequence_with_coin = []
            for j in range(self.lookback):
                # Combine price with coin encoding
                timestep = np.concatenate([
                    [price_sequence[j]],  # Price
                    coin_encoding  # Coin one-hot encoding
                ])
                sequence_with_coin.append(timestep)
            
            X.append(sequence_with_coin)
            y.append(scaled_data[i + self.lookback, 0])
        
        return np.array(X), np.array(y)
    
    def generate_all_sequences(self, coin_data: Dict[str, Dict], coin_encodings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate sequences for all coins and combine them"""
        all_X, all_y = [], []
        sequence_info = {}
        
        for coin in coin_data.keys():
            if coin not in coin_encodings:
                print(f"Warning: No encoding for {coin}. Skipping...")
                continue
                
            # Generate sequences for training data
            coin_X, coin_y = self.generate_sequences_single_coin(
                coin_data[coin]['scaled_train'], 
                coin_encodings[coin]
            )
            
            if len(coin_X) > 0:
                all_X.append(coin_X)
                all_y.append(coin_y)
                sequence_info[coin] = {
                    'num_sequences': len(coin_X),
                    'start_idx': len(all_y) - 1 if len(all_y) > 1 else 0
                }
                
                print(f"{coin}: Generated {len(coin_X)} sequences")
        
        if all_X:
            combined_X = np.concatenate(all_X, axis=0)
            combined_y = np.concatenate(all_y, axis=0)
            
            print(f"Total sequences: {len(combined_X)}")
            print(f"Sequence shape: {combined_X.shape}")
            
            return combined_X, combined_y, sequence_info
        else:
            raise ValueError("No sequences generated")


