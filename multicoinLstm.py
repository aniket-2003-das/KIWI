# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping




# =============================================================================
# 5. TRAINER MODULE
# =============================================================================

class MultiCoinTrainer:
    """Handles training of the multi-coin LSTM model"""
    
    def __init__(self, model: MultiCoinLSTM, model_save_path: str = "models/"):
        self.model = model
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        self.history = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              patience: int = 10, verbose: int = 1):
        """Train the model"""
        
        # Setup callbacks
        checkpoint_path = self.model_save_path / "best_model.keras"
        
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        callbacks = [checkpoint, early_stopping]
        
        # Train model
        self.history = self.model.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print("No training history available")
    
    def load_best_model(self):
        """Load the best saved model"""
        checkpoint_path = self.model_save_path / "best_model.keras"
        if checkpoint_path.exists():
            self.model.model = load_model(str(checkpoint_path))
            print("Best model loaded successfully")
        else:
            print("No saved model found")

# =============================================================================
# 6. PREDICTOR MODULE
# =============================================================================

class MultiCoinPredictor:
    """Handles predictions for multiple coins"""
    
    def __init__(self, model: MultiCoinLSTM, preprocessor: MultiCoinPreprocessor, 
                 sequence_generator: SequenceGenerator):
        self.model = model
        self.preprocessor = preprocessor
        self.sequence_generator = sequence_generator
        
    def predict_single_coin(self, coin_name: str, coin_data: Dict, 
                           num_future_days: int = 14) -> Dict:
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
    
    def predict_all_coins(self, coin_data: Dict, num_future_days: int = 14) -> Dict:
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

# =============================================================================
# 7. EVALUATOR MODULE
# =============================================================================

class MultiCoinEvaluator:
    """Evaluates and visualizes model performance"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, predictions: Dict) -> Dict:
        """Calculate performance metrics for all coins"""
        metrics = {}
        
        for coin_name, pred_data in predictions.items():
            test_actual = pred_data['test_actual']
            test_pred = pred_data['test_predictions']
            
            if len(test_actual) > 0 and len(test_pred) > 0:
                # Calculate metrics
                rmse = math.sqrt(mean_squared_error(test_actual, test_pred))
                r2 = r2_score(test_actual, test_pred)
                
                # Calculate daily returns
                daily_returns_actual = np.diff(test_actual) / test_actual[:-1] * 100
                daily_returns_pred = np.diff(test_pred) / test_pred[:-1] * 100
                
                metrics[coin_name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'mean_actual_return': np.mean(daily_returns_actual),
                    'mean_predicted_return': np.mean(daily_returns_pred),
                    'std_actual_return': np.std(daily_returns_actual),
                    'std_predicted_return': np.std(daily_returns_pred)
                }
        
        self.metrics = metrics
        return metrics
    
    def print_metrics_summary(self):
        """Print metrics summary"""
        if not self.metrics:
            print("No metrics calculated yet")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        for coin, metrics in self.metrics.items():
            print(f"\n{coin}:")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R²: {metrics['r2']:.3f}")
            print(f"  Mean Actual Return: {metrics['mean_actual_return']:.2f}%")
            print(f"  Mean Predicted Return: {metrics['mean_predicted_return']:.2f}%")
    
    def plot_predictions(self, predictions: Dict, figsize: Tuple[int, int] = (15, 10)):
        """Plot predictions for all coins"""
        num_coins = len(predictions)
        if num_coins == 0:
            print("No predictions to plot")
            return
        
        # Calculate grid dimensions
        cols = min(3, num_coins)
        rows = (num_coins + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (coin_name, pred_data) in enumerate(predictions.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Plot test predictions
            if len(pred_data['test_actual']) > 0:
                ax.plot(pred_data['test_actual'], label='Actual', marker='o', markersize=2)
                ax.plot(pred_data['test_predictions'], label='Predicted', marker='s', markersize=2)
            
            # Plot future predictions
            if len(pred_data['future_predictions']) > 0:
                future_start = len(pred_data['test_actual'])
                future_x = range(future_start, future_start + len(pred_data['future_predictions']))
                ax.plot(future_x, pred_data['future_predictions'], 
                       label='Future Forecast', marker='^', markersize=2, linestyle='--')
            
            ax.set_title(f'{coin_name} Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_coins, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# 8. MAIN PIPELINE CLASS
# =============================================================================

class MultiCoinLSTMPipeline:
    """Main pipeline orchestrating the entire multi-coin LSTM workflow"""
    
    def __init__(self, lookback: int = 14, prediction_days: int = 60):
        self.lookback = lookback
        self.prediction_days = prediction_days
        
        # Initialize components
        self.data_loader = MultiCoinDataLoader()
        self.preprocessor = MultiCoinPreprocessor(prediction_days)
        self.sequence_generator = SequenceGenerator(lookback)
        self.model = None
        self.trainer = None
        self.predictor = None
        self.evaluator = MultiCoinEvaluator()
        
        # Data storage
        self.combined_data = None
        self.coin_data = None
        self.coin_encodings = None
        self.predictions = None
        
    def load_data(self, coin_configs: Dict[str, str]):
        """Load data for multiple coins"""
        print("Loading data...")
        self.combined_data = self.data_loader.load_multiple_coins(coin_configs)
        
    def preprocess_data(self):
        """Preprocess data for all coins"""
        print("Preprocessing data...")
        self.coin_data = self.preprocessor.prepare_coin_data(self.combined_data)
        self.coin_encodings = self.preprocessor.create_coin_encoding(list(self.coin_data.keys()))
        
    def prepare_sequences(self):
        """Generate training sequences"""
        print("Generating sequences...")
        self.X_train, self.y_train, self.sequence_info = self.sequence_generator.generate_all_sequences(
            self.coin_data, self.coin_encodings
        )
        
        # For simplicity, we'll use a portion of training data as validation
        # In production, you might want to use separate validation coins
        val_split = 0.2
        split_idx = int(len(self.X_train) * (1 - val_split))
        
        self.X_val = self.X_train[split_idx:]
        self.y_val = self.y_train[split_idx:]
        self.X_train = self.X_train[:split_idx]
        self.y_train = self.y_train[:split_idx]
        
        print(f"Training sequences: {len(self.X_train)}")
        print(f"Validation sequences: {len(self.X_val)}")
        
    def build_model(self, lstm_units: List[int] = [128, 64], dropout_rate: float = 0.2):
        """Build the LSTM model"""
        print("Building model...")
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.model = MultiCoinLSTM(input_shape)
        self.model.build_model(lstm_units, dropout_rate)
        self.model.get_model_summary()
        
    def train_model(self, epochs: int = 100, batch_size: int = 32, patience: int = 10):
        """Train the model"""
        print("Training model...")
        self.trainer = MultiCoinTrainer(self.model)
        self.trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=epochs, batch_size=batch_size, patience=patience
        )
        
        # Load best model
        self.trainer.load_best_model()
        
    def make_predictions(self, num_future_days: int = 14):
        """Make predictions for all coins"""
        print("Making predictions...")
        self.predictor = MultiCoinPredictor(self.model, self.preprocessor, self.sequence_generator)
        self.predictions = self.predictor.predict_all_coins(self.coin_data, num_future_days)
        
    def evaluate_results(self):
        """Evaluate and visualize results"""
        print("Evaluating results...")
        metrics = self.evaluator.calculate_metrics(self.predictions)
        self.evaluator.print_metrics_summary()
        self.evaluator.plot_predictions(self.predictions)
        
        return metrics
    
    def run_complete_pipeline(self, coin_configs: Dict[str, str], 
                            epochs: int = 100, num_future_days: int = 14):
        """Run the complete pipeline"""
        print("Starting Multi-Coin LSTM Pipeline...")
        
        # Execute pipeline steps
        self.load_data(coin_configs)
        self.preprocess_data()
        self.prepare_sequences()
        self.build_model()
        self.train_model(epochs=epochs)
        self.make_predictions(num_future_days)
        metrics = self.evaluate_results()
        
        print("Pipeline completed successfully!")
        return metrics
    
    def save_pipeline(self, save_dir: str = "saved_pipeline/"):
        """Save the complete pipeline"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor(save_path / "preprocessor.pkl")
        
        # Save model
        if self.model and self.model.model:
            self.model.model.save(save_path / "model.keras")
        
        print(f"Pipeline saved to {save_path}")

# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the multi-coin LSTM pipeline"""
    
    # Define coin configurations
    coin_configs = {
        'BTC': 'BTC_USD.csv',
        'ETH': 'ETH_USD.csv',
        'ADA': 'ADA_USD.csv',
        'DOGE': 'DOGE_USD.csv'
    }
    
    # Initialize pipeline
    pipeline = MultiCoinLSTMPipeline(
        lookback=14,
        prediction_days=60
    )
    
    # Run complete pipeline
    metrics = pipeline.run_complete_pipeline(
        coin_configs=coin_configs,
        epochs=50,  # Reduced for example
        num_future_days=14
    )
    
    # Save pipeline
    pipeline.save_pipeline()
    
    return pipeline, metrics

# =============================================================================
# 10. UTILITY FUNCTIONS
# =============================================================================

def create_sample_data(coins: List[str], days: int = 1000, save_dir: str = "sample_data/"):
    """Create sample cryptocurrency data for testing"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    base_date = pd.Timestamp('2020-01-01')
    dates = [base_date + pd.Timedelta(days=i) for i in range(days)]
    
    for coin in coins:
        # Generate synthetic price data
        np.random.seed(hash(coin) % 1000)  # Consistent seed per coin
        
        # Random walk with trend
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [1000 * (1 + i * 0.1)]  # Starting price varies by coin
        
        for i in range(1, days):
            prices.append(prices[-1] * (1 + returns[i]))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
        
        # Save to CSV
        df.to_csv(save_path / f"{coin}_USD.csv", index=False)
        print(f"Created sample data for {coin}")

if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data(['BTC', 'ETH', 'ADA', 'DOGE'])
    
    # Run example
    pipeline, metrics = example_usage()