# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score



# =============================================================================
# 8. EVALUATOR MODULE
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