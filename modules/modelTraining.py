# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = Filter out INFO and WARNING, show only ERROR
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from modules.modelBuilding import MultiCoinLSTM

# =============================================================================
# 6. TRAINER MODULE
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
        """Plot and save training history"""
        if self.history:
            assets_path = Path("assets")
            assets_path.mkdir(exist_ok=True)  # Ensure the assets folder exists

            plt.figure(figsize=(12, 4))
            
            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Save the plot
            plot_path = assets_path / "training_history.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free memory
            print(f"Training history plot saved to {plot_path}")
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