# # Multi-Coin LSTM Framework - Usage Examples
# # Various ways to use the framework for different scenarios

# # =============================================================================
# # EXAMPLE 1: BASIC USAGE - Quick Start
# # =============================================================================

# def basic_usage():
#     """Simple example for getting started"""
    
#     # Define your coin data files
#     coin_configs = {
#         'BTC': 'data/BTC_USD.csv',
#         'ETH': 'data/ETH_USD.csv',
#         'ADA': 'data/ADA_USD.csv'
#     }
    
#     # Initialize and run pipeline
#     pipeline = MultiCoinLSTMPipeline(lookback=14, prediction_days=60)
#     metrics = pipeline.run_complete_pipeline(
#         coin_configs=coin_configs,
#         epochs=100,
#         num_future_days=14
#     )
    
#     return pipeline, metrics

# # =============================================================================
# # EXAMPLE 2: STEP-BY-STEP USAGE - Full Control
# # =============================================================================

# def step_by_step_usage():
#     """Example with full control over each step"""
    
#     # Initialize pipeline
#     pipeline = MultiCoinLSTMPipeline(lookback=21, prediction_days=30)
    
#     # Step 1: Load data
#     coin_configs = {
#         'BTC': 'data/BTC_USD.csv',
#         'ETH': 'data/ETH_USD.csv',
#         'LTC': 'data/LTC_USD.csv',
#         'XRP': 'data/XRP_USD.csv'
#     }
#     pipeline.load_data(coin_configs)
    
#     # Step 2: Preprocess
#     pipeline.preprocess_data()
    
#     # Step 3: Generate sequences
#     pipeline.prepare_sequences()
    
#     # Step 4: Build custom model
#     pipeline.build_model(
#         lstm_units=[256, 128, 64],  # 3-layer LSTM
#         dropout_rate=0.3
#     )
    
#     # Step 5: Train with custom parameters
#     pipeline.train_model(
#         epochs=200,
#         batch_size=64,
#         patience=15
#     )
    
#     # Step 6: Make predictions
#     pipeline.make_predictions(num_future_days=30)
    
#     # Step 7: Evaluate
#     metrics = pipeline.evaluate_results()
    
#     # Step 8: Save pipeline
#     pipeline.save_pipeline("models/multi_coin_model/")
    
#     return pipeline, metrics

# # =============================================================================
# # EXAMPLE 3: LARGE SCALE DEPLOYMENT - 20+ Coins
# # =============================================================================

# def large_scale_deployment():
#     """Example for handling many coins efficiently"""
    
#     # Define 20+ coins
#     major_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TRX', 'LTC', 'MATIC']
#     altcoins = ['LINK', 'UNI', 'ATOM', 'VET', 'ICP', 'FIL', 'THETA', 'MANA', 'SAND', 'AXS']
    
#     all_coins = major_coins + altcoins
    
#     # Create coin configs
#     coin_configs = {coin: f'data/{coin}_USD.csv' for coin in all_coins}
    
#     # Initialize with optimized parameters for large scale
#     pipeline = MultiCoinLSTMPipeline(
#         lookback=30,  # Longer lookback for more complex patterns
#         prediction_days=90  # Longer test period
#     )
    
#     # Load data
#     pipeline.load_data(coin_configs)
    
#     # Preprocess
#     pipeline.preprocess_data()
    
#     # Generate sequences
#     pipeline.prepare_sequences()
    
#     # Build larger model for complex patterns
#     pipeline.build_model(
#         lstm_units=[512, 256, 128],
#         dropout_rate=0.4
#     )
    
#     # Train with larger batch size and more epochs
#     pipeline.train_model(
#         epochs=300,
#         batch_size=128,
#         patience=20
#     )
    
#     # Make predictions
#     pipeline.make_predictions(num_future_days=7)
    
#     # Evaluate
#     metrics = pipeline.evaluate_results()
    
#     # Save for production
#     pipeline.save_pipeline("production_models/")
    
#     return pipeline, metrics

# =============================================================================
# EXAMPLE 4: USING INDIVIDUAL MODULES
# =============================================================================

def individual_modules_usage():
    """Example of using individual modules for custom workflows"""
    
    # 1. Data Loading
    data_loader = MultiCoinDataLoader("data/")
    coin_configs = {'BTC': 'BTC_USD.csv', 'ETH': 'ETH_USD.csv'}
    combined_data = data_loader.load_multiple_coins(coin_configs)
    
    # 2. Preprocessing
    preprocessor = MultiCoinPreprocessor(prediction_days=60)
    coin_data = preprocessor.prepare_coin_data(combined_data)
    coin_encodings = preprocessor.create_coin_encoding(list(coin_data.keys()))
    
    # 3. Sequence Generation
    seq_generator = SequenceGenerator(lookback=14)
    X_train, y_train, seq_info = seq_generator.generate_all_sequences(coin_data, coin_encodings)
    
    # 4. Model Building
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = MultiCoinLSTM(input_shape)
    model.build_model([128, 64], dropout_rate=0.2)
    
    # 5. Training
    trainer = MultiCoinTrainer(model)
    
    # Simple train/val split
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)
    trainer.load_best_model()
    
    # 6. Prediction
    predictor = MultiCoinPredictor(model, preprocessor, seq_generator)
    predictions = predictor.predict_all_coins(coin_data, num_future_days=14)
    
    # 7. Evaluation
    evaluator = MultiCoinEvaluator()
    metrics = evaluator.calculate_metrics(predictions)
    evaluator.print_metrics_summary()
    evaluator.plot_predictions(predictions)
    
    return predictions, metrics

# =============================================================================
# EXAMPLE 5: CUSTOM COIN ADDITION
# =============================================================================

def add_new_coin_to_existing_model():
    """Example of adding a new coin to an existing trained model"""
    
    # Load existing pipeline
    pipeline = MultiCoinLSTMPipeline()
    
    # Load existing preprocessor
    pipeline.preprocessor.load_preprocessor("saved_pipeline/preprocessor.pkl")
    
    # Add new coin data
    new_coin_configs = {
        'BTC': 'data/BTC_USD.csv',
        'ETH': 'data/ETH_USD.csv',
        'NEW_COIN': 'data/NEW_COIN_USD.csv'  # New coin
    }
    
    # Retrain with new coin
    pipeline.load_data(new_coin_configs)
    pipeline.preprocess_data()
    pipeline.prepare_sequences()
    
    # Build model with updated input shape
    pipeline.build_model()
    
    # Train (you might want to use transfer learning here)
    pipeline.train_model(epochs=100)
    
    # Make predictions
    pipeline.make_predictions()
    
    # Evaluate
    metrics = pipeline.evaluate_results()
    
    return pipeline, metrics

# =============================================================================
# EXAMPLE 6: HYPERPARAMETER TUNING
# =============================================================================

def hyperparameter_tuning():
    """Example of hyperparameter tuning"""
    
    # Define hyperparameter combinations
    hyperparams = [
        {'lookback': 14, 'lstm_units': [128, 64], 'dropout': 0.2, 'batch_size': 32},
        {'lookback': 21, 'lstm_units': [256, 128], 'dropout': 0.3, 'batch_size': 64},
        {'lookback': 30, 'lstm_units': [512, 256, 128], 'dropout': 0.4, 'batch_size': 128},
    ]
    
    coin_configs = {
        'BTC': 'data/BTC_USD.csv',
        'ETH': 'data/ETH_USD.csv',
        'ADA': 'data/ADA_USD.csv'
    }
    
    best_metrics = {}
    best_params = None
    
    for i, params in enumerate(hyperparams):
        print(f"\nTesting hyperparameters {i+1}/{len(hyperparams)}: {params}")
        
        # Initialize pipeline with current parameters
        pipeline = MultiCoinLSTMPipeline(
            lookback=params['lookback'],
            prediction_days=60
        )
        
        # Load and preprocess data
        pipeline.load_data(coin_configs)
        pipeline.preprocess_data()
        pipeline.prepare_sequences()
        
        # Build model
        pipeline.build_model(
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout']
        )
        
        # Train
        pipeline.train_model(
            epochs=50,  # Reduced for tuning
            batch_size=params['batch_size']
        )
        
        # Predict and evaluate
        pipeline.make_predictions()
        metrics = pipeline.evaluate_results()
        
        # Store results
        avg_r2 = np.mean([m['r2'] for m in metrics.values()])
        best_metrics[i] = {'params': params, 'avg_r2': avg_r2, 'metrics': metrics}
        
        if best_params is None or avg_r2 > best_metrics[best_params]['avg_r2']:
            best_params = i
    
    print(f"\nBest parameters: {best_metrics[best_params]['params']}")
    print(f"Best average R²: {best_metrics[best_params]['avg_r2']:.3f}")
    
    return best_metrics

# =============================================================================
# EXAMPLE 7: REAL-TIME PREDICTION PIPELINE
# =============================================================================

class RealTimePredictionPipeline:
    """Real-time prediction pipeline for production use"""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.pipeline = MultiCoinLSTMPipeline()
        self.load_trained_model(model_path, preprocessor_path)
        
    def load_trained_model(self, model_path: str, preprocessor_path: str):
        """Load pre-trained model and preprocessor"""
        # Load preprocessor
        self.pipeline.preprocessor.load_preprocessor(preprocessor_path)
        
        # Load model
        self.pipeline.model = MultiCoinLSTM((14, len(self.pipeline.preprocessor.coins) + 1))
        self.pipeline.model.model = load_model(model_path)
        
        # Initialize other components
        self.pipeline.sequence_generator = SequenceGenerator(14)
        self.pipeline.predictor = MultiCoinPredictor(
            self.pipeline.model,
            self.pipeline.preprocessor,
            self.pipeline.sequence_generator
        )
    
    def predict_coin_price(self, coin: str, recent_prices: np.ndarray, days_ahead: int = 1):
        """
        Predict price for a specific coin given recent prices
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            recent_prices: Array of recent prices (length = lookback)
            days_ahead: Number of days to predict ahead
        """
        if coin not in self.pipeline.preprocessor.scalers:
            raise ValueError(f"Coin {coin} not supported")
        
        # Scale recent prices
        scaler = self.pipeline.preprocessor.scalers[coin]
        scaled_prices = scaler.transform(recent_prices.reshape(-1, 1))
        
        # Get coin encoding
        coin_encoding = self.pipeline.preprocessor.coin_encoders[coin]
        
        # Prepare sequence
        sequence_with_coin = []
        for i in range(len(scaled_prices)):
            timestep = np.concatenate([
                [scaled_prices[i, 0]],
                coin_encoding
            ])
            sequence_with_coin.append(timestep)
        
        # Predict
        predictions = []
        current_sequence = np.array([sequence_with_coin])
        
        for _ in range(days_ahead):
            pred = self.pipeline.model.model.predict(current_sequence, verbose=0)
            pred_price = scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
            predictions.append(pred_price)
            
            # Update sequence for next prediction
            if days_ahead > 1:
                # Roll sequence and add new prediction
                new_timestep = np.concatenate([[pred[0, 0]], coin_encoding])
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = new_timestep
        
        return np.array(predictions)

def real_time_example():
    """Example of real-time prediction"""
    
    # Initialize real-time pipeline
    rt_pipeline = RealTimePredictionPipeline(
        "saved_pipeline/model.keras",
        "saved_pipeline/preprocessor.pkl"
    )
    
    # Simulate recent BTC prices
    recent_btc_prices = np.random.uniform(45000, 55000, 14)
    
    # Predict next 3 days
    predictions = rt_pipeline.predict_coin_price('BTC', recent_btc_prices, days_ahead=3)
    
    print(f"Recent BTC prices: {recent_btc_prices[-5:]}")
    print(f"Predicted prices for next 3 days: {predictions}")
    
    return predictions

# =============================================================================
# EXAMPLE 8: ADVANCED EVALUATION AND BACKTESTING
# =============================================================================

def advanced_backtesting():
    """Advanced backtesting with rolling windows"""
    
    # Load data
    coin_configs = {'BTC': 'data/BTC_USD.csv', 'ETH': 'data/ETH_USD.csv'}
    
    # Parameters
    lookback = 14
    prediction_days = 30
    roll_days = 7  # Re-train every 7 days
    
    # Initialize results storage
    backtest_results = {}
    
    # Get total data length
    data_loader = MultiCoinDataLoader()
    combined_data = data_loader.load_multiple_coins(coin_configs)
    
    # Perform rolling backtests
    for coin in coin_configs.keys():
        coin_data = combined_data[combined_data['Coin'] == coin]
        total_days = len(coin_data)
        
        # Storage for this coin
        backtest_results[coin] = {
            'predictions': [],
            'actuals': [],
            'dates': []
        }
        
        # Rolling window backtest
        for start_day in range(100, total_days - prediction_days, roll_days):
            end_day = start_day + prediction_days
            
            # Create training data up to start_day
            train_data = coin_data.iloc[:start_day]
            test_data = coin_data.iloc[start_day:end_day]
            
            # Train model on this window
            pipeline = MultiCoinLSTMPipeline(lookback, prediction_days)
            
            # Prepare mini dataset
            mini_combined = train_data.copy()
            mini_combined['Coin'] = coin
            
            # Quick training
            pipeline.combined_data = mini_combined
            pipeline.preprocess_data()
            pipeline.prepare_sequences()
            pipeline.build_model()
            pipeline.train_model(epochs=20)  # Quick training
            
            # Make prediction
            pipeline.make_predictions(num_future_days=len(test_data))
            
            # Store results
            if coin in pipeline.predictions:
                pred_data = pipeline.predictions[coin]
                backtest_results[coin]['predictions'].extend(pred_data['future_predictions'])
                backtest_results[coin]['actuals'].extend(test_data['Close'].values)
                backtest_results[coin]['dates'].extend(test_data['Date'].values)
    
    # Calculate backtest metrics
    for coin in backtest_results:
        if len(backtest_results[coin]['predictions']) > 0:
            preds = np.array(backtest_results[coin]['predictions'])
            actuals = np.array(backtest_results[coin]['actuals'])
            
            rmse = np.sqrt(np.mean((preds - actuals) ** 2))
            r2 = r2_score(actuals, preds)
            
            print(f"{coin} Backtest Results:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.3f}")
            print(f"  Predictions: {len(preds)}")
    
    return backtest_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Choose which example to run
    print("Multi-Coin LSTM Framework Examples")
    print("1. Basic Usage")
    print("2. Step-by-Step Usage")
    print("3. Large Scale Deployment")
    print("4. Individual Modules")
    print("5. Hyperparameter Tuning")
    print("6. Real-time Prediction")
    print("7. Advanced Backtesting")
    
    choice = input("Enter choice (1-7): ")
    
    if choice == '1':
        pipeline, metrics = basic_usage()
    elif choice == '2':
        pipeline, metrics = step_by_step_usage()
    elif choice == '3':
        pipeline, metrics = large_scale_deployment()
    elif choice == '4':
        predictions, metrics = individual_modules_usage()
    elif choice == '5':
        best_metrics = hyperparameter_tuning()
    elif choice == '6':
        predictions = real_time_example()
    elif choice == '7':
        backtest_results = advanced_backtesting()
    else:
        print("Invalid choice. Running basic usage...")
        pipeline, metrics = basic_usage()