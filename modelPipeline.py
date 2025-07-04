from modules.dataRequestor import DataRequestor
from modules.dataLoader import MultiCoinDataLoader
from modules.dataPreProcessor import MultiCoinPreprocessor
from modules.sequenceGenerator import SequenceGenerator
from modules.modelBuilding import MultiCoinLSTM
from modules.modelTraining import MultiCoinTrainer









# Example usage
if __name__ == "__main__":
    # Step 1:
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
    start_date = "2015-10-01"
    end_date = "2025-05-31"
    interval = "1d"
    dr = DataRequestor(tickers, start_date, end_date, interval)
    dr.fetch_and_save_all()

    # Step 2:
    coin_files = {
        'BTC': 'BTC-USD.csv',
        'ETH': 'ETH-USD.csv',
        'SOL': 'SOL-USD.csv',
    }

    loader = MultiCoinDataLoader()

    # Load and save combined data
    # loader.load_multiple_coins(coin_files)
    # loader.save_combined_data()

    # Reload Combined Data
    loader2 = MultiCoinDataLoader(data_dir="data")
    loader2.load_combined_from_csv()
    # eth_data = loader2.get_coin_data('ETH')
    combined_data = loader2.load_combined_from_csv()
    
    # Step 3:
    preprocessor = MultiCoinPreprocessor(prediction_days=60)
    coin_data = preprocessor.prepare_coin_data(combined_data)
    coin_encodings = preprocessor.create_coin_encoding(list(coin_data.keys()))
    # Save preprocessor data (scalers and encodings)
    preprocessor.save_preprocessor("assets/preprocessor_data.pkl")
    # Load preprocessor data (scalers and encodings)
    preprocessor.load_preprocessor("assets/preprocessor_data.pkl")

    # Step 4: Generate sequences
    seq_generator = SequenceGenerator(lookback=7)
    X_train, y_train, seq_info = seq_generator.generate_all_sequences(coin_data, coin_encodings)

    # Step 5. Model Building
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = MultiCoinLSTM(input_shape)
    model.build_model([128, 64], dropout_rate=0.2)
    
    print(model)  # Ensure this is not None
    print(model.model)  # Ensure this is a valid Keras model
    
    # Step 6. Training
    trainer = MultiCoinTrainer(model)
    
    # Simple train/val split
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=200)
    trainer.plot_training_history()
    trainer.load_best_model()