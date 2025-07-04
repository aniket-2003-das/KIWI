import pandas as pd
from pathlib import Path

# =============================================================================
# 2. DATA LOADER MODULE
# =============================================================================

class MultiCoinDataLoader:
    """Handles loading, combining, saving, and reloading cryptocurrency datasets."""

    def __init__(self, data_dir: str = "../data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.combined_data = None

    def load_single_coin_data(self, filepath: str, coin_name: str) -> pd.DataFrame:
        """Load and preprocess a single coin CSV file."""
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        daily_data = df.groupby('Date')['Close'].mean().reset_index()
        daily_data['Coin'] = coin_name
        return daily_data

    def load_multiple_coins(self, coin_configs: dict) -> pd.DataFrame:
        """Load multiple coins using load_single_coin_data method and combine them."""
        all_data = []

        for coin, file in coin_configs.items():
            file_path = self.data_dir / file
            try:
                coin_data = self.load_single_coin_data(file_path, coin)
                all_data.append(coin_data)
                print(f"Loaded {len(coin_data)} rows for {coin}")
            except Exception as e:
                print(f"Failed to load {coin}: {e}")

        if not all_data:
            raise ValueError("No valid coin data loaded.")

        self.combined_data = pd.concat(all_data).sort_values(['Coin', 'Date'])
        return self.combined_data

    def save_combined_data(self, filename: str = "ALL-COINS-USD.csv") -> None:
        """Save the combined dataset to a CSV file."""
        if self.combined_data is not None:
            path = self.data_dir / filename
            self.combined_data.to_csv(path, index=False)
            print(f"Saved combined data to {path}")
        else:
            raise ValueError("No data to save.")

    def load_combined_from_csv(self, filename: str = "ALL-COINS-USD.csv") -> pd.DataFrame:
        """Load previously saved combined coin data."""
        path = self.data_dir / filename
        print(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        self.combined_data = df.sort_values(['Coin', 'Date'])
        return self.combined_data

    def get_coin_data(self, coin_name: str) -> pd.DataFrame:
        """Extract data for a specific coin from combined dataset."""
        if self.combined_data is None:
            raise ValueError("Combined data not loaded.")
        return self.combined_data[self.combined_data['Coin'] == coin_name].copy()
    

if __name__ == "__main__":
    coin_files = {
        'BTC': 'BTC-USD.csv',
        'ETH': 'ETH-USD.csv',
        'SOL': 'SOL-USD.csv',
    }

    loader = MultiCoinDataLoader()

    # Load and save combined data
    loader.load_multiple_coins(coin_files)
    loader.save_combined_data()

    # Reload Combined Data
    loader2 = MultiCoinDataLoader()
    loader2.load_combined_from_csv()
    eth_data = loader2.get_coin_data('ETH')
