import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import os

# =============================================================================
# 1. DATA REQUESTOR MODULE
# =============================================================================

class DataRequestor:
    def __init__(self, tickers: list, start: str, end: str, interval: str = "1d", save_dir: str = "../data"):
        self.tickers = [t.upper() for t in tickers]
        self.start = start
        self.end = end
        self.interval = interval
        self.save_dir = save_dir

    def fetch_and_save_all(self):
        for ticker in self.tickers:
            print(f"\n================= {ticker} =================")
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=self.start, end=self.end, interval=self.interval)

                if data.empty:
                    print(f"⚠️ No data found for {ticker}. Skipping.")
                    continue

                self.preview(data)

                # Save to CSV
                os.makedirs(self.save_dir, exist_ok=True)
                filepath = os.path.join(self.save_dir, f"{ticker}.csv")
                data.to_csv(filepath)
                print(f"✅ Saved to {filepath}")

            except Exception as e:
                print(f"❌ Error fetching/saving data for {ticker}: {e}")

    @staticmethod
    def preview(data, rows: int = 3):
        print("\n📊 Preview:")
        print(data.head(rows))
        print(data.tail(rows))
        print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Example usage
if __name__ == "__main__":
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
    start_date = "2015-10-01"
    end_date = "2025-05-31"
    interval = "1d"
    dr = DataRequestor(tickers, start_date, end_date, interval)
    dr.fetch_and_save_all()


    
