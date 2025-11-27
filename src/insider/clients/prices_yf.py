# src/insider/clients/prices_yf.py
from __future__ import annotations
import pandas as pd
import yfinance as yf
import requests_cache
from datetime import datetime, timedelta

_session = requests_cache.CachedSession(
    cache_name="data/.cache/yf_cache",
    backend="sqlite",
    expire_after=60*60*24  # 1 day
)

def download_adj_close(tickers: list[str], start: str|datetime, end: str|datetime) -> pd.DataFrame:
    """
    Returns a wide DataFrame: index=Date (tz-naive), columns=tickers, values=Adj Close.
    """
    # yfinance uses its own session internally; caching still helps via its CDN
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(df, pd.Series):  # single ticker => make it 2D
        df = df.to_frame(tickers[0])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()



'''Testing method
if __name__ == "__main__":
    # Test the function with AAPL data from 2025-01-01 to 2025-01-07
    import matplotlib.pyplot as plt
    
    tickers = ['AAPL']
    start = '2025-01-01'
    end = '2025-01-07'
    
    df = download_adj_close(tickers, start, end)
    print("Downloaded data:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['AAPL'], marker='o')
    plt.title('AAPL Adjusted Close Prices (2025-01-01 to 2025-01-07)')
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.grid(True)
    plt.show()
'''