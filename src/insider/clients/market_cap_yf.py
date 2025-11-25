# Create a new file with the following content, mirroring the structure of prices_yf.py

from __future__ import annotations
import yfinance as yf
from typing import Dict, List, Optional

def get_market_caps(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Fetches market capitalization for a list of tickers using yfinance.
    
    Returns a dictionary with tickers as keys and market cap (in USD) as values.
    If a ticker is invalid or data unavailable, the value will be None.
    """
    market_caps = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_caps[ticker] = info.get('marketCap')
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            market_caps[ticker] = None
    return market_caps

"""
# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'INVALID']
    caps = get_market_caps(tickers)
    print("Market Caps:", caps)
"""