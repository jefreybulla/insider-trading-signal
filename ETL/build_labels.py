#!/usr/bin/env python
# coding: utf-8

# In[28]:


# notebooks/02_build_labels.ipynb (conceptual cells)
import sys
sys.path.insert(0, '../src')

from insider.clients.prices_yf import download_adj_close
from insider.clients.market_cap_yf import get_market_caps
from insider.calendar import nyse_days
from insider.features import compute_forward_return, log1p_safe, size_vs_cap
from insider.config import SETTINGS
import pandas as pd
import numpy as np



# events_2025Q1.parquet is the jopined data obtained after mergest relevant data from Transactions, Submissions and Report Owner
events = pd.read_parquet("data/raw/events_2025Q1.parquet")  
events.head()


# In[29]:


tickers = ['^GSPC'] + sorted(events["ticker"].unique())
print(len(tickers))

start = events["date"].min() - pd.Timedelta(days=10)    # -10 cushion for non-trading days
end   = events["date"].max() + pd.Timedelta(days=180)   # +180 ensure data for t+63
print(f"start={start}, end={end}")


# ## Fetch stock prices

# In[30]:


#start = '2025-01-01'
#end = '2025-01-15'
#tickers = ['^GSPC', 'MSFT', 'META', 'GOOGL']
#px = download_adj_close(tickers, start, end)


# In[31]:


# Data fetch happened on 11-24-2025. Number of columns with any NaN: 1189 out of 2374
# px.to_parquet("data/engineered/price_data_2025Q1.parquet") 
# price_data_2025Q1.parquet stores the fetched stock market data
px = pd.read_parquet("data/engineered/price_data_2025Q1.parquet")  
print(f"px shape: {px.shape}")
print(f"Number of columns with any NaN: {px.isnull().any().sum()}")

# Drop all columns with any NaN
px = px.dropna(axis=1, how='any')
print(f"px shape after dropping columns with NaN: {px.shape}")


# In[32]:


td = nyse_days(start, end)

events = compute_forward_return(events, px, td, n=63)
events = events.dropna(subset=["P_t","P_t_plus_63"])

events["forward_ret_63"] = events["P_t_plus_63"]/events["P_t"] - 1
events["label_up"] = (events["forward_ret_63"] > 0).astype(int)


# In[33]:


events.head()


# ## Add market benchmark with S&P500

# In[34]:


# Add SP500 (^GSPC) forward returns to events
# Get SP500 prices at event date and t+63

def get_sp500_prices(row):
    """Get SP500 prices at transaction date and 63 days later"""
    t = pd.Timestamp(row["date"])

    # Find SP500 price at or nearest to transaction date
    sp500_prices = px['^GSPC'].dropna()
    pos_t = sp500_prices.index.searchsorted(t, "left")
    if pos_t >= len(sp500_prices):
        pos_t = len(sp500_prices) - 1
    sp500_t = sp500_prices.iloc[pos_t]

    # Find SP500 price 63 trading days later
    # Estimate position (roughly 63 trading days ≈ 90 calendar days)
    t_plus_63_approx = t + pd.Timedelta(days=90)
    pos_t63 = sp500_prices.index.searchsorted(t_plus_63_approx, "left")
    if pos_t63 >= len(sp500_prices):
        pos_t63 = len(sp500_prices) - 1
    sp500_t63 = sp500_prices.iloc[pos_t63]

    if pd.isna(sp500_t) or pd.isna(sp500_t63) or sp500_t == 0:
        return pd.Series({"SP500_t": np.nan, "SP500_t_plus_63": np.nan, "forward_ret_SP500": np.nan})

    forward_ret_sp500 = sp500_t63 / sp500_t - 1
    return pd.Series({"SP500_t": float(sp500_t), "SP500_t_plus_63": float(sp500_t63), "forward_ret_SP500": forward_ret_sp500})

# Apply the function
sp500_data = events.apply(get_sp500_prices, axis=1)
events = pd.concat([events, sp500_data], axis=1)

print("Added SP500 columns:")
print(events[['date', 'SP500_t', 'SP500_t_plus_63', 'forward_ret_SP500']].head())


# In[35]:


events.head()


# In[36]:


# Add label_up_market: 1 if stock outperforms SP500, 0 otherwise
events["label_up_market"] = (events["forward_ret_63"] > events["forward_ret_SP500"]).astype(int)

print("Added label_up_market column:")
print(events[['forward_ret_63', 'forward_ret_SP500', 'label_up_market']].head(10))
print(f"\nLabel distribution:")
print(events['label_up_market'].value_counts())


# In[37]:


events.head(10)


# In[38]:


print(len(events))


# ## Fetch Market Cap

# In[39]:


# Get all tickers from px (excluding the index)
tickers_list = px.columns.tolist()
#print(f"Fetching market caps for {len(tickers_list)} tickers...")


# In[40]:


# Fetch market caps from API
# Fetch completed in 11/25/2025. Took ~14 minutes to run. 
# Market caps fetched: 1178 with data, 5 missing
#market_caps = get_market_caps(tickers_list)
#market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['ticker', 'market_cap'])


# In[41]:


# Convert to DataFrame for easier viewing
#print(f"\nMarket caps fetched: {len(market_cap_df[market_cap_df['market_cap'].notna()])} with data, {len(market_cap_df[market_cap_df['market_cap'].isna()])} missing")
#print("\nSample market caps:")
#print(market_cap_df.head(10))

#market_cap_df.to_parquet("data/engineered/market_cap_data_2025Q1.parquet")   

# Create a mapping for events
#market_cap_map = market_caps
#print(f"\nTotal tickers with market cap data: {sum(1 for v in market_cap_map.values() if v is not None)}")



# In[42]:


# market_cap_data_2025Q1.parquet stores the market cap data fetched from APIs
market_cap_df = pd.read_parquet("data/engineered/market_cap_data_2025Q1.parquet")  


# In[43]:


print(f"px shape: {market_cap_df.shape}")


# In[44]:


# Create a mapping of ticker to market cap
market_cap_map = dict(zip(market_cap_df['ticker'], market_cap_df['market_cap']))

# Add size_vs_cap column: dollar_value / market_cap
events['market_cap'] = events['ticker'].map(market_cap_map)
events['size_vs_cap'] = events['dollar_value'] / events['market_cap']

# Replace inf and -inf with NaN
events['size_vs_cap'] = events['size_vs_cap'].replace([np.inf, -np.inf], np.nan)

print("Added size_vs_cap column:")
print(events[['ticker', 'dollar_value', 'market_cap', 'size_vs_cap']].head(10))
print(f"\nMissing values in size_vs_cap: {events['size_vs_cap'].isnull().sum()}")
print(f"Summary statistics for size_vs_cap:")
print(events['size_vs_cap'].describe())


# In[45]:


events.head()


# In[46]:


print("Total rows in events:", len(events))
print("Rows with any missing or NaN values:", events.isnull().any(axis=1).sum())
print("\nNaN counts per column:")
print(events.isnull().sum())


# In[47]:


# Drop rows with any NaN or missing values
events = events.dropna()

# Verify the changes
print("Total rows after dropping NaN:", len(events))
print("Rows with any missing or NaN values after drop:", events.isnull().any(axis=1).sum())


# In[48]:


# Add natural log of size_vs_cap column
events['log_size_vs_cap'] = np.log(events['size_vs_cap'])
events.head()


# In[49]:


events = events.drop(columns=['ACCESSION_NUMBER', 'date', 'dollar_value', 'ticker', 'P_t', 'P_t_plus_63', 'forward_ret_63', 'label_up', 'SP500_t', 'SP500_t_plus_63', 'forward_ret_SP500', 'market_cap', 'size_vs_cap'  ])


# In[50]:


events.head(10)


# In[51]:


print(len(events))


# In[52]:


# Get current column order
cols = events.columns.tolist()

# Find indices of the columns to swap
idx_label = cols.index('label_up_market')
idx_log = cols.index('log_size_vs_cap')

# Swap their positions
cols[idx_label], cols[idx_log] = cols[idx_log], cols[idx_label]

# Reorder the DataFrame
events = events[cols]

# Verify the new order
print("New column order:")
print(events.columns.tolist())


# In[53]:


events.head(10)


# In[54]:


print(len(events))


# In[55]:


events.to_parquet("data/engineered/final_data_2025Q1.parquet") 

