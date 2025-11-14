#!/usr/bin/env python
# coding: utf-8

# # Part 1 – NONDERIV_TRANS.tsv Exploratory Data Analysis
# 
# This notebook performs exploratory data analysis (EDA) on the **NONDERIV_TRANS.tsv** file. We focus on:
# - Basic structure and missing values
# - Filtering to **COMMON stock** and **P/S (Buy/Sell)** transactions
# - Distribution of transaction codes
# - Transaction dates and time trends
# - Shares, price per share, and dollar-value distributions

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ## Load data
# We load only the relevant columns used in the analysis. That is: 
# 
# | **Field Name**        | **Description**                                                                                               | **Data Type**       | **Nullable** | **Key** |
# |------------------------|---------------------------------------------------------------------------------------------------------------|---------------------|---------------|----------|
# | SECURITY_TITLE         | Security title                                                                                                | VARCHAR2 (60)       | No            |          |
# | TRANS_DATE             | Transaction date in (DD-MON-YYYY) format.                                                                     | DATE                | No            |          |
# | TRANS_SHARES           | Transaction shares reported when Securities Acquired (A) or Disposed of (D).                                  | NUMBER(16,2)        | Yes           |          |
# | TRANS_PRICEPERSHARE    | Price of non-Derivative Transaction Security.                                                                 | NUMBER(16,2)        | Yes           |          |
# | TRANS_CODE             | Transaction code (values and descriptions are listed in the Appendix section 6.2 Trans Code List).            | VARCHAR2 (1)        | Yes           |          |
# 
# 
# If the file path is different on your system, update `NONDERIV_PATH` accordingly.

# In[3]:


NONDERIV_PATH = r"/Users/aaniaadap/Desktop/KDDM Project/Dataset/NONDERIV_TRANS.tsv"  # <-- update this

nonderiv_cols = ['SECURITY_TITLE', 'TRANS_DATE', 'TRANS_SHARES',
                 'TRANS_PRICEPERSHARE', 'TRANS_CODE']

df_nonderiv = pd.read_csv(NONDERIV_PATH, sep='\t', usecols=nonderiv_cols, low_memory=False)
df_nonderiv.head()


# ## Basic information and missing values
# We check the **shape**, **data types**, and **missing values** to understand data quality before applying any filters.

# In[4]:


print('Shape:', df_nonderiv.shape)
df_nonderiv.dtypes


# In[5]:


missing = df_nonderiv.isnull().sum()
missing_pct = (missing / len(df_nonderiv) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df


# ## Filter for COMMON stock and P/S transactions
# We restrict the dataset to:
# - Rows where `SECURITY_TITLE` contains **'COMMON'** (case-insensitive), and
# - `TRANS_CODE` is either **'P' (Purchase)** or **'S' (Sale)**.
# 
# This focuses the analysis on standard insider buy/sell transactions in common stock.

# In[6]:


df_filtered = df_nonderiv[
    df_nonderiv['SECURITY_TITLE'].str.contains('COMMON', case=False, na=False)
    & df_nonderiv['TRANS_CODE'].isin(['P', 'S'])
].copy()

print('Records after filtering:', len(df_filtered))
print('Percentage retained:', round(len(df_filtered) / len(df_nonderiv) * 100, 2), '%')

df_filtered['TRANS_CODE'].value_counts()


# ## 4. Transaction date analysis
# We convert `TRANS_DATE` to datetime and derive **year** and **year-month** fields to study temporal patterns.

# In[7]:


df_filtered['TRANS_DATE'] = pd.to_datetime(df_filtered['TRANS_DATE'], errors='coerce')
df_filtered['year'] = df_filtered['TRANS_DATE'].dt.year
df_filtered['year_month'] = df_filtered['TRANS_DATE'].dt.to_period('M')

print('Date range:', df_filtered['TRANS_DATE'].min(), 'to', df_filtered['TRANS_DATE'].max())

transactions_by_year = df_filtered['year'].value_counts().sort_index()
transactions_by_year


# In[8]:


plt.figure()
df_filtered['year'].value_counts().sort_index().plot(kind='bar')
plt.title('Transactions by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ## 5. Shares, price per share, and dollar value
# We examine the distributions of:
# - `TRANS_SHARES`
# - `TRANS_PRICEPERSHARE`
# - `dollar_value = TRANS_SHARES × TRANS_PRICEPERSHARE`
# including summary statistics and selected percentiles.

# In[9]:


numeric_cols = ['TRANS_SHARES', 'TRANS_PRICEPERSHARE']
for col in numeric_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

df_filtered['dollar_value'] = df_filtered['TRANS_SHARES'] * df_filtered['TRANS_PRICEPERSHARE']

stats = df_filtered[['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'dollar_value']].describe()
stats


# In[10]:


percentiles = [10, 25, 50, 75, 90, 95, 99]
dv = df_filtered['dollar_value'].dropna()

pd.DataFrame({f'{p}th': [dv.quantile(p/100)] for p in percentiles})


# ## 6. Visualizations
# Below we plot:
# - Buy vs Sell transaction counts
# - Distribution of dollar values (log scale to handle skew)
# These plots help visually confirm patterns seen in the summary statistics.

# In[11]:


plt.figure()
df_filtered['TRANS_CODE'].value_counts().plot(kind='bar')
plt.title('Buy (P) vs Sell (S) Transactions')
plt.xlabel('TRANS_CODE')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure()
dv_log = np.log10(df_filtered['dollar_value'].dropna() + 1)
dv_log.plot(kind='hist', bins=50)
plt.title('Distribution of Dollar Value (log10 scale)')
plt.xlabel('log10(dollar_value + 1)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ##  Monthly trends: counts, dollar value, buy/sell ratio

# In[12]:


# Aggregate by month
monthly = (
    df_filtered
    .groupby('year_month')
    .agg(
        n_trades=('TRANS_CODE', 'count'),
        total_dollar_value=('dollar_value', 'sum'),
        buys=('TRANS_CODE', lambda x: (x == 'P').sum()),
        sells=('TRANS_CODE', lambda x: (x == 'S').sum())
    )
)

monthly['buy_sell_ratio'] = monthly['buys'] / monthly['sells'].replace(0, np.nan)

monthly.tail()


# In[13]:


# Plot: trades per month
plt.figure()
monthly['n_trades'].plot()
plt.title('Number of Transactions per Month')
plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: total dollar value per month
plt.figure()
monthly['total_dollar_value'].plot()
plt.title('Total Dollar Value per Month')
plt.xlabel('Year-Month')
plt.ylabel('Total Dollar Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: buy/sell ratio over time
plt.figure()
monthly['buy_sell_ratio'].plot()
plt.title('Buy/Sell Transaction Count Ratio Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Buy/Sell Ratio')
plt.xticks(rotation=45)
plt.axhline(1, linestyle='--')
plt.tight_layout()
plt.show()


# Here we analyse when insiders are more net-buy vs net-sell and how intensity changes over time

# ## Compare buy vs sell behaviour (size, price, dollar value)

# In[15]:


buy_sell_summary = (
    df_filtered
    .groupby('TRANS_CODE')
    .agg(
        n_trades=('dollar_value', 'count'),
        median_dollar_value=('dollar_value', 'median'),
        mean_dollar_value=('dollar_value', 'mean'),
        median_shares=('TRANS_SHARES', 'median'),
        mean_shares=('TRANS_SHARES', 'mean'),
        median_price=('TRANS_PRICEPERSHARE', 'median'),
        mean_price=('TRANS_PRICEPERSHARE', 'mean')
    )
)

buy_sell_summary


# In[16]:


# Visual: distribution of log-dollar value by P vs S
plt.figure()
for code in ['P', 'S']:
    subset = np.log10(df_filtered.loc[df_filtered['TRANS_CODE'] == code, 'dollar_value'] + 1)
    subset.plot(kind='kde', label=code)
plt.title('Log Dollar Value Distribution: Buys (P) vs Sells (S)')
plt.xlabel('log10(dollar_value + 1)')
plt.legend()
plt.tight_layout()
plt.show()


# Checking if sells are larger than buys, if buys happen at lower/higher prices, etc.

# ## Flag and inspect large / extreme trades (outliers)

# In[17]:


# Define threshold as 99th percentile of dollar_value
threshold_99 = df_filtered['dollar_value'].quantile(0.99)

large_trades = df_filtered[df_filtered['dollar_value'] >= threshold_99].copy()
large_trades_sorted = large_trades.sort_values('dollar_value', ascending=False)

threshold_99, large_trades_sorted.head(20)


# In[18]:


# How many large trades are buys vs sells?
large_trades['TRANS_CODE'].value_counts(normalize=True)


# ## Security-level view: which securities see most insider activity?

# In[19]:


group_cols = []
if 'ISSUER_CIK' in df_filtered.columns:
    group_cols.append('ISSUER_CIK')
if 'ISSUER_TRADING_SYMBOL' in df_filtered.columns:
    group_cols.append('ISSUER_TRADING_SYMBOL')

# Always include security title as a fallback
group_cols.append('SECURITY_TITLE')

security_activity = (
    df_filtered
    .groupby(group_cols)
    .agg(
        n_trades=('dollar_value', 'count'),
        total_dollar_value=('dollar_value', 'sum'),
        net_dollar_value=('dollar_value', lambda x: x[df_filtered.loc[x.index, 'TRANS_CODE'] == 'P'].sum()
                                              - x[df_filtered.loc[x.index, 'TRANS_CODE'] == 'S'].sum())
    )
    .sort_values('total_dollar_value', ascending=False)
)

security_activity.head(20)


# Shows which companies/securities attract the most insider trading and whether insiders are net buyers or sellers.

# # Stability and anomaly: month-over-month changes

# In[20]:


monthly_changes = monthly[['n_trades', 'total_dollar_value']].copy()
monthly_changes['n_trades_change_pct'] = monthly_changes['n_trades'].pct_change() * 100
monthly_changes['dollar_value_change_pct'] = monthly_changes['total_dollar_value'].pct_change() * 100

monthly_changes.tail()


# In[21]:


plt.figure()
monthly_changes['n_trades_change_pct'].plot()
plt.title('MoM % Change in Number of Trades')
plt.xlabel('Year-Month')
plt.ylabel('% Change')
plt.axhline(0, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
monthly_changes['dollar_value_change_pct'].plot()
plt.title('MoM % Change in Total Dollar Value')
plt.xlabel('Year-Month')
plt.ylabel('% Change')
plt.axhline(0, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Spikes here = months where insider activity is unusually high/low → candidate for deeper investigation.

# In[ ]:




