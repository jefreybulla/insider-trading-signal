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

# In[460]:


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

# In[461]:


NONDERIV_PATH = r"../NONDERIV_TRANS.tsv"  # <-- update this

nonderiv_cols = ['ACCESSION_NUMBER', 'SECURITY_TITLE', 'TRANS_DATE', 'TRANS_SHARES',
                 'TRANS_PRICEPERSHARE', 'TRANS_CODE']

df_nonderiv = pd.read_csv(NONDERIV_PATH, sep='\t', usecols=nonderiv_cols, low_memory=False)
df_nonderiv.head()


# ## Basic information and missing values
# We check the **shape**, **data types**, and **missing values** to understand data quality before applying any filters.

# In[462]:


print('Shape:', df_nonderiv.shape)
df_nonderiv.dtypes


# In[463]:


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

# In[464]:


df_filtered = df_nonderiv[
    df_nonderiv['SECURITY_TITLE'].str.contains('COMMON', case=False, na=False)
    & df_nonderiv['TRANS_CODE'].isin(['P', 'S'])
].copy()

print('Records after filtering:', len(df_filtered))
print('Percentage retained:', round(len(df_filtered) / len(df_nonderiv) * 100, 2), '%')

df_filtered['TRANS_CODE'].value_counts()


# ## 4. Transaction date analysis
# We convert `TRANS_DATE` to datetime and derive **year** and **year-month** fields to study temporal patterns.

# In[465]:


df_filtered['TRANS_DATE'] = pd.to_datetime(df_filtered['TRANS_DATE'], errors='coerce')
df_filtered['year'] = df_filtered['TRANS_DATE'].dt.year
df_filtered['year_month'] = df_filtered['TRANS_DATE'].dt.to_period('M')

print('Date range:', df_filtered['TRANS_DATE'].min(), 'to', df_filtered['TRANS_DATE'].max())

transactions_by_year = df_filtered['year'].value_counts().sort_index()
transactions_by_year


# In[466]:


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

# In[467]:


numeric_cols = ['TRANS_SHARES', 'TRANS_PRICEPERSHARE']
for col in numeric_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

df_filtered['dollar_value'] = df_filtered['TRANS_SHARES'] * df_filtered['TRANS_PRICEPERSHARE']

stats = df_filtered[['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'dollar_value']].describe()
stats


# In[468]:


percentiles = [10, 25, 50, 75, 90, 95, 99]
dv = df_filtered['dollar_value'].dropna()

pd.DataFrame({f'{p}th': [dv.quantile(p/100)] for p in percentiles})


# In[469]:


# Drop rows with transaction with no value
df_filtered = df_filtered[df_filtered['TRANS_SHARES'] > 0]
print('Records after filtering:', len(df_filtered))
df_filtered = df_filtered[df_filtered['TRANS_PRICEPERSHARE'] > 0]
print('Records after filtering:', len(df_filtered))

# Recompiled dollar value
df_filtered['dollar_value'] = df_filtered['TRANS_SHARES'] * df_filtered['TRANS_PRICEPERSHARE']

stats = df_filtered[['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'dollar_value']].describe()
stats


# Let's remove the rows with transaction dates that did not happen in year 2025

# In[470]:


df_filtered = df_filtered[df_filtered['year'] == 2025]
print('Records after filtering for 2025 only:', len(df_filtered))


# ## Multiple transactions per submision 

# In[471]:


df_filtered.head()


# In[472]:


# Let's check how many rows have unique ACCESSION_NUMBER
print('Number of unique ACCESSION_NUMBER:', df_filtered['ACCESSION_NUMBER'].nunique())


# We have that about half of the transaction have unique ids. We now know that is common to have multiple transaction for one SEC submision

# In[473]:


# Group by ACCESSION_NUMBER and calculate the time span for each
grouped_dates = df_filtered.groupby('ACCESSION_NUMBER')['TRANS_DATE']
time_windows = grouped_dates.max() - grouped_dates.min()

# Find the largest time window
largest_window = time_windows.max()
print(f'Largest time window between transactions for the same ACCESSION_NUMBER: {largest_window}')

# Find the ACCESSION_NUMBER with the largest time window
accession_with_largest_window = time_windows.idxmax()
print(f'ACCESSION_NUMBER with largest time window: {accession_with_largest_window}')

# Show all rows for that ACCESSION_NUMBER
rows_for_largest = df_filtered[df_filtered['ACCESSION_NUMBER'] == accession_with_largest_window]
rows_for_largest


# In[474]:


# Group by ACCESSION_NUMBER and count unique TRANS_CODE
grouped_codes = df_filtered.groupby('ACCESSION_NUMBER')['TRANS_CODE'].nunique()
conflicting_accessions = grouped_codes[grouped_codes > 1]

print(f'Number of ACCESSION_NUMBER with conflicting TRANS_CODE: {len(conflicting_accessions)}')

if len(conflicting_accessions) > 0:
    print('Examples of conflicting ACCESSION_NUMBER:')
    for acc in conflicting_accessions.index[:5]:  # Show first 5 examples
        print(f'\nACCESSION_NUMBER: {acc}')
        print(df_filtered[df_filtered['ACCESSION_NUMBER'] == acc][['TRANS_CODE', 'TRANS_DATE']].sort_values('TRANS_DATE'))
else:
    print('No conflicting TRANS_CODE found.')


# A small number of rows with same ACCESSION_NUMBER have conflicting TRANS_CODE. Let's drop those rows

# In[475]:


# Drop rows with conflicting TRANS_CODE
df_filtered = df_filtered[~df_filtered['ACCESSION_NUMBER'].isin(conflicting_accessions.index)]
print('Records after dropping conflicting TRANS_CODE:', len(df_filtered))


# Let's combine the transactions that are part of the same submision (same ACCESSION_NUMBER)

# In[476]:


# Sort by ACCESSION_NUMBER and TRANS_DATE
df_filtered = df_filtered.sort_values(['ACCESSION_NUMBER', 'TRANS_DATE'])

# Group by ACCESSION_NUMBER and aggregate
df_combined = df_filtered.groupby('ACCESSION_NUMBER').agg({
    'SECURITY_TITLE': 'last',
    'TRANS_CODE': 'last',
    'TRANS_SHARES': 'sum',
    'TRANS_PRICEPERSHARE': lambda x: (df_filtered.loc[x.index, 'dollar_value'].sum() / 
                                       df_filtered.loc[x.index, 'TRANS_SHARES'].sum()) 
                                      if df_filtered.loc[x.index, 'TRANS_SHARES'].sum() > 0 else 0,
    'dollar_value': 'sum',
    'year': 'last',
    'year_month': 'last',
    'TRANS_DATE': 'last'
}).reset_index()

print('Shape after combining:', df_combined.shape)
df_combined.head()


# In[477]:


# Show combined row for the ACCESSION_NUMBER =  0000947871-25-000303
rows_for_largest = df_combined[df_combined['ACCESSION_NUMBER'] == accession_with_largest_window]
rows_for_largest


# In[478]:


stats = df_combined[['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'dollar_value']].describe()
stats


# ## 6. Visualizations
# Below we plot:
# - Buy vs Sell transaction counts
# - Distribution of dollar values (log scale to handle skew)
# These plots help visually confirm patterns seen in the summary statistics.

# In[489]:


plt.figure()
df_combined['TRANS_CODE'].value_counts().plot(kind='bar')
plt.title('Buy (P) vs Sell (S) Transactions')
plt.xlabel('TRANS_CODE')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure()
dv_log = np.log10(df_combined['dollar_value'].dropna() + 1)    # log base 20
#dv_log = np.log1p(df_combined['dollar_value'].dropna() + 1)     # natural log
dv_log.plot(kind='hist', bins=50)
plt.title('Distribution of Dollar Value (log10 scale)')
plt.xlabel('log10(dollar_value + 1)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ##  Monthly trends: counts, dollar value, buy/sell ratio

# In[490]:


# Aggregate by month
monthly = (
    df_combined
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


# In[481]:


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

# In[491]:


buy_sell_summary = (
    df_combined
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


# In[483]:


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

# In[492]:


# Define threshold as 99th percentile of dollar_value
threshold_99 = df_combined['dollar_value'].quantile(0.99)

large_trades = df_combined[df_combined['dollar_value'] >= threshold_99].copy()
large_trades_sorted = large_trades.sort_values('dollar_value', ascending=False)

threshold_99, large_trades_sorted.head(20)


# In[485]:


# How many large trades are buys vs sells?
large_trades['TRANS_CODE'].value_counts(normalize=True)


# ## Security-level view: which securities see most insider activity?

# In[493]:


group_cols = []
if 'ISSUER_CIK' in df_combined.columns:
    group_cols.append('ISSUER_CIK')
if 'ISSUER_TRADING_SYMBOL' in df_combined.columns:
    group_cols.append('ISSUER_TRADING_SYMBOL')

# Always include security title as a fallback
group_cols.append('SECURITY_TITLE')

security_activity = (
    df_combined
    .groupby(group_cols)
    .agg(
        n_trades=('dollar_value', 'count'),
        total_dollar_value=('dollar_value', 'sum'),
        net_dollar_value=('dollar_value', lambda x: x[df_combined.loc[x.index, 'TRANS_CODE'] == 'P'].sum()
                                              - x[df_combined.loc[x.index, 'TRANS_CODE'] == 'S'].sum())
    )
    .sort_values('total_dollar_value', ascending=False)
)

security_activity.head(20)


# Shows which companies/securities attract the most insider trading and whether insiders are net buyers or sellers.

# # Stability and anomaly: month-over-month changes

# In[494]:


monthly_changes = monthly[['n_trades', 'total_dollar_value']].copy()
monthly_changes['n_trades_change_pct'] = monthly_changes['n_trades'].pct_change() * 100
monthly_changes['dollar_value_change_pct'] = monthly_changes['total_dollar_value'].pct_change() * 100

monthly_changes.tail()


# In[495]:


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




