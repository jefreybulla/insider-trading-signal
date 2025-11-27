#!/usr/bin/env python
# coding: utf-8

# # Join data from Transactions, reporting owner and submission
# 
# In this notebook we create a transactions_data table as follows
# - Take valid transactions from NONDERIV_TRANS.tsv (~11k rows)
# - Add a 'reporter' column using `RPTOWNER_RELATIONSHIP` from REPORTINGOWNER.tsv
# - Add a 'ticker' column using `ISSUERTRADINGSYMBOL` from SUBMISSION.tsv

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# In[15]:


NONDERIV_PATH = r"../NONDERIV_TRANS.tsv"  # <-- update this

nonderiv_cols = ['ACCESSION_NUMBER', 'SECURITY_TITLE', 'TRANS_DATE', 'TRANS_SHARES',
                 'TRANS_PRICEPERSHARE', 'TRANS_CODE']

df_nonderiv = pd.read_csv(NONDERIV_PATH, sep='\t', usecols=nonderiv_cols, low_memory=False)
df_nonderiv.head()


# ## Basic information and missing values
# We check the **shape**, **data types**, and **missing values** to understand data quality before applying any filters.

# In[16]:


print('Shape:', df_nonderiv.shape)


# In[17]:


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

# In[18]:


df_filtered = df_nonderiv[
    df_nonderiv['SECURITY_TITLE'].str.contains('COMMON', case=False, na=False)
    & df_nonderiv['TRANS_CODE'].isin(['P', 'S'])
].copy()

print('Records after filtering:', len(df_filtered))
print('Percentage retained:', round(len(df_filtered) / len(df_nonderiv) * 100, 2), '%')

df_filtered['TRANS_CODE'].value_counts()


# ## Clean data

# In[19]:


# Drop rows with transaction with no value
df_filtered = df_filtered[df_filtered['TRANS_SHARES'] > 0]
print('Records after filtering:', len(df_filtered))
df_filtered = df_filtered[df_filtered['TRANS_PRICEPERSHARE'] > 0]
print('Records after filtering:', len(df_filtered))

stats = df_filtered[['TRANS_SHARES', 'TRANS_PRICEPERSHARE']].describe()
stats


# In[20]:


df_filtered['TRANS_DATE'] = pd.to_datetime(df_filtered['TRANS_DATE'], errors='coerce')
df_filtered['year'] = df_filtered['TRANS_DATE'].dt.year
#print(df_filtered.head())

# Keep 2025 transactions only
df_filtered = df_filtered[df_filtered['year'] == 2025]
print('Records after filtering for 2025 only:', len(df_filtered))

# Transactions per month in 2025
df_filtered['month'] = df_filtered['TRANS_DATE'].dt.month
rows_per_month = df_filtered.groupby('month').size()
print("Rows per month in 2025:")
print(rows_per_month)


# There are a couple of invalid transactions with date after end of Q1 (after March). Let's drop those transactions

# In[21]:


df_filtered = df_filtered[df_filtered['month'].isin([1, 2, 3, 4])]
print('Records after filtering for January to April 2025:', len(df_filtered))

# Recheck transactions per month in 2025
df_filtered['month'] = df_filtered['TRANS_DATE'].dt.month
rows_per_month = df_filtered.groupby('month').size()
print("Rows per month in 2025:")
print(rows_per_month)


# ## Creating dollar_value

# In[22]:


numeric_cols = ['TRANS_SHARES', 'TRANS_PRICEPERSHARE']
for col in numeric_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

df_filtered['dollar_value'] = df_filtered['TRANS_SHARES'] * df_filtered['TRANS_PRICEPERSHARE']
stats = df_filtered[['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'dollar_value']].describe()
stats


# ## Merging transactions that belong to the same submission

# Drop rows with same ACCESSION_NUMBER have conflicting TRANS_CODE.

# In[23]:


# Drop rows with conflicting TRANS_CODE
grouped_codes = df_filtered.groupby('ACCESSION_NUMBER')['TRANS_CODE'].nunique()
conflicting_accessions = grouped_codes[grouped_codes > 1]

df_filtered = df_filtered[~df_filtered['ACCESSION_NUMBER'].isin(conflicting_accessions.index)]
print('Records after dropping conflicting TRANS_CODE:', len(df_filtered))


# Let's combine the transactions that are part of the same submision (same ACCESSION_NUMBER)

# In[24]:


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
    'TRANS_DATE': 'last'
}).reset_index()

print('Shape after combining:', df_combined.shape)
df_combined.head()


# In[25]:


# Drop columns that will not be used for modeling 

df_combined = df_combined.drop(columns=['SECURITY_TITLE', 'year', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE'])
df_combined = df_combined.rename(columns={'TRANS_DATE': 'date', 'TRANS_CODE': 'side'})

# Convert TRANS_CODE to buy/sell
df_combined['side'] = df_combined['side'].map({'P': 'buy', 'S': 'sell'})

print('Columns after dropping:', df_combined.columns.tolist())
df_combined.head()


# ## Visualizations
# Below we plot:
# - Buy vs Sell transaction counts
# - Distribution of dollar values (log scale to handle skew)
# These plots help visually confirm patterns seen in the summary statistics.

# In[27]:


plt.figure()
dv_log = np.log10(df_combined['dollar_value'].dropna() + 1)    # log base 20
#dv_log = np.log1p(df_combined['dollar_value'].dropna() + 1)     # natural log
dv_log.plot(kind='hist', bins=50)
plt.title('Distribution of Dollar Value (log10 scale)')
plt.xlabel('log10(dollar_value + 1)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ## Log Transformation for dollar_value

# In[28]:


df_combined["log_dollar_value"] = np.log1p(df_combined["dollar_value"].clip(lower=0))
df_combined.head()


# ## Bringing Role from REPORTINGOWNER.tsv

# In[29]:


REPORTINGOWNER_PATH = r"../REPORTINGOWNER.tsv"  # <-- update this

df_owner = pd.read_csv(REPORTINGOWNER_PATH, sep='\t', low_memory=False)
df_owner.head()


# In[30]:


print('Number of unique ACCESSION_NUMBER in df_owner:', df_owner['ACCESSION_NUMBER'].nunique())
print(f'Total rows {len(df_owner)}')


# Let's merge rows with the same id and also simplify the RPTOWNER_RELATIONSHIP to have only four possible values

# In[31]:


def get_simplified_role(rel):
    if not isinstance(rel, str):
        return 'OTHER'
    r = rel.upper()
    if 'OFFICER' in r:
        return 'OFFICER'
    elif 'DIRECTOR' in r:
        return 'DIRECTOR'
    elif 'TENPERCENTOWNER' in r or '10% OWNER' in r:
        return 'TENPERCENTOWNER'
    else:
        return 'OTHER'

# Apply simplification
df_owner['simplified'] = df_owner['RPTOWNER_RELATIONSHIP'].apply(get_simplified_role)

# Assign sort key for precedence: OFFICER > DIRECTOR > TENPERCENTOWNER > OTHER
sort_order = {'OFFICER': 1, 'DIRECTOR': 2, 'TENPERCENTOWNER': 3, 'OTHER': 4}
df_owner['sort_key'] = df_owner['simplified'].map(sort_order)

# Sort by ACCESSION_NUMBER and sort_key, then drop duplicates keeping the first (highest precedence)
df_owner = df_owner.sort_values(['ACCESSION_NUMBER', 'sort_key']).drop_duplicates('ACCESSION_NUMBER', keep='first')

# Update the column and drop temp columns
df_owner['RPTOWNER_RELATIONSHIP'] = df_owner['simplified']
df_owner = df_owner.drop(columns=['simplified', 'sort_key'])

print('Shape after merging and simplifying:', df_owner.shape)
print('Updated unique RPTOWNER_RELATIONSHIP values:')
print(df_owner['RPTOWNER_RELATIONSHIP'].value_counts())
df_owner.head()


# In[32]:


df_join = df_combined.merge(df_owner[['ACCESSION_NUMBER', 'RPTOWNER_RELATIONSHIP']], on='ACCESSION_NUMBER', how='inner')
df_join = df_join.rename(columns={'RPTOWNER_RELATIONSHIP': 'role'})
print('Shape of df_join:', df_join.shape)
df_join.head()


# ## Bringing ticker from SUBMISSION.tsv

# In[33]:


SUBMISSION_PATH = r"../SUBMISSION.tsv"  # <-- update this

df_submission = pd.read_csv(SUBMISSION_PATH, sep='\t', low_memory=False)
df_submission = df_submission[['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL', 'AFF10B5ONE']]
print(len(df_submission))
df_submission.head()


# Let's clean the data

# In[34]:


# Handle AFF10B5ONE values
if 'AFF10B5ONE' in df_submission.columns:
    # Replace string/boolean values
    df_submission['AFF10B5ONE'] = df_submission['AFF10B5ONE'].replace({'false': 0, 'true': 1, False: 0, True: 1})

    # Drop rows where AFF10B5ONE is NaN
    df_submission = df_submission.dropna(subset=['AFF10B5ONE'])

    # Ensure it's integer type
    df_submission['AFF10B5ONE'] = df_submission['AFF10B5ONE'].astype(int)

    print('Updated AFF10B5ONE value counts:')
    print(df_submission['AFF10B5ONE'].value_counts())
    print('Shape after processing:', df_submission.shape)
else:
    print('AFF10B5ONE column not found in df_submission.')


# In[35]:


# Remove rows with missing ticker
df_submission = df_submission.dropna(subset=['ISSUERTRADINGSYMBOL'])
print('Shape after dropping rows with missing ISSUERTRADINGSYMBOL:', df_submission.shape)

"""
missing_tickers = df_submission["ISSUERTRADINGSYMBOL"].isnull().sum()
total_rows = len(df_submission)
missing_pct = (missing_tickers / total_rows * 100).round(2) if total_rows > 0 else 0

print(f"Total rows: {total_rows}")
print(f"Missing tickers: {missing_tickers} ({missing_pct}%)")

if missing_tickers > 0:
    print("Sample rows with missing tickers:")
    display(df_submission[df_submission["ticker"].isnull()].head())

"""


# In[36]:


df_join = df_join.merge(df_submission[['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL', 'AFF10B5ONE']], on='ACCESSION_NUMBER', how='inner')
df_join = df_join.rename(columns={'ISSUERTRADINGSYMBOL': 'ticker', 'AFF10B5ONE': 'is_10b5_1'})
print('Shape of df_join:', df_join.shape)


# In[37]:


df_join.head()


# In[38]:


value_counts = df_join['is_10b5_1'].value_counts()
percentages = (value_counts / len(df_submission)) * 100
print('Percentages:')
print(percentages.round(2))


# In[39]:


# Number of unique tickers
value_counts = df_join['ticker'].value_counts()
print(value_counts)


# ## Last check for missing values before storing data in Parquet

# In[40]:


missing = df_join.isnull().sum()
missing_pct = (missing / len(df_join) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})

print("Missing values in df_join:")
print(missing_df)


# In[42]:


df_join.to_parquet("data/raw/events_2025Q1.parquet") 


# In[ ]:




