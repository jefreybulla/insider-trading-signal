#!/usr/bin/env python
# coding: utf-8

# # Part 3 – SUBMISSION.tsv Exploratory Data Analysis & Summary
# 
# This notebook analyzes **SUBMISSION.tsv**, with a focus on:
# - Basic structure and key fields (e.g., `ACCESSION_NUMBER`, `CIK`, `PERIOD_OF_REPORT`)
# - Unique filers and filing activity
# - Presence/absence of **Rule 10b5-1 trading plans** (if a corresponding field exists)

# In[1]:


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
# Update `SUBMISSION_PATH` as needed for your environment.

# In[2]:


SUBMISSION_PATH = r"../SUBMISSION.tsv"  # <-- update this

df_submission = pd.read_csv(SUBMISSION_PATH, sep='\t', low_memory=False)
df_submission.head()


# In[ ]:





# ## Basic info and missing values
# We inspect dimensions, data types, and missingness to understand submission-level data quality.

# In[3]:


print('Shape:', df_submission.shape)
df_submission.dtypes


# In[4]:


# Find rows with duplicate ACCESSION_NUMBER
duplicates = df_submission[df_submission.duplicated('ACCESSION_NUMBER', keep=False)]

if not duplicates.empty:
    # Pick the first duplicate ACCESSION_NUMBER as a sample
    sample_acc = duplicates['ACCESSION_NUMBER'].iloc[0]
    sample_rows = df_submission[df_submission['ACCESSION_NUMBER'] == sample_acc]

    print(f'Sample ACCESSION_NUMBER with duplicates: {sample_acc}')
    print(f'Number of rows for this ACCESSION_NUMBER: {len(sample_rows)}')
    display(sample_rows)
else:
    print('No duplicate ACCESSION_NUMBER found in df_submission.')


# In[5]:


missing = df_submission.isnull().sum()
missing_pct = (missing / len(df_submission) * 100).round(2)
pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})


# ## Filing activity overview
# We examine key identifiers such as `CIK` and filing dates to understand how active different filers are.

# In[6]:


if 'CIK' in df_submission.columns:
    unique_ciks = df_submission['CIK'].nunique()
    print('Unique CIKs:', unique_ciks)
    top_ciks = df_submission['CIK'].value_counts().head(10)
    display(top_ciks)

if 'PERIOD_OF_REPORT' in df_submission.columns:
    df_submission['PERIOD_OF_REPORT'] = pd.to_datetime(df_submission['PERIOD_OF_REPORT'], errors='coerce')
    filings_by_year = df_submission['PERIOD_OF_REPORT'].dt.year.value_counts().sort_index()
    display(filings_by_year)


# **PERIOD_OF_REPORT ≠ filing date.** 
# - It refers to the actual period the transaction occurred, not when the filing was submitted. So a form filed in early 2025 can still report insider trades that happened in late 2024 or earlier.
# 
# **Back-dated or amended filings.** 
# - Companies sometimes file amendments or delayed disclosures (Form 4/A, Form 5) that reference older transaction periods, which explains 2023 or 2024 dates.
# 
# **Carry-over data from merged sources.** 
# - If your dataset combines multiple SEC feeds (non-derivative, derivative, reporting-owner files), older report periods can appear even though the data was extracted during 2025 Q1.

# In[7]:


df_submission['PERIOD_OF_REPORT'].value_counts().sort_index()


# In[8]:


import re

def detect_format(val):
    s = str(val).strip()
    if re.fullmatch(r"\d{4}", s):
        return "YYYY"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "YYYY-MM-DD"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return "MM/DD/YYYY"
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):
        return "DD-MM-YYYY"
    if re.fullmatch(r"\d{8}", s):
        return "YYYYMMDD"
    return "OTHER/WEIRD"

df_submission["POR_format"] = df_submission["PERIOD_OF_REPORT"].apply(detect_format)

print(df_submission["POR_format"].value_counts(dropna=False))
print(df_submission[df_submission["POR_format"] == "OTHER/WEIRD"]["PERIOD_OF_REPORT"].unique()[:50])


# In[9]:


# Number of unique report dates in 2025
df_submission['PERIOD_OF_REPORT'].dt.year.value_counts().sort_index()

# How many unique 2025 report dates exist
df_2025 = df_submission[df_submission['PERIOD_OF_REPORT'].dt.year == 2025]
print("Unique dates in 2025:", df_2025['PERIOD_OF_REPORT'].nunique())

# Top 10 most frequent report dates
print(df_2025['PERIOD_OF_REPORT'].value_counts().head(10))


# - The PERIOD_OF_REPORT column is in a valid datetime format — there are no formatting or parsing errors.
# 
# - The data for 2025 covers only Q1, with 90 unique report dates in total.
# 
# - Filings are heavily concentrated on a few specific dates — e.g., Jan 2 (2522), Mar 3 (2090), and Feb 28 (1973) — which is unusually high for typical insider trading volumes.
# 
# - This suggests that the spike in 2025 filings is not due to increased activity, but rather repeated or expanded entries in the dataset.
# 
# - The duplication likely comes from multiple transaction-level rows per filing (e.g., one per security, derivative, or issuer), or merged data sources causing repetition.
# 
# - Therefore, the 2025 Q1 surge represents data duplication or reporting granularity, not an actual rise in insider trading.

# In[10]:


if 'PERIOD_OF_REPORT' in df_submission.columns:
    plt.figure()
    df_submission['PERIOD_OF_REPORT'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Filings by Year (SUBMISSION.tsv)')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# In[11]:


# Drop non 2025 rows
df_submission = df_submission[df_submission['PERIOD_OF_REPORT'].dt.year == 2025]
print('Shape after dropping non-2025 rows:', df_submission.shape)


# ## 4. 10b5-1 plan indicator (if available)
# If the dataset contains a column indicating Rule 10b5-1 trading plans (e.g., `IS_10B5_1` or similar), we can compute the share of filings with vs. without such a plan.

# In[12]:


plan_cols = [c for c in df_submission.columns if '10b5' in c.lower() or '10b5-1' in c.lower()]
plan_cols


# In[13]:


if plan_cols:
    col = plan_cols[0]
    print('Using column as 10b5-1 indicator:', col)
    value_counts = df_submission[col].value_counts(dropna=False)
    display(value_counts)

    plt.figure()
    value_counts.plot(kind='bar')
    plt.title('10b5-1 Plan Indicator Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
else:
    print('No explicit 10b5-1 indicator column detected.')


# ## Handle AFF10B5ONE values

# In[14]:


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


# In[15]:


len(df_submission)


# ## Remove rows with missing ISSUERTRADINGSYMBOL

# In[ ]:


df_submission = df_submission.dropna(subset=['ISSUERTRADINGSYMBOL'])
print('Shape after dropping rows with missing ISSUERTRADINGSYMBOL:', df_submission.shape)

