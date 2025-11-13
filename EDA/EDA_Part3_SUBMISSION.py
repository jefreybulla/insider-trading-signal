#!/usr/bin/env python
# coding: utf-8

# # Part 3 – SUBMISSION.tsv Exploratory Data Analysis & Summary
# 
# This notebook analyzes **SUBMISSION.tsv**, with a focus on:
# - Basic structure and key fields (e.g., `ACCESSION_NUMBER`, `CIK`, `PERIOD_OF_REPORT`)
# - Unique filers and filing activity
# - Presence/absence of **Rule 10b5-1 trading plans** (if a corresponding field exists)

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
# Update `SUBMISSION_PATH` as needed for your environment.

# In[3]:


SUBMISSION_PATH = r"/Users/aaniaadap/Desktop/KDDM Project/Dataset/SUBMISSION.tsv"  # <-- update this

df_submission = pd.read_csv(SUBMISSION_PATH, sep='\t', low_memory=False)
df_submission.head()


# ## Basic info and missing values
# We inspect dimensions, data types, and missingness to understand submission-level data quality.

# In[4]:


print('Shape:', df_submission.shape)
df_submission.dtypes


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

# In[8]:


df_submission['PERIOD_OF_REPORT'].value_counts().sort_index()


# In[11]:


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


# In[13]:


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

# In[7]:


if 'PERIOD_OF_REPORT' in df_submission.columns:
    plt.figure()
    df_submission['PERIOD_OF_REPORT'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Filings by Year (SUBMISSION.tsv)')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# ## 4. 10b5-1 plan indicator (if available)
# If the dataset contains a column indicating Rule 10b5-1 trading plans (e.g., `IS_10B5_1` or similar), we can compute the share of filings with vs. without such a plan.

# In[8]:


plan_cols = [c for c in df_submission.columns if '10b5' in c.lower() or '10b5-1' in c.lower()]
plan_cols


# In[9]:


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


# ## 5. Combined high-level summary
# - **NONDERIV_TRANS.tsv**: Transaction-level data (buy/sell, shares, prices, dollar values) for common stock insider trades.
# - **REPORTINGOWNER.tsv**: Describes who the reporters are and their roles (officers, directors, large owners).
# - **SUBMISSION.tsv**: Captures filing-level metadata, linking transactions and owners at the form level.
# 
