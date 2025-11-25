#!/usr/bin/env python
# coding: utf-8

# # Part 2 – REPORTINGOWNER.tsv Exploratory Data Analysis
# 
# This notebook analyzes **REPORTINGOWNER.tsv** to understand who the reporting owners are and how they are categorized.
# Key goals:
# - Inspect structure and missingness
# - Analyze distribution of `RPTOWNER_RELATIONSHIP`
# - Group roles into simplified categories (e.g., Officer, Director, 10% Owner)
# - Visualize the composition of reporting owners

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ## 1. Load data
# Update `REPORTINGOWNER_PATH` as needed for your environment.

# In[70]:


REPORTINGOWNER_PATH = r"../REPORTINGOWNER.tsv"  # <-- update this

df_owner = pd.read_csv(REPORTINGOWNER_PATH, sep='\t', low_memory=False)
df_owner.head(15)


# ## 2. Basic info and missing values
# We first examine the shape, data types, and missing values to assess data quality.

# In[71]:


print('Shape:', df_owner.shape)
df_owner.dtypes


# In[72]:


missing = df_owner.isnull().sum()
missing_pct = (missing / len(df_owner) * 100).round(2)
pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})


# ## Relationship type distribution
# We look at `RPTOWNER_RELATIONSHIP` to understand what kinds of roles are most common among reporting owners.

# In[73]:


rel_col = 'RPTOWNER_RELATIONSHIP'
relationship_counts = df_owner[rel_col].value_counts(dropna=False)
relationship_counts.head(20)


# ### Simplified role categories
# We map the detailed relationship strings into broader categories, such as:
# - **Officer Only**
# - **Director Only**
# - **10% Owner Only**
# - **Officer & Director**
# - **Other / Multiple**
# 
# This helps interpret the mix of insider types.

# In[74]:


def categorize_role(rel: str) -> str:
    if not isinstance(rel, str):
        return 'Unknown'
    r = rel.upper()
    is_officer = 'OFFICER' in r
    is_director = 'DIRECTOR' in r
    is_owner = '10%' in r or '10% OWNER' in r

    if is_officer and not is_director and not is_owner:
        return 'Officer Only'
    if is_director and not is_officer and not is_owner:
        return 'Director Only'
    if is_owner and not is_officer and not is_director:
        return '10% Owner Only'
    if is_officer and is_director and not is_owner:
        return 'Officer & Director'
    if is_owner and (is_officer or is_director):
        return 'Owner & Officer/Director'
    return 'Other/Multiple'

df_owner['role_category'] = df_owner[rel_col].apply(categorize_role)
simplified_counts = df_owner['role_category'].value_counts()
simplified_counts


# ### Detailed role frequency (handling multi-role entries)
# 
# We now break down `RPTOWNER_RELATIONSHIP` into individual roles.
# This helps us see how often labels like "OFFICER", "DIRECTOR",
# "10% OWNER", etc. appear across all reporting owners,
# even when multiple roles are combined in one field.
# 

# In[75]:


# Ensure relationship column is string and uppercased
df_owner['RPTOWNER_RELATIONSHIP'] = df_owner['RPTOWNER_RELATIONSHIP'].fillna('').str.upper()

# Split on common delimiters (comma, semicolon, slash)
df_owner['role_list'] = df_owner['RPTOWNER_RELATIONSHIP'].str.split(r'[,;/]')

# Explode into one role per row
roles_exploded = df_owner.explode('role_list')
roles_exploded['role_list'] = roles_exploded['role_list'].str.strip()
roles_exploded = roles_exploded[roles_exploded['role_list'] != '']

# Count individual role tokens
role_counts = roles_exploded['role_list'].value_counts().head(20)
role_counts


# In[76]:


plt.figure()
role_counts.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 20 Individual Role Mentions (Exploded)')
plt.xlabel('Count')
plt.ylabel('Role')
plt.tight_layout()
plt.show()


# ### Visualizations
# We visualize:
# - Top relationship types
# - Distribution of simplified role categories

# In[77]:


plt.figure()
relationship_counts.head(10).plot(kind='barh')
plt.title('Top 10 Reporting Owner Relationships')
plt.xlabel('Count')
plt.ylabel('Relationship Type')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure()
simplified_counts.plot(kind='bar')
plt.title('Simplified Role Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Handling multiple entries per subimission and simplifying RPTOWNER_RELATIONSHIP

# In[78]:


print('Number of unique ACCESSION_NUMBER in df_owner:', df_owner['ACCESSION_NUMBER'].nunique())
print(f'Total rows {len(df_owner)}')


# There are a few thosand rows with non-unique ids. Let's merge them. 
# We will also simplify the RPTOWNER_RELATIONSHIP to have only four possible values

# In[79]:


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

