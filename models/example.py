#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


df = pd.read_parquet("../ETL/data/engineered/final_data_2025Q1.parquet")
df.head(20)

