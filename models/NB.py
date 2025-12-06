#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_parquet("../ETL/data/engineered/final_data_2025Q1.parquet")
df.head()


# In[ ]:


print(df[df.isna().any(axis=1)]) # shows NaN values in data frame if it exists


# In[ ]:


# Calculate percentage of 1 and 0 values for label_up_market
percentages = df['label_up_market'].value_counts(normalize=True) * 100
print(percentages)


# In[ ]:


# Print data types of each column using row 1
print("Data types of each column:")
print(df.dtypes)


# In[ ]:


attr = df.drop(columns=[ 'label_up_market'])
target = df['label_up_market']
attr.head()


# One-hot encode categorical columns
attr = pd.get_dummies(attr, columns=['side', 'role'], drop_first=False)
attr.head()


# In[ ]:


attr_train, attr_test, target_train,target_test  = train_test_split(attr, target,test_size = 0.3, random_state =82 , shuffle = True)

gnb = GaussianNB()

model = gnb
# Train model
model.fit(attr_train, target_train)

# Make predictions on the test set
target_pred = model.predict(attr_test)


# In[ ]:


# Evaluate the model
accuracy = accuracy_score(target_test, target_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f"Precision (for class 1): {precision_score(target_test, target_pred)}")
print('Classification Report')
print(classification_report(target_test, target_pred))


# # Performance
# - Accuracy: 0.59
# - Precision (class 1): 0.364
