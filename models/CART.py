#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[13]:


# Define the file path 
path = "final_data_2025Q1.parquet"

# Load the data with fastparquet
df = pd.read_parquet(path, engine="fastparquet")

# Display the first 20 rows
df.head(20)


# In[14]:


# Convert categorical columns ('side' and 'role') into numeric dummy variables (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['side', 'role'], drop_first=True)


# # Define Features and Target

# In[15]:


# Define the target variable
target_col = 'label_up_market'
# Create the Feature Matrix (X) by dropping the target column
X = df_encoded.drop(columns=[target_col])

# Create the Target Vector (y)
y = df_encoded[target_col]


# # Split Data

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Model Training

# In[17]:


clf = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=4) 
clf.fit(X_train, y_train)


# In[18]:


# Generate predictions on the unseen testing data
y_pred = clf.predict(X_test)


# # Evaluation

# In[19]:


print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[20]:


plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Down/Flat', 'Up Market'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("CART Decision Logic for Insider Trading")
plt.show()


# In[21]:


# Calculate feature importance scores and creates a sorted Series
importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
# Print the features ranked by their influence on the prediction
print("\n--- Feature Importance ---")
print(importance)


# In[22]:


# Calculate the baseline accuracy (predicting the majority class for every instance)
default_accuracy = y.value_counts().max() / len(y)
print("Default (baseline) accuracy:", default_accuracy)


# In[23]:


import os
os.rename("CART.pynub", "CART.py")


# In[ ]:




