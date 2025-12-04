#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[13]:


# Define the file path 
path = "final_data_2025Q1.parquet"

# Load the data with fastparquet
df = pd.read_parquet(path, engine="fastparquet")

# Display the first 20 rows
df.head(20)


# In[14]:


# drop_first=True helps avoid multicollinearity (e.g., if it's not "Buy", it must be "Sell")
df_encoded = pd.get_dummies(df, columns=['side', 'role'], drop_first=True)


# # Define Features and Target

# In[15]:


# The target is what we want to predict: 'label_up_market' (Will the stock go up?)
target_col = 'label_up_market'
# X contains all the features used to make the prediction
X = df_encoded.drop(columns=[target_col])
# y contains the actual outcomes (Up Market: 1 or 0)
y = df_encoded[target_col]


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Initialize Model

# In[17]:


# Configure the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,       # Limit depth to prevent overfitting noise
    min_samples_leaf=5, # Each leaf must have at least 5 trades
    random_state=42,
    n_jobs=-1
)


# In[18]:


print("Training Random Forest...")
# Fit the model to the training data
rf_model.fit(X_train, y_train)


# In[19]:


# Predict outcomes for the test set
y_pred = rf_model.predict(X_test)


# # Evaluation

# In[20]:


print("--- Random Forest Performance ---")
# Print the accuracy score (percentage of correct guesses)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
# Show detailed metrics: Precision (accuracy of positive predictions), Recall (ability to find positives), F1-Score
print(classification_report(y_test, y_pred))


# # Calculate Feature Importance

# In[21]:


# Sort them in ascending order for plotting
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)


# In[22]:


# Plot a horizontal bar chart of feature importance
plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='teal')
plt.title("What Drives Insider Trading Signals? (Feature Importance)")
plt.xlabel("Importance Score")
plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script RF.ipynb')


# In[ ]:




