#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score


# In[ ]:


df = pd.read_parquet("../ETL/data/engineered/final_data_2025Q1.parquet")
print(len(df))
df.head(10)
print(df.describe())


# In[ ]:





# ## Splitting data into attributes & target

# In[ ]:


attr = df.drop(columns=[ 'label_up_market'])
target = df['label_up_market']
attr.head()


# ## Normalizing/Centering data

# In[ ]:


# One-hot encode categorical columns
attr = pd.get_dummies(attr, columns=['side', 'role'], drop_first=False)

# normalize/center numeric features
# Normalize only numeric features
numeric_cols = ['log_dollar_value', 'log_size_vs_cap']
scaler = StandardScaler()
attr[numeric_cols] = scaler.fit_transform(attr[numeric_cols])

attr.head(10)


# ## Splitting data into training and testing

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(attr, target,random_state=82, test_size=0.3)
print(X_train.shape)
print(y_train.shape)


# ## Artificial Neural Network

# In[ ]:


#  one hidden layer containing 4 neurons, default to ReLU activation
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10000, random_state=82)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ## Evaluation

# In[ ]:


# Accuracy Score
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: {} ({:.2%})".format(accuracy, accuracy))

print(classification_report(y_test, y_pred, digits=3))


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f"Precision (for class 1): {precision_score(y_test, y_pred)}")
print('Classification Report')
print(classification_report(y_test, y_pred))


# # Performance
# - Accuracy: 0.642
# - Precision (class 1): 0.565
