#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Model Execution

# In[2]:


# Load dataset
df = pd.read_parquet("../ETL/data/engineered/final_data_2025Q1.parquet")


# In[3]:


target = "label_up_market"
X = df.drop(columns=[target])
y = df[target]


# In[4]:


# Categorical vs Numeric columns
cat_cols = ["side", "role"]
num_cols = [c for c in X.columns if c not in cat_cols]


# In[5]:


# Preprocess: One-hot encode categoricals
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
],
remainder="passthrough")   # keep numeric columns as-is


# In[6]:


# C5.0-style tree (entropy split)
c50 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,           # similar to C5.0 settings
    min_samples_split=10,
    random_state=42
)


# In[7]:


pipe = Pipeline([
    ("prep", preprocess),
    ("c50", c50)
])


# In[8]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[9]:


# Fit model
pipe.fit(X_train, y_train)


# In[10]:


# Predictions
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test) 

# ---- Accuracy ----
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")


# In[11]:


# ---- Confusion Matrix (Test) ----
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test):")
print(cm)


# In[12]:


# ---- Classification Report (Test) ----
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))


# # **Model Interpretation & Inferences (C5.0 Decision Tree)**
# 
# ## Overall accuracy improved (0.62), but the model is biased
# 
# * The model classifies **most observations as Class 0** because:
# 
#   * Class 0 is more frequent.
#   * Decision trees (especially shallow ones) prefer splits that maximize majority purity.
#   * The minority class (1) is harder to learn due to weaker signal.
# * This creates a situation where accuracy looks acceptable, but performance on Class 1 is poor.
# 
# ---
# 
# ## Class 0 (Down/Not-Up) → Very strong performance
# 
# * **Recall = 0.92**
#   → The model detects almost every instance where the stock does *not* outperform the market.
# * This greatly boosts overall accuracy but hides weaknesses in the minority class.
# * High recall + moderate precision (0.63) = strong performance for Class 0.
# 
# ---
# 
# ## Class 1 (Up-Market) → Very weak performance
# 
# * **Recall = 0.16**
# 
#   > The model identifies only 16 out of every 100 true “up-market” cases.
# * **Precision = 0.56**
#   → When it predicts class 1, it is only moderate at being correct.
# * This makes the model **ineffective for detecting positive market movements**, which is often the primary use case in insider trading prediction tasks.
# 
# ---
# 
# ## Why recall is so low for Class 1
# 
# * **Data imbalance**: Class 0 significantly outnumbers Class 1.
# * **Limited feature strength**: Insider trade metadata alone (side, role, size) does not fully explain price movement.
# * **Shallow tree depth (max_depth=4)** leads to underfitting and oversimplified decision boundaries.
# * **Minority class patterns are noisy**, making them harder to separate using entropy-based splits.
# 
# ---
# 
# ## Why entropy-based C5.0 trees behave this way
# 
# * C5.0 uses **entropy (information gain)** to select splits.
# * Information gain tends to favor splits that:
# 
#   * Maximize purity of the large class
#   * Reduce impurity where the majority dominates
# * When the signal is weak and the dataset is imbalanced, the algorithm naturally becomes **biased toward the majority class**.
# * As a result, minority-class recall becomes extremely low.
# 
