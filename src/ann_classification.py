import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Authored: @AymaRehman

# "One of the algorithms that must be used is artificial neural networks."
# This file implements the mandatory ANN classification for Part III of the assignment.

# "This part of the assignment aims to apply at least 3 classification
# algorithms to the previously analysed dataset and features of data objects selected
# in Part I of the assignment."
# Input: heart_cleaned.csv produced by preprocessing.py (features selected in Part I).

df = pd.read_csv("data/heart_cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

print("=== Loaded Cleaned Dataset ===")
print(f"Total data objects : {len(df)}")
print(f"Features used      : {list(X.columns)}")
print(f"Feature count      : {X.shape[1]}")
print()
