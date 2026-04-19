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


# "Split the dataset into training and test datasets (mandatory!)."
# Stratified split preserves the class ratio from the full dataset in both subsets.
# 80/20 is standard for a dataset of ~280 rows.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# "Information on the test and training datasets:
# (a) the total number of data objects added to the test and training datasets (number and %)
# and (b) information on how many data objects from each class are included in the
# training and test datasets (number and %)."

total = len(df)
print("=== Train / Test Split Summary ===")
print(f"Training set : {len(X_train)} objects ({len(X_train) / total * 100:.1f}%)")
print(f"Test set     : {len(X_test)} objects  ({len(X_test) / total * 100:.1f}%)")
print()

for split_name, y_split in [("Training", y_train), ("Test", y_test)]:
    print(f"{split_name} set — class breakdown:")
    counts = y_split.value_counts().sort_index()
    for label, count in counts.items():
        meaning = "No disease" if label == 0 else "Disease present"
        print(
            f"Class {label} ({meaning}): {count} objects "
            f"({count / len(y_split) * 100:.1f}%)"
        )
print()

# Scaler is fit on training data only to prevent data leakage into the test set.
# As noted in preprocessing.py, normalisation is required due to large scale
# differences across continuous features (e.g. age [29–77] vs chol [126–564]).

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
