import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    print(f"{split_name} set - class breakdown:")
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


# "Perform at least 3 experiments with each algorithm using the
# training dataset, changing the values of the algorithm hyperparameters and
# analysing the performance metrics of the algorithms."

# Each experiment has 1 clear, isolated purpose:

#   Experiment 1 : Baseline (small shallow network)
#   One hidden layer of 32 neurons, alpha=0.0001 (minimal regularisation).
#   Establishes a lower-bound reference for a simple ANN on this dataset.

#   Experiment 2 : Effect of stronger regularisation
#   Same architecture as Exp. 1, but alpha raised from 0.0001 to 0.1.
#   Tests whether heavily penalising large weights improves generalisation
#   on this small dataset.

#   Experiment 3 : Effect of added depth
#   Two hidden layers (64, 32), alpha=0.01 (moderate regularisation).
#   Tests whether depth helps the model learn more complex non-linear
#   interactions compared to the shallow baseline.

# Shared across all 3:
# activation=relu, solver=adam, learning_rate_init=0.001, max_iter=3000, random_state=42.

# relu controls which neurons deliver info and which stay silent,
# learning rate is the pace at which the model learns,
# Adam maintains understanding of context to adapt to situations dynamically

# "Hyperparameter values used in the experiments with each of the
# algorithms (in a table format) and screenshots showing these values and the
# performance metrics of the experiments."

experiments = [
    {
        "name": "Experiment 1",
        "purpose": "Baseline - shallow network (32 neurons), minimal regularisation",
        "params": dict(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=3000,
            random_state=42,
        ),
    },
    {
        "name": "Experiment 2",
        "purpose": "Strong regularisation - same architecture, alpha raised to 0.1",
        "params": dict(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.1,
            learning_rate_init=0.001,
            max_iter=3000,
            random_state=42,
        ),
    },
    {
        "name": "Experiment 3",
        "purpose": "Deeper network - two hidden layers (64, 32), moderate regularisation",
        "params": dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=3000,
            random_state=42,
        ),
    },
]


def evaluate_model(model, X, y):
    """Return a dict of key classification metrics."""
    y_pred = model.predict(X)
    return {
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y, y_pred, zero_division=0), 4),
    }


print("======================================================================")
print("PART III - ANN EXPERIMENTS (training data, 5-fold cross-validation)")
print("======================================================================")

results = []
trained_models = {}

for exp in experiments:
    name = exp["name"]
    params = exp["params"]

    model = MLPClassifier(**params)

    # Cross-validation on training set only - test set not touched here
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="accuracy"
    )

    # Fit on full training set to get train-set metrics for the report table
    model.fit(X_train_scaled, y_train)
    train_metrics = evaluate_model(model, X_train_scaled, y_train)
    trained_models[name] = model

    result_row = {
        "Experiment": name,
        "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
        "activation": params["activation"],
        "alpha": params["alpha"],
        "learning_rate_init": params["learning_rate_init"],
        "max_iter": params["max_iter"],
        "CV Acc (mean)": round(cv_scores.mean(), 4),
        "CV Acc (std)": round(cv_scores.std(), 4),
    }

    for key, value in train_metrics.items():
        new_key = f"Train {key}"
        result_row[new_key] = value

    results.append(result_row)

    print(f"\n=== {name} : {exp['purpose']} ===")
    print("Hyperparameters:")
    for key, value in params.items():
        print(f"  {key:25s}: {value}")
    print(f"\nCV accuracy (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Training-set metrics :")
    for key, value in train_metrics.items():
        print(f"  {key:12s}: {value:.4f}")


# Summary table (need to screenshot this for the report)
print("\n\n=== EXPERIMENT SUMMARY TABLE ===")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
print()

# "For each algorithm, select the trained model that provides
# the best algorithm performance."
# Criterion: highest mean CV accuracy reflects generalisation, not just training fit.

best_result = max(results, key=lambda r: r["CV Acc (mean)"])
best_exp_name = best_result["Experiment"]
best_model = trained_models[best_exp_name]
best_params = next(e["params"] for e in experiments if e["name"] == best_exp_name)

print(f"=== Best Model Selected: {best_exp_name} ===")
print("Selection criterion : highest mean CV accuracy")
print(f"Best CV accuracy    : {best_result['CV Acc (mean)']:.4f}")
print(f"Hyperparameters     : {best_params}")
print()

# "Conclusions on the performance of the models in the experiments
# performed, clearly identifying the model that will be used for testing."
print("=== Conclusions from Training Experiments ===")
for r in results:
    gap = round(r["Train Accuracy"] - r["CV Acc (mean)"], 4)
    print(
        f"  {r['Experiment']}: CV Acc = {r['CV Acc (mean)']:.4f} ± {r['CV Acc (std)']:.4f} | "
        f"Train Acc = {r['Train Accuracy']:.4f} | Train-CV gap = {gap:.4f} | "
        f"Train F1 = {r['Train F1']:.4f}"
    )
print(
    f"\n  {best_exp_name} is selected for testing: it achieved the highest mean "
    f"CV accuracy, indicating the best generalisation to unseen folds. "
    f"The Train-CV gap column above shows how much each model overfit its training data - "
    f"a smaller gap with a higher CV accuracy is the ideal combination."
)
print()

# "Apply the trained model for each algorithm to the test dataset."
# "Results of the testing of the trained models and a comparison and
# interpretation of their performance, clearly separated from the training experiments."

print("======================================================================")
print(f"TESTING RESULTS - {best_exp_name} applied to held-out test set")
print("======================================================================")

test_metrics = evaluate_model(best_model, X_test_scaled, y_test)
y_pred_test = best_model.predict(X_test_scaled)

print("\nTest-set performance metrics:")
for key, value in test_metrics.items():
    print(f"  {key:12s}: {value:.4f}")

print("\nConfusion Matrix (rows = actual, cols = predicted):")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: No disease", "Actual: Disease"],
    columns=["Predicted: No disease", "Predicted: Disease"],
)
print(cm_df.to_string())

print("\nFull Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_test,
        target_names=["No disease (0)", "Disease present (1)"],
    )
)

# "Evaluate and compare the performance of the trained models."
# Cross-algorithm comparison (ANN vs the two other classifiers) is in classification.py.
print("=== Interpretation ===")
print(
    f"The best ANN model ({best_exp_name}) achieved a test accuracy of "
    f"{test_metrics['Accuracy']:.4f} and F1-score of {test_metrics['F1']:.4f} "
    f"on the held-out test set.\n"
    f"These results will be compared against the two other classifiers "
    f"in classification.py."
)
