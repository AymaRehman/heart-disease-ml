import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/heart_cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

print("=== Loaded Cleaned Dataset ===")
print(f"Total data objects : {len(df)}")
print(f"Features used      : {list(X.columns)}")
print(f"Feature count      : {X.shape[1]}")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

total = len(df)

print("=== Train / Test Split Summary ===")
print(f"Training set : {len(X_train)} objects ({len(X_train) / total * 100:.1f}%)")
print(f"Test set     : {len(X_test)} objects ({len(X_test) / total * 100:.1f}%)")
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


continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = MinMaxScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(
    X_train[continuous_features]
)

X_test_scaled[continuous_features] = scaler.transform(
    X_test[continuous_features]
)

print("=== Normalisation Applied ===")
print("Continuous features normalised:")
for feature in continuous_features:
    print(f"  {feature}")
print()


def evaluate_model(model, X_data, y_data):
    """Return main classification metrics."""
    y_pred = model.predict(X_data)

    return {
        "Accuracy": round(accuracy_score(y_data, y_pred), 4),
        "Precision": round(precision_score(y_data, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_data, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_data, y_pred, zero_division=0), 4),
    }


experiments = [
    {
        "algorithm": "KNN",
        "experiment": "KNN Experiment 1",
        "purpose": "Small neighbourhood, more sensitive to local patterns",
        "model": KNeighborsClassifier(
            n_neighbors=3,
            weights="uniform",
            metric="minkowski"
        ),
        "params": {
            "n_neighbors": 3,
            "weights": "uniform",
            "metric": "minkowski",
        },
    },
    {
        "algorithm": "KNN",
        "experiment": "KNN Experiment 2",
        "purpose": "Medium neighbourhood, balanced baseline",
        "model": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski"
        ),
        "params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski",
        },
    },
    {
        "algorithm": "KNN",
        "experiment": "KNN Experiment 3",
        "purpose": "Larger neighbourhood, smoother decision boundary",
        "model": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski"
        ),
        "params": {
            "n_neighbors": 7,
            "weights": "distance",
            "metric": "minkowski",
        },
    },
    {
        "algorithm": "Decision Tree",
        "experiment": "Decision Tree Experiment 1",
        "purpose": "Shallow tree to reduce overfitting",
        "model": DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
            random_state=42
        ),
        "params": {
            "criterion": "gini",
            "max_depth": 3,
            "random_state": 42,
        },
    },
    {
        "algorithm": "Decision Tree",
        "experiment": "Decision Tree Experiment 2",
        "purpose": "Medium depth tree for more detailed decision rules",
        "model": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=42
        ),
        "params": {
            "criterion": "gini",
            "max_depth": 5,
            "random_state": 42,
        },
    },
    {
        "algorithm": "Decision Tree",
        "experiment": "Decision Tree Experiment 3",
        "purpose": "Entropy criterion with medium depth",
        "model": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            random_state=42
        ),
        "params": {
            "criterion": "entropy",
            "max_depth": 5,
            "random_state": 42,
        },
    },
]


print("=" * 70)
print("PART 3 - SUPERVISED CLASSIFICATION EXPERIMENTS")
print("=" * 70)

results = []
trained_models = {}

for exp in experiments:
    algorithm = exp["algorithm"]
    experiment_name = exp["experiment"]
    purpose = exp["purpose"]
    model = exp["model"]

    print(f"\n=== {experiment_name} ===")
    print(f"Algorithm: {algorithm}")
    print(f"Purpose  : {purpose}")
    print("Hyperparameters:")

    for param_name, value in exp["params"].items():
        print(f"  {param_name:15s}: {value}")

    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    model.fit(X_train_scaled, y_train)

    train_metrics = evaluate_model(model, X_train_scaled, y_train)

    trained_models[experiment_name] = model

    result_row = {
        "Algorithm": algorithm,
        "Experiment": experiment_name,
        "Purpose": purpose,
        "CV Accuracy Mean": round(cv_scores.mean(), 4),
        "CV Accuracy Std": round(cv_scores.std(), 4),
        "Train Accuracy": train_metrics["Accuracy"],
        "Train Precision": train_metrics["Precision"],
        "Train Recall": train_metrics["Recall"],
        "Train F1": train_metrics["F1"],
        "Hyperparameters": str(exp["params"]),
    }

    results.append(result_row)

    print(f"\nCV accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Training-set metrics:")

    for metric_name, value in train_metrics.items():
        print(f"  {metric_name:12s}: {value:.4f}")


print("\n\n=== FULL EXPERIMENT SUMMARY TABLE ===")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
print()

summary_path = os.path.join(
    OUTPUT_DIR,
    "classification_full_experiment_summary.csv"
)
summary_df.to_csv(summary_path, index=False)
print(f"Classification experiment summary saved as: {summary_path}")
print()


print("=== BEST EXPERIMENT PER ALGORITHM ===")

best_per_algorithm = []

for algorithm in summary_df["Algorithm"].unique():
    algorithm_results = summary_df[summary_df["Algorithm"] == algorithm]
    best_row = algorithm_results.loc[
        algorithm_results["CV Accuracy Mean"].idxmax()
    ]
    best_per_algorithm.append(best_row)

best_per_algorithm_df = pd.DataFrame(best_per_algorithm)

print(best_per_algorithm_df.to_string(index=False))
print()

best_per_algorithm_path = os.path.join(
    OUTPUT_DIR,
    "classification_best_per_algorithm.csv"
)
best_per_algorithm_df.to_csv(best_per_algorithm_path, index=False)
print(f"Best per algorithm table saved as: {best_per_algorithm_path}")
print()


best_result = summary_df.loc[summary_df["CV Accuracy Mean"].idxmax()]
best_experiment_name = best_result["Experiment"]
best_algorithm = best_result["Algorithm"]
best_model = trained_models[best_experiment_name]

print(f"=== Best Supervised Model Selected: {best_experiment_name} ===")
print(f"Algorithm           : {best_algorithm}")
print("Selection criterion : highest mean CV accuracy")
print(f"Best CV accuracy    : {best_result['CV Accuracy Mean']:.4f}")
print(f"Hyperparameters     : {best_result['Hyperparameters']}")
print()


print("=" * 70)
print(f"TESTING RESULTS — {best_experiment_name}")
print("=" * 70)

test_metrics = evaluate_model(best_model, X_test_scaled, y_test)
y_pred_test = best_model.predict(X_test_scaled)

print("\nTest-set performance metrics:")

for metric_name, value in test_metrics.items():
    print(f"  {metric_name:12s}: {value:.4f}")


print("\nConfusion Matrix (rows = actual, columns = predicted):")

cm = confusion_matrix(y_test, y_pred_test)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No disease", "Actual: Disease"],
    columns=["Predicted: No disease", "Predicted: Disease"]
)

print(cm_df.to_string())
print()

cm_csv_path = os.path.join(OUTPUT_DIR, "classification_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print(f"Confusion matrix table saved as: {cm_csv_path}")

cm_plot_path = os.path.join(OUTPUT_DIR, "classification_confusion_matrix.png")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No disease", "Disease"],
    yticklabels=["No disease", "Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_experiment_name}")
plt.tight_layout()
plt.savefig(cm_plot_path, dpi=300)
plt.close()

print(f"Confusion matrix plot saved as: {cm_plot_path}")
print()


print("Full Classification Report:")

report_text = classification_report(
    y_test,
    y_pred_test,
    target_names=["No disease (0)", "Disease present (1)"]
)

print(report_text)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Classification report saved as: {report_path}")


print("\n=== FINAL SUPERVISED MODEL COMPARISON TABLE ===")

final_comparison_rows = []

for _, row in best_per_algorithm_df.iterrows():
    experiment_name = row["Experiment"]
    model = trained_models[experiment_name]

    test_metrics_for_model = evaluate_model(model, X_test_scaled, y_test)

    final_comparison_rows.append(
        {
            "Algorithm": row["Algorithm"],
            "Best Experiment": experiment_name,
            "CV Accuracy Mean": row["CV Accuracy Mean"],
            "Test Accuracy": test_metrics_for_model["Accuracy"],
            "Test Precision": test_metrics_for_model["Precision"],
            "Test Recall": test_metrics_for_model["Recall"],
            "Test F1": test_metrics_for_model["F1"],
            "Hyperparameters": row["Hyperparameters"],
        }
    )

final_comparison_df = pd.DataFrame(final_comparison_rows)

print(final_comparison_df.to_string(index=False))
print()

final_comparison_path = os.path.join(
    OUTPUT_DIR,
    "classification_final_model_comparison.csv"
)
final_comparison_df.to_csv(final_comparison_path, index=False)
print(f"Final classification comparison table saved as: {final_comparison_path}")
print()


print("=== Rationale ===")
print(
    "KNN was selected because it is a simple distance-based supervised algorithm "
    "that can classify patients by comparing them with similar examples in the training set. "
    "Decision Tree was selected because it is interpretable and can represent decision rules "
    "based on medical attributes."
)
print()


print("=== Interpretation ===")
print(
    f"The best supervised classification model was {best_experiment_name} "
    f"from the {best_algorithm} algorithm. It achieved a test accuracy of "
    f"{test_metrics['Accuracy']:.4f} and an F1-score of {test_metrics['F1']:.4f}. "
    f"Two supervised algorithms were compared: KNN and Decision Tree. "
    f"For each algorithm, three experiments were performed by changing "
    f"hyperparameters. The final comparison table shows the best experiment "
    f"for each algorithm and can be compared with the ANN results from "
    f"ann_classification.py."
)