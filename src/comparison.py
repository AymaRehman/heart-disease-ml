import os
import sys

import pandas as pd


# Cross-algorithm comparison for Part III of the assignment.

# "Results of the testing of the trained models and a comparison and
# interpretation of their performance, clearly separated from the training
# experiments."

# This script loads the best test results from ann_classification.py and classification.py,
# merges them into a single table, and prints a summary comparison.
# The final table is also saved as a CSV for reference in the report.

OUTPUT_DIR = "outputs"

ann_path = os.path.join(OUTPUT_DIR, "ann_best_test_results.csv")
classification_path = os.path.join(OUTPUT_DIR, "classification_best_test_results.csv")

missing = [p for p in [ann_path, classification_path] if not os.path.exists(p)]
if missing:
    print("ERROR: required input files are missing:")
    for p in missing:
        print(f"  {p}")
    print(
        "\nPlease run src/ann_classification.py and src/classification.py first, "
        "then re-run this script."
    )
    sys.exit(1)


print("=== Loading Best-Model Test Results ===")
ann_df = pd.read_csv(ann_path)
classification_df = pd.read_csv(classification_path)
print(f"ANN results loaded from           : {ann_path}")
print(f"KNN/DT results loaded from        : {classification_path}")
print()


final_comparison_df = pd.concat([ann_df, classification_df], ignore_index=True)


print("=" * 70)
print("FINAL CROSS-ALGORITHM TEST RESULTS (ANN vs KNN vs Decision Tree)")
print("=" * 70)
print()
print(final_comparison_df.to_string(index=False))
print()

final_path = os.path.join(OUTPUT_DIR, "final_model_comparison.csv")
final_comparison_df.to_csv(final_path, index=False)
print(f"Final comparison table saved as: {final_path}")
print()

print("=== Comparison Summary ===")

best_by_accuracy = final_comparison_df.loc[
    final_comparison_df["Test Accuracy"].idxmax()
]
best_by_f1 = final_comparison_df.loc[final_comparison_df["Test F1"].idxmax()]
best_by_recall = final_comparison_df.loc[final_comparison_df["Test Recall"].idxmax()]

print(
    f"Highest Test Accuracy : {best_by_accuracy['Algorithm']} "
    f"({best_by_accuracy['Best Experiment']}) - {best_by_accuracy['Test Accuracy']:.4f}"
)
print(
    f"Highest Test F1       : {best_by_f1['Algorithm']} "
    f"({best_by_f1['Best Experiment']}) - {best_by_f1['Test F1']:.4f}"
)
print(
    f"Highest Test Recall   : {best_by_recall['Algorithm']} "
    f"({best_by_recall['Best Experiment']}) - {best_by_recall['Test Recall']:.4f}"
)
print()
