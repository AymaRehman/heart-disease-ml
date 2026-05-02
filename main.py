import os
import runpy
import sys
import time

# Entry point for the full heart-disease-ml pipeline.
# Runs each stage in the correct order

# Stages:
#   1. Preprocessing                -> produces data/heart_cleaned.csv
#   2. K-Means clustering           -> Part II (unsupervised)
#   3. Hierarchical clustering      -> Part II (unsupervised)
#   4. ANN classification           -> Part III (supervised, mandatory algorithm)
#   5. KNN + Decision Tree          -> Part III (supervised, two freely chosen algorithms)
#   6. Cross-algorithm comparison   -> merges Part III test results into one table

# The EDA notebook (notebooks/exploration.ipynb) is not run from here as it is
# intended to be opened and executed interactively in Jupyter for Part I.


PIPELINE = [
    ("Preprocessing", "src/preprocessing.py"),
    ("K-Means Clustering (Part II)", "src/Kmeans_clustering.py"),
    ("Hierarchical Clustering (Part II)", "src/hierarchical_clustering.py"),
    ("ANN Classification (Part III)", "src/ann_classification.py"),
    ("KNN + Decision Tree Classification (Part III)", "src/classification.py"),
    ("Cross-algorithm Comparison (Part III)", "src/comparison.py"),
]


def banner(title):
    bar = "====================================================================="
    print(f"\n{bar}")
    print(f"{title}")
    print(f"{bar}\n")


def run_stage(title, script_path):
    banner(title)
    if not os.path.exists(script_path):
        print(f"ERROR: required script not found: {script_path}")
        sys.exit(1)

    start = time.time()
    runpy.run_path(script_path, run_name="__main__")
    elapsed = time.time() - start
    print(f"\n[{title}] completed in {elapsed:.2f}s")


def main():
    banner("HEART DISEASE ML PIPELINE")
    print("Stages to execute:")
    for i, (title, script_path) in enumerate(PIPELINE, start=1):
        print(f"{i}. {title}")
    print()
    print("Note: the EDA notebook (notebooks/exploration.ipynb) is run separately")
    print("in Jupyter for Part I and is not included in this pipeline.")

    total_start = time.time()
    for title, script_path in PIPELINE:
        run_stage(title, script_path)

    total_elapsed = time.time() - total_start
    banner("PIPELINE FINISHED")
    print(f"All stages completed successfully in {total_elapsed:.2f}s.")
    print("Outputs are saved in the 'outputs/' directory.")


if __name__ == "__main__":
    main()
