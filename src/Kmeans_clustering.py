import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# K-means clustering for the unsupervised learning part.
# Input file is produced by preprocessing.py: data/heart_cleaned.csv

df = pd.read_csv("data/heart_cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

print("=== Loaded Cleaned Dataset ===")
print(f"Total data objects : {len(df)}")
print(f"Features used      : {list(X.columns)}")
print(f"Feature count      : {X.shape[1]}")
print()


# K-means is distance-based, so normalisation is required.
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = MinMaxScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])

print("=== Normalisation Applied ===")
print("Continuous features normalised:")
for feature in continuous_features:
    print(f"  {feature}")
print()


experiments = [
    {
        "name": "Experiment 1",
        "purpose": "Baseline clustering with k=2, matching the binary target structure",
        "params": dict(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        ),
    },
    {
        "name": "Experiment 2",
        "purpose": "Testing k=3 to check whether three natural groups exist",
        "params": dict(
            n_clusters=3,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        ),
    },
    {
        "name": "Experiment 3",
        "purpose": "Testing k=4 to check whether more detailed grouping improves clustering quality",
        "params": dict(
            n_clusters=4,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        ),
    },
]


print("=" * 70)
print("K-MEANS CLUSTERING EXPERIMENTS")
print("=" * 70)

results = []
trained_models = {}

for exp in experiments:
    name = exp["name"]
    params = exp["params"]

    model = KMeans(**params)
    cluster_labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)

    trained_models[name] = model

    result_row = {
        "Experiment": name,
        "n_clusters": params["n_clusters"],
        "init": params["init"],
        "n_init": params["n_init"],
        "max_iter": params["max_iter"],
        "Inertia": round(inertia, 4),
        "Silhouette Score": round(silhouette, 4),
        "Adjusted Rand Index": round(ari, 4),
    }

    results.append(result_row)

    print(f"\n=== {name}: {exp['purpose']} ===")
    print("Hyperparameters:")
    for k, v in params.items():
        print(f"  {k:15s}: {v}")

    print("\nClustering metrics:")
    print(f"  Inertia             : {inertia:.4f}")
    print(f"  Silhouette Score    : {silhouette:.4f}")
    print(f"  Adjusted Rand Index : {ari:.4f}")

    print("\nCluster distribution:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(
            f"  Cluster {cluster}: {count} objects "
            f"({count / len(cluster_labels) * 100:.1f}%)"
        )


print("\n\n=== EXPERIMENT SUMMARY TABLE ===")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
print()


# Best model selection.
# Silhouette Score is used because K-means is unsupervised.
best_result = max(results, key=lambda r: r["Silhouette Score"])
best_exp_name = best_result["Experiment"]
best_model = trained_models[best_exp_name]

print(f"=== Best K-Means Model Selected: {best_exp_name} ===")
print("Selection criterion : highest Silhouette Score")
print(f"Best Silhouette     : {best_result['Silhouette Score']:.4f}")
print(f"Number of clusters  : {best_result['n_clusters']}")
print()


best_labels = best_model.predict(X_scaled)

print("=" * 70)
print(f"FINAL K-MEANS RESULTS — {best_exp_name}")
print("=" * 70)

comparison_df = pd.DataFrame({
    "Cluster": best_labels,
    "Actual target": y
})

print("\n=== Cluster vs Actual Target Table ===")
cluster_target_table = pd.crosstab(
    comparison_df["Cluster"],
    comparison_df["Actual target"],
    rownames=["Cluster"],
    colnames=["Actual target"]
)
print(cluster_target_table.to_string())
print()


# PCA is only for possible 2D visualisation in the report.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": best_labels,
    "Actual target": y
})

print("=== PCA Representation Data Snapshot ===")
print(pca_df.head().to_string(index=False))
print()


print("=== Interpretation ===")
print(
    f"The best K-means model was {best_exp_name} with k={best_result['n_clusters']}. "
    f"It achieved a Silhouette Score of {best_result['Silhouette Score']:.4f}. "
    f"Since K-means is an unsupervised algorithm, the target variable was not used "
    f"during clustering. The target column was used only after clustering to compare "
    f"the discovered groups with the actual heart disease classes. "
    f"The Adjusted Rand Index was {best_result['Adjusted Rand Index']:.4f}."
)