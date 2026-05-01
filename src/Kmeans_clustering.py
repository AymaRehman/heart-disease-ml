import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#2a9d8f", "#c1121f", "#457b9d", "#f4a261", "#6a4c93", "#ffb703"]


df = pd.read_csv("data/heart_cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

print("=== Loaded Cleaned Dataset ===")
print(f"Total data objects : {len(df)}")
print(f"Features used      : {list(X.columns)}")
print(f"Feature count      : {X.shape[1]}")
print()


continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

scaler = MinMaxScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])

print("=== Normalisation Applied ===")
print("Continuous features normalised:")
for feature in continuous_features:
    print(f"  {feature}")
print()


k_values = [2, 3, 4, 5, 6]

experiments = []

for i, k in enumerate(k_values, start=1):
    experiments.append(
        {
            "name": f"Experiment {i}",
            "purpose": f"Testing K-means clustering with k={k}",
            "params": dict(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=42,
            ),
        }
    )


print("=" * 70)
print("K-MEANS CLUSTERING EXPERIMENTS")
print("=" * 70)

results = []
trained_models = {}
trained_labels = {}

for exp in experiments:
    name = exp["name"]
    params = exp["params"]

    model = KMeans(**params)
    cluster_labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)

    trained_models[name] = model
    trained_labels[name] = cluster_labels

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
    for param_name, value in params.items():
        print(f"  {param_name:15s}: {value}")

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

summary_csv_path = os.path.join(OUTPUT_DIR, "kmeans_experiment_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Experiment summary saved as: {summary_csv_path}")
print()


# Plot 1: Silhouette coefficient for at least 5 different k values.
silhouette_plot_path = os.path.join(OUTPUT_DIR, "kmeans_silhouette_scores.png")

plt.figure(figsize=(8, 5))
plt.plot(
    summary_df["n_clusters"],
    summary_df["Silhouette Score"],
    marker="o",
    color=COLORS[0]
)
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(silhouette_plot_path, dpi=300)
plt.close()

print(f"Silhouette coefficient plot saved as: {silhouette_plot_path}")
print()


# Best model selection.
best_result = max(results, key=lambda r: r["Silhouette Score"])
best_exp_name = best_result["Experiment"]
best_model = trained_models[best_exp_name]
best_labels = trained_labels[best_exp_name]
best_k = best_result["n_clusters"]

print(f"=== Best K-Means Model Selected: {best_exp_name} ===")
print("Selection criterion : highest Silhouette Score")
print(f"Best Silhouette     : {best_result['Silhouette Score']:.4f}")
print(f"Number of clusters  : {best_k}")
print()


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

cluster_target_path = os.path.join(OUTPUT_DIR, "kmeans_cluster_vs_target.csv")
cluster_target_table.to_csv(cluster_target_path)
print(f"Cluster vs target table saved as: {cluster_target_path}")
print()


# PCA for 2D scatterplot.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": best_labels,
    "Actual target": y
})

pca_csv_path = os.path.join(OUTPUT_DIR, "kmeans_pca_best_clusters.csv")
pca_df.to_csv(pca_csv_path, index=False)
print(f"PCA cluster data saved as: {pca_csv_path}")


# Plot 2: Scatterplot according to best Silhouette coefficient.
scatter_path = os.path.join(OUTPUT_DIR, "kmeans_best_clusters_scatter.png")

plt.figure(figsize=(8, 6))

for cluster in sorted(pca_df["Cluster"].unique()):
    cluster_data = pca_df[pca_df["Cluster"] == cluster]
    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}",
        color=COLORS[cluster % len(COLORS)],
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4
    )

plt.title(f"K-Means Best Clusters Scatter Plot (k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(scatter_path, dpi=300)
plt.close()

print(f"Best-cluster scatter plot saved as: {scatter_path}")


# Plot 3: Per-sample silhouette plot for best k.
sample_silhouette_values = silhouette_samples(X_scaled, best_labels)

silhouette_samples_path = os.path.join(
    OUTPUT_DIR,
    "kmeans_best_k_silhouette_samples.png"
)

plt.figure(figsize=(9, 6))

y_lower = 10

for cluster in range(best_k):
    cluster_silhouette_values = sample_silhouette_values[best_labels == cluster]
    cluster_silhouette_values.sort()

    cluster_size = cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size

    plt.fill_betweenx(
        y=range(y_lower, y_upper),
        x1=0,
        x2=cluster_silhouette_values,
        facecolor=COLORS[cluster % len(COLORS)],
        edgecolor=COLORS[cluster % len(COLORS)],
        alpha=0.75
    )

    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster))
    y_lower = y_upper + 10

plt.axvline(
    x=best_result["Silhouette Score"],
    color="#c1121f",
    linestyle="--",
    label="Average Silhouette Score"
)

plt.title(f"Silhouette Plot for Best K-Means Model (k={best_k})")
plt.xlabel("Silhouette coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(silhouette_samples_path, dpi=300)
plt.close()

print(f"Per-sample silhouette plot saved as: {silhouette_samples_path}")
print()


print("=== PCA Representation Data Snapshot ===")
print(pca_df.head().to_string(index=False))
print()


print("=== Interpretation ===")
print(
    f"The best K-means model was {best_exp_name} with k={best_k}. "
    f"It achieved a Silhouette Score of {best_result['Silhouette Score']:.4f}. "
    f"Since K-means is an unsupervised algorithm, the target variable was not used "
    f"during clustering. The target column was used only after clustering to compare "
    f"the discovered groups with the actual heart disease classes. "
    f"The Adjusted Rand Index was {best_result['Adjusted Rand Index']:.4f}."
)