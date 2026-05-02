# Hierarchical Clustering Analysis - Part II
# Authored: @nandana-subhash

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/heart_cleaned.csv")

print(" Dataset Info ")
print(f"Total patients: {len(df)}")
print(f"Features: {list(df.columns)}")
print(f"Class distribution:")
print(df["target"].value_counts())

# separating features from target
# we dont want the computer to see the target when clustering
X = df.drop(columns=["target"])
y = df["target"]

# scaling all features to 0 and 1
# because ward measures similarity using straight line distance
# unscaled features would dominate the distance calculation

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

print(" Normalisation ")
print("MinMax scaling applied to all features")
print(f"Features scaled: {list(X.columns)}")

# building the linkage matrix using ward method
# ward method minimizes the variance within each cluster

linkage_matrix = linkage(X_scaled, method="ward")

print(" Linkage Matrix ")
print("Linkage method used: Ward")
print(f"Linkage matrix shape: {linkage_matrix.shape}")
   

# Experiment 1 - low cutoff
# cutoff 3 gives us many small clusters

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram - Experiment 1 (cutoff=3)")
plt.xlabel("Patient Index")
plt.ylabel("Distance")
plt.axhline(y=3, color="r", linestyle="--", label="cutoff = 3")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/exp1_dendrogram.png")
plt.show()

clusters_exp1 = fcluster(linkage_matrix, t=3, criterion="distance")

print(" Experiment 1 (cutoff = 3) ")
print(f"Number of clusters formed: {len(set(clusters_exp1))}")
print("Patients per cluster:")
print(pd.Series(clusters_exp1).value_counts().sort_index().to_string())

# Experiment 2 - medium cutoff
# cutoff 6 gives us fewer but bigger clusters than exp1

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram - Experiment 2 (cutoff=6)")
plt.xlabel("Patient Index")
plt.ylabel("Distance")
plt.axhline(y=6, color="g", linestyle="--", label="cutoff = 6")
# green line so we can visually tell experiments apart in screenshots
plt.legend()
plt.tight_layout()
plt.savefig("outputs/exp2_dendrogram.png")
plt.show()

# same linkage matrix as exp1, only the cutoff changes across all experiments
clusters_exp2 = fcluster(linkage_matrix, t=6, criterion="distance")

print(" Experiment 2 (cutoff = 6) ")
print(f"Number of clusters formed: {len(set(clusters_exp2))}")
print("Patients per cluster:")
print(pd.Series(clusters_exp2).value_counts().sort_index().to_string())

# Experiment 3 - high cutoff
# cutoff 11 gives us very few clusters
# aiming for 2 clusters to match our 2 known classes

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram - Experiment 3 (cutoff=11)")
plt.xlabel("Patient Index")
plt.ylabel("Distance")
plt.axhline(y=11, color="b", linestyle="--", label="cutoff = 11")
# blue line to distinguish from exp1 (red) and exp2 (green)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/exp3_dendrogram.png")
plt.show()

# same linkage matrix, only cutoff value changes
clusters_exp3 = fcluster(linkage_matrix, t=11, criterion="distance")

print(" Experiment 3 (cutoff = 11) ")
print(f"Number of clusters formed: {len(set(clusters_exp3))}")
print("Patients per cluster:")
print(pd.Series(clusters_exp3).value_counts().sort_index().to_string())


# comparing clusters with actual heart disease labels
# this tells us if the computer naturally separated sick from healthy patients
# we use crosstab to see how many patients from each class ended up in each cluster

print(" Cluster vs Heart Disease Labels ")

print("Experiment 1 (cutoff = 3):")
print(pd.crosstab(clusters_exp1, y, rownames=["Cluster"], colnames=["Heart Disease"]).to_string())
print()

print("Experiment 2 (cutoff = 6):")
print(pd.crosstab(clusters_exp2, y, rownames=["Cluster"], colnames=["Heart Disease"]).to_string())
print()

print("Experiment 3 (cutoff = 11):")
print(pd.crosstab(clusters_exp3, y, rownames=["Cluster"], colnames=["Heart Disease"]).to_string())
print()

# silhouette score measures how well the clusters are separated
# ranges from -1 to 1, higher is better

score1 = silhouette_score(X_scaled, clusters_exp1)
score2 = silhouette_score(X_scaled, clusters_exp2)
score3 = silhouette_score(X_scaled, clusters_exp3)

print(" Silhouette Scores ")
print(f"Experiment 1 (cutoff = 3): {round(score1, 4)}")
print(f"Experiment 2 (cutoff = 6): {round(score2, 4)}")
print(f"Experiment 3 (cutoff = 11): {round(score3, 4)}")


# results summary

print(" Conclusions ")
print(f"Total patients analysed: {len(df)}")
print(f"Class 0 (no disease): {sum(y == 0)} patients")
print(f"Class 1 (disease present): {sum(y == 1)} patients")
print()
print("Experiment summary:")
print(f"Experiment 1 (cutoff = 3): {len(set(clusters_exp1))} clusters formed")
print(f"Experiment 2 (cutoff = 6): {len(set(clusters_exp2))} clusters formed")
print(f"Experiment 3 (cutoff = 11): {len(set(clusters_exp3))} clusters formed")
print()

# higher cutoff = fewer clusters, lower cutoff = more clusters
# ward linkage stayed the same in all 3, only cutoff changed
print("Ward linkage method was used across all 3 experiments")
print("Only the cutoff value was changed between experiments")
print()

# silhouette scores show how well separated the clusters are
print("The silhouette scores show how well the clusters are separated")
