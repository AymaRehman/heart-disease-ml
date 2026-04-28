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
print("Class distribution:")
print(df["target"].value_counts())

# separating features from target
# we dont want the computer to see the target when clustering
X = df.drop(columns=["target"])
y = df["target"]

# normalisation is needed because ward linkage uses euclidean distance
# if we only scale continuous features, the unscaled categorical features
# like cp, ca, thal dominate the distance calculation
# so we scale all features to bring everything between 0 and 1

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

print(" Normalisation ")
print("MinMax scaling applied to all features")
print(f"Features scaled: {list(X.columns)}")

# building the linkage matrix using ward method
# ward method minimizes the variance within each cluster
# this is the foundation of hierarchical clustering
linkage_matrix = linkage(X_scaled, method="ward")

print(" Linkage Matrix ")
print("Linkage method used: Ward")
print(f"Linkage matrix shape: {linkage_matrix.shape}")

# Experiment 1 - low cutoff
# keeping the same ward linkage but moving the cutoff line to see how clusters change
# cutoff 3 means we cut the tree at a low height so we get more smaller groups

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
# raising the cutoff to 6 means we cut higher up the tree
# so groups that were separate in exp1 now merge together
# this gives us fewer but bigger clusters

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
# cutting at 10 means we go very high up the tree
# most groups merge together giving us very few clusters
# this is the broadest view of the data structure
# aiming for around 2 clusters to match our 2 classes

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

# silhouette score measures how well each patient fits in its cluster
# score close to 1 means patients are well grouped
# score close to 0 means clusters overlap
# score close to -1 means patients are in wrong clusters

score1 = silhouette_score(X_scaled, clusters_exp1)
score2 = silhouette_score(X_scaled, clusters_exp2)
score3 = silhouette_score(X_scaled, clusters_exp3)

print(" Silhouette Scores ")
print(f"Experiment 1 (cutoff = 3): {round(score1, 4)}")
print(f"Experiment 2 (cutoff = 6): {round(score2, 4)}")
print(f"Experiment 3 (cutoff = 11): {round(score3, 4)}")


# wrapping up the results from all 3 experiments

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

# silhouette score tells us which experiment gave the best clusters
print("The silhouette scores show how well the clusters are separated")
