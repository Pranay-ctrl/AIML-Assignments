import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data with three groups
X, _ = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=0.8,
    random_state=42
)

# Standardise the data
X = StandardScaler().fit_transform(X)

# Create K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model and obtain cluster labels
labels = model.fit_predict(X)

# Display cluster labels
print("First 15 Cluster Labels:")
print(labels[:15])

# Display cluster centres
print("\nCluster Centres:")
print(model.cluster_centers_)

# Visualise clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()