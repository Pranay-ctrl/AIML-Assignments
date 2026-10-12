import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data

print("Original Dataset Shape:", X.shape)

# Standardise the data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Display the results
print("Reduced Dataset Shape:", X_reduced.shape)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Variance Retained:")
print(pca.explained_variance_ratio_.sum())

# Visualise the reduced data
plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=iris.target
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset After PCA")
plt.show()