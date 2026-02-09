import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import umap.umap_ as UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
from sklearn.datasets import make_blobs

centers = [
    [2, -6, -6],
    [-1, 9,  4],
    [-8, 7,  2],
    [4,  7,  9]
]
cluster_std = [1,1,2,3.5]

# Make the blobs and return the data and blob labels
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)

# Displaying the data in an interactive 3D plot
#Create a Dataframe for plotly
df = pd.DataFrame(X, columns=['x', 'y', 'z'])

# Create interactive 3D scatters plotly
fig = px.scatter_3d(df, x='x', y='y', z='z', color=labels_.astype(str), opacity=0.7, color_discrete_sequence=px.colors.qualitative.G10, title='3D Scatter Plot of Blobs')

fig.update_traces(marker=dict(size=5, line=dict(width=1, color='black')), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800)

fig.show()

# Standardize the data to prepare it for the three projection methods
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# Apply t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne = tsne.fit_transform(X_scaled)

# the 2D t-SNE result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(tsne[:, 0], tsne[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7)
ax.set_title('t-SNE Projection of Blobs')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# Compare UMAP and PCA dimensionality reduction to two dimensions
# Apply UMAP to reduce the dimensionality to 2D
umap_model = UMAP.UMAP(n_components=2, random_state=42, min_dist=0.5, spread=1.0, n_jobs=-1)

X_umap = umap_model.fit_transform(X_scaled)

# Plot the 2D UMAP projection result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
ax.set_title('UMAP Projection of Blobs')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# Apply PCA to reduce the dimensionality to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


fig = plt.figure(figsize=(8, 6))

# Plot the 2D PCA result (right)
ax2 = fig.add_subplot(111)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax2.set_title("2D PCA Projection of 3-D Data")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()