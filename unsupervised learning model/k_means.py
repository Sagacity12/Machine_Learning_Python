import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# K-Means on a synthetic dataset
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2, -3], [1,1], ], cluster_std=0.90)
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80)

# setting up k-means
k_means = KMeans(init = 'k-means++', n_clusters= 4, n_init= 12)
k_means.fit(X)
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# Creating the visual plot
# Initialize the plot with the specified dimensions
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means.labels_))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2,-1], [2, -3], [1,1], ])), colors):
    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means.labels_ == k)
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

