import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.rio.helpers import coords
from sklearn.cluster import DBSCAN
import hdbscan
import requests
import io
import zipfile

from sklearn.preprocessing import StandardScaler

# geographical tools
# pandas dataframe-like geodataframes for geographical data
import geopandas as gpd
# used for obtianing a basemap of Canada
import contextily as ctx
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'

output_dir = './'
os.makedirs(output_dir, exist_ok=True)

response = requests.get(zip_file_url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted {file_name}")

# function that plots clustered location and overlays them on a basemap
def plot_clustered_locations(df, title='Museums Clustered by Proximity'):
    df_plot = df.copy()
    df_plot['Latitude'] = pd.to_numeric(df_plot['Latitude'], errors='coerce')
    df_plot['Longitude'] = pd.to_numeric(df_plot['Longitude'], errors='coerce')
    df_plot = df_plot.dropna(subset=['Latitude', 'Longitude'])
    """
    Plots clustered locations and overlays on a basemap of Canada.

    Parameters:
        - df: Dataframe containing 'Latitude', 'Longitude', 'Cluster' columns.
        - title: str, title of the plot.
    """

    # coordinates into a GeoDataFrame
    # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot['Longitude'], df_plot['Latitude']),
        crs="EPSG:4326"
    )
    # Reproject to Web Mercator to align with basemap
    gdf = gdf.to_crs(epsg=3857)

    # Creating the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    # plot the points
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')

    # Plot clustered points, coloured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)

    # Add basemap of Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)

    # Format plot
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")
df.head()

# Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
coords = df[['Latitude', 'Longitude']].copy()
coords['Latitude'] = pd.to_numeric(coords['Latitude'], errors='coerce')
coords['Longitude'] = pd.to_numeric(coords['Longitude'], errors='coerce')

mask = coords.notna().all(axis=1)
coords = coords.loc[mask]

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)


# Apply DBSCAN with Euclidean distance to the scaled coordinates
min_samples = 3
eps = 1.0
metrix = 'euclidean'

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metrix).fit(coords_scaled)

# Add cluster labels to the DataFrame
clusters = dbscan.fit_predict(coords_scaled)
df.loc[mask, 'Cluster'] = clusters
df['Cluster'].value_counts()

# Plot the museums on a basemap of Canada, colored by cluster label
# plot_clustered_locations(df, title='Museums Clustered by Proximity')

# Build an HDBSCAN clustering model
coords = df[['Latitude', 'Longitude']].copy()
coords['Latitude'] = pd.to_numeric(coords['Latitude'], errors='coerce')
coords['Longitude'] = pd.to_numeric(coords['Longitude'], errors='coerce')

mask = coords.notna().all(axis=1)
coords = coords.loc[mask]

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)



min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')

# Assign labels
clusters = hdb.fit_predict(coords_scaled)
df.loc[mask, 'Cluster'] = clusters

# Display the size of each cluster
df['Cluster'].value_counts()
print(df['Cluster'].value_counts(dropna=False))
plot_clustered_locations(df, title='Museums Clustered by Proximity')
# Plot the museum clusters
# plot_clustered_locations(df, title='Museums Hierarchically Clustered by Proximity')