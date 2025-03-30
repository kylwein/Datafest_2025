import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load CSV files (adjust paths as needed)
leases_df = pd.read_csv("./Leases.csv")
occupancy_df = pd.read_csv("./Major Market Occupancy Data-revised.csv")
price_df = pd.read_csv("./Price and Availability Data.csv")
unemployment_df = pd.read_csv("./Unemployment.csv")

# Merge datasets on common keys
df_merged = pd.merge(leases_df, price_df, on=["year", "quarter", "market"], how="left", suffixes=("", "_price"))
df_merged = pd.merge(df_merged,
                     occupancy_df[['year', 'quarter', 'market', 'starting_occupancy_proportion', 'avg_occupancy_proportion']],
                     on=["year", "quarter", "market"], how="left")
df_merged = pd.merge(df_merged, unemployment_df[['year', 'quarter', 'state', 'unemployment_rate']],
                     on=["year", "quarter", "state"], how="left")

# Define numeric columns to aggregate
agg_cols = [
    'leasing', 'RBA', 'availability_proportion', 'available_space',
    'starting_occupancy_proportion', 'avg_occupancy_proportion',
    'direct_availability_proportion', 'direct_available_space',
    'direct_internal_class_rent', 'direct_overall_rent', 'internal_class_rent',
    'leasedSF', 'overall_rent', 'sublet_availability_proportion',
    'sublet_available_space', 'sublet_internal_class_rent', 'sublet_overall_rent',
    'unemployment_rate'
]

# Aggregate by year, quarter, market, and internal_industry (from leases.csv)
df_agg = df_merged.groupby(['year', 'quarter', 'market', 'internal_industry'])[agg_cols].mean().reset_index()

selected_industries = [
    'Technology, Advertising, Media, and Information',
    'Healthcare',
    'Financial Services',
    'Retail',
    'Legal Services',
    'Government',
    'Construction, Engineering and Architecture',
    'Manufacturing (except Pharmaceutical, Retail, and Computer Tech)',
    'Energy & Utilities'
]

# Filter aggregated data to only include selected industries
df_selected = df_agg[df_agg['internal_industry'].isin(selected_industries)].copy()
print("Filtered DataFrame shape:", df_selected.shape)

# Choose numeric predictors for clustering
features = ['leasing', 'available_space', 'leasedSF', 'overall_rent', 'unemployment_rate']
df_selected = df_selected.dropna(subset=features)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(df_selected[features])

# Apply KMeans clustering (using 3 clusters; adjust as needed)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_selected['cluster'] = kmeans.fit_predict(X)

print("Clustering results:")
print(df_selected[['market', 'internal_industry', 'cluster']].head())

# Reduce dimensions for visualization with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df_selected.index)
df_pca['cluster'] = df_selected['cluster']
df_pca['internal_industry'] = df_selected['internal_industry']
df_pca['market'] = df_selected['market']

# Plot the data points colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='cluster', palette='Set1', s=50, alpha=0.8)

# Compute centroids for each cluster (in PCA space)
centroids = df_pca.groupby('cluster')[['PC1', 'PC2']].mean().reset_index()



plt.title("Market Segmentation by Leasing and Industry Patterns")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


