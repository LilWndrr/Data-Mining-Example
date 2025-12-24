import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'my_location_features.csv')
df = pd.read_csv(csv_path)

# 2. Prepare Features
features = df[['Distance_m', 'Duration_min', 'Speed_m_min']]

# 3. Scale Data (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Print Statistics
print("\n--- Cluster Averages ---")
# This shows you what each cluster represents (e.g., Cluster 0 = Short Walks)
print(df.groupby('Cluster')[['Distance_m', 'Duration_min', 'Speed_m_min']].mean())

# 6. Plotting
plt.figure(figsize=(10, 6))

# FIXED: X=Duration, Y=Distance (Now matches your labels)
scatter = plt.scatter(
    df['Duration_min'],   # X Axis
    df['Distance_m'],     # Y Axis
    c=df['Cluster'], 
    cmap='viridis', 
    alpha=0.6,
    edgecolors='k'
)

plt.title('K-Means Clustering of Location Data')
plt.xlabel('Duration (minutes)')
plt.ylabel('Distance (meters)')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, alpha=0.3)

# Save and Show
plt.savefig('cluster_results.png')
print("\nPlot saved as 'cluster_results.png'")
plt.show()