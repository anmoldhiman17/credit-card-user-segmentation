# Segmenting Credit Card Users with K-Means, Elbow Method, and PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load dataset
df = pd.read_csv('Intenship_Project_Gncipl_Week3/archive (7).zip')  # Download from Kaggle and keep in same folder

# 2. Clean & preprocess
df = df.drop('CUST_ID', axis=1)
df = df.fillna(df.mean())

# 3. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 4. Elbow Method to choose k
wcss = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 5. Fit K-Means with chosen k (example: k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 6. PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.title('Credit Card User Segmentation (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.show()

# 7. Cluster summary
summary = pd.Series(labels).value_counts().reset_index()
summary.columns = ['Cluster', 'Count']
print(summary)
