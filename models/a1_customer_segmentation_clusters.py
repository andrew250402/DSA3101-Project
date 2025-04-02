import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.lines as mlines
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score

# ==========================================================
#                     DATA LOADING                          
# ==========================================================
# Load datasets
Transaction = pd.read_csv('transactions_summary.csv')
Usage = pd.read_csv('digital_usage.csv')
Customers = pd.read_csv('customers.csv')

# ==========================================================
#                     DATA CLEANING                         
# ==========================================================
# Cleaning Transaction Dataset
Transaction = Transaction[Transaction['current_balance'] >= 0]
Transaction = Transaction[Transaction['peak_month_spending'] >= 0]

# Cleaning Usage Dataset
## Remove columns that are not needed for our analysis.
Usage['has_mobile_app'] = Usage['has_mobile_app'].replace({'Yes': 1, 'No': 0}) 
Usage['has_web_account'] = Usage['has_web_account'].replace({'Yes': 1, 'No': 0})
Usage = Usage.drop(columns=['last_mobile_login', 'last_web_login']) 

# Cleaning Customers Dataset
# Filter out entries with erroneous data such as negative income and created_at dates from the future.
# For our project, we also consider customers below the age of 21 to be erroneous entries.
binary_columns = ['credit_default', 'credit_card', 'personal_loan', 'mortgage', 'savings_account', 'investment_product', 'auto_loan', 'wealth_management']
Customers[binary_columns] = Customers[binary_columns].replace({'yes': 1, 'no': 0}) 
Customers = Customers[(Customers['income'] >= 0) & (Customers['age'] >= 21)] 
Customers = Customers[Customers['created_at'] < '2025-03-01']
Customers = Customers.drop(columns=['customer_segment']) # To avoid having our results influenced by the old clusters, they are also removed.
customers_copy = Customers.copy() # To be used for saving as CSV
Customers = Customers.drop(columns=['created_at']) # Not useful for clustering
customer_data_original = Customers.copy() # To be used for merging
Customers = Customers.drop(columns=['customer_id']) # Customer_id is not informative and should not be used in the analysis.



# ==========================================================
#                 FEATURE ENGINEERING                      
# ==========================================================
# One-hot encoding for categorical variables
# PCA (Principal Component Analysis) works by calculating variance between different columns, so our data needs to be numerical.
# One-hot encoding converts all categorical variables into values of 0 and 1, allowing PCA to do its computations.
customer_data_encoded = pd.get_dummies(Customers)

# Standardizing data
# PCA is sensitive to the scale of different features.
# Standardising every feature to have a mean of 0 and standard deviation of 1 ensures that all features have equal importance in PCA.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data_encoded)

# ==========================================================
#              HIERARCHICAL CLUSTERING                     
# ==========================================================
# Perform PCA for dimensionality reduction
# Create a Scree Plot to decide the number of components to use

pca = PCA().fit(scaled_data)

# Below is the scree plot
# A Scree Plot shows the explained variance of each component from PCA.
# We should pick just enough components to ensure the cumulative explained variance is at least 95%, so we chose to keep 22 components.
'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()
'''
pca = PCA(n_components=22)  # Chosen from Scree Plot
scaled_data_pca = pca.fit_transform(scaled_data)

# Apply Hierarchical Clustering
linkage_matrix = linkage(scaled_data_pca, method='ward')

# Below is the dendrogram plot
# A dendrogram is a tree-like diagram that shows the hierarchical relationship between data points.
# We choose 4 clusters based on the dendrogram plot, where the red line intersects the dendrogram.
'''
# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.axhline(y=150, color='r', linestyle='--')  # Adjust threshold based on the plot
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
'''
n_clusters = 4 # Chosen from Dendrogram
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_data_pca)

# Assign cluster labels
customer_data_with_labels = customers_copy
customer_data_with_labels['cluster'] = cluster_labels
print("Number of customers in each cluster: For New Customers (Hierarchical Clustering)")
print(customer_data_with_labels['cluster'].value_counts())

# ==========================================================
#                 K-MEANS CLUSTERING                       
# ==========================================================
# Find optimal number of clusters using Silhouette Score
silhouette_scores = []
cluster_range = range(2, 10)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data_pca)
    silhouette_scores.append(silhouette_score(scaled_data_pca, cluster_labels))

# Below is the Silhouette Score plot to find the chosen K
# The Silhouette Score is a measure of how similar an object is to its own cluster compared to other clusters.
# K was chosen to be 4 as it has the highest Silhouette Score between 2 to 5 clusters.
'''
# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='r')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()
'''

# Apply K-Means with chosen K (K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data_pca)

# Assign cluster labels
customer_data_with_labels_kmean = Customers.copy()
customer_data_with_labels_kmean['cluster'] = cluster_labels
print("Number of customers in each cluster: For New Customers (K-Means Clustering)")
print(customer_data_with_labels_kmean['cluster'].value_counts())

# ==========================================================
#        MERGING DATASETS FOR EXISTING CUSTOMERS           
# ==========================================================
df = pd.merge(customer_data_original, Transaction, on='customer_id', how='inner')
df = pd.merge(df, Usage, on='customer_id', how='inner')
df = df.drop('customer_id', axis=1) # Remove 'customer_id' column
# Check the shape of the merge dataset
# print(df.shape) # 9429 rows
# print(df.columns)


# ==========================================================
#                 FEATURE ENGINEERING                      
# ==========================================================
# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# print(df_encoded.head())

# Standardise the data
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(df_encoded)
#scaled_data.shape # 9429 Rows


# ==========================================================
#              HIERARCHICAL CLUSTERING                     
# ==========================================================

# Perform PCA for dimensionality reduction
# Create a Scree Plot to decide the number of components to use
pca = PCA().fit(scaled_data)

# Below is the scree plot
# A Scree Plot shows the explained variance of each component from PCA.
# We should pick just enough components to ensure the cumulative explained variance is at least 95%, so we chose to keep 40 components.
'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()
'''

pca = PCA(n_components=40) # Reduce to 40 dimensions (chosen from Scree Plot)
scaled_data_pca = pca.fit_transform(scaled_data)

# Apply Hierarchical Clustering
linkage_matrix = linkage(scaled_data_pca, method='ward')

# Below is the dendrogram plot
'''
# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.axhline(y=250, color='r', linestyle='--')  # Adjust threshold based on the plot
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
'''

n_clusters = 3 # Chosen from Dendrogram
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_data_pca)

# Assign cluster labels
customer_data_with_labels_extras = customers_copy
customer_data_with_labels_extras['cluster'] = cluster_labels
print("Number of existing customers in each cluster: For Existing Customers (Hierarchical Clustering)")
print(customer_data_with_labels_extras['cluster'].value_counts())


# ==========================================================
#                 K-MEANS CLUSTERING                       
# ==========================================================

# Find the optimal number of clusters for KMeans
silhouette_scores = []
cluster_range = range(2, 10)  # Start from 3 clusters to max 9 clusters
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data_pca)
    silhouette_scores.append(silhouette_score(scaled_data_pca, cluster_labels))

# Below is the Silhouette Score plot to find the chosen K
# K was chosen to be 3 as it has the highest Silhouette Score.
'''
# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='r')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()
'''

# Apply K-Means with chosen K (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data_pca)
customer_data_with_labels_extras_kmean = df_encoded.copy()
customer_data_with_labels_extras_kmean['cluster'] = cluster_labels
print("Number of existing customers in each cluster: For Existing Customers (K-Means Clustering)")
print(customer_data_with_labels_extras_kmean['cluster'].value_counts())

# ==========================================================
#                  CONCLUSION                  
# ==========================================================

# Hierarchical Clustering is chosen for both new and existing customers. Both K-Means and hierarchical clustering are able to segments the clusters well. 
# But hierarchical edges over k-means due to slightly clearer segmentation and its slight robustness to outliers as compared to K-means.
# Additionally, hierarchical clustering can handle mixed data types better, and has no assumption on equal sized clusters (KMeans have the tendancy to create equally sized clusters)


# ==========================================================
#                  SAVING RESULTS                          
# ==========================================================
# customer_data_with_labels.to_csv('customer_data_with_labels_new_customers.csv', index=False) (For New Customers) (4 clusters)
# customer_data_with_labels_extras.to_csv('customer_data_with_labels_existing_customers.csv', index=False) (For Existing Customers) (To be used for sub-questions) (3 clusters)
