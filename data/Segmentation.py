# Loading of packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.lines as mlines
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Import the datasets
Transaction = pd.read_csv('transactions_summary.csv')
Usage = pd.read_csv('digital_usage.csv')
Customers = pd.read_csv('customers.csv')




# Cleaning the Transaction dataset
# Check for missing values
# print(Transaction.isnull().sum()) # 0

# Check for 0 values in the dataset
# print((Transaction == 0).sum()) # Multiple 0 values in the dataset

# Check the Statistics of the dataset
# print(Transaction.describe()) # Negative Balance, negative peak_month_spending, 0 values in the dataset

# Remove all the negative rows in the dataset
Transaction = Transaction[Transaction['current_balance'] >= 0]
Transaction = Transaction[Transaction['peak_month_spending'] >= 0]

#print(Transaction.describe())

# Shape of the dataset
#print(Transaction.shape) # 9940 rows and 24 columns

# Find the rows where average_transaction_amount_12m is 0
zero_avg_transaction = Transaction[Transaction['average_transaction_amount_12m'] == 0]
# Show in table format
#display(zero_avg_transaction)





# Cleaning the Usage dataset
# Check for missing values
# print(Usage.isnull().sum())

# Check for 0 values in the datasets
# print((Usage == 0).sum())

# Check if Usage dataset is clean
#print(Usage.describe())

# Check if last_mobile_login is null, the mobile_logins_per_week is 0, avg_mobile_session_duration is 0, and has_mobile_app is No
sum_mobile_null = Usage['last_mobile_login'].isnull().sum()
#print(sum_mobile_null) # 1659
sum_mobile = 0
for _, row in Usage.iterrows():
    if row['has_mobile_app'] == 'No' and pd.isnull(row['last_mobile_login']) and row['mobile_logins_per_week'] == 0 and row['avg_mobile_session_duration'] == 0:
        sum_mobile += 1
#print(sum_mobile) # 1659, the number matches as expected

# Check if last_web_login is null, the web_logins_per_week is 0, avg_web_session_duration is 0, and has_web_account is No
sum_web_null = Usage['last_web_login'].isnull().sum()
#print(sum_web_null) # 1539
sum_web = 0
for _, row in Usage.iterrows():
    if row['has_web_account'] == 'No' and pd.isnull(row['last_web_login']) and row['web_logins_per_week'] == 0 and row['avg_web_session_duration'] == 0:
        sum_web += 1
#print(sum_web) # 1539, the number matches as expected

# Check if last_mobile_login is null, the sum of mobile_logins_per_week is 0, avg_mobile_session_duration is 0, and has_mobile_app is No
sum_mobile = 0
for _, row in Usage.iterrows():
    if pd.isnull(row['last_mobile_login']):
        sum_mobile += row['mobile_logins_per_week'] + row['avg_mobile_session_duration'] + (1 if row['has_mobile_app'] == 'Yes' else 0)
#print(sum_mobile) # 0, the number matches as expected

# Check if last_web_login is null, the sum of web_logins_per_week is 0, avg_web_session_duration is 0, and has_web_account is No
sum_web = 0
for _, row in Usage.iterrows():
    if pd.isnull(row['last_web_login']):
        sum_web += row['web_logins_per_week'] + row['avg_web_session_duration'] + (1 if row['has_web_account'] == 'Yes' else 0)
#print(sum_web) # 0, the number matches as expected

# Changing Yes/No response to numerical response
columns_to_convert = ['has_mobile_app', 'has_web_account']
Usage[columns_to_convert] = Usage[columns_to_convert].replace({'Yes': 1, 'No': 0})

# Remove the last web/app log in
Usage = Usage.drop(columns=['last_mobile_login', 'last_web_login'])

#print(Usage.head())




# Cleaning the Customers dataset
columns_to_convert = ['credit_default', 'credit_card', 'personal_loan', 'mortgage', 'savings_account', 'investment_product', 'auto_loan', 'wealth_management']
Customers[columns_to_convert] = Customers[columns_to_convert].replace({'yes': 1, 'no': 0})
Customers = Customers[Customers['income'] >= 0] # Filter out entries with negative income
Customers = Customers[Customers['age'] >= 21] # Filter out non-adults
Customers = Customers[Customers['created_at'] < '2025-03-01'] # Filter out erroneous entries
Customers = Customers.drop(columns=['created_at']) # Remove column
Customers = Customers.drop(columns=['customer_segment']) # Remove column
customer_data_original = Customers.copy() # To be used for merging
Customers = Customers.drop(columns=['customer_id']) # Remove column

# Show first few entries
#Customers.head()



# Feature engineering
# Create one hot encoding for categorical variables
customer_data_encoded = pd.get_dummies(Customers)

# Standardise the data
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(customer_data_encoded)
scaled_data.shape



# Create a Scree Plot to decide the number of components to use
pca = PCA().fit(scaled_data)
'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()
'''
pca = PCA(n_components=22)  # Reduce to 22 dimensions (chosen from Scree Plot)
scaled_data_pca = pca.fit_transform(scaled_data)  # Transform data

# Compute hierarchical clustering
linkage_matrix = linkage(scaled_data_pca, method='ward')  # Ward's method for variance minimisation

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


# Assign cluster labels (decide number of clusters from dendrogram)
n_clusters = 4
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_data_pca)

#print(f"Cluster assignments: {np.unique(cluster_labels, return_counts=True)}")



# Merge cluster labels with the customer dataset
customer_data_with_labels = Customers
customer_data_with_labels['cluster'] = cluster_labels

# Count the numbers in each cluster
print("Number of customers in each cluster: For new customers (Hierarchical Clustering)")
print(customer_data_with_labels['cluster'].value_counts())







# KMeans Clustering
# Find the optimal number of clusters for KMeans
silhouette_scores = []
cluster_range = range(2, 10)  # Start from 3 clusters to max 9 clusters

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data_pca)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(scaled_data_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

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

# Add cluster labels to the dataset
customer_data_with_labels_kmean = Customers
customer_data_with_labels_kmean['cluster'] = cluster_labels


# Count the numbers in each cluster
print("Number of customers in each cluster: For new customers (K-Means Clustering)")
print(customer_data_with_labels_kmean['cluster'].value_counts())





# Merge the datasets
df = pd.merge(customer_data_original, Transaction, on='customer_id', how='inner')
df = pd.merge(df, Usage, on='customer_id', how='inner')

# Check the shape of the merge dataset
#print(df.shape) # 9429 rows
#print(df.columns)



# Data Preprocessing
# Remove 'customer_id' column
df = df.drop('customer_id', axis=1)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
#print(df_encoded.head())

# Standardise the data
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(df_encoded)
#scaled_data.shape # 9429 Rows



# Hierarchical Clustering
# Create a Scree Plot to decide the number of components to use
pca = PCA().fit(scaled_data)
'''
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()
'''


pca = PCA(n_components=40)  # Reduce to 40 dimensions (chosen from Scree Plot)
scaled_data_pca = pca.fit_transform(scaled_data)  # Transform data

# Compute hierarchical clustering
linkage_matrix = linkage(scaled_data_pca, method='ward')  # Ward's method for variance minimisation

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



# Assign cluster labels (decide number of clusters from dendrogram)
n_clusters = 3
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_data_pca)

#print(f"Cluster assignments: {np.unique(cluster_labels, return_counts=True)}")
#cluster_labels

# Merge cluster labels with the customer dataset
customer_data_with_labels_extras = df_encoded
customer_data_with_labels_extras['cluster'] = cluster_labels

# Count the numbers in each cluster
print("Number of customers in each cluster: For existing customers (Hierarchical Clustering)")
print(customer_data_with_labels_extras['cluster'].value_counts())



#KMeans Clustering
# Find the optimal number of clusters for KMeans
silhouette_scores = []
cluster_range = range(2, 10)  # Start from 3 clusters to max 9 clusters

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data_pca)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(scaled_data_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

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

# Add cluster labels to the dataset
customer_data_with_labels_extras_kmean = df_encoded
customer_data_with_labels_extras_kmean['cluster'] = cluster_labels

# Count the numbers in each cluster
print("Number of customers in each cluster: For existing customers (K-Means Clustering)")
print(customer_data_with_labels_extras_kmean['cluster'].value_counts())
#print(customer_data_with_labels_extras_kmean.columns)


# These are the 4 datasets that we have created with the cluster labels
# customer_data_with_labels # Hierarchical Clustering for new customers (4 clusters) 
# customer_data_with_labels_kmean # K-Means Clustering for new customers (4 clusters)
# customer_data_with_labels_extras # Hierarchical Clustering for existing customers (3 clusters)
# customer_data_with_labels_extras_kmean # K-Means Clustering for existing customers (3 clusters) 