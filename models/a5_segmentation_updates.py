# Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Load customer data
customer_data = pd.read_csv('../Data DSA3101/customers.csv')
# Data cleaning and processing
## Filter out entries with erroneous data such as negative income and created_at dates from the future.
## For our project, we also consider customers below the age of 21 to be erroneous entries.
customer_data = customer_data[customer_data['income'] >= 0]
customer_data = customer_data[customer_data['age'] >= 21]
customer_data = customer_data[customer_data['created_at'] < '2025-03-01']
customer_data = customer_data.drop(columns=['created_at'])
# Show first few entries
customer_data.head()

# Load usage data
usage_data = pd.read_csv('../Data DSA3101/digital_usage.csv')
# Data cleaning and processing
## Remove columns that are not needed for our analysis.
usage_data = usage_data.drop(columns=['last_mobile_login', 'last_web_login'])
# Show first few entries
usage_data.head()

# Load transaction data
transaction_data = pd.read_csv('../Data DSA3101/transactions_summary.csv')
# Data cleaning and processing
## Filter out entries with erroneous data, specifically entries with negative balance and spending.
transaction_data = transaction_data[transaction_data['current_balance'] >= 0]
transaction_data = transaction_data[transaction_data['peak_month_spending'] >= 0]
# Show first few entries
transaction_data.head()

# Merge the datasets
df = pd.merge(customer_data, transaction_data, on='customer_id', how='inner')
df = pd.merge(df, usage_data, on='customer_id', how='inner')
# Check the shape of the merged dataset
print("Dimension of combined dataset: " + str(df.shape)) # 9429 rows

# Drop customer_id and customer_segment
## customer_id is not informative and should not be used in the analysis.
## To avoid having our results influenced by the old clusters, they are also removed.
df_no_id = df.drop('customer_id', axis=1)
df_no_id = df.drop('customer_segment', axis=1)
# Create one-hot encoding for categorical variables
## PCA (Principal Component Analysis) works by calculating variance between different columns, so our data needs to be numerical.
## One-hot encoding converts all categorical variables into values of 0 and 1, allowing PCA to do its computations.
df_encoded_no_id = pd.get_dummies(df_no_id)
# Standardise the data
## PCA is sensitive to the scale of different features.
## Standardising every feature to have a mean of 0 and standard deviation of 1 ensures that all features have equal importance in PCA.
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(df_encoded_no_id)
scaled_data.shape

# Define batch size and initialise Incremental PCA
## Incremental PCA's advantage over normal PCA is its ability to process the dataset in mini-batches, making it more memory-efficient and allowing it to work well with large datasets.
## For our dataset of about 10000 rows, we selected a batch size of 100.
batch_size = 100
ipca = IncrementalPCA()
# Fit in mini-batches
for i in range(0, scaled_data.shape[0], batch_size):
    ipca.partial_fit(scaled_data[i:i+batch_size])  # Partial fit on each batch

# Reduce to 40 dimensions (chosen from Scree Plot)
## A Scree Plot shows the explained variance of each component from PCA.
## We should pick just enough components to ensure the cumulative explained variance is at least 95%, so we chose to keep 40 components.
ipca = IncrementalPCA(n_components=40)  
scaled_data_ipca = ipca.fit_transform(scaled_data)  # Transform data
# Train BIRCH on historical data
## BIRCH's (Balanced Iterative Reducing and Clustering using Hierarchies) main advantage over Hierarchical Clustering is its ability to learn incrementally.
## This means the model is able to handle and cluster incoming data without the need to retrain the model from scratch.
## In the context of real-time segmentation, new customers can be added and clustered easily while saving memory and time.
birch = Birch(threshold=0.5, branching_factor=50, n_clusters=3) # Number of clusters are decided from hierarchical clustering
birch.fit(scaled_data_ipca)
birch_labels = birch.predict(scaled_data_ipca)
birch_labels_df = pd.DataFrame(birch_labels, columns=['cluster'])
print("Cluster results from BIRCH:")
print(birch_labels_df.head())

# Add clusters to the customers dataset
## After obtaining the clusters from BIRCH, we add the results to the customer dataset to be used for other tasks.
customer_data_with_BIRCH_clusters = pd.concat([customer_data.reset_index(drop=True), birch_labels_df.reset_index(drop=True)], axis=1)
print("New dataset dimension after adding cluster labels" + str(customer_data_with_BIRCH_clusters.shape))

# Create new customer
## We can test the real-time segmentation capabilities of our models by simulating a new customer.
print("Testing customer segmentation with new customer...")
new_customer = pd.DataFrame([(8, 56, 'management', 'divorced', 'tertiary', 'yes', 'Suburban', 6548,
       'yes', 'no', 'no', 'no', 'yes', 'yes', 'no', 19,
       11440.758, 9, 35896, 51, 147, 242, 539,
       51454.288, 127104.169, 191752.301,
       350.029, 525.223, 355.755, 49.0,
       40.333, 44.916, 0.176,
       0.123, 0.142, 0.767,
       0.697, 0.762, 'No', 'no', 0, 7, 0.0, 2.6)], # Change values here                     
    columns=df_no_id.columns)
print(new_customer)
new_customer = pd.concat([new_customer, df_no_id[0:50]], axis=0) # Include other customer data to create a sufficient batch size for IPCA

# Process new data in the same way as the original dataset
new_customer = pd.get_dummies(new_customer).reindex(columns=df_encoded_no_id.columns, fill_value=0)

# Partial fit Incremental PCA to the new customer
new_customer_scaled = data_scaler.transform(new_customer)
print("Running Incremental PCA on new customer...")
ipca.partial_fit(new_customer_scaled)
new_customer_ipca = ipca.transform(new_customer_scaled)

# Run BIRCH on the new customer
print("Running BIRCH on new customer...")
new_cluster_label = birch.predict(new_customer_ipca)
print("New customer assigned to cluster:", new_cluster_label[0])