# Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from joblib import dump
import pickle

def load_data_and_model():
    # Load customer data
    customer_data = pd.read_csv('../Data DSA3101/customers.csv')

    # Data cleaning and processing
    ## Filter out entries with erroneous data such as negative income and created_at dates from the future.
    ## For our project, we also consider customers below the age of 21 to be erroneous entries.
    customer_columns_to_convert = ['credit_default', 'credit_card', 'personal_loan', 'mortgage', 'savings_account', 'investment_product', 'auto_loan', 'wealth_management']
    customer_data[customer_columns_to_convert] = customer_data[customer_columns_to_convert].replace({'yes': 1, 'no': 0})
    customer_data = customer_data[customer_data['income'] >= 0]
    customer_data = customer_data[customer_data['age'] >= 21]
    customer_data = customer_data[customer_data['created_at'] < '2025-03-01']
    customer_data = customer_data.drop(columns=['created_at'])

    customer_df = customer_data

    # Drop customer_id and customer_segment
    ## customer_id is not informative and should not be used in the analysis.
    ## To avoid having our results influenced by the old clusters, they are also removed.
    customer_df = customer_df.drop('customer_id', axis=1)
    customer_df = customer_df.drop('customer_segment', axis=1)

    # Create one-hot encoding for categorical variables
    ## PCA (Principal Component Analysis) works by calculating variance between different columns, so our data needs to be numerical.
    ## One-hot encoding converts all categorical variables into values of 0 and 1, allowing PCA to do its computations.
    customer_df_encoded = pd.get_dummies(customer_df)

    # Standardise the data
    ## PCA is sensitive to the scale of different features.
    ## Standardising every feature to have a mean of 0 and standard deviation of 1 ensures that all features have equal importance in PCA.
    customer_data_scaler = StandardScaler()
    customer_scaled_data = customer_data_scaler.fit_transform(customer_df_encoded)
    customer_scaled_data.shape

    # Reduce to 22 dimensions (chosen from Scree Plot)
    ## A Scree Plot shows the explained variance of each component from PCA.
    ## We should pick just enough components to ensure the cumulative explained variance is at least 95%, so we chose to keep 22 components.
    ipca = IncrementalPCA(n_components=22)
    
    customer_scaled_data_ipca = ipca.fit_transform(customer_scaled_data)  # Transform data

    # Train BIRCH on historical data
    ## BIRCH's (Balanced Iterative Reducing and Clustering using Hierarchies) main advantage over Hierarchical Clustering is its ability to learn incrementally.
    ## This means the model is able to handle and cluster incoming data without the need to retrain the model from scratch.
    ## In the context of real-time segmentation, new customers can be added and clustered easily while saving memory and time.
    birch_customers = Birch(threshold=0.5, branching_factor=50, n_clusters=3) # Number of clusters are decided from hierarchical clustering
    birch_customers.fit(customer_scaled_data_ipca)

    return birch_customers, customer_data_scaler, ipca, customer_data, customer_df_encoded

# Save the model for Streamlit
def save_model_artifacts():
    os.makedirs('saved', exist_ok=True)
    birch_model, sc, ipca_model, customers, customer_encoded = load_data_and_model()
    birch_model.__dict__.pop('_root', None)

    with open('saved/a5-streamlit-segment.pkl', 'wb') as f:
        pickle.dump({
            'birch_model': birch_model,
            'scaler': sc,
            'ipca_model': ipca_model,
            'customers': customers,
            'customer_encoded': customer_encoded
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_model_artifacts()
    print("Model artifacts saved successfully in 'saved' directory")