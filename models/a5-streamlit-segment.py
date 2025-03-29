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
    customer_data = pd.read_csv('../Data DSA3101/customers.csv')

    # Data cleaning and processing
    customer_columns_to_convert = ['credit_default', 'credit_card', 'personal_loan', 'mortgage', 'savings_account', 'investment_product', 'auto_loan', 'wealth_management']
    customer_data[customer_columns_to_convert] = customer_data[customer_columns_to_convert].replace({'yes': 1, 'no': 0})
    customer_data = customer_data[customer_data['income'] >= 0]
    customer_data = customer_data[customer_data['age'] >= 21]
    customer_data = customer_data[customer_data['created_at'] < '2025-03-01']
    customer_data = customer_data.drop(columns=['created_at'])

    customer_df = customer_data

    # Drop customer_id and customer_segment
    customer_df = customer_df.drop('customer_id', axis=1)
    customer_df = customer_df.drop('customer_segment', axis=1)

    # Create one-hot encoding for categorical variables
    customer_df_encoded = pd.get_dummies(customer_df)

    # Standardise the data
    customer_data_scaler = StandardScaler()
    customer_scaled_data = customer_data_scaler.fit_transform(customer_df_encoded)
    customer_scaled_data.shape


        # Reduce to 40 dimensions (chosen from Scree Plot)
    ipca = IncrementalPCA(n_components=22)
    
    customer_scaled_data_ipca = ipca.fit_transform(customer_scaled_data)  # Transform data

    # Train BIRCH on historical data
    birch_customers = Birch(threshold=0.5, branching_factor=50, n_clusters=3) # Number of clusters are decided from hierarchical clustering
    birch_customers.fit(customer_scaled_data_ipca)

    return birch_customers, customer_data_scaler, ipca, customer_data, customer_df_encoded

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