import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os
from datetime import datetime

def load_data_and_model():
    """
    Loads and preprocesses customer data, then trains a logistic regression model for churn prediction.
    Returns:
        tuple: (feature_columns, trained_model, scaler_object, cluster_column_names)
    """
    # Load raw data files
    df_customers = pd.read_csv("../data/customers.csv")  # Customer demographics
    df_churn = pd.read_csv("../data/churn.csv")          # Churn status labels
    df_transactions = pd.read_csv("../data/transactions_summary.csv")  # Transaction history
    df_customers_clustered = pd.read_csv("../data/customer_data_with_labels_only.csv")  # Pre-computed clusters
    
    # Merge datasets on customer_id
    df = pd.merge(df_customers, df_churn, on='customer_id')
    df = df.merge(df_customers_clustered[['customer_id', 'cluster']], on='customer_id', how='left')
    df = df.dropna(subset=['cluster'])  # Remove customers without cluster assignments

    # Feature Engineering
    # ------------------
    # Create account activity flag (inactive if no transactions in 180+ days)
    df['account_active'] = df_transactions['days_since_last_transaction'].apply(lambda x: 0 if x > 180 else 1)
    
    # Convert yes/no columns to binary (1/0)
    binary_columns = df.columns[10:17]  # Columns like credit_card, mortgage etc.
    df[binary_columns] = df[binary_columns].map(lambda x: 1 if x == 'yes' else 0)
    df['churn_status'] = df['churn_status'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['credit_default'] = df['credit_default'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Create aggregated features
    df['total_products_purchased'] = df[binary_columns].sum(axis=1)  # Sum of all financial products
    
    # Calculate tenure in years
    df['created_at'] = pd.to_datetime(df['created_at'])
    current_date = datetime.now()
    df['tenure'] = (current_date - df['created_at']).apply(lambda x: np.ceil(x.days / 365)).astype(int)
    
    # Create utilization rate features
    # To explore the relationship between the number of products purchased and income/tenure
    df['product_utilization_rate_by_income'] = df.apply(
        lambda row: 0 if row['income'] == 0 else row['total_products_purchased'] / row['income'], axis=1)
    df['product_utilization_rate_by_tenure'] = df.apply(
        lambda row: 0 if row['tenure'] == 0 else row['total_products_purchased'] / row['tenure'], axis=1)
    
    # One-hot encode cluster labels to better understand customer segments
    # This will create binary columns for each cluster label
    df_encoded = pd.get_dummies(df['cluster'], prefix='cluster')
    df = pd.concat([df, df_encoded], axis=1)
    
    # Prepare model features
    y = df["churn_status"]  # Target variable
    # Drop non-feature columns, including a few categorical and identifier columns
    # The columns to drop are identified based on the initial data exploration
    X = df.drop(columns=['customer_id','churn_status', 'job', 'marital', 'education', 
                        'customer_segment', 'region', 'created_at', 'churn_id','churn_date', 'cluster'])
    
    # Model Training
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    
    # Standardize numerical features
    sc = StandardScaler()
    num_cols = X.columns  # All columns are numerical post-processing
    X_train[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test[num_cols] = sc.transform(X_test[num_cols])
    
    # Train logistic regression with regularization (C=0.1) and solver (newton-cg)
    # ratios like product utilization rate can have high variance, so regularization is important
    # The solver 'newton-cg' is converges reliably even when churn classes are imbalanced
    logr_model = LogisticRegression(C=0.1, solver='newton-cg')
    logr_model.fit(X_train, y_train)
    
    return num_cols, logr_model, sc, df_encoded.columns.tolist()

def save_model_artifacts():
    """
    Saves trained model, scaler, and feature metadata to a pickle file.
    Output file: saved/b5-customer-retention-strategies.pkl
    """
    # Create directory if missing
    os.makedirs('saved', exist_ok=True)
    
    # Load trained artifacts
    num_cols, logr_model, sc, cluster_columns = load_data_and_model()
    
    # Serialize objects
    with open('saved/b5-customer-retention-strategies.pkl', 'wb') as f:
        pickle.dump({
            'logr_model': logr_model,      # Trained model
            'scaler': sc,                  # StandardScaler instance
            'num_cols': num_cols,          # Feature column names
            'cluster_columns': cluster_columns  # One-hot encoded cluster columns
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Execute model training and saving when run directly
    save_model_artifacts()
    print("Model artifacts saved successfully in 'saved' directory")