import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os
from datetime import datetime

def load_data_and_model():
    df_customers = pd.read_csv("../Data DSA3101/customers.csv")
    df_churn = pd.read_csv("../Data DSA3101/churn.csv")
    df_transactions = pd.read_csv("../Data DSA3101/transactions_summary.csv")
    df_customers_clustered = pd.read_csv("../Data DSA3101/customer_data_with_labels_only.csv")
    
    # Merge and preprocess data
    df = pd.merge(df_customers, df_churn, on='customer_id')
    df = df.merge(df_customers_clustered[['customer_id', 'cluster']], on='customer_id', how='left')
    df = df.dropna(subset=['cluster'])
    
    # Create features
    df['account_active'] = df_transactions['days_since_last_transaction'].apply(lambda x: 0 if x > 180 else 1)
    binary_columns = df.columns[10:17]
    df[binary_columns] = df[binary_columns].map(lambda x: 1 if x == 'yes' else 0)
    df['churn_status'] = df['churn_status'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['credit_default'] = df['credit_default'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['total_products_purchased'] = df[binary_columns].sum(axis=1)
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    current_date = datetime.now()
    df['tenure'] = (current_date - df['created_at']).apply(lambda x: np.ceil(x.days / 365)).astype(int)
    df['product_utilization_rate_by_income'] = df.apply(
            lambda row: 0 if row['income'] == 0 else row['total_products_purchased'] / row['income'], axis=1)
    df['product_utilization_rate_by_tenure'] = df.apply(
            lambda row: 0 if row['tenure'] == 0 else row['total_products_purchased'] / row['tenure'], axis=1)
    
    # Encode clusters
    df_encoded = pd.get_dummies(df['cluster'], prefix='cluster')
    df = pd.concat([df, df_encoded], axis=1)
    
    # Prepare model data
    y = df["churn_status"]
    X = df.drop(columns=['customer_id','churn_status', 'job', 'marital', 'education', 
                        'customer_segment', 'region', 'created_at', 'churn_id','churn_date', 'cluster'])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    sc = StandardScaler()
    num_cols = X.columns
    X_train[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test[num_cols] = sc.transform(X_test[num_cols])
    
    logr_model = LogisticRegression(C=0.1, solver='newton-cg')
    logr_model.fit(X_train, y_train)
    
    return num_cols, logr_model, sc, df_encoded.columns.tolist()

def save_model_artifacts():
    os.makedirs('saved', exist_ok=True)
    num_cols, logr_model, sc, cluster_columns = load_data_and_model()
    with open('saved/b5-customer-retention-strategies.pkl', 'wb') as f:
        pickle.dump({
            'logr_model': logr_model,
            'scaler': sc,
            'num_cols': num_cols,
            'cluster_columns': cluster_columns
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_model_artifacts()
    print("Model artifacts saved successfully in 'saved' directory")