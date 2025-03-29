import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE



# loading in relevant datasets, preprocessing data/feature engineering

def load_and_preprocess_data():
    # loading relevant data
    campaigns = pd.read_csv("../Data DSA3101/campaigns.csv")
    customer_engagement = pd.read_csv("../Data DSA3101/customer_engagement.csv")
    customers = pd.read_csv("../Data DSA3101/customers.csv")
    transactions = pd.read_csv("../Data DSA3101/transactions_summary.csv")

    # feature engineering conversion_rate
    # calculate total engagements (sent = "Yes"), engagement is irregardless of active/passive participation
    total_engagements = (
        customer_engagement[customer_engagement["sent"] == "Yes"]
        .groupby("campaign_id")
        .size().reset_index(name = "total_engagements")
    )
    # calculate successful conversions
    successful_conversions = (
        customer_engagement[customer_engagement["conversion_status"] == "Yes"]
        .groupby("campaign_id")
        .size().reset_index(name="successful_conversions")
    )
    # merge data calculate conversion_rate
    conversion_data = total_engagements.merge(successful_conversions, on="campaign_id", how="left")
    conversion_data["successful_conversions"] = conversion_data["successful_conversions"].fillna(0)
    conversion_data["conversion_rate"] = conversion_data["successful_conversions"] / conversion_data["total_engagements"]

    # feature engineering acquisition_cost
    acquisition_data = conversion_data.merge(campaigns[["campaign_id","total_campaign_cost"]], on="campaign_id")
    #computing acquisition_cost
    acquisition_data["acquisition_cost"] = acquisition_data["total_campaign_cost"] / acquisition_data["successful_conversions"]

    # feature engineering avg_targeted_clv
    # merging customers and transaction data
    customers = customers.merge(transactions, on = "customer_id", how = "left")

    # assign simplified churn probability based on days_since_last_transaction
    churn_threshold = 90
    customers["churn_prob"] = np.where(customers["days_since_last_transaction"] > churn_threshold, 0.8, 0.2)

    # calculate estimated lifetime for each customer using simplified churn probability (in years)
    customers["expected_lifetime"] = 1/customers["churn_prob"]

    # calculate clv of each customer
    customers["CLV"] = customers["expected_lifetime"] * customers ["average_transaction_amount_12m"]

    # obtain clv of targeted customers for each campaign (avg_targeted_clv)
    engagement_clv = customer_engagement.merge(customers[["customer_id","CLV"]], on = "customer_id", how = "left")
    campaign_clv = engagement_clv.groupby("campaign_id")["CLV"].mean().reset_index()
    campaign_clv.rename(columns = {"CLV": "avg_targeted_clv"}, inplace = True)

    # generate ROI for each campaign
    roi_data = campaigns[["campaign_id","total_campaign_cost","total_revenue_generated"]]
    roi_data["roi"] = (roi_data["total_revenue_generated"] - roi_data["total_campaign_cost"])/roi_data["total_campaign_cost"]

    # data preparation for machine learning
    # one-hot enconding for categorical variables
    campaigns_encoded = pd.get_dummies(campaigns, columns = ["campaign_type"], drop_first = True) # campaign type
    
    campaigns_encoded.rename(columns = {"recommended_product_name": "product"}, inplace = True)
    campaigns_encoded = pd.get_dummies(campaigns_encoded, columns=['product'], drop_first=True) # product

    campaigns_encoded = pd.get_dummies(campaigns_encoded, columns = ['customer_segment'], drop_first = True) # customer segment
    
    #calculate campaign duration and add it into the dataset
    campaigns_encoded["start_date"] = pd.to_datetime(campaigns_encoded["start_date"])
    campaigns_encoded["end_date"] = pd.to_datetime(campaigns_encoded["end_date"])
    
    campaigns_encoded["campaign_duration"] = (campaigns_encoded["end_date"] - campaigns_encoded["start_date"]).dt.days
        
    # merge campaign data with feature engineered columns
    campaigns_encoded = campaigns_encoded.merge(conversion_data[["campaign_id","conversion_rate"]], on = "campaign_id", how = "left")
    campaigns_encoded = campaigns_encoded.merge(acquisition_data[["campaign_id","acquisition_cost"]], on = "campaign_id", how = "left")
    campaigns_encoded = campaigns_encoded.merge(campaign_clv[["campaign_id","avg_targeted_clv"]], on = "campaign_id", how = "left")

    # define categories for ROI
    bins = [0, 0.67, 1.33, float('inf')] # Define ROI thresholds
    labels = ['Low', 'Medium', 'High']  # Define ROI categories
    campaigns_encoded['roi_category'] = pd.cut(roi_data["roi"], bins=bins, labels=labels)

    return campaigns_encoded

def selected_features():
    # features (column names) for easier accessibility. features were selected based on feature importances
    features = ['total_campaign_cost',
                         'campaign_duration',
                         'conversion_rate',
                         'acquisition_cost',
                         'customer_segment_Retired',
                         'customer_segment_Young Professionals',
                         'product_Credit Card',
                         'product_Personal Loan',
                         'campaign_type_Email',
                         'campaign_type_Mobile App Notifications',
                         'campaign_type_SMS']
    return features
    
def train_model():
    data = load_and_preprocess_data()

    # select features and target variable (roi_category)
    X = data[selected_features()]
    y = data["roi_category"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # train model using best parameters from GridSearchCV
    pipeline = make_pipeline(
        SMOTE(k_neighbors=2,random_state=3),
        GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=5,
            n_estimators=50,
            random_state=1)
    )
    
    best_model = pipeline.fit(X_train, y_train)
    return best_model

def evaluate_model(model):
    data = load_and_preprocess_data()
    X = data[selected_features()]
    y = data["roi_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    y_pred = model.predict(X_test)
    
    labels = ['Low', 'Medium', 'High'] 
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

def save_model():
    model_path = "saved/b3-measuring-campaign-roi.pkl"
    model = train_model()

    with open(model_path, "wb") as f:
        pickle.dump(model,f)

    print(f"Model trained and saved at {model_path}")
    
