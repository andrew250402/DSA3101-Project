import pandas as pd
import numpy as np
import pickle
import os
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, make_scorer, f1_score

from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "Data DSA3101", "customer_data_with_labels_only.csv")
data = pd.read_csv(file_path)

# Drop identifier column
if 'customer_id' in data.columns:
    data.drop('customer_id', axis=1, inplace=True)

# Define target product columns
product_cols = ['credit_card', 'personal_loan', 'mortgage', 'savings_account',
                'investment_product', 'auto_loan', 'wealth_management']

# Encode categorical features
categorical_cols = ['job', 'marital', 'education', 'cluster', 'region']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Convert product columns to boolean
data[product_cols] = data[product_cols].astype(bool)

# Feature engineering on created_at
data['days_since_acc_created'] = (pd.Timestamp.now() - pd.to_datetime(data['created_at'])) / pd.Timedelta(days=1)
data.drop('created_at', axis=1, inplace=True)

# Define feature columns
feature_cols = [col for col in data.columns if col not in product_cols]

# Fill missing values
data[feature_cols] = data[feature_cols].fillna(method='ffill')

# Apriori analysis
frequent_itemsets = apriori(data[product_cols], min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Filter frequent bundles of 2+ products
frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]

# Create engineered bundle features
for idx, row in frequent_itemsets.iterrows():
    bundle = row['itemsets']
    feature_name = 'has_' + '_'.join(sorted(bundle))
    data[feature_name] = data[list(bundle)].all(axis=1)

# Update product_cols to include bundle features
product_cols.extend([col for col in data.columns if col.startswith('has_')])

# Prepare data for multi-label classification
X = data[feature_cols]
y = data[product_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBClassifier as base
base_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
chain = ClassifierChain(base_estimator=base_clf, random_state=42)

scorer = make_scorer(f1_score, average='macro')

param_grid = {
    'base_estimator__n_estimators': [50, 100],
    'base_estimator__max_depth': [3, 5],
    'base_estimator__learning_rate': [0.01, 0.1],
    'base_estimator__subsample': [0.7, 1.0],
    'base_estimator__colsample_bytree': [0.7, 1.0]
}

grid = GridSearchCV(chain, param_grid=param_grid, scoring=scorer, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
best_chain = grid.best_estimator_
y_pred = best_chain.predict(X_test)

print("Best Parameters:")
print(grid.best_params_)
print("Best Cross-Validated Macro F1 Score:", grid.best_score_)
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))
print("Final Hamming Loss:", hamming_loss(y_test, y_pred))

# Save model to disk
file_path_model = os.path.join(base_dir, "saved", "multi_label_model.pkl")
with open(file_path_model, "wb") as f:
    pickle.dump(best_chain, f)
