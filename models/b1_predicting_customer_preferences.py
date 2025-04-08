"""
This script performs a **multi-label classification** task on customer data.
It uses **pandas** for data manipulation, **mlxtend** for Apriori analysis to extract frequent itemsets,
and **XGBClassifier** wrapped in a **ClassifierChain** for predicting multiple target labels simultaneously.
The workflow includes data loading, preprocessing, feature engineering, model training with hyperparameter tuning using **GridSearchCV**, evaluation, and model saving.
"""

# =============================================================================
# Import Necessary Libraries and Modules
# =============================================================================
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier  # Note: This import is unused in the current script.
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, make_scorer, f1_score

from mlxtend.frequent_patterns import apriori, association_rules

# =============================================================================
# Data Loading and Initial Setup
# =============================================================================
"""
Load the dataset containing customer information.
The file path is constructed relative to the current file's directory.
The CSV file is then read into a pandas DataFrame for further processing.
"""
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "Data DSA3101", "customer_data_with_labels_only.csv")
data = pd.read_csv(file_path)

# =============================================================================
# Data Preprocessing
# =============================================================================
"""
Remove the identifier column 'customer_id' if it exists, as it is not informative for the model.
"""
if 'customer_id' in data.columns:
    data.drop('customer_id', axis=1, inplace=True)

"""
Define the target product columns that represent various financial products.
These will be used as labels in the multi-label classification.
"""
product_cols = ['credit_card', 'personal_loan', 'mortgage', 'savings_account',
                'investment_product', 'auto_loan', 'wealth_management']

"""
Encode categorical features into numerical format using one-hot encoding.
Dropping the first category helps to avoid the **dummy variable trap**.
"""
categorical_cols = ['job', 'marital', 'education', 'cluster', 'region']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

"""
Convert the product columns to boolean type to ensure binary representation of product ownership.
"""
data[product_cols] = data[product_cols].astype(bool)

"""
Perform **feature engineering** on the 'created_at' column:
- Compute 'days_since_acc_created' by calculating the difference between the current timestamp and the account creation date.
- Drop the original 'created_at' column since the new feature now encapsulates the needed information.
"""
data['days_since_acc_created'] = (pd.Timestamp.now() - pd.to_datetime(data['created_at'])) / pd.Timedelta(days=1)
data.drop('created_at', axis=1, inplace=True)

"""
Define the feature columns (X) by excluding the target product columns.
These features will serve as the input for the classification model.
"""
feature_cols = [col for col in data.columns if col not in product_cols]

"""
Fill missing values in the feature columns using the **forward fill** method,
ensuring that any gaps in the data are appropriately addressed.
"""
data[feature_cols] = data[feature_cols].fillna(method='ffill')

# =============================================================================
# Apriori Analysis for Bundle Feature Engineering
# =============================================================================
"""
Apply the **Apriori algorithm** to the product columns to discover frequent itemsets with a minimum support of 0.5.
Then, derive **association rules** with a minimum confidence threshold of 0.7.
We chose this method as applying the Apriori algorithm helps uncover frequent product combinations, 
while association rules identify strong purchase relationships, enabling better recommendations and targeted marketing. 
"""
frequent_itemsets = apriori(data[product_cols], min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

"""
Filter the frequent itemsets to retain only those with two or more items,
as these represent potential product bundles that can provide additional insights.
"""
frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]

"""
Create new engineered features for each detected product bundle.
For each bundle, a new column is added that indicates if a customer owns all products within that bundle.
"""
for idx, row in frequent_itemsets.iterrows():
    bundle = row['itemsets']
    feature_name = 'has_' + '_'.join(sorted(bundle))
    data[feature_name] = data[list(bundle)].all(axis=1)

"""
Update the target product columns list to include the newly created bundle features.
This ensures that the model will predict both individual products and product bundles.
"""
product_cols.extend([col for col in data.columns if col.startswith('has_')])

# =============================================================================
# Preparing Data for Multi-Label Classification
# =============================================================================
"""
Separate the dataset into features (X) and target labels (y).
Then, split the data into training and testing sets using an 80/20 ratio,
ensuring reproducibility with a fixed random state.
"""
X = data[feature_cols]
y = data[product_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Model Setup and Hyperparameter Tuning using GridSearchCV
# =============================================================================
"""
Initialize an **XGBClassifier** as the base estimator for the **ClassifierChain**.
ClassifierChain is employed to handle the multi-label classification scenario.
We chose this model as ClassifierChain with XGBClassifier improves prediction accuracy by 
modeling interdependencies between customer preferences, making it superior to independent multi-label approaches
A **macro F1 score** is defined as the evaluation metric to provide balanced performance across labels.
Hyperparameters for the base estimator are tuned using **GridSearchCV** with 3-fold cross-validation.
"""
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

# =============================================================================
# Model Evaluation and Saving
# =============================================================================
"""
After hyperparameter tuning, retrieve the best model from **GridSearchCV** and predict on the test set.
The performance is evaluated using the **classification_report** (which includes precision, recall, and F1 score)
and **hamming_loss**, which measures the fraction of labels that are incorrectly predicted.
"""
best_chain = grid.best_estimator_
y_pred = best_chain.predict(X_test)

print("Best Parameters:")
print(grid.best_params_)
print("Best Cross-Validated Macro F1 Score:", grid.best_score_)
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))
print("Final Hamming Loss:", hamming_loss(y_test, y_pred))

"""
Save the best-performing model to disk using **pickle**.
This allows for later deployment or further evaluation without retraining.
The model is stored in a 'saved' directory relative to the base directory.
"""
file_path_model = os.path.join(base_dir, "saved", "multi_label_model.pkl")
with open(file_path_model, "wb") as f:
    pickle.dump(best_chain, f)
