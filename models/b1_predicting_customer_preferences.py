"""
Multi-Label Classification Pipeline with ClassifierChain + RandomForest
Compatible with scikit-learn >= 1.6.1 (includes `monotonic_cst`)
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import pickle
import os
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, make_scorer, f1_score

from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

# =============================================================================
# Setup
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "customer_data_with_labels_only.csv")
model_path = os.path.join(base_dir, "saved", "multi_label_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# =============================================================================
# Data Loading
# =============================================================================
df = pd.read_csv(data_path)

if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

product_cols = [
    'credit_card', 'personal_loan', 'mortgage', 'savings_account',
    'investment_product', 'auto_loan', 'wealth_management'
]

categorical_cols = ['job', 'marital', 'education', 'cluster', 'region']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df[product_cols] = df[product_cols].astype(bool)

df['days_since_acc_created'] = (
    pd.Timestamp.now() - pd.to_datetime(df['created_at'])
) / pd.Timedelta(days=1)
df.drop('created_at', axis=1, inplace=True)

feature_cols = [col for col in df.columns if col not in product_cols]
df[feature_cols] = df[feature_cols].ffill()

# =============================================================================
# Apriori-Based Bundle Feature Engineering
# =============================================================================
frequent_itemsets = apriori(df[product_cols], min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]

for _, row in frequent_itemsets.iterrows():
    bundle = sorted(row['itemsets'])
    feature_name = 'has_' + '_'.join(bundle)
    df[feature_name] = df[bundle].all(axis=1)

bundle_cols = [col for col in df.columns if col.startswith('has_')]
product_cols += bundle_cols

# =============================================================================
# Train/Test Split
# =============================================================================
X = df[feature_cols]
y = df[product_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# Model Definition: RandomForest in ClassifierChain
# =============================================================================
base_clf = RandomForestClassifier(random_state=42)
chain = ClassifierChain(base_estimator=base_clf, random_state=42)

scorer = make_scorer(f1_score, average='macro')

param_grid = {
    'base_estimator__n_estimators': [50, 100],
    'base_estimator__max_depth': [5, 10, None],
    'base_estimator__min_samples_split': [2, 5],
    'base_estimator__min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    estimator=chain,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# =============================================================================
# Evaluation
# =============================================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Best Cross-Validated Macro F1:", round(grid.best_score_, 4))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("⚠️ Hamming Loss:", round(hamming_loss(y_test, y_pred), 4))

# =============================================================================
# Save Model
# =============================================================================
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"📦 Model saved successfully to: {model_path}")
