import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import classification_report, hamming_loss, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Using RandomForest instead of XGBoost
import os
from mlxtend.frequent_patterns import apriori

# Configure the page
st.set_page_config(page_title="Bank Product Recommendation", layout="wide")

# Sidebar Page Navigation
page = st.sidebar.selectbox("Select Page", ["Project Flow", "Model Performance", "Apriori Analysis","Product Recommendation"])

# Path Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "Data DSA3101", "customer_data_with_labels_only.csv")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv(csv_path)
    if "customer_id" in df.columns:
        df.drop("customer_id", axis=1, inplace=True)

    product_cols = [
        "credit_card", "personal_loan", "mortgage", "savings_account",
        "investment_product", "auto_loan", "wealth_management"
    ]
    categorical_cols = ["job", "marital", "education", "cluster", "region"]

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df[product_cols] = df[product_cols].astype(bool)
    
    # Calculate days since account creation and drop the original date column
    df["days_since_acc_created"] = (pd.Timestamp.now() - pd.to_datetime(df["created_at"])) / pd.Timedelta(days=1)
    df.drop("created_at", axis=1, inplace=True)

    # Create new features based on frequent product bundles
    frequent_bundles = [
        ['credit_card', 'personal_loan'],
        ['credit_card', 'savings_account'],
        ['auto_loan', 'credit_card'],
        ['personal_loan', 'savings_account'],
        ['auto_loan', 'savings_account'],
        ['auto_loan', 'credit_card', 'savings_account']
    ]
    for bundle in frequent_bundles:
        name = 'bundle_' + '_'.join(sorted(bundle))
        df[name] = df[bundle].all(axis=1)
    product_cols += ['bundle_' + '_'.join(sorted(b)) for b in frequent_bundles]

    # Define feature columns and fill missing values
    feature_cols = [col for col in df.columns if col not in product_cols]
    df[feature_cols] = df[feature_cols].fillna(method="ffill")
    X = df[feature_cols]
    y = df[product_cols]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForest as base estimator
    base_clf = RandomForestClassifier(random_state=42)
    chain = ClassifierChain(base_estimator=base_clf, random_state=42)

    # Define a scorer and hyperparameter grid specific to RandomForest
    scorer = make_scorer(f1_score, average="macro")
    param_grid = {
        'base_estimator__n_estimators': [50, 100],
        'base_estimator__max_depth': [None, 5, 10],
        'base_estimator__min_samples_split': [2, 5]
    }

    # Grid search with cross-validation
    grid = GridSearchCV(chain, param_grid=param_grid, scoring=scorer, cv=3, verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    return best_model, X, y, X_test, y_test, y_pred, y_prob, feature_cols, product_cols, grid

(best_model, X_all, y_all, X_test, y_test, y_pred, y_prob, feature_cols, product_cols, grid) = load_and_train_model()

# ---------------------------
# Page 1: Product Recommendation
# ---------------------------
if page == "Product Recommendation":
    st.title("🔮 Customer Product Recommendation Form")
    
    # Define options for dropdowns
    jobs = ['admin', 'technician', 'services', 'management', 'unemployed', 'blue-collar', 'entrepreneur', 'self-employed', 'student', 'retired']
    maritals = ['married', 'single', 'divorced']
    educations = ['primary', 'secondary', 'tertiary']
    regions = ['Urban', 'Suburban', 'Rural']
    
    # Create a form for user input
    with st.form("customer_form"):
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 52)
            income = st.slider("Income", 0, 10000, 1200, step=100)
            tenure = st.slider("Tenure (years)", 0, 30, 5)
        with col2:
            days_since_acc_created = st.slider("Days Since Account Created", 0, 10000, 1000)
    
        st.subheader("Demographics")
        job = st.selectbox("Job", options=jobs)
        marital = st.selectbox("Marital Status", options=maritals)
        education = st.selectbox("Education", options=educations)
        region = st.selectbox("Region", options=regions)
    
        st.subheader("Financial Products (Already Owned)")
        col3, col4, col5 = st.columns(3)
        with col3:
            credit_default = st.radio("Credit Default", options=["no", "yes"])
            credit_card = st.radio("Has Credit Card", options=["no", "yes"])
            auto_loan = st.radio("Has Auto Loan", options=["no", "yes"])
        with col4:
            personal_loan = st.radio("Has Personal Loan", options=["no", "yes"])
            mortgage = st.radio("Has Mortgage", options=["no", "yes"])
            wealth_management = st.radio("Uses Wealth Management", options=["no", "yes"])
        with col5:
            savings_account = st.radio("Has Savings Account", options=["no", "yes"])
            investment_product = st.radio("Has Investment Product", options=["no", "yes"])
    
        submitted = st.form_submit_button("🎯 Predict Recommended Products")
    
    if submitted:
        # Build input dictionary from form data
        input_dict = {
            "age": age,
            "income": income,
            "days_since_acc_created": days_since_acc_created,
            "tenure": tenure,
            "credit_default": 1 if credit_default == "yes" else 0,
            "credit_card": 1 if credit_card == "yes" else 0,
            "auto_loan": 1 if auto_loan == "yes" else 0,
            "personal_loan": 1 if personal_loan == "yes" else 0,
            "mortgage": 1 if mortgage == "yes" else 0,
            "wealth_management": 1 if wealth_management == "yes" else 0,
            "savings_account": 1 if savings_account == "yes" else 0,
            "investment_product": 1 if investment_product == "yes" else 0,
        }

        # One-hot encode categorical fields based on training feature columns
        for col in feature_cols:
            if col.startswith("job_") and col == f"job_{job}":
                input_dict[col] = 1
            elif col.startswith("marital_") and col == f"marital_{marital}":
                input_dict[col] = 1
            elif col.startswith("education_") and col == f"education_{education}":
                input_dict[col] = 1
            elif col.startswith("region_") and col == f"region_{region}":
                input_dict[col] = 1
            elif col not in input_dict:
                input_dict[col] = 0

        # Prepare input DataFrame ensuring the same column order as in training
        input_df = pd.DataFrame([input_dict])[feature_cols]

        # Predict probabilities for each product
        probs = best_model.predict_proba(input_df)  
        # Extract the single row of probabilities
        prob_vector = probs[0]

        # Create a DataFrame mapping each product to its probability
        result_df = pd.DataFrame({
            "Product": product_cols,   # Assumes product_cols order matches the model's output
            "Probability": prob_vector
        }).sort_values(by="Probability", ascending=False)

        # Split products into individual and bundle categories
        result_individual = result_df[~result_df["Product"].str.startswith("bundle")].reset_index(drop=True)
        result_bundle = result_df[result_df["Product"].str.startswith("bundle")].reset_index(drop=True)

        # Display two separate tables
        st.subheader("Individual Products")
        st.dataframe(result_individual)

        st.subheader("Bundled Products")
        st.dataframe(result_bundle)

elif page == "Model Performance":
    st.title("📈 Model Training and Test Results")
    
    st.markdown("---")
    st.markdown(
        "This section displays the **training** and **test** performance of the model using **GridSearchCV** "
        "and a **ClassifierChain** with a **XGBoost** base estimator. Below, you will find key performance metrics, "
        "a detailed classification report, and a legend mapping product indices to individual products and bundles."
    )
    
    
    # Detailed Classification Report in an Expander
    st.subheader("Classification Report on Test Set")
    with st.expander("Click to view the full classification report"):
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)
    
    st.markdown("---")
    
    # Legend: Mapping Product Indices to Names
    st.subheader("Legend: Product Index Mapping")
    individual_products = [
        "credit_card", "personal_loan", "mortgage", "savings_account",
        "investment_product", "auto_loan", "wealth_management"
    ]
    bundle_products = [
        "bundle_credit_card_personal_loan",
        "bundle_credit_card_savings_account",
        "bundle_auto_loan_credit_card",
        "bundle_personal_loan_savings_account",
        "bundle_auto_loan_savings_account",
        "bundle_auto_loan_credit_card_savings_account"
    ]
    all_products = individual_products + bundle_products  # total of 13 items (indices 0 to 12)
    legend_lines = [f"**{i}**: {name}" for i, name in enumerate(all_products)]
    legend_text = "<br>".join(legend_lines)
    st.markdown(legend_text, unsafe_allow_html=True)


elif page == "Apriori Analysis":
    st.title("📊 Apriori Analysis Results")
    
    # Load the CSV file (reuse csv_path from earlier)
    df_apriori = pd.read_csv(csv_path)
    if "customer_id" in df_apriori.columns:
        df_apriori.drop("customer_id", axis=1, inplace=True)
    
    # Define the product columns
    product_cols_apriori = [
        "credit_card", "personal_loan", "mortgage", "savings_account",
        "investment_product", "auto_loan", "wealth_management"
    ]
    df_apriori[product_cols_apriori] = df_apriori[product_cols_apriori].astype(bool)
    
    # Compute frequent itemsets using mlxtend's apriori
    from mlxtend.frequent_patterns import apriori, association_rules
    frequent_itemsets = apriori(df_apriori[product_cols_apriori], min_support=0.05, use_colnames=True)


    # Sort by support (descending order)
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False)

    
    # Display the frequent itemsets table
    st.subheader("Frequent Itemsets (Sorted by Support)")
    st.dataframe(frequent_itemsets_sorted)
    


elif page == "Project Flow":
    st.title("🔍 Project Flow: From Apriori to Machine Learning")

    st.markdown("### Overview")
    st.markdown(
        "This project integrates **association rule mining** (using the **Apriori algorithm**) with **machine learning** to build a multi-label product recommendation system. "
        "The flow below outlines the process from data collection to model deployment."
    )

    flow_diagram = """
    digraph {
        // Layout: left-to-right
        rankdir=UD;
        node [shape=box, style=filled, color=lightgreen, fontsize=13];
        nodesep=0.5;
        ranksep=1.2;

        // --- TOP ROW NODES ---
        DataCollection [label="Data Collection & Integration"];
        Preprocessing [label="Data Preprocessing\\n(One-hot Encoding, Time Features)"];
        Apriori [label="Apriori Analysis\\n(Frequent Itemsets)"];
        FeatureEngineering [label="Feature Engineering\\n(Bundle Features)"];

        // --- BOTTOM ROW NODES ---
        // We insert an invisible blank node in the leftmost column to align columns properly
        blank1 [label="", style=invis, width=0.1];
        Prediction [label="Product Recommendation\\n(Predict & Visualize)"];
        Evaluation [label="Model Evaluation\\n(Classification Report, Hamming Loss)"];
        ModelTraining [label="Model Training\\n(ClassifierChain, GridSearchCV)"];

        // Force each row's nodes to share the same rank
        { rank = same; DataCollection; Preprocessing; Apriori; FeatureEngineering }
        { rank = same; blank1; Prediction; Evaluation; ModelTraining }

        // --- EDGES: TOP ROW (left → right) ---
        DataCollection -> Preprocessing
        Preprocessing -> Apriori
        Apriori -> FeatureEngineering

        // --- EDGES: DOWN from top row col4 to bottom row col4 ---
        FeatureEngineering -> ModelTraining

        // --- EDGES: BOTTOM ROW (right → left visually, but "ModelTraining -> Evaluation -> Prediction" in code) ---
        ModelTraining -> Evaluation
        Evaluation -> Prediction

        // --- INVISIBLE EDGES to align columns ---
        DataCollection -> blank1 [style=invis]
        Preprocessing -> Prediction [style=invis]
        Apriori -> Evaluation [style=invis]
    }
    """




    st.graphviz_chart(flow_diagram)

    st.markdown("### Detailed Process Description")
    st.markdown(
        """
    **1. Data Collection & Integration:**  
    - **Collect** raw data such as customer details and their product ownership.

    **2. Data Preprocessing:**  
    - **Clean** and **transform** the raw data.  
    - Apply **one-hot encoding** to categorical variables and compute **time-based features** (e.g., days since account creation).

    **3. Apriori Analysis:**  
    - Run the **Apriori algorithm** on the product usage data to extract **frequent itemsets**.  
    - Identify common product bundles to inform the next step.

    **4. Feature Engineering:**  
    - Create new binary features based on the frequent itemsets discovered, representing whether a customer holds a specific product bundle.

    **5. Model Training:**  
    - Build a **multi-label classification** model using a **ClassifierChain** with a base estimator (e.g., **Random Forest**).  
    - Use **GridSearchCV** for **hyperparameter tuning** to optimize model performance.

    **6. Model Evaluation:**  
    - Evaluate the model using metrics such as **Recall** and **F1 Score**.  
    - Generate a **classification report** to assess test performance.

    **7. Product Recommendation:**  
    - Deploy the model to predict and recommend financial products for individual customers based on their inputs.  
    - Visualize prediction probabilities to aid in decision-making.
        """
    )

    st.markdown(
        "This systematic approach ensures that insights from **association rule mining** are effectively integrated with **predictive modeling** to enhance the product recommendation process."
<<<<<<< Updated upstream
    )
=======
    )
>>>>>>> Stashed changes
