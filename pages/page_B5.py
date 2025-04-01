import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_utilities import read_model, read_image

def plot_churn_likelihood(churn_prob):
    # Calculate probabilities
    retention_prob = 1 - churn_prob
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 1), facecolor='white')
    
    # Create single horizontal bar
    ax.barh([''], [1], color='lightgray', alpha=0.3)  # Background bar
    ax.barh([''], [churn_prob], color='salmon', label=f'Churn ({churn_prob:.1%})')
    ax.barh([''], [retention_prob], left=churn_prob, 
            color='lightgreen', label=f'Retain ({retention_prob:.1%})')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')

    # Customize plot
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.1)])
    
    # Remove spines and y-axis label
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig
# Configure page
st.set_page_config(page_title="Logistic Regression for Scalable Churn Prediction", layout="wide")

# Title and introduction
st.title("🚀 Why Logistic Regression Works for Scalable Churn Prediction")
st.markdown("""
<div style='color: #aaa; margin-bottom: 20px;'>
Key advantages of logistic regression for enterprise-scale churn prediction
</div>
""", unsafe_allow_html=True)

# Create the comparison table
data = {
    "Feature": [
        "Computational Efficiency",
        "Memory Footprint", 
        "Prediction Speed",
        "Business Interpretability"
    ],
    "Logistic Regression 📊": [
        "✅ <strong>Relatively fast training</strong> (converges in few iterations)",
        "✅ <strong>Minimal storage</strong> (just stores coefficient weights)",
        "✅ <strong>Instant predictions</strong> (single dot product + sigmoid)",
        "✅ <strong>Clear feature importance</strong> (direct from coefficients)"
    ],
    "Impact on Churn Prediction 🔍": [
        "🔄 <strong>Processes millions of records</strong> with minimal compute",
        "📉 <strong>O(n*d) space complexity</strong> (d = features, n = datapoints) scales easily   ",
        "🔄 <strong>Adapts to changing behavior</strong> without full retraining",
        "💡 <strong>Explainable predictions</strong> allow for simpler inference"
    ]
}

df = pd.DataFrame(data)

# Generate HTML with Pandas Styler
html = df.style.set_properties(**{
    'font-size': '16pt',
    'padding': '12px',
    'color': '#ffffff',
    'background-color': 'transparent'
}).to_html(escape=False)

# Dark theme CSS with your exact styling
html = f"""
<style>
    table {{
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        color: #ffffff !important;
        background-color: transparent !important;
    }}
    table th {{
        font-weight: bold !important;
        background-color: #333333 !important;
        text-align: left;
        padding: 12px;
        border: 1px solid #444444 !important;
    }}
    table td {{
        padding: 12px;
        border-bottom: 1px solid #444444 !important;
        background-color: #222222 !important;
    }}
    table tr:hover td {{
        background-color: #2a2a2a !important;
    }}
    .dataframe {{
        background-color: transparent !important;
    }}
    strong {{
        color: #4fc3f7 !important;
    }}
</style>
{html}
"""

st.markdown(html, unsafe_allow_html=True)

# Additional notes with matching style
st.markdown("""
<style>
    .highlight {{
        color: #4fc3f7;
        font-weight: bold;
    }}
</style>

<div style='margin-top: 30px; color: #aaa;'>
""", unsafe_allow_html=True)

st.image(read_image("B5_churn_coefficient.png"))

def load_data_and_model():
    seg_model = read_model('a5-streamlit-segment.pkl')
    logr_model = read_model('b5-customer-retention-strategies.pkl')
    segment, logr = ([seg_model['birch_model'], seg_model['scaler'], seg_model['ipca_model'], seg_model['customers'], seg_model['customer_encoded']],
                    [logr_model['num_cols'], logr_model['logr_model'], logr_model['scaler'], logr_model['cluster_columns']])
    return segment, logr

# Load model and data
birch_customers, customer_data_scaler, ipca, customer_data, customer_df_encoded = load_data_and_model()[0]
n, logr_model, sc, cluster_columns = load_data_and_model()[1]

# Streamlit app
st.title("Customer Behavior Segmentation and Churn Prediction")
st.subheader("Enter New Customer Details")

# Get unique values from original data for dropdowns
jobs = customer_data['job'].unique()
maritals = customer_data['marital'].unique()
educations = customer_data['education'].unique()
regions = customer_data['region'].unique()

def inject_css():
    st.markdown("""
    <style>
        /* Targets ALL widget labels */
        div[data-testid="stWidgetLabel"] > label, 
        div[data-testid="stWidgetLabel"] > p,
        div[data-testid="stMarkdownContainer"] > p {
            font-size: 22px !important;
        }

    </style>
    """, unsafe_allow_html=True)

# Call this before AND after your form
inject_css()

# Create form for user input
with st.form("customer_form"):
    # Numerical inputs
    age = st.slider("Age", min_value=18, max_value=100, value=52, step=1)
    income = st.slider("Income", min_value=0, max_value=10000, value=1200, step=100)
    
    # Categorical inputs
    job = st.selectbox("Job", options=jobs,placeholder= "student")
    marital = st.selectbox("Marital Status", options=maritals, placeholder= "married")
    education = st.selectbox("Education", options=educations, placeholder= "tertiary")
    region = st.selectbox("Region", options=regions, placeholder= "Suburban")

    st.write("Financial Products:")
    col1, col2, col3 = st.columns(3)
    # Binary inputs
    with col1:
        credit_default = st.radio("Credit Default", options=['no', 'yes'])
        credit_card = st.radio("Has Credit Card", options=['no', 'yes'])
        auto_loan = st.radio("Has Auto Loan", options=['no', 'yes'])
    
    with col2:
        personal_loan = st.radio("Has Personal Loan", options=['no', 'yes'])
        mortgage = st.radio("Has Mortgage", options=['no', 'yes'])
        wealth_management = st.radio("Uses Wealth Management", options=['no', 'yes'])
    
    with col3:
        savings_account = st.radio("Has Savings Account", options=['no', 'yes'])
        investment_product = st.radio("Has Investment Product", options=['no', 'yes'])
    
    st.subheader("Account Status")
    col4, col5 = st.columns(2)
    with col4:
        account_active = st.radio("Is the account active?", options=['no', 'yes'])
    with col5:
        tenure = st.slider("Tenure", min_value=0, max_value=18, value=5, step=1)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create new customer DataFrame
    new_customer = pd.DataFrame([[
        50, age, job, marital, education, credit_default, "Budget-Conscious", region, income,
        credit_card, personal_loan, mortgage, savings_account, 
        investment_product, auto_loan, wealth_management
    ]], columns=customer_data.columns)
    # Include other customer data to create a sufficient batch size for IPCA

    # Convert yes/no to 1/0
    customer_columns_to_convert = [
        'credit_default', 'credit_card', 'personal_loan', 'mortgage', 
        'savings_account', 'investment_product', 'auto_loan', 'wealth_management'
    ]
    
    customer_df = pd.concat([new_customer, customer_data[0:50]], axis=0)
    customer_df[customer_columns_to_convert] = customer_df[customer_columns_to_convert].replace({'yes': 1, 'no': 0})
    
    # One-hot encode
    new_customer_encoded = pd.get_dummies(customer_df).reindex(columns=customer_df_encoded.columns, fill_value=0)
    
    # Scale and transform
    new_customer_scaled = customer_data_scaler.transform(new_customer_encoded)
    new_customer_ipca = ipca.transform(new_customer_scaled)
    
    # Predict cluster
    cluster = birch_customers.predict(new_customer_ipca)[0]

    # Define your cluster mapping
    CLUSTER_MAPPING = {
        0: "Conservative Savers",
        1: "High-Value Customers", 
        2: "Everyday Spenders"
    }

    # Display the result
    cluster_name = CLUSTER_MAPPING.get(cluster, "Unknown Cluster")
    st.success(f"""
    ### 🎯 Customer Segment Identified  
    **Cluster {cluster}: {cluster_name}**  
    *This customer belongs to the {cluster_name} group*
    """)

    # Add cluster-specific advice
    CLUSTER_ADVICE = {
        0: """
        <div style='font-size:22px'>
        <b>🎯 Conservative Savers Strategy</b><br>
        - 📈 Push investment products & credit cards<br>
        - 💳 Offer cashback/discounts to increase spending<br>
        - 📱 Prioritize mobile app engagement<br>
        - 🏆 Low-risk investment bundles<br>
        </div>
        """,
        
        1: """
        <div style='font-size:22px'>
        <b>✨ High-Value Customers Strategy</b><br>
        - 💰 Promote personal loans with competitive rates<br>
        - ✈️ Travel miles/rewards for high spending<br>
        - ⏳ Priority service & wealth management<br>
        - 📲 VIP mobile features<br>
        </div>
        """,
        
        2: """
        <div style='font-size:22px'>
        <b>🛒 Everyday Spenders Strategy</b><br>
        - 📊 Investment product education<br>
        - 🎯 Threshold-based spending incentives<br>
        - 🔄 Usage-based app recommendations<br>
        - 💡 Bundle product offers<br>
        </div>
        """
    }

    cluster = f"{birch_customers.predict(new_customer_ipca)[0]}.0"
    
    def yes_no_to_binary(value):
        return 1 if value == "yes" else 0

    total_products_purchased = sum([
        yes_no_to_binary(credit_card),
        yes_no_to_binary(personal_loan),
        yes_no_to_binary(mortgage),
        yes_no_to_binary(savings_account),
        yes_no_to_binary(investment_product),
        yes_no_to_binary(auto_loan),
        yes_no_to_binary(wealth_management)
    ])
    
    # Create a template DataFrame with ALL features the model expects
    # Initialize all features to 0 first
    input_data = pd.DataFrame(columns=logr_model.feature_names_in_).fillna(0)
    
    # Now fill in the values we collected from the form
    input_data['age'] = [age]
    input_data['credit_default'] = [yes_no_to_binary(credit_default)]
    input_data['income'] = [income]
    input_data['credit_card'] = [yes_no_to_binary(credit_card)]
    input_data['personal_loan'] = [yes_no_to_binary(personal_loan)]
    input_data['mortgage'] = [yes_no_to_binary(mortgage)]
    input_data['savings_account'] = [yes_no_to_binary(savings_account)]
    input_data['investment_product'] = [yes_no_to_binary(investment_product)]
    input_data['auto_loan'] = [yes_no_to_binary(auto_loan)]
    input_data['wealth_management'] = [yes_no_to_binary(wealth_management)]
    input_data['account_active'] = [yes_no_to_binary(account_active)]
    input_data["total_products_purchased"] = [total_products_purchased]
    input_data['tenure'] = [tenure]
    input_data['product_utilization_rate_by_income'] = [total_products_purchased / income if income > 0 else 0]
    input_data['product_utilization_rate_by_tenure'] = [total_products_purchased / tenure if tenure > 0 else 0]


    for cluster_col in cluster_columns:
        input_data[cluster_col] = [1 if f"cluster_{cluster}" == cluster_col else 0]

    input_data = input_data[logr_model.feature_names_in_]
    
    # Scale and predict
    data_scaled = sc.transform(input_data)
    churn_prob = logr_model.predict_proba(data_scaled)[:, 1][0]
   
    # Display results
    st.markdown("---")
    st.subheader("Churn Prediction Results")
    col1, padding, padding2, col2 = st.columns(4)

    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")

    with padding:
        st.empty()
        
    with padding2:
        st.empty()  

    with col2:
        st.metric("Retention Probability", f"{(1 - churn_prob):.1%}")
    churn_plot = plot_churn_likelihood(churn_prob)
    st.pyplot(churn_plot, transparent=True)
    
    cluster = birch_customers.predict(new_customer_ipca)[0]
    churn_prob = round(churn_prob * 100, 1)
    if churn_prob >= 50:
        st.markdown(f"""
        <div style='font-size:24px; color:#FF6B6B; padding:10px; border-radius:5px; border-left:5px solid #FFA000'>
        ⚠️ High Churn Risk ({churn_prob}%) - Cluster {cluster}: {cluster_name}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(CLUSTER_ADVICE.get(cluster, ""), unsafe_allow_html=True)
        
    elif churn_prob >= 30:
        st.markdown(f"""
        <div style='font-size:22px; padding:10px; border-radius:5px; border-left:5px solid #2196F3'>
        ⚖️ Moderate Churn Risk ({churn_prob}%) - Monitor Cluster {cluster}
        </div>
        <div style='font-size:20px; margin-top:10px'>
        Consider light engagement: {cluster_name} may respond to periodic check-ins
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div style='font-size:22px; padding:10px; border-radius:5px; border-left:5px solid #4CAF50'>
        ✅ Low Churn Risk ({churn_prob}%) - Cluster {cluster} is stable
        </div>
        <div style='font-size:20px; margin-top:10px'>
        Maintain standard service levels for {cluster_name}
        </div>
        """, unsafe_allow_html=True)

elif 'new_customer' not in locals() and 'new_customer' not in globals():
    st.info("Please fill out the form and click 'Predict' to see results.")
