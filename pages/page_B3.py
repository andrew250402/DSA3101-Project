import streamlit as st
import os
from streamlit_utilities import read_csv, read_image, read_model
import numpy as np
import pandas as pd

# load saved model
model = read_model('b3_measuring_campaign_roi.pkl')

def user_input_features():
    # Numerical inputs
    
                                    
    # Categorical inputs
    return


st.title("Measuring and maximising return on investment ROI for personalised marketing efforts")
st.header("📊 Model Overview:")
st.markdown("""
Our model, a **Gradient Boosting Classifier (GBC)** was trained on feature-engineered
campaign data, utilising 11 important features that were selected via feature selection
to further improve predictive accuracy.
""")
st.subheader("Why GBC?")
st.markdown("""
Gradient Boosting Classifiers are known to perform well even with small datasets, just like
the campaign dataset that was used for model training to predict ROI
""")
st.subheader("ROI Category Classification:")
st.markdown("""Our ROI classification categorises campaign performance into three tiers
based on the following benchmarks:
""")

roi_class_table ={
    'Tier': ['Low 🔴','Medium 🟡','High 🟢'],
    'Raw ROI Value': ['0.66 and below', '0.67-1.32', '1.33 and above']
}
st.dataframe(roi_class_table)
st.markdown("""
- **Low ROI 🔴(<= 0.66)**: Campaigns fail to generate significant returns, potentially caused by factors such as poor targeting or weak conversion rates.
- **Medium ROI 🟡(0.67 - 1.32)**: Campaigns are profitable but can be further optimised.
- **High ROI 🟢(>= 1.33)**: Campaign is performing extremely well, due to a combination of effective targeting, high conversion rates etc.

Apart from improving model performance, converting raw ROI values into distinct tiers
can also enable strategic decision making by translating complex metrics into more
comprehensible and actionable business insights.
""")

st.subheader("Campaign ROI Prediction")
st.markdown("""
Adjust sliders and dropdown menus below to create a personalised marketing campaign 
targeting a customer segment of your choice. Afterwards, click "Predict" to see the
ROI prediction of your campaign in real-time.
""")

# Taking in User Input
st.header("Campaign Parameters:")
customer_segment = st.selectbox(
    "Customer Segment",
    ["Middle Market", "Retired", "Budget-Conscious","High-Value","Young Professionals"]
)
total_campaign_cost = st.slider("Total Campaign Cost ($)", 10000, 110000)
campaign_duration = st.slider("Duration of Campaign (Days)", 30, 60)
conversion_rate = st.slider("Conversion Rate (%)", 5, 15)
acquisition_cost = st.slider("Acquisition Cost ($)", 150, 7000)
product_type = st.selectbox(
    "Product Type",
    ["Credit Card", "Investment Product", "Mortgage", "Personal Loan", "Savings Account", "Wealth Management"]
)
campaign_type = st.selectbox(
    "Campaign Channel",
    ["Email", "Mobile App Notifications", "SMS"]
)

# creating an input_dictionary
input_dict = {
    'total_campaign_cost': total_campaign_cost,
    'campaign_duration': campaign_duration,
    'conversion_rate': conversion_rate/100,
    'acquisition_cost': acquisition_cost,
    'customer_segment_Retired': 1 if customer_segment == "Retired" else 0,
    'customer_segment_Young Professionals': 1 if customer_segment == "Young Professionals" else 0,
    'product_Credit Card': 1 if product_type == "Credit Card" else 0,
    'product_Personal Loan': 1 if product_type == "Personal Loan" else 0,
    'campaign_type_Email': 1 if campaign_type == "Email" else 0,
    'campaign_type_Mobile App Notifications': 1 if campaign_type == "Mobile App Notifications" else 0,
    'campaign_type_SMS': 1 if campaign_type == "SMS" else 0
}

model_features = [
    'total_campaign_cost',
    'campaign_duration',
    'conversion_rate', 
    'acquisition_cost',
    'customer_segment_Retired',
    'customer_segment_Young Professionals',
    'product_Credit Card',
    'product_Personal Loan',
    'campaign_type_Email',
    'campaign_type_Mobile App Notifications',
    'campaign_type_SMS'
]

# make prediction
input_df = pd.DataFrame([input_dict])[model_features]

# adding predict button
if st.button("Predict!"):
    prediction = model.predict(input_df)
    roi_class = prediction[0]
    color = {
        "Low": "red",
        "Medium": "orange",
        "High": "green"
    }[roi_class]
    st.markdown(
        f"### Predicted ROI Category: <span style='color:{color}; font-size:28px'>{roi_class}</span>",
        unsafe_allow_html=True
    )
    st.markdown("#### Follow-up Actions:")
    if roi_class == "High":
        st.markdown("""
        - Scale such campaigns to similar audiences
        - Allocate additional budget to such campaigns
        - Monitor campaign for any slow down in engagement
        """)
    if roi_class == "Low" or roi_class == "Medium":
        st.markdown("""
        - Reallocate budget from underperforming segments
        - Consider alternative marketing channels
        - Re-evaluate selection of target audience
        """)
        
        

st.divider()

# Model Evaluation

# Display results
st.subheader("Prediction Results")

# Show the input values
st.write(model.feature_names_in_)
