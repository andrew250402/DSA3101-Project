import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import numpy as np
import pandas as pd

st.title("Personalisation with Cost-Effectiveness in Marketing Campaigns")



st.title("Campaigns Dataset")
try:
    campaigns = read_csv('campaigns.csv')
    st.dataframe(campaigns)
except FileNotFoundError as e:
    st.error(str(e))
#Get ROI
campaigns["ROI"] = (campaigns["total_revenue_generated"] - campaigns["total_campaign_cost"]) / campaigns["total_campaign_cost"]
# Define multipliers for campaign type and customer segment
campaigns_type_mult = {
    "Mobile App Notifications": 1.3,
    "Email": 1.2,
    "SMS": 1.1,
    "Direct Mail": 1.0
}

campaigns_seg_mult = {
    "High-Value": 1.4,
    "Young Professionals": 1.3,
    "Middle-Market": 1.2,
    "Retired": 1.1,
    "Budget-Conscious": 1.0
}

product_mult = {
    "Personal Loan": 1.3,
    "Wealth Management": 1.5,
    "Credit Card": 1.2,
    "Savings Account": 1.0,
    "Auto Loan": 1.1,
    "Mortgage": 1.4,
    "Investment Product": 1.3
}

# Apply the multipliers using .map()
campaigns["type_mult"] = campaigns["campaign_type"].map(campaigns_type_mult)
campaigns["seg_mult"] = campaigns["customer_segment"].map(campaigns_seg_mult)
campaigns["product_pers_score"] = campaigns["recommended_product_name"].map(product_mult)

# Compute personalization score
campaigns["pers_score"] = campaigns["type_mult"] * campaigns["seg_mult"] * campaigns["product_pers_score"]
# Display the updated DataFrame
print(campaigns[["campaign_id", "campaign_type", "customer_segment", "pers_score"]].head())


st.title("Personalisation Score Multipliers and Rationales")

st.write("### Campaign Types")
campaign_type = {
    "Campaign Type": ["Mobile App Notifications", "Email", "SMS", "Direct Mail"],
    "Personalisation Multiplier": [1.3, 1.2, 1.1, 1.0],
    "Reasoning": ["Highly dynamic; can be triggered based on real-time behaviors (e.g., abandoned carts, app activity). Allows deep customisation with user data.",
                  "Can be personalised with names, purchase history, and dynamic content. Supports segmentation and A/B testing.",
                  "Allows for some personalisation (name, limited targeting), but is mostly short and lacks interactive elements.",
                  "Least personalised. Typically static, batch-sent to predefined segments. High production cost limits granularity."]
}

campaign_type_table = pd.DataFrame(campaign_type)
campaign_type_table = campaign_type_table.style.set_table_styles([
    {'selector': 'td', 'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]}
])
st.dataframe(campaign_type_table)

st.write("### Customer Segment")
customer_segment = {
    "Campaign Type": ["High-Value", "Young Professionals", "Middle-Market", "Retired", "Budget-Conscious"],
    "Personalisation Multiplier": [1.4, 1.3, 1.2, 1.1, 1.0],
    "Reasoning": ["Likely has detailed transaction history & preferences, making personalised campaigns highly effective. High ROI justifies more personalised marketing.",
                  "More digitally engaged, responsive to personalised app notifications, emails, and targeted ads. Campaigns can be optimized for their behaviours.",
                  "Moderate spending & engagement levels, some personalisation possible but less impactful than high-value or young professionals.",
                  "Likely to have stable, predictable behavior, but digital engagement may be lower. Personalisation is useful but limited.",
                  "Price-sensitive; campaigns are more deal-focused than personalised. Broad targeting is more effective than individual customisation."]
}

customer_segment_table = pd.DataFrame(customer_segment)
customer_segment_table = customer_segment_table.style.set_table_styles([
    {'selector': 'td', 'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]}
])
st.dataframe(customer_segment_table)

st.write("### Recommended Product")

recommended_product = {
    "Campaign Type": ["Wealth Management", "Mortgage", "Personal Loan", "Investment Product", "Credit Card", "Auto Loan", "Savings Account"],
    "Personalisation Multiplier": [1.5, 1.4, 1.3, 1.3, 1.2, 1.1, 1.0],
    "Reasoning": ["Highly personalised; involves tailored investment strategies, complex financial products.",
                  "Highly personalised based on home type, loan term, location, and financial situation.",
                  "High demand, often customised to the individual’s credit score, income, and loan amount.",
                  "Investment options are tailored to risk profiles and financial goals, but generally standardised.",
                  "Personalised offers based on spending habits, credit score, and rewards preferences.",
                  "Moderately personalised based on credit and vehicle preferences, but not as much as personal loans.",
                  "Somewhat personalised (interest rates, account features), but largely standardised."]
}

recommended_product_table = pd.DataFrame(recommended_product)
recommended_product_table = recommended_product_table.style.set_table_styles([
    {'selector': 'td', 'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]}
])
st.dataframe(recommended_product_table)

# Create correlation tables for different categories
correlation_campaign_type = campaigns.groupby('campaign_type').apply(lambda group: group['ROI'].corr(group['pers_score']), include_groups=False).reset_index()
correlation_campaign_type.columns = ['Campaign Type', 'Correlation (ROI vs. Pers. Score)']

correlation_customer_segment = campaigns.groupby('customer_segment').apply(lambda group: group['ROI'].corr(group['pers_score']), include_groups=False).reset_index()
correlation_customer_segment.columns = ['Customer Segment', 'Correlation (ROI vs. Pers. Score)']

correlation_recommended_product = campaigns.groupby('recommended_product_name').apply(lambda group: group['ROI'].corr(group['pers_score']), include_groups=False).reset_index()
correlation_recommended_product.columns = ['Recommended Product', 'Correlation (ROI vs. Pers. Score)']

# Function to return the action and strategy based on correlation
def get_action_strategy(correlation_value):
    if 0.7 <= correlation_value <= 1:
        return "High Personalization Focus", "Double down on personalization efforts (e.g., high personalization emails, notifications)"
    elif 0.3 <= correlation_value < 0.7:
        return "Moderate Personalization", "Continue with personalized efforts, test different strategies"
    elif 0 <= correlation_value < 0.3:
        return "Minimal Personalization", "Focus on light personalization or segment-based targeting"
    elif 0 <= correlation_value < -0.3:
        return "Cost-Effective or Generic", "Use broad targeting with minimal personalization"
    elif -1 <= correlation_value < 0:
        return "Avoid Excessive Personalization", "Use mass-market campaigns, avoid deep personalization"
    else:
        return None, None  # In case of invalid correlation values

# Streamlit UI
st.title("ROI vs. Personalization Score Analysis")

# Radio button for plot selection
plot_choice = st.radio(
    "Select a plot:",
    ["Campaign Type", "Customer Segment", "Recommended Product"]
)

# Select category (simulate hover behavior)
if plot_choice == "Campaign Type":
    category = st.selectbox("Select Campaign Type:", ["All"] + list(campaigns['campaign_type'].unique()))
    if category == "All":
        filtered_data = campaigns
    else:
        filtered_data = campaigns[campaigns['campaign_type'] == category]
    
    # Display Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x="pers_score", y="ROI", hue="campaign_type", alpha=0.7, ax=ax)
    ax.set_title(f"ROI vs. Personalization Score (Campaign Type: {category})")
    ax.legend(title="Campaign Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    # Display Correlation Table for selected category
    if category == "All":
        st.write(correlation_campaign_type)
    else:
        st.write(correlation_campaign_type[correlation_campaign_type['Campaign Type'] == category])

    # Display Action and Strategy for selected category (if not "All")
    if category != "All":
        correlation_value = correlation_campaign_type[correlation_campaign_type['Campaign Type'] == category]['Correlation (ROI vs. Pers. Score)'].values[0]
        action, strategy = get_action_strategy(correlation_value)
        st.write(f"**Correlation Coefficient:** {correlation_value:.2f}")
        st.write(f"**Action to Take:** {action}")
        st.write(f"**Strategy:** {strategy}")

elif plot_choice == "Customer Segment":
    category = st.selectbox("Select Customer Segment:", ["All"] + list(campaigns['customer_segment'].unique()))
    if category == "All":
        filtered_data = campaigns
    else:
        filtered_data = campaigns[campaigns['customer_segment'] == category]
    
    # Display Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x="pers_score", y="ROI", hue="customer_segment", alpha=0.7, ax=ax)
    ax.set_title(f"ROI vs. Personalization Score (Customer Segment: {category})")
    ax.legend(title="Customer Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    # Display Correlation Table for selected category
    if category == "All":
        st.write(correlation_customer_segment)
    else:
        st.write(correlation_customer_segment[correlation_customer_segment['Customer Segment'] == category])

    # Display Action and Strategy for selected category (if not "All")
    if category != "All":
        correlation_value = correlation_customer_segment[correlation_customer_segment['Customer Segment'] == category]['Correlation (ROI vs. Pers. Score)'].values[0]
        action, strategy = get_action_strategy(correlation_value)
        st.write(f"**Correlation Coefficient:** {correlation_value:.2f}")
        st.write(f"**Action to Take:** {action}")
        st.write(f"**Strategy:** {strategy}")

elif plot_choice == "Recommended Product":
    category = st.selectbox("Select Recommended Product:", ["All"] + list(campaigns['recommended_product_name'].unique()))
    if category == "All":
        filtered_data = campaigns
    else:
        filtered_data = campaigns[campaigns['recommended_product_name'] == category]
    
    # Display Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x="pers_score", y="ROI", hue="recommended_product_name", alpha=0.7, ax=ax)
    ax.set_title(f"ROI vs. Personalization Score (Recommended Product: {category})")
    ax.legend(title="Recommended Product", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    # Display Correlation Table for selected category
    if category == "All":
        st.write(correlation_recommended_product)
    else:
        st.write(correlation_recommended_product[correlation_recommended_product['Recommended Product'] == category])

    # Display Action and Strategy for selected category (if not "All")
    if category != "All":
        correlation_value = correlation_recommended_product[correlation_recommended_product['Recommended Product'] == category]['Correlation (ROI vs. Pers. Score)'].values[0]
        action, strategy = get_action_strategy(correlation_value)
        st.write(f"**Correlation Coefficient:** {correlation_value:.2f}")
        st.write(f"**Action to Take:** {action}")
        st.write(f"**Strategy:** {strategy}")