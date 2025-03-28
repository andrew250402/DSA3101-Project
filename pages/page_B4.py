import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import numpy as np
import pandas as pd

st.title("Personalisation with Cost-Effectiveness in Marketing Campaigns")



st.title("Campaigns Dataset")
try:
    df = read_csv('campaigns.csv')
    st.dataframe(df)
except FileNotFoundError as e:
    st.error(str(e))

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

st.title("Display png image")
try:
    image = read_image('apple.png')
    st.write("Image from apple.png:")
    st.image(image)
except FileNotFoundError as e:
    st.error(str(e))



# read the saved model (.pkl file)
model = read_model('logistic_regression_model.pkl')

st.title("Dummmy Logistic Regression Model Predictor")
st.write("""
Adjust the sliders to set feature values and see the model's prediction in real-time.
The model was trained on randomly generated data with 5 features.
""")

# Create sliders for each feature in the sidebar
st.sidebar.header("Feature Controls")
sliders = []
for i in range(5):
    # Using normal distribution range (-3 to 3) since data was randomly generated
    slider = st.sidebar.slider(
        label=f"Feature {i + 1}",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        key=f"feature_{i}"
    )
    sliders.append(slider)

# Convert slider values to numpy array for prediction
input_data = np.array(sliders).reshape(1, -1)

# Make prediction
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction Results")

# Show prediction with color (red for class 0, green for class 1)
if prediction == 1:
    st.success(f"Predicted Class: {prediction} (Positive)")
else:
    st.error(f"Predicted Class: {prediction} (Negative)")

# Show probability bar chart
st.write("Class Probabilities:")
st.bar_chart({
    "Class 0": probabilities[0],
    "Class 1": probabilities[1]
})

# Show raw probabilities
st.write(f"Probability of Class 0: {probabilities[0]:.4f}")
st.write(f"Probability of Class 1: {probabilities[1]:.4f}")

# Show the input values
st.subheader("Current Input Values")
for i, value in enumerate(sliders):
    st.write(f"Feature {i + 1}: {value:.2f}")
