import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import pandas as pd

# Load data
customer_engagement_df = read_csv('customer_engagement.csv')
customer_df = read_csv('customers.csv')

# Set page config
st.set_page_config(page_title="Marketing Campaign Analysis", layout="wide")

# Create sidebar for page navigation
st.sidebar.title("Navigation")

# Define pages
pages = ["A3", "A4"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)


# Campaign Analysis Page
if selected_page == "A3":
    st.title("How do customer behaviors vary across different segments?")

    st.subheader("Mean Product Usage across Customer Segments")
    st.write("[Insert Chart Here]")
    st.write("""
             ### Insights:
- **Cluster 0**: Low usage of investment products and credit cards.
- **Cluster 1**: High usage of most products except personal loans.
- **Cluster 2**: Low usage of investment products.

### Recommendations:
- **Cluster 0**: Increase promotions for investment products and credit cards.
- **Cluster 1**: Offer personal loans with lower interest rates to boost adoption.
- **Cluster 2**: Promote investment products to drive higher usage.

### Overall Strategy:
- Focus on personalized, targeted promotions for each cluster to increase product adoption and maximize marketing ROI.
    """)

    st.subheader("Peak Month Spending and Frequency across Customer Segments")
    st.write("[Insert Chart Here]")
    st.write("""
### Insights:
- **Cluster 0**: Lowest peak month spending and frequency.
- **Cluster 1**: Highest peak month spending and frequency.
- **Cluster 2**: Moderate peak month spending frequency and amounts.

### Recommendations:
- **Cluster 0**: Offer re-engagement campaigns like cashback and discounts to increase spending frequency.
- **Cluster 1**: Provide exclusive perks (e.g., priority customer service) and incentives (e.g., travel miles) to maintain high spending and encourage loyalty.
- **Cluster 2**: Encourage higher spending by offering discounts or incentives for reaching specific spending thresholds.

### Overall Strategy:
- Tailor promotions and incentives based on cluster spending habits to maximize engagement and increase overall spending.

    """)

    st.subheader("Mobile and Web Logins across Customer Segments")
    st.write("Insert Chart Here")
    st.write("""
### Insights:
- **All Clusters**: Higher average logins per week on mobile compared to web.

### Recommendations:
- Invest more resources into improving the mobile app to enhance customer experience.
- Push personalized recommendations based on customer usage patterns within the app.

### Overall Strategy:
- Focus on optimizing the mobile experience to increase engagement and deliver targeted, personalized offers.

    """)

elif selected_page == "A4":
    st.title("What are the key performance indicators (KPIs) for assessing the success of marketing campaigns?")
