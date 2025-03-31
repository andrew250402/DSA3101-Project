import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import pandas as pd

# Load data
customer_engagement_df = read_csv('customer_engagement.csv')
customer_df = read_csv('customers.csv')

# load in plots
product_usage = read_image("A3_mean_product_usage.png")
spending_behaviour = read_image("A3_spending_behaviour.png")
mobile_web_logins = read_image("A3_mobile_web_logins.png")
profit_clv_per_product = read_image("A4_profit_clv_per_product.jpg")

# Set page config
st.set_page_config(page_title="Marketing Campaign Analysis", layout="wide")

# Create sidebar for page navigation
st.sidebar.title("Navigation")

# Define pages
pages = ["A3: Behavioural Patterns", "A4: Campaign Impact"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)


# Campaign Analysis Page
if selected_page == "A3: Behavioural Patterns":
    st.title("How do customer behaviors vary across different segments?")

    st.subheader("Mean Product Usage across Customer Segments")
    st.image(product_usage)
    st.markdown("""
The graph above highlights the **mean product usage** across customer segments, revealing
which popularity of products  within their respective segments. These enable
targeted advertising, as we are able to efficiently allocate resources to maximise returns
from every segment.

### Insights and Recommendations:
- **Cluster 0 (Conservative Savers)**: It appears that **investment products are underutilised**
in this segment. Marketing efforts for investment products can be scaled up to improve product
utlisation in this segment.
- **Cluster 1 (High-value Customers)**: Product usage is relatiely **high across all 
categories** as compared to the other clusters, except for Personal Loans. Promoting personal
Loan services with lower interest rates to customers in this segment can boost adoption.
- **Cluster 2 (Everyday Spenders)**: a **low usage of investment products** is evident. Promoting
investment products can drive up usage.

### Overall Strategy:
- Focus on personalized, targeted promotions for each cluster to increase product adoption
and maximize marketing ROI. Nevertheless, it is important to note that some products are
underutilised in segments as the customer segment simply **does not want** the product.
Extensive advertising may not improve returns. Hence, an alternative marketing approach
that can be considered is to promote the products which are of high usages to further
capitalise on the already proven product preferences.
    """)

    st.subheader("Peak Month Spending and Frequency across Customer Segments")
    st.image(spending_behaviour)
    st.markdown("""
The graph above depicts the **peak monthly spending** and **peak monthly transaction count**
across the different clusters. This visualisation gives us insights into the spending
habits of the different clusters to better cater promotions and incentives towards them.

### Insights and Recommendations:
- **Cluster 0**: Cluster 0 has the **lowest** peak monthly spending and transaction frequency.
Offering re-engagement campaigns such as cashbacks and discount can incentivise higher
spending frequency and in greater amounts.
- **Cluster 1**: Cluster 1 has the **highest** peak spending habits across the three segments.
Providing exclusive perks such as priority customer service can encourage their continued
engagement with the bank. Spending habits can be further incentivised through promotions such
as travel miles.
-**Cluster 2**: Has a moderate peak month spending frequency but **middling spend per transaction**.
Higher spending per transaction can be incentivised by offering discounts after achieving
certain spending thresholds.

### Overall Strategy:
Tailor promotions and incentives based on the spending habits of the segment to maximize engagement
and increase overall spending.
    """)

    st.subheader("Mobile and Web Logins across Customer Segments")
    st.image(mobile_web_logins)
    st.markdown("""
The visualisation above depicts the **average number of logins** per week for the
Mobile and Web banking platforms, across three customer segments. 

### Insights and Recommendations:
We observe a consistent trend across all the displayed clusters that the **mobile app
receives more interaction** than the web platform. More resources can be invested into the
mobile application to improve customer experience, through which personalised marketing
recommendations can also be pushed out.

### Overall Strategy:
Focus on optimizing the mobile experience to increase engagement and deliver targeted, personalized offers.
    """)

elif selected_page == "A4: Campaign Impact":
    st.title("What are the key performance indicators (KPIs) for assessing the success of marketing campaigns?")
    st.markdown("""
### Overview
To evaluate campaign performance, we developed metrics such as engagement score, conversion
rate, total profit and customer lifetime value (CLV) from each campaign. We then used
clustering and correlation matrix to identify patterns.
""")

    st.image(profit_clv_per_product)
    st.markdown("""

    """)
