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
heatmap_campaigns_kpis = read_image("A4_heatmap_campaigns_kpis.png")
correlation_matrix_kpis = read_image("A4_correlation_matrix_kpis.png")

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
    st.write("""
### Campaign Performance Evaluation  

To evaluate campaign performance, we identified and developed key performance indicators (KPIs) such as **Engagement Score, Conversion Rate, and Customer Lifetime Value (CLV)**. We also calculated the **Total Profit** for each campaign.  

#### Key Metrics:  

Before we move on to discuss about our insights regarding campaign success indicators, let's first examine the key metrics that enabled our analysis:

- **Conversion Rate**: Percentage of customers who converted after the campaign.  
- **Engagement Score**: Aggregated score based on user interaction with the campaign.  
  - Derived from **open rate, click rate, and average engagement time**, normalized to a value between 0 and 1.  
  - **Formula**:  
""")
    st.latex(r"Engagement Score = 0.3 \times Open\ Rate + 0.5 \times Click\ Rate + 0.2 \times Scaled\ Engagement\ Time")
    st.write("""
- **Customer Lifetime Value (CLV)**:  
  - For customers who converted, CLV was calculated as:  
""")
    st.latex(r"\text{CLV} = \text{Monthly Revenue} \times \frac{1}{\text{Churn Probability}}")
    st.write("""
  - The **average CLV** of the targeted customerswas then determined for each campaign.  
- **Total Profit**:  
  - Calculated as:  
""")
    st.latex(r"\text{Total Profit} = \text{Total Revenue} - \text{Total Campaign Cost}")

    st.subheader("Heatmap Analysis")
    st.write("""
A Campaign Engagement Heatmap was generated to allow us to visually analyse how users interacted with campaigns.
Color gradients are used to highlight areas performed in metrics such as `click_rate`, `open_rate` and `conversion_rate`.
Green is used for areas of high engagement, while Red indicates that the campaign is performing poorly in this metric.
""")
    st.image(heatmap_campaigns_kpis)
    st.write("""
### Insights:  
- **Delivery Issues**:  
  - The **"Mortgage Campaign for Retired"** and **"Savings Account Campaign for Retired"** campaigns had particularly low delivery rates.  
  - Campaigns targeting retired individuals require further investigation to understand delivery issues.  
  - As a result, these campaigns also had low click and open rates.  

- **Overall Campaign Performance**:  
  - Open, click, and conversion rates are generally low across all campaigns, even the well-delivered ones.  
  - This shows that there is **room for improvement** in refining messaging and targeting tactics to enhance campaign engagement and conversion rates.  
""")
    
    st.subheader("CLV and Profit Analysis")
    st.write("""
In this section, we analyse the two key performance metrics: Total Profit and average CLV (per targeted customer), across different customer segments and recommended products.
""")

    st.image(profit_clv_per_product)
    st.markdown("""
### Insights:  
- **Total Profit Analysis**:  
  - **Wealth Management for the Retired** attained the highest total profit.  
  - **Auto Loan for Young Professionals** amassed the least total profit.  

- **Average CLV Analysis**:  
  - **Investment Products for Young Professionals** had the highest CLV.  
  - The **Retired** customer segment consistently showed high CLV across multiple products.  

- **CLV vs. Profitability**:  
  - The **Retired** customer segment is a particularly strong target group, consistently performing well in Total Profit and average CLV across the different recommended products.
  - Targeting a segment with  high CLV does not necessarily result in high profitability. For example, the **Auto Loans for Young Professionals** campaign targeted customers with high CLV but still resulted in low Total Profits.  
    """)

    st.subheader("Correlation Matrix Analysis")
    st.write("""
In this section, a Correlation Matrix is utilised to allow us to better understand the strength and direction of relationships between key metrics in our dataset. Values range from -1 to 1:
- +1: indicates a perfect positive correlation between two variables
- -1: indicates a perfect negative correlation
- 0: the two variables are (likely) unrelated. No apparent relationship.
""")
    st.image(correlation_matrix_kpis)
    st.write("""
### Insights:  
- **Engagement Score & Conversion/Profitability**:  
  - Engagement score is **negatively correlated** with conversion rate and has **almost zero correlation** with total profit.  
  - This suggests that the current engagement score definition may not reflect tangible business outcomes.  
  - Resources may be misallocated toward **vanity metrics** (e.g., opens, clicks) that do not drive conversions or profits.  

- **Conversion Rate & Profitability**:  
  - Conversion rate is **positively correlated** with CLV and total profit.  
  - Customers who convert tend to have **higher value** and contribute more to overall profit.  
    """)

    st.subheader("Overall Recommendation")
    st.write("""
#### Key Strategic Recommendations:

1. **Refocus Marketing and Resource Allocation**  
   - Shift focus away from **superficial engagement metrics** (e.g., opens, clicks) and instead prioritize **conversion-driven strategies**.  
   - Target **high CLV customer segments** such as the **retired** and **middle market** to maximize profitability.  
   - Allocate more resources toward **higher-profit campaigns**, such as **wealth management for the retired**.  

2. **Improve Campaign Effectiveness**  
   - **Investigate delivery rate issues** for campaigns targeting the retired segment to ensure better reach and engagement.  
   - **Enhance marketing messaging and targeting strategies** to improve engagement and conversion rates.  
   - **Redefine engagement metrics** to align with actual business outcomes rather than vanity metrics.  

3. **Optimize Product Strategy**  
   - **Prioritize marketing efforts** toward the **Retired** segment, as they show both high CLV and profitability.  
   - **Investigate the Auto Loans for Young Professionals Campaign**, which scored high in CLV metrics but performed poorly in profitability, to improve financial returns.  

4. **Drive Higher Conversions and Profitability**  
   - **Shift resources** from vanity engagement metrics to tactics that are proven to **increase conversions and profits**.  
   - **Balance profitability and long-term customer value** when designing product strategies and marketing campaigns.  
    """)
