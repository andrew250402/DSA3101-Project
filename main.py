import streamlit as st


st.set_page_config(
    page_title="DSA3101 project",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DSA3101 project")

# Create sidebar for page navigation
st.sidebar.title("Navigation")

# Define pages
pages = ["Introduction/Context", "Analysis/Insights", "Proposed Solution/ Deployment Strategy", "Conclusion"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "Introduction/Context":
    st.title("Introduction/Context")
    

elif selected_page == "Analysis/Insights":
    st.title("Analysis/Insights")
    st.subheader("Smart Customer Segmentation for Precision Targeting")
    st.markdown("""
**Method:**
- Applied **Hierarchical Clustering** to identify customer segments based on:
  - Transaction history  
  - Digital engagement  
  - Product usage  
- Chosen over traditional segmentation due to its adaptability to new customer data and reduced retraining needs.

**Insights:**
- Identified **three distinct customer segments**:
  - **Conservative Savers** – Older, risk-averse, focused on retirement and low-risk savings.  
  - **High-Value Customers** – High transaction volume, digitally engaged, responsive to premium services.  
  - **Everyday Spenders** – Frequent card users who value mobile banking and cashback rewards.

**Action:**
- Deploy **segment-specific campaigns** to boost engagement:
  - Retirement-focused offers for **Conservative Savers**  
  - Exclusive investment opportunities for **High-Value Customers**  
  - Cashback and mobile-first promotions for **Everyday Spenders**
                """)

    st.subheader("Optimized Campaigns for Maximum ROI")
    st.markdown("""
**Method:**
- Analyzed marketing channel performance across time and campaign types.
- Used **Gradient Boosting** to develop an ROI prediction model based on:
  - Historical conversion rates  
  - Campaign costs  
  - Engagement metrics  

**Insights:**
- **SMS campaigns** perform best on **Mondays**.  
- **Mobile app notifications** see the highest engagement on **Thursdays**.  
- **Mortgage and savings promotions** receive the strongest customer response.  
- High engagement doesn't always equal high profitability — e.g., some well-clicked campaigns delivered low ROI, while **Wealth Management for retirees** proved highly profitable.

**Action:**
- Reallocate marketing budgets based on predicted ROI rather than engagement alone.  
- Reduce spending on **underperforming high-engagement** campaigns.  
- Prioritize **high-ROI strategies**, such as targeting retirees for wealth management.  
- Optimize **channel and timing** to ensure each campaign drives measurable returns.
                """)

    st.subheader("Predictive Analytics for Proactive Engagement")
    st.markdown("""
**Method**
- Developed a **predictive recommendation engine** using the **Apriori algorithm** to uncover hidden product affinities.  
- Combined with a **Random Forest model** to predict customer likelihood of product adoption.  
- Built a **Logistic Regression churn prediction model** using customer segmentation insights to identify at-risk behaviors.

**Insights**
- Customers with **credit cards** are more likely to adopt **personal loans** (product affinity).  
- High-Value Customers show churn signals via **declining transaction frequency**.  
- Everyday Spenders may churn if there's a **drop in mobile app logins**.  
- Segment-specific behavior patterns are critical in early churn detection.

**Action**
- Use predictive models to **proactively recommend relevant products** to customers.  
- Deploy **targeted retention incentives** based on early churn indicators.  
- Prevent customer loss by **addressing at-risk behavior before churn happens**, improving long-term loyalty and product adoption.
                """)

    st.subheader("Balancing Personalization & Cost Efficiency")
    st.markdown("""
**Method**
- Created an artificial **personalization score** for each campaign to assess the level of targeting.  
- Used **ROI** as the primary metric to evaluate **cost-effectiveness** of personalization.  
- Analyzed the relationship between personalization score and ROI across different customer segments.

**Insights**
- **Young Professionals** show the **strongest correlation** between personalization and ROI.  
  - Highly personalized campaigns (e.g., wealth management or mortgages via email/mobile notifications) are most effective.  
- **Middle-Market** customers have the **weakest correlation**.  
  - General marketing strategies (e.g., SMS or direct mail promoting savings accounts or auto loans) are more cost-effective for this group.  
- Over-personalization can lead to inflated costs with minimal ROI benefit in low-response segments.

**Action**
- **Prioritize personalization** for high-response segments like **Young Professionals**.  
- **Reduce targeting costs** by using **general campaigns** for segments like the **Middle-Market**.  
- Use data-driven personalization strategies to **maximize ROI** while keeping marketing spend efficient.
                """)


elif selected_page == "Proposed Solution/ Deployment Strategy":
    st.title("Proposed Solution/ Deployment Strategy")

    st.subheader("Segment-Specific A/B Testing")
    st.markdown("""
    Conduct controlled experiments across all customer segments to determine the optimal:

- **Channel preferences**  
  (e.g., SMS vs. Email vs. Push Notifications)

- **Timing strategies**  
  (e.g., Day of Week / Time of Day)

- **Creative approaches**  
  (e.g., Personalized Messaging vs. Product-Focused Messaging)
                """)
    st.subheader("Real-Time Model Updates")
    st.markdown("""
    Establish automated BIRCH and IPCA pipeline models using:
    
- Latest transaction data  
                
- Updated digital engagement patterns  
                
- Seasonal behavioral trends          
                """)
    st.subheader("Privacy")
    st.markdown("""
    Implement differential privacy techniques to:

- Anonymize customer data used for clustering

- Add statistical noise to protect individual identities

- Maintain marketing effectiveness while complying with regulations
                """)

    st.subheader("Deployment Timeline")
    st.markdown("""
Phase 1 (Months 1–3): Pilot Implementation + Baseline A/B Tests
- Deploy core segmentation models  
- Establish testing framework for 2 smaller segments  
- Capture initial optimization benchmarks  

Phase 2 (Months 4–6): Scaling + Continuous Learning
- Expand to all segments  
- Implement automated model updates  
- Introduce privacy safeguards  
- Analyze A/B test results to refine strategies  

Phase 3 (Months 7–12): Mature Optimization
- Full automation of testing and model updates  
- Privacy-by-design across all components
                """)
else:
    st.title("Conclusion")

    st.subheader("Revenue Growth Through Precision Engagement")
    st.markdown("""
- Higher conversion rates from segment-specific personalization
- Increase in CLV via predictive next-best-product recommendations
                """)
    st.subheader("Cost Efficiency & Resource Optimization")
    st.markdown("""
- Reduction in wasted ad spend through AI-driven budget allocation
- Lower customer acquisition costs via improved targeting accuracy
- Relieves the need for manual segmentation through automated clustering
                """)
    st.subheader("Future-Proof Competitive Advantage")
    st.markdown("""
- Real-time adaptation to shifting customer behaviors and market trends
- Continuous performance improvement through self-learning A/B testing frameworks
- Regulatory resilience with built-in privacy protections and compliance safeguards
                """)