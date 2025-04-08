import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_utilities import read_csv, read_image, read_model


st.set_page_config(
    page_title="Optibank",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.image(read_image("optibank_logo.png"))


# Create sidebar for page navigation
st.sidebar.title("Navigation")

# Define pages
pages = ["Introduction/Context", "Analysis/Insights", "Proposed Solution/ Deployment Strategy", "Conclusion"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "Introduction/Context":
    st.markdown("""
    <style>
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li, 
    div[data-testid="stMarkdownContainer"] ul {
        font-size: 18pt !important;
        line-height: 1.5 !important;
        margin-bottom: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Introduction/Context")
    st.subheader("🚀 AI-Powered Marketing vs. Traditional Banking Marketing")

    # Split into two columns
    col1, col2 = st.columns(2)

    # --- Left Side (Traditional Marketing) ---
    with col1:
        st.markdown("### ❌ Traditional Banking Marketing")

        # Pain points in bullet points
        st.markdown("""
        **Challenges:**  
        - **Static campaigns** (one-size-fits-all)  
        - **Poor segment targeting** (wasted budget)  
        - **Low conversion rates** (irrelevant offers)  
        - **Missed cross-sell opportunities**  
        - **Unmitigated customer churn**  
        """)

    # --- Right Side (AI-Powered Marketing) ---
    with col2:
        st.markdown("### ✅ AI-Optimized Marketing")
        
        # Benefits in bullet points
        st.markdown("""
        **Solutions:**   
        - **Predictive analytics** (anticipate needs)
        - **Dynamic customer segmentation**  
        - **Personalized campaign offers**  
        - **Higher ROI on marketing spend**  
        - **Improved customer loyalty**  
        """)
        
    # Split into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Additional Visual: Side-by-Side Bar Chart ---
        st.markdown("---")
        st.subheader("📊 Performance Comparison: AI vs. Traditional")
        fig, ax = plt.subplots()
        metrics = ["Conversion Rate", "Cost Per Acquisition", "Customer Retention"]
        traditional_values = [5, 50, 60]
        ai_values = [15, 20, 85]

        ax.bar([x - 0.2 for x in range(3)], traditional_values, width=0.4, label="Traditional", color="red")
        ax.bar([x + 0.2 for x in range(3)], ai_values, width=0.4, label="AI-Optimized", color="green")
        ax.set_xticks(range(3))
        ax.set_xticklabels(metrics)
        ax.legend()

        st.pyplot(fig)
    with col2:
        st.empty()

    # Footer
    st.markdown("---")
    

elif selected_page == "Analysis/Insights":

    st.markdown("""
        <style>
        div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stMarkdownContainer"] li, 
        div[data-testid="stMarkdownContainer"] ul {
            font-size: 18pt !important;
            line-height: 1.5 !important;
            margin-bottom: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Analysis/Insights")
    st.header("📊 Smart Customer Segmentation Analysis")
    st.markdown("---")

    # Main Content Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Key Insights & Methodology")
        st.markdown("""
        #### 🧪 **Clustering Approach**
        - Leveraged **Hierarchical Clustering** for dynamic customer grouping
        - Key dimensions:
            - Transaction patterns
            - Digital engagement metrics
            - Product/service utilization
            - Demographic indicators
        - *Advantage:* Hierarchical clustering naturally reveals data hierarchies through dendrograms

        #### 🎯 **Identified Customer Segments**
        """)
        
        # Segment Details with Emojis
        with st.expander("👴 Conservative Savers", expanded=False):
            st.markdown("""
            - **Profile:** Older demographic, risk-averse
            - **Needs:** Retirement planning, FDIC-insured products
            - **Engagement:** Digital banking engagement
            """)
        
        with st.expander("👔 High-Value Customers", expanded=False):
            st.markdown("""
            - **Profile:** High net-worth, tech-savvy professionals
            - **Needs:** Premium services, investment products
            - **Engagement:** Active on digital channels
            """)
        
        with st.expander("💳 Everyday Spenders", expanded=False):
            st.markdown("""
            - **Profile:** Frequent card users
            - **Needs:** Cashback rewards, instant financing
            - **Engagement:** Mobile-first banking users
            """)

    with col2:
        st.subheader("📈 Cluster Visualization")
        st.image(read_image("a5_dendrogram.png"), use_container_width=True)
        st.caption("Figure 1: Customer segmentation dendrogram showing 3 distinct clusters")


    st.divider()
    st.divider()

        # --- Optimized Campaigns Section ---
    st.header("🎯 Optimized Campaigns for Maximum ROI")
    st.markdown("---")
    with st.container():
        col1, col2 = st.columns([2, 1], gap="medium")
        
        with col1:
            st.subheader("🔍 Key Insights")
            
            with st.expander("⚛️ Engagement Paradox", expanded=False):
                st.markdown("""
                35% of high-engagement campaigns underperformed on ROI
                """)

            with st.expander("💸 High-Value Offers", expanded=False):
                st.markdown("""
                - 👴 Wealth Management: 3× ROI from retirees
                - 🏦 Mortgage promotions: highest conversion rates
                """)

            with st.expander("⏰ Peak Timing", expanded=False):
                st.markdown("""
                - 📱 SMS campaigns: **Monday** effectiveness (2× avg. ROI)
                - 📲 App notifications: **Thursday** engagement peaks
                - 📧**Email**: Highest average conversion rate
                """)
        with col2:
            st.markdown("""
            #### 🚀 Action Plan
            """)
            st.success("""
            **Budget Reallocation Strategy**
            - 🕒 Implement day-specific channel outreach            
            - ⬆️ Boost retiree-focused wealth management
            - ⬇️ Reduce underperforming engagement campaigns
            """)
    st.divider()
    st.divider()

    # --- Predictive Analytics Section ---
    st.header("🔮 Predictive Analytics for Proactive Engagement")
    st.markdown("---")
    with st.container():
        st.subheader("🔍 Key Insights & Methodology")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("**🔄 Apriori Product Affinities Engine**", expanded=True):
                st.markdown("""
                - Product affinity detection
                - Market basket analysis
                - Cross-sell recommendations
                """)
            
        with col2:
            st.markdown("""
            #### 🔍 Behavioral Insights
            - **Credit Card Holders**: 73% loan adoption likelihood
            - **Affinity Patterns**: Savings + Insurance bundle potential
            """)

    st.divider()
    st.divider()

    # --- Personalization Section ---
    st.header("⚖️ Balancing Personalization & Cost Efficiency")
    st.markdown("---")
    with st.container():
        st.subheader("🔍 Key Insights & Methodology")
        cols = st.columns([2,1])
        
        with cols[0]:
            st.markdown("""
            #### 🎯 Segment-Specific Strategies
            """)
            
            with st.expander("👔 High-Value Customers", expanded=False):
                st.markdown("""
                - **High-Value Personalization**:
                    - 🏠 Mortgage rate alerts via mobile
                    - 📈 Investment auto-recommendations
                - Optimal Channels: Email + Push
                - ROI Multiplier: 1.8-2.4×
                """)
            
            with st.expander("💳 Everyday Spenders", expanded=False):
                st.markdown("""
                - **Cost-Effective Approach**:
                    - 🚗 Auto loan SMS blasts
                    - 🏦 Savings account mailers
                - Optimal Channels: SMS + Direct Mail
                - Personalization Cap: 35% score
                """)

        with cols[1]:
            st.markdown("""
            #### 📊 Metric Framework
            - **Personalization score** for each campaign to assess the level of targeting.  
            - **ROI** as the primary metric to evaluate **cost-effectiveness** of personalization.  
            - Analyzed the relationship between personalization score and ROI across different customer segments.
            """)
        
        st.markdown("""
        <style>
        .stContainer {border-radius: 15px; padding: 25px; margin: 15px 0;}
        </style>
        """, unsafe_allow_html=True)


elif selected_page == "Proposed Solution/ Deployment Strategy":
    st.markdown("""
        <style>
        div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stMarkdownContainer"] li, 
        div[data-testid="stMarkdownContainer"] ul {
            font-size: 18pt !important;
            line-height: 1.5 !important;
            margin-bottom: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # --- Real-Time Personalization Engine ---
    st.title("Proposed Solution/Deployment Strategy")
    st.header("📦 Deployment and Implementation")
    st.markdown("---")
    st.markdown("""
            #### 🐳 Docker Deployment Business Consideration
            """)
    with st.container():
        
        cols = st.columns(3)
        
        with cols[0]:
            with st.expander("📈 Scalability", expanded=False):
                st.markdown("""
                            - Each model runs in its own container, allowing independent scaling based on real-time usage.
                            - High-demand services (e.g., real-time segmentation) can scale without affecting other components.
                            """)

        with cols[1]:
            with st.expander("💰 Cost-Efficiency", expanded=False):
                st.markdown("""
                            - Resources are allocated only to the services currently in demand, reducing idle compute time.
                            - Modular structure minimizes infrastructure overhead and avoids unnecessary full-system scaling.
                            """)

        with cols[2]:
            with st.expander("🔁 Long Term Adaptability", expanded=False):
                st.markdown("""
                            - Individual models can be updated, replaced, or retrained without disrupting the entire system.
                            - Supports CI/CD practices for rapid iteration and deployment of improvements.
                            """)
                
    st.divider()
    st.divider()
    
    st.header("📊 Smart Customer Segmentation Solution")
    st.markdown("---")
    with st.container():
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("""
            #### 🧠 Dynamic Clustering Framework
            """)
            
            with st.expander("📊 Data Processing Architecture", expanded=False):
                st.markdown("""
                            - **Initial Data Clustering**: Hierarchical clustering for segment identification
                            - **Real-time Data Processing**: Incremental PCA for updating insights
                            - **Scalable Clustering**: BIRCH for lower cost and faster clustering building on the initial clustering
                            """)
        with col2:
            st.markdown("""
            #### 🧠 Benefits
            """)
            st.success("""
            **Budget Reallocation Strategy**
            - 📱 Consumer Trends Tracking
            - 🚀 Real-Time Adaptive Model
            - 🎯 Competitive Positioning
            """)
    st.markdown("---")
    with st.container():
        
        with st.expander("🚀 Recommended Segment Approaches", expanded=False):
        
            cols = st.columns(3)
                
            with cols[0]:
                st.markdown("""
                **👴 Conservative Savers**
                - Retirement planning workshops
                - CD ladder strategies
                - Estate planning services
                """)
                
            with cols[1]:
                st.markdown("""
                **👔 High-Value Customers**
                - Private banking offers
                - Personal loans with lower rates
                - Priority concierge service
                """)
                
            with cols[2]:
                st.markdown("""
                **💳 Everyday Spenders**
                - Cashback boost campaigns
                - Introduce entry-level investment tools
                - Personalised mobile app features
                """)
                
            st.markdown("""
                <style>
                .stContainer {background-color: #f5faff; border-radius: 10px; padding: 20px;}
                </style>
                """, unsafe_allow_html=True)
        
    st.divider()
    st.divider()

    st.header("🧠 AI-Powered Campaign ROI Optimisation Solution")
    st.markdown("---")

    col1, col2 = st.columns([2,1], gap = "large")
    
    with col1:
        with st.expander("**📉 Gradient Boosting Classification**", expanded=False):
            st.markdown("""
                - **Predict** the ROI outcome of campaigns before they are even rolled out
                - Raw ROI values converted into distinct tiers
                - Classes are Low 🔴, Medium 🟡 and High 🟢
                """)
        st.markdown("""
        #### ♟️ Campaign Optimisation Strategy
        """)  
        st.markdown("""
            - Develop new, **personalised** campaigns based on derived insights
            - Input parameters into model to obtain an early ROI prediction
            - Result: Model output predicts potential ROI gains from campaign
            - Low/Medium ROI 🔴: Re-evaluate campaign parameters
            - High ROI 🟢: Roll out campaign, monitor performance for any red flags
            """)

    with col2:
        st.markdown("""
        #### 📈 Benefits of Pre-emptive ROI Prediction
        """)
        st.success("""
        - ⬇️ Lower risk of inefficient budget allocation towards unprofitable campaigns
        - 💰 Maximize returns by focusing resources on campaigns with the highest predicted ROI
        - ⚡ Classification nature of model enables immediate, actionable insights with regards to campaign planning
        """)
                
    st.divider()
    st.divider()

    st.header("🔮 Predictive Solutions for Proactive Engagement")
    # Action Section with Highlight
    st.markdown("---")

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        with st.expander("**📉 Logistic Model Churn Prediction**", expanded=False):
            st.markdown("""
                - Churn risk prediction
                - Early warning system
                - Retention scoring
                """)

        st.markdown("""
            #### 🛡️ Retention Strategy
            - **Churn Signals**:
                - 💸 High-Value: Transaction decline >40%
                - 📱 Everyday: Mobile logins drop >60%
                - 👴 General: Inactivity >180 days
            - Targeted products and offers to retain at-risk customers
            """)
        
    with col2:
    # --- Predictive Retention Benefits ---
        st.markdown("""
        #### 📈 Predictive Retention Benefits
        """)
        st.success("""
        **Benefits of Predictive Retention** 
        ✓ **Smart Targeting**:  
        - Auto-match products to customer life stages  
        - Trigger offers when churn risk >50%  

        ✓ **Measurable Impact**:  
        - ↓ preventable churn  
        - ↑ ROI on retention spend  
        - ↑ meaningful customer engagement
        """)


else:
    st.markdown("""
        <style>
        div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stMarkdownContainer"] li, 
        div[data-testid="stMarkdownContainer"] ul {
            font-size: 18pt !important;
            line-height: 1.5 !important;
            margin-bottom: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Conclusion")
    st.markdown("---")

    # Main Content Section
    st.header("🏆 Banking Analytics Impact Summary")

    # Revenue Growth Section
    with st.expander("🎯 Revenue Growth Through Precision Engagement", expanded=False):
        st.markdown("""
        - Higher conversion rates from segment-specific personalization
        - Increase in CLV via predictive next-best-product recommendations
        - Enhanced cross-selling opportunities through data-driven insights
        """)

    # Cost Efficiency Section
    with st.expander("📈 Cost Efficiency & Resource Optimization", expanded=False):
        st.markdown("""
        - Reduction in wasted ad spend through AI-driven budget allocation
        - Lower customer acquisition costs via improved targeting accuracy
        - Relieves the need for manual segmentation through automated clustering
        """)

    # Competitive Advantage Section
    with st.expander("💪 Future-Proof Competitive Advantage", expanded=False):
        st.markdown("""
        - Real-time adaptation to shifting customer behaviors and market trends
        - Regular model improvement through retraining on fresh campaign data
        - Predict and prevent customer attrition to maintain edge over competitors
        """)

