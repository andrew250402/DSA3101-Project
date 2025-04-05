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
pages = ["Introduction/Context", "Analysis/Insights", "Proposed Solution/ Deployment Strategy"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "Introduction/Context":
    st.title("Introduction/Context")
    

elif selected_page == "Analysis/Insights":
    st.title("Analysis/Insights")
    st.subheader("Smart Customer Segmentation for Precision Targeting")

    st.subheader("Optimized Campaigns for Maximum ROI")

    st.subheader("Predictive Analytics for Proactive Engagement")

    st.subheader("Balancing Personalization & Cost Efficiency")
else:
    st.title("Proposed Solution/ Deployment Strategy")

    st.subheader("Segment-Specific A/B Testing")

    st.subheader("Real-Time Model Updates")

    st.subheader("Privacy")

    st.subheader("Deployment Timeline")
