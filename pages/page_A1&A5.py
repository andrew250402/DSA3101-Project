import streamlit as st
from streamlit_utilities import read_csv, read_image, read_model
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("page_A1&A5")
st.title("1.1 Customer Segmentation")
st.write("### Picking the right clustering algorithm")
col1, col2 = st.columns(2)
with col1:
    st.header("Customer Segmentation for Existing Customers")
    st.image(read_image("A1_H_existing_cluster.png"))
    st.subheader("Hierarchical Clustering")
    st.image(read_image("A1_K_exisiting_cluster.png"))
    st.subheader("K Means Clustering")
with col2:
    st.header("Customer Segmentation for Incoming Customers")
    st.image(read_image("A1_H_new_cluster.png"))
    st.subheader("Hierarchical Clustering")
    st.image(read_image("A1_K_new_cluster.png"))
    st.subheader("K Means Clustering")
pad1, mark, pad2 = st.columns([1, 2, 1])
st.empty()
st.markdown("#### Both K-Means and hierarchical clustering are able to segments the clusters well. But hierarchical edges over k-means due to slightly clearer segmentation and its slight robustness to outliers as compared to K-means. "
"Additionally, hierarchical clustering can handle mixed data types better, and has no assumption on equal sized clusters (KMeans have the tendancy to create equally sized clusters).")
st.empty()
st.markdown("#### As such, hierarchical clustering will be chosen to segment existing customers.")
st.empty()
st.title("1.2 Real-time and Scalable Segmentation")
st.empty()
st.markdown(
    """
    ## 🔍 Understanding Hierarchical Clustering(HC)
    #### HC builds a tree-like structure (dendrogram) to group data.
    ### However, it struggles with **large datasets** due to:
    #### - 🛑 High memory usage (storing distance matrices)
    #### - 🛑 Slow computation (O(n²) complexity)

    #### **Solution:** We use **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)** and **Incremental PCA** to make HC scalable! 🚀
    """
)
padding, col1, padding2 = st.columns([1, 2, 1])
with padding:
    st.empty()
with col1:
    st.header("Dendrogram")
    st.image(read_image("A5_dendrogram.png"))
with padding2:
    st.empty()

import streamlit as st

import streamlit as st

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

st.markdown("### ⚡ How BIRCH (CF Trees) and Incremental PCA Accelerate Hierarchical Clustering")

# Create columns with visible divider
col1, divider, col2 = st.columns([5, 1, 5], gap="small")

with col1:
    st.markdown("""
    #### **CF Trees (BIRCH)**
    🏗️ **Balanced Iterative Reducing and Clustering using Hierarchies**  
    - **Tree structure** that stores data summaries (Clustering Features) instead of raw points  
    - Each CF contains:  
      ✓ `N` (point count)  
      ✓ `LS` (linear sum)  
      ✓ `SS` (squared sum)  
    - **Benefits for HC**:  
      ▸ 90%+ memory reduction via compression  
      ▸ Avoids O(n²) distance matrix calculations  
      ▸ Enables streaming data support  

    #### **Incremental PCA**  
    📉 **Dimensionality Reduction for Scalable HC**  
    - Processes data in **mini-batches** (unlike standard PCA)  
    - **Key advantages**:  
      ✓ Reduces "curse of dimensionality" in HC  
      ✓ Removes noise/irrelevant features  
      ✓ Updates model with new data (online learning)  
      ✓ Preserves 95%+ variance with fewer dimensions  
    """)

with divider:
    # Vertical divider line
    st.markdown("""
    <div style='border-left: 2px solid white; height: 800px; margin: 0 auto;'></div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    #### **Synergistic Effects**  
    🔄 **Combined workflow**:  
    1. IPCA first reduces dimensions  
    2. BIRCH then clusters the compressed data  
    3. Traditional HC merges the final subclusters  
    
    **Key Advantages**:  
    ▸ 10-100x faster execution  
    ▸ Processes >1M data points  
    ▸ Real-time streaming capability  
    ▸ Lower memory requirements  
    </div>
    """, unsafe_allow_html=True)

# Horizontal divider at bottom
st.divider()

st.subheader("⚡ Benefits of BIRCH and Incremental PCA")

data = {
    "Feature": [
        "Scalability (Large Datasets)",
        "Memory Efficiency",
        "Computational Speed",
        "Streaming/Online Learning",
        "Impact on HC"
    ],
    "BIRCH 🏗️ (Balanced Iterative Reducing and Clustering using Hierarchies)": [
        "✅ <strong>Highly scalable</strong> (uses CF Trees for compression)",
        "✅ <strong>Dramatically reduces memory</strong> by summarizing data into subclusters",
        "✅ <strong>Faster than traditional HC</strong> (avoids pairwise distance matrices)",
        "✅ <strong>Incremental clustering</strong> (adapts to new data points efficiently)",
        "✅ <strong>Optimizes HC</strong> via CF Tree structure + threshold-based merging"
    ],
    "Incremental PCA 📉 (Principal Component Analysis)": [
        "✅ <strong>Handles large datasets</strong> via batch processing",
        "✅ <strong>Low-memory usage</strong> (processes data in mini-batches)",
        "✅ <strong>Accelerates HC</strong> by reducing dimensionality first",
        "✅ <strong>Supports online updates</strong> (model adjusts to new data)",
        "✅ <strong>Improves HC efficiency</strong> by removing noise/redundant features"
    ]
}

df = pd.DataFrame(data)

# Generate HTML with Pandas Styler
html = df.style.set_properties(**{
    'font-size': '18pt',
    'padding': '12px',
    'color': '#ffffff',  # White text
    'background-color': 'transparent'  # Transparent background
}).to_html(escape=False)

# Dark theme CSS
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
        color: #4fc3f7 !important;  # Light blue for bold text
    }}
</style>
{html}
"""

st.markdown(html, unsafe_allow_html=True)