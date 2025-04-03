import streamlit as st
import pandas as pd
import plotly.express as px
# 
df = pd.read_csv("../data/A4_metrics.csv")
df['ROI'] = df['total_profit']/df['total_campaign_cost']
st.set_page_config(page_title="Marketing Campaign Analysis", layout="wide")

st.title("Marketing Campaign Analysis")
st.subheader("What are the key performance indicators (KPIs) for assessing the success of marketing campaigns?")
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

heatmap_metrics = ['deliver_rate', 'open_rate', 'click_rate', 'conversion_rate', 'engagement_score']

# Create a DataFrame just for the heatmap
heatmap_data = df[['campaign_name'] + heatmap_metrics].set_index('campaign_name').round(2)

# Generate the heatmap
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Metric", y="Campaign", color="Value"),
    x=heatmap_metrics,
    y=heatmap_data.index,
    color_continuous_scale="RdYlGn",
    text_auto = True,
    aspect="auto",
)

# Display in Streamlit
fig.update_layout(
    title="📊 Campaign Performance Heatmap",
    xaxis_title=None,
    yaxis_title=None,
    font=dict(size=12),
    margin=dict(l=200, r=20, t=50, b=20),  
    height=600,
    coloraxis_colorbar=dict(title="Score")
)

# Streamlit display
st.plotly_chart(fig, use_container_width=True)
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
st.subheader("Comparing different metrics across different campaigns")

# Define metrics
metrics = ['total_revenue', 'total_campaign_cost', 'total_profit', 'clv','ROI']
names = [m.replace("_", " ").title() for m in metrics]

# Layout: 4 columns for 4 buttons
cols = st.columns(5)
selected_metric = None

# Render buttons
for i, label in enumerate(names):
    if cols[i].button(label):
        selected_metric = metrics[i]

# Optional: Output selected metric
if selected_metric:
    st.success(f"📊 You selected: **{selected_metric.replace('_', ' ').title()}**")
# Checkbox to enable sorting
sort_enabled = st.checkbox("Sort by value (descending)", value=False)
# Display chart if a metric is selected

if selected_metric:
    chart_data = df.copy()
    if sort_enabled:
        chart_data = chart_data.sort_values(by = selected_metric, ascending=False)
    fig = px.bar(
        chart_data,
        y='campaign_name',
        x=selected_metric,
        orientation='h',
        title=f"{selected_metric.replace('_', ' ').title()} by Campaign"
    )

    fig.update_layout(xaxis_title=selected_metric.replace("_", " ").title(),
        yaxis_title="Campaign Name",
        showlegend=False)
    
    st.plotly_chart(fig)
else:
    st.info("👆 Click a metric button to visualize it.")

corr_metrics = ['engagement_score', 'conversion_rate', 'clv', 'total_profit']
corr_data = df[corr_metrics].corr()

# Plot correlation matrix
fig_corr = px.imshow(
    corr_data,
    text_auto=True,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    labels=dict(x="Metric", y="Metric", color="Correlation"),
    title="📈 Correlation Matrix",
    aspect="auto"
)

st.plotly_chart(fig_corr, use_container_width=True)

bub_metrics = ['engagement_score', 'conversion_rate', 'clv']
x_metric = st.selectbox("X-axis metric", options=bub_metrics)
y_metric = st.selectbox("Y-axis metric", options=[m for m in bub_metrics if m != x_metric])
color_by = st.selectbox("Color by", options=['customer_segment', 'recommended_product_name'])
size_by = st.selectbox("Size by", options=['total_revenue', 'total_campaign_cost','total_profit'])
fig = px.scatter(
    df,
    x=x_metric,
    y=y_metric,
    color=color_by,
    size=size_by,  # Optional
    hover_data=['campaign_name'],
    title=f"{x_metric} vs {y_metric} by {color_by}"
)
st.plotly_chart(fig, use_container_width=True)


# Unique values
segments = df['customer_segment'].unique().tolist()
products = df['recommended_product_name'].unique().tolist()

# 🔁 Persistent session state for filters
if 'selected_segments' not in st.session_state:
    st.session_state.selected_segments = segments.copy()

if 'selected_products' not in st.session_state:
    st.session_state.selected_products = products.copy()

# 🔘 Toggle segment selection
st.markdown("### Filter by Customer Segment")
seg_cols = st.columns(len(segments))
for i, seg in enumerate(segments):
    if seg in st.session_state.selected_segments:
        color = "#4CAF50"  # Active: Green
    else:
        color = "#E0E0E0"  # Inactive: Grey

    if seg_cols[i].button(seg, key=f"seg_btn_{seg}"):
        if seg in st.session_state.selected_segments:
            st.session_state.selected_segments.remove(seg)
        else:
            st.session_state.selected_segments.append(seg)

    seg_cols[i].markdown(f"<div style='height: 6px; background-color: {color}; border-radius: 4px;'></div>", unsafe_allow_html=True)

# 🔘 Toggle product selection
st.markdown("### Filter by Recommended Product")
prod_cols = st.columns(len(products))
for i, prod in enumerate(products):
    if prod in st.session_state.selected_products:
        color = "#2196F3"  # Active: Blue
    else:
        color = "#E0E0E0"  # Inactive: Grey

    if prod_cols[i].button(prod, key=f"prod_btn_{prod}"):
        if prod in st.session_state.selected_products:
            st.session_state.selected_products.remove(prod)
        else:
            st.session_state.selected_products.append(prod)

    prod_cols[i].markdown(f"<div style='height: 6px; background-color: {color}; border-radius: 4px;'></div>", unsafe_allow_html=True)

# 🧹 Filter dataframe
filtered_df = df[
    (df['customer_segment'].isin(st.session_state.selected_segments)) &
    (df['recommended_product_name'].isin(st.session_state.selected_products))
]

# 🎯 Select metric and show filtered data shape
group_metric = st.selectbox("Select a metric to visualize:", ['clv', 'total_profit','ROI'])

st.write(f"Filtered Data Shape: {filtered_df.shape}")
# Plot grouped bar chart
fig = px.bar(
    filtered_df,
    x='customer_segment',
    y=group_metric,
    color='recommended_product_name',
    barmode='group',
    title=f"{group_metric.replace('_', ' ').title()} per Recommended Product by Customer Segment",
    labels={
        'customer_segment': 'Customer Segment',
        'recommended_product_name': 'Recommended Product',
        group_metric: group_metric.replace('_', ' ').title()
    }
)

st.plotly_chart(fig, use_container_width=True)