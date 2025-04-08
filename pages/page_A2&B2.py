import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_utilities import read_csv, read_image, read_model

# Load data
customer_engagement_df = read_csv('customer_engagement.csv')
customer_df = read_csv('customers.csv')

# Set page config
st.set_page_config(page_title="Marketing Campaign Analysis", layout="wide")

# Create sidebar for page navigation
st.sidebar.title("Navigation")

# Define pages
pages = ["A2", "B2"]

# Radio buttons for page selection
selected_page = st.sidebar.radio("Go to", pages)


# Function to get best advertising platform
def get_best_advertising_platform(cust_id, threshold=0.3):
    best_platform_list = []
    all_possible_channels = ['Email', 'Mobile App Notifications', 'SMS', 'Direct Mail']

    # Case 1: Not an existing customer
    if cust_id not in customer_df["customer_id"].values:
        st.warning("This customer is not an existing customer in our database.")
        return all_possible_channels, None

    # Case 2: Existing customer with marketing history
    elif (cust_id in customer_df["customer_id"].values) and (
            cust_id in customer_engagement_df["customer_id"].values):
        historical_campaign_data = customer_engagement_df[customer_engagement_df["customer_id"] == cust_id]

        successful_delivery_frequency = historical_campaign_data.groupby("channel").agg(
            send_count=('channel', 'count'),
            delivered_count=('delivered', lambda x: (x == "Yes").sum())
        ).reset_index()

        successful_delivery_frequency["success_rate"] = (
                successful_delivery_frequency['delivered_count'] /
                successful_delivery_frequency['send_count']
        )

        filter_not_receptive_channels = successful_delivery_frequency[
            successful_delivery_frequency["success_rate"] < threshold
            ]

        not_receptive_channels = filter_not_receptive_channels["channel"].unique().tolist()

        for channel in all_possible_channels:
            if channel in not_receptive_channels:
                continue
            else:
                best_platform_list.append(channel)

        return best_platform_list, successful_delivery_frequency

    # Case 3: Existing customer but no marketing history
    else:
        st.warning("This customer exists but has never been advertised to before.")
        return all_possible_channels, None


ctr_img = read_image("A2_ctr.jpg")
cv_rate_img = read_image("A2_conversion_rate.png")
platform_img = read_image("B2_platform.png")
width_image = 850

# Campaign Analysis Page
if selected_page == "A2":
    st.title("What factors are most strongly correlated with customer engagement in marketing campaigns?")

    # Display first plot with analysis
    st.subheader("Click-Through Rate Analysis by Day & Channel")
    st.image(ctr_img, width=width_image)

    # Analysis of the CTR plot
    st.write("""
    ### Key Insights from Click-Through Rate Analysis:

    1. **Channel Performance:**
       - **Mobile App Notifications** consistently show the highest click-through rates across most days of the week, peaking on Thursday.
       - **Email** and **Direct Mail** show relatively consistent performance but are generally less effective than mobile channels.

    2. **Day of Week Patterns:**
       - **Thursday** appears to be the optimal day for Mobile App Notifications with the highest CTR.
       - **Sunday** shows strong performance for SMS campaigns.

    """)
    st.write("***")

    # Display second plot with analysis
    st.subheader("Conversion Rate Analysis by Product Type")
    st.image(cv_rate_img, width=width_image)

    # Analysis of the Conversion Rate plot
    st.write("""
    ### Key Insights from Conversion Rate Analysis:

    1. **Product Performance:**
       - **Savings Accounts** have the highest conversion rate (approximately 12%), making them the most successful product offering.
       - **Mortgage** products show the second-highest conversion rate (approximately 11.5%).
       - **Investment Products** and **Credit Cards** demonstrate moderate conversion rates.
       - **Personal Loans** and **Auto Loans** have the lowest conversion rates.

    2. **Product Category Patterns:**
       - Savings products (Savings Accounts, Wealth Management) generally outperform lending products.
       - Mortgages performs better than Personal Loans and Auto Loans.
    """)
    st.write("***")

    # Define function to calculate weighted metrics
    def weighted_metrics(df):
        total_delivered = df["delivered"].sum()
        if total_delivered == 0:  # Avoid division by zero
            return pd.Series({"conversion_rate": 0, "click_through_rate": 0, "open_rate": 0})
    
        return pd.Series({
            "conversion_rate": (df["conversion_status"] * df["delivered"]).sum() / total_delivered,
            "click_through_rate": (df["clicked"] * df["delivered"]).sum() / total_delivered,
            "open_rate": (df["opened"] * df["delivered"]).sum() / total_delivered,
        })
    a2_data = read_csv('A2_metrics.csv')
    a2_data.rename(columns={"engagement_day": "day"}, inplace=True)
    engagement_metrics = ['click_through_rate', 'conversion_rate', 'open_rate']
    x_metrics = ['day', 'channel', 'recommended_product_name', 'cluster']
    y_metric = st.selectbox("Y-axis metric", options=engagement_metrics)
    x_metric = st.selectbox("X-axis metric", options=x_metrics)
    colour_options = ["None"] + [m for m in x_metrics if m != x_metric and m != "cluster"]
    color_by = st.selectbox("Color by", options=colour_options)
    # Groupby based on color_by (if selected)
    group_cols = [x_metric] if color_by == "None" else [x_metric, color_by]
    new_df = a2_data.groupby(group_cols).apply(weighted_metrics).reset_index()
    day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if x_metric == "day" or color_by == "day":
        new_df["day"] = new_df["day"].map(lambda x: day_labels[x])
    fig = px.bar(
        new_df,
        x=x_metric,
        y=y_metric,
        color=None if color_by == "None" else color_by,
        barmode="group",
        title=f"{y_metric} by {x_metric}" + (f" & {color_by}" if color_by != "None" else "")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Conclusion")

    st.write("""
    -   Channels and product type have the strongest influence on engagement. SMS campaigns see peak conversions on Mondays, while Mobile App Notifications generate higher open and click-through rates on Thursdays. 
    -   Mortgage and Savings Accounts campaigns lead in conversions. 
    -   To maximise efficiency, banks should prioritise SMS campaigns on Mondays for higher conversions and mobile notifications on Thursday for engagement. 
    -   As for products, Mortgage and Savings Accounts should be marketing priorities. We should also emphasise channel and product-focused engagement efforts more than cluster-based and day-specific strategies.
    """)
# Advertising Recommender Page
elif selected_page == "B2":
    st.title("What strategies can we implement to optimise marketing campaigns in real-time?")

    # Display and analyze the platform chart
    st.subheader("Advertisement Delivery Success Rate by Channel")
    st.image(platform_img, width=width_image)

    # Analysis of the platform chart
    st.write("""
    ### Key Insights from Advertisement Delivery Analysis:

    1. **Channel Delivery Performance:**
       - **Direct Mail** has the lowest delivery success rate.
       - **Email**, **SMS**, and **Mobile App Notifications** demonstrate excellent delivery rates (90%+ success).
       - **Mobile App Notifications** show a small percentage (approximately 10%) of failed deliveries.

    2. **Business impact:**
       - Advertising funds are wasted on customers who are not receptive to advertisement through certain channels, especially direct mail.

    """)
    st.write("***")
    # Explanation of the recommendation logic
    st.write("""
    ### How Our Recommendation System Works:

    The system follows these decision rules when making channel recommendations:

    1. **For New Customers (not in database):**
       - All channels (Email, Mobile App Notifications, SMS, Direct Mail) will be recommended

    2. **For Existing Customers Without Advertising History:**
       - All channels will be recommended

    3. **For Existing Customers With Advertising History:**
       - Success rate is calculated for each channel based on past delivery success
       - Only channels with success rates above the specified threshold are recommended
       - This personalized approach focuses on channels most likely to reach the specific customer, saving costs in the long run.

    The adjustable threshold allows marketers to balance between reach (lower threshold) and efficiency (higher threshold) in their campaigns.
    """)

    # Input form in the main content area (not sidebar)
    with st.container():
        st.subheader("Input Parameters")
        st.write("Note: Existing customer ID is between 0 and 10000 but you can enter any value and see what happens")
        # Use columns for a better layout
        col1, col2 = st.columns([1, 2])


        with col1:
            # Customer ID input
            cust_id = st.text_input("Enter Customer ID:", value="123")

            # Threshold slider
            threshold = st.slider(
                "Select Success Rate Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Channels with success rate below this threshold will be excluded"
            )

            submit_button = st.button("Submit")

        with col2:
            st.write("### How It Works")
            st.write("""
            This tool recommends the best advertising channels for a specific customer based on their past engagement history.

            - Enter a customer ID to analyze their past interactions
            - Adjust the threshold to control how strict the recommendations should be
            - Click Submit to see the recommendations
            """)

    # Results section
    if submit_button:
        if cust_id.isnumeric():
            cust_id = int(cust_id)

        st.markdown("---")
        st.subheader(f"Recommendation for Customer ID: {cust_id}")

        # Get recommendations
        recommended_channels, delivery_data = get_best_advertising_platform(cust_id, threshold)

        # Use columns for results display
        col1, col2 = st.columns(2)

        with col1:
            # Display delivery data if available
            if delivery_data is not None:
                st.subheader("Customer Advertisement History")
                st.dataframe(delivery_data)

        with col2:
            # Always display recommendations
            st.subheader("Recommended Advertising Channels")

            if isinstance(recommended_channels, list):
                recommendations_df = pd.DataFrame({
                    "Recommended Channels": recommended_channels
                })
                st.dataframe(recommendations_df)
            else:
                st.error("No recommendations could be generated for this customer.")
