import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


file_path = "../Data DSA3101/"

# Read the CSV files
# campaigns_df = pd.read_csv(file_path + "campaigns.csv")
# churn_df = pd.read_csv(file_path + "churn.csv")
customer_engagement_df = pd.read_csv(file_path + "customer_engagement.csv")
customer_df = pd.read_csv(file_path + "customers.csv")
# digital_usage_df = pd.read_csv(file_path + "digital_usage.csv")
# transactions_summary_df = pd.read_csv(file_path + "campaigns.csv")


def get_best_advertising_platform(cust_id):
    """
    :param cust_id: the customer id
    :return: list of the best advertising platforms
    """
    best_platform_list = []
    all_possible_channels = ['Email', 'Mobile App Notifications', 'SMS', 'Direct Mail']
    print("list of all possible channels:", all_possible_channels)

    # case 1: cust_id is not an existing customer, recommend all possible advertising channels
    if cust_id not in customer_df["customer_id"]:
        print("\nNot an existing customer")
        return all_possible_channels

    # case 2: cust_id is an existing customer and has been marketed to before, find the channels they are not receptive to
    elif (cust_id in customer_df["customer_id"].values) and (cust_id in customer_engagement_df["customer_id"].values):
        # get the historical campaign data for this customer id
        historical_campaign_data = customer_engagement_df[customer_engagement_df["customer_id"] == cust_id]

        # for each channel utilised to advertise to the customer, count the number of times the advertisement was delivered successfully.
        successful_delivery_frequency = historical_campaign_data.groupby("channel").agg(
            send_count=('channel','count'),
            delivered_count=('delivered', lambda x: (x == "Yes").sum())
        ).reset_index()


        # find the ratio of delivered_count/send_count
        successful_delivery_frequency["success_rate"] = successful_delivery_frequency['delivered_count'] / successful_delivery_frequency['send_count']

        # if the ratio of successful delivery is less than 30%, the customer is deemed as not receptive of the advertising channel
        filter_not_receptive_channels = successful_delivery_frequency[successful_delivery_frequency["success_rate"] < 0.3]

        # get the unique channels where the customer is not receptive
        not_receptive_channels = filter_not_receptive_channels["channel"].unique().tolist()


        print(f"\nAdvertisment history of customer id: {cust_id}")
        print(successful_delivery_frequency)

        # keep only channels the customer is receptive to or has not been used before
        # an empty list will be returned if the customer is not receptive to ALL possible methods of advertising
        for channel in all_possible_channels:
            if channel in not_receptive_channels:
                continue
            else:
                best_platform_list.append(channel)


        if not_receptive_channels != []:
            print("\nConclusion: customer is not receptive to these channel(s):", not_receptive_channels)

        if best_platform_list == []:
            print("\nConclusion: customer is not receptive of all possible advertising channels")

        if best_platform_list == all_possible_channels:
            print("\nConclusion: customer is receptive of all possible advertising channels")

        else:
            return best_platform_list


    # case 3: cust_id is an existing customer in the database but has not been marketed to before, recommend all possible advertising channels
    else:
        print("\nCustomer has never been advertised to before")
        return all_possible_channels

    return best_platform_list



def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


# print("Test cases:")
# print("1) Not a customer (cust id outside range) -> recommend all channels")
# print("2) Never advertised to before (cust id 1234) -> recommend all channels")
# print("3) Receptive to all advertising channels (cust id 123) -> recommend all channels")
# print("4) Not receptive to direct mail (cust id 5) -> recommend all channels except direct mail")

while True:
    print("\n-------------------------------------------------------------------------")
    customer_id = safe_cast(input("Please enter a customer id (0 to 10000): "), int)
    print("\nrecommended channels:",get_best_advertising_platform(customer_id))
    print("-------------------------------------------------------------------------\n\n\n\n\n\n\n")


