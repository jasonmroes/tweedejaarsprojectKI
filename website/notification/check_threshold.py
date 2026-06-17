import pandas as pd
from datetime import datetime, timedelta
from website.notification.send_email import send_mail
from pathlib import Path
# import argparse
# import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def check_threshold(filename):
    dataframe = pd.read_csv(filename)
    df = dataframe.copy()
    df = df[["price","date","hour"]]
    df = df.head()
    # group by hour and date
    df.groupby(['date', 'hour'])['price'].sum()
    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')

    # send alerts only over the upcoming 3 weeks
    now = datetime.now()
    over_three_weeks = now + timedelta(weeks=3)
    three_week_reservations_df = df[df['date'] < over_three_weeks]

    # If the grouped price is higher than the threshold, show in the table
    threshold = 400
    three_week_reservations_df = three_week_reservations_df[three_week_reservations_df["price"] > threshold]
    three_week_reservations_df.to_csv("website/incoming_reservations/incoming_data/risk_alerts.csv")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Reading the incoming data from Duke Of Tokyo API/")
    # parser.add_argument("--config", required=True,
    #                     help="Path(s) to YAML config file(s). Pass multiple to train in sequence.")
    # args = parser.parse_args()

    # config_path = args.config
    # with open(config_path) as f:
    #     cfg = yaml.load(f, Loader = yaml.FullLoader)

    # CONFIG = cfg["data"]
    incoming_reservations = "../../data/incoming_data/preprocessed_mollie_incoming.csv"
    check_threshold(incoming_reservations)