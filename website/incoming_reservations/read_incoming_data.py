import json
import pandas as pd
import streamlit as st
import requests
import os
import argparse
import yaml
from pathlib import Path

# run from project root with:
# python3 website/incoming_reservations/read_incoming_data.py --config configs/standard_config.yaml
BASE_DIR = Path(__file__).resolve().parent.parent.parent


####### get incoming payments from API ########
def get_payments(payments_url, headers, config):
    response_payments = requests.get(payments_url, headers=headers)

    if response_payments.status_code==200:
        payments_data = response_payments.json()
    else:
        print(f"Error: {response_payments.status_code}, {response_payments.text}")

    if isinstance(payments_data, str):
        payments_data = json.loads(payments_data)

    df_embedded = payments_data['_embedded']
    df_payments = df_embedded['payments']

    df = pd.json_normalize(df_payments, sep="_")
    
    df.to_csv(BASE_DIR/config["incoming_data_folder"]/"mollie_incoming_payments.csv", index=False)
    return df


####### get incoming refunds from API ########
def get_refunds(refunds_url, headers, config):
    response_refunds = requests.get(refunds_url, headers=headers)

    if response_refunds.status_code==200:
        data_refunds = response_refunds.json()
    else:
        print(f"Error: {response_refunds.status_code}, {response_refunds.text}")

    if isinstance(data_refunds, str):
        data_refunds = json.loads(data_refunds)

    df_embedded = data_refunds['_embedded']
    df_payments = df_embedded['refunds']

    df = pd.json_normalize(df_payments, sep="_")
    df.to_csv(BASE_DIR/config["incoming_data_folder"]/"mollie_incoming_refunds.csv", index=False)

    return df


####### get incoming chargebacks from API ########
def get_chargebacks(chargebacks_url, headers, config):
    response_chargebacks = requests.get(chargebacks_url, headers=headers)

    if response_chargebacks.status_code==200:
        data_chargebacks = response_chargebacks.json()
    else:
        print(f"Error: {response_chargebacks.status_code}, {response_chargebacks.text}")

    if isinstance(data_chargebacks, str):
        data_chargebacks = json.loads(data_chargebacks)

    df_embedded = data_chargebacks['_embedded']
    df_payments = df_embedded['chargebacks']

    df = pd.json_normalize(df_payments, sep="_")
    df.to_csv(BASE_DIR/config["incoming_data_folder"]/"mollie_incoming_chargebacks.csv", index=False)

    return df


if __name__ == "__main__":
    # The incoming reservations from API are incoming as 
    # json files. That is done in terminal by -o payments.json
    # This code turns json file into csv file, ready for preprocessing

    parser = argparse.ArgumentParser(description="Reading the incoming data from Duke Of Tokyo API/")
    parser.add_argument("--config", required=True,
                        help="Path(s) to YAML config file(s). Pass multiple to train in sequence.")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        cfg = yaml.load(f, Loader = yaml.FullLoader)

    CONFIG = cfg["data"]

    payments_url = "https://api.mollie.com/v2/payments"
    refunds_url = "https://api.mollie.com/v2/refunds"
    chargebacks_url = "https://api.mollie.com/v2/chargebacks"

    mollie_api_key_amsterdam = st.secrets["MOLLIE_API_KEY_AMSTERDAM"]
    headers = {"Authorization": f"Bearer {mollie_api_key_amsterdam}"}

    if os.path.exists(CONFIG["incoming_data_folder"]):
        if os.path.exists(CONFIG["incoming_data_folder"]/"mollie_incoming_refunds"):
            os.remove(CONFIG["incoming_data_folder"]/"mollie_incoming_refunds.csv")
            print("Previous refunds deleted")
        if os.path.exists(CONFIG["incoming_data_folder"]/"mollie_incoming_payments.csv"):
            os.remove(CONFIG["incoming_data_folder"]/"mollie_incoming_payments.csv")
            print("Previous payments deleted")
        if os.path.exists(CONFIG["incoming_data_folder"]/"mollie_incoming_chargebacks.csv"):
            os.remove(CONFIG["incoming_data_folder"]/"mollie_incoming_chargebacks.csv")
            print("Previous chargebacks deleted")
        
    else:
        print("The folder did not exist")
        os.makedirs(CONFIG["incoming_data_folder"], exist_ok=True)

    get_payments(payments_url, headers, CONFIG)
    get_refunds(refunds_url, headers, CONFIG)
    get_chargebacks(chargebacks_url, headers, CONFIG)


