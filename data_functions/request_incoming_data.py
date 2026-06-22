import json
import pandas as pd
import streamlit as st
import requests
import os
from pathlib import Path

# run from project root with:
# python3 data-functions/request_incoming_data.py --config configs/standard_config.yaml
BASE_DIR = Path(__file__).resolve().parent.parent

####### get incoming payments from API ########
def get_payments(payments_url, headers):
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
    #df.to_csv("incoming_data/mollie_incoming_payments.csv", index=False)
    return df


####### get incoming refunds from API ########
def get_refunds(refunds_url, headers):
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
    #df.to_csv("incoming_data/mollie_incoming_refunds.csv", index=False)

    return df


####### get incoming chargebacks from API ########
def get_chargebacks(chargebacks_url, headers):
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

    return df



