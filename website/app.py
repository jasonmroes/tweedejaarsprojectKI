import streamlit as st
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_functions.incoming_data_functions import get_payments
from data_functions.incoming_weather_functions import get_incoming_weather
# Run with streamlit run website/app.py
st.set_page_config(page_title="Duke Of Tokyo", page_icon="microphone.png", layout="wide")
                   
st.write("""
# Duke Of Tokyo*
""")

st.image("Duke.jpeg")
df = pd.read_csv("../experiments/CatBoostModel_future_forecast.csv")

st.dataframe(df)
##########################################################################################
########################## PREPARE #######################################################
##########################################################################################

load_dotenv()
KNMI_API_KEY = os.getenv("KNMI_API_KEY")
MOLLIE_API_KEY_AMSTERDAM = st.secrets["MOLLIE_API_KEY_AMSTERDAM"]
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

if not KNMI_API_KEY:
    sys.exit("ERROR: KNMI_API_KEY not found in .env file.")
if not MOLLIE_API_KEY_AMSTERDAM:
    sys.exit("ERROR: MOLLIE_API_KEY_AMSTERDAM not found in .env file.")
if not VISUAL_CROSSING_API_KEY:
    sys.exit("ERROR: VISUAL_CROSSING_API_KEY not found in .env file.")

##########################################################################################
########################## GET UPCOMING RESERVATIONS #####################################
##########################################################################################
payments_url = "https://api.mollie.com/v2/payments"
refunds_url = "https://api.mollie.com/v2/refunds"
chargebacks_url = "https://api.mollie.com/v2/chargebacks"


headers = {"Authorization": f"Bearer {MOLLIE_API_KEY_AMSTERDAM}"}
upcoming_reservations = get_payments(payments_url, headers)

##########################################################################################
########################## GET UPCOMING WEATHER #####################################
##########################################################################################

location      = "52.3105,4.7683"   #Schiphol Amsterdam Coordinates
base_url      = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
forecast_days = 15
total_days    = 21

upcoming_weather = get_incoming_weather(VISUAL_CROSSING_API_KEY, base_url, location, forecast_days, total_days)

# merge into 1 dataset and preprocess
# run through the model
# output is a df with predictions for 3 weeks


