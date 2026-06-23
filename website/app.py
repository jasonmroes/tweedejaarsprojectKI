import streamlit as st
import pandas as pd
import datetime
import sys
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_functions.incoming_data_functions import get_payments, get_refunds, get_chargebacks
from data_functions.functions import process_output
from data_functions.incoming_weather_functions import get_incoming_weather
from data_functions.preprocess_incoming_mollie import preprocess_incoming_data
from src.darts_prophet.inference import run_inference


# Run with streamlit run website/app.py
# st.set_page_config(page_title="Duke Of Tokyo", page_icon="microphone.png", layout="wide")
                   
# st.write("""
# # Duke Of Tokyo*
# """)

# st.image("Duke.jpeg")
# df = pd.read_csv("../experiments/CatBoostModel_future_forecast.csv")

# st.dataframe(df)
##########################################################################################
########################## PREPARE #######################################################
##########################################################################################

INFERENCE_CONFIG = BASE_DIR / "configs" / "TiDE_hid256_dr0.5.yaml"
with open(INFERENCE_CONFIG) as f:
    _inference_cfg = yaml.load(f, Loader=yaml.FullLoader)
TRAINING_DATA_PATH = BASE_DIR / _inference_cfg["data"]["data_path"]

load_dotenv()
KNMI_API_KEY = os.getenv("KNMI_API_KEY")
MOLLIE_API_KEY_AMSTERDAM = os.getenv("MOLLIE_API_KEY_AMSTERDAM")
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
upcoming_chargebacks = get_chargebacks(chargebacks_url, headers)
upcoming_refunds = get_refunds(refunds_url, headers)

##########################################################################################
########################## GET UPCOMING WEATHER #####################################
##########################################################################################

location      = "52.3105,4.7683"   #Schiphol Amsterdam Coordinates
base_url      = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
forecast_days = 15
total_days    = 21

upcoming_weather = get_incoming_weather(VISUAL_CROSSING_API_KEY, base_url, location, forecast_days, total_days)


##########################################################################################
########################## PREPROCESS UPCOMING DATA #####################################
##########################################################################################
location = 'AM'
res_days = [7,14,21]
from_date = datetime.datetime(2026, 5, 31)
sort = True
old_future_file = BASE_DIR/"data/old_future_AM.csv"
df_cancellations = pd.concat([upcoming_chargebacks, upcoming_refunds])

upcoming_weeks_data = preprocess_incoming_data(upcoming_reservations, upcoming_weather, old_future_file=old_future_file, location=location,
                                       df_cancellations=df_cancellations, from_date=from_date, sort=sort)

# The live weather feed uses different column names than the training data,
# and has no sunshine-hours forecast at all. Rename the ones with a direct
# equivalent (covers both past_covariates and future_covariates column
# names), and backfill sun_hours with its historical monthly average
# (climatology) since no live forecast for it exists.

sun_hours_by_month = pd.read_csv(TRAINING_DATA_PATH).groupby("month")["sun_hours"].mean()
upcoming_weeks_data["sun_hours"] = upcoming_weeks_data["month"].map(sun_hours_by_month)


##########################################################################################
#################### MAKE PREDICTIONS FOR UPCOMING 3 WEEKS ###############################
##########################################################################################

# inference.py uses flat imports ("from model_utils import ...") so its own
# directory must be on sys.path, and its YAML paths (data_path, save_dir) are
# relative to the repo root, so cwd must be BASE_DIR while it runs.


# Horizon => 336/24=14 days prediction
# Offset ensures that the prediction is made per week, starting from Monday
# Offset is 31 may + offset hours
HORIZON = 336
OFFSET = 24

predictions = run_inference(str(INFERENCE_CONFIG), upcoming_weeks_data, HORIZON, OFFSET)
df_predictions = process_output(predictions, OFFSET, HORIZON)


how_many_weeks = 2

for week in range(how_many_weeks):
    fromm = week*24*7
    to = (week+1)*24*7
    table = df_predictions[fromm:to]
    table = table.pivot(index="hour", columns="date", values="revenue")
    table = pd.concat([table.iloc[0:4], table.iloc[12:]])

    st.markdown(table.to_html(index=True), unsafe_allow_html=True)
