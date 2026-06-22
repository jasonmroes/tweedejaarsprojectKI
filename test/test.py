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


from data_functions.functions import process_output

predictions = pd.read_csv(BASE_DIR/"test/predictions.csv", index_col=0)

output = process_output(predictions, 24, 336)