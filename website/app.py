import streamlit as st
import pandas as pd
 
# Run with streamlit run app.py
st.set_page_config(page_title="Duke Of Tokyo", page_icon="microphone.png", layout="wide")
                   
st.write("""
# Duke Of Tokyo*
""")

st.image("Duke.jpeg")
df = pd.read_csv("../experiments/CatBoostModel_future_forecast.csv")

st.dataframe(df)

