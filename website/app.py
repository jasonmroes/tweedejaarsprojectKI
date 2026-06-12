import streamlit as st
import pandas as pd
 
# Run with streamlit run app.py
st.set_page_config(page_title="Duke Of Tokyo", page_icon="website/microphone.png", layout="wide")
                   
st.write("""
# Duke Of Tokyo*
""")

st.image("website/Duke.jpeg")
# df = pd.read_csv("experiments/CatBoostModel_future_forecast.csv")

# st.dataframe(df)

df = pd.read_csv("website/output_processed.csv")

# df = df.groupby("hour")
st.table(df)



