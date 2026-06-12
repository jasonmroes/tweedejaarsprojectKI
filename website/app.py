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

# df = pd.read_csv("website/output_processed.csv")
df = pd.read_csv("website/output_processed.csv", header=None,
                 names=["date", "hour", "revenue", "other"])

# table = df.pivot(index="hour", columns="date", values="revenue")

# choose how many weeks in advance the model predicts
how_many_weeks = 3

for week in range(how_many_weeks):
    fromm = week*24*7
    to = (week+1)*24*7
    table = df[fromm:to]
    table = table.pivot(index="hour", columns="date", values="revenue")
    st.table(table)

# first_week = table.head(how_many_weeks*)




