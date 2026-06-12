import streamlit as st
import pandas as pd
 
# Run with streamlit run app.py
st.set_page_config(page_title="Duke Of Tokyo", page_icon="website/microphone.png", layout="wide")
                   
st.write("""
# Duke Of Tokyo*
""")

st.image("website/Duke.jpeg")

df = pd.read_csv("website/output_processed.csv", header=None,
                 names=["date", "hour", "revenue", "other"])

# choose how many weeks in advance the model predicts
how_many_weeks = 3
st.markdown("""
        <style>
        table {
            border-collapse: collapse;
            font-size: 20px;
            border: 6px solid salmon !important;
            margin: auto;
        }
        th, td {
            border: 2px solid black;
            padding: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

for week in range(how_many_weeks):
    fromm = week*24*7
    to = (week+1)*24*7
    table = df[fromm:to]
    table = table.pivot(index="hour", columns="date", values="revenue")
    table = pd.concat([table.iloc[0:4], table.iloc[12:]])

    st.markdown(table.to_html(index=True), unsafe_allow_html=True)






