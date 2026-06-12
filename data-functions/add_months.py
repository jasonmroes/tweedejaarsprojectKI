import pandas as pd 

def add_months(df):

    date = df.dropna(subset=['date']).copy()
    

    months = []
    for idx, desc in df['date'].items():

        month = desc[5:7]
        months.append(int(month))
    
    months = pd.Series(months)
    date["month"] = months


    return date

if __name__ == "__main__":


    df = pd.read_csv("data/zeros_past_data_AM.csv")
    dataframe = add_months(df)
    dataframe.to_csv("data/zeros_past_data_AM_months.csv")