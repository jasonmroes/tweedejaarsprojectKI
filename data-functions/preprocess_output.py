import pandas as pd
import date_functions
import re

def seperate_datetime(df):
    df = df.dropna(subset=['datetime']).copy()

    dates = []
    hours = []
    

    for idx, desc in df['datetime'].items():

        date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", desc)
        time = re.search("([0-9]{2}):([0-9]{2})", desc)

        dates.append(date.group())
        hours.append(time.group())


    df['date'] = dates
    df['hour'] = hours

    dataframe = df[['date', 'hour', 'price']]
    return dataframe

def add_day_of_week(df):
    
    dates = df['date']
    days = []
    for date in dates:
        date_dct = date_functions.get_date_dict(date)
        day_of_week = date_functions.date_to_day(date_dct)
        days.append(day_of_week)
      
    df['day of week'] = days
   
    return df


if __name__ == "__main__":
    output_file = "experiments/CatBoostModel_future_forecast.csv"
    df = pd.read_csv(output_file)
    three_columns = seperate_datetime(df)
    four_columns = add_day_of_week(three_columns)
    print(four_columns.head(10))
    df.to_csv()