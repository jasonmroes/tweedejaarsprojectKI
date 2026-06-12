import pandas as pd
import date_functions
import re
import numpy as np

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

def correct_opening_hours(df):
    ''''
    The expected revenue is 0 for not opening hours
    '''

    df = df.copy()
    df.loc[(df['hour'] == "04:00"), 'price'] = 0
    df.loc[(df['hour'] == "05:00"), 'price'] = 0
    df.loc[(df['hour'] == "06:00"), 'price'] = 0
    df.loc[(df['hour'] == "07:00"), 'price'] = 0
    df.loc[(df['hour'] == "08:00"), 'price'] = 0
    df.loc[(df['hour'] == "09:00"), 'price'] = 0
    df.loc[(df['hour'] == "10:00"), 'price'] = 0
    df.loc[(df['hour'] == "11:00"), 'price'] = 0
    df.loc[(df['hour'] == "12:00"), 'price'] = 0
    df.loc[(df['hour'] == "13:00"), 'price'] = 0
    df.loc[(df['hour'] == "14:00"), 'price'] = 0
    df.loc[(df['hour'] == "15:00"), 'price'] = 0
    df.loc[(df['hour'] == "16:00"), 'price'] = 0
    df.loc[(df['hour'] == "17:00"), 'price'] = 0



    return df



if __name__ == "__main__":
    output_file = "experiments/CatBoostModel_future_forecast.csv"
    df = pd.read_csv(output_file)
    three_columns = seperate_datetime(df)
    four_columns = add_day_of_week(three_columns)
    four_columns = four_columns[2:]
    df = correct_opening_hours(four_columns)

    df.to_csv("website/output_processed.csv", header=False, index=False)