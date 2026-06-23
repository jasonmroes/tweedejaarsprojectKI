import pandas as pd
import datetime
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_functions.functions import merge_weather, add_months, get_future, add_zeros, agg_by, \
                      add_day_of_week_and_holiday, negative_to_zero, filter_paid, \
                      seperate_description, createdAt_seperate, get_location_split, \
                      remove_big_values, add_reservations_xd, get_start_end_date \
                      



def preprocess_incoming_data(df_reservations:pd.DataFrame, df_weather:pd.DataFrame, old_future_file:str="", 
                             df_cancellations:pd.DataFrame=None, location:str="AM",
                             from_date:datetime.datetime=datetime.datetime(2026, 5, 31), sort:bool=False, incoming:bool=True):
    """
    df_reservations: The not preprocessed newest incoming mollie reservations dataframe.

    df_weather: The dataframe containing the weather data per date

    old_future_file: The path to the csv file of the last concatenated data. This file will be automatically
    made/updated by this function if you give the path.

    df_cancellations: the dataframe concatenated of the incoming refunds and chargebacks

    location: Location to get the dataframe for. 
    'AM': Amsterdam, 'UT': Utrecht, 'RO': Rotterdam, 'GR': Groningen

    from_date: datetime.datetime(year, month, day). 
    Leave this None except if you only want the data from a certain day onwards.

    sort: True if you want to sort the data by date and hour. By default False.

    res_days: a list with ints for extra columns with the total reservations revenue of reservations
    that were made at least xn days before the date they go with the list being [x1, x2 .., xn]

    incoming: If you are using incoming data put this as True for preprocess_mollie() to work
    properly. If you are not using incoming data put this as False.

    Returns the preprocessed dataframe.
    """
    full_df = df_reservations
    
    full_df = filter_paid(full_df)

    if df_cancellations is not None:
        df_cancellations = df_cancellations.rename(columns={'settlementAmount_value': 'amountRemaining_value'})
        full_df = pd.concat([full_df, df_cancellations])
        full_df = full_df.reset_index(drop=True)
        
    full_df = seperate_description(full_df)

    if incoming:
        full_df = createdAt_seperate(full_df)

    if from_date:
        full_df = get_future(full_df, from_date)
    if location != "":
        full_df = get_location_split(full_df, location)
    full_df = full_df.rename(columns={"amountRemaining_value": "price"})
    full_df = full_df.dropna(subset=['price']).copy()
    full_df['price'] = full_df['price'].astype(float)
    full_df = remove_big_values(full_df)

    # full_df = add_reservations_xd(full_df, res_days)
    # keep_columns = ['description', 'date', 'hour', 'price'] + [f'reservations_{x}d' for x in res_days]
    keep_columns = ['description', 'date', 'hour', 'price']

    # delete all but description, price, date, hour, and reservations_xds
    full_df = full_df[keep_columns]
    
    full_df = merge_weather(full_df, df_weather)

    if old_future_file != "":
        if os.path.isfile(old_future_file):
            df_old_future = pd.read_csv(old_future_file)
            df_old_future = df_old_future[keep_columns]
            full_df = pd.concat([full_df, df_old_future])
            full_df = full_df.drop_duplicates(subset=['description', 'price'])

        full_df.to_csv(old_future_file, index=False)

    full_df = agg_by(full_df, ['description'])
    full_df = full_df.reset_index(drop=True)
    full_df = full_df.drop(columns=['description'])
    # full_df = negative_to_zero(full_df, ['price'] + [f'reservations_{x}d' for x in res_days])
    full_df = negative_to_zero(full_df, ['price'])

    start_date, end_date = get_start_end_date(full_df)
    full_df = add_zeros(full_df, start_date=start_date, end_date=end_date)
    full_df['walk-in'] = [0] * len(full_df['price'])
    full_df['consumption'] = [0] * len(full_df['price'])

    full_df = agg_by(full_df, ['date', 'hour'])
    full_df = full_df.reset_index(drop=True)

    full_df = add_day_of_week_and_holiday(full_df)
    full_df = add_months(full_df)

    full_df = merge_weather(full_df, df_weather)

    if sort:
        full_df = full_df.sort_values(by=['date', 'hour'])

    # Weather columns (whatever df_weather actually supplied) are left NaN
    # here wherever no real forecast was available — the caller fills those
    # with historical climatology instead of an unrealistic hard zero.
    weather_cols = [c for c in df_weather.columns if c != "date"]
    non_weather_cols = [c for c in full_df.columns if c not in weather_cols]
    full_df[non_weather_cols] = full_df[non_weather_cols].fillna(0)

    full_df = full_df.rename(columns={
    "temp": "temp_avg",
    "tempmin": "temp_min",
    "tempmax": "temp_max",
    "precip": "rain_mm",
    "windspeed": "wind_avg",
    "cloudcover": "cloud",
})
    
    return full_df