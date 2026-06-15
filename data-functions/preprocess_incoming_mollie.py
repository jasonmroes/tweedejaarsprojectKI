from functions import preprocess_mollie, merge_weather, add_months, \
                      get_start_end_date, add_day_of_week_and_holiday, negative_to_zero

import pandas as pd
import datetime

def preprocess_incoming_data(df_weather, df_reservations, location="", from_date=None):

    # preprocess
    full_df = preprocess_mollie(df = df_reservations, location = location, from_date = from_date)

    full_df = negative_to_zero(full_df)

    full_df = add_day_of_week_and_holiday(full_df)

    full_df = add_months(full_df)

    full_df = merge_weather(full_df, df_weather)

    return full_df

if __name__ == "__main__":
    data_path = "../data/mollie_reservations.csv"

    weather_file = "../data/weather/WeatherAmsterdam.csv"

    save_file = "../data/fully_preprocessed_data_AM_incoming.csv"

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    location = 'AM'

    from_date = None

    # if you only want to get data from a certain point onwards (not including inputed date). datetime.datetime(year, month, day):
    # from_date = datetime.datetime(2026, 5, 31)

    # read csvs
    df_weather = pd.read_csv(weather_file)
    df_reservations = pd.read_csv(data_path)

    # preprocess into one df
    full_df = preprocess_incoming_data(df_weather, df_reservations, location, from_date)
    
    # write to csv
    full_df.to_csv(save_file, index=False)

    print("DONE!")