from functions import preprocess_exports, preprocess_mollie, merge_weather, get_past, add_months, filter_paid, \
                      get_start_end_date, add_day_of_week_and_holiday, abbreviate_locations, negative_to_zero

import pandas as pd

def preprocess_past_data(df_reservations:pd.DataFrame, df_sales:pd.DataFrame, df_weather:pd.DataFrame, 
                         location:str="AM", sort:bool=False, res_days:list[int]=[7,14,21], incoming:bool=False):
    """
    df_reservations: The not preprocessed mollie reservations dataframe from "mollie-all-payments.csv"

    df_sales: The not preprocessed sales area exports dataframe from "untill_exports_sales_area.csv"

    df_weather: The dataframe containing the weather data per date

    location: Location to get the dataframe for. 
    'AM': Amsterdam, 'UT': Utrecht, 'RO': Rotterdam, 'GR': Groningen

    sort: True if you want to sort the data by date and hour. By default False.

    res_days: a list with ints for extra columns with the total reservations revenue of reservations
    that were made at least xn days before the date they go with the list being [x1, x2 .., xn]

    incoming: If you are using incoming data put this as True for preprocess_mollie() to work
    properly. If you are not using incoming data put this as False.

    Returns the preprocessed dataframe.
    """

    df_sales = abbreviate_locations(df_sales)

    # get end date
    start_date, end_date = get_start_end_date(df_sales)

    # preprocess into 3 dfs
    df_walkin, df_consumptions = preprocess_exports(df_sales, start_date, end_date, location)
    df_reservations = filter_paid(df_reservations)
    df_reservations = preprocess_mollie(df_reservations, start_date, end_date, location, res_days=res_days, incoming=incoming)


    # get concatenated dfs
    full_df_karaoke = pd.concat((df_reservations, df_walkin))
    full_df = pd.concat((full_df_karaoke, df_consumptions))

    full_df = get_past(full_df, end_date)

    full_df = negative_to_zero(full_df)

    full_df = add_day_of_week_and_holiday(full_df)

    full_df = add_months(full_df)

    full_df = merge_weather(full_df, df_weather)

    if sort:
        full_df = full_df.sort_values(by=['date', 'hour'])
    
    full_df = full_df.fillna(0)

    return full_df

if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    sales_file = "../data/extern/untill_exports_sales_area.csv"

    weather_file = "../data/weather/WeatherAmsterdam.csv"

    save_file = "../data/fully_preprocessed_data_AM_past.csv"

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    location = 'AM'
    res_days = [7,14,21]

    sort = True

    # read csvs
    df_weather = pd.read_csv(weather_file)
    df_reservations = pd.read_csv(reservations_file)
    df_sales = pd.read_csv(sales_file)

    # preprocess into one df
    full_df = preprocess_past_data(df_reservations, df_sales, df_weather, location, sort, res_days=res_days)
    
    # write to csv
    full_df.to_csv(save_file, index=False)

    print("DONE!")