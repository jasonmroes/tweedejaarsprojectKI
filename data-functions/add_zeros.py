import pandas as pd
from preprocess_mollie import add_day_of_week_and_holiday
from preprocessing import merge_weather

def sort_by_date(df):
    df['datetime'] = df['date'].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by=['datetime'], ascending=False)
    df = df.drop(columns=['datetime'])
    return df

def add_zeros(df, weather_file):
    dates = []
    dates_range = pd.date_range(start='2023-12-05', end='2026-05-31')
    dates_range = [date.strftime('%Y-%m-%d') for date in dates_range]
    hours = list(range(0,24)) * len(dates_range)
    for date in dates_range:
         dates += [date] * 24
    zeros_list = [0] * len(hours)
    locations = ['AM'] * len(hours)
    zeros_dct = {'price': zeros_list, 'date': dates, 'hour': hours, 'location': locations,'walk-in': zeros_list,'consumption': zeros_list}
    zeros_df = pd.DataFrame(zeros_dct)
    zeros_df = add_day_of_week_and_holiday(zeros_df)
    zeros_df = merge_weather(zeros_df, weather_file)
    df1 = pd.concat((df, zeros_df))
    return df1



if __name__ == "__main__":
    past_file = "../data/past_data_AM.csv"
    save_file = "../data/zeros_past_data_AM.csv"
    weather_file = "../data/weather/WeatherAmsterdam.csv"

    df_weather = pd.read_csv(weather_file)
    df = pd.read_csv(past_file)

    df = add_zeros(df, df_weather)
    df = sort_by_date(df)

    df.to_csv(save_file, index=False)
    print("DONE!")

    



