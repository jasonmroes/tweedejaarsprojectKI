import pandas as pd
from preprocess_mollie import add_day_of_week_and_holiday

def sort_by_date(df):
    df['datetime'] = df['date'].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by=['hour'])
    df = df.sort_values(by=['datetime'])
    df = df.drop(columns=['datetime'])
    return df

def add_zeros(df):
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
    df1 = pd.concat((df, zeros_df))
    return df1



if __name__ == "__main__":
    past_file = "../data/past_data_AM.csv"
    save_file = "../data/zeros_past_data_AM.csv"

    df = pd.read_csv(past_file)
    df = add_zeros(df)
    # df = sort_by_date(df)

    df.to_csv(save_file)
    print("DONE!")

    



