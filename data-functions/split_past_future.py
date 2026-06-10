import pandas as pd
from preprocessing import get_location_split
from datetime import datetime

def get_future(df, start_date):
    return df[pd.to_datetime(df['date']) > start_date]

def get_past(df, end_date):
    return df[pd.to_datetime(df['date']) <= end_date]

def get_between_dates(df, start_date, end_date):
    df1 = get_future(df, start_date)
    return get_past(df1, end_date)

if __name__ == "__main__":

    save_folder = "../data"

    file = "../data/preprocessed_data/preprocessed_all.csv"

    full_df = pd.read_csv(file)

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    full_df = get_location_split(full_df, 'UT')

    full_df_future = get_future(full_df, datetime(2026,5,31))
    full_df_past = get_past(full_df, datetime(2026,5,31))
    full_df_future_days = get_between_dates(full_df, datetime(2026,5,31), datetime(2026,6,21))
    
    # write to csvs (location specific)
    full_df_future.to_csv(f'{save_folder}/future_data_UT.csv', index=False)
    full_df_past.to_csv(f'{save_folder}/past_data_UT.csv', index=False)
    full_df_future_days.to_csv(f'{save_folder}/future_days_data_UT.csv', index=False)

    print("DONE!")