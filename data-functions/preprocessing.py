from preprocess_mollie import preprocess_mollie
from preprocess_exports import preprocess_exports
import pandas as pd

def get_location_split(df, location:str):
    # location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df = df[df['location'] == location]
    return df

def merge_weather(df, df_weather):
    merged_df = pd.merge(df, df_weather, on='date', how='left')
    return merged_df


# ratio of cancelled reservations: 0.06921421289128875 of valid reservation data 
# (don’t forget to not count the walkins)

# to get rooms uncomment the code for rooms in preprocess_mollie.py - seperate_description(df)
# to get other columns go to remove their name from the drop command in 
# preprocess_mollie.py - preprocess_mollie / preprocess_exports.py - preprocess_exports

if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    sales_file = "../data/extern/untill_exports_sales_area.csv"
    tables_file = "../data/extern/untill_exports_tables.csv"

    weather_file = "../data/weather/WeatherAmsterdam.csv"

    save_folder = "../data/preprocessed_data/AM"
    save_folder2 = "../data/preprocessed_data"

    # get weather df
    df_weather = pd.read_csv(weather_file)

    # preprocess into 3 dfs
    df_reservations = preprocess_mollie(reservations_file)
    df_karaoke, df_consumptions = preprocess_exports(sales_file)

    # add ids
    df_reservations['id'] = range(0, len(df_reservations['date']))
    df_karaoke['id'] = range(len(df_reservations['date']), len(df_karaoke['date']) + len(df_reservations))
    df_consumptions['id'] = range(1000000, 1000000 + len(df_consumptions['date']))

    # add weather data
    df_reservations = merge_weather(df_reservations, df_weather)
    df_karaoke = merge_weather(df_karaoke, df_weather)
    df_consumptions = merge_weather(df_consumptions, df_weather)

    # get concatenated dfs
    full_df_karaoke = pd.concat((df_reservations, df_karaoke))
    full_df = pd.concat((full_df_karaoke, df_consumptions))

    # write to csvs
    df_reservations.to_csv(f'{save_folder2}/preprocessed_mollie.csv', index=False)
    df_karaoke.to_csv(f'{save_folder2}/preprocessed_walkins_karaoke.csv', index=False)
    df_consumptions.to_csv(f'{save_folder2}/preprocessed_consumptions.csv', index=False)
    full_df_karaoke.to_csv(f'{save_folder2}/preprocessed_all_karaoke.csv', index=False)
    full_df.to_csv(f'{save_folder2}/preprocessed_all.csv', index=False)

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df_reservations = get_location_split(df_reservations, 'AM')
    df_karaoke = get_location_split(df_karaoke, 'AM')
    df_consumptions = get_location_split(df_consumptions, 'AM')
    full_df_karaoke = get_location_split(full_df_karaoke, 'AM')
    full_df = get_location_split(full_df, 'AM')
    
    # write to csvs (location specific)
    df_reservations.to_csv(f'{save_folder}/preprocessed_mollie.csv', index=False)
    df_karaoke.to_csv(f'{save_folder}/preprocessed_walkins_karaoke.csv', index=False)
    df_consumptions.to_csv(f'{save_folder}/preprocessed_consumptions.csv', index=False)
    full_df_karaoke.to_csv(f'{save_folder}/preprocessed_all_karaoke.csv', index=False)
    full_df.to_csv(f'{save_folder}/preprocessed_all.csv', index=False)

    print("DONE!")
