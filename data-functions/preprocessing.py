from preprocess_mollie import preprocess_mollie
from preprocess_exports import preprocess_exports
import pandas as pd

def get_location_split(df, location:str):
    # location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df = df[df['location'] == location]
    return df

# ratio of cancelled reservations: 0.06921421289128875 of valid reservation data 
# (don’t forget to not count the walkins)

# to get rooms uncomment the code for rooms in preprocess_mollie.py - seperate_description(df)
# to get other columns go to remove their name from the drop command in 
# preprocess_mollie.py - preprocess_mollie / preprocess_exports.py - preprocess_exports

if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    sales_file = "../data/extern/untill_exports_sales_area.csv"
    tables_file = "../data/extern/untill_exports_tables.csv"

    save_folder = "../data/preprocessed_data/AM"

    # preprocess into 4 dfs
    df_reservations = preprocess_mollie(reservations_file)
    df_karaoke, df_consumptions = preprocess_exports(sales_file, tables_file)
    full_df_karaoke = pd.concat((df_reservations, df_karaoke))

    # add ids
    df_reservations['id'] = range(0, len(df_reservations['date']))
    df_karaoke['id'] = range(len(df_reservations['date']), len(full_df_karaoke['date']))
    full_df_karaoke['id'] = range(0, len(full_df_karaoke['date']))
    df_consumptions['id'] = range(1000000, 1000000 + len(df_consumptions['date']))

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df_reservations = get_location_split(df_reservations, 'AM')
    df_karaoke = get_location_split(df_karaoke, 'AM')
    df_consumptions = get_location_split(df_consumptions, 'AM')
    full_df_karaoke = get_location_split(full_df_karaoke, 'AM')

    # write to csvs
    df_reservations.to_csv(f'{save_folder}/preprocessed_mollie.csv', index=False)
    df_karaoke.to_csv(f'{save_folder}/preprocessed_walkins_karaoke.csv', index=False)
    df_consumptions.to_csv(f'{save_folder}/preprocessed_consumptions.csv', index=False)
    full_df_karaoke.to_csv(f'{save_folder}/preprocessed_all_karaoke.csv', index=False)
