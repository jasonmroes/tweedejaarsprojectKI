from preprocess_mollie import preprocess_mollie
from preprocess_exports import preprocess_exports
import pandas as pd

# ratio of cancelled reservations: 0.06921421289128875 of valid reservation data 
# (don’t forget to not count the walkins)

if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    sales_file = "../data/extern/untill_exports_sales_area.csv"
    tables_file = "../data/extern/untill_exports_tables.csv"

    save_folder = "../data/preprocessed_data"

    df_reservations = preprocess_mollie(reservations_file)
    df_karaoke, df_consumptions = preprocess_exports(sales_file, tables_file)
    full_df_karaoke = pd.concat((df_reservations, df_karaoke))

    df_reservations.to_csv(f'{save_folder}/preprocessed_mollie.csv')
    df_karaoke.to_csv(f'{save_folder}/preprocessed_walkins_karaoke.csv')
    df_consumptions.to_csv(f'{save_folder}/preprocessed_consumptions.csv')
    full_df_karaoke.to_csv(f'{save_folder}/preprocessed_all_karaoke.csv')
