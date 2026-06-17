from preprocess_mollie import preprocess_mollie
from preprocessing import get_location_split, merge_weather

if __name__ == '__main__':
    save_folder = "../data"
    data_path = "../data/mollie_reservations.csv"
    # weather_path = ""

    # preprocess
    df = preprocess_mollie(data_path, incoming=True)

    # make location specific, location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df = get_location_split(df, 'AM')

    # add weather data
    # df = merge_weather(df, df_weather)

    # write to csv
    df.to_csv(f'{save_folder}/preprocessed_mollie_incoming.csv', index=False)
    print("DONE!")