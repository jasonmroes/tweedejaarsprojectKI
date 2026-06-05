import pandas as pd
from preprocess_mollie import add_day_of_week



def filter_and_concat(df1, df2):
    # delete all columns but 'location', 'price', 'date', 'hour', 'departmentName' for both

    df1 = df1.drop(columns=['timestamp', 'salesAreaId', 'is_karaoke_sku',
                              'salesAreaName', 'billNumber', 'billSuffix', 
                              'articleId', 'articleName', 
                              'quantity', 'waiter_item', 'row_key'])
    df2 = df2.drop(columns=['timestamp', 'salesAreaId', 'is_karaoke_sku',
                                'salesAreaName', 'billNumber', 'billSuffix',
                                'tableNumber', 'articleId', 'articleName',
                                'quantity', 'waiter_item', 'row_key'])
    
    return pd.concat((df1, df2))

def abbreviate_locations(df):
    df = df.replace({'location': {'amsterdam': 'AM', 'utrecht': 'UT',
                                  'rotterdam': 'RO', 'karaoke_bar': 'GR'}})
    return df

def get_karaoke(df):
    filtered_df = df[df['departmentName'] == 'KARAOKE']
    return filtered_df

def get_consumptions(df):
    filtered_df = df[df['departmentName'] != 'KARAOKE']
    return filtered_df

    

def preprocess_exports(sales_file: str, tables_file: str):
    """
    sales_file: location of untill_exports_sales_area.csv
    tables_file: location of untill_exports_sales_tables.csv
    
    Drops columns except for 'date', 'hour', 'location' and adds 'day of week'.
    Concatenates the two files into one file.
    Adds column 'walk-in' with value 1 (True) for all columns.
    Splits the one file into one with karaoke room walkins and one with only consumptions.
    Returns df_karaoke, df_consumptions
    """

    df_sales = pd.read_csv(sales_file)
    df_tables = pd.read_csv(tables_file)

    df1 = filter_and_concat(df_sales, df_tables)
    df1 = add_day_of_week(df1)
    df1 = abbreviate_locations(df1)
    df1['walk-in'] = [1] * len(df1['location'])

    df_karaoke = get_karaoke(df1)
    df_consumptions = get_consumptions(df1)

    df_karaoke = df_karaoke.drop(columns=['departmentName'])
    df_consumptions = df_consumptions.drop(columns=['departmentName'])
    
    return df_karaoke, df_consumptions

if __name__ == "__main__":
    sales_file = "../data/extern/untill_exports_sales_area.csv"
    tables_file = "../data/extern/untill_exports_tables.csv"

    df_karaoke, df_consumptions = preprocess_exports(sales_file, tables_file)
    df_karaoke.to_csv('data_new/preprocessed_exports_karaoke.csv')
    df_consumptions.to_csv('data_new/preprocessed_exports_consumptions.csv')





