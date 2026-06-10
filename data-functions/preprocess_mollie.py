import pandas as pd
import reservations
import date_functions
import holidays

# ratio of cancelled reservations: 0.06921421289128875 of valid reservation data 
# (don’t forget to not count the walkins)


def calculate_refunded_ratio(df):
    refunded_count = 0
    for val in df['amountRefunded_value']:
        if val != None and val > 0:
            refunded_count += 1
    return refunded_count / len(df['id'])

# ------------------------------------------------------------


def drop_refunded(df):
    invalid_rows = []
    for idx, val in df['amountRefunded_value'].items():
        if val != None and val > 0:
            invalid_rows.append(idx)
    
    df = df.drop(index=invalid_rows)
    return df


def seperate_description(df):
    df = df.dropna(subset=['description']).copy()

    dates = []
    hours = []
    locations = []
    # rooms = []

    # Some rows don't contain date and time in their description, which makes them unusable
    invalid_rows = []

    for idx, desc in df['description'].items():
        dct = reservations.get_date_time_room(desc)

        dates.append(dct['date'])
        hours.append(dct['hour'])
        locations.append(dct['location'])
        # rooms.append(dct['room'])

        if dct['location'] == 'None':
            invalid_rows.append(idx)

    df['date'] = dates
    df['hour'] = hours
    df['location'] = locations
    # df['room'] = rooms

    df = df.drop(index=invalid_rows)

    return df

def createdAt_seperate(df):
    df = df.dropna(subset=['createdAt']).copy()

    dates = []

    # Some rows don't contain date and time in their description, which makes them unusable
    invalid_rows = []

    for idx, desc in df['createdAt'].items():
        date = reservations.get_createdAt_date(desc)

        dates.append(date)

        if date == None:
            invalid_rows.append(idx)

    df['createdAt'] = dates

    df = df.drop(index=invalid_rows)

    return df

def add_day_of_week_and_holiday(df):
    holiday_dct = holidays.NL(years=[2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 
                                 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036])
    dates = df['date']
    days = []
    holidays_lst = []
    for date in dates:
        date_dct = date_functions.get_date_dict(date)
        day_of_week = date_functions.date_to_day(date_dct)
        days.append(day_of_week)
        holiday = date_functions.date_to_holiday(date_dct, holiday_dct)
        holidays_lst.append(holiday)
    df['day of week'] = days
    df['national holiday'] = holidays_lst
    return df

def filter_paid(df):
    invalid_rows = []
    for idx, desc in df['status'].items():
        if desc != 'paid':
            invalid_rows.append(idx)

    df = df.drop(index=invalid_rows)

    return df

def preprocess_mollie(reservations_file: str, incoming: bool = False):
    """
    reservervations_file: location of mollie-all-payments.csv
    
    Drops refunded, seperates the description into 'date', 'hour',
    'location' and adds 'day of week'. Removes all invalid
    data (data with an incomplete description). 
    Adds column 'walk-in' with value 0 (False) for all columns
    Returns the new dataframe.
    """
    df = pd.read_csv(reservations_file)
    df = seperate_description(df)
    df = add_day_of_week_and_holiday(df)
    df = filter_paid(df)
    # delete all but createdAt, amount_value, date, hour, location, day of week
    df = df[['createdAt', 'amountRemaining_value', 'date', 'hour', 'location', 'day of week', 'national holiday']]
    
    df['walk-in'] = [0] * len(df['date'])
    df['consumption'] = [0] * len(df['date'])
    df = df.rename(columns={"amountRemaining_value": "price"})
    df = df.dropna(subset=['price']).copy()

    if incoming:
        df = createdAt_seperate(df)
    return df



if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    preprocess_mollie(reservations_file)


        


