import pandas as pd
import reservations
import date_functions

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

        if dct['hour'] == 0:
            invalid_rows.append(idx)

    df['date'] = dates
    df['hour'] = hours
    df['location'] = locations
    # df['room'] = rooms

    df = df.drop(index=invalid_rows)

    return df

def add_day_of_week(df):
    dates = df['date']
    days = []
    for date in dates:
        date_dct = date_functions.get_date_dict(date)
        day_of_week = date_functions.date_to_day(date_dct)
        days.append(day_of_week)
    df['day of week'] = days
    return df

def preprocess_mollie(reservations_file: str):
    """
    reservervations_file: location of mollie-all-payments.csv
    
    Drops refunded, seperates the description into 'date', 'hour',
    'location' and adds 'day of week'. Removes all invalid
    data (data with an incomplete description). 
    Adds column 'walk-in' with value 0 (False) for all columns
    Returns the new dataframe.
    """
    df = pd.read_csv(reservations_file)
    df = drop_refunded(df)
    df2 = seperate_description(df)
    df3 = add_day_of_week(df2)
    # delete all but id, amount_value, date, hour, location, day of week
    df3 = df3.drop(columns=['mode', 'status', 'isCancelable', 'description', 'method', 
                     'metadata', 'details', 'profileId','mandateId',
                     'customerId','sequenceType','locale','webhookUrl',
                     'redirectUrl','expiresAt','cancelledAt','failedAt',
                     'settledAt','_links','amount_currency',
                     'amountRefunded_currency','amountRefunded_value',
                     'amountRemaining_currency','amountRemaining_value',
                     'amountCaptured_currency','amountCaptured_value',
                     'amountChargedBack_currency','amountChargedBack_value',
                     'settlementAmount_currency','settlementAmount_value','mollie_client'])
    
    df3['walk-in'] = [0] * len(df3['id'])
    df3.rename(columns={"amount_value": "price"})
    return df3


if __name__ == "__main__":
    reservations_file = "../data/extern/mollie-all-payments.csv"
    preprocess_mollie(reservations_file)


        


