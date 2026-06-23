import pandas as pd
import re
import calendar
import holidays
import os
from datetime import timedelta
import datetime

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def add_reservations_xd(df, days_list):
    col_names = []
    lst = [[] for _ in range(len(days_list))]
    for i in days_list:
        xd = f'reservations_{i}d'
        col_names.append(xd)

    for idx, d in df['createdAt'].items():
        days_between = abs((pd.to_datetime(d) - pd.to_datetime(df.loc[idx, 'date'])).days)
        for j in range(len(days_list)):
            if days_between >= days_list[j]:
                lst[j].append(df.loc[idx, 'price'])
            else:
                lst[j].append(0)
                
    for idx in range(len(col_names)):
        df[col_names[idx]] = lst[idx]

    return df

def add_zeros(df, start_date, end_date):
    dates = []
    dates_range = pd.date_range(start=start_date, end=end_date)
    dates_range = [date.strftime('%Y-%m-%d') for date in dates_range]
    hours = list(range(0,24)) * len(dates_range)
    for date in dates_range:
         dates += [date] * 24
    zeros_list = [0] * len(hours)
    zeros_dct = {'price': zeros_list, 'date': dates, 'hour': hours}
    zeros_df = pd.DataFrame(zeros_dct)
    df1 = pd.concat((df, zeros_df))
    return df1


def date_to_day(date_dict: dict): # Use get_date_dict first
    # wd = ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag']  # days
    # return wd[calendar.weekday(date_dict['year'], date_dict['month'], date_dict['day'])]

    return calendar.weekday(date_dict['year'], date_dict['month'], date_dict['day'])

def date_to_holiday(date_dict: dict, holiday_dct): # Use get_date_dict first
    # Deze functie geeft alleen officieel erkende feestdagen (waarop je vrij hebt)
    # Andere feestdagen/dagen krijgen None
    # holiday_translation = {"New Year's Day": "Nieuwjaarsdag",
    #                        "Good Friday": "Goede Vrijdag",
    #                        "Easter Sunday": "Paaszondag",
    #                        "Easter Monday": "Paasmaandag",
    #                        "King's Day": "Koningsdag",
    #                        "Ascension Day": "Hemelvaart",
    #                        "Whit Sunday": "Pinksterzondag",
    #                        "Whit Monday": "Pinkstermaandag",
    #                        "Christmas Day": "Eerste Kerstdag",
    #                        "Second Day of Christmas": "Tweede Kerstdag"
    #                        }
    if datetime.date(date_dict['year'], date_dict['month'], date_dict['day']) in holiday_dct:
        return 1
    else:
        return 0

def get_date_dict(date1:str, yearfirst:bool = True):
    # if formatted YYYY-MM-DD yearfirst = True, if formatted DD-MM-YYYY yearfirst = False
    if yearfirst: 
        year, month, day = map(int, date1.split("-"))
    else:
        day, month, year = map(int, date1.split("-"))

    return {'year': year, 'month': month, 'day': day}

def get_time_dict(time: str): # formatted HH:MM:(SS)
    time_lst = map(int, time.split(":"))
    hour = time_lst[0]
    minute = time_lst[1]
    return {'hour': hour, 'minute': minute}


def add_months(df):

    date = df.dropna(subset=['date']).copy()
    

    months = []
    for idx, desc in df['date'].items():

        month = desc[5:7]
        months.append(int(month))
    
    months = pd.Series(months)
    date["month"] = months


    return date

def group_by_date(df):
    df1 = df.copy()
    df1 = df1.groupby(['date', 'hour'], as_index=False).sum(numeric_only=True)
    return df1

def agg_by(df, columns):
    grouped = df.groupby(columns, as_index=False)

    dct = {}
    for column in df.columns:
        if column == 'price' or 'reservations' in column:
            dct[column] = 'sum'
        else:
            dct[column] = lambda x: x.iloc[0]

    return grouped.agg(dct)

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


def get_start_end_date(df):
    df1 = df.copy()
    df1['date'] = pd.to_datetime(df1['date'])
    last_date = df1['date'].max()
    first_date = df1['date'].min()
    return first_date, last_date
    

def preprocess_exports(df_sales, start_date, end_date, location:str = ""):

    if location != "":
        df1 = get_location_split(df_sales, location)

    df1 = df_sales.drop(columns=['timestamp', 'salesAreaId', 'is_karaoke_sku',
                              'salesAreaName', 'billNumber', 'billSuffix', 
                              'articleId', 'articleName', 
                              'quantity', 'waiter_item', 'row_key', 'location'])

    df_walkin = get_karaoke(df1)
    df_consumptions = get_consumptions(df1)

    df_walkin = df_walkin.drop(columns=['departmentName'])
    df_consumptions = df_consumptions.drop(columns=['departmentName'])

    df_walkin = df_walkin.dropna(subset=['price']).copy()
    df_consumptions = df_consumptions.dropna(subset=['price']).copy()

    df_walkin = remove_big_values(df_walkin)
    df_consumptions = remove_big_values(df_consumptions)

    df_walkin = add_zeros(df_walkin, start_date=start_date, end_date=end_date)
    df_consumptions = add_zeros(df_consumptions, start_date=start_date, end_date=end_date)

    df_walkin = group_by_date(df_walkin)
    df_consumptions = group_by_date(df_consumptions)

    df_walkin['consumption'] = [0] * len(df_walkin['price'])
    df_consumptions['consumption'] = [1] * len(df_consumptions['price'])

    df_walkin['walk-in'] = [1] * len(df_walkin['price'])
    df_consumptions['walk-in'] = [1] * len(df_consumptions['price'])
    
    return df_walkin, df_consumptions

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

def negative_to_zero(df, columns):
    for column in columns:
        for idx, val in df[column].items():
            if val < 0:
                df.loc[idx, column] = 0
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
        dct = get_date_time_room(desc)

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
        date = get_createdAt_date(desc)

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
        date_dct = get_date_dict(date)
        day_of_week = date_to_day(date_dct)
        days.append(day_of_week)
        holiday = date_to_holiday(date_dct, holiday_dct)
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

def remove_big_values(df):
    invalid_rows = []

    for idx, price in df['price'].items():

        if price > 1000 or price < -1000:
            invalid_rows.append(idx)

    df = df.drop(index=invalid_rows)

    return df

def preprocess_mollie(df, start_date=None, end_date=None, location: str = "", from_date = None, res_days = [7]):

    df = seperate_description(df)

    if from_date:
        df = get_future(df, from_date)
    if location != "":
        df = get_location_split(df, location)

    df = df.rename(columns={"amountRemaining_value": "price"})
    df = df.dropna(subset=['price']).copy()
    df = remove_big_values(df)

    df = add_reservations_xd(df, res_days)
    keep_columns = ['date', 'hour', 'price'] + [f'reservations_{x}d' for x in res_days]
    # delete all but price, date, hour, and reservations_xds
    df = df[keep_columns]
    if start_date == None or end_date == None:
        start_date, end_date = get_start_end_date(df)
    df = add_zeros(df, start_date=start_date, end_date=end_date)
    df = group_by_date(df)
    df['walk-in'] = [0] * len(df['price'])
    df['consumption'] = [0] * len(df['price'])
    return df

def get_location_split(df, location:str):
    # location: (Amsterdam - 'AM', Utrecht - 'UT', Rotterdam - 'RO', Groningen - 'GR')
    df = df[df['location'] == location]
    return df


def merge_weather(df, df_weather):
    merged = df.merge(
        df_weather,
        on="date",
        how="left",
        suffixes=("", "_new")
    )

    weather_cols = [c for c in df_weather.columns if c != "date"]

    for col in weather_cols:
        if col in df.columns:
            merged[col] = merged[f"{col}_new"].combine_first(merged[col])
            merged.drop(columns=f"{col}_new", inplace=True)

    return merged

def get_date_time_room(description:str):
    # description formatted as: LL0000000 YYYY-MM-DD HH:MM @ Room - email
    date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", description)
    time = re.search("([0-9]{2}):([0-9]{2})", description)
    room = re.search("@ ([A-Z][a-z]*)", description)
    location = re.search("([A-Z][A-Z])", description)
    if date == None or time == None or room == None or location == None:
        return {'date':'None', 'hour':-1, 'minute':-1, \
            'location':'None', 'room':'None'}
    # return dict with year, month, day, hour, minute, room (int, int, int, int, int, str)
    return {'date':date.group(1), 'hour':int(time.group(1)), 'minute':int(time.group(2)), \
            'location':location.group(1), 'room':room.group(1)}

def get_createdAt_date(description:str):
    date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", description)
    if date == None:
        return None
    return date.group(0)


def get_days_to_date(date_dict1: dict, date_dict2 : dict): #date1/2 formatted as YYYY-MM-DD
    
    d1 = datetime.date(date_dict1['year'], date_dict1['month'], date_dict1['day'])
    d2 = datetime.date(date_dict2['year'], date_dict2['month'], date_dict2['day'])
    dif = abs(d2 - d1)
    return dif.days # Return number of days between date1 and date2


def get_future(df, start_date):
    return df[pd.to_datetime(df['date']) > start_date]

def get_past(df, end_date):
    return df[pd.to_datetime(df['date']) <= end_date]

def get_between_dates(df, start_date, end_date):
    df1 = get_future(df, start_date)
    return get_past(df1, end_date)

def seperate_datetime(df):
    # turn a datetime column into 2 columns "date" and "time"
    df = df.dropna(subset=['datetime']).copy()

    dates = []
    hours = []
    

    for idx, desc in df['datetime'].items():

        date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", desc)
        time = re.search("([0-9]{2}):([0-9]{2})", desc)

        dates.append(date.group())
        hours.append(time.group())


    df['date'] = dates
    df['hour'] = hours

    dataframe = df[['date', 'hour', 'price']]
    return dataframe

def process_output(df, offset, horizon):
    # turn index e.g. 83002 into datetime column

    df = df.reset_index(names="steps")

    may_31 = datetime.datetime(2026, 5, 31, 23, 0)
    next_hour = may_31 + timedelta(hours=offset)
    datetimes = []

    for _ in range(horizon):
        next_hour = next_hour + timedelta(hours=1)
        datetimes.append(str(next_hour))
    
    datetimes = pd.Series(datetimes).T

    df['datetime'] = datetimes
    df = df[['price', 'datetime']]
    df['price'] = df['price'].round().astype(int)

    
    df = seperate_datetime(df)
    print(df.head())

    return df


