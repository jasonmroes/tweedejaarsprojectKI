import pandas as pd
import datetime

def difference_hours(df: pd.DataFrame):
    """
    date: datetime.datetime(year, month, day, hour)
    """
    monday = get_last_monday()
    df1 = df.copy()
    df1['date'] = pd.to_datetime(df1['date'])
    last_date = df1['date'].max()
    df1 = df1[df1['date'] == last_date]
    last_hour = df1['hour'].max()
    last_date_hour = last_date + datetime.timedelta(hours=int(last_hour))
    duration = last_date_hour - monday
    duration_in_s = duration.total_seconds()
    duration_in_h = divmod(duration_in_s, 3600)[0]
    return duration_in_h


def get_last_monday():
    dt = datetime.datetime.now()
    monday = dt - datetime.timedelta(days=dt.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    return monday