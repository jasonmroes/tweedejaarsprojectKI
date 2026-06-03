import re
from datetime import date

def get_date_time_room(description:str):
    # description formatted as: LL0000000 YYYY-MM-DD HH:MM @ Room - email
    date = re.search("([0-9]{4})-([0-9]{2})-([0-9]{2})", description)
    time = re.search("([0-9]{2}):([0-9]{2})", description)
    room = re.search("@ ([A-Z][a-z]*)", description)
    # return dict with year, month, day, hour, minute, room (int, int, int, int, int, str)
    return {'year':int(date.group(1)), 'month':int(date.group(2)), 'day':int(date.group(3)), \
            'hour':int(time.group(1)), 'minute':int(time.group(2)), 'room':room.group(1)}


def get_days_to_date(date_dict1: dict, date_dict2 : dict): #date1/2 formatted as YYYY-MM-DD
    d1 = date(date_dict1['year'], date_dict1['month'], date_dict1['day'])
    d2 = date(date_dict2['year'], date_dict2['month'], date_dict2['day'])
    dif = abs(d2 - d1)
    return dif.days # Return number of days between date1 and date2


# info = get_date_time_room("AMC374YR 2026-03-27 15:30 @ Akihabara - sdgr@gmail.com")
# info2 = get_date_time_room("AMC37AYR 2026-02-27 15:30 @ Akihabara - sdgr@gmail.com")
# print(get_days_to_date(info, info2))