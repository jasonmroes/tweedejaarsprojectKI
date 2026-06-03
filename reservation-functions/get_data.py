import re
from datetime import date

def get_date_time_room(description:str): # description formatted as: LL0000000 YYYY-MM-DD HH:MM @ Room - email
    date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", description)
    time = re.search("[0-9]{2}:[0-9]{2}", description)
    room = re.search("@ ([A-Z][a-z]*)", description)
    return date.group(0), time.group(0), room.group(1)

def description_to_seperate(in_file, out_file):
    pass

def get_days_to_date(date1:str, date2:str): #date1/2 formatted as YYYY-MM-DD
    year1, month1, day1 = date1.split("-")
    year2, month2, day2 = date2.split("-")
    d1 = date(year1, month1, day1)
    d2 = date(year2, month2, day2)
    dif = d2 - d1
    return dif.days # Return number of days between date1 and date2