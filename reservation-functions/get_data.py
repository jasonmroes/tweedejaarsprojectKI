import re

def get_date_time(description:str): # description formatted as: LL0000000 YYYY-MM-DD HH:MM @ <room> - email
    date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", description)
    time = re.search("[0-9]{2}:[0-9]{2}", description)
    return date.group(0), time.group(0)