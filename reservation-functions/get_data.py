import re

def get_date_time_room(description:str): # description formatted as: LL0000000 YYYY-MM-DD HH:MM @ Room - email
    date = re.search("([0-9]{4}-[0-9]{2}-[0-9]{2})", description)
    time = re.search("[0-9]{2}:[0-9]{2}", description)
    room = re.search("@ ([A-Z][a-z]*)", description)
    return date.group(0), time.group(0), room.group(1)