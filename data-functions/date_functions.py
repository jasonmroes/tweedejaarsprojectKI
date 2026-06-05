import calendar
from datetime import date

def date_to_day(date_dict: dict): # Use get_date_dict first
    wd = ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag']  # days
    return wd[calendar.weekday(date_dict['year'], date_dict['month'], date_dict['day'])]

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
    if date(date_dict['year'], date_dict['month'], date_dict['day']) in holiday_dct:
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


# date1 = get_date_dict("25-12-2025")
# date2 = get_date_dict("2025-12-26", yearfirst=True)

# print(date_to_day(date1))
# print(date_to_day(date2))
# print(date_to_holiday(date1))
