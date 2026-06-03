import calendar
import holidays
import csv

def date_to_day(date: str): # date is formatted DD-MM-YYYY
    day, month, year = map(int, date.split("-"))
    wd = ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag']  # days
    return wd[calendar.weekday(year, month, day)]

def date_to_holiday(date:str): # date is formatted DD-MM-YYYY
    # Deze functie geeft alleen officieel erkende feestdagen (waarop je vrij hebt)
    # Andere feestdagen krijgen gewoon None
    holiday = holidays.NL().get(date)
    holiday_translation = {"New Year's Day": "Nieuwjaarsdag",
                           "Good Friday": "Goede Vrijdag",
                           "Easter Sunday": "Paaszondag",
                           "Easter Monday": "Paasmaandag",
                           "King's Day": "Koningsdag",
                           "Ascension Day": "Hemelvaart",
                           "Whit Sunday": "Pinksterzondag",
                           "Whit Monday": "Pinkstermaandag",
                           "Christmas Day": "Eerste Kerstdag",
                           "Second Day of Christmas": "Tweede Kerstdag"
                           }
    
    return holiday_translation[holiday]