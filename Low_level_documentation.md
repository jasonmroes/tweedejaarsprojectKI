## preprocess_past_data

To train the model you first need the preprocessed past data. You can get this by using preprocess_past_data(df_reservations:pd.DataFrame, df_sales:pd.DataFrame, df_weather:pd.DataFrame, location:str="AM", sort:bool=False,res_days:list[int]=[7,14,21]) in file preprocess_past_data.py
You need the following variables for this and the functions in the file functions.py:

  df_reservations: The not preprocessed mollie reservations dataframe from "mollie-all-payments.csv"

  df_sales: The not preprocessed sales area exports dataframe from "untill_exports_sales_area.csv"

  df_weather: The dataframe containing the weather data per date

  location: Location to get the dataframe for. 
  'AM': Amsterdam, 'UT': Utrecht, 'RO': Rotterdam, 'GR': Groningen

  sort: True if you want to sort the data by date and hour. By default False.

  res_days: a list with ints for extra columns with the total reservations revenue of reservations
  that were made at least xn days before the date they go with the list being [x1, x2 .., xn]

  incoming: If you are using incoming data put this as True for preprocess_mollie() to work
  properly. If you are not using incoming data put this as False.

  Returns the preprocessed dataframe.

You can also use the file itself which runs it automaticcally with all required variables as long as you make sure all used file paths are correct. 

The function does the following things:

Abbreviate the locations of df_sales to make it usable for filtering by location.

Seperate df_sales by walk-in karaoke data and consumptions data.

Seperate description of df_reservations into seperate columns of date, hour and location.

Filter data to only have data that has status 'paid'

Drop all rows with empty price values and price values bigger than 1000 euros or smaller than -1000 euros (to make sure bad data such as infinite numbers are not used for training).

Add a column for how much money reservations made for each number of days in res_days 


Add zeros for all hours and dates where there is no data from the first date in df_sales to the last date.

Group by date and hour.

Add rows consumption (0 if not a consumption, 1 if consumption) and walk-in (1 if consumption or walk-in, 0 if reservation)

Convert negative values for price and reservations_xd to 0.

Add days of the week as int 0-6 (0 = monday, 6 = sunday) and add if it's a holiday (0 if no, 1 if yes)

Merge the dataframe with the weather data.

Drop rows except for the weather data, date, hour, price, reservations_xd

If sort = True sort the dataframe by date and hour.

Return the preprocessed dataframe ready for training.

## preprocess_incoming_data
To get predictions using upcoming reservations you use in the preprocess_incoming_mollie.py the function
preprocess_incoming_data(df_reservations:pd.DataFrame, df_weather:pd.DataFrame, old_future_file:str="", 
                             df_cancellations:pd.DataFrame=None, location:str="AM",
                             from_date:datetime.datetime=datetime.datetime(2026, 5, 31), sort:bool=False, res_days:list[int]=[7,14,21], incoming:bool=True)


  df_reservations: The not preprocessed newest incoming mollie reservations dataframe.

  df_weather: The dataframe containing the weather data per date

  old_future_file: The path to the csv file of the last concatenated data. This file will be automatically
  made/updated by this function if you give the path.

  df_cancellations: the dataframe concatenated of the incoming refunds and chargebacks

  location: Location to get the dataframe for. 
  'AM': Amsterdam, 'UT': Utrecht, 'RO': Rotterdam, 'GR': Groningen

  from_date: datetime.datetime(year, month, day). 
  Leave this None except if you only want the data from a certain day onwards.

  sort: True if you want to sort the data by date and hour. By default False.

  res_days: a list with ints for extra columns with the total reservations revenue of reservations
  that were made at least xn days before the date they go with the list being [x1, x2 .., xn]

  incoming: If you are using incoming data put this as True for preprocess_mollie() to work
  properly. If you are not using incoming data put this as False.

  Returns the preprocessed dataframe.


You can also use the file itself which runs it automaticcally with all required variables as long as you make sure all used file paths are correct. 

The function does the following things:

Concatenate df_cancellations with df_reservations

Seperate description of df_reservations into seperate columns of date, hour and location.

Filter data to only have data that has status 'paid'

Drop all rows with empty price values and price values bigger than 1000 euros or smaller than -1000 euros (to make sure bad data such as infinite numbers are not used for training).

Add a column for how much money reservations made for each number of days in res_days.

Merge the dataframe with the weather data.

Concatenate with old_future_file and save that concatenated file as the new old_future_file.

Aggregate by description to refund data that has been refunded.

Convert negative values for price and reservations_xd to 0 (which also removed unnecessary refunds)

Add zeros for all hours and dates where there is no data from the first date in df_sales to the last date.

Group by date and hour.

Add rows consumption (0 if not a consumption, 1 if consumption) and walk-in (1 if consumption or walk-in, 0 if reservation)

Add days of the week as int 0-6 (0 = monday, 6 = sunday) and add if it's a holiday (0 if no, 1 if yes)

Merge the dataframe with the weather data. (for data that has not merged with the weather data yet)

Drop rows except for the weather data, date, hour, price, reservations_xd

If sort = True sort the dataframe by date and hour.

Return the preprocessed dataframe ready for training.

## preprocess_average_revenue

To get a csv with the average revenue per combination of hour/day of week/month over a certain number of years,
use preprocess_average_revenue(df:pd.DataFrame, years:int=1).

  df: Fully preprocessed past DataFrame

  years: amount of years before the last date in the dataset to use for the average

  Return the dataframe with data of the average revenues.

Gets the maximum date of the dataset and subtracts years from it --> first_date

Make the dataframe only contain data after first_date

Aggregate and sum by date/hour

Only keep columns 'month', 'day of week', 'hour', 'price'

Group by 'month', 'day of week', 'hour' and take the mean of the price

Round all prices to 2 decimals

Return the dataframe with data of the average revenues.

## need_send_risk_alert
To get the difference between the average and the predicted price and to see if that the difference is 25% more or less, 
use need_send_risk_alert(predicted_row, avg_rev_df:pd.DataFrame, less:float=-100, more:float=100).

  predicted_row: the row of the results for which you want to see if you need a risk alert

  avg_rev_df: the dataframe read from average_revenue_AM.csv containing the average revenue for each combination
  of month, day of week and hour.

  less: the minimum (negative) difference for the model to return less

  more: the minimum difference for the model to return more

  returns: a string "more", "less" or "same" and a float with the difference between the predicted and average revenue.

Create columns 'day of week', 'month' for predicted_row using data from 'date' and update hour to not contain minutes

Locate the average that has the same 'day of week', 'month' and hour as predicted_row

Calculate the difference between the predicted price and the average

If the difference is equal to or more that 25% of the average or variable more, whichever is bigger, return 'more', difference

If the difference is equal to or more that -25% of the average or variable less, whichever is more negative, return 'less', difference

Otherwise return 'same', difference



# Model
For low level information on the model pipeline, refer to src/darts_prophet/PIPELINE.md.

Note that the chosen model and its parameters can be simply edited using the config file. For more details on the config file, refer to the provided example


## get_incoming_weather

To get forecast weather data for upcoming dates, use "get_incoming_weather(API_KEY, BASE_URL, LOCATION, FORECAST_DAYS, TOTAL_DAYS)" in "incoming_weather.py".

"API_KEY": Your Visual Crossing API key, loaded from the environment.

"BASE_URL": The base URL for the Visual Crossing Timeline API.

"LOCATION": Coordinates or location string to fetch weather for (e.g. Schiphol coordinates).

"FORECAST_DAYS": Number of days to request from the API.

"TOTAL_DAYS": Total number of days the output DataFrame must cover. If the API returns fewer days than this, the remainder is padded with empty rows.

Returns a DataFrame with one row per date containing forecast weather fields.

The function does the following things:

- Construct the request URL from "LOCATION", today's date, and today + FORECAST_DAYS - 1.
- Fetch daily forecast data from the Visual Crossing API for the fields: datetime, temp, tempmin, tempmax, humidity, precip, precipprob, cloudcover, windspeed.
- Exit with a descriptive error message on HTTP 429 (quota exceeded), 401 (invalid key), or any other non-OK response.
- Convert "cloudcover" from Visual Crossing's 0–100% scale to oktas (0–8) using _pct_to_oktas, to match the scale used in training data. Values already ≤ 8 are left untouched.
- Pad the DataFrame with empty date rows if the API returned fewer days than "TOTAL_DAYS".
- Format all dates as YYYY-MM-DD strings.
- Return the forecast DataFrame ready to be passed to preprocess_incoming_data.


## build weather dataset

To build a historical weather dataset from KNMI for a date range, run "weather.py" from the project root with a config file

"start_date": Start of the date range as a YYYY-MM-DD string.

"end_date": End of the date range as a YYYY-MM-DD string.

The script requires the following to be set before calling build_dataset:

"DATASET", "VERSION": The KNMI Open Data dataset name and version to query.

"SCHIPHOL_STATION": The KNMI station ID for Schiphol (0-20000-0-06240). This can be changed to a different location if you for example want the data from utrecht

"HEADERS": Authorization header built from KNMI_API_KEY loaded via dotenv.

Returns a DataFrame with one row per date containing historical weather observations. The output is saved to the path specified by "weather_file" in the config.

The function does the following things:

- Iterate over each date in the range from "start_date" to "end_date" inclusive.
- For each date, fetch the corresponding .nc (NetCDF) file from the KNMI Open Data API via a two-step process: first retrieve a temporary download URL from the metadata endpoint, then download the file contents.
- Open the NetCDF file in memory using xarray and select the Schiphol station slice.
- Extract the fields: temp_avg (TG), temp_min (TN), temp_max (TX), rain_mm (RH), sun_hours (SQ), wind_avg (FG), cloud (NG), humidity (UG).
- Skip dates that fail (e.g. missing files) with a printed error and continue.
- Collect all rows into a DataFrame and save it as a CSV to the configured output path.