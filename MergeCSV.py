import pandas as pd
 
df1 = pd.read_csv("weather_amsterdam1.csv")
df2 = pd.read_csv("weather_amsterdam.csv")
 
merged = pd.concat([df1, df2], ignore_index=True)
merged.to_csv("WeatherAmsterdam.csv", index=False)
 
print(f"Done! {len(merged):,} rows written to 'weather_amsterdam_merged.csv'")
