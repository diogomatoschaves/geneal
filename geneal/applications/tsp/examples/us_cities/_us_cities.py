import os

import pandas as pd


try:
    module_path = os.path.abspath(os.path.join("."))
    us_cities = pd.read_csv(os.path.join(module_path, "data/us_cities.csv"))
except FileNotFoundError:
    module_path = os.path.abspath(os.path.join(".."))
    us_cities = pd.read_csv(os.path.join(module_path, "data/us_cities.csv"))

us_cities.rename(columns={"LAT": "lat", "LON": "lon"}, inplace=True)

us_cities = us_cities[(us_cities["State"] != "AK") & (us_cities["State"] != "HI")]

us_cities["lon"] = -us_cities["lon"]

us_cities.reset_index(drop=True, inplace=True)
us_cities.index += 1

us_cities_dict = us_cities.to_dict(orient="index")
