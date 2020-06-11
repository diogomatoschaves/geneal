import os

import pandas as pd


try:
    module_path = os.path.abspath(os.path.join("."))
    world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))
except FileNotFoundError:
    module_path = os.path.abspath(os.path.join(".."))
    world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))


world_capitals.dropna(subset=["CapitalName"], axis=0, inplace=True)

world_capitals.reset_index(drop=True, inplace=True)
world_capitals.index += 1

world_capitals_dict = world_capitals.to_dict(orient="index")
