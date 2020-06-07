import os

import networkx as nx
import pandas as pd
import turf

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

G = nx.Graph()

G.add_nodes_from(us_cities.index)
nx.set_node_attributes(G, us_cities_dict)

for city, city_attr in us_cities_dict.items():
    edges = [
        (
            city,
            to_city,
            round(
                turf.distance(
                    [city_attr["lon"], city_attr["lat"]],
                    [to_city_attr["lon"], to_city_attr["lat"]],
                    {"units": "kilometers"},
                ),
                2,
            ),
        )
        for to_city, to_city_attr in us_cities_dict.items()
        if to_city != city
    ]

    edges = sorted(edges, key=lambda x: x[2])

    G.add_weighted_edges_from(edges)
