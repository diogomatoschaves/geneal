import os

import networkx as nx
import pandas as pd
import turf

try:
    module_path = os.path.abspath(os.path.join(".."))
    world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))
except FileNotFoundError:
    module_path = os.path.abspath(os.path.join("."))
    world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))


world_capitals.dropna(subset=["CapitalName"], axis=0, inplace=True)

world_capitals.reset_index(drop=True, inplace=True)
world_capitals.index += 1

world_capitals_dict = world_capitals.to_dict(orient="index")

G = nx.Graph()

G.add_nodes_from(world_capitals.index)
nx.set_node_attributes(G, world_capitals_dict)

for city, city_attr in world_capitals_dict.items():
    edges = [
        (
            city,
            to_city,
            round(
                turf.distance(
                    [city_attr["CapitalLongitude"], city_attr["CapitalLatitude"]],
                    [to_city_attr["CapitalLongitude"], to_city_attr["CapitalLatitude"]],
                    {"units": "kilometers"},
                ),
                2,
            ),
        )
        for to_city, to_city_attr in world_capitals_dict.items()
        if to_city != city
    ]

    edges = sorted(edges, key=lambda x: x[2])

    G.add_weighted_edges_from(edges)
