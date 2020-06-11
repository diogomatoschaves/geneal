import turf

from geneal.applications.tsp.examples.world_capitals import world_capitals_dict
from geneal.applications.tsp.helpers import create_graph

G = create_graph(
    world_capitals_dict,
    turf.distance,
    lon=lambda x: x["CapitalLongitude"],
    lat=lambda x: x["CapitalLatitude"],
)
