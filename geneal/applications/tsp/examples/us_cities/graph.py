import turf

from geneal.applications.tsp.examples.us_cities._us_cities import us_cities_dict
from geneal.applications.tsp.helpers import create_graph

G = create_graph(us_cities_dict, turf.distance)
