import networkx as nx
import numpy as np

from geneal.applications.tsp.helpers import create_graph


def dist(xy1, xy2):
    return round(np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2), 2)


cities_dict = {
    1: {"coords": [392.8, 356.4], "name": "Oklahoma City"},
    2: {"coords": [559.6, 404.8], "name": "Montgomery"},
    3: {"coords": [451.6, 186.0], "name": "Saint Paul"},
    4: {"coords": [698.8, 239.6], "name": "Trenton"},
    5: {"coords": [204.0, 243.2], "name": "Salt Lake City"},
    6: {"coords": [590.8, 263.2], "name": "Columbus"},
    7: {"coords": [389.2, 448.4], "name": "Austin"},
    8: {"coords": [179.6, 371.2], "name": "Phoenix"},
    9: {"coords": [719.6, 205.2], "name": "Hartford"},
    10: {"coords": [489.6, 442.0], "name": "Baton Rouge"},
    11: {"coords": [80.0, 139.2], "name": "Salem"},
    12: {"coords": [469.2, 367.2], "name": "Little Rock"},
    13: {"coords": [673.2, 293.6], "name": "Richmond"},
    14: {"coords": [501.6, 409.6], "name": "Jackson"},
    15: {"coords": [447.6, 246.0], "name": "Des Moines"},
    16: {"coords": [563.6, 216.4], "name": "Lansing"},
    17: {"coords": [293.6, 274.0], "name": "Denver"},
    18: {"coords": [159.6, 182.8], "name": "Boise"},
    19: {"coords": [662.0, 328.8], "name": "Raleigh"},
    20: {"coords": [585.6, 376.8], "name": "Atlanta"},
    21: {"coords": [500.8, 217.6], "name": "Madison"},
    22: {"coords": [548.0, 272.8], "name": "Indianapolis"},
    23: {"coords": [546.4, 336.8], "name": "Nashville"},
    24: {"coords": [632.4, 364.8], "name": "Columbia"},
    25: {"coords": [735.2, 201.2], "name": "Providence"},
    26: {"coords": [738.4, 190.8], "name": "Boston"},
    27: {"coords": [594.8, 434.8], "name": "Tallahassee"},
    28: {"coords": [68.4, 254.0], "name": "Sacramento"},
    29: {"coords": [702.0, 193.6], "name": "Albany"},
    30: {"coords": [670.8, 244.0], "name": "Harrisburg"},
}


G = create_graph(
    cities_dict, dist, lon=lambda x: x["coords"][0], lat=lambda x: x["coords"][1]
)
