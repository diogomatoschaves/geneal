import numpy as np

from geneal.utils.helpers import create_graph


def dist(xy1, xy2, **kwargs):
    return round(np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2), 2)


cities = {
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
}

G = create_graph(
    cities, dist, lon=lambda x: x["coords"][0], lat=lambda x: x["coords"][1]
)
