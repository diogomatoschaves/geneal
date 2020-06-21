import networkx as nx


def create_graph(
    cities_dict, distance_f, lon=lambda x: x["lon"], lat=lambda x: x["lat"]
):

    G = nx.Graph()

    G.add_nodes_from(cities_dict.keys())
    nx.set_node_attributes(G, cities_dict, "coords")

    for city, city_attr in cities_dict.items():
        edges = [
            (
                city,
                to_city,
                round(
                    distance_f(
                        [lon(city_attr), lat(city_attr)],
                        [lon(to_city_attr), lat(to_city_attr)],
                    ),
                    2,
                ),
            )
            for to_city, to_city_attr in cities_dict.items()
            if to_city != city
        ]

        edges = sorted(edges, key=lambda x: x[2])

        G.add_weighted_edges_from(edges)

    return G
