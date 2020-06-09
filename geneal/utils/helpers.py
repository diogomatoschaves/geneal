import networkx as nx


def get_input_dimensions(lst, n_dim=0):
    if isinstance(lst, (list, tuple)):
        return get_input_dimensions(lst[0], n_dim + 1) if len(lst) > 0 else 0
    else:
        return n_dim


def get_elapsed_time(start_time, end_time):

    runtime = (end_time - start_time).seconds

    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_str = ""

    if hours:
        time_str += f"{hours} hours, "

    if minutes:
        time_str += f"{minutes} minutes, "

    if seconds:
        time_str += f"{seconds} seconds"

    return time_str


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
