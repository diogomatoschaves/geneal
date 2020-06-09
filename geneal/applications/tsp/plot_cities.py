import plotly.graph_objects as go
import numpy as np


def add_trace(fig, G, city_1, city_2):

    city_1_lon = G.nodes[city_1]["lon"]
    city_1_lat = G.nodes[city_1]["lat"]
    city_2_lon = G.nodes[city_2]["lon"]
    city_2_lat = G.nodes[city_2]["lat"]

    if (
        city_1_lon < 0 < city_2_lon
        and np.abs(city_1_lon) > 90
        and np.abs(city_2_lon) > 90
    ):
        city_1_lon = city_1_lon + 360
    elif (
        city_1_lon > 0 > city_2_lon
        and np.abs(city_1_lon) > 90
        and np.abs(city_2_lon) > 90
    ):
        city_2_lon = city_2_lon + 360

    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            mode="lines",
            lon=[city_1_lon, city_2_lon],
            lat=[city_1_lat, city_2_lat],
            marker={"size": 10},
            line=dict(width=1, color="red"),
        )
    )

    return fig


def plot_cities(cities_graph, us_cities, ga_obj):

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=us_cities["lon"],
            lat=us_cities["lat"],
            hoverinfo="text",
            text=us_cities["Location"] + ", " + us_cities["State"],
            mode="markers",
            marker=dict(
                size=2,
                color="rgb(255, 0, 0)",
                line=dict(width=3, color="rgba(68, 68, 68, 0)"),
            ),
        )
    )

    fig.update_layout(
        title_text="Feb. 2011 American Airline flight paths<br>(Hover for airport names)",
        showlegend=False,
        geo=dict(
            scope="north america",
            projection_type="azimuthal equal area",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
        ),
    )

    for city_1, city_2 in zip(ga_obj.best_individual_, ga_obj.best_individual_[1:]):
        fig = add_trace(fig, cities_graph, city_1, city_2)

    fig = add_trace(
        fig, cities_graph, ga_obj.best_individual_[0], ga_obj.best_individual_[-1]
    )

    fig.show()
