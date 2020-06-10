import plotly.graph_objects as go
import numpy as np


def add_trace(
    fig, cities_dict, city_1, city_2, lon=lambda x: x["lon"], lat=lambda x: x["lat"]
):

    city_1_lon = lon(cities_dict[city_1])
    city_1_lat = lat(cities_dict[city_1])
    city_2_lon = lon(cities_dict[city_2])
    city_2_lat = lat(cities_dict[city_2])

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


def plot_cities(
    cities_dict,
    ga_obj,
    lon=lambda x: x["lon"],
    lat=lambda x: x["lat"],
    name=lambda x: x["Location"] + ", " + x["State"],
    scope=None,
    projection_type="mercator",
):

    lons = [lon(city) for city in cities_dict.values()]
    lats = [lat(city) for city in cities_dict.values()]
    names = [name(city) for city in cities_dict.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lon=lons,
            lat=lats,
            hoverinfo="text",
            text=names,
            mode="markers",
            marker=dict(
                size=2,
                color="rgb(255, 0, 0)",
                line=dict(width=3, color="rgba(68, 68, 68, 0)"),
            ),
        )
    )

    geo = dict(
        projection_type=projection_type,
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
    )

    if scope:
        geo.update({"scope": scope})

    fig.update_layout(
        title_text="Best Route", showlegend=False, geo=geo,
    )

    for city_1, city_2 in zip(ga_obj.best_individual_, ga_obj.best_individual_[1:]):
        add_trace(fig, cities_dict, city_1, city_2, lon, lat)

    add_trace(
        fig,
        cities_dict,
        ga_obj.best_individual_[0],
        ga_obj.best_individual_[-1],
        lon,
        lat,
    )

    fig.show()
