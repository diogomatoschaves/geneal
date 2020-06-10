import pickle

import pytest

from geneal.applications.tsp.travelling_salesman_problem import (
    TravellingSalesmanProblemSolver,
)
from geneal.applications.tsp.helpers.plot_cities import plot_cities
from geneal.applications.tsp.examples.us_cities import us_cities_dict
from geneal.applications.tsp.examples.cities import cities_dict
from geneal.applications.tsp.examples.world_capitals import world_capitals_dict
from tests.mock_fixtures.mock_fixtures import (
    mock_matplotlib,
    mock_logging,
    mock_random_sample,
    mock_plotly_figure_show,
)
from tests.applications.fixtures.tsp_test_fixture import G


class TestHelpersAndExamples:
    @pytest.mark.parametrize(
        "cities, graph_file, lat, lon, name",
        [
            pytest.param(
                cities_dict,
                'cities',
                lambda x: x["coords"][0],
                lambda x: x["coords"][1],
                lambda x: x["name"],
                id='cities'
            ),
            pytest.param(
                us_cities_dict,
                'us_cities',
                lambda x: x["lon"],
                lambda x: x["lat"],
                lambda x: x["Location"] + ", " + x["State"],
                id='us_cities'
            ),
            pytest.param(
                world_capitals_dict,
                'world_capitals',
                lambda x: x["CapitalLongitude"],
                lambda x: x["CapitalLatitude"],
                lambda x: x["CapitalName"],
                id='world_capitals'
            ),
        ],
    )
    def test_plot_cities(
        self,
        mocker,
        mock_logging,
        mock_matplotlib,
        mock_plotly_figure_show,
        cities,
        graph_file,
        lat,
        lon,
        name,
    ):

        mocked_add_trace = mocker.patch(
            "geneal.applications.tsp.helpers.plot_cities.add_trace"
        )

        with open(f'tests/applications/fixtures/{graph_file}.pickle', 'rb') as f:
            G = pickle.load(f)

        tsp_solver = TravellingSalesmanProblemSolver(G, max_gen=2, random_state=42,)

        tsp_solver.solve()

        plot_cities(
            cities,
            tsp_solver,
            lon=lon,
            lat=lat,
            name=name,
        )

        mocked_add_trace.assert_called()

        assert mocked_add_trace.call_count == len(G.nodes)
