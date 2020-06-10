import logging
import random
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go


@pytest.fixture
def mock_matplotlib(mocker):
    mocker.patch.object(plt, "show", lambda: None)


@pytest.fixture
def mock_logging(mocker):
    mocker.patch.object(logging, "info", lambda x: None)


@pytest.fixture
def mock_random_sample(mocker):
    mocker.patch.object(random, "sample", lambda x, y: ["random_inversion"])


@pytest.fixture
def mock_plotly_figure_show(mocker):
    mocker.patch.object(go.Figure, "show", lambda self: None)
