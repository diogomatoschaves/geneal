import logging

import pytest
import matplotlib.pyplot as plt


@pytest.fixture
def mock_matplotlib(mocker):
    mocker.patch.object(plt, "show", lambda: None)


@pytest.fixture
def mock_logging(mocker):
    mocker.patch.object(logging, "info", lambda x: None)