import pytest
from pathlib import Path
from scTenifold.data import fetch_data


@pytest.fixture(scope="session")
def morphine_datasets():
    morphine = fetch_data("morphine")
    return morphine


@pytest.fixture(scope="session")
def aging_datasets():
    aging = fetch_data("aging")
    return aging

