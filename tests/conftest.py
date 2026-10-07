"""
conftest.py — Shared pytest fixtures.
"""
import pandas as pd
import pytest

from engine.config import Config


@pytest.fixture
def sample_df():
    """The provided sample healthcare dataset."""
    return pd.read_csv("data/sample.csv")


@pytest.fixture
def default_cfg():
    """Default Config with all defaults."""
    return Config()


@pytest.fixture
def synthetic_df():
    """
    A completely different dataset with different entities and indicators.
    Used to prove there is no hardcoded entity/indicator logic.
    """
    data = {
        "period": ["2024-01", "2024-01", "2024-01",
                   "2024-02", "2024-02", "2024-02",
                   "2024-03", "2024-03", "2024-03"],
        "zone": ["Pune", "Nagpur", "Nashik",
                 "Pune", "Nagpur", "Nashik",
                 "Pune", "Nagpur", "Nashik"],
        "tb_cure_rate": [82, 75, 78,
                         70, 74, 80,
                         68, 73, 79],
        "malaria_cases": [120, 200, 150,
                          110, 210, 145,
                          140, 180, 155],
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_cfg():
    """Config configured for the synthetic dataset."""
    return Config(
        entity_col="zone",
        period_col="period",
        polarity={"tb_cure_rate": 1, "malaria_cases": -1},
    )
