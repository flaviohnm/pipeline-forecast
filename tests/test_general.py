import warnings

import pandas as pd
import pytest

from src.utils.general import DATASETS_INFO

# Intervalo real entre observações de cada dataset de alta frequência
EXPECTED_STEP = {
    "ETTh1": pd.Timedelta("1h"),
    "ETTh2": pd.Timedelta("1h"),
    "ETTm1": pd.Timedelta("15min"),
    "ECL": pd.Timedelta("1h"),
    "Weather": pd.Timedelta("10min"),
}


@pytest.mark.parametrize("name", DATASETS_INFO)
def test_freq_is_valid_and_not_deprecated(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # alias obsoleto (ex.: "t") vira falha
        pd.date_range("2020-01-01", periods=2, freq=DATASETS_INFO[name]["freq"])


@pytest.mark.parametrize("name, step", EXPECTED_STEP.items())
def test_freq_matches_sampling_interval(name, step):
    dates = pd.date_range("2020-01-01", periods=2, freq=DATASETS_INFO[name]["freq"])
    assert dates[1] - dates[0] == step
