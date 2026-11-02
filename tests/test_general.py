import warnings

import pandas as pd
import pytest

from src.utils.general import DATASETS_INFO

# Intervalo real entre observações de cada dataset de alta frequência
EXPECTED_STEP = {
    "ETTh1": pd.Timedelta("1h"),
    "ETTh2": pd.Timedelta("1h"),
    "ETTm1": pd.Timedelta("15min"),
    "ETTm2": pd.Timedelta("15min"),
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


# Comprimento mínimo de cada arquivo para caber treino + validação + teste do protocolo rolling
ROLLING_MIN_ROWS = {"ETTh1": 14400, "ETTh2": 14400, "ETTm1": 57600, "ETTm2": 57600, "ECL": 26304, "Weather": 52696}


@pytest.mark.parametrize("name, n_rows", ROLLING_MIN_ROWS.items())
def test_rolling_config_fits_dataset_and_horizons(name, n_rows):
    from src.utils.general import get_rolling_origins

    info = DATASETS_INFO[name]
    origins = get_rolling_origins(info, n_rows=n_rows)
    assert origins and origins[-1] + info["max_horizon"] <= n_rows
    # a validação precisa comportar ao menos uma janela de max_horizon (escolha de alpha)
    assert info["rolling"]["val_size"] >= info["max_horizon"]
