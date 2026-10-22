import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from src.metrics.evaluate_metrics import naive_forecasts
from src.utils.general import check_rolling_cutoffs, get_protocol, get_rolling_origins, rolling_input

INFO = {
    "freq": "h",
    "max_horizon": 48,
    "seasonal_period": 24,
    "train_size": 240,
    "rolling": {"val_size": 48, "test_size": 168, "step_size": 24},
}


@pytest.fixture
def df():
    """Série horária com ciclo diário, tendência e ruído; sobra cauda após o bloco de teste."""
    rng = np.random.default_rng(1)
    n = 24 * 25
    t = np.arange(n)
    y = 20 + 0.01 * t + 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"unique_id": "SYN", "ds": pd.date_range("2021-01-01", periods=n, freq="h"), "y": y})


def _cv(df, model, refit=False):
    df_cv, origins, kwargs = rolling_input(df, INFO)
    sf = StatsForecast(models=[model], freq=INFO["freq"], n_jobs=1)
    return sf.cross_validation(df=df_cv, refit=refit, **kwargs).reset_index(), origins


def test_rolling_input_cuts_at_end_of_last_window(df):
    df_cut, origins, kwargs = rolling_input(df, INFO)
    assert origins == get_rolling_origins(INFO, len(df))
    assert len(df_cut) == origins[-1] + INFO["max_horizon"]
    assert kwargs == {"h": 48, "n_windows": len(origins), "step_size": 24}


def test_statsforecast_cutoffs_match_protocol(df):
    cv, origins = _cv(df, SeasonalNaive(season_length=24), refit=True)  # SeasonalNaive não tem forward
    check_rolling_cutoffs(cv, df, origins)  # não levanta
    assert len(cv) == len(origins) * INFO["max_horizon"]
    # A 1ª previsão de cada janela cai exatamente na origem
    first_ds = cv.groupby("cutoff")["ds"].min().sort_values().tolist()
    assert first_ds == df["ds"].iloc[origins].tolist()


def test_statsforecast_seasonal_naive_equals_ours(df):
    """Mesma previsão do SeasonalNaive do Nixtla e do nosso baseline: janelas e origem alinhadas, sem vazamento."""
    cv, origins = _cv(df, SeasonalNaive(season_length=24), refit=True)  # SeasonalNaive não tem forward
    df_raw = df.rename(columns={"y": "y_real"})[["ds", "y_real"]]
    for start in origins:
        ours = naive_forecasts(df_raw, df["ds"].iloc[start], INFO["max_horizon"], 24)
        theirs = cv[cv["cutoff"] == df["ds"].iloc[start - 1]].sort_values("ds")
        np.testing.assert_allclose(theirs["SeasonalNaive"].to_numpy(), ours["SeasonalNaive"].to_numpy())


def test_statsforecast_ets_refit_false_uses_new_data(df):
    """Com refit=False os parâmetros ficam fixos, mas cada janela parte do nível observado até o seu cutoff."""
    cv, origins = _cv(df, AutoETS(season_length=24, model="ZZA"))
    check_rolling_cutoffs(cv, df, origins)
    first_step = cv.sort_values("ds").groupby("cutoff").first()
    assert first_step["AutoETS"].nunique() == len(origins)  # previsões diferentes por janela
    assert not cv["AutoETS"].isna().any()


def test_check_rolling_cutoffs_detects_misalignment(df):
    _, origins, kwargs = rolling_input(df, INFO)
    sf = StatsForecast(models=[SeasonalNaive(season_length=24)], freq="h", n_jobs=1)
    cv_wrong = sf.cross_validation(df=df, refit=True, **kwargs).reset_index()  # sem cortar a cauda
    with pytest.raises(ValueError, match="Cutoffs gerados"):
        check_rolling_cutoffs(cv_wrong, df, origins)


@pytest.mark.parametrize(
    "config, expected",
    [({}, "single_split"), ({"evaluation": {"protocol": "rolling"}}, "rolling")],
)
def test_get_protocol(config, expected):
    assert get_protocol(config) == expected


def test_get_protocol_invalid():
    with pytest.raises(ValueError, match="protocol"):
        get_protocol({"evaluation": {"protocol": "kfold"}})
