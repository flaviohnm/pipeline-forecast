import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from neuralforecast.models import NHITS

from src.models.arima_baseline import arima_rolling
from src.models.deep_baselines import fit_predict_rolling
from src.models.hybrid_combiner import combine_rolling
from src.models.hybrid_prophet_nhits import hybrid_prophet_rolling
from src.models.nhits_residual import residuals_on_full_dates
from src.models.prophet_baseline import prophet_rolling
from src.utils.general import get_rolling_origins
from src.visualization.plot_results import ResultsPlotter

warnings.filterwarnings("ignore")
for _name in ("pytorch_lightning", "lightning", "cmdstanpy", "prophet"):
    logging.getLogger(_name).setLevel(logging.ERROR)

H = 48
INFO = {
    "freq": "h",
    "max_horizon": H,
    "seasonal_period": 24,
    "train_size": 240,
    "rolling": {"val_size": 48, "test_size": 144, "step_size": 48},
}
QUIET = {
    "enable_progress_bar": False,
    "logger": False,
    "enable_model_summary": False,
    "enable_checkpointing": False,
    "num_workers_loader": 0,
    "accelerator": "cpu",
}


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 24 * 20
    t = np.arange(n)
    y = 20 + 0.02 * t + 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"unique_id": "SYN", "ds": pd.date_range("2021-01-01", periods=n, freq="h"), "y": y})


@pytest.fixture
def origins(df):
    return get_rolling_origins(INFO, len(df))


def _tamper(df, start, delta=30.0):
    out = df.copy()
    out.loc[start:, "y"] += delta
    return out


# --- ARIMA: previsões + resíduos com parâmetros fixos -------------------------


def test_arima_rolling_residuals_cover_until_last_cutoff(df, origins):
    forecast, residuals = arima_rolling(df, INFO, n_jobs=1)
    assert forecast["cutoff"].nunique() == len(origins)
    assert list(forecast.columns) == ["unique_id", "cutoff", "ds", "y", "ARIMA"]
    assert residuals["ds"].min() == df["ds"].iloc[0]
    assert residuals["ds"].max() == df["ds"].iloc[origins[-1] - 1]  # termina no último cutoff
    assert not residuals["y"].isna().any()


def test_arima_rolling_residuals_are_causal(df, origins):
    """Alterar a série a partir da origem k não muda os resíduos anteriores a ela."""
    _, base = arima_rolling(df, INFO, n_jobs=1)
    _, out = arima_rolling(_tamper(df, origins[1]), INFO, n_jobs=1)
    before = base["ds"] < df["ds"].iloc[origins[1]]
    np.testing.assert_allclose(out.loc[before, "y"], base.loc[before, "y"])
    assert not np.allclose(out.loc[~before, "y"], base.loc[~before, "y"])


# --- N-HiTS nos resíduos ------------------------------------------------------


def test_residuals_on_full_dates_leaves_tail_empty(df, origins):
    _, residuals = arima_rolling(df, INFO, n_jobs=1)
    full = residuals_on_full_dates(residuals, df)
    assert len(full) == len(df)
    assert full["y"].iloc[: origins[-1]].notna().all()
    assert full["y"].iloc[origins[-1] :].isna().all()


def test_nhits_on_residuals_rolling_runs_with_empty_tail(df, origins):
    _, residuals = arima_rolling(df, INFO, n_jobs=1)
    full = residuals_on_full_dates(residuals, df)
    model = NHITS(h=H, input_size=96, max_steps=5, random_seed=1, **QUIET)
    out = fit_predict_rolling("NHITS", model, full, INFO, input_size=96)
    assert out["cutoff"].nunique() == len(origins)
    assert not out["NHITS"].isna().any()


def test_fit_predict_rolling_rejects_missing_values_in_training(df):
    broken = df.copy()
    broken.loc[10, "y"] = np.nan
    model = NHITS(h=H, input_size=96, max_steps=5, random_seed=1, **QUIET)
    with pytest.raises(ValueError, match="treino\\+val"):
        fit_predict_rolling("NHITS", model, broken, INFO, input_size=96)


# --- Combinação ARIMA + N-HiTS ------------------------------------------------


def _long(cutoffs, h, col, value):
    frames = [
        pd.DataFrame(
            {"unique_id": "S", "cutoff": c, "ds": c + pd.to_timedelta(np.arange(1, h + 1), unit="h"), col: value}
        )
        for c in cutoffs
    ]
    return pd.concat(frames, ignore_index=True)


def test_combine_rolling_sums_components():
    cutoffs = list(pd.date_range("2021-01-01", periods=3, freq="D"))
    out = combine_rolling(_long(cutoffs, 4, "ARIMA", 10.0), _long(cutoffs, 4, "NHITS", -1.5))
    assert len(out) == 12
    assert (out["Hybrid_ARIMA_NHITS"] == 8.5).all()


def test_combine_rolling_rejects_misaligned_windows():
    cutoffs = list(pd.date_range("2021-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError, match="desalinhadas"):
        combine_rolling(_long(cutoffs, 4, "ARIMA", 1.0), _long(cutoffs[:2], 4, "NHITS", 1.0))


# --- Prophet ------------------------------------------------------------------


def test_prophet_rolling_alignment_and_no_leakage(df, origins):
    base = prophet_rolling(df, INFO)  # check_rolling_cutoffs (com h) roda dentro
    assert len(base) == len(origins) * H
    out = prophet_rolling(_tamper(df, origins[1]), INFO)
    cutoffs = sorted(base["cutoff"].unique())
    before = base["cutoff"].isin(cutoffs[:2])  # janelas 0 e 1 só veem dados < origins[1]
    np.testing.assert_allclose(out.loc[before, "Prophet"], base.loc[before, "Prophet"], rtol=1e-4)
    last = base["cutoff"] == cutoffs[2]
    assert not np.allclose(out.loc[last, "Prophet"], base.loc[last, "Prophet"])


# --- Híbrido Prophet + N-HiTS -------------------------------------------------


def test_hybrid_prophet_rolling(df, origins):
    model = NHITS(h=H, input_size=96, max_steps=5, scaler_type="standard", random_seed=1, **QUIET)
    out = hybrid_prophet_rolling(df, INFO, model, input_size=96)
    assert list(out.columns) == [
        "unique_id", "cutoff", "ds", "Prophet_component", "NHITS_component", "Hybrid_Prophet_NHITS"
    ]
    assert out["cutoff"].nunique() == len(origins)
    np.testing.assert_allclose(out["Hybrid_Prophet_NHITS"], out["Prophet_component"] + out["NHITS_component"])
    # O componente Prophet é o mesmo procedimento do baseline (reajuste em cada origem)
    baseline = prophet_rolling(df, INFO)
    np.testing.assert_allclose(out["Prophet_component"], baseline["Prophet"], rtol=1e-4)


# --- Gráficos do protocolo rolling --------------------------------------------


def test_rolling_plots_are_written(tmp_path):
    cutoffs = list(pd.date_range("2021-01-05", periods=3, freq="D"))
    hybrid = _long(cutoffs, 48, "ARIMA", 10.0)
    hybrid["NHITS"] = 0.5
    hybrid["Hybrid_ARIMA_NHITS"] = 10.5
    raw = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=24 * 12, freq="h")})
    raw["OT"] = 10 + np.sin(np.arange(len(raw)) / 4)
    summary = pd.DataFrame(
        {
            "Horizonte": [24, 24, 48, 48],
            "Modelo": ["ARIMA", "Ingênuo Sazonal"] * 2,
            "MSE_mean": [0.1, 0.2, 0.3, 0.4],
            "MSE_std": [0.01, 0.02, 0.03, 0.04],
            "MAE_mean": [0.1, 0.2, 0.3, 0.4],
            "MAE_std": [0.01, 0.02, 0.03, 0.04],
            "n_janelas": 3,
            "Normalizacao": "train",
        }
    )
    for sub in ("metrics", "plots", "hybrid", "raw"):
        (tmp_path / sub).mkdir()
    summary.to_csv(tmp_path / "metrics" / "SYN_evaluation_rolling.csv", index=False)
    hybrid.to_csv(tmp_path / "hybrid" / "SYN_hybrid_rolling.csv", index=False)
    raw.to_csv(tmp_path / "raw" / "SYN.csv", index=False)

    plotter = ResultsPlotter.__new__(ResultsPlotter)
    plotter.metrics_dir, plotter.plots_dir = tmp_path / "metrics", tmp_path / "plots"
    plotter.forecast_hybrid, plotter.raw_dir = tmp_path / "hybrid", tmp_path / "raw"
    plotter.plot_mse_benchmark_rolling("SYN")
    plotter.plot_time_series_forecast_rolling("SYN", window=1, zoom_horizon=24)

    assert (tmp_path / "plots" / "SYN_mse_benchmark_rolling.png").stat().st_size > 0
    assert (tmp_path / "plots" / "SYN_forecast_lines_rolling.png").stat().st_size > 0
