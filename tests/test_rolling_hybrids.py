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
from src.models.nhits_residual import residual_check, residuals_on_full_dates
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


def _long_h(cutoffs, h_list, col, value):
    frames = []
    for h in h_list:
        f = _long(cutoffs, h, col, value)
        f.insert(1, "H", h)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_combine_rolling_per_horizon_uses_first_h_steps_of_arima():
    cutoffs = list(pd.date_range("2021-01-01", periods=3, freq="D"))
    arima = _long(cutoffs, 8, "ARIMA", 10.0)  # ARIMA cobre o maior H
    nhits = _long_h(cutoffs, [4, 8], "NHITS", -1.0)
    out = combine_rolling(arima, nhits)
    assert list(out.columns) == ["unique_id", "H", "cutoff", "ds", "ARIMA", "NHITS", "Hybrid_ARIMA_NHITS"]
    assert len(out) == 3 * (4 + 8)
    assert (out["Hybrid_ARIMA_NHITS"] == 9.0).all()
    assert out.groupby(["H", "cutoff"]).size().eq(out.groupby(["H", "cutoff"])["H"].first()).all()


def test_combine_rolling_per_horizon_rejects_missing_arima_windows():
    cutoffs = list(pd.date_range("2021-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError, match="desalinhadas"):
        combine_rolling(_long(cutoffs[:2], 8, "ARIMA", 1.0), _long_h(cutoffs, [4], "NHITS", 1.0))


# --- Checagem do N-HiTS nos resíduos ------------------------------------------


def test_residual_check_reports_skill_on_validation_block(df, origins):
    _, residuals = arima_rolling(df, INFO, n_jobs=1)
    full = residuals_on_full_dates(residuals, df)
    h = 24
    model = NHITS(h=h, input_size=96, max_steps=5, random_seed=1, **QUIET)
    captured = []
    fit_predict_rolling(
        "NHITS", model, full, INFO, input_size=96, horizon=h,
        on_fitted=lambda nf: captured.append(residual_check(nf, full, INFO, 96, h)),
    )
    row = captured[0]
    assert row["H"] == h and row["janelas_validacao"] >= 1
    assert np.isfinite(row["skill"]) and row["skill"] <= 1.0
    assert row["mse_zero"] > 0


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
    out = hybrid_prophet_rolling(df, INFO, model, input_size=96)  # modelo único: formato legado, sem coluna H
    assert list(out.columns) == [
        "unique_id", "cutoff", "ds", "Prophet_component", "NHITS_component", "Hybrid_Prophet_NHITS"
    ]
    assert out["cutoff"].nunique() == len(origins)
    np.testing.assert_allclose(out["Hybrid_Prophet_NHITS"], out["Prophet_component"] + out["NHITS_component"])
    # O componente Prophet é o mesmo procedimento do baseline (reajuste em cada origem)
    baseline = prophet_rolling(df, INFO)
    np.testing.assert_allclose(out["Prophet_component"], baseline["Prophet"], rtol=1e-4)


def test_hybrid_prophet_rolling_per_horizon(df, origins):
    models = {
        h: NHITS(h=h, input_size=96, max_steps=5, scaler_type="standard", random_seed=1, **QUIET) for h in (24, H)
    }
    out = hybrid_prophet_rolling(df, INFO, models, input_size=96)
    assert list(out.columns)[:2] == ["unique_id", "H"]
    for h in (24, H):
        sub = out[out["H"] == h]
        assert len(sub) == len(origins) * h
    np.testing.assert_allclose(out["Hybrid_Prophet_NHITS"], out["Prophet_component"] + out["NHITS_component"])


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


# --- Variantes do híbrido: previsões na validação -----------------------------
def test_arima_validation_covers_validation_block_without_peeking(df):
    from src.models.arima_baseline import arima_validation, validation_origins

    info = {**INFO, "rolling": {"val_size": 96, "test_size": 144, "step_size": 24}}
    vo = validation_origins(info, len(df))
    assert vo == [240, 264, 288]
    val = arima_validation(df, info, n_jobs=1)
    assert val["cutoff"].nunique() == 3 and (val.groupby("cutoff").size() == H).all()
    first = pd.Timestamp(df["ds"].iloc[239])
    assert first in set(val["cutoff"])
    # dados após o fim da validação (e, no refit=False, após cada cutoff) não alteram as previsões
    changed = arima_validation(_tamper(df, 336), info, n_jobs=1)
    pd.testing.assert_frame_equal(val.drop(columns="y"), changed.drop(columns="y"))


def test_arima_validation_empty_when_validation_shorter_than_horizon(df):
    from src.models.arima_baseline import arima_validation

    info = {**INFO, "rolling": {"val_size": 24, "test_size": 144, "step_size": 24}}
    assert arima_validation(df, info, n_jobs=1).empty


def test_nhits_validation_predictions_and_alpha_end_to_end(df, origins):
    from src.models.hybrid_variants import alpha_by_horizon
    from src.models.nhits_residual import nhits_validation_predictions

    info = {**INFO, "rolling": {"val_size": 96, "test_size": 144, "step_size": 24}}
    arima_val = __import__("src.models.arima_baseline", fromlist=["x"]).arima_validation(df, info, n_jobs=1)
    rng = np.random.default_rng(0)
    res = pd.DataFrame({"unique_id": "SYN", "ds": df["ds"], "y": rng.normal(0, 1, len(df))})
    res.loc[get_rolling_origins(info, len(df))[-1] :, "y"] = np.nan
    vo = [240, 264, 288]
    held = {}

    def after(nf):
        held["val"] = nhits_validation_predictions(nf, res, 48, 24, vo)

    model = NHITS(h=24, input_size=48, max_steps=3, val_check_steps=3, **QUIET)
    fit_predict_rolling("NHITS", model, res, info, 48, horizon=24, on_fitted=after)
    val = held["val"]
    assert val["H"].eq(24).all() and val["cutoff"].nunique() == 3 and (val.groupby("cutoff").size() == 24).all()
    table = alpha_by_horizon(arima_val, val, step=0.5)
    assert table.loc[0, "H"] == 24 and 0.0 <= table.loc[0, "alpha"] <= 1.0
    assert table.loc[0, "mse_val"] <= min(table.loc[0, "mse_val_arima"], table.loc[0, "mse_val_aditivo"]) + 1e-12


def test_combiner_writes_alpha_blocks_and_table(tmp_path):
    from src.models.hybrid_combiner import HybridCombiner, combine_rolling
    from src.models.hybrid_variants import ALPHA_COL, BLOCKS_ALPHA_COL, load_hybrid_config

    cuts = pd.to_datetime(["2020-01-01", "2020-01-02"])
    ar = pd.DataFrame(
        [{"unique_id": "s", "cutoff": c, "ds": c + pd.Timedelta(hours=i), "ARIMA": 10.0, "y": 10.5}
         for c in cuts for i in range(1, 5)]
    )
    nh = pd.DataFrame(
        [{"H": h, "cutoff": c, "ds": c + pd.Timedelta(hours=i), "NHITS": 1.0}
         for h in (2, 4) for c in cuts for i in range(1, h + 1)]
    )
    comb = HybridCombiner.__new__(HybridCombiner)
    comb.hybrid_cfg = load_hybrid_config({})
    comb.hybrid_dir, comb.blocks_dir, comb.metrics_dir = tmp_path / "h", tmp_path / "b", tmp_path / "m"
    comb.hybrid_dir.mkdir()
    comb.arima_dir = comb.resid_dir = tmp_path / "v"
    comb.arima_dir.mkdir()
    ar.to_csv(comb.arima_dir / "S_arima_val_rolling.csv", index=False)
    nh.to_csv(comb.resid_dir / "S_nhits_residuals_val_rolling.csv", index=False)

    final = combine_rolling(ar[["unique_id", "cutoff", "ds", "ARIMA"]], nh)
    comb.add_variants("S", final, nh, ar)
    out = pd.read_csv(comb.hybrid_dir / "S_hybrid_rolling.csv")
    assert np.allclose(out[ALPHA_COL], 10.5)  # alpha 0.5 com erro 0.5 e correção 1
    blocks = pd.read_csv(comb.blocks_dir / "S_hybrid_blocks_rolling.csv")
    assert "H" not in blocks.columns and np.allclose(blocks[BLOCKS_ALPHA_COL], 10.5)
    table = pd.read_csv(comb.metrics_dir / "S_hybrid_alpha.csv")
    assert set(table["variante"]) == {"alpha_H", "alpha_bloco"}
