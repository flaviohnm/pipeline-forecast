import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from neuralforecast.models import NHITS

from src.models.arima_baseline import arima_rolling, arima_validation, validation_origins
from src.models.hybrid_exog_backtest import (
    EXOG_ALPHA_COL,
    EXOG_COL,
    EXOG_NAME,
    arima_backtest,
    backtest_origins,
    hybrid_exog_rolling,
    inference_frames,
    load_exog_config,
    positions_for_horizon,
    training_frame,
)
from src.utils.general import get_rolling_origins

warnings.filterwarnings("ignore")
for _name in ("pytorch_lightning", "lightning"):
    logging.getLogger(_name).setLevel(logging.ERROR)

H = 24
INPUT = 48
INFO = {
    "freq": "h",
    "max_horizon": H,
    "seasonal_period": 24,
    "train_size": 384,
    "rolling": {"val_size": 96, "test_size": 96, "step_size": 24},
}
QUIET = {
    "enable_progress_bar": False, "logger": False, "enable_model_summary": False,
    "enable_checkpointing": False, "num_workers_loader": 0, "accelerator": "cpu",
}


@pytest.fixture
def df():
    rng = np.random.default_rng(5)
    n = 24 * 25
    t = np.arange(n)
    y = 20 + 0.02 * t + 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"unique_id": "SYN", "ds": pd.date_range("2021-01-01", periods=n, freq="h"), "y": y})


@pytest.fixture
def arima_files(df):
    fc, _ = arima_rolling(df, INFO, n_jobs=1)
    val = arima_validation(df, INFO, n_jobs=1)
    return fc, val


def test_config_defaults_e_validacao():
    assert load_exog_config({})["fit_frac"] == 0.5
    with pytest.raises(ValueError):
        load_exog_config({"hybrid_exog": {"fit_frac": 1.0}})
    with pytest.raises(ValueError):
        load_exog_config({"hybrid_exog": {"stride": 0}})


def test_origens_do_backtest_nao_tocam_o_teste():
    origins = backtest_origins(INFO, 0.5, 24)
    assert origins[0] == 192
    assert origins[-1] + H <= INFO["train_size"] + INFO["rolling"]["val_size"]


def test_backtest_fora_da_amostra_e_sem_teste(df):
    bt = arima_backtest(df, INFO, 0.5, 24, n_jobs=1)
    test_start = df["ds"].iloc[INFO["train_size"] + INFO["rolling"]["val_size"]]
    assert bt["ds"].max() < test_start
    assert bt["cutoff"].nunique() == len(backtest_origins(INFO, 0.5, 24))
    assert (bt.groupby("cutoff").size() == H).all()


def test_training_frame_forma_alvo_e_exogena(df):
    bt = arima_backtest(df, INFO, 0.5, 24, n_jobs=1)
    pos = [int(np.flatnonzero(df["ds"].to_numpy() == c)[0]) + 1 for c in sorted(bt["cutoff"].unique())]
    tf = training_frame(df, bt, pos, INPUT, H)
    one = tf[tf["unique_id"] == f"o{pos[0]}"].reset_index(drop=True)
    assert len(one) == INPUT + H
    # contexto: alvo 0 e exógena = série original; futuro: exógena = previsão do ARIMA, alvo = erro de h passos
    assert (one["y"].iloc[:INPUT] == 0).all()
    np.testing.assert_allclose(one[EXOG_NAME].iloc[:INPUT], df["y"].iloc[pos[0] - INPUT : pos[0]])
    cut = bt[bt["cutoff"] == df["ds"].iloc[pos[0] - 1]].sort_values("ds")
    np.testing.assert_allclose(one[EXOG_NAME].iloc[INPUT:], cut["ARIMA"])
    np.testing.assert_allclose(one["y"].iloc[INPUT:], cut["y"].to_numpy() - cut["ARIMA"].to_numpy())


def test_alvo_do_treino_nunca_passa_do_limite(df):
    pos = positions_for_horizon([100, 200, 300, 360], 384, 24)
    assert pos == [100, 200, 300, 360]
    assert positions_for_horizon([100, 200, 300, 380], 384, 24) == [100, 200, 300]


def test_inference_frames_nao_usa_valores_futuros(df, arima_files):
    fc, _ = arima_files
    test_pos = get_rolling_origins(INFO, len(df))
    hist, futr = inference_frames(df, fc, test_pos[:2], INPUT, H)
    assert len(hist) == 2 * INPUT and len(futr) == 2 * H
    assert (hist["y"] == 0).all()
    tampered = df.copy()
    tampered.loc[test_pos[0]:, "y"] += 500.0  # muda só o futuro da 1ª origem
    hist2, futr2 = inference_frames(tampered, fc, test_pos[:1], INPUT, H)
    h1 = hist[hist["unique_id"] == f"o{test_pos[0]}"]
    np.testing.assert_allclose(h1[EXOG_NAME].to_numpy(), hist2[EXOG_NAME].to_numpy())


def test_hybrid_exog_rolling_ponta_a_ponta(df, arima_files):
    fc, val = arima_files
    bt = arima_backtest(df, INFO, 0.5, 24, n_jobs=1)

    def factory(h):
        return NHITS(h=h, input_size=INPUT, futr_exog_list=[EXOG_NAME], scaler_type=None, max_steps=15,
                     early_stop_patience_steps=-1, random_seed=1, **QUIET)

    out, table = hybrid_exog_rolling(
        df, INFO, {}, "SYN", fc, val, bt, [H], INPUT, load_exog_config({}), model_factory=factory
    )
    test_pos = get_rolling_origins(INFO, len(df))
    assert out["cutoff"].nunique() == len(test_pos)
    assert (out.groupby(["H", "cutoff"]).size() == H).all()
    np.testing.assert_allclose(out[EXOG_COL], out["ARIMA"] + out["NHITS_exog"])
    assert table["alpha"].between(0, 1).all()
    assert {"skill_residual", "mse_val_arima", "mse_val_aditivo"} <= set(table.columns)
    # alpha = 0 reproduz o ARIMA
    z = out["H"].map(dict(zip(table["H"], table["alpha"])))
    np.testing.assert_allclose(out[EXOG_ALPHA_COL], out["ARIMA"] + z * out["NHITS_exog"])
    assert len(validation_origins(INFO, len(df))) > 0
