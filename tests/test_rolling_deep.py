import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from neuralforecast.models import MLP, NHITS

from src.models.deep_baselines import DLBaselinesTrainer, fit_predict_rolling
from src.utils.general import get_rolling_origins

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

H = 24
INPUT_SIZE = 48
INFO = {
    "freq": "h",
    "max_horizon": H,
    "train_size": 480,
    "rolling": {"val_size": 96, "test_size": 192, "step_size": 48},
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
    n = 24 * 35
    t = np.arange(n)
    y = 10 + 3 * np.sin(2 * np.pi * t / 24) + 0.005 * t
    return pd.DataFrame({"unique_id": "SYN", "ds": pd.date_range("2021-01-01", periods=n, freq="h"), "y": y})


def _mlp():
    return MLP(h=H, input_size=INPUT_SIZE, max_steps=10, scaler_type="standard", random_seed=7, **QUIET)


def test_fit_predict_rolling_format_and_alignment(df):
    out = fit_predict_rolling("MLP", _mlp(), df, INFO, INPUT_SIZE)
    origins = get_rolling_origins(INFO, len(df))
    assert list(out.columns) == ["unique_id", "cutoff", "ds", "MLP"]
    assert len(out) == len(origins) * H
    assert not out["MLP"].isna().any()
    # check_rolling_cutoffs (com h) já roda dentro; aqui confirmamos a 1ª data de cada janela
    first = out.groupby("cutoff")["ds"].min().sort_values().tolist()
    assert first == df["ds"].iloc[origins].tolist()


def test_fit_predict_rolling_no_leakage(df):
    """Alterar a série a partir da origem k não muda as janelas com cutoff anterior a ela."""
    origins = get_rolling_origins(INFO, len(df))
    k = 2
    base = fit_predict_rolling("MLP", _mlp(), df, INFO, INPUT_SIZE)

    tampered = df.copy()
    tampered.loc[origins[k] :, "y"] += 50.0
    out = fit_predict_rolling("MLP", _mlp(), tampered, INFO, INPUT_SIZE)

    cutoffs = sorted(base["cutoff"].unique())
    before = base["cutoff"].isin(cutoffs[: k + 1])  # janelas 0..k só veem dados < origins[k]
    np.testing.assert_allclose(out.loc[before, "MLP"], base.loc[before, "MLP"], rtol=1e-5)
    after = base["cutoff"] == cutoffs[k + 1]  # a janela k+1 vê os dados alterados
    assert not np.allclose(out.loc[after, "MLP"], base.loc[after, "MLP"])


def test_fit_predict_rolling_trains_only_on_train_plus_val(df):
    """O bloco de teste não entra no treino: alterá-lo inteiro não muda a 1ª janela."""
    origins = get_rolling_origins(INFO, len(df))
    base = fit_predict_rolling("NHITS", NHITS(h=H, input_size=INPUT_SIZE, max_steps=10, random_seed=7, **QUIET), df, INFO, INPUT_SIZE)
    tampered = df.copy()
    tampered.loc[origins[0] :, "y"] = 0.0
    out = fit_predict_rolling(
        "NHITS", NHITS(h=H, input_size=INPUT_SIZE, max_steps=10, random_seed=7, **QUIET), tampered, INFO, INPUT_SIZE
    )
    first = base["cutoff"] == base["cutoff"].min()
    np.testing.assert_allclose(out.loc[first, "NHITS"], base.loc[first, "NHITS"], rtol=1e-5)


def test_fit_predict_rolling_rejects_short_history(df):
    with pytest.raises(ValueError, match="input_size"):
        fit_predict_rolling("MLP", _mlp(), df, INFO, input_size=1000)


def test_build_models_shared_definitions():
    trainer = DLBaselinesTrainer.__new__(DLBaselinesTrainer)
    trainer.config = {"random_seed": 42}
    models = trainer.build_models(horizon=720, input_size=1440)
    assert list(models) == ["MLP", "LSTM", "NHITS", "Informer"]
    assert all(m.h == 720 for m in models.values())
    assert models["MLP"].input_size == 1440
    assert models["Informer"].input_size == 720  # Informer mantém input_size = horizonte
