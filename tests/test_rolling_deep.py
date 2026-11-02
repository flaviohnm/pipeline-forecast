import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from neuralforecast.models import MLP, NHITS

from src.models.deep_baselines import DLBaselinesTrainer, fit_predict_rolling
from src.models.deep_profile import load_deep_config
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
    trainer.config = {"random_seed": 42, "compute": {"accelerator": "cpu"}}
    trainer.deep_cfg = load_deep_config(trainer.config)
    models = trainer.build_models(horizon=720)
    assert list(models) == ["MLP", "LSTM", "NHITS", "Informer"]
    assert all(m.h == 720 for m in models.values())
    assert all(m.input_size == 336 for m in models.values())  # lookback único para todas as redes


def _trainer(tmp_path, models=None, resume=False, strategy_h=(24, 48)):
    trainer = DLBaselinesTrainer.__new__(DLBaselinesTrainer)
    trainer.config = {"random_seed": 7, "compute": {"accelerator": "cpu"}}
    trainer.deep_cfg = load_deep_config(trainer.config)
    trainer.deep_cfg["input_size"] = INPUT_SIZE
    trainer.deep_cfg["common"]["max_steps"] = 5
    trainer.forecast_dir = tmp_path
    trainer.models = models or ["MLP"]
    trainer.resume = resume
    trainer.failed = []
    return trainer


def _info_two_horizons():
    return {**INFO, "max_horizon": 48, "forecast_horizon": [24, 48], "rolling": {"val_size": 96, "test_size": 192, "step_size": 48}}


def test_each_model_gets_its_own_file_and_failure_is_isolated(df, tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, models=["MLP", "NHITS"])
    info = _info_two_horizons()
    real = DLBaselinesTrainer.run_rolling_model

    def flaky(self, name, info_, df_, model_name, resume=False):
        if model_name == "MLP":
            raise RuntimeError("bad allocation")
        return real(self, name, info_, df_, model_name, resume=resume)

    monkeypatch.setattr(DLBaselinesTrainer, "run_rolling_model", flaky)
    failed = trainer.run_rolling("SYN", info, df, models=["MLP", "NHITS"])
    assert failed == ["MLP"]
    assert not (tmp_path / "SYN_deep_mlp_rolling.csv").exists()
    out = pd.read_csv(tmp_path / "SYN_deep_nhits_rolling.csv")
    assert sorted(out["H"].unique()) == [24, 48]
    assert "NHITS" in out.columns and "MLP" not in out.columns


def test_resume_skips_finished_horizons(df, tmp_path):
    trainer = _trainer(tmp_path)
    info = _info_two_horizons()
    trainer.run_rolling_model("SYN", info, df, "MLP")
    path = tmp_path / "SYN_deep_mlp_rolling.csv"
    full = pd.read_csv(path)
    # simula uma execução interrompida depois do H=24
    full[full["H"] == 24].to_csv(path, index=False)
    sentinel = full[full["H"] == 24]["MLP"].to_numpy().copy()
    trainer.run_rolling_model("SYN", info, df, "MLP", resume=True)
    out = pd.read_csv(path)
    assert sorted(out["H"].unique()) == [24, 48]
    # o H=24 já concluído não foi retreinado
    assert np.allclose(out[out["H"] == 24]["MLP"].to_numpy(), sentinel)


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        DLBaselinesTrainer(models=["Transformer"])


def test_informer_inference_batch_is_small_by_default():
    cfg = load_deep_config({})
    assert cfg["models"]["Informer"]["windows_batch_size"] <= 32
    assert cfg["models"]["Informer"]["inference_windows_batch_size"] <= 64
