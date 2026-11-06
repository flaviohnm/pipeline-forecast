import json
import logging
import warnings

import numpy as np
import optuna
import pandas as pd
import pytest

from src.metrics.check_outputs import missing_for_dataset
from src.models import hpo
from src.models.deep_profile import (
    HPO_MODEL_KEYS,
    build_models,
    hpo_best_path,
    load_deep_config,
    load_hpo_config,
    load_tuned,
    tuned_input_size,
)

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

H = 24
INFO = {
    "date_column": "date",
    "target_column": "OT",
    "freq": "h",
    "seasonal_period": 24,
    "forecast_horizon": [12, 24],
    "max_horizon": H,
    "train_size": 480,
    "rolling": {"val_size": 96, "test_size": 192, "step_size": 48},
}


def _config(tmp_path, **hpo_over):
    return {
        "random_seed": 7,
        "compute": {"accelerator": "cpu", "devices": 1},
        "hpo": {"results_dir": str(tmp_path), "trial_max_steps": 5, "n_trials": 3, "n_trials_by_model": {}, **hpo_over},
    }


@pytest.fixture
def csv_info(tmp_path):
    n = 24 * 35
    t = np.arange(n)
    y = 10 + 3 * np.sin(2 * np.pi * t / 24) + 0.005 * t
    path = tmp_path / "syn.csv"
    pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="h"), "OT": y}).to_csv(path, index=False)
    return {**INFO, "path": str(path)}


def test_hpo_config_padroes_e_validacao():
    cfg = load_hpo_config({})
    assert cfg["enabled"] is False and cfg["use_tuned"] is True and cfg["n_trials"] == 20
    assert load_hpo_config({"hpo": {"n_trials_by_model": {"MLP": 5}}})["n_trials_by_model"] == {"Informer": 8, "MLP": 5}
    with pytest.raises(ValueError):
        load_hpo_config({"hpo": {"n_trials": 0}})
    with pytest.raises(ValueError):
        load_hpo_config({"hpo": {"n_trials_by_model": {"Inexistente": 3}}})


@pytest.mark.parametrize("key", HPO_MODEL_KEYS)
def test_espaco_de_busca_gera_kwargs_validos(key):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    for _ in range(5):
        trial = study.ask()
        params = hpo.suggest_params(key, trial)
        assert params["input_size"] in hpo.SEARCH_SPACE[key]["input_size"]
        assert params["learning_rate"] > 0
        study.tell(trial, 0.0)


@pytest.mark.parametrize("key", HPO_MODEL_KEYS)
def test_tentativa_de_referencia_e_a_configuracao_atual(key):
    deep_cfg = load_deep_config({})
    ref = hpo.reference_params(key, {}, deep_cfg, hybrid_input_size=96 if key == "NHITS_RES" else None)
    study = optuna.create_study()
    study.enqueue_trial(ref)
    params = hpo.suggest_params(key, study.ask())
    assert params["input_size"] == (96 if key == "NHITS_RES" else 336)
    base = hpo.BASE_MODEL[key]
    assert params["learning_rate"] == pytest.approx(deep_cfg["models"].get(base, {}).get("learning_rate", hpo.NF_DEFAULTS[base]["learning_rate"]))


def test_load_tuned_e_input_size(tmp_path):
    cfg = _config(tmp_path)
    assert load_tuned(cfg, "SYN", "MLP") is None
    assert tuned_input_size(cfg, "SYN", "MLP", 336) == 336
    hpo_best_path(cfg, "SYN", "MLP").write_text(json.dumps({"params": {"input_size": 24, "learning_rate": 0.01}}))
    assert load_tuned(cfg, "SYN", "MLP")["params"]["input_size"] == 24
    assert tuned_input_size(cfg, "SYN", "MLP", 336) == 24
    assert load_tuned(cfg, None, "MLP") is None
    assert load_tuned({**cfg, "hpo": {**cfg["hpo"], "use_tuned": False}}, "SYN", "MLP") is None


def test_build_models_aplica_hiperparametros_otimizados(tmp_path):
    cfg = _config(tmp_path)
    quiet = {"enable_progress_bar": False, "logger": False, "enable_model_summary": False, "enable_checkpointing": False}
    base = build_models(H, cfg, names=["MLP"], extra_kwargs=quiet)["MLP"]
    assert base.input_size == 336
    hpo_best_path(cfg, "SYN", "MLP").write_text(
        json.dumps({"params": {"input_size": 48, "learning_rate": 0.005, "hidden_size": 128, "num_layers": 3}})
    )
    tuned = build_models(H, cfg, names=["MLP"], extra_kwargs=quiet, dataset="SYN")["MLP"]
    assert tuned.input_size == 48 and tuned.hparams.learning_rate == 0.005 and tuned.hparams.hidden_size == 128
    # outro dataset (sem arquivo) e chamada sem dataset continuam com o perfil do YAML
    assert build_models(H, cfg, names=["MLP"], extra_kwargs=quiet, dataset="OUTRO")["MLP"].input_size == 336
    assert build_models(H, cfg, names=["MLP"], extra_kwargs=quiet)["MLP"].input_size == 336


def test_tuning_horizon_cabe_na_validacao():
    deep_cfg = load_deep_config({})
    cfg = load_hpo_config({})
    assert hpo.tuning_horizon(INFO, deep_cfg, cfg) == 24  # meio de [12, 24]
    small = {**INFO, "rolling": {**INFO["rolling"], "val_size": 20}}
    assert hpo.tuning_horizon(small, deep_cfg, {**cfg, "tuning_horizon": 24}) == 12
    assert all(s + 24 <= INFO["train_size"] + INFO["rolling"]["val_size"] for s in hpo.validation_starts(INFO, 24))


def test_tune_grava_evidencia_e_retoma(tmp_path, csv_info):
    cfg = _config(tmp_path)
    result = hpo.tune(cfg, "SYN", csv_info, "MLP", quiet=True)
    best_file = hpo_best_path(cfg, "SYN", "MLP")
    assert best_file.exists() and (tmp_path / "SYN_MLP_trials.csv").exists() and (tmp_path / "SYN_MLP.db").exists()
    saved = json.loads(best_file.read_text())
    assert saved["n_trials_completed"] == 3 and saved["seed"] == 7 and saved["tuning_horizon"] == 24
    assert "max_steps" not in saved["params"]  # o limite de passos da busca não vai para o treino final
    # a 1ª tentativa é a referência, então o melhor nunca é pior que ela
    assert saved["best_value"] <= saved["reference_value"] + 1e-12
    trials = pd.read_csv(tmp_path / "SYN_MLP_trials.csv")
    assert len(trials) == 3 and trials["state"].eq("COMPLETE").all()
    again = hpo.tune(cfg, "SYN", csv_info, "MLP", quiet=True)  # retomada: nada a rodar
    assert again["n_trials_completed"] == 3 and again["best_value"] == result["best_value"]


def test_busca_nao_depende_do_bloco_de_teste(tmp_path, csv_info):
    """Trocar os dados do teste por lixo não altera o MSE de validação de uma configuração fixa."""
    df = pd.read_csv(csv_info["path"])
    cfg = _config(tmp_path)
    objective = lambda frame, info: hpo.Objective(
        "MLP", "SYN", info, frame.rename(columns={"date": "ds", "OT": "y"}).assign(unique_id="SYN", ds=lambda d: pd.to_datetime(d["ds"])),
        cfg, 24, trial_max_steps=5,
    )
    params = {"input_size": 48, "learning_rate": 1e-3, "hidden_size": 64, "num_layers": 2}
    test_start = INFO["train_size"] + INFO["rolling"]["val_size"]
    suja = df.copy()
    suja.loc[test_start:, "OT"] = 1e6
    a = objective(df, csv_info).score(params)
    b = objective(suja, csv_info).score(params)
    assert a == pytest.approx(b)


def test_conferencia_final_exige_arquivos_do_hpo(tmp_path):
    cfg = _config(tmp_path, enabled=True)
    faltas = missing_for_dataset("SYN", INFO, tmp_path / "metrics", n_rows=800, config=cfg)
    for key in HPO_MODEL_KEYS:
        assert any(f"SYN_{key}_best.json" in f for f in faltas)
    for key in HPO_MODEL_KEYS:
        hpo_best_path(cfg, "SYN", key).write_text("{}")
    assert not any("best.json" in f for f in missing_for_dataset("SYN", INFO, tmp_path / "metrics", n_rows=800, config=cfg))
    # desligado: não exige nada
    off = _config(tmp_path, enabled=False)
    assert not any("best.json" in f for f in missing_for_dataset("SYN", INFO, tmp_path / "metrics", n_rows=800, config=off))
